"""Resolve run directories and checkpoints for the runner.

Two sources of truth, in order of preference:

1. The runner state registry — the *exact* run directory/checkpoint a stage
   produced (deterministic even after later runs are added).
2. A strict seed+tag filesystem scan of the output tree (covers runs
   launched manually, outside the runner).

A run directory is an eligible artifact only if it carries the
``<run_dir>.completed`` sibling marker that ``RunWriter.mark_completed``
writes at the very end of a successful run — ``run_meta.json`` alone proves
nothing (it is written when a run *starts*, so crashed runs have it too).
The scan therefore can never select a partial run whose early checkpoints
survived a crash.

The tag disambiguates variants that share a model output tree, e.g. the
``data=robot_only`` BC cell recorded with ``project.tag=robot_only`` next to
the default BC runs under ``outputs/bc/stage1/``.

Since seeds became a directory dimension, runs live under
``{model}/stage{N}/seed{S}/{run}/``; legacy runs directly under
``{model}/stage{N}/`` are still resolved (both layouts are scanned). The
``seed`` filter is applied from ``run_meta.json`` regardless of layout, so
resolution stays seed-exact either way.

Tag semantics here are *strict*: ``tag=None`` matches only runs whose
``run_meta.json`` records no tag (the default cell must never resolve to the
``robot_only`` variant, and vice versa). This is intentionally stricter than
:func:`phaseforge.utils.config.find_latest_checkpoint`, whose ``tag=None``
means "no constraint".
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from phaseforge.runner.protocol import Method
from phaseforge.runner.registry import RunnerState

_REQUIRED_CKPT_REL = "checkpoints/checkpoint_best.pt"
_SEED_DIR_RE = re.compile(r"^seed\d+$")


def is_seed_dir(path: str | Path) -> bool:
    """Return whether ``path`` is a ``seed{N}`` directory level.

    Multi-seed runs are organised as ``{model}/stage{N}/seed{S}/{run}/``;
    this predicate lets the resolver accept both that layout and the legacy
    ``{model}/stage{N}/{run}/`` layout (runs written before seeds were a
    directory dimension).
    """
    return bool(_SEED_DIR_RE.match(Path(path).name))


class CheckpointError(RuntimeError):
    """Raised when a required run directory or checkpoint cannot be resolved."""


def _read_run_meta(run_dir: Path) -> dict[str, Any]:
    meta_path = run_dir / "run_meta.json"
    if not meta_path.is_file():
        return {}
    try:
        loaded = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _tag_matches(meta_tag: Any, expected: str | None) -> bool:
    if expected is None:
        return meta_tag is None
    return meta_tag == expected


def _is_completed(run_dir: Path) -> bool:
    """A run is an eligible artifact only if the run *finished* successfully.

    ``run_meta.json`` is written when a run *starts*, so it exists for
    crashed/killed runs too; only the ``<run_dir>.completed`` sibling marker
    (written by ``RunWriter.mark_completed`` at the very end of a successful
    run) proves the run actually finished. Requiring it here guarantees the
    resolver can never select a partial run whose early checkpoints were
    saved before a crash.
    """
    return run_dir.with_name(run_dir.name + ".completed").is_file()


def _iter_runs_newest_first(search_dir: Path):
    """Collect candidate run dirs across the dual output layout, newest-first.

    Current runs live under ``stage{N}/seed{S}/{run}/``; legacy runs (written
    before seeds were a directory dimension) sit directly under ``stage{N}/``.
    Both are collected, then sorted globally by run name so the newest-first
    contract holds regardless of layout.
    """
    runs: list[Path] = []
    for child in search_dir.iterdir():
        if not child.is_dir():
            continue
        if is_seed_dir(child):
            runs.extend(sub for sub in child.iterdir() if sub.is_dir())
        else:
            runs.append(child)
    runs.sort(key=lambda p: p.name, reverse=True)
    return iter(runs)


def _find_run(
    search_dir: Path,
    model_name: str,
    kind: str,
    *,
    seed: int,
    tag: str | None,
    expected_commit: str | None = None,
    expected_config_hash: str | None = None,
) -> Path:
    if not search_dir.is_dir():
        raise CheckpointError(f"No {kind} runs found under {search_dir} for seed {seed}.")
    for run in _iter_runs_newest_first(search_dir):
        if not _is_completed(run):
            continue
        meta = _read_run_meta(run)
        if meta.get("seed") != seed:
            continue
        if not _tag_matches(meta.get("tag"), tag):
            continue
        if expected_commit and meta.get("git_commit") != expected_commit:
            continue
        if expected_config_hash is not None:
            # Fail closed: a run without a recorded config hash is not
            # eligible when the contract is enforced.
            if meta.get("config_hash") != expected_config_hash:
                continue
        return run
    raise CheckpointError(
        f"No completed {kind} run for {model_name} seed {seed}"
        + (f" with tag {tag!r}" if tag else " with no tag")
        + (
            f" and config_hash {expected_config_hash!r}"
            if expected_config_hash
            else ""
        )
        + (
            f" and git commit {expected_commit!r}"
            if expected_commit
            else ""
        )
        + f" under {search_dir}."
    )


def resolve_run_dir(
    outputs_base: Path,
    model_name: str,
    stage: int,
    *,
    seed: int,
    tag: str | None = None,
    expected_commit: str | None = None,
    expected_config_hash: str | None = None,
) -> Path:
    """Return the newest *completed* run directory for ``model/stage`` matching
    seed+tag (and, when given, commit and config hash).

    Searches ``<outputs_base>/<model_name>/stage<stage>/`` (including
    ``seed{S}/`` sub-directories when present) newest-first and returns the
    first directory whose ``<run_dir>.completed`` marker exists and whose
    ``run_meta.json`` records the expected seed and tag. ``tag=None`` matches
    only untagged runs. Requiring the completion marker means a run that
    crashed after saving early checkpoints is never selected. Newest-first
    matters because a sweep can legitimately produce several runs for one
    ``(model, stage, seed)`` cell (e.g. a re-run after a crash); under the
    runner's "latest successful wins" policy the newest completed run is the
    intended artifact.

    When ``expected_commit`` is given, only runs whose ``run_meta.json``
    ``git_commit`` matches it are eligible — stale pre-fix checkpoints from
    an earlier git revision are never selected for a re-run. When
    ``expected_config_hash`` is given, only runs whose recorded
    ``config_hash`` matches are eligible; runs without a recorded hash are
    rejected (fail closed). This is the robust artifact-contract gate: it
    holds even if a sweep legitimately spans several commits.

    Raises:
        CheckpointError: No matching run directory exists.
    """
    stage_dir = outputs_base / model_name / f"stage{stage}"
    return _find_run(
        stage_dir,
        model_name,
        f"{model_name} stage{stage}",
        seed=seed,
        tag=tag,
        expected_commit=expected_commit,
        expected_config_hash=expected_config_hash,
    )


def resolve_checkpoint_path(
    outputs_base: Path,
    method: Method,
    stage: int,
    *,
    seed: int,
    state: RunnerState | None = None,
    expected_commit: str | None = None,
    expected_config_hash: str | None = None,
) -> Path:
    """Resolve the absolute ``checkpoint_best.pt`` for a completed stage.

    Prefers the exact artifact recorded in the runner state; falls back to a
    strict seed+tag scan (so manually launched runs work too). The returned
    path is the one the evaluation step loads. With ``expected_commit`` /
    ``expected_config_hash``, both the registry lookup and the scan are
    gated to runs produced at that git revision / resolved config.
    """
    rel = state.get_ckpt(method.phase_key, seed, stage) if state is not None else None
    if rel:
        candidate = (outputs_base / rel).resolve()
        if candidate.is_file():
            return candidate
    run_dir = resolve_run_dir(
        outputs_base,
        method.model_name,
        stage,
        seed=seed,
        tag=method.output_tag,
        expected_commit=expected_commit,
        expected_config_hash=expected_config_hash,
    )
    ckpt = run_dir / _REQUIRED_CKPT_REL
    if not ckpt.is_file():
        raise CheckpointError(
            f"Run {run_dir} completed but has no {_REQUIRED_CKPT_REL} — the "
            "training run did not persist a best checkpoint."
        )
    return ckpt.resolve()


def resolve_stage_ckpt(
    outputs_base: Path,
    model_name: str,
    stage: int,
    *,
    seed: int,
    tag: str | None = None,
    expected_commit: str | None = None,
    expected_config_hash: str | None = None,
) -> Path:
    """Resolve the absolute ``checkpoint_best.pt`` of a provider run.

    This is what a stage-2 training subprocess loads via
    ``train.stage1_ckpt_path``. It reuses the strict seed+tag resolution so
    the exact, completed, *untagged* provider run is selected — never the CLI
    auto-detect (:func:`phaseforge.utils.config.find_latest_checkpoint`),
    whose ``tag=None`` means "no constraint" and can therefore pick a tagged
    sibling variant that shares the provider's output tree, crashing the
    stage-2 load with a dimension mismatch. With ``expected_commit`` /
    ``expected_config_hash``,
    only provider checkpoints produced at that git revision / resolved
    config are eligible.
    """
    run_dir = resolve_run_dir(
        outputs_base,
        model_name,
        stage,
        seed=seed,
        tag=tag,
        expected_commit=expected_commit,
        expected_config_hash=expected_config_hash,
    )
    ckpt = run_dir / _REQUIRED_CKPT_REL
    if not ckpt.is_file():
        raise CheckpointError(
            f"Run {run_dir} completed but has no {_REQUIRED_CKPT_REL} — the "
            "training run did not persist a best checkpoint."
        )
    return ckpt.resolve()


def checkpoint_exists(
    outputs_base: Path,
    model_name: str,
    stage: int,
    *,
    seed: int,
    tag: str | None = None,
    expected_commit: str | None = None,
    expected_config_hash: str | None = None,
) -> bool:
    """Return whether a ``checkpoint_best.pt`` exists for the exact seed+tag."""
    try:
        run_dir = resolve_run_dir(
            outputs_base,
            model_name,
            stage,
            seed=seed,
            tag=tag,
            expected_commit=expected_commit,
            expected_config_hash=expected_config_hash,
        )
    except CheckpointError:
        return False
    return (run_dir / _REQUIRED_CKPT_REL).is_file()


def stage_checkpoint_relative(
    outputs_base: Path, run_dir: Path, model_name: str, stage: int
) -> str:
    """Record the run's checkpoint as a path relative to the outputs base."""
    ckpt = run_dir / _REQUIRED_CKPT_REL
    if not ckpt.is_file():
        raise CheckpointError(
            f"Run {run_dir} completed but has no {_REQUIRED_CKPT_REL} — the "
            "training run did not persist a best checkpoint."
        )
    return ckpt.relative_to(outputs_base).as_posix()


def resolve_eval_run_dir(
    outputs_base: Path,
    model_name: str,
    *,
    seed: int,
    tag: str | None = None,
    expected_commit: str | None = None,
    expected_config_hash: str | None = None,
) -> Path:
    """Return the newest evaluation run directory matching seed+tag.

    Evaluation runs live under ``<outputs_base>/eval/<model_name>/``.
    ``tag=None`` matches only untagged runs.
    """
    eval_dir = outputs_base / "eval" / model_name
    return _find_run(
        eval_dir,
        model_name,
        f"evaluation for {model_name}",
        seed=seed,
        tag=tag,
        expected_commit=expected_commit,
        expected_config_hash=expected_config_hash,
    )


def verify_provider_task(
    ckpt_path: str | Path, *, expected_task: str | None, include_data_hash: bool = False
) -> dict[str, Any]:
    """Fail closed unless a provider checkpoint was trained on the expected task.

    A seed-exact checkpoint from the wrong task lives in a different output
    tree symmetry (same model, same seed, different ``data``) and must never
    satisfy a Stage 2 provider request. The task is read from
    the provider run's own ``resolved_config.yaml``
    (``data.source.task_name``); a missing/unreadable sidecar or a mismatch
    raises :class:`CheckpointError` before any subprocess launches.

    Returns a summary dict (``task``, plus ``data_config_hash`` only when
    ``include_data_hash`` is set — recomputing it re-stats the raw dataset,
    so resolution callers leave it off). ``expected_task=None``
    (task-agnostic callers) skips the check but still reports what was found.
    """
    ckpt_path = Path(ckpt_path)
    run_dir = ckpt_path.parent.parent
    resolved = run_dir / "resolved_config.yaml"
    if not resolved.is_file():
        raise CheckpointError(
            f"Provider run {run_dir} has no resolved_config.yaml; cannot verify "
            "its task. Refusing to consume an untraceable artifact."
        )
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(str(resolved))
    except Exception as exc:
        raise CheckpointError(
            f"Provider run {run_dir} has an unreadable resolved_config.yaml "
            f"({type(exc).__name__}: {exc}); cannot verify its task."
        ) from exc
    try:
        data = cfg.get("data", None)
        source = data.get("source", None) if data is not None else None
        task = str(source.get("task_name", "")) if source is not None else ""
        data_hash = None
        if include_data_hash:
            try:
                from phaseforge.data.ingestion.cache_manager import CacheManager

                data_hash = CacheManager.compute_hash(cfg.data)
            except Exception:
                data_hash = None
    except Exception as exc:
        raise CheckpointError(
            f"Provider run {run_dir} resolved config has no data.source.task_name; "
            "cannot verify its task."
        ) from exc
    if expected_task is not None and task != expected_task:
        raise CheckpointError(
            f"Provider checkpoint {ckpt_path} was trained on task {task!r}; "
            f"the consumer requires task {expected_task!r}. Run the provider "
            "Stage 1 for the consuming task first."
        )
    return {"task": task, "data_config_hash": data_hash}


_EXPERT_KEY_RE = re.compile(r"^moe_layer\.experts\.(\d+)\.")


def _count_experts_from_state_keys(state_keys) -> int | None:
    """Return the expert count implied by ``moe_layer.experts.<i>.`` keys.

    Returns ``None`` when the state dict carries no MoE expert keys (dense
    models such as BC), in which case the expert-count contract check is not
    applicable.
    """
    indices = {
        int(m.group(1))
        for key in state_keys
        if (m := _EXPERT_KEY_RE.match(str(key))) is not None
    }
    if not indices:
        return None
    expected = set(range(len(indices)))
    if indices != expected:
        raise CheckpointError(
            "Checkpoint expert indices are not contiguous "
            f"(found {sorted(indices)}); the artifact is corrupt or from an "
            "incompatible architecture."
        )
    return len(indices)


def verify_checkpoint_contract(
    ckpt_path: Path,
    *,
    expected_model_name: str | None = None,
    expected_num_experts: int | tuple[int, ...] | set[int] | list[int] | None = None,
    expected_stage: int | None = None,
) -> dict[str, Any]:
    """Fail closed unless a checkpoint satisfies the requested contract.

    Loads the checkpoint header (CPU) and its sibling ``run_meta.json``
    (``<run_dir>/run_meta.json`` next to ``checkpoints/checkpoint_best.pt``)
    and verifies:

    * the sidecar's ``model_name`` matches ``expected_model_name`` (when
      given) — catches artifacts selected from the wrong model's tree;
    * the sidecar's ``stage`` matches ``expected_stage`` (when given);
    * the state dict's MoE expert count matches ``expected_num_experts``
      (when given and the checkpoint contains experts — dense checkpoints
      skip this check). ``expected_num_experts`` may be a single ``int`` or
      a collection of allowed counts. This is the guard against pre-final
      artifacts of the same filesystem name (e.g. the retired 8-expert
      ``phaseforge``).

    Returns a summary dict (``num_experts``, sidecar fields) for logging.

    Raises:
        CheckpointError: The checkpoint violates the contract, cannot be
            read, or the sidecar is missing a required field.
    """
    import torch

    ckpt_path = Path(ckpt_path)
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - any load failure must fail closed
        raise CheckpointError(
            f"Checkpoint {ckpt_path} cannot be loaded for contract "
            f"verification: {type(exc).__name__}: {exc}"
        ) from exc
    state = ckpt.get("model_state_dict") if isinstance(ckpt, dict) else None
    if state is None:
        raise CheckpointError(
            f"Checkpoint {ckpt_path} has no 'model_state_dict' entry; not a "
            "PhaseForge training artifact."
        )

    num_experts = _count_experts_from_state_keys(state.keys())
    if expected_num_experts is not None and num_experts is not None:
        if isinstance(expected_num_experts, (tuple, set, list)):
            allowed: set[int] = {int(x) for x in expected_num_experts}
        else:
            allowed = {int(expected_num_experts)}
        if num_experts not in allowed:
            if len(allowed) == 1:
                need = str(next(iter(allowed)))
            else:
                need = f"one of {sorted(allowed)}"
            raise CheckpointError(
                f"Checkpoint {ckpt_path} has {num_experts} experts; the final "
                f"protocol requires {need}. Refusing to consume a "
                "pre-final/incompatible artifact."
            )

    # run_dir/checkpoints/checkpoint_best.pt -> run_dir (the sidecar lives
    # at run_dir/run_meta.json; _read_run_meta appends the file name).
    run_dir = ckpt_path.parent.parent
    meta = _read_run_meta(run_dir)
    if expected_model_name is not None:
        got = meta.get("model_name")
        if got != expected_model_name:
            raise CheckpointError(
                f"Checkpoint {ckpt_path} belongs to model {got!r}; expected "
                f"{expected_model_name!r}."
            )
    if expected_stage is not None:
        got = meta.get("stage")
        if got is not None and int(got) != int(expected_stage):
            raise CheckpointError(
                f"Checkpoint {ckpt_path} is a stage-{got} artifact; expected "
                f"stage {expected_stage}."
            )

    return {
        "num_experts": num_experts,
        "model_name": meta.get("model_name"),
        "stage": meta.get("stage"),
        "config_hash": meta.get("config_hash"),
        "seed": meta.get("seed"),
    }

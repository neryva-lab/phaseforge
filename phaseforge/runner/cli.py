"""``phaseforge-sweep`` — the experiment runner CLI.

Runs complete experiments from the frozen protocol manifest: for every
selected method and seed it executes each training stage in order and then
the offline evaluation of that method's final-stage checkpoint, honouring
the protocol's Stage 1 source dependencies. A resumable state registry
(``<outputs>/_runner/state.json``) records what completed and the *exact*
artifact each stage produced, so evaluation never targets a stale or
seed-mixed checkpoint and an interrupted sweep resumes in place.

When a selected stage-2 method's Stage 1 provider (historical ``bc`` /
``phaseforge`` or a final ``*_stage1`` identity) is not selected and its
checkpoint is missing from the output tree, the runner auto-trains the
provider's Stage 1 as a dependency step before running the consumer, so a
partial ``--methods`` selection no longer fails pre-flight on a missing
provider. Explicitly scoped runs (``--stage``/``--eval-only``) keep the
strict pre-flight check: a missing prerequisite then fails loudly
instead of silently training.

Examples::

    phaseforge-sweep                                    # full five-task matrix x seeds
    phaseforge-sweep --methods precision_residual_phaseforge               # the method on every task
    phaseforge-sweep --methods precision_residual_phaseforge --tasks Lift  # method x task subset
    phaseforge-sweep --methods precision_residual_phaseforge@Lift bc@Can   # exact cells by name@task
    phaseforge-sweep --tasks Lift Can                   # every method on those tasks
    phaseforge-sweep --methods 1,5,9                    # by manifest index (legacy)
    phaseforge-sweep --methods 1 --seeds 42
    phaseforge-sweep --stage 1                          # train stage 1 only
    phaseforge-sweep --eval-only                        # (re)run evaluations only
    phaseforge-sweep --methods 3 --dry-run              # preview commands
    phaseforge-sweep --list                             # show the method matrix
    phaseforge-sweep --expect-steps 285 --dry-run       # assert the plan size first

``python -m phaseforge.runner --help`` is equivalent.

The five-task manifest contains task-specific entries for Lift, Can, Square,
Tool Hang, and Transport. Tool Hang subprocesses are routed to the dedicated
robosuite 1.5.0 interpreter; the other tasks use the current environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from phaseforge.runner.executor import (
    CommandError,
    preflight_toolhang_python,
    resolve_toolhang_python,
    run_step,
    step_command,
)
from phaseforge.runner.protocol import (
    Method,
    Protocol,
    ProtocolError,
    Step,
    build_plan,
    is_final_provider,
    load_protocol,
    provider_method_name,
)
from phaseforge.runner.registry import RegistryError, RunnerState
from phaseforge.runner.resolver import (
    CheckpointError,
    resolve_checkpoint_path,
    resolve_eval_run_dir,
    resolve_run_dir,
    resolve_stage_ckpt,
    stage_checkpoint_relative,
    verify_checkpoint_contract,
    verify_provider_task,
)
from phaseforge.runner.selection import (
    SelectionResult,
    SelectionSpec,
    format_selection_table,
    resolve_selection,
)
from phaseforge.utils.config import config_hash, git_info

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Expert-count contract for every Stage 1 MoE checkpoint consumed by
#: the final protocol (the canonical method and Stage 1 providers are six-expert).
#: Dense checkpoints (BC family) carry no ``moe_layer.experts.*`` keys and
#: skip the check inside ``verify_checkpoint_contract``. This is the
#: fail-closed guard against pre-final artifacts that share a filesystem
#: name with the canonical method.
FINAL_EXPERT_CONTRACT = 6

#: Default protocol manifest for the locked final experiment matrix.
DEFAULT_MANIFEST = "experiments/final_causal_matrix.json"


def _split_list(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values or []:
        out.extend(part.strip() for part in value.split(",") if part.strip())
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="phaseforge-sweep",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help=f"Protocol manifest JSON (default: {DEFAULT_MANIFEST}).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[],
        help="Method tokens: a name ('precision_residual_phaseforge' — every task), an exact cell "
        "('precision_residual_phaseforge@Lift'), or a manifest index ('1', legacy). "
        "Default: all methods.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=[],
        help="Task facet filter (e.g. 'Lift Can'); intersects with --methods. "
        "Default: no task filtering.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=[],
        help="Protocol seeds to run (e.g. '42,44'). Default: all protocol seeds.",
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=(1, 2),
        default=None,
        help="Run only this training stage (no evaluation).",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run only the evaluation steps (final-stage checkpoints must exist).",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Run all training stages but skip the evaluation steps.",
    )
    parser.add_argument(
        "--with-dependencies",
        action="store_true",
        help="Include the required Stage 1 pretraining of a shared provider in "
        "the printed plan. The runtime auto-injects it for unscoped sweeps "
        "regardless, so this flag mainly controls plan visibility.",
    )
    parser.add_argument(
        "--expect-steps",
        type=int,
        default=None,
        metavar="N",
        help="Refuse to run unless the built plan has exactly N steps — a "
        "machine-checked guard against manifest or selection drift, verified "
        "before any step executes.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run steps that the state registry already marks completed.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record a failed step and keep going instead of stopping.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and exact commands without executing anything.",
    )
    parser.add_argument(
        "--verify-gates",
        action="store_true",
        help="Run the read-only final-matrix verification gates (DRY-01..DRY-05: "
        "inspection record, plan expansion, provider ordering, command "
        "contract, historical isolation) and exit without executing anything.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the method matrix from the manifest and exit.",
    )
    parser.add_argument(
        "--outputs",
        default="outputs",
        help="Output base directory, relative to the project root (default: outputs).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Forward project.log_level=INFO to the runs (default WARNING).",
    )
    parser.add_argument(
        "--toolhang-python",
        default=None,
        help="Python interpreter for Tool Hang steps (also PHASEFORGE_TOOLHANG_PYTHON). "
        "Defaults to .venv-toolhang/bin/python or .venv-toolhang/Scripts/python.exe.",
    )
    parser.add_argument(
        "--no-commit-gate",
        action="store_true",
        help="Disable git commit gating: "
        "allow reusing completed checkpoints from previous git commits.",
    )
    return parser.parse_args(argv)


def _manifest_path(args: argparse.Namespace) -> Path:
    p = Path(args.manifest)
    target = p if p.is_absolute() else PROJECT_ROOT / p
    if target.is_dir():
        main_json = target / "main.json"
        if main_json.exists():
            return main_json
    return target


def _outputs_base(args: argparse.Namespace) -> Path:
    return (PROJECT_ROOT / args.outputs).resolve()


def _selection_spec(args: argparse.Namespace) -> SelectionSpec:
    return SelectionSpec(
        method_tokens=tuple(_split_list(args.methods)),
        tasks=tuple(_split_list(args.tasks)),
    )


def _build_plan(
    protocol: Protocol, args: argparse.Namespace
) -> tuple[list[Step], SelectionResult]:
    selection = resolve_selection(protocol, _selection_spec(args))
    seeds: list[int] | None = None
    if args.seeds:
        try:
            seeds = [int(s) for s in _split_list(args.seeds)]
        except ValueError as exc:
            raise ProtocolError(f"--seeds must be integers: {args.seeds}") from exc
    plan = build_plan(
        protocol,
        list(selection.methods),
        seeds=seeds,
        stage=args.stage,
        eval_only=args.eval_only,
        skip_eval=args.skip_eval,
        with_dependencies=args.with_dependencies,
    )
    return plan, selection


def _provider_stage1_overrides(
    protocol: Protocol, provider: Method, seed: int, outputs_base: Path
) -> list[str]:
    """Build the Hydra override list for a provider's Stage 1 config.

    Mirrors :func:`phaseforge.runner.commands.train_command` for a stage-1
    step of the provider method exactly (same keys, same order semantics),
    so the composed config hashes identically to the provider run's
    recorded ``resolved_config.yaml`` when the provider was trained from
    this manifest at this commit.
    """
    cmd = [
        f"models={provider.model}",
        "train=stage1",
        f"project.seed={seed}",
        f"project.output_dir={outputs_base}",
    ]
    if provider.data != "common":
        cmd.append(f"data={provider.data}")
    if provider.output_tag:
        cmd.append(f"project.tag={provider.output_tag}")
    if provider.name:
        cmd.append(f"project.method={provider.name}")
    cmd.extend(protocol.defaults)
    if provider.overrides:
        cmd.extend([o for o in provider.overrides if not o.startswith("eval.")])
    return cmd


def _expected_stage1_config_hash(
    protocol: Protocol,
    provider: Method,
    seed: int,
    outputs_base: Path,
    device: str | None,
) -> str:
    """Compose a provider's Stage 1 config and hash it.

    Applies the same training-time mutations the CLI applies before
    hashing (per-model ``freeze_encoder`` override; effective device), so
    the digest is directly comparable with the provider run's recorded
    ``config_hash``. Raises :class:`CheckpointError` when composition
    itself fails — a provider whose expected config cannot even be built
    must fail closed, never silently pass.
    """
    from hydra import compose, initialize

    overrides = _provider_stage1_overrides(protocol, provider, seed, outputs_base)
    if device is not None:
        overrides.append(f"project.device={device}")
    try:
        with initialize(version_base="1.3", config_path="../config"):
            cfg = compose(config_name="main", overrides=overrides)
    except Exception as exc:
        raise CheckpointError(
            f"Cannot compose the expected Stage 1 config for provider "
            f"{provider.name!r} (task={provider.task!r}, seed={seed}): "
            f"{type(exc).__name__}: {exc}. Refusing to consume a provider "
            "whose identity cannot be established."
        ) from exc
    # Mirror phaseforge.cli.train: per-model freeze override lands in
    # train.freeze_encoder BEFORE run metadata / config hash are written.
    try:
        model_freeze = cfg.models.get("freeze_encoder", None)
    except Exception:
        model_freeze = None
    if model_freeze is not None:
        cfg.train.freeze_encoder = bool(model_freeze)
    return config_hash(cfg)


def _check_provider_config_identity(
    protocol: Protocol,
    step: Step,
    provider: Method,
    ckpt: Path,
    outputs_base: Path,
) -> None:
    """Enforce the provider config-identity gate.

    Reads the resolved provider run's recorded ``device`` and
    ``config_hash``, recomposes the expected Stage 1 config for that
    device, and rejects mismatches before any subprocess launches. The
    device is environment identity (not science) and is normalized to the
    recorded value; every other key must match exactly.
    """
    import json as _json

    run_dir = ckpt.parent.parent
    meta_path = run_dir / "run_meta.json"
    try:
        meta = _json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CheckpointError(
            f"Provider run {run_dir} has no readable run_meta.json; cannot "
            "verify its config identity."
        ) from exc
    if not isinstance(meta, dict) or not meta.get("config_hash"):
        raise CheckpointError(
            f"Provider run {run_dir} records no config_hash; failing closed "
            "instead of consuming an unidentified artifact."
        )
    expected = _expected_stage1_config_hash(
        protocol, provider, step.seed, outputs_base, meta.get("device"),
    )
    if meta.get("config_hash") != expected:
        raise CheckpointError(
            f"Provider {provider.name!r} (task={provider.task!r}, seed={step.seed}) "
            f"config mismatch: run {run_dir.name} records "
            f"config_hash {meta.get('config_hash')!r} but the manifest resolves "
            f"{expected!r}. The provider checkpoint was trained from a "
            "different configuration — re-train the provider Stage 1 from "
            "this manifest instead of consuming a stale artifact."
        )


def _require_stage2_prereq(
    step: Step,
    outputs_base: Path,
    expected_commit: str | None = None,
    protocol: Protocol | None = None,
) -> Path | None:
    """Resolve the exact Stage 1 checkpoint a stage-2 step bootstraps from.

    Returns ``None`` for steps that load nothing (a stage-1 step, or a
    stage-2 step whose model has no Stage 1 source). The returned path is
    passed to the subprocess as ``train.stage1_ckpt_path`` so it loads this
    exact artifact rather than re-running its own auto-detect
    (:func:`phaseforge.utils.config.find_latest_checkpoint`), whose
    ``tag=None`` means "no constraint" and can select a newer *tagged* sibling
    variant that shares the provider's output tree, crashing the load with a
    dimension mismatch. With ``expected_commit``, only provider checkpoints
    from that git revision are eligible.

    Gate stack (all before any subprocess launches):

    * seed-exact completed-run resolution + checkpoint contract
      (model tree, six-expert count, stage) — every source;
    * task match: the provider run's own resolved config must
      name the consuming task. Historical sources skip the check when the
      run predates ``resolved_config.yaml`` sidecars; final
      ``*_stage1`` identities always require it;
    * config identity: for final providers, and only when the
      calling sweep passes its ``protocol``, the provider run's recorded
      config hash must equal the manifest-resolved Stage 1 hash.
    """
    req = step.required_checkpoint()
    if req is None:
        return None
    model, stage = req
    try:
        source = step.method.stage2_source
        source_tag = (
            step.method.output_tag if source == "self" else step.method.task
        )
        ckpt = resolve_stage_ckpt(
            outputs_base,
            model,
            stage,
            seed=step.seed,
            # A Stage 2 dependency consumes either this method's own tagged
            # Stage 1 cell or the task's default provider cell.
            tag=source_tag,
            expected_commit=expected_commit,
        )
    except CheckpointError as exc:
        raise CheckpointError(
            f"{step.method.name} stage 2 needs a {model} stage 1 checkpoint for "
            f"seed {step.seed} under {outputs_base}. Run {model} stage 1 first, "
            "or re-run with --with-dependencies."
        ) from exc
    # Fail closed on contract violations (wrong model tree, wrong expert
    # count, wrong stage) before the subprocess consumes the artifact.
    expected_experts = (
        step.method.expected_num_experts
        if step.method.stage2_source == "self"
        else FINAL_EXPERT_CONTRACT
    )
    verify_checkpoint_contract(
        ckpt,
        expected_model_name=model,
        expected_num_experts=expected_experts,
        expected_stage=stage,
    )
    # Exact task match from the provider run's own config.
    is_final = is_final_provider(step.method.stage2_source)
    if step.method.task is not None:
        try:
            verify_provider_task(ckpt, expected_task=step.method.task)
        except CheckpointError:
            if is_final:
                raise
            # Historical sources: runs predating resolved_config.yaml
            # sidecars stay resolvable (frozen behavior); anything carrying
            # the sidecar is still checked.
            run_dir = ckpt.parent.parent
            if (run_dir / "resolved_config.yaml").is_file():
                raise
    # Config-identity gate for final providers.
    if is_final and protocol is not None:
        provider_name = provider_method_name(step.method.stage2_source)
        provider = (
            protocol.method_by_name(provider_name, task=step.method.task)
            if provider_name is not None
            else None
        )
        if provider is None:  # pragma: no cover - validated at load time
            raise CheckpointError(
                f"{step.method.name} stage 2 provider "
                f"{step.method.stage2_source!r} is missing from the protocol."
            )
        _check_provider_config_identity(protocol, step, provider, ckpt, outputs_base)
    return ckpt


def _auto_dependency_provider(
    step: Step, protocol: Protocol, args: argparse.Namespace
) -> Method | None:
    """Return the Stage 1 provider the runner should auto-train, or ``None``.

    Mirrors the plan-level dependency policy: providers are the default
    Stage 1 cells of the consumer's task (historical ``bc``/``phaseforge``
    or the final ``*_stage1`` identities resolved through
    :func:`provider_method_name`), and explicit scoping (``--stage``,
    ``--eval-only``) disables injection so a deliberately narrowed sweep
    still fails pre-flight.
    """
    if args.eval_only or args.stage is not None:
        return None
    if step.kind != "train" or step.stage != 2:
        return None
    req = step.required_checkpoint()
    if req is None:
        return None
    model, stage = req
    if stage != 1:
        return None
    source = step.method.stage2_source
    if source == "self":
        # Only self-sourced Stage-1 providers re-inject their own checkpoint.
        if model not in ("bc", "precision_residual_phaseforge", "final_aligned_static_rule"):
            return None
        return protocol.method_by_name(model, task=step.method.task)
    provider_name = provider_method_name(source)
    if provider_name is None:
        return None
    return protocol.method_by_name(provider_name, task=step.method.task)


def _run_dependency_step(
    step: Step,
    protocol: Protocol,
    outputs_base: Path,
    state: RunnerState,
    args: argparse.Namespace,
    toolhang_python: Path | None,
    expected_commit: str | None = None,
) -> None:
    """Execute one auto-injected Stage 1 dependency step and record it."""
    print(f"\n[runner] auto-injecting missing dependency: {step.label}", flush=True)
    try:
        run_step(
            step,
            ckpt_path=None,
            outputs_base=outputs_base,
            defaults=protocol.defaults,
            cwd=PROJECT_ROOT,
            log_level="INFO" if args.verbose else "WARNING",
            toolhang_python=toolhang_python,
        )
    except CommandError as exc:
        raise CheckpointError(f"Auto-injected dependency {step.label} failed: {exc}") from exc
    if step.stage is None:  # pragma: no cover - guard for type checkers
        raise CheckpointError(f"Dependency step {step.label} has no stage.")
    run_dir = resolve_run_dir(
        outputs_base,
        step.method.model_name,
        step.stage,
        seed=step.seed,
        tag=step.method.output_tag,
        expected_commit=expected_commit,
    )
    ckpt_rel = stage_checkpoint_relative(
        outputs_base, run_dir, step.method.model_name, step.stage
    )
    state.mark(
        step.method.phase_key,
        step.seed,
        step.registry_phase,
        run_dir=run_dir.relative_to(outputs_base).as_posix(),
        ckpt=ckpt_rel,
    )


def _resolve_stage2_with_auto_dependency(
    step: Step,
    protocol: Protocol,
    outputs_base: Path,
    state: RunnerState,
    args: argparse.Namespace,
    toolhang_python: Path | None,
    expected_commit: str | None = None,
) -> Path | None:
    """Resolve a stage-2 prerequisite, auto-training a missing provider first.

    In an unscoped sweep, a stage-2 step whose Stage 1 provider checkpoint is
    absent from the output tree no longer fails pre-flight: the runner trains
    the provider's Stage 1 (a dependency step) and then resolves the consumer
    exactly as before. A provider that already exists is never retrained, and
    providers of explicitly scoped runs (``--stage``/``--eval-only``) still
    fail loudly. Returns ``None`` for steps that load no prerequisite.
    """
    try:
        return _require_stage2_prereq(step, outputs_base, expected_commit, protocol)
    except CheckpointError:
        provider = _auto_dependency_provider(step, protocol, args)
        if provider is None:
            raise
        dep = Step(kind="train", method=provider, seed=step.seed, stage=1, dependency=True)
        _run_dependency_step(
            dep, protocol, outputs_base, state, args, toolhang_python, expected_commit
        )
        return _require_stage2_prereq(step, outputs_base, expected_commit, protocol)


def _eval_target(
    step: Step, outputs_base: Path, state: RunnerState, expected_commit: str | None = None
) -> Path:
    ckpt = resolve_checkpoint_path(
        outputs_base,
        step.method,
        step.method.final_stage,
        seed=step.seed,
        state=state,
        expected_commit=expected_commit,
    )
    # Fail closed before evaluation consumes a pre-final or cross-model
    # artifact (same filesystem name, wrong contract).
    verify_checkpoint_contract(
        ckpt,
        expected_model_name=step.method.model_name,
        expected_num_experts=step.method.expected_num_experts,
        expected_stage=step.method.final_stage,
    )
    return ckpt


def _should_skip_eval(
    step: Step,
    outputs_base: Path,
    state: RunnerState,
    force: bool,
    expected_commit: str | None = None,
) -> bool:
    if force:
        return False
    entry = state.get(step.method.phase_key, step.seed, "eval")
    if entry is None or entry.get("status") != "completed":
        return False
    if not state.is_complete(step.method.phase_key, step.seed, "eval"):
        return False
    try:
        target = _eval_target(step, outputs_base, state, expected_commit)
    except CheckpointError:
        return False
    return entry.get("ckpt") == target.relative_to(outputs_base).as_posix()


def _print_plan(
    plan: list[Step],
    outputs_base: Path,
    state: RunnerState,
    args: argparse.Namespace,
    expected_commit: str | None = None,
) -> None:
    print(f"\n[runner] plan ({len(plan)} steps, outputs base: {outputs_base})")
    print(f"[runner] steps: {_plan_breakdown(plan)}")
    for index, step in enumerate(plan, start=1):
        tag = " [dependency]" if step.dependency else ""
        status = (
            "done"
            if _step_done(step, outputs_base, state, args.force, expected_commit)
            else "pending"
        )
        print(f"  {index:>3}. {step.label:<48} {status}{tag}")


def _plan_breakdown(plan: list[Step]) -> str:
    counts: dict[str, int] = {}
    deps = 0
    for step in plan:
        if step.kind == "eval":
            key = "eval"
        else:
            key = f"stage{step.stage}"
        counts[key] = counts.get(key, 0) + 1
        if step.dependency:
            deps += 1
    parts = " ".join(f"{key}={counts[key]}" for key in sorted(counts))
    return f"{parts} auto-dependencies={deps}"


def _print_selection(
    protocol: Protocol, selection: SelectionResult, manifest_sha: str
) -> None:
    print(
        f"\n[runner] selection: {len(selection.methods)} cells from {protocol.name} "
        f"(manifest sha256 {manifest_sha[:12]})"
    )
    print(format_selection_table(selection, protocol))


def _step_done(
    step: Step,
    outputs_base: Path,
    state: RunnerState,
    force: bool,
    expected_commit: str | None = None,
) -> bool:
    if force:
        return False
    if step.kind == "train":
        return state.is_complete(step.method.phase_key, step.seed, step.registry_phase)
    return _should_skip_eval(
        step, outputs_base, state, force=False, expected_commit=expected_commit
    )


def _print_methods(protocol: Protocol, rows: list[Method] | None = None) -> None:
    rows = list(protocol.methods) if rows is None else list(rows)
    print(f"Protocol: {protocol.name} — {protocol.task} — {protocol.description}")
    for method in rows:
        stages = ",".join(f"stage{s}" for s in method.stages)
        print(
            f"  {method.index:>2}  {method.name:<32} {method.model:<40} "
            f"data={method.data:<10} {stages:<12} "
            f"eval={'yes' if method.evaluate else 'no'}"
        )
    print(f"seeds: {list(protocol.seeds)}")


def _print_summary(protocol: Protocol, state: RunnerState, counts: dict[str, int]) -> None:
    print("\n[runner] summary")
    print(f"  ran={counts['run']} skipped={counts['skip']} failed={counts['failed']}")
    for method in protocol.methods:
        phases = [f"stage{s}" for s in method.stages]
        if method.evaluate:
            phases.append("eval")
        cells: list[str] = []
        for seed in protocol.seeds:
            statuses = []
            for phase in phases:
                entry = state.get(method.phase_key, seed, phase)
                if state.is_complete(method.phase_key, seed, phase):
                    statuses.append("ok")
                elif entry is not None and entry.get("status") == "failed":
                    statuses.append("FAIL")
                else:
                    statuses.append("-")
            cells.append(f"seed{seed}=[{' '.join(statuses)}]")
        print(f"  {method.name:<32} {'  '.join(cells)}")


def _current_commit() -> str | None:
    """Return the current git commit, or ``None`` when unavailable.

    Used as the commit gate for the state registry and checkpoint
    resolution: a sweep at revision X must not reuse artifacts recorded or
    produced at revision Y (e.g. pre-fix checkpoints) — that silently
    corrupts the sweep. When git is unavailable (no repo), gating is
    disabled and resolution falls back to "newest completed" behaviour.
    """
    commit = git_info().get("commit") or ""
    return commit.strip() or None


def _manifest_sha(args: argparse.Namespace) -> str:
    """SHA-256 of the manifest file (it loaded successfully, so it exists)."""
    return hashlib.sha256(_manifest_path(args).read_bytes()).hexdigest()


def _write_plan_artifact(
    outputs_base: Path,
    protocol: Protocol,
    args: argparse.Namespace,
    selection: SelectionResult,
    plan: list[Step],
    manifest_sha: str,
    commit: str | None,
) -> None:
    """Atomically write the resolved-sweep provenance record.

    ``<outputs>/_runner/plan.json`` describes the *latest* invocation: what
    was asked (argv + tokens), what it resolved to (cells, manifest hash),
    and the expected/executed step count. Together with ``state.json`` it
    makes every sweep reconstructible after the fact.
    """
    path = outputs_base / "_runner" / "plan.json"
    payload = {
        "version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": commit or "",
        "argv": list(sys.argv),
        "manifest": {"path": str(_manifest_path(args)), "sha256": manifest_sha},
        "selection": {
            "method_tokens": list(_split_list(args.methods)),
            "tasks": list(_split_list(args.tasks)),
            "seeds": list(_split_list(args.seeds)),
        },
        "resolved_cells": [
            {
                "index": m.index,
                "phase_key": m.phase_key,
                "model": m.model,
                "stages": list(m.stages),
            }
            for m in selection.methods
        ],
        "step_count": len(plan),
        "expect_steps": args.expect_steps,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".plan_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    print(f"[runner] plan artifact: {path.relative_to(outputs_base).as_posix()}")


def run(args: argparse.Namespace) -> int:
    outputs_base = _outputs_base(args)
    try:
        protocol = load_protocol(_manifest_path(args))
    except (ProtocolError, OSError) as exc:
        print(f"[runner] ERROR loading manifest: {exc}", file=sys.stderr)
        return 2

    if args.verify_gates:
        from phaseforge.runner.verify import format_gates_report, run_final_gates

        try:
            seeds = [int(s) for s in _split_list(args.seeds)] or None
        except ValueError:
            print(f"[runner] ERROR: --seeds must be integers: {args.seeds}", file=sys.stderr)
            return 2
        try:
            report = run_final_gates(_manifest_path(args), outputs_base, seeds=seeds)
        except ProtocolError as exc:
            print(f"[runner] ERROR: {exc}", file=sys.stderr)
            return 2
        print(format_gates_report(report))
        failed = [g for g, r in report["gates"].items() if r["status"] != "pass"]
        return 2 if failed else 0

    if args.list:
        try:
            selection = resolve_selection(protocol, _selection_spec(args))
        except ProtocolError as exc:
            print(f"[runner] ERROR: {exc}", file=sys.stderr)
            return 2
        _print_methods(protocol, list(selection.methods))
        return 0

    commit = None if args.no_commit_gate else _current_commit()
    if commit is None:
        if args.no_commit_gate:
            print(
                "[runner] commit gating disabled via --no-commit-gate; reusing "
                "completed checkpoints from previous revisions.",
                file=sys.stderr,
            )
        else:
            print(
                "[runner] WARNING: git unavailable — commit gating disabled; stale "
                "pre-fix checkpoints will NOT be filtered. Verify the tree is clean "
                "at the intended revision before trusting results.",
                file=sys.stderr,
            )
    else:
        print(f"[runner] commit gate: {commit}")

    try:
        state = RunnerState(RunnerState.default_path(outputs_base), expected_commit=commit)
    except RegistryError as exc:
        print(f"[runner] ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        plan, selection = _build_plan(protocol, args)
    except ProtocolError as exc:
        print(f"[runner] ERROR: {exc}", file=sys.stderr)
        return 2

    if not plan:
        print("[runner] No steps match the current selection.", file=sys.stderr)
        return 0

    manifest_sha = _manifest_sha(args)
    _print_selection(protocol, selection, manifest_sha)
    _print_plan(plan, outputs_base, state, args, commit)

    if args.expect_steps is not None and args.expect_steps != len(plan):
        print(
            f"[runner] ERROR: --expect-steps {args.expect_steps} but the plan has "
            f"{len(plan)} steps — refusing to run (manifest or selection drift).",
            file=sys.stderr,
        )
        return 2

    toolhang_python: Path | None = None
    has_toolhang = any(step.method.task == "ToolHang" for step in plan)
    if has_toolhang and not args.dry_run:
        try:
            toolhang_python = resolve_toolhang_python(PROJECT_ROOT, args.toolhang_python)
            preflight_toolhang_python(toolhang_python)
            print(f"[runner] Tool Hang interpreter: {toolhang_python}")
        except CommandError as exc:
            print(f"[runner] ERROR: {exc}", file=sys.stderr)
            return 2

    if not args.dry_run:
        try:
            _write_plan_artifact(
                outputs_base, protocol, args, selection, plan, manifest_sha, commit
            )
        except OSError as exc:
            # An unrecordable sweep must not start: the artifact is the
            # provenance guarantee every later audit relies on.
            print(
                f"[runner] ERROR: cannot write plan artifact under {outputs_base}: {exc}",
                file=sys.stderr,
            )
            return 2

    counts = {"run": 0, "skip": 0, "failed": 0}
    total = len(plan)
    for index, step in enumerate(plan, start=1):
        prefix = f"[{index}/{total}]"
        try:
            if _step_done(step, outputs_base, state, args.force, commit):
                print(f"{prefix} skip {step.label} (already completed)")
                counts["skip"] += 1
                continue

            if args.dry_run:
                _print_dry_run(step, protocol, outputs_base, state, args, commit)
                counts["run"] += 1
                continue

            ckpt_abs: Path | None = None
            if step.kind == "eval":
                ckpt_abs = _eval_target(step, outputs_base, state, commit)
            else:
                ckpt_abs = _resolve_stage2_with_auto_dependency(
                    step, protocol, outputs_base, state, args, toolhang_python, commit
                )

            run_step(
                step,
                ckpt_path=ckpt_abs,
                outputs_base=outputs_base,
                defaults=protocol.defaults,
                cwd=PROJECT_ROOT,
                log_level="INFO" if args.verbose else "WARNING",
                toolhang_python=toolhang_python,
            )

            if step.kind == "eval":
                if ckpt_abs is None:  # pragma: no cover - guard for type checkers
                    raise CheckpointError(f"Eval step {step.label} has no checkpoint.")
                run_dir = resolve_eval_run_dir(
                    outputs_base,
                    step.method.model_name,
                    seed=step.seed,
                    tag=step.method.output_tag,
                    expected_commit=commit,
                )
                state.mark(
                    step.method.phase_key,
                    step.seed,
                    "eval",
                    ckpt=ckpt_abs.relative_to(outputs_base).as_posix(),
                    run_dir=run_dir.relative_to(outputs_base).as_posix(),
                )
            else:
                if step.stage is None:  # pragma: no cover - guard for type checkers
                    raise CheckpointError(f"Train step {step.label} has no stage.")
                run_dir = resolve_run_dir(
                    outputs_base,
                    step.method.model_name,
                    step.stage,
                    seed=step.seed,
                    tag=step.method.output_tag,
                    expected_commit=commit,
                )
                ckpt_rel = stage_checkpoint_relative(
                    outputs_base, run_dir, step.method.model_name, step.stage
                )
                state.mark(
                    step.method.phase_key,
                    step.seed,
                    step.registry_phase,
                    run_dir=run_dir.relative_to(outputs_base).as_posix(),
                    ckpt=ckpt_rel,
                )
            counts["run"] += 1
            print(f"{prefix} OK {step.label}")
        except (CheckpointError, CommandError) as exc:
            state.mark_failed(step.method.phase_key, step.seed, step.registry_phase, str(exc))
            counts["failed"] += 1
            print(f"{prefix} FAILED {step.label}: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                _print_summary(protocol, state, counts)
                return 1

    _print_summary(protocol, state, counts)
    return 0 if counts["failed"] == 0 else 1


def _print_dry_run(
    step: Step,
    protocol: Protocol,
    outputs_base: Path,
    state: RunnerState,
    args: argparse.Namespace,
    expected_commit: str | None = None,
) -> None:
    try:
        ckpt_abs: Path | None = None
        if step.kind == "eval":
            ckpt_abs = _eval_target(step, outputs_base, state, expected_commit)
        else:
            ckpt_abs = _require_stage2_prereq(step, outputs_base, expected_commit, protocol)
        cmd = step_command(
            step, ckpt_path=ckpt_abs, outputs_base=outputs_base, defaults=protocol.defaults
        )
        print(f"  [dry-run] WOULD RUN  {step.label}")
        print(f"    $ {' '.join(cmd)}")
    except CheckpointError as exc:
        provider = _auto_dependency_provider(step, protocol, args)
        if provider is not None:
            dep = Step(kind="train", method=provider, seed=step.seed, stage=1, dependency=True)
            cmd = step_command(
                dep, ckpt_path=None, outputs_base=outputs_base, defaults=protocol.defaults
            )
            print(f"  [dry-run] AUTO-INJECT dependency: {dep.label}")
            print(f"    $ {' '.join(cmd)}")
            return
        print(f"  [dry-run] BLOCKED  {step.label} — prerequisite missing: {exc}")


def main() -> None:
    args = parse_args()
    try:
        exit_code = run(args)
    except KeyboardInterrupt:
        print("\n[runner] Interrupted by user.", file=sys.stderr)
        exit_code = 130
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

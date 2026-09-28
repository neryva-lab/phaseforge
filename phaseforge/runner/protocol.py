"""Locked experiment protocol: load + validate the JSON matrix, build the plan.

The final protocol manifest is the single source of truth describing every
method, its training stages, its Stage 1 source dependency, and whether a
complete run includes evaluation. ``load_protocol`` validates it loudly so a
malformed matrix never silently produces a partial sweep.

A *plan* is the ordered list of concrete steps produced by
:func:`build_plan`: for every selected method and seed, all training stages
in order, then the evaluation of the final-stage checkpoint, with any
required pretraining stages injected first when ``with_dependencies`` is set.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

_VALID_STAGES = frozenset({1, 2})
# Per-task data configs (lift, can, square, tool_hang, transport) plus the
# two task-agnostic variants (common, robot_only). Each maps to a YAML
# file under ``phaseforge/config/data/`` whose ``source.task_name`` drives
# the ingestion path for that benchmark task.
_VALID_DATA = frozenset(
    {
        "common",
        "robot_only",
        "robot_only_lift",
        "robot_only_can",
        "robot_only_square",
        "robot_only_tool_hang",
        "robot_only_transport",
        "lift",
        "can",
        "square",
        "tool_hang",
        "transport",
    }
)
_VALID_EVAL_MODES = frozenset({"rollout", "offline"})

#: Final provider identities: explicit Stage 1 sources for the
#: final causal matrix. Each maps to ``(provider method name, provider model
#: name)``. The model name is the filesystem/config identity consumed via
#: ``train.stage1_ckpt_path``; the method name is the manifest identity
#: whose Stage 1 cell is the provider (used for cross-checks, plan
#: dependencies, and runner auto-injection).
_FINAL_STAGE1_PROVIDERS: dict[str, tuple[str, str]] = {
    "precision_residual_phaseforge_stage1": (
        "precision_residual_phaseforge",
        "precision_residual_phaseforge",
    ),
    "final_aligned_bc_stage1": ("bc", "final_aligned_bc"),
    "final_aligned_static_rule_stage1": (
        "final_aligned_static_rule",
        "final_aligned_static_rule",
    ),
}

#: Method-name prefixes marking final-family rows.
_FINAL_METHOD_PREFIXES = ("precision_residual_", "final_aligned_")


def is_final_method(name: str) -> bool:
    """Return whether a method name belongs to the final matrix family."""
    return str(name).startswith(_FINAL_METHOD_PREFIXES)


def is_final_provider(stage2_source: str | None) -> bool:
    """Return whether a Stage 2 source is an explicit final provider identity."""
    return stage2_source in _FINAL_STAGE1_PROVIDERS


def provider_method_name(stage2_source: str | None) -> str | None:
    """Return the manifest method name providing a Stage 2 source.

    ``"self"`` is resolved by the caller (the consumer's own method); final
    identities resolve through :data:`_FINAL_STAGE1_PROVIDERS`. Returns
    ``None`` for a null source (Stage 1 / provider-less rows).
    """
    if stage2_source is None or stage2_source == "self":
        return None
    if stage2_source in _FINAL_STAGE1_PROVIDERS:
        return _FINAL_STAGE1_PROVIDERS[stage2_source][0]
    return stage2_source


def provider_model_name(stage2_source: str | None, consumer_model_name: str) -> str | None:
    """Return the model (filesystem/config) identity of a Stage 2 source.

    Mirrors :func:`provider_method_name` but yields the model name consumed
    via ``train.stage1_ckpt_path`` (which differs from the method name for
    the BC provider: method ``bc``, model ``final_aligned_bc``).
    """
    if stage2_source is None:
        return None
    if stage2_source == "self":
        return consumer_model_name
    if stage2_source in _FINAL_STAGE1_PROVIDERS:
        return _FINAL_STAGE1_PROVIDERS[stage2_source][1]
    return stage2_source


def verify_output_namespace(outputs_base: str | Path) -> Path:
    """Fail closed unless an output namespace is fresh.

    A namespace is fresh when its directory is absent or contains no run
    artifacts (no ``*.completed`` markers, no ``checkpoint_best.pt``, no
    runner state). Final sweeps must target a fresh namespace so
    historical outputs can never be overwritten or mixed into the final
    matrix. Raises :class:`ProtocolError` describing what was found.
    """
    base = Path(outputs_base)
    if not base.exists():
        return base.resolve()
    offenders: list[str] = []
    try:
        markers = sorted(
            str(p.relative_to(base))
            for p in base.rglob("*.completed")
            if p.is_file()
        )
        offenders.extend(markers[:8])
        if not markers:
            ckpts = sorted(
                str(p.relative_to(base))
                for p in base.rglob("checkpoint_best.pt")
                if p.is_file()
            )
            offenders.extend(ckpts[:8])
        state = base / "_runner" / "state.json"
        if state.is_file():
            offenders.append("_runner/state.json")
    except OSError as exc:
        raise ProtocolError(
            f"Cannot inspect output namespace {base}: {exc}. Refusing to run."
        ) from exc
    if offenders:
        shown = ", ".join(offenders)
        extra = "" if len(offenders) < 8 else ", ..."
        raise ProtocolError(
            f"Output namespace {base} is not fresh (found {shown}{extra}). "
            "Final sweeps require a fresh namespace: choose an absent or "
            "empty --outputs directory so historical artifacts cannot be "
            "overwritten."
        )
    return base.resolve()


class ProtocolError(ValueError):
    """Raised when a protocol manifest is malformed or inconsistent."""


@dataclass(frozen=True)
class Method:
    """One baseline method in the protocol."""

    index: int
    name: str
    role: str
    model: str
    data: str
    stages: tuple[int, ...]
    stage2_source: str | None
    evaluate: bool
    tag: str | None = None
    evaluate_mode: str = "rollout"
    task: str | None = None
    experiment_id: str | None = None
    overrides: tuple[str, ...] = ()

    @property
    def model_name(self) -> str:
        """Filesystem model name (e.g. ``baselines/bc`` -> ``bc``)."""
        return self.model.rsplit("/", 1)[-1]

    @property
    def final_stage(self) -> int:
        """The stage whose checkpoint a complete run evaluates."""
        return self.stages[-1]

    @property
    def phase_key(self) -> str:
        """Registry key prefix for this method's phases (stable identity).

        For the five-task protocol, methods trained on different tasks share
        the same ``name`` but must have distinct output trees. Including the
        task in the phase_key prevents cross-task checkpoint collisions.
        """
        if self.task:
            return f"{self.task}/{self.name}"
        return self.name

    @property
    def output_tag(self) -> str | None:
        """Tag used to isolate this method's artifacts from other tasks."""
        parts: list[str] = []
        if self.task:
            parts.append(self.task)
        if self.tag:
            parts.append(self.tag)
        return "__".join(parts) if parts else None

    @property
    def expected_num_experts(self) -> int:
        """The number of MoE experts expected for this method (defaults to 6).

        Parsed from ``models.router.num_experts=<N>`` in ``self.overrides`` if
        specified (such as the K-sweep capacity ablations ``pf_k3`` -> 3 or
        ``pf_k12`` -> 12); otherwise defaults to the canonical 6 experts.
        Dense models skip expert verification in the verifier.
        """
        for ov in self.overrides:
            if ov.startswith("models.router.num_experts="):
                return int(ov.split("=", 1)[1])
        return 6


@dataclass(frozen=True)
class Protocol:
    """The validated protocol document."""

    name: str
    task: str
    description: str
    seeds: tuple[int, ...]
    defaults: tuple[str, ...]
    methods: tuple[Method, ...]

    def method_by_name(self, name: str, *, task: str | None = None) -> Method | None:
        for m in self.methods:
            if m.name == name and (task is None or m.task == task):
                return m
        return None

    def method_by_index(self, index: int, *, task: str | None = None) -> Method | None:
        for m in self.methods:
            if m.index == index and (task is None or m.task == task):
                return m
        return None

    @property
    def known_tasks(self) -> tuple[str, ...]:
        """The concrete task dimension of this protocol, in manifest order.

        Returns the per-row task values when rows carry them (the final
        multi-task protocol); otherwise falls back to the protocol-level task
        name for single-task manifests, ignoring the multi-task placeholder
        ``"all"``. Empty when the protocol declares no task at
        all — such manifests reject a ``--tasks`` filter loudly instead of
        silently selecting nothing.
        """
        tasks: list[str] = []
        seen: set[str] = set()
        for m in self.methods:
            if m.task is None or m.task.lower() in seen:
                continue
            seen.add(m.task.lower())
            tasks.append(m.task)
        if not tasks:
            nominal = self.task.strip()
            if nominal and nominal.lower() != "all":
                tasks.append(nominal)
        return tuple(tasks)

    def select_methods(self, refs: list[str]) -> list[Method]:
        """Resolve a list of user references to ordered :class:`Method`s.

        Each reference may be a method name or its numeric index. Unknown
        references raise a :class:`ProtocolError` listing every valid option.
        """
        selected: list[Method] = []
        seen: set[tuple[str | None, str]] = set()
        for ref in refs:
            method: Method | None = None
            # isdecimal, not isdigit: superscript digits satisfy isdigit but
            # crash int() with a ValueError that is not a ProtocolError.
            if ref.isdecimal():
                method = self.method_by_index(int(ref))
                if method is None:
                    raise ProtocolError(
                        f"Unknown method index {ref!r}. Valid indices: "
                        f"{[m.index for m in self.methods]}."
                    )
            else:
                matches = [m for m in self.methods if m.name == ref]
                if not matches:
                    raise ProtocolError(
                        f"Unknown method {ref!r}. Valid names: {[m.name for m in self.methods]}."
                    )
                if len(matches) > 1:
                    tasks = [m.task or "default" for m in matches]
                    raise ProtocolError(
                        f"Method name {ref!r} is ambiguous across tasks {tasks}; "
                        "select it by manifest index."
                    )
                method = matches[0]
            identity = (method.task, method.name)
            if identity not in seen:
                selected.append(method)
                seen.add(identity)
        return sorted(selected, key=lambda m: m.index)


def _require(data: dict[str, Any], key: str, label: str) -> Any:
    if key not in data or data[key] in (None, ""):
        raise ProtocolError(f"Protocol method is missing {label} ({key!r}).")
    return data[key]


def _parse_method(raw: dict[str, Any]) -> Method:
    if not isinstance(raw, dict):
        raise ProtocolError(f"Protocol method must be a JSON object, got {type(raw).__name__}.")

    index = raw.get("index")
    if not isinstance(index, int) or isinstance(index, bool) or index < 1:
        raise ProtocolError(f"Method {raw.get('name', '?')!r}: 'index' must be an int >= 1.")
    name = _require(raw, "name", "a name")
    if not isinstance(name, str) or not name.replace("_", "").isalnum():
        raise ProtocolError(f"Method {index}: 'name' must be [a-z0-9_]+ (got {name!r}).")
    role = str(_require(raw, "role", "a role description"))
    model = _require(raw, "model", "a model config path")
    if not isinstance(model, str) or not model:
        raise ProtocolError(f"Method {name!r}: 'model' must be a non-empty string.")
    data = str(raw.get("data", "common"))
    if data not in _VALID_DATA:
        raise ProtocolError(
            f"Method {name!r}: 'data' must be one of {sorted(_VALID_DATA)}, got {data!r}."
        )

    stages_raw = raw.get("stages")
    if not isinstance(stages_raw, list) or not stages_raw:
        raise ProtocolError(f"Method {name!r}: 'stages' must be a non-empty list.")
    stages: list[int] = []
    for s in stages_raw:
        if isinstance(s, bool) or not isinstance(s, int) or s not in _VALID_STAGES:
            raise ProtocolError(
                f"Method {name!r}: stage {s!r} must be one of {sorted(_VALID_STAGES)}."
            )
        if s not in stages:
            stages.append(s)
    if stages != sorted(stages):
        raise ProtocolError(f"Method {name!r}: 'stages' must be ascending and unique.")

    stage2_source = raw.get("stage2_source")
    if stage2_source is not None:
        allowed_sources = frozenset({"self"}) | frozenset(_FINAL_STAGE1_PROVIDERS)
        if stage2_source not in allowed_sources:
            raise ProtocolError(
                f"Method {name!r}: 'stage2_source' must be null, 'self' or one of "
                f"{sorted(_FINAL_STAGE1_PROVIDERS)}, "
                f"got {stage2_source!r}."
            )
        if 2 not in stages:
            raise ProtocolError(f"Method {name!r}: 'stage2_source' set but method has no stage 2.")

    evaluate = raw.get("evaluate", True)
    if not isinstance(evaluate, bool):
        raise ProtocolError(f"Method {name!r}: 'evaluate' must be a boolean.")

    tag = raw.get("tag")
    if tag is not None and (not isinstance(tag, str) or not tag):
        raise ProtocolError(f"Method {name!r}: 'tag' must be a non-empty string or null.")

    evaluate_mode = str(raw.get("evaluate_mode", "rollout"))
    if evaluate_mode not in _VALID_EVAL_MODES:
        raise ProtocolError(
            f"Method {name!r}: 'evaluate_mode' must be one of "
            f"{sorted(_VALID_EVAL_MODES)}, got {evaluate_mode!r}."
        )

    task = raw.get("task")
    if task is not None and (not isinstance(task, str) or not task):
        raise ProtocolError(f"Method {name!r}: 'task' must be a non-empty string or null.")

    experiment_id = raw.get("experiment_id")
    if experiment_id is not None and (not isinstance(experiment_id, str) or not experiment_id):
        raise ProtocolError(f"Method {name!r}: 'experiment_id' must be a non-empty string or null.")

    overrides_raw = raw.get("overrides", [])
    if not isinstance(overrides_raw, list) or not all(isinstance(x, str) for x in overrides_raw):
        raise ProtocolError(f"Method {name!r}: 'overrides' must be a list of strings.")

    return Method(
        index=index,
        name=name,
        role=role,
        model=model,
        data=data,
        stages=tuple(stages),
        stage2_source=stage2_source,
        evaluate=evaluate,
        tag=tag,
        evaluate_mode=evaluate_mode,
        task=task,
        experiment_id=experiment_id,
        overrides=tuple(overrides_raw),
    )


def load_protocol(path: str | Path) -> Protocol:
    """Load and strictly validate a protocol manifest.

    Raises:
        ProtocolError: The document is malformed (bad JSON, missing keys,
            duplicate indices, non-int seeds, ...).
        OSError: The file cannot be read.
    """
    p = Path(path)
    if p.is_dir():
        main_json = p / "main.json"
        if main_json.exists():
            p = main_json
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"Protocol {p} is not valid JSON: {exc.msg}.") from exc
    if not isinstance(raw, dict):
        raise ProtocolError(f"Protocol {p} must be a JSON object.")

    name = _require(raw, "name", "a protocol name")
    task = str(_require(raw, "task", "a task name"))
    description = str(raw.get("description", ""))

    seeds_raw = raw.get("seeds")
    if not isinstance(seeds_raw, list) or not seeds_raw:
        raise ProtocolError(f"Protocol {name!r}: 'seeds' must be a non-empty list of ints.")
    seeds: list[int] = []
    for s in seeds_raw:
        if isinstance(s, bool) or not isinstance(s, int):
            raise ProtocolError(f"Protocol {name!r}: seed {s!r} must be an int.")
        if s not in seeds:
            seeds.append(s)

    defaults_raw = raw.get("defaults", [])
    if not isinstance(defaults_raw, list) or not all(
        isinstance(d, str) and d for d in defaults_raw
    ):
        raise ProtocolError(f"Protocol {name!r}: 'defaults' must be a list of override strings.")
    defaults = tuple(defaults_raw)

    methods_raw = raw.get("methods")
    manifests_ref = raw.get("manifests") or raw.get("includes")

    if not methods_raw and not manifests_ref:
        raise ProtocolError(f"Protocol {name!r}: either 'methods' or 'manifests' must be provided.")

    if manifests_ref and not methods_raw:
        if isinstance(manifests_ref, dict):
            sub_paths = list(manifests_ref.values())
        elif isinstance(manifests_ref, list):
            sub_paths = manifests_ref
        else:
            raise ProtocolError(f"Protocol {name!r}: 'manifests' must be a list or dict of paths.")

        sub_methods: list[Method] = []
        sub_dir = p.parent
        global_idx = 1
        for sub_p_str in sub_paths:
            sub_p = Path(sub_p_str)
            if not sub_p.is_absolute():
                sub_p = sub_dir / sub_p
            if not sub_p.exists():
                raise ProtocolError(
                    f"Protocol {name!r}: referenced manifest {sub_p} does not exist."
                )
            sub_proto = load_protocol(sub_p)
            for m in sub_proto.methods:
                eff_task = m.task or (sub_proto.task if sub_proto.task != "all" else None)
                m_updated = replace(m, index=global_idx, task=eff_task)
                global_idx += 1
                sub_methods.append(m_updated)
        methods = sub_methods
    else:
        if not isinstance(methods_raw, list) or not methods_raw:
            raise ProtocolError(f"Protocol {name!r}: 'methods' must be a non-empty list.")
        methods = [_parse_method(m) for m in methods_raw]
    seen_indices: dict[tuple[str | None, int], str] = {}
    seen_names: set[tuple[str | None, str]] = set()
    for m in methods:
        task_key: str | None = m.task
        # Method identity is (task, name) under the final multi-task
        # protocol: the same method name appears once per task. Single-task
        # protocols leave ``task`` null and the identity collapses to
        # ``name``.
        identity: tuple[str | None, str] = (task_key, m.name)
        index_key: tuple[str | None, int] = (task_key, m.index)
        if index_key in seen_indices:
            raise ProtocolError(
                f"Duplicate method index {m.index} for task {task_key!r} "
                f"({m.name!r} vs {seen_indices[index_key]!r})."
            )
        if identity in seen_names:
            raise ProtocolError(
                f"Duplicate method name {m.name!r}"
                + (f" for task {task_key!r}" if task_key else ".")
            )
        seen_indices[index_key] = m.name
        seen_names.add(identity)
    methods.sort(key=lambda m: (m.task is None, m.task or "", m.index))

    # Cross-checks: a stage2_source provider must be a real method in the
    # SAME task, and the provided stage 1 must come from the default
    # (no variant tag) cell of that task — otherwise dependency injection
    # would point at the wrong output tree. Final provider identities
    # resolve to their manifest method through provider_method_name().
    for m in methods:
        if m.stage2_source is None or m.stage2_source == "self":
            continue
        provider_name = provider_method_name(m.stage2_source)
        assert provider_name is not None
        provider = next(
            (p for p in methods if p.name == provider_name and p.task == m.task),
            None,
        )
        if provider is None:
            raise ProtocolError(
                f"Method {m.name!r} (task={m.task!r}): 'stage2_source' "
                f"{m.stage2_source!r} is not a method in this task."
            )
        if 1 not in provider.stages:
            raise ProtocolError(
                f"Method {m.name!r}: provider {provider.name!r} has no stage 1."
            )
        if provider.tag is not None:
            raise ProtocolError(
                f"Method {m.name!r}: provider {provider.name!r} must be the default "
                "(common-data) cell, but it carries a variant tag."
            )

    return Protocol(
        name=str(name),
        task=task,
        description=description,
        seeds=tuple(seeds),
        defaults=defaults,
        methods=tuple(methods),
    )


@dataclass(frozen=True)
class Step:
    """One concrete run step: a training stage or the evaluation of a method."""

    kind: str
    method: Method
    seed: int
    stage: int | None = None
    dependency: bool = False

    @property
    def label(self) -> str:
        if self.kind == "eval":
            return f"{self.method.name} seed={self.seed} eval"
        return f"{self.method.name} seed={self.seed} stage{self.stage}"

    @property
    def registry_phase(self) -> str:
        """Registry phase name (``stage1``/``stage2``/``eval``)."""
        if self.kind == "eval":
            return "eval"
        return f"stage{self.stage}"

    @property
    def phase_key(self) -> str:
        """Registry phase key (stable identity for resume/skip logic)."""
        return f"{self.method.phase_key}/{self.seed}/{self.registry_phase}"

    def required_checkpoint(self) -> tuple[str, int] | None:
        """The ``(model, stage)`` artifact this step must load, or ``None``.

        * A stage-2 step loads the Stage 1 checkpoint of its provider
          (``"self"`` -> this method's own stage 1; a historical alias or a
          final ``*_stage1`` identity -> the mapped provider model's
          stage 1).
        * An eval step loads the method's final-stage checkpoint.
        * A stage-1 step loads nothing.
        """
        if self.kind == "eval":
            return (self.method.model_name, self.method.final_stage)
        if self.kind == "train" and self.stage == 2:
            source = self.method.stage2_source
            if source is None:
                return None
            model = provider_model_name(source, self.method.model_name)
            if model is None:  # pragma: no cover - validated at load time
                raise ProtocolError(
                    f"Method {self.method.name!r}: cannot resolve Stage 1 "
                    f"provider for 'stage2_source' {source!r}."
                )
            return (model, 1)
        return None


def _validate_seed_filter(protocol: Protocol, seeds_filter: list[int] | None) -> list[int]:
    if not seeds_filter:
        return list(protocol.seeds)
    for s in seeds_filter:
        if s not in protocol.seeds:
            raise ProtocolError(
                f"Seed {s} is not in the protocol ({list(protocol.seeds)}). "
                "Adjust the protocol manifest or choose a protocol seed."
            )
    return list(seeds_filter)


def build_plan(
    protocol: Protocol,
    methods: list[Method],
    *,
    seeds: list[int] | None = None,
    stage: int | None = None,
    eval_only: bool = False,
    skip_eval: bool = False,
    with_dependencies: bool = False,
) -> list[Step]:
    """Build the ordered plan of steps for the selected methods and seeds.

    Ordering: methods by manifest index (so a provider runs before its
    consumers in a full sweep); within a method, seeds in protocol order and
    each seed's training stages ascending, followed by the method's eval.
    When ``with_dependencies`` is set and a selected method's Stage 1
    provider is not itself selected, the provider's Stage 1 (train only,
    marked ``dependency``) is prepended so a partial selection still runs to
    completion. Stage filtering disables dependency injection (the scope was
    explicitly narrowed; a missing prerequisite then fails pre-flight). The
    CLI additionally auto-injects a missing provider at runtime for unscoped
    sweeps, so a partial selection completes even without ``--with-dependencies``.
    """
    if eval_only and stage is not None:
        raise ProtocolError("--stage and --eval-only are mutually exclusive.")
    if eval_only and skip_eval:
        raise ProtocolError("--eval-only and --skip-eval are mutually exclusive.")
    if stage is not None and stage not in (1, 2):
        raise ProtocolError(f"--stage must be 1 or 2, got {stage}.")

    seed_list = _validate_seed_filter(protocol, seeds)
    selected_identities = {(m.task, m.name) for m in methods}

    deps: list[Step] = []
    if with_dependencies and not eval_only and stage is None:
        for m in methods:
            if m.stage2_source is None or m.stage2_source == "self":
                continue
            provider_name = provider_method_name(m.stage2_source)
            if provider_name is None:  # pragma: no cover - validated at load
                continue
            provider_identity = (m.task, provider_name)
            if provider_identity not in selected_identities:
                provider = protocol.method_by_name(provider_name, task=m.task)
                if provider is None:
                    raise ProtocolError(
                        f"Method {m.name!r}: provider {m.stage2_source!r} is missing "
                        "from the protocol."
                    )
                for seed in seed_list:
                    deps.append(
                        Step(kind="train", method=provider, seed=seed, stage=1, dependency=True)
                    )
        seen: set[tuple[str | None, str, int]] = set()
        deduped: list[Step] = []
        for step in deps:
            key = (step.method.task, step.method.name, step.seed)
            if key not in seen:
                seen.add(key)
                deduped.append(step)
        deps = sorted(deduped, key=lambda s: (s.method.index, seed_list.index(s.seed)))

    steps: list[Step] = []
    for method in methods:
        for seed in seed_list:
            if eval_only:
                if method.evaluate:
                    steps.append(Step(kind="eval", method=method, seed=seed))
                continue
            for stage_n in method.stages:
                if stage is not None and stage_n != stage:
                    continue
                steps.append(Step(kind="train", method=method, seed=seed, stage=stage_n))
            if method.evaluate and not skip_eval and stage is None:
                steps.append(Step(kind="eval", method=method, seed=seed))

    return deps + steps

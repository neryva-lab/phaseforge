"""Config preflight for the full experiment matrix.

Composes every (method, task, stage, seed) cell from the frozen manifest
via Hydra's compose API (no CLI, so no argparse/Python-3.14 interaction)
and validates each cell's resolved config before any GPU compute is
spent. Catches config-level failures that would otherwise abort the sweep
mid-matrix:

* the cell composes at all (bad override, missing group, schema error),
* ``data`` group exists and matches the method's task,
* ``models`` group exists and ``models.name`` is set (resolution alias),
* ``num_phases`` consistency across data / model / phase head,
* the checkpoint ``monitor`` matches the predeclared protocol rule
  (stage 1: ``val/loss_action``, stage 2: ``val/loss_action``),
* stage-2 ``freeze_encoder`` is enabled and a stage-1 source is resolved,
* scheduler ``T_max`` covers the full epoch budget (no premature decay),
* ``eval`` group + ``eval.mode`` are consistent for evaluation cells.

Usage::

    uv run python scripts/protocol/preflight_configs.py                 # locked final matrix
    uv run python scripts/protocol/preflight_configs.py --methods bc     # subset
    uv run python scripts/protocol/preflight_configs.py --tasks Lift     # subset
    uv run python scripts/protocol/preflight_configs.py --manifest experiments/final_causal_matrix.json

Exit code is 0 when every cell passes; errors are collected and reported
per cell with the exact failing override.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import DictConfig
from omegaconf.errors import OmegaConfBaseException

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "experiments" / "final_causal_matrix.json"

MONITOR_BY_STAGE = {1: "val/loss_action", 2: "val/loss_action"}

@dataclass
class Cell:
    """One (method, task, stage, seed) cell, plus its preflight findings."""

    method: str
    model: str
    data: str
    task: str
    stage: int
    seed: int
    tag: str | None = None
    defaults: tuple[str, ...] = ()
    overrides: tuple[str, ...] = ()
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return (
            f"{self.method}/{self.task}/seed{self.seed}/stage{self.stage}"
            + (f" (tag={self.tag})" if self.tag else "")
        )

    def fail(self, msg: str) -> None:
        self.errors.append(msg)


def _compose_cell(cell: Cell) -> DictConfig | None:
    """Compose the Hydra config for one cell; returns None on composition error."""
    from hydra import compose

    overrides = [
        f"models={cell.model}",
        f"train=stage{cell.stage}",
        f"data={cell.data}",
        f"project.seed={cell.seed}",
    ]
    if cell.tag:
        overrides.append(f"project.tag={cell.tag}")
    if cell.method:
        overrides.append(f"project.method={cell.method}")
    overrides.extend(cell.defaults)
    overrides.extend(cell.overrides)
    try:
        cfg = compose(config_name="main", overrides=overrides)
        return cfg
    except (OmegaConfBaseException, Exception) as exc:  # noqa: BLE001
        cell.fail(f"composition failed: {type(exc).__name__}: {exc}")
        return None


def _check_train_cell(cfg: DictConfig, cell: Cell) -> None:
    if "data" not in cfg or cfg.data is None:
        cell.fail("cfg.data missing after composition")
        return
    task_name = str(cfg.data.get("task_name") or "")
    if task_name and task_name != cell.task:
        cell.fail(f"cfg.data.task_name={task_name!r} != manifest task {cell.task!r}")
    if "models" not in cfg or cfg.models is None:
        cell.fail("cfg.models missing after composition")
        return

    model_name = getattr(cfg.models, "name", None)
    if not model_name:
        cell.fail("cfg.models.name unset — checkpoint-source resolution alias broken")

    # num_phases consistency (mirrors the corresponding guard in the training pipeline).
    data_phases = None
    labeler = cfg.data.get("phase_labeler") if cfg.data.get("phase_labeler") else None
    if labeler is not None:
        data_phases = labeler.get("num_phases")
    candidates = []
    if data_phases is not None:
        candidates.append(("data.phase_labeler.num_phases", int(data_phases)))
    phase_head = cfg.models.get("phase_head")
    if phase_head is not None and phase_head.get("num_phases") is not None:
        candidates.append(("models.phase_head.num_phases", int(phase_head.num_phases)))
    if cfg.models.get("num_phases") is not None:
        candidates.append(("models.num_phases", int(cfg.models.num_phases)))
    if len({v for _, v in candidates}) > 1:
        cell.fail(f"num_phases mismatch across {candidates}")
    elif not candidates:
        cell.warnings.append("no num_phases declared anywhere (BC-style model)")

    train = cfg.train
    # Checkpoint monitor must follow the predeclared protocol rule.
    monitor = str(train.checkpoint.get("monitor", ""))
    expected = MONITOR_BY_STAGE[cell.stage]
    if monitor != expected:
        cell.fail(f"checkpoint.monitor={monitor!r} != predeclared {expected!r}")

    epochs = int(train.get("epochs", 0))
    scheduler = train.get("scheduler")
    if scheduler is not None:
        t_max = scheduler.get("T_max")
        if t_max is not None and int(t_max) < epochs:
            cell.warnings.append(
                f"scheduler T_max={t_max} < epochs={epochs} (premature LR decay)"
            )

    # Compute effective freeze and encoder_lr_scale following exact precedence:
    # models config wins if key present, else train config, else defaults
    models_cfg = cfg.get("models")
    if (
        models_cfg is not None
        and "freeze_encoder" in models_cfg
        and models_cfg.get("freeze_encoder") is not None
    ):
        effective_freeze = bool(models_cfg.get("freeze_encoder"))
    elif train.get("freeze_encoder") is not None:
        effective_freeze = bool(train.get("freeze_encoder"))
    else:
        effective_freeze = True

    if (
        models_cfg is not None
        and "encoder_lr_scale" in models_cfg
        and models_cfg.get("encoder_lr_scale") is not None
    ):
        effective_lr_scale = float(models_cfg.get("encoder_lr_scale"))
    elif train.get("encoder_lr_scale") is not None:
        effective_lr_scale = float(train.get("encoder_lr_scale"))
    else:
        effective_lr_scale = 1.0

    if cell.stage == 2:
        is_unfrozen = (
            model_name in ("precision_residual_scratch_moe",)
            or cell.method in ("precision_residual_scratch_moe",)
        )
        if not effective_freeze and not is_unfrozen:
            cell.fail(
                "stage-2 train.freeze_encoder=false (protocol requires true for frozen models)"
            )
        if effective_freeze and effective_lr_scale != 1.0:
            cell.fail(
                f"encoder_lr_scale={effective_lr_scale} != 1.0 with effective freeze (dead config)"
            )

        src = train.get("stage1_ckpt_path")
        if not src:
            # The CLI auto-resolves; verify the alias chain exists here too.
            from phaseforge.utils.config import resolve_checkpoint_source

            source_model = resolve_checkpoint_source(model_name)
            cell.warnings.append(
                f"stage1_ckpt_path auto-resolves via {model_name} -> {source_model}"
            )
    else:
        if bool(train.get("freeze_encoder", False)):
            cell.fail("stage-1 train.freeze_encoder must be false")

    # Validate router top_k <= num_experts
    router_cfg = cfg.models.get("router")
    if router_cfg is not None:
        top_k = int(router_cfg.get("top_k", 2))
        num_experts = int(router_cfg.get("num_experts", 6))
        if top_k > num_experts:
            cell.fail(f"router top_k ({top_k}) > num_experts ({num_experts})")

    # Validate phase_head router init requirement
    router_init = cfg.models.get("router_init")
    if router_init is not None and router_init.get("type") == "phase_head":
        if cfg.models.get("phase_head") is None:
            cell.fail("router_init=phase_head requires models.phase_head in config")
        from phaseforge.utils.config import resolve_checkpoint_source

        src_model = resolve_checkpoint_source(model_name)
        if src_model not in ("precision_residual_phaseforge", "self") and cell.stage == 2:
            cell.fail(
                "router_init=phase_head requires phase-supervised Stage 1 source "
                "'precision_residual_phaseforge', "
                f"got '{src_model}'"
            )

    # Validate corruption conflicts
    corruption_rate = float(cfg.data.get("phase_corruption_rate", 0.0))
    if corruption_rate > 0.0 and cell.method in (
        "precision_residual_teacher_forced",
        "precision_residual_oracle",
    ):
        cell.fail(f"Phase corruption not allowed for {cell.method}")

    # phase_class_weight sanity: the mode must be known, and cui requires
    # a valid decay in (0, 1).
    pw = str(train.get("phase_class_weight", "none"))
    if pw not in ("none", "balanced", "cui"):
        cell.fail(f"train.phase_class_weight={pw!r} (expected 'none', 'balanced' or 'cui')")
    if pw == "cui":
        beta = float(train.get("cui_beta", 0.999))
        if not 0.0 < beta < 1.0:
            cell.fail(f"train.cui_beta={beta!r} must be in (0, 1) for cui weighting")

    # sticky_coeff sanity: negative weights would reward switching.
    if float(train.get("sticky_coeff", 0.0)) < 0.0:
        cell.fail(f"train.sticky_coeff={train.get('sticky_coeff')!r} must be >= 0")

    # teacher_routing sanity: lambda0 must be >= 0, and the model must carry
    # the soft mapping M the teacher KL target is built from.
    tr = train.get("teacher_routing") or {}
    if float(tr.get("lambda0", 0.0)) < 0.0:
        cell.fail(f"train.teacher_routing.lambda0={tr.get('lambda0')!r} must be >= 0")
    models_tr = (cfg.models or {}).get("teacher_routing") or {}
    if bool(models_tr.get("enabled", False)) and cfg.models.get("soft_mapping") is None:
        cell.fail(
            "models.teacher_routing.enabled=true requires the soft_mapping "
            "config block (M is built at bootstrap)"
        )


def _check_eval_cell(cfg: DictConfig, cell: Cell, mode: str) -> None:
    if "eval" not in cfg or cfg.eval is None:
        cell.fail("cfg.eval missing after composition")
        return
    eval_mode = str(cfg.eval.get("mode", ""))
    if eval_mode != mode:
        cell.fail(f"eval.mode={eval_mode!r} != manifest evaluate_mode {mode!r}")
    if mode == "rollout":
        for key in ("bank", "env", "episodes"):
            if cfg.eval.get(key) is None:
                cell.fail(f"eval.{key} missing (rollout schema) — eval group not selected?")
        # V2-E router-mode sanity: the intervention must be one of the known
        # modes, and oracle requires the soft_mapping block (M is built at
        # bootstrap on the models with a router).
        router_mode = str(cfg.eval.episodes.get("router_mode", "learned")).lower()
        if router_mode not in ("learned", "sticky", "uniform", "oracle"):
            cell.fail(
                f"eval.episodes.router_mode={router_mode!r} must be one of "
                "learned, sticky, uniform, oracle"
            )
        if router_mode == "oracle" and (cfg.models or {}).get("soft_mapping") is None:
            cell.fail(
                "eval.episodes.router_mode=oracle requires the soft_mapping "
                "config block (M is built at bootstrap)"
            )


def _iter_manifest_cells(manifest: dict) -> list[Cell]:
    cells: list[Cell] = []
    for m in manifest.get("methods", []):
        stages = list(m.get("stages") or [])
        cell_overrides = tuple(m.get("overrides") or [])
        for stage in stages:
            for seed in manifest.get("seeds", []):
                cells.append(
                    Cell(
                        method=str(m.get("name", "")),
                        model=str(m.get("model", "")),
                        data=str(m.get("data", "")),
                        task=str(m.get("task", "")),
                        stage=int(stage),
                        seed=int(seed),
                        tag=(str(m["tag"]) if m.get("tag") else None),
                        defaults=tuple(manifest.get("defaults", [])),
                        overrides=cell_overrides,
                    )
                )
    return cells


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--methods", nargs="*", default=None, help="filter by method name")
    parser.add_argument("--tasks", nargs="*", default=None, help="filter by task")
    parser.add_argument(
        "--stages", nargs="*", type=int, default=None, help="filter by stage (1/2)"
    )
    args = parser.parse_args()

    import json

    from hydra import initialize_config_dir

    manifest_path = Path(args.manifest)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        print(f"preflight: cannot read manifest {manifest_path}: {exc}", file=sys.stderr)
        return 2

    cells = _iter_manifest_cells(manifest)
    if args.methods:
        cells = [c for c in cells if c.method in args.methods]
    if args.tasks:
        cells = [c for c in cells if c.task in args.tasks]
    if args.stages:
        cells = [c for c in cells if c.stage in args.stages]
    if not cells:
        print("preflight: no cells match the filters", file=sys.stderr)
        return 2

    config_dir = str(PROJECT_ROOT / "phaseforge" / "config")
    bad: list[Cell] = []
    eval_count = 0
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        for cell in cells:
            cfg = _compose_cell(cell)
            if cfg is None:
                bad.append(cell)
                print(f"[FAIL] {cell.label}: {cell.errors[0]}")
                continue
            try:
                _check_train_cell(cfg, cell)
            except Exception as exc:  # noqa: BLE001
                cell.fail(f"validation crashed: {type(exc).__name__}: {exc}")
            if cell.errors:
                bad.append(cell)

        # Evaluation cells: recompose each (method, task, seed) with the
        # runner's eval overrides (eval=<group>, eval.mode=<mode>) for every
        # method marked evaluate=true in the manifest.
        for m in manifest.get("methods", []):
            if not m.get("evaluate", False):
                continue
            mode = str(m.get("evaluate_mode", "rollout"))
            group = "rollout" if mode == "rollout" else "metrics"
            for seed in manifest.get("seeds", []):
                if args.methods and str(m.get("name", "")) not in args.methods:
                    continue
                if args.tasks and str(m.get("task", "")) not in args.tasks:
                    continue
                cell = Cell(
                    method=str(m.get("name", "")),
                    model=str(m.get("model", "")),
                    data=str(m.get("data", "")),
                    task=str(m.get("task", "")),
                    stage=0,
                    seed=int(seed),
                    tag=(str(m["tag"]) if m.get("tag") else None),
                    defaults=tuple(manifest.get("defaults", [])),
                )
                eval_count += 1
                from hydra import compose

                overrides = [
                    f"models={cell.model}",
                    f"data={cell.data}",
                    f"project.seed={cell.seed}",
                    f"eval={group}",
                    f"eval.mode={mode}",
                ]
                if cell.tag:
                    overrides.append(f"project.tag={cell.tag}")
                overrides.extend(cell.defaults)
                try:
                    eval_cfg = compose(config_name="main", overrides=overrides)
                    _check_eval_cell(eval_cfg, cell, mode)
                except (OmegaConfBaseException, Exception) as exc:  # noqa: BLE001
                    cell.fail(f"eval composition failed: {type(exc).__name__}: {exc}")
                if cell.errors:
                    bad.append(cell)
                    print(f"[FAIL] eval {cell.label} ({mode}): {cell.errors[0]}")

    for cell in bad:
        if cell.errors and "composition failed" not in cell.errors[0]:
            print(f"[FAIL] {cell.label}")
            for err in cell.errors:
                print(f"       - {err}")
    for cell in cells:
        for w in cell.warnings:
            print(f"[warn] {cell.label}: {w}")
    if bad:
        print(f"preflight: {len(bad)} of {len(cells)} cell(s) FAILED", file=sys.stderr)
        return 1

    print(f"preflight: all {len(cells)} train cell(s) and {eval_count} eval cell(s) passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

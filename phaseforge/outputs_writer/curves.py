"""Per-run training curves and final summary persistence.

Writes ``metrics/training_curves.jsonl`` (one schema-validated row per
epoch) and ``metrics/summary.json`` (the final per-run scalars at
``on_train_end``) inside a run directory. Mirrors the ``RunWriter`` and
``results.jsonl`` discipline: every row is validated before it reaches
disk, appends hold a cross-process ``FileLock`` + ``fsync``, and the
per-epoch curve append is idempotent (a resumed run never duplicates an
already-written epoch).

Schema: a single row schema with required core fields plus optional
stage/model-specific fields. A ``bc`` row carries only the core block;
a Stage 2 ``phaseforge`` row adds the routing fields. Absence is honest.
"""

from __future__ import annotations

import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from filelock import FileLock

from phaseforge.outputs_writer.schema import SchemaError

#: Required core curve fields (every run, every stage).
CURVE_CORE_REQUIRED: tuple[str, ...] = (
    "run_id",
    "epoch",
    "global_step",
    "train/lr",
    "epoch_wall_seconds",
    "train_steps_per_second",
    "train/loss_total",
    "train/loss_action",
    "val/loss_total",
    "val/loss_action",
)

#: Optional numeric fields. Stage 1 phase fields, Stage 2 routing fields,
#: the teacher diagnostic routing-accuracy fields, the V2-C stickiness and
#: V2-D teacher-KL training scalars, the per-epoch checkpoint monitor
#: value, and the V2-C validation switch rate all live here
#: (type-checked only when present).
CURVE_OPTIONAL_NUMERIC: tuple[str, ...] = (
    "train/loss_phase",
    "train/grad_cos_action_phase",
    "train/lambda_phase",
    "val/loss_phase",
    "train/phase_acc",
    "val/phase_acc",
    "train/phase_balanced_acc",
    "val/phase_balanced_acc",
    "train/loss_balance",
    "val/routing_entropy",
    "val/phase_expert_nmi",
    "val/topk_balance_score",
    "val/top1_balance_score",
    "val/topk_collapse_rate",
    "val/top1_collapse_rate",
    "val/routing_accuracy",
    "val/routing_balanced_accuracy",
    "train/loss_sticky",
    "train/loss_teacher_kl",
    "train/teacher_lambda",
    "val/routing_switch_rate",
    "checkpoint_monitor_value",
    # IS-PhaseForge Stage 1: Supervised Contrastive learning
    "train/loss_supcon",
    "train/supcon_lambda",
    "val/loss_supcon",
    # IS-PhaseForge Stage 2: Large-margin prototype router
    "train/loss_margin",
    "train/margin_lambda",
    "val/loss_margin",
    # IS-PhaseForge Stage 2: Local contraction / Lipschitz regularizer
    "train/loss_lip",
    "train/lip_lambda",
    "val/loss_lip",
    # IS-PhaseForge Stage 2: Gain regularization
    "train/loss_gain",
    "train/gain_lambda",
    "val/loss_gain",
    # Precision-Residual PhaseForge Stage 2: Release kinematics auxiliary loss
    "train/loss_release",
    "train/release_lambda",
    "val/loss_release",
    # Precision-Residual PhaseForge Stage 2: Fine-grained subspace action MSEs
    "train/loss_action_pos",
    "val/loss_action_pos",
    "train/loss_action_rot",
    "val/loss_action_rot",
    "train/loss_action_grip",
    "val/loss_action_grip",
    "train/loss_action_place",
    "val/loss_action_place",
)

_CURVE_INT_FIELDS = ("epoch", "global_step")


def _is_real_int(value: Any) -> bool:
    """``bool`` is technically ``int`` in Python — exclude it."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_real_numeric(value: Any) -> bool:
    """Numeric (int or float, excluding ``bool``). NaN allowed, inf rejected."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return not (isinstance(value, float) and math.isinf(value))


def validate_curve_row(row: dict[str, Any]) -> None:
    """Strict validator for one ``training_curves.jsonl`` row.

    Raises :class:`SchemaError` on missing required keys, wrong-typed
    values, unknown top-level keys, or a ``checkpoint_monitor`` present
    without its ``checkpoint_monitor_value``. ``peak_gpu_memory_mb`` may be
    ``None`` on CPU; every other optional numeric field
    must be finite-or-NaN when present.
    """
    if not isinstance(row, dict):
        raise SchemaError(f"Curve row must be a dict, got {type(row).__name__}")
    missing = [k for k in CURVE_CORE_REQUIRED if k not in row]
    if missing:
        raise SchemaError(f"Curve row missing required keys: {missing}")
    known = (
        set(CURVE_CORE_REQUIRED)
        | set(CURVE_OPTIONAL_NUMERIC)
        | {"checkpoint_monitor", "peak_gpu_memory_mb", "extra"}
    )
    unknown = sorted(set(row) - known)
    if unknown:
        raise SchemaError(f"Curve row has unknown top-level keys: {unknown}")
    if not isinstance(row["run_id"], str):
        raise SchemaError(f"Curve row['run_id'] must be str, got {type(row['run_id']).__name__}")
    for key in _CURVE_INT_FIELDS:
        if not _is_real_int(row[key]):
            raise SchemaError(f"Curve row[{key!r}] must be int, got {type(row[key]).__name__}")
    for key in CURVE_CORE_REQUIRED:
        if key in _CURVE_INT_FIELDS or key == "run_id":
            continue
        if not _is_real_numeric(row[key]):
            raise SchemaError(
                f"Curve row[{key!r}] must be finite or NaN numeric, got {type(row[key]).__name__}"
            )
    for key in CURVE_OPTIONAL_NUMERIC:
        if key not in row:
            continue
        if not _is_real_numeric(row[key]):
            raise SchemaError(
                f"Curve row[{key!r}] must be finite or NaN numeric, got {type(row[key]).__name__}"
            )
    if "checkpoint_monitor" in row:
        if not isinstance(row["checkpoint_monitor"], str):
            raise SchemaError(
                "Curve row['checkpoint_monitor'] must be str, got "
                f"{type(row['checkpoint_monitor']).__name__}"
            )
        if "checkpoint_monitor_value" not in row:
            raise SchemaError("Curve row has checkpoint_monitor without checkpoint_monitor_value")
    if "peak_gpu_memory_mb" in row and row["peak_gpu_memory_mb"] is not None:
        if not _is_real_numeric(row["peak_gpu_memory_mb"]):
            raise SchemaError(
                "Curve row['peak_gpu_memory_mb'] must be finite or NaN "
                "numeric or null, got "
                f"{type(row['peak_gpu_memory_mb']).__name__}"
            )
    if "extra" in row and not isinstance(row["extra"], dict):
        raise SchemaError(f"Curve row['extra'] must be dict, got {type(row['extra']).__name__}")


#: Required fields of ``metrics/summary.json``.
SUMMARY_REQUIRED: tuple[str, ...] = (
    "run_id",
    "kind",
    "model",
    "stage",
    "seed",
    "config_hash",
    "data_config_hash",
    "data_provenance_path",
    "git_sha",
    "device",
    "started_at",
    "finished_at",
    "wall_seconds",
    "epochs_run",
    "trainable_params",
    "total_params",
    "best_epoch",
    "final_val",
    "extra",
)

_SUMMARY_NUMERIC_NULLABLE = ("wall_seconds", "best_val_monitor", "peak_gpu_memory_mb")


def _validate_summary_numeric_nullable(row: dict[str, Any], key: str) -> None:
    value = row.get(key)
    if value is None:
        return
    if not _is_real_numeric(value):
        raise SchemaError(
            f"Summary[{key!r}] must be finite or NaN numeric or null, got {type(value).__name__}"
        )


def validate_summary(summary: dict[str, Any]) -> None:
    """Strict validator for ``metrics/summary.json`` and ledger rows.

    Unknown top-level keys are rejected (forward-compat lives in
    ``extra``). ``seed``, ``best_epoch`` and ``global_steps`` may be null
    when the run did not configure them; ``best_val_monitor``,
    ``peak_gpu_memory_mb`` and ``wall_seconds`` may be null/NaN. Every
    other field is type-checked strictly.
    """
    if not isinstance(summary, dict):
        raise SchemaError(f"Summary must be a dict, got {type(summary).__name__}")
    missing = [k for k in SUMMARY_REQUIRED if k not in summary]
    if missing:
        raise SchemaError(f"Summary missing required keys: {missing}")
    known = set(SUMMARY_REQUIRED) | {
        "best_val_monitor",
        "best_checkpoint",
        "best_checkpoint_sha256",
        "source_stage1",
        "global_steps",
        "peak_gpu_memory_mb",
        "lambda_phase",
        "balance_coeff",
        "freeze_encoder",
        "tag",
        "method",
        "run_dir",
    }
    unknown = sorted(set(summary) - known)
    if unknown:
        raise SchemaError(f"Summary has unknown top-level keys: {unknown}")

    for key in (
        "run_id",
        "model",
        "config_hash",
        "data_config_hash",
        "git_sha",
        "device",
        "started_at",
        "finished_at",
        "data_provenance_path",
    ):
        value = summary[key]
        if value is not None and not isinstance(value, str):
            raise SchemaError(f"Summary[{key!r}] must be str or null, got {type(value).__name__}")
    for key in ("tag", "method"):
        if key not in summary:
            continue
        value = summary[key]
        if value is not None and not isinstance(value, str):
            raise SchemaError(f"Summary[{key!r}] must be str or null, got {type(value).__name__}")
    if summary["kind"] not in ("train", "eval"):
        raise SchemaError(f"Summary['kind'] must be 'train' or 'eval', got {summary['kind']!r}")
    if not _is_real_int(summary["stage"]):
        raise SchemaError(f"Summary['stage'] must be int, got {type(summary['stage']).__name__}")
    seed = summary["seed"]
    if seed is not None and not _is_real_int(seed):
        raise SchemaError(f"Summary['seed'] must be int or null, got {type(seed).__name__}")
    for key in ("epochs_run", "trainable_params", "total_params"):
        if not _is_real_int(summary[key]):
            raise SchemaError(f"Summary[{key!r}] must be int, got {type(summary[key]).__name__}")
    best_epoch = summary["best_epoch"]
    if best_epoch is not None and not _is_real_int(best_epoch):
        raise SchemaError(
            f"Summary['best_epoch'] must be int or null, got {type(best_epoch).__name__}"
        )
    for key in _SUMMARY_NUMERIC_NULLABLE:
        _validate_summary_numeric_nullable(summary, key)
    if not isinstance(summary["final_val"], dict):
        raise SchemaError(
            f"Summary['final_val'] must be dict, got {type(summary['final_val']).__name__}"
        )
    if not isinstance(summary["extra"], dict):
        raise SchemaError(f"Summary['extra'] must be dict, got {type(summary['extra']).__name__}")
    for key in ("best_checkpoint", "best_checkpoint_sha256", "run_dir"):
        if key not in summary:
            continue
        value = summary[key]
        if value is not None and not isinstance(value, str):
            raise SchemaError(f"Summary[{key!r}] must be str or null, got {type(value).__name__}")
    if "source_stage1" in summary:
        value = summary["source_stage1"]
        if value is not None and not isinstance(value, dict):
            raise SchemaError(
                f"Summary['source_stage1'] must be dict or null, got {type(value).__name__}"
            )
    if "global_steps" in summary:
        value = summary["global_steps"]
        if value is not None and not _is_real_int(value):
            raise SchemaError(
                f"Summary['global_steps'] must be int or null, got {type(value).__name__}"
            )
    for key in ("lambda_phase", "balance_coeff"):
        if key not in summary:
            continue
        value = summary[key]
        if value is not None and not _is_real_numeric(value):
            raise SchemaError(
                f"Summary[{key!r}] must be finite or NaN numeric or null, "
                f"got {type(value).__name__}"
            )
    if "freeze_encoder" in summary:
        value = summary["freeze_encoder"]
        if value is not None and not isinstance(value, bool):
            raise SchemaError(
                f"Summary['freeze_encoder'] must be bool or null, got {type(value).__name__}"
            )


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _iso_now() -> str:
    return datetime.now(UTC).isoformat()


class TrainingCurveWriter:
    """Appends validated curve rows + the final summary for one run dir.

    ``append_curve_row`` is idempotent per ``(run_id, epoch)``: a resumed
    run that already persisted an epoch will not duplicate it. Rows are
    validated before any write, exactly like the results ledger.
    """

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.metrics_dir = self.run_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.curves_path = self.metrics_dir / "training_curves.jsonl"
        self.lock = FileLock(str(self.metrics_dir / ".curves.lock"))

    def append_curve_row(self, row: dict[str, Any]) -> Path:
        """Validate and append one epoch row; no-op if already persisted."""
        validate_curve_row(row)
        payload = json.dumps(row, default=_json_default) + "\n"
        with self.lock:
            existing = self._read_curves_locked()
            if any(
                r.get("run_id") == row["run_id"] and r.get("epoch") == row["epoch"]
                for r in existing
            ):
                return self.curves_path
            with open(self.curves_path, "a", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
        return self.curves_path

    def _read_curves_locked(self) -> list[dict[str, Any]]:
        """Tolerant read (crash-truncated trailing line skipped)."""
        if not self.curves_path.exists():
            return []
        text = self.curves_path.read_text(encoding="utf-8")
        ends_with_newline = text.endswith(("\n", "\r"))
        lines = text.splitlines()
        out: list[dict[str, Any]] = []
        for index, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            is_last = index == len(lines) - 1
            try:
                out.append(json.loads(stripped))
            except json.JSONDecodeError:
                if is_last and not ends_with_newline:
                    continue
                raise
        return out

    def read_curves(self) -> list[dict[str, Any]]:
        """Public tolerant read of every epoch row written so far."""
        with self.lock:
            return self._read_curves_locked()

    def write_summary(self, summary: dict[str, Any]) -> Path:
        """Validate and persist ``metrics/summary.json``."""
        validate_summary(summary)
        path = self.metrics_dir / "summary.json"
        path.write_text(
            json.dumps(summary, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        return path


__all__ = [
    "CURVE_CORE_REQUIRED",
    "CURVE_OPTIONAL_NUMERIC",
    "SUMMARY_REQUIRED",
    "TrainingCurveWriter",
    "validate_curve_row",
    "validate_summary",
]

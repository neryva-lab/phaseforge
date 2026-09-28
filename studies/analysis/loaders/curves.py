"""Typed loader for training curves (training_curves.jsonl).

Field set is the full audited schema (figures_tables_plan.md section 1):
efficiency, train losses, and every Stage-2 routing diagnostic. Fields absent
for a stage (e.g. NMI in Stage 1) stay ``None`` — only the identity columns
are required.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from studies.analysis.common import io as cio

_REQUIRED = ("epoch", "global_step")


@dataclass(frozen=True)
class CurvePoint:
    epoch: int
    global_step: int
    # losses
    train_loss_action: float | None = None
    train_loss_total: float | None = None
    train_loss_balance: float | None = None
    train_loss_sticky: float | None = None
    train_loss_teacher_kl: float | None = None
    val_loss_action: float | None = None
    val_loss_total: float | None = None
    train_lr: float | None = None
    # routing diagnostics (Stage 2)
    nmi: float | None = None
    routing_entropy: float | None = None
    switch_rate: float | None = None
    top1_balance: float | None = None
    top1_collapse: float | None = None
    topk_balance: float | None = None
    topk_collapse: float | None = None
    # efficiency
    epoch_wall_seconds: float | None = None
    steps_per_second: float | None = None
    peak_gpu_memory_mb: float | None = None


_JSON_TO_ATTR: dict[str, str] = {
    "train/loss_action": "train_loss_action",
    "train/loss_total": "train_loss_total",
    "train/loss_balance": "train_loss_balance",
    "train/loss_sticky": "train_loss_sticky",
    "train/loss_teacher_kl": "train_loss_teacher_kl",
    "val/loss_action": "val_loss_action",
    "val/loss_total": "val_loss_total",
    "train/lr": "train_lr",
    "val/phase_expert_nmi": "nmi",
    "val/routing_entropy": "routing_entropy",
    "val/routing_switch_rate": "switch_rate",
    "val/top1_balance_score": "top1_balance",
    "val/top1_collapse_rate": "top1_collapse",
    "val/topk_balance_score": "topk_balance",
    "val/topk_collapse_rate": "topk_collapse",
    "epoch_wall_seconds": "epoch_wall_seconds",
    "train_steps_per_second": "steps_per_second",
    "peak_gpu_memory_mb": "peak_gpu_memory_mb",
}


@dataclass(frozen=True)
class TrainingCurve:
    path: Path
    points: tuple[CurvePoint, ...]

    def epochs(self) -> list[int]:
        return [p.epoch for p in self.points]

    def series(self, field_name: str) -> list[tuple[int, float]]:
        """(epoch, value) pairs for a CurvePoint attribute, skipping Nones.

        ``field_name`` may be the JSON key (``val/phase_expert_nmi``) or the
        attribute name (``nmi``).
        """
        attr = _JSON_TO_ATTR.get(field_name, field_name)
        return [(p.epoch, v) for p in self.points if (v := getattr(p, attr, None)) is not None]

    def last(self, field_name: str) -> float | None:
        pairs = self.series(field_name)
        return pairs[-1][1] if pairs else None


def load_curve(run_dir: Path) -> TrainingCurve:
    # Support both layouts: run_dir/training_curves.jsonl and run_dir/metrics/training_curves.jsonl (final_output)
    curve_path = run_dir / "training_curves.jsonl"
    if not curve_path.is_file():
        curve_path = run_dir / "metrics" / "training_curves.jsonl"
    points: list[CurvePoint] = []
    for row in cio.iter_jsonl(curve_path):
        missing = [k for k in _REQUIRED if k not in row]
        if missing:
            raise ValueError(f"{run_dir / 'training_curves.jsonl'}: missing {missing}")
        kwargs: dict[str, Any] = {}
        for json_key, attr in _JSON_TO_ATTR.items():
            if json_key in row and row[json_key] is not None:
                kwargs[attr] = float(row[json_key])
        points.append(
            CurvePoint(epoch=int(row["epoch"]), global_step=int(row["global_step"]), **kwargs)
        )
    if not points:
        raise ValueError(f"{run_dir / 'training_curves.jsonl'}: no curve points")
    points.sort(key=lambda p: p.epoch)
    return TrainingCurve(path=run_dir, points=tuple(points))

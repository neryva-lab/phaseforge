"""Persistence callback: exact epoch curves + final run summary to disk.

The per-run provenance is written from the
existing ``on_epoch_start`` / ``on_train_batch`` / ``on_epoch_end`` /
``on_train_end`` hooks:

* ``on_epoch_start`` — capture ``train/lr`` (learning rate at the start of
  the epoch, definition fixed) and reset the epoch accumulator.
* ``on_train_batch`` — accumulate the on-device loss tensors sample-weighted
  without per-batch host/device syncs; the exact epoch mean is materialized
  once at ``on_epoch_end``.
* ``on_epoch_end`` — write one schema-validated ``training_curves.jsonl``
  row: core fields + per-epoch timing + checkpoint monitor name/value
  + stage-specific fields (phase accuracy from
  ``Stage1Trainer.epoch_train_metrics``/validation pool, routing diagnostics
  from the validation metrics, teacher-forced routing accuracy).
* ``on_train_end`` — write ``metrics/summary.json``: final scalars, best
  checkpoint + SHA-256, source Stage 1 identity, parameter counts.

The curve append is idempotent per ``(run_id, epoch)`` and the summary is
derived from final state, so a resumed run produces no duplicate rows.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from phaseforge.outputs_writer.curves import (
    CURVE_OPTIONAL_NUMERIC,
    TrainingCurveWriter,
)
from phaseforge.outputs_writer.provenance import sha256_file
from phaseforge.trains.callbacks.base import Callback
from phaseforge.utils.config import config_hash, git_info

logger = logging.getLogger(__name__)

#: Unprefixed validation metrics and their persisted ``val/`` key.
_VAL_KEY_MAP = {
    "loss_total": "val/loss_total",
    "loss_action": "val/loss_action",
    "loss_phase": "val/loss_phase",
    "loss_supcon": "val/loss_supcon",
    "loss_margin": "val/loss_margin",
    "loss_lip": "val/loss_lip",
    "loss_gain": "val/loss_gain",
    "loss_release": "val/loss_release",
    "loss_action_pos": "val/loss_action_pos",
    "loss_action_rot": "val/loss_action_rot",
    "loss_action_grip": "val/loss_action_grip",
    "loss_action_place": "val/loss_action_place",
}

_EMPTY_SOURCE_STAGE1 = {
    "run_id": None,
    "checkpoint": None,
    "sha256": None,
    "model": None,
    "seed": None,
    "config_hash": None,
    "git_commit": None,
}


class MetricPersistenceCallback(Callback):
    """Persists per-epoch curves and the final summary for one run."""

    def __init__(
        self,
        run_dir: str | Path,
        run_id: str,
        *,
        data_config_hash: str | None = None,
        source_stage1: dict[str, Any] | None = None,
        kind: str = "train",
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.data_config_hash = data_config_hash
        self.source_stage1 = source_stage1 or dict(_EMPTY_SOURCE_STAGE1)
        self.kind = kind
        self.writer = TrainingCurveWriter(self.run_dir)

        self._started_at: str | None = None
        self._finished_at: str | None = None
        self._wall_t0: float | None = None
        self._lr: float | None = None
        self._train_acc: dict[str, torch.Tensor] = {}
        self._train_n = 0
        self._last_val_metrics: dict[str, float] = {}
        self._peak_memory_mb: float | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_train_start(self, trainer: Any) -> None:
        self._started_at = datetime.now(UTC).isoformat()
        self._wall_t0 = time.perf_counter()

    def on_epoch_start(self, trainer: Any) -> None:
        self._train_acc = {}
        self._train_n = 0
        self._lr = None
        optimizer = getattr(trainer, "optimizer", None)
        if optimizer is not None and optimizer.param_groups:
            lr = optimizer.param_groups[0].get("lr")
            if isinstance(lr, (int, float)):
                self._lr = float(lr)

    def on_train_batch(
        self,
        trainer: Any,
        batch: dict[str, torch.Tensor],
        out: Any,
        metrics: dict[str, Any],
        n: int,
        step: int,
    ) -> None:
        if n <= 0:
            return
        device = trainer.device
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                val = value.detach().to(device=device, dtype=torch.float32)
            else:
                val = torch.as_tensor(float(value), device=device, dtype=torch.float32)
            acc = self._train_acc.get(key, torch.zeros((), device=device, dtype=torch.float32))
            self._train_acc[key] = acc + val * n
        self._train_n += n

    def on_epoch_end(self, trainer: Any, val_metrics: dict[str, float]) -> None:
        self._last_val_metrics = {k: float(v) for k, v in val_metrics.items()}
        timing = trainer.epoch_timing()

        val_total = val_metrics.get("loss_total", val_metrics.get("val/loss_total", float("nan")))
        val_action = val_metrics.get(
            "loss_action", val_metrics.get("val/loss_action", float("nan"))
        )

        row: dict[str, Any] = {
            "run_id": self.run_id,
            "epoch": int(trainer.current_epoch),
            "global_step": int(trainer.global_step),
            "train/lr": self._lr if self._lr is not None else float("nan"),
            "epoch_wall_seconds": float(timing.get("epoch_wall_seconds", float("nan"))),
            "train_steps_per_second": float(timing.get("train_steps_per_second", float("nan"))),
            "val/loss_total": float(val_total),
            "val/loss_action": float(val_action),
        }

        # Exact sample-weighted epoch means of the training losses. ``bc``
        # has no phase head, so its constant ``loss_phase`` is omitted
        # (schema-optional; absence is honest).
        if self._train_n > 0:
            has_phase_head = hasattr(trainer.model, "phase_head")
            for key, acc in self._train_acc.items():
                mean = float((acc / self._train_n).item())
                train_key = f"train/{key}"
                if train_key == "train/loss_phase" and not has_phase_head:
                    continue
                row[train_key] = mean

        # Stage-specific training scalars (e.g. train/phase_acc).
        row.update(trainer.epoch_train_metrics())

        # Validation extras: allowlisted ``val/`` diagnostics only.
        for key, value in val_metrics.items():
            if key in _VAL_KEY_MAP:
                row[_VAL_KEY_MAP[key]] = float(value)
            elif key.startswith("val/") and (
                key in CURVE_OPTIONAL_NUMERIC or key in ("val/loss_total", "val/loss_action")
            ):
                row[key] = float(value)
            elif key.startswith("val/"):
                logger.warning("Dropping validation metric %r not in the curve schema.", key)

        # Per-epoch checkpoint monitor name + value.
        # val_metrics mixes unprefixed losses (``loss_total``) and prefixed
        # routing diagnostics (``val/routing_entropy``), so try both forms.
        monitor = self._monitor_name(trainer)
        if monitor:
            monitor_value = val_metrics.get(
                monitor.replace("val/", ""), val_metrics.get(monitor, float("nan"))
            )
            row["checkpoint_monitor"] = monitor
            row["checkpoint_monitor_value"] = float(monitor_value)

        peak = timing.get("peak_gpu_memory_mb")
        if peak is not None:
            peak_mb = float(peak)
            row["peak_gpu_memory_mb"] = peak_mb
            if self._peak_memory_mb is None or peak_mb > self._peak_memory_mb:
                self._peak_memory_mb = peak_mb

        self.writer.append_curve_row(row)

    def on_train_end(self, trainer: Any) -> None:
        self._finished_at = datetime.now(UTC).isoformat()
        wall = float("nan")
        if self._started_at is not None and self._wall_t0 is not None:
            wall = time.perf_counter() - self._wall_t0

        counts = trainer.parameter_counts()
        best_epoch, best_monitor, best_ckpt, best_sha = self._best_checkpoint_info(trainer)

        models_cfg = trainer.cfg.get("models")
        model_name = str(
            models_cfg.get("name", type(trainer.model).__name__)
            if models_cfg is not None
            else type(trainer.model).__name__
        )
        stage = int(getattr(trainer.model, "stage", trainer.train_cfg.get("stage", 1)))

        summary: dict[str, Any] = {
            "run_id": self.run_id,
            "kind": self.kind,
            "model": model_name,
            "stage": stage,
            "seed": trainer.cfg.project.get("seed"),
            "tag": trainer.cfg.project.get("tag"),
            "method": trainer.cfg.project.get("method"),
            "config_hash": config_hash(trainer.cfg),
            "data_config_hash": self.data_config_hash,
            "data_provenance_path": "metadata/data_provenance.json",
            "git_sha": git_info()["commit"],
            "device": str(trainer.device),
            "started_at": self._started_at or "",
            "finished_at": self._finished_at,
            "wall_seconds": float(wall),
            "epochs_run": int(trainer.current_epoch),
            "global_steps": int(trainer.global_step),
            "trainable_params": counts["trainable_params"],
            "total_params": counts["total_params"],
            "best_epoch": best_epoch,
            "best_val_monitor": best_monitor,
            "final_val": dict(self._last_val_metrics),
            "best_checkpoint": best_ckpt,
            "best_checkpoint_sha256": best_sha,
            "source_stage1": dict(self.source_stage1),
            "extra": {},
        }

        # Config-derived convenience fields.
        if "lambda_phase" in trainer.train_cfg:
            summary["lambda_phase"] = float(trainer.train_cfg.lambda_phase)
        if "freeze_encoder" in trainer.train_cfg:
            summary["freeze_encoder"] = bool(trainer.train_cfg.freeze_encoder)
        balance_coeff = self._balance_coeff(trainer)
        if balance_coeff is not None:
            summary["balance_coeff"] = balance_coeff
        if self._peak_memory_mb is not None:
            summary["peak_gpu_memory_mb"] = self._peak_memory_mb

        self.writer.write_summary(summary)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _monitor_name(self, trainer: Any) -> str:
        checkpoint_cfg = trainer.train_cfg.get("checkpoint")
        if checkpoint_cfg is None:
            return ""
        return str(checkpoint_cfg.get("monitor", ""))

    def _balance_coeff(self, trainer: Any) -> float | None:
        models_cfg = trainer.cfg.get("models")
        if models_cfg is None:
            return None
        router_cfg = models_cfg.get("router")
        if router_cfg is None:
            return None
        coeff = router_cfg.get("balance_coeff")
        return float(coeff) if coeff is not None else None

    def _best_checkpoint_info(
        self, trainer: Any
    ) -> tuple[int | None, float | None, str | None, str | None]:
        for callback in trainer.callbacks:
            best = getattr(callback, "best_ckpt_path", None)
            if best is None or not Path(best).is_file():
                continue
            best_path = Path(best)
            try:
                rel = best_path.relative_to(self.run_dir).as_posix()
            except ValueError:
                rel = str(best_path)
            sha = sha256_file(best_path)
            topk = getattr(callback, "_topk", None)
            best_epoch = int(topk[0][1]) if topk else None
            best_score = getattr(callback, "best_score", None)
            best_monitor = float(best_score) if isinstance(best_score, (int, float)) else None
            return best_epoch, best_monitor, rel, sha
        return None, None, None, None


__all__ = ["MetricPersistenceCallback"]

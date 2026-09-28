"""CheckpointCallback: Saves model weights periodically and tracks best models."""

from __future__ import annotations

import logging
import math
import random
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from phaseforge.trains.callbacks.base import Callback

if TYPE_CHECKING:
    from phaseforge.trains.loops.base import BaseTrainer

logger = logging.getLogger(__name__)


class CheckpointCallback(Callback):
    """Saves checkpoints during training.

    Args:
        output_dir: Directory to save checkpoints.
        every_n_epochs: Save frequency for periodic snapshots.
        monitor: Metric to monitor for the 'best' checkpoint.
        mode: 'min' or 'max' for the monitored metric.
        save_top_k: How many best checkpoints to keep. The callback
            maintains a top-k collection (``checkpoint_best_epoch_*.pt``),
            deletes evicted members, and keeps ``checkpoint_best.pt`` as an
            alias for the current rank-1 checkpoint so downstream checkpoint
            discovery stays stable. Periodic snapshots
            (``checkpoint_epoch_*.pt``) are always kept regardless of
            ``save_top_k``.

    Checkpoints also record optimizer/scheduler state, RNG state
    (torch/numpy/random/cuda) and callback state so
    :meth:`BaseTrainer.resume` can restore an interrupted run: the model,
    optimizer, scheduler, global step and epoch are restored exactly. Note:
    the DataLoader shuffle generator is seeded from ``project.seed`` at
    pipeline build time and is *not* captured in the checkpoint, so after a
    resume the per-epoch batch *order* may differ from an uninterrupted run
    (the data content and splits are identical). Resume is intended for
    manual continuation of an interrupted run, not for bit-exact
    reproducibility.
    """

    def __init__(
        self,
        output_dir: Path | str,
        every_n_epochs: int = 10,
        monitor: str = "val/loss_total",
        mode: str = "min",
        save_top_k: int = 1,
        min_epoch: int = 0,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = max(1, int(save_top_k))
        self.min_epoch = max(0, int(min_epoch))

        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.best_ckpt_path: Path | None = None
        # (score, epoch, snapshot path) sorted best-first.
        self._topk: list[tuple[float, int, Path]] = []

    def on_epoch_end(self, trainer: BaseTrainer, val_metrics: dict[str, float]) -> None:
        epoch = trainer.current_epoch

        # Metric keys might be passed as e.g., 'loss_total' instead of 'val/loss_total'
        # so we strip 'val/' if necessary. Some loops (stage 2) also return
        # already-prefixed keys ('val/routing_entropy'); try both forms so a
        # routing-diagnostics monitor never silently selects nothing.
        monitor_key = self.monitor.replace("val/", "")

        current_score = val_metrics.get(monitor_key, val_metrics.get(self.monitor))

        if current_score is not None:
            self._update_topk(trainer, epoch, float(current_score))

        if epoch % self.every_n_epochs == 0:
            self._save_periodic(trainer, epoch)

    def _update_topk(self, trainer: BaseTrainer, epoch: int, score: float) -> None:
        """Insert the epoch into the top-k collection and prune evicted files."""
        # When min_epoch is configured, skip top-k selection for earlier epochs
        # so early transient noise dips cannot permanently freeze the best checkpoint
        # before the learning rate schedule has annealed into fine-tuning.
        if epoch < self.min_epoch:
            return

        # Never admit a non-finite score: when the top-k collection is not
        # yet full, an early NaN monitor value would otherwise be pinned as
        # "best" (float-NaN comparisons are False in both directions, so a
        # NaN at the head can never be replaced), silently exporting a
        # garbage best checkpoint to the eval path.
        if not math.isfinite(score):
            logger.warning(
                "Skipping checkpoint at epoch %d: monitor %s is non-finite (%s)",
                epoch,
                self.monitor,
                score,
            )
            return
        # Skip snapshotting when the collection is full and this score cannot
        # enter it.
        if len(self._topk) >= self.save_top_k:
            worst_score = self._topk[-1][0]
            enters = score < worst_score if self.mode == "min" else score > worst_score
            if not enters:
                return

        snapshot = self.output_dir / f"checkpoint_best_epoch_{epoch:04d}.pt"
        self._topk.append((score, epoch, snapshot))
        self._topk.sort(key=lambda entry: entry[0], reverse=(self.mode == "max"))

        evicted = self._topk[self.save_top_k :]
        self._topk = self._topk[: self.save_top_k]
        for _, _, path in evicted:
            path.unlink(missing_ok=True)
            logger.info("Evicted checkpoint outside top-%d: %s", self.save_top_k, path.name)

        # Rank-1 alias for downstream checkpoint discovery.
        best_score, best_epoch, best_path = self._topk[0]
        self.best_score = best_score
        alias = self.output_dir / "checkpoint_best.pt"
        self.best_ckpt_path = alias

        # Build the payload only after the top-k state has been updated. This
        # makes a checkpoint self-consistent when it is used for resume.
        torch.save(self._build_state(trainer, epoch), snapshot)
        shutil.copyfile(best_path, alias)
        if epoch == best_epoch:
            logger.info(f"Saved new best checkpoint (epoch {epoch}) to {alias.name}")

    def _save_periodic(self, trainer: BaseTrainer, epoch: int) -> None:
        path = self.output_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        torch.save(self._build_state(trainer, epoch), path)
        logger.info(f"Saved periodic checkpoint to {path.name}")

    def _build_state(self, trainer: BaseTrainer, epoch: int) -> dict:
        return {
            "epoch": epoch,
            "global_step": trainer.global_step,
            "should_stop": trainer.should_stop,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": trainer.scheduler.state_dict() if trainer.scheduler else None,
            "stage": trainer.model.stage if hasattr(trainer.model, "stage") else 1,
            "rng_state": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(),
                "random": random.getstate(),
            },
            "callbacks": {
                type(cb).__name__: cb.state_dict()
                for cb in trainer.callbacks
                if hasattr(cb, "state_dict")
            },
        }

    def state_dict(self) -> dict:
        return {
            "best_score": self.best_score,
            "best_ckpt_path": str(self.best_ckpt_path) if self.best_ckpt_path else None,
            "topk": [(score, epoch, path.name) for score, epoch, path in self._topk],
        }

    def load_state_dict(self, state: dict) -> None:
        self.best_score = state.get("best_score", self.best_score)
        saved_path = state.get("best_ckpt_path")
        if saved_path:
            candidate = Path(saved_path)
            self.best_ckpt_path = candidate if candidate.is_file() else None
        topk = state.get("topk", [])
        self._topk = []
        for score, epoch, name in topk:
            path = self.output_dir / name
            if path.is_file():
                self._topk.append((float(score), int(epoch), path))
        self._topk.sort(key=lambda entry: entry[0], reverse=(self.mode == "max"))

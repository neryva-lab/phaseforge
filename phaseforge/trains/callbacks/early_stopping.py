"""Early stopping callback."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from phaseforge.trains.callbacks.base import Callback

if TYPE_CHECKING:
    from phaseforge.trains.loops.base import BaseTrainer

logger = logging.getLogger(__name__)


class EarlyStoppingCallback(Callback):
    """Stops training if a monitored metric stops improving."""

    def __init__(
        self,
        monitor: str = "val/loss_total",
        mode: str = "min",
        patience: int = 10,
        min_delta: float = 0.0,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.enabled = enabled

        self.wait_count = 0
        self.best_score = math.inf if mode == "min" else -math.inf

    def on_epoch_end(self, trainer: BaseTrainer, val_metrics: dict[str, float]) -> None:
        """Check if we should stop training."""
        if not self.enabled:
            return

        # The training loop does not prefix keys with 'val/', so we strip it if provided
        monitor_key = self.monitor.replace("val/", "")

        if monitor_key not in val_metrics:
            logger.warning(
                "Early stopping monitor '%s' (resolved to '%s') not found "
                "in val_metrics. Available metrics: %s",
                self.monitor,
                monitor_key,
                list(val_metrics.keys()),
            )
            return

        current_score = val_metrics[monitor_key]

        improved = False
        if self.mode == "min":
            if current_score < self.best_score - self.min_delta:
                improved = True
        else:
            if current_score > self.best_score + self.min_delta:
                improved = True

        if improved:
            self.best_score = current_score
            self.wait_count = 0
        else:
            self.wait_count += 1
            logger.info(
                "EarlyStopping: %s did not improve. Patience: %d/%d",
                self.monitor,
                self.wait_count,
                self.patience,
            )

            if self.wait_count >= self.patience:
                logger.info(
                    "EarlyStopping: Patience of %d reached. Signaling trainer to stop.",
                    self.patience,
                )
                trainer.should_stop = True

    def state_dict(self) -> dict:
        """Serialize early-stopping state for checkpoint resume."""
        return {
            "wait_count": self.wait_count,
            "best_score": self.best_score,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore early-stopping state from a checkpoint."""
        self.wait_count = int(state.get("wait_count", self.wait_count))
        self.best_score = state.get("best_score", self.best_score)

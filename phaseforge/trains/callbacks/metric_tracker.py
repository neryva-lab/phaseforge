"""MetricTrackerCallback: In-memory buffer for custom metric aggregations."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from phaseforge.trains.callbacks.base import Callback

if TYPE_CHECKING:
    from phaseforge.trains.loops.base import BaseTrainer


class MetricTrackerCallback(Callback):
    """Tracks raw metric values over time.

    Useful for metrics that require history (like TimeToStableRouting)
    or for producing summary plots at the end of training.
    """

    def __init__(self) -> None:
        self.history: dict[str, list[float]] = defaultdict(list)

    def on_train_step(self, trainer: BaseTrainer, step: int, metrics: dict[str, float]) -> None:
        for k, v in metrics.items():
            self.history[f"train/{k}"].append(v)

    def on_epoch_end(self, trainer: BaseTrainer, val_metrics: dict[str, float]) -> None:
        for k, v in val_metrics.items():
            self.history[f"val/{k}"].append(v)

    def get_history(self, key: str) -> list[float]:
        return self.history.get(key, [])

    def state_dict(self) -> dict:
        """Serialize tracked history for checkpoint resume."""
        return {"history": {k: list(v) for k, v in self.history.items()}}

    def load_state_dict(self, state: dict) -> None:
        """Restore tracked history from a checkpoint."""
        self.history = defaultdict(list)
        for k, v in state.get("history", {}).items():
            self.history[k] = list(v)

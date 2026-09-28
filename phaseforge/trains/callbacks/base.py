"""Callback base class."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phaseforge.trains.loops.base import BaseTrainer


class Callback:
    """Base class for all training callbacks."""

    def on_train_start(self, trainer: BaseTrainer) -> None:
        pass

    def on_epoch_start(self, trainer: BaseTrainer) -> None:
        pass

    def on_train_batch(
        self,
        trainer: BaseTrainer,
        batch: dict,
        out: object,
        metrics: dict,
        n: int,
        step: int,
    ) -> None:
        pass

    def on_train_step(self, trainer: BaseTrainer, step: int, metrics: dict[str, float]) -> None:
        pass

    def on_epoch_end(self, trainer: BaseTrainer, val_metrics: dict[str, float]) -> None:
        pass

    def on_train_end(self, trainer: BaseTrainer) -> None:
        pass

    def state_dict(self) -> dict:
        """Serializable state for checkpoint resume. Default: no state."""
        return {}

    def load_state_dict(self, state: dict) -> None:
        """Restore state saved by :meth:`state_dict`. Default: no-op."""

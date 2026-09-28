"""WandbLoggerCallback: Integrates weights & biases logging."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from phaseforge.trains.callbacks.base import Callback

if TYPE_CHECKING:
    from phaseforge.trains.loops.base import BaseTrainer

logger = logging.getLogger(__name__)


class WandbLoggerCallback(Callback):
    """Logs metrics and configuration to Weights & Biases."""

    def __init__(self) -> None:
        # wandb init should be called prior to the trainer loop
        import wandb

        self.wandb = wandb

    def on_train_step(self, trainer: BaseTrainer, step: int, metrics: dict[str, float]) -> None:
        log_dict = {f"train/{k}": v for k, v in metrics.items()}
        log_dict["global_step"] = step
        log_dict["epoch"] = trainer.current_epoch

        lr = trainer.optimizer.param_groups[0]["lr"]
        log_dict["train/lr"] = lr

        if self.wandb.run is not None:
            self.wandb.log(log_dict, step=step)

    def on_epoch_end(self, trainer: BaseTrainer, val_metrics: dict[str, float]) -> None:
        if not val_metrics:
            return

        log_dict = {f"val/{k}": v for k, v in val_metrics.items()}
        log_dict["global_step"] = trainer.global_step
        log_dict["epoch"] = trainer.current_epoch

        if self.wandb.run is not None:
            self.wandb.log(log_dict, step=trainer.global_step)

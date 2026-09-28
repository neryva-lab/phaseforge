"""CPU-only tests for the early-stopping callback."""

from __future__ import annotations

from phaseforge.trains.callbacks.early_stopping import EarlyStoppingCallback


class FakeTrainer:
    def __init__(self) -> None:
        self.should_stop = False


def test_disabled_callback_never_stops() -> None:
    trainer = FakeTrainer()
    cb = EarlyStoppingCallback(monitor="val/loss_total", mode="min", patience=1, enabled=False)
    cb.on_epoch_end(trainer, {"loss_total": 1.0})
    cb.on_epoch_end(trainer, {"loss_total": 1.0})
    assert trainer.should_stop is False
    assert cb.wait_count == 0


def test_enabled_callback_stops_after_patience() -> None:
    trainer = FakeTrainer()
    cb = EarlyStoppingCallback(monitor="val/loss_total", mode="min", patience=2, enabled=True)
    cb.on_epoch_end(trainer, {"loss_total": 1.0})
    cb.on_epoch_end(trainer, {"loss_total": 1.0})
    assert trainer.should_stop is False
    cb.on_epoch_end(trainer, {"loss_total": 1.0})
    assert trainer.should_stop is True


def test_monitor_key_prefix_is_stripped() -> None:
    trainer = FakeTrainer()
    cb = EarlyStoppingCallback(monitor="val/loss_total", mode="min", patience=1)
    cb.on_epoch_end(trainer, {"loss_total": 1.0})
    cb.on_epoch_end(trainer, {"loss_total": 1.0})
    assert trainer.should_stop is True

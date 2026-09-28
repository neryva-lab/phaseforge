"""CPU-only tests for checkpointing: top-k best checkpoints and resume.

Verifies the CheckpointCallback's real top-k behavior (collection,
eviction of files, rank-1 alias) and that ``BaseTrainer.resume`` restores
epoch/step, RNG state and callback state from a saved checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import DictConfig

from phaseforge.trains.callbacks.checkpointing import CheckpointCallback
from phaseforge.trains.callbacks.early_stopping import EarlyStoppingCallback
from phaseforge.trains.loops.base import BaseTrainer


class _StageModel(torch.nn.Linear):
    """Linear with a checkpointed stage attribute (like BaseManipulationModel)."""

    def __init__(self) -> None:
        super().__init__(4, 4)
        self.stage = 1


class _TinyTrainer(BaseTrainer):
    """Minimal concrete trainer: never actually trains."""

    def _compute_loss(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        return torch.tensor(0.0), {}


def _make_cfg() -> DictConfig:
    return DictConfig(
        {
            "project": {"device": "cpu"},
            "train": {
                "epochs": 1,
                "grad_clip_norm": 0.0,
                "log_every_n_steps": 10,
                "optimizer": {
                    "_target_": "torch.optim.AdamW",
                    "lr": 1.0e-3,
                },
                "scheduler": {
                    "_target_": "torch.optim.lr_scheduler.CosineAnnealingLR",
                    "T_max": 1,
                },
            },
        }
    )


def _make_trainer() -> _TinyTrainer:
    return _TinyTrainer(cfg=_make_cfg(), model=_StageModel(), train_loader=None, val_loader=None)


def test_save_top_k_prunes_and_updates_alias(tmp_path: Path) -> None:
    cb = CheckpointCallback(
        output_dir=tmp_path,
        every_n_epochs=100,  # never trigger periodic snapshots
        monitor="loss_total",
        mode="min",
        save_top_k=2,
    )
    trainer = _make_trainer()

    # Scores: epoch 1 = 3.0, epoch 2 = 1.0 (best), epoch 3 = 2.0.
    for epoch, score in [(1, 3.0), (2, 1.0), (3, 2.0)]:
        trainer.current_epoch = epoch
        cb.on_epoch_end(trainer, {"loss_total": score})

    names = sorted(p.name for p in tmp_path.iterdir())
    # Epoch 1 (worst of the three) is evicted; the rank-1 alias exists.
    assert names == [
        "checkpoint_best.pt",
        "checkpoint_best_epoch_0002.pt",
        "checkpoint_best_epoch_0003.pt",
    ]

    alias = torch.load(tmp_path / "checkpoint_best.pt", weights_only=False)
    best = torch.load(tmp_path / "checkpoint_best_epoch_0002.pt", weights_only=False)
    assert alias["epoch"] == 2
    assert best["epoch"] == 2
    # The alias is a copy of the rank-1 snapshot.
    assert alias["global_step"] == best["global_step"]


def test_prefixed_monitor_matches_stage2_val_metrics(tmp_path: Path) -> None:
    # Stage 2 validation returns already-prefixed keys ('val/routing_entropy').
    # A prefixed monitor must still drive best-checkpoint selection instead of
    # silently selecting nothing.
    cb = CheckpointCallback(
        output_dir=tmp_path,
        every_n_epochs=100,
        monitor="val/routing_entropy",
        mode="max",
        save_top_k=1,
    )
    trainer = _make_trainer()
    trainer.current_epoch = 1
    cb.on_epoch_end(trainer, {"val/routing_entropy": 0.5})
    assert cb.best_ckpt_path is not None
    assert cb.best_score == pytest.approx(0.5)
    assert (tmp_path / "checkpoint_best.pt").exists()


def test_nonfinite_monitor_never_admitted_as_best(tmp_path: Path) -> None:
    # A NaN/Inf monitor at epoch 1 must not pin a garbage "best" checkpoint:
    # float-NaN comparisons are False in both directions, so without the
    # guard the NaN would sit at the top-k head forever and be exported to
    # the eval path via checkpoint_best.pt.
    cb = CheckpointCallback(
        output_dir=tmp_path,
        every_n_epochs=100,
        monitor="loss_total",
        mode="min",
        save_top_k=1,
    )
    trainer = _make_trainer()

    trainer.current_epoch = 1
    cb.on_epoch_end(trainer, {"loss_total": float("nan")})
    assert cb.best_ckpt_path is None
    assert not (tmp_path / "checkpoint_best.pt").exists()

    # The next finite score becomes best instead of being locked out.
    trainer.current_epoch = 2
    cb.on_epoch_end(trainer, {"loss_total": 0.7})
    assert cb.best_score == pytest.approx(0.7)
    assert (tmp_path / "checkpoint_best.pt").exists()
    assert cb.best_ckpt_path is not None

    # A later Inf/NaN still never replaces the finite best.
    trainer.current_epoch = 3
    cb.on_epoch_end(trainer, {"loss_total": float("inf")})
    trainer.current_epoch = 4
    cb.on_epoch_end(trainer, {"loss_total": float("nan")})
    assert cb.best_score == pytest.approx(0.7)


def test_resume_restores_rng_epoch_step_and_callback_state(tmp_path: Path) -> None:
    # Trainer A: simulate a run at epoch 2, step 5, with early-stopping
    # state already accumulated, and save a checkpoint through the callback.
    trainer_a = _make_trainer()
    trainer_a.current_epoch = 2
    trainer_a.global_step = 5
    trainer_a.model.stage = 2
    early_stopping = EarlyStoppingCallback(monitor="loss_total", patience=3)
    early_stopping.wait_count = 2
    early_stopping.best_score = 1.0
    trainer_a.callbacks = [early_stopping]

    cb = CheckpointCallback(output_dir=tmp_path, every_n_epochs=10, save_top_k=1)
    # Capture the RNG state the checkpoint will record, immediately before
    # saving (model construction above already consumed RNG).
    torch.manual_seed(999)
    saved_rng = torch.get_rng_state()
    ckpt_path = tmp_path / "resume.pt"
    torch.save(cb._build_state(trainer_a, epoch=2), ckpt_path)

    # Corrupt the process RNG state: resuming must restore it exactly.
    torch.manual_seed(0)
    assert not torch.equal(torch.get_rng_state(), saved_rng)

    trainer_b = _make_trainer()
    trainer_b.add_callback(EarlyStoppingCallback(monitor="loss_total", patience=3))
    trainer_b.resume(ckpt_path)
    trainer_b._apply_resume_payload()

    assert trainer_b.current_epoch == 2
    assert trainer_b.global_step == 5
    assert trainer_b.model.stage == 2
    assert torch.equal(torch.get_rng_state(), saved_rng)

    es_b = trainer_b.callbacks[0]
    assert isinstance(es_b, EarlyStoppingCallback)
    assert es_b.wait_count == 2
    assert es_b.best_score == 1.0


def test_resume_rejects_missing_file(tmp_path: Path) -> None:
    trainer = _make_trainer()
    with pytest.raises(FileNotFoundError, match="not found"):
        trainer.resume(tmp_path / "does_not_exist.pt")


def test_min_epoch_ignores_early_epochs(tmp_path: Path) -> None:
    """Scores before min_epoch must never enter the top-k collection.

    This prevents transient noise dips during high-LR training from
    permanently locking checkpoint_best.pt.
    """
    cb = CheckpointCallback(
        output_dir=tmp_path,
        every_n_epochs=100,  # never trigger periodic snapshots
        monitor="loss_total",
        mode="min",
        save_top_k=1,
        min_epoch=50,
    )
    trainer = _make_trainer()

    # A very good score *before* min_epoch: must be ignored.
    trainer.current_epoch = 10
    cb.on_epoch_end(trainer, {"loss_total": 0.001})
    assert cb.best_ckpt_path is None
    assert not (tmp_path / "checkpoint_best.pt").exists()

    # Another early epoch, right before the cutoff.
    trainer.current_epoch = 49
    cb.on_epoch_end(trainer, {"loss_total": 0.0005})
    assert cb.best_ckpt_path is None

    # At min_epoch: this score should be accepted.
    trainer.current_epoch = 50
    cb.on_epoch_end(trainer, {"loss_total": 0.5})
    assert cb.best_ckpt_path is not None
    assert cb.best_score == pytest.approx(0.5)
    assert (tmp_path / "checkpoint_best.pt").exists()

    # A later, better score replaces it normally.
    trainer.current_epoch = 100
    cb.on_epoch_end(trainer, {"loss_total": 0.3})
    assert cb.best_score == pytest.approx(0.3)
    alias = torch.load(tmp_path / "checkpoint_best.pt", weights_only=False)
    assert alias["epoch"] == 100


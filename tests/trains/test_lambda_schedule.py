"""Tests for the stage-1 λ schedule (train.lambda_schedule)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize
from torch.utils.data import DataLoader

from phaseforge.data.common.collator import PhaseAwareCollator
from phaseforge.trains.callbacks.persistence import MetricPersistenceCallback
from phaseforge.trains.loops.stage1_loop import Stage1Trainer
from phaseforge.utils.registry import build_model, build_trainer


def _build_trainer(overrides: list[str], epochs: int = 1):
    with initialize(version_base="1.3", config_path="../../phaseforge/config"):
        cfg = compose(
            config_name="main",
            overrides=[
                "models=precision_residual_phaseforge",
                "data=lift",
                "train=stage1",
                "project.device=cpu",
                f"train.epochs={epochs}",
                *overrides,
            ],
        )
    model = build_model(cfg)
    batch = [
        {
            "state": torch.randn(10, 19),
            "action": torch.randn(10, 7),
            "phase": torch.randint(0, 6, (10,)),
            "task_id": 0,
            "trajectory_id": 0,
            "trajectory_position": 0,
        },
        {
            "state": torch.randn(8, 19),
            "action": torch.randn(8, 7),
            "phase": torch.randint(0, 6, (8,)),
            "task_id": 0,
            "trajectory_id": 1,
            "trajectory_position": 0,
        },
    ]
    collator = PhaseAwareCollator()
    loader = DataLoader(batch, batch_size=2, collate_fn=collator)
    return build_trainer(cfg, model, loader, loader), cfg


def _fit_and_read_rows(tmp_path: Path, trainer) -> list[dict]:
    cb = MetricPersistenceCallback(tmp_path / "run", "run0001", data_config_hash="h")
    trainer.add_callback(cb)
    trainer.fit()
    rows = (tmp_path / "run" / "metrics" / "training_curves.jsonl").read_text().splitlines()
    assert rows
    return [json.loads(line) for line in rows]


def test_default_schedule_is_constant_protocol() -> None:
    trainer, _ = _build_trainer([])
    assert isinstance(trainer, Stage1Trainer)
    assert trainer._effective_lambda_phase() == pytest.approx(1.0)


def test_linear_schedule_endpoints() -> None:
    trainer, _ = _build_trainer(["train.lambda_schedule.type=linear"], epochs=10)
    trainer.current_epoch = 0
    assert trainer._effective_lambda_phase() == pytest.approx(1.0)
    trainer.current_epoch = 10
    assert trainer._effective_lambda_phase() == pytest.approx(0.0)
    trainer.current_epoch = 5
    assert trainer._effective_lambda_phase() == pytest.approx(0.5)


def test_linear_schedule_scales_base_lambda() -> None:
    trainer, _ = _build_trainer(
        [
            "train.lambda_schedule.type=linear",
            "train.lambda_phase=2.0",
            "train.lambda_schedule.end=0.5",
        ],
        epochs=10,
    )
    trainer.current_epoch = 10
    assert trainer._effective_lambda_phase() == pytest.approx(1.0)  # 2.0 * 0.5
    trainer.current_epoch = 0
    assert trainer._effective_lambda_phase() == pytest.approx(2.0)


def test_linear_schedule_rejects_invalid_bounds() -> None:
    trainer, _ = _build_trainer(
        ["train.lambda_schedule.type=linear", "train.lambda_schedule.end=1.5"]
    )
    with pytest.raises(ValueError, match="0 <= end <= start <= 1"):
        trainer._effective_lambda_phase()
    trainer2, _ = _build_trainer(
        [
            "train.lambda_schedule.type=linear",
            "train.lambda_schedule.start=0.5",
            "train.lambda_schedule.end=0.8",
        ]
    )
    with pytest.raises(ValueError, match="0 <= end <= start <= 1"):
        trainer2._effective_lambda_phase()


def test_unknown_schedule_type_rejects() -> None:
    trainer, _ = _build_trainer(["train.lambda_schedule.type=banana"])
    with pytest.raises(ValueError, match="banana"):
        trainer._effective_lambda_phase()


def test_linear_schedule_emits_lambda_curve_and_weights_loss(tmp_path: Path) -> None:
    trainer, _ = _build_trainer(
        [
            "train.lambda_schedule.type=linear",
            "train.lambda_schedule.start=1.0",
            "train.lambda_schedule.end=0.0",
        ],
        epochs=3,
    )
    rows = _fit_and_read_rows(tmp_path, trainer)
    assert len(rows) == 3
    for i, row in enumerate(rows):
        assert "train/lambda_phase" in row
        # current_epoch is 1-indexed inside the fit (1, 2, 3); start=1 -> end=0.
        expected = 1.0 - ((i + 1) / 3.0)
        assert row["train/lambda_phase"] == pytest.approx(expected, abs=1e-4)
    # The phase loss is still reported raw (not masked by the schedule).
    assert all("train/loss_phase" in row for row in rows)


def test_constant_schedule_matches_no_schedule(tmp_path: Path) -> None:
    torch.manual_seed(1234)
    rows_default = _fit_and_read_rows(tmp_path, _build_trainer([])[0])
    torch.manual_seed(1234)
    rows_constant = _fit_and_read_rows(
        tmp_path / "constant", _build_trainer(["train.lambda_schedule.type=constant"])[0]
    )
    assert len(rows_default) == len(rows_constant)
    _TIMING_KEYS = {"epoch_wall_seconds", "train_steps_per_second"}
    for a, b in zip(rows_default, rows_constant):
        shared_a = {k: v for k, v in a.items() if k not in _TIMING_KEYS | {"train/lambda_phase"}}
        shared_b = {k: v for k, v in b.items() if k not in _TIMING_KEYS | {"train/lambda_phase"}}
        assert shared_a == shared_b, (
            "constant schedule must not change training dynamics"
        )

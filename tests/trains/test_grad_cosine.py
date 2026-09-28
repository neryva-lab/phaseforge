"""Tests for the stage-1 grad-cosine auxiliary-conflict diagnostic."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from hydra import compose, initialize
from torch.utils.data import DataLoader

from phaseforge.data.common.collator import PhaseAwareCollator
from phaseforge.trains.callbacks.persistence import MetricPersistenceCallback
from phaseforge.trains.loops.stage1_loop import _grad_cosine_similarity
from phaseforge.utils.registry import build_model, build_trainer


class _TinyNet(nn.Module):
    """Shared encoder + two heads, mirroring the stage-1 structure."""

    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Linear(4, 4)
        self.action_head = nn.Linear(4, 2)
        self.phase_head = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.relu(self.shared(x))
        return self.action_head(h), self.phase_head(h)


def test_grad_cosine_unit_range_and_sign() -> None:
    torch.manual_seed(0)
    net = _TinyNet()
    x = torch.randn(8, 4)
    act, ph = net(x)
    loss_a = act.mean()
    loss_p = ph.mean()

    cos = _grad_cosine_similarity(loss_a, loss_p, net.parameters())
    assert cos is not None
    assert -1.0 <= float(cos) <= 1.0

    # Identical losses => identical gradients => cos == +1.
    cos_same = _grad_cosine_similarity(loss_a, loss_a, net.parameters())
    assert cos_same is not None
    assert float(cos_same) == pytest.approx(1.0)

    # Opposite losses => cos == -1.
    cos_opp = _grad_cosine_similarity(loss_a, -loss_a, net.parameters())
    assert cos_opp is not None
    assert float(cos_opp) == pytest.approx(-1.0)


def test_grad_cosine_ignores_non_participating_params() -> None:
    torch.manual_seed(0)
    net = _TinyNet()
    x = torch.randn(8, 4)
    act, ph = net(x)
    # The action head gradient does not touch the phase head and vice versa;
    # unused params must contribute zeros (same-shaped vectors) rather than
    # raising or mismatching lengths.
    cos = _grad_cosine_similarity(act.mean(), ph.mean(), net.parameters())
    assert cos is not None
    assert -1.0 <= float(cos) <= 1.0


def test_grad_cosine_degenerate_returns_none() -> None:
    torch.manual_seed(0)
    net = _TinyNet()
    x = torch.randn(8, 4)
    act, _ = net(x)
    zero = torch.zeros((), requires_grad=False)
    # A constant loss (no grad) must yield None, not a RuntimeError.
    assert _grad_cosine_similarity(act.mean(), zero, net.parameters()) is None
    assert _grad_cosine_similarity(zero, act.mean(), net.parameters()) is None

    # Frozen parameters are skipped entirely.
    net.shared.weight.requires_grad = False
    cos = _grad_cosine_similarity(act.mean(), act.mean(), net.parameters())
    assert cos is not None


def _build_stage1_trainer(
    grad_cosine: bool, model_cfg: str = "precision_residual_phaseforge"
):
    with initialize(version_base="1.3", config_path="../../phaseforge/config"):
        cfg = compose(
            config_name="main",
            overrides=[
                f"models={model_cfg}",
                "data=lift",
                "train=stage1",
                "project.device=cpu",
                "train.epochs=1",
                f"train.grad_cosine={str(grad_cosine).lower()}",
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
    trainer = build_trainer(cfg, model, loader, loader)
    return trainer, cfg


def _fit_and_read_row(tmp_path: Path, trainer) -> dict:
    """Fit one epoch with a persistence callback and return the curve row."""
    cb = MetricPersistenceCallback(tmp_path / "run", "run0001", data_config_hash="h")
    trainer.add_callback(cb)
    trainer.fit()
    rows = list(
        (tmp_path / "run" / "metrics" / "training_curves.jsonl")
        .read_text()
        .splitlines()
    )
    assert rows, "expected at least one curve row"
    import json

    return json.loads(rows[-1])


def test_grad_cosine_emitted_when_enabled(tmp_path: Path) -> None:
    trainer, _ = _build_stage1_trainer(True)
    row = _fit_and_read_row(tmp_path, trainer)
    assert "train/grad_cos_action_phase" in row
    assert -1.0 <= row["train/grad_cos_action_phase"] <= 1.0


def test_grad_cosine_absent_by_default(tmp_path: Path) -> None:
    trainer, _ = _build_stage1_trainer(False)
    row = _fit_and_read_row(tmp_path, trainer)
    assert "train/grad_cos_action_phase" not in row


def test_grad_cosine_absent_without_phase_head(tmp_path: Path) -> None:
    # bc has no phase head: the phase loss carries no gradient, so the
    # diagnostic must be skipped (absence is honest, never a fabricated zero).
    trainer, _ = _build_stage1_trainer(True, model_cfg="final_aligned_bc")
    row = _fit_and_read_row(tmp_path, trainer)
    assert "train/grad_cos_action_phase" not in row

"""CPU-only tests for the Stage 2 margin term."""

from __future__ import annotations

import pytest
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from phaseforge.models.base import ModelOutput
from phaseforge.trains.loops.stage2_loop import Stage2Trainer
from tests.trains.test_stage2_loop import _DictDataset


class _MarginMoEModel(torch.nn.Module):
    """Fake MoE fronting a real PrototypeRouter (distances as gate logits)."""

    def __init__(self, num_experts: int = 3) -> None:
        super().__init__()
        from phaseforge.models.components.prototype_router import PrototypeRouter

        self.probe = torch.nn.Parameter(torch.zeros(1))
        self.moe_layer = _Shim(PrototypeRouter(latent_dim=4, num_experts=num_experts))

    def forward(self, batch: dict[str, torch.Tensor]) -> ModelOutput:
        states = batch["state"] if batch["state"].ndim == 2 else batch["state"][:, 0]
        out = self.moe_layer.router(states[:, :4].contiguous())
        action_pred = self.probe * torch.zeros(batch["action"].shape[0], 2)
        return ModelOutput(
            action_pred=action_pred,
            phase_logits=None,
            routing_weights=out.weights,
            expert_indices=out.indices,
            gate_logits=out.gate_logits,
            aux_losses={"balance": out.balance_loss},
        )

    def freeze_encoder(self) -> None:
        pass


class _Shim:
    def __init__(self, router) -> None:
        self.router = router


def _make_trainer(model, margin: dict | None) -> Stage2Trainer:
    train_cfg: dict = {
        "epochs": 1,
        "grad_clip_norm": 0.0,
        "log_every_n_steps": 1,
        "freeze_encoder": False,
        "optimizer": {"_target_": "torch.optim.AdamW", "lr": 1.0e-4},
        "scheduler": {"_target_": "torch.optim.lr_scheduler.CosineAnnealingLR", "T_max": 1},
    }
    if margin is not None:
        train_cfg["margin"] = margin
    cfg = DictConfig({"project": {"device": "cpu"}, "train": train_cfg})
    loader = DataLoader(_DictDataset(num=32, num_phases=3, seed=0), batch_size=8)
    return Stage2Trainer(cfg=cfg, model=model, train_loader=loader, val_loader=loader)


def test_margin_disabled_by_default_preserves_metric_keys() -> None:
    from tests.trains.test_stage2_loop import CountingMoEModel

    trainer = _make_trainer(CountingMoEModel(num_experts=3), margin=None)
    batch = next(iter(trainer.train_loader))
    _total, metrics = trainer._compute_loss(batch)
    assert set(metrics) == {
        "loss_total",
        "loss_action",
        "loss_balance",
        "loss_sticky",
        "loss_teacher_kl",
        "teacher_lambda",
    }


def test_margin_enabled_adds_term() -> None:
    torch.manual_seed(0)
    trainer = _make_trainer(
        _MarginMoEModel(),
        margin={"enabled": True, "lambda_margin": 1.0, "margin": 0.5},
    )
    batch = next(iter(trainer.train_loader))
    total, metrics = trainer._compute_loss(batch)
    assert "loss_margin" in metrics and "margin_lambda" in metrics
    assert metrics["margin_lambda"] == pytest.approx(1.0)
    assert metrics["loss_margin"].item() >= 0.0
    expected = metrics["loss_action"] + metrics["loss_balance"] + metrics["loss_margin"]
    assert total.detach().item() == pytest.approx(expected.item(), rel=1e-5)


def test_margin_enabled_zero_lambda_disables_term() -> None:
    trainer = _make_trainer(
        _MarginMoEModel(),
        margin={"enabled": True, "lambda_margin": 0.0, "margin": 0.5},
    )
    batch = next(iter(trainer.train_loader))
    total, metrics = trainer._compute_loss(batch)
    assert total.detach().item() == pytest.approx(
        (metrics["loss_action"] + metrics["loss_balance"]).item(), rel=1e-5
    )


def test_margin_enabled_without_prototype_router_fails_closed() -> None:
    from tests.trains.test_stage2_loop import CountingMoEModel

    trainer = _make_trainer(
        CountingMoEModel(num_experts=3),
        margin={"enabled": True, "lambda_margin": 1.0, "margin": 0.5},
    )
    with pytest.raises(RuntimeError, match="prototype router"):
        trainer._compute_loss(next(iter(trainer.train_loader)))


def test_default_stage2_config_has_margin_disabled() -> None:
    from pathlib import Path

    from omegaconf import OmegaConf

    repo = Path(__file__).resolve().parents[2]
    cfg = OmegaConf.load(str(repo / "phaseforge" / "config" / "train" / "stage2.yaml"))
    assert cfg["margin"]["enabled"] is False

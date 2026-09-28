"""CPU-only tests for the full Stage 2 IS objective.

Exercises L_total = L_action + margin + lip + gain + balance (plus the
legacy sticky/teacher terms at zero) through a real IS-PhaseForge model so
the Stage 1 -> Stage 2 seams (latent/info/router) are covered, not mocked.
"""

from __future__ import annotations

import pytest
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from phaseforge.models.components.action_head import ActionHead
from phaseforge.models.components.encoder import StateEncoder
from phaseforge.models.components.impedance_expert import ImpedanceExpert
from phaseforge.models.components.phase_head import PhaseClassificationHead
from phaseforge.models.components.prototype_router import PrototypeRouter
from phaseforge.models.phase_moe import PhaseBootstrappedMoE
from phaseforge.trains.loops.stage2_loop import Stage2Trainer
from tests.trains.test_stage2_loop import _DictDataset


def _is_model() -> PhaseBootstrappedMoE:
    torch.manual_seed(0)
    encoder = StateEncoder(input_dim=19, hidden_dims=[16], latent_dim=8)
    model = PhaseBootstrappedMoE(
        encoder=encoder,
        action_head=ActionHead(input_dim=8, output_dim=7, hidden_dim=16),
        phase_head=PhaseClassificationHead(latent_dim=8, num_phases=2),
        router=PrototypeRouter(latent_dim=8, num_experts=2, top_k=1),
        expert=ImpedanceExpert(input_dim=8, hidden_dim=16),
        router_init={"type": "centroid", "prototype_source": "rule", "seed": 0},
        expert_init={"type": "random"},
    )
    model.stage = 2
    return model


def _rows(num: int = 32, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    return [
        {
            "state": torch.randn(19, generator=gen),
            "action": torch.randn(7, generator=gen),
            "phase": torch.tensor(int(i % 2)),
        }
        for i in range(num)
    ]


def _make_trainer(model, extra: dict | None = None) -> Stage2Trainer:
    train_cfg: dict = {
        "epochs": 1,
        "grad_clip_norm": 0.0,
        "log_every_n_steps": 1,
        "freeze_encoder": False,
        "optimizer": {"_target_": "torch.optim.AdamW", "lr": 1.0e-4},
        "scheduler": {"_target_": "torch.optim.lr_scheduler.CosineAnnealingLR", "T_max": 1},
    }
    if extra:
        train_cfg.update(extra)
    cfg = DictConfig({"project": {"device": "cpu"}, "train": train_cfg})
    loader = DataLoader(_rows(), batch_size=8)
    return Stage2Trainer(cfg=cfg, model=model, train_loader=loader, val_loader=loader)


def test_full_objective_sums_all_terms() -> None:
    trainer = _make_trainer(
        _is_model(),
        extra={
            "margin": {"enabled": True, "lambda_margin": 1.0, "margin": 0.5},
            "lipschitz": {"enabled": True, "lambda_lip": 0.5, "rho": 0.8, "num_pairs": 64},
            "gain_reg": {"enabled": True, "lambda_gain": 1.0e-4, "kappa_nominal": 1.0},
        },
    )
    batch = next(iter(trainer.train_loader))
    total, metrics = trainer._compute_loss(batch)
    for key in ("loss_margin", "loss_lip", "loss_gain", "margin_lambda"):
        assert key in metrics
    assert metrics["loss_lip"].item() >= 0.0
    assert metrics["loss_gain"].item() > 0.0
    expected = (
        metrics["loss_action"]
        + metrics["loss_balance"]
        + metrics["loss_margin"]
        + 0.5 * metrics["loss_lip"]
        + 1.0e-4 * metrics["loss_gain"]
    )
    assert total.detach().item() == pytest.approx(expected.item(), rel=1e-5)
    total.backward()
    assert torch.isfinite(trainer.model.moe_layer.experts[0].target_head.weight.grad).all()


def test_all_zero_lambdas_match_disabled() -> None:
    enabled = {
        "margin": {"enabled": True, "lambda_margin": 0.0, "margin": 0.5},
        "lipschitz": {"enabled": True, "lambda_lip": 0.0, "rho": 0.8, "num_pairs": 64},
        "gain_reg": {"enabled": True, "lambda_gain": 0.0, "kappa_nominal": 1.0},
    }
    torch.manual_seed(0)
    on_total, _metrics = _make_trainer(_is_model(), extra=enabled)._compute_loss(
        next(iter(DataLoader(_rows(), batch_size=8)))
    )
    torch.manual_seed(0)
    off_total, off_metrics = _make_trainer(_is_model())._compute_loss(
        next(iter(DataLoader(_rows(), batch_size=8)))
    )
    assert on_total.detach().item() == pytest.approx(off_total.detach().item())
    assert set(off_metrics) == {
        "loss_total",
        "loss_action",
        "loss_balance",
        "loss_sticky",
        "loss_teacher_kl",
        "teacher_lambda",
    }


def test_lip_gain_require_impedance_info() -> None:
    from tests.trains.test_stage2_loop import CountingMoEModel

    loader = DataLoader(_DictDataset(num=16, seed=1), batch_size=8)
    cfg = DictConfig(
        {
            "project": {"device": "cpu"},
            "train": {
                "epochs": 1,
                "grad_clip_norm": 0.0,
                "log_every_n_steps": 1,
                "freeze_encoder": False,
                "optimizer": {"_target_": "torch.optim.AdamW", "lr": 1.0e-4},
                "scheduler": {
                    "_target_": "torch.optim.lr_scheduler.CosineAnnealingLR",
                    "T_max": 1,
                },
                "lipschitz": {"enabled": True},
            },
        }
    )
    trainer = Stage2Trainer(
        cfg=cfg, model=CountingMoEModel(num_experts=3), train_loader=loader, val_loader=loader
    )
    with pytest.raises(RuntimeError, match="impedance"):
        trainer._compute_loss(next(iter(loader)))

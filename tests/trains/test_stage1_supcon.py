"""CPU-only tests for Stage 1 SupCon wiring."""

from __future__ import annotations

import pytest
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from phaseforge.models.base import ModelOutput
from phaseforge.trains.loops.stage1_loop import Stage1Trainer
from phaseforge.trains.losses.supcon import supcon_loss


def _normed(n: int, dim: int = 8, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(n, dim, generator=gen), dim=-1)


def test_supcon_loss_separates_regimes() -> None:
    latents = _normed(16)
    labels = torch.tensor([0] * 8 + [1] * 8)
    taut = supcon_loss(latents, labels, temperature=0.07)
    assert torch.isfinite(taut).item()
    assert taut.item() > 0.0
    # Colder temperature sharpens the same logits -> larger loss here.
    assert supcon_loss(latents, labels, temperature=0.01).item() >= taut.item()


def test_supcon_loss_gradient_flows() -> None:
    latents = _normed(8)
    latents.requires_grad_(True)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    supcon_loss(latents, labels).backward()
    assert latents.grad is not None
    assert torch.isfinite(latents.grad).all()


def test_supcon_loss_singleton_safe() -> None:
    latents = _normed(4)
    labels = torch.tensor([0, 1, 2, 3])
    assert supcon_loss(latents, labels).item() == pytest.approx(0.0)
    with pytest.raises(ValueError):
        supcon_loss(latents, labels, temperature=0.0)
    with pytest.raises(ValueError):
        supcon_loss(torch.randn(4), labels)


class _DictDataset(Dataset):
    def __init__(self, num: int = 32, seed: int = 0) -> None:
        gen = torch.Generator().manual_seed(seed)
        self.states = torch.randn(num, 4, generator=gen)
        self.actions = torch.randn(num, 2, generator=gen)
        self.phases = torch.randint(0, 2, (num,), generator=gen)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "state": self.states[idx],
            "action": self.actions[idx],
            "phase": self.phases[idx],
        }


class _SupconModel(torch.nn.Module):
    """Fake Stage 1 model exposing latents (mirrors PhaseBootstrappedMoE)."""

    def __init__(self) -> None:
        super().__init__()
        self.probe = torch.nn.Parameter(torch.zeros(1))
        gen = torch.Generator().manual_seed(0)
        self.fixed_latents = torch.nn.functional.normalize(torch.randn(32, 8, generator=gen))

    def forward(self, batch: dict[str, torch.Tensor]) -> ModelOutput:
        action_pred = self.probe * torch.zeros_like(batch["action"])
        return ModelOutput(
            action_pred=action_pred,
            phase_logits=None,
            routing_weights=None,
            expert_indices=None,
            gate_logits=None,
            latent=self.fixed_latents[: batch["action"].size(0)],
        )

    def freeze_encoder(self) -> None:
        pass


class _NoLatentModel(_SupconModel):
    def forward(self, batch: dict[str, torch.Tensor]) -> ModelOutput:
        out = super().forward(batch)
        out.latent = None
        return out


def _make_trainer(model, supcon: dict | None) -> Stage1Trainer:
    train_cfg: dict = {
        "epochs": 1,
        "grad_clip_norm": 0.0,
        "log_every_n_steps": 1,
        "lambda_phase": 0.0,
        "optimizer": {"_target_": "torch.optim.AdamW", "lr": 1.0e-4},
        "scheduler": {"_target_": "torch.optim.lr_scheduler.CosineAnnealingLR", "T_max": 1},
    }
    if supcon is not None:
        train_cfg["supcon"] = supcon
    cfg = DictConfig({"project": {"device": "cpu"}, "train": train_cfg})
    loader = DataLoader(_DictDataset(), batch_size=8)
    return Stage1Trainer(cfg=cfg, model=model, train_loader=loader, val_loader=loader)


def test_stage1_disabled_by_default_is_bit_identical() -> None:
    trainer = _make_trainer(_SupconModel(), supcon=None)
    batch = next(iter(trainer.train_loader))
    total, metrics = trainer._compute_loss(batch)
    assert "loss_supcon" not in metrics
    assert "supcon_lambda" not in metrics
    assert total.requires_grad


def test_stage1_enabled_adds_supcon_term() -> None:
    trainer = _make_trainer(
        _SupconModel(),
        supcon={"enabled": True, "lambda_sc": 1.0, "temperature": 0.07, "label_field": "phase"},
    )
    batch = next(iter(trainer.train_loader))
    total, metrics = trainer._compute_loss(batch)
    assert metrics["supcon_lambda"] == pytest.approx(1.0)
    assert metrics["loss_supcon"].item() > 0.0
    expected = metrics["loss_action"] + metrics["loss_supcon"]
    assert total.detach().item() == pytest.approx(expected.item())


def test_stage1_enabled_zero_lambda_disables_term() -> None:
    trainer = _make_trainer(
        _SupconModel(),
        supcon={"enabled": True, "lambda_sc": 0.0, "temperature": 0.07, "label_field": "phase"},
    )
    batch = next(iter(trainer.train_loader))
    total, metrics = trainer._compute_loss(batch)
    assert metrics["loss_supcon"].item() > 0.0
    assert total.detach().item() == pytest.approx(metrics["loss_action"].item())


def test_stage1_missing_latents_or_labels_fail_closed() -> None:
    trainer = _make_trainer(
        _NoLatentModel(),
        supcon={"enabled": True, "lambda_sc": 1.0, "temperature": 0.07, "label_field": "phase"},
    )
    with pytest.raises(RuntimeError, match="latent"):
        trainer._compute_loss(next(iter(trainer.train_loader)))
    trainer_bad_field = _make_trainer(
        _SupconModel(),
        supcon={
            "enabled": True,
            "lambda_sc": 1.0,
            "temperature": 0.07,
            "label_field": "phase_topo",
        },
    )
    with pytest.raises(RuntimeError, match="phase_topo"):
        trainer_bad_field._compute_loss(next(iter(trainer_bad_field.train_loader)))


def test_default_stage1_config_has_supcon_disabled() -> None:
    from pathlib import Path

    from omegaconf import OmegaConf

    repo = Path(__file__).resolve().parents[2]
    cfg = OmegaConf.load(str(repo / "phaseforge" / "config" / "train" / "stage1.yaml"))
    assert cfg["supcon"]["enabled"] is False

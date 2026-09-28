"""Tests for V2-A stage-1 phase-loss changes: soft targets (label smoothing)
and CUI class weights (train.phase_class_weight="cui")."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from phaseforge.cli import _phase_class_weights
from phaseforge.models.base import ModelOutput
from phaseforge.trains.loops.stage1_loop import Stage1Trainer, _phase_ce


class FakePhaseModel(torch.nn.Module):
    """Deterministic model exposing an action head and a phase head."""

    def __init__(self, num_phases: int = 4) -> None:
        super().__init__()
        self.probe = torch.nn.Parameter(torch.zeros(1))

    def forward(self, batch: dict[str, torch.Tensor]) -> ModelOutput:
        B = batch["action"].size(0)
        action = batch["action"]
        T = action.size(-2) if action.dim() == 3 else 1
        if action.dim() == 3:
            action_pred = torch.zeros(B, T, action.size(-1)) + self.probe * 0.0
            logits = torch.zeros(B, T, 4) + self.probe
        else:
            action_pred = torch.zeros(B, action.size(-1)) + self.probe * 0.0
            logits = torch.zeros(B, 4) + self.probe
        return ModelOutput(
            action_pred=action_pred,
            phase_logits=logits,
            routing_weights=None,
            expert_indices=None,
            gate_logits=None,
            aux_losses={},
        )


def _make_trainer(overrides: dict | None = None) -> Stage1Trainer:
    train_cfg = {
        "epochs": 1,
        "grad_clip_norm": 0.0,
        "log_every_n_steps": 1,
        "optimizer": {
            "_target_": "torch.optim.AdamW",
            "lr": 1.0e-4,
        },
        "scheduler": {
            "_target_": "torch.optim.lr_scheduler.CosineAnnealingLR",
            "T_max": 1,
        },
    }
    if overrides:
        train_cfg.update(overrides)
    cfg = DictConfig({"project": {"device": "cpu"}, "train": train_cfg})
    loader = DataLoader([], batch_size=1)
    return Stage1Trainer(cfg=cfg, model=FakePhaseModel(), train_loader=loader, val_loader=loader)


def _masked_batch(B: int = 4, T: int = 3, seed: int = 0) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return {
        "action": torch.randn(B, T, 2, generator=gen),
        "phase": torch.randint(0, 4, (B, T), generator=gen),
        "padding_mask": torch.rand(B, T, generator=gen) > 0.3,
    }


def _logits(B: int = 4, T: int = 3, C: int = 4, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, C, generator=gen)


def test_phase_ce_eps_zero_bit_identical_to_cross_entropy() -> None:
    logits = _logits()
    targets = torch.randint(0, 4, (4 * 3,))
    flat = logits.view(-1, 4)
    assert torch.equal(_phase_ce(flat, targets, None, 0.0), F.cross_entropy(flat, targets))


def test_phase_ce_eps_zero_with_weights_matches_cross_entropy() -> None:
    logits = _logits()
    targets = torch.randint(0, 4, (4 * 3,))
    flat = logits.view(-1, 4)
    weights = torch.tensor([0.5, 1.0, 2.0, 0.25])
    assert torch.equal(
        _phase_ce(flat, targets, weights, 0.0),
        F.cross_entropy(flat, targets, weight=weights),
    )


def test_phase_ce_smoothing_matches_manual_formula() -> None:
    logits = _logits()
    targets = torch.randint(0, 4, (4 * 3,))
    flat = logits.view(-1, 4)
    eps = 0.2
    got = _phase_ce(flat, targets, None, eps)
    log_probs = F.log_softmax(flat, dim=-1)
    nll = -log_probs.gather(1, targets.unsqueeze(-1)).squeeze(-1)
    uniform_nll = -log_probs.mean(dim=-1)
    expected = ((1.0 - eps) * nll + eps * uniform_nll).mean()
    assert got == pytest.approx(expected, rel=1e-6)


def test_phase_ce_smoothing_applies_weight_once() -> None:
    logits = _logits()
    targets = torch.randint(0, 4, (4 * 3,))
    flat = logits.view(-1, 4)
    eps = 0.2
    weights = torch.tensor([0.5, 1.0, 2.0, 0.25])
    got = _phase_ce(flat, targets, weights, eps)
    log_probs = F.log_softmax(flat, dim=-1)
    nll = -log_probs.gather(1, targets.unsqueeze(-1)).squeeze(-1)
    uniform_nll = -log_probs.mean(dim=-1)
    per_sample = (1.0 - eps) * nll + eps * uniform_nll
    expected = (per_sample * weights[targets]).mean()
    assert got == pytest.approx(expected, rel=1e-6)


def test_compute_loss_soft_targets_masked_batch() -> None:
    trainer = _make_trainer({"soft_target_eps": 0.2})
    batch = _masked_batch()
    out = trainer.model(batch)
    total, metrics = trainer._compute_loss(batch, out=out)

    mask = batch["padding_mask"]
    logits_valid = out.phase_logits.view(-1, 4)[mask.view(-1)]
    targets_valid = batch["phase"].view(-1)[mask.view(-1)]
    log_probs = F.log_softmax(logits_valid, dim=-1)
    nll = -log_probs.gather(1, targets_valid.unsqueeze(-1)).squeeze(-1)
    uniform_nll = -log_probs.mean(dim=-1)
    expected_phase = ((1.0 - 0.2) * nll + 0.2 * uniform_nll).mean()
    assert metrics["loss_phase"].detach() == pytest.approx(
        expected_phase.detach(), rel=1e-6
    )
    assert total.requires_grad
    total.backward()
    assert trainer.model.probe.grad is not None


def test_compute_loss_soft_targets_with_weights_matches_manual() -> None:
    weights = [0.5, 1.0, 2.0, 0.25]
    trainer = _make_trainer(
        {"soft_target_eps": 0.1, "phase_weights": weights}
    )
    batch = _masked_batch(seed=7)
    out = trainer.model(batch)
    _, metrics = trainer._compute_loss(batch, out=out)

    mask = batch["padding_mask"]
    logits_valid = out.phase_logits.view(-1, 4)[mask.view(-1)]
    targets_valid = batch["phase"].view(-1)[mask.view(-1)]
    log_probs = F.log_softmax(logits_valid, dim=-1)
    nll = -log_probs.gather(1, targets_valid.unsqueeze(-1)).squeeze(-1)
    uniform_nll = -log_probs.mean(dim=-1)
    w = torch.tensor(weights)
    per_sample = (0.9 * nll + 0.1 * uniform_nll) * w[targets_valid]
    assert metrics["loss_phase"].detach() == pytest.approx(
        per_sample.detach().mean(), rel=1e-6
    )


def test_soft_target_eps_out_of_range_rejected() -> None:
    trainer = _make_trainer({"soft_target_eps": 1.5})
    batch = _masked_batch()
    with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
        trainer._compute_loss(batch)


def test_phase_class_weights_balanced() -> None:
    counts = {0: 10, 1: 100, 2: 40}
    weights = _phase_class_weights("balanced", counts)
    total = sum(counts.values())
    assert weights == pytest.approx(
        [total / (3 * counts[c]) for c in range(3)]
    )


def test_phase_class_weights_cui_formula_and_bounds() -> None:
    counts = {0: 1, 1: 10, 2: 1000}
    beta = 0.9
    weights = _phase_class_weights("cui", counts, beta=beta)
    assert weights == pytest.approx(
        [(1.0 - beta) / (1.0 - beta ** counts[c]) for c in range(3)]
    )
    # Monotone non-increasing in class count; singleton classes cap at 1.
    assert weights[0] == pytest.approx(1.0)
    assert weights[0] >= weights[1] >= weights[2]
    assert all(0.0 < w <= 1.0 for w in weights)


def test_phase_class_weights_cui_invalid_beta_rejected() -> None:
    with pytest.raises(ValueError, match="must be in \\(0, 1\\)"):
        _phase_class_weights("cui", {0: 5}, beta=1.0)
    with pytest.raises(ValueError, match="must be in \\(0, 1\\)"):
        _phase_class_weights("cui", {0: 5}, beta=0.0)


def test_phase_class_weights_unknown_mode_rejected() -> None:
    with pytest.raises(ValueError, match="unknown phase_class_weight"):
        _phase_class_weights("banana", {0: 1})


def test_validate_aggregation_exact_python_float_math() -> None:
    """The base-loop validation aggregation (on-device float64) must
    reproduce the former per-batch Python-float arithmetic EXACTLY, with a
    short final batch exercising the sample weighting."""
    from torch.utils.data import DataLoader, Dataset

    from phaseforge.trains.loops.stage1_loop import Stage1Trainer  # noqa: F401 (already imported)

    class _DS(Dataset):
        def __init__(self, num: int = 7) -> None:
            g = torch.Generator().manual_seed(31)
            self.state = torch.randn(num, 4, generator=g)
            self.action = torch.randn(num, 2, generator=g)
            self.phase = torch.randint(0, 4, (num,), generator=g)

        def __len__(self) -> int:
            return len(self.action)

        def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
            return {"state": self.state[i], "action": self.action[i], "phase": self.phase[i]}

    train_cfg = {
        "epochs": 1,
        "grad_clip_norm": 0.0,
        "log_every_n_steps": 1,
        "lambda_phase": 1.0,
        "optimizer": {"_target_": "torch.optim.AdamW", "lr": 1.0e-4},
        "scheduler": {
            "_target_": "torch.optim.lr_scheduler.CosineAnnealingLR",
            "T_max": 1,
        },
    }
    cfg = DictConfig({"project": {"device": "cpu"}, "train": train_cfg})
    loader = DataLoader(_DS(), batch_size=4)
    model = FakePhaseModel()
    trainer = Stage1Trainer(cfg=cfg, model=model, train_loader=loader, val_loader=loader)

    agg: dict[str, float] = {}
    total = 0
    for batch in loader:
        out = model(batch)
        _, metrics = trainer._compute_loss(batch, out=out)
        n = batch["action"].shape[0]
        for k, v in metrics.items():
            f = float(v.detach().cpu().item()) if isinstance(v, torch.Tensor) else float(v)
            agg[k] = agg.get(k, 0.0) + f * n
        total += n
    expected = {k: v / total for k, v in agg.items()}

    val_metrics = trainer._validate()
    for k, v in expected.items():
        assert k in val_metrics, f"metric {k} missing from validation output"
        assert val_metrics[k] == v, f"{k}: {val_metrics[k]!r} != {v!r}"

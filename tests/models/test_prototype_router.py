"""CPU-only tests for PrototypeRouter."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from phaseforge.evaluations.metrics.init_diagnostics import (
    compute_init_routing_diagnostics,
)
from phaseforge.models.components.action_head import ActionHead
from phaseforge.models.components.encoder import StateEncoder
from phaseforge.models.components.expert import ExpertMLP
from phaseforge.models.components.moe_layer import MoELayer
from phaseforge.models.components.phase_head import PhaseClassificationHead
from phaseforge.models.components.prototype_router import PrototypeRouter
from phaseforge.models.components.router import RouterOutput


def _router(num_experts: int = 3, top_k: int = 1) -> PrototypeRouter:
    torch.manual_seed(0)
    return PrototypeRouter(latent_dim=8, num_experts=num_experts, top_k=top_k)


def test_init_rejects_bad_config() -> None:
    with pytest.raises(ValueError):
        PrototypeRouter(latent_dim=0, num_experts=3)
    with pytest.raises(ValueError):
        PrototypeRouter(latent_dim=8, num_experts=0)
    with pytest.raises(ValueError):
        PrototypeRouter(latent_dim=8, num_experts=2, top_k=3)
    with pytest.raises(ValueError):
        PrototypeRouter(latent_dim=8, num_experts=3, margin=-0.1)


def test_forward_top1_shapes_and_argmin() -> None:
    router = _router()
    with torch.no_grad():
        router.prototypes.copy_(
            torch.tensor([[10.0] + [0.0] * 7, [0.0] * 8, [-10.0] + [0.0] * 7])
        )
    latent = torch.tensor([[9.0] + [0.0] * 7, [-9.0] + [0.0] * 7, [0.0] * 8])
    out = router(latent)
    assert isinstance(out, RouterOutput)
    assert out.weights.shape == (3, 1)
    assert out.indices.shape == (3, 1)
    assert out.gate_logits.shape == (3, 3)
    assert out.indices.flatten().tolist() == [0, 2, 1]
    assert torch.allclose(out.weights, torch.ones(3, 1))
    assert torch.allclose(out.gate_logits, -torch.cdist(latent, router.prototypes))
    with pytest.raises(ValueError):
        router(torch.randn(2, 5))


def test_margin_loss_exact_cases() -> None:
    router = _router()
    dists = torch.tensor([[0.0, 1.0, 2.0]])
    targets = torch.tensor([0])
    assert router.margin_loss(dists, targets, margin=0.5).item() == pytest.approx(0.0)
    # Gaps are [0, 1, 2]; m=1.5 -> [0, 0.5, 0] after masking j=y; sum = 0.5.
    assert router.margin_loss(dists, targets, margin=1.5).item() == pytest.approx(0.5)
    multi = torch.tensor([[0.0, 0.2, 5.0], [3.0, 0.0, 0.1]])
    got = router.margin_loss(multi, torch.tensor([0, 1]), margin=0.5).item()
    # Row 0: max(0, .5-.2)+0 = .3; row 1: 0+max(0, .5-.1)=.4; mean = .35.
    assert got == pytest.approx(0.35)
    assert router.margin_loss_from_logits(-dists, targets, 0.5).item() == pytest.approx(0.0)
    with pytest.raises(ValueError):
        router.margin_loss(dists, torch.tensor([7]), margin=0.5)
    empty = torch.zeros(0, 3)
    assert router.margin_loss(empty, torch.zeros(0, dtype=torch.long)).item() == 0.0


def test_balance_loss_finite_and_scaled() -> None:
    router = PrototypeRouter(latent_dim=4, num_experts=2, balance_coeff=0.5)
    out = router(torch.randn(16, 4))
    assert torch.isfinite(out.balance_loss).item()
    assert out.balance_loss.item() >= 0.0


def test_uniform_ok_others_raise_and_reset_noop() -> None:
    router = _router()
    latent = torch.randn(4, 8)
    weights, indices = router.uniform_selection(latent)
    assert weights.shape == (4, 3) and indices.shape == (4, 3)
    assert torch.allclose(weights.sum(-1), torch.ones(4))
    with pytest.raises(RuntimeError):
        router.sticky_selection(torch.randn(4, 3))
    with pytest.raises(RuntimeError):
        router.oracle_selection(torch.randn(4, 3), torch.ones(3, 3))
    router.reset_sticky_ema()


def test_ema_update_moves_only_present_regimes() -> None:
    router = _router()
    before = router.prototypes.data.clone()
    latents = torch.zeros(6, 8)
    latents[:3, 0] = 4.0
    router.ema_update(latents, torch.tensor([0, 0, 0, 1, 1, 1]), decay=0.0)
    assert torch.allclose(router.prototypes.data[0], torch.tensor([4.0] + [0.0] * 7))
    assert torch.allclose(router.prototypes.data[1], torch.zeros(8))
    assert torch.equal(router.prototypes.data[2], before[2])
    with pytest.raises(ValueError):
        router.ema_update(latents, torch.zeros(6, dtype=torch.long), decay=1.0)


def test_moe_top1_dispatch_equals_selected_expert() -> None:
    torch.manual_seed(1)
    router = PrototypeRouter(latent_dim=8, num_experts=3, top_k=1)
    experts = torch.nn.ModuleList(
        [ExpertMLP(input_dim=8, hidden_dims=[16], output_dim=7) for _ in range(3)]
    )
    layer = MoELayer(router=router, experts=experts)
    latent = torch.randn(8, 8)
    out = layer(latent)
    assert out.combined_output.shape == (8, 7)
    for row in range(8):
        expert_idx = int(out.expert_indices[row, 0].item())
        expected = experts[expert_idx](latent[row : row + 1])
        assert torch.allclose(out.combined_output[row : row + 1], expected)


def test_init_diagnostics_compatible_with_prototype_router() -> None:
    from phaseforge.models.phase_moe import PhaseBootstrappedMoE

    torch.manual_seed(0)
    encoder = StateEncoder(input_dim=6, hidden_dims=[16], latent_dim=8)
    head = ActionHead(input_dim=8, output_dim=4, hidden_dim=16)
    phase_head = PhaseClassificationHead(latent_dim=8, num_phases=2)
    router = PrototypeRouter(latent_dim=8, num_experts=2, top_k=1)
    expert = ExpertMLP(input_dim=8, hidden_dims=[16], output_dim=4)
    model = PhaseBootstrappedMoE(
        encoder=encoder, action_head=head, phase_head=phase_head,
        router=router, expert=expert, expert_init={"type": "random"},
    )
    gen = torch.Generator().manual_seed(0)
    rows = [
        {
            "state": torch.randn(6, generator=gen),
            "phase": torch.tensor(int(i % 2)),
        }
        for i in range(32)
    ]
    loader = DataLoader(rows, batch_size=8)
    model.bootstrap_moe(dataloader=loader, device="cpu")
    assert model.stage == 2
    diagnostics = compute_init_routing_diagnostics(model, loader, device="cpu")
    assert "t0_nmi_phase_top1" in diagnostics

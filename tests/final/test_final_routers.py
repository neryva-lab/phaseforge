"""Final router/loss gates (ledger TEST-07, TEST-08, ROUTER-05).

TEST-07: both router classes implement the registered multiclass margin
formula with finite gradients. ROUTER-05: prototype distance/logit forms
agree numerically. TEST-08: softmax exploration noise is training-only;
evaluation and margin logits are deterministic.
"""

from __future__ import annotations

import pytest
import torch

from phaseforge.models.components.prototype_router import PrototypeRouter
from phaseforge.models.components.router import TopKRouter


def test_common_margin_loss_matches_registered_formula() -> None:
    """TEST-07: both routers compute 1/|B| sum_{j!=y} max(0, m-(s_y-s_j))."""
    proto = PrototypeRouter(latent_dim=4, num_experts=3)
    topk = TopKRouter(latent_dim=4, num_experts=3)
    logits = torch.tensor([[0.0, -1.0, -2.0], [-3.0, 0.0, -0.1]])
    targets = torch.tensor([0, 1])
    expected = 0.2
    assert proto.margin_loss_from_logits(logits, targets, 0.5).item() == pytest.approx(expected)
    assert topk.margin_loss_from_logits(logits, targets, 0.5).item() == pytest.approx(expected)
    # Exact single-row case: l=[5.0, 4.8, 0.0], y=0, m=0.5 -> 0.3.
    one = torch.tensor([[5.0, 4.8, 0.0]])
    assert proto.margin_loss_from_logits(one, torch.tensor([0]), 0.5).item() == pytest.approx(0.3)
    assert topk.margin_loss_from_logits(one, torch.tensor([0]), 0.5).item() == pytest.approx(0.3)


def test_common_margin_loss_gradients_finite() -> None:
    """TEST-07: margin losses backpropagate finite gradients to the gates."""
    routers = (
        PrototypeRouter(latent_dim=4, num_experts=3),
        TopKRouter(latent_dim=4, num_experts=3),
    )
    for router in routers:
        gates = torch.randn(8, 3, requires_grad=True)
        loss = router.margin_loss_from_logits(gates, torch.randint(0, 3, (8,)), 0.5)
        assert torch.isfinite(loss).item()
        loss.backward()
        assert gates.grad is not None and torch.isfinite(gates.grad).all()


def test_common_margin_loss_rejects_bad_inputs() -> None:
    """TEST-07: both routers fail closed on range/width/empty/margin errors."""
    routers = (
        PrototypeRouter(latent_dim=4, num_experts=3),
        TopKRouter(latent_dim=4, num_experts=3),
    )
    for router in routers:
        with pytest.raises(ValueError, match="out of range"):
            router.margin_loss_from_logits(torch.zeros(1, 3), torch.tensor([7]), 0.5)
        with pytest.raises(ValueError):
            router.margin_loss_from_logits(torch.zeros(1, 4), torch.tensor([0]), 0.5)
        with pytest.raises(ValueError, match="margin"):
            router.margin_loss_from_logits(torch.zeros(1, 3), torch.tensor([0]), -1.0)
        empty = router.margin_loss_from_logits(
            torch.zeros(0, 3), torch.zeros(0, dtype=torch.long), 0.5
        )
        assert empty.item() == 0.0


def test_prototype_distance_logit_equivalence() -> None:
    """ROUTER-05: d_j-d_y and the logit form agree numerically."""
    torch.manual_seed(0)
    router = PrototypeRouter(latent_dim=4, num_experts=3)
    gates = torch.randn(16, 3)
    targets = torch.randint(0, 3, (16,))
    via_dists = router.margin_loss(-gates, targets, 0.5)
    via_logits = router.margin_loss_from_logits(gates, targets, 0.5)
    assert abs(via_dists.item() - via_logits.item()) < 1e-9


def test_softmax_noise_is_training_only() -> None:
    """TEST-08: noise perturbs training dispatches, never eval or clean."""
    torch.manual_seed(0)
    router = TopKRouter(latent_dim=4, num_experts=3, noise_std=0.5)
    latent = torch.randn(16, 4)
    router.train()
    first, second = router(latent), router(latent)
    assert not torch.allclose(first.gate_logits, second.gate_logits)
    assert torch.allclose(first.clean_gate_logits, second.clean_gate_logits)
    router.eval()
    out = router(latent)
    assert torch.allclose(out.gate_logits, out.clean_gate_logits)


def test_margin_on_clean_logits_is_stable_across_noisy_forwards() -> None:
    """TEST-08/ROUTER-04: margin targets do not inherit exploration noise."""
    torch.manual_seed(1)
    router = TopKRouter(latent_dim=4, num_experts=3, noise_std=0.5)
    router.train()
    latent = torch.randn(8, 4)
    targets = torch.randint(0, 3, (8,))
    margins = {
        router.margin_loss_from_logits(out.clean_gate_logits, targets, 0.5).item()
        for out in (router(latent), router(latent), router(latent))
    }
    assert len(margins) == 1

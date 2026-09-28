"""Dispatch-loop regression tests.

The MoE dispatch loop does not early-skip experts that received zero
tokens: converting a CUDA tensor to bool (``match_mask.any()``) is a
host-device synchronization, and an expert invoked on an empty ``(0, D)``
slice is an exact no-op. These tests prove the no-skip loop is bit-identical
to the skip-based loop — including the case where experts receive no
tokens at all — by replicating the skip-based loop verbatim as the reference.
"""

from __future__ import annotations

import torch

from phaseforge.models.components.expert import ExpertMLP
from phaseforge.models.components.moe_layer import MoELayer
from phaseforge.models.components.router import TopKRouter, _zero_scalar


def _build_layer(num_experts: int = 6, top_k: int = 2, latent_dim: int = 16) -> MoELayer:
    torch.manual_seed(7)
    router = TopKRouter(latent_dim, num_experts=num_experts, top_k=top_k, noise_std=0.1)
    layer = MoELayer(router=router, experts=ExpertMLP(latent_dim, [32], 3))
    layer.eval()  # eval: no routing noise -> deterministic gate
    return layer


def _reference_with_skip(layer: MoELayer, latent: torch.Tensor):
    """Verbatim replica of the skip-based loop, INCLUDING the `.any()` skip."""
    router_out = layer.router(latent)
    weights, indices = router_out.weights, router_out.indices
    out_dim = layer.experts[0].output_dim
    combined = torch.zeros((latent.size(0), out_dim), dtype=latent.dtype)
    for expert_idx, expert_net in enumerate(layer.experts):
        match_mask = indices == expert_idx
        if not match_mask.any():
            continue
        batch_idx, k_idx = torch.where(match_mask)
        combined.index_add_(
            0, batch_idx, expert_net(latent[batch_idx]) * weights[batch_idx, k_idx].unsqueeze(-1)
        )
    return combined, router_out


def test_dispatch_bit_identical_with_unselected_experts() -> None:
    """Gate biases force every token onto experts {0, 1}; experts 2-5 receive
    zero tokens. The no-skip loop must match the skip-based reference
    bit-for-bit (empty slices are exact no-ops)."""
    layer = _build_layer()
    with torch.no_grad():
        layer.router.gate_linear.weight.zero_()
        layer.router.gate_linear.bias.copy_(torch.tensor([10.0, 5.0, 0, 0, 0, 0.0]))

    torch.manual_seed(11)
    latent = torch.randn(64, 16)

    with torch.no_grad():
        out = layer(latent)
        ref, router_out = _reference_with_skip(layer, latent)

    # The forcing actually concentrates routing on two experts.
    used = torch.unique(router_out.indices)
    assert set(used.tolist()) == {0, 1}
    assert torch.equal(out.combined_output, ref)


def test_dispatch_bit_identical_with_all_experts_active() -> None:
    """Random init, full batch: typically every expert is selected; the
    loop output must still match the skip-based reference bitwise."""
    layer = _build_layer()
    torch.manual_seed(12)
    latent = torch.randn(256, 16)

    with torch.no_grad():
        out = layer(latent)
        ref, router_out = _reference_with_skip(layer, latent)

    used = torch.unique(router_out.indices)
    assert used.numel() == 6, "expected every expert active for a full random batch"
    assert torch.equal(out.combined_output, ref)


def test_router_sticky_default_is_cached_zero_scalar() -> None:
    """The sticky-loss default is a cached read-only 0.0 scalar matching the
    value and dtype of an inline ``torch.tensor(0.0)``."""
    zero = _zero_scalar(torch.device("cpu"))
    assert zero.item() == 0.0
    assert zero.dtype == torch.float32
    assert _zero_scalar(torch.device("cpu")) is zero

    layer = _build_layer(num_experts=4, top_k=1)
    layer.eval()
    with torch.no_grad():
        router_out = layer.router(torch.randn(8, 16))
    assert router_out.sticky_loss.item() == 0.0
    assert router_out.sticky_loss.dtype == torch.float32

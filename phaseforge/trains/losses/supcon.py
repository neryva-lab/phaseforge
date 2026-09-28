"""Supervised contrastive (SupCon) loss for regime-aligned encoders.

Shapes the latent space so same-regime states cluster and different-regime
states separate, making prototype-based routing geometric rather than
fragile::

    L_SupCon = 1/|B| Σ_i -1/|P(i)| Σ_{p ∈ P(i)}
               log[ exp(z_i · z_p / τ) / Σ_{a ≠ i} exp(z_i · z_a / τ) ]

with L2-normalized latents ``z`` (see ``StateEncoder(normalize_output=True)``),
positives ``P(i)`` sharing sample ``i``'s regime, and temperature ``τ``.
"""

from __future__ import annotations

import torch
from torch import Tensor


def supcon_loss(latents: Tensor, labels: Tensor, temperature: float = 0.07) -> Tensor:
    """Supervised contrastive loss over one batch.

    Args:
        latents: L2-normalized latents of shape ``(N, Dz)``.
        labels: Integer regime labels of shape ``(N,)``.
        temperature: Positive softmax temperature ``τ``.

    Returns:
        Scalar loss tensor (gradients flow to ``latents``). Samples with
        no same-regime partner contribute zero; an all-singleton batch
        returns exactly ``0.0`` (never NaN).
    """
    tau = float(temperature)
    if tau <= 0.0:
        raise ValueError(f"temperature must be > 0.0, got {tau}.")
    if latents.ndim != 2:
        raise ValueError(f"Expected latents shape (N, Dz), got {tuple(latents.shape)}.")
    targets = labels.reshape(-1).long()
    if targets.numel() != latents.size(0):
        raise ValueError("latents and labels must have equal batch size.")
    count = latents.size(0)
    if count < 2:
        return torch.zeros((), device=latents.device, dtype=latents.dtype)

    logits = (latents @ latents.T) / tau
    logits = logits - logits.max(dim=-1, keepdim=True).values.detach()
    self_mask = torch.eye(count, dtype=torch.bool, device=latents.device)
    logits = logits.masked_fill(self_mask, float("-inf"))
    log_denom = torch.logsumexp(logits, dim=-1)

    pos_mask = targets.unsqueeze(1) == targets.unsqueeze(0)
    pos_mask = pos_mask & ~self_mask
    pos_counts = pos_mask.sum(dim=-1)
    log_prob = logits - log_denom.unsqueeze(-1)
    # The self entries are -inf (masked above); zero them *after* the
    # log-softmax so the `log_prob * pos_mask` product below is 0 there
    # instead of `-inf * 0 = NaN`.
    log_prob = log_prob.masked_fill(self_mask, 0.0)
    per_sample = -(log_prob * pos_mask.float()).sum(dim=-1) / pos_counts.clamp(min=1)
    valid_weight = (pos_counts > 0).to(latents.dtype)
    valid_count = valid_weight.sum()
    # Fused zero-sync reduction: when valid_count == 0, numerator is 0 and
    # clamped denominator produces exactly 0.0 without host-device synchronization.
    return (per_sample * valid_weight).sum() / valid_count.clamp(min=1.0)


def pairwise_distances(latents: Tensor) -> Tensor:
    """Euclidean pairwise distances, shape ``(N, N)`` (diagnostic helper)."""
    return torch.cdist(latents.float(), latents.float())


__all__ = ["pairwise_distances", "supcon_loss"]

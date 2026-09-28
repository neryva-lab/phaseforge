"""Expert utilization and load balancing metrics.

Terminology: ``expert_utilization`` counts the
**top-k routing assignments** (every selected expert index contributes
one assignment), whereas ``phase_expert_nmi`` in :mod:`phase_alignment`
uses only the **top-1** assignment. Both are valid; the distinction is
deliberate and documented.
"""

from __future__ import annotations

import torch
from torch import Tensor


def expert_utilization(expert_indices: Tensor, num_experts: int) -> Tensor:
    """Compute the fraction of items routed to each expert.

    Counts every **top-k** assignment in ``expert_indices`` (see module
    docstring for the top-k vs top-1 distinction).

    Args:
        expert_indices: Tensor of shape (B, K) containing chosen expert indices.
        num_experts: Total number of experts (E).

    Returns:
        Tensor of shape (E,) representing the fraction [0, 1] of routing assignments.

    Raises:
        ValueError: If ``num_experts < 1``, ``expert_indices`` is empty,
            or any index is outside ``[0, num_experts)``.
    """
    if int(num_experts) < 1:
        raise ValueError(f"num_experts must be >= 1, got {num_experts}")
    if expert_indices.numel() == 0:
        raise ValueError("expert_indices must not be empty")
    if expert_indices.is_floating_point():
        raise ValueError("expert_indices must be integer indices")
    flat = expert_indices.view(-1)
    if flat.min() < 0 or flat.max() >= num_experts:
        raise ValueError(
            f"expert_indices outside [0, {num_experts}): "
            f"min={flat.min().item()}, max={flat.max().item()}"
        )

    indices_flat = expert_indices.view(-1)

    # Bincount to get absolute usage, minlength ensures output shape is (E,)
    counts = torch.bincount(indices_flat, minlength=num_experts).float()

    total_assignments = max(1, len(indices_flat))
    fractions = counts / total_assignments

    return fractions


def expert_utilization_top1(expert_indices: Tensor, num_experts: int) -> Tensor:
    """Compute utilization from only the top-1 assignment per item.

    ``expert_utilization`` intentionally counts every selected top-k expert.
    This companion metric is needed when comparing utilization with
    top-1-only diagnostics such as phase/expert NMI. For a 1-D index tensor,
    each element is already treated as one item's top-1 assignment.
    """
    if expert_indices.ndim == 0:
        raise ValueError("expert_indices must have at least one dimension")
    top1 = expert_indices if expert_indices.ndim == 1 else expert_indices[..., 0]
    return expert_utilization(top1, num_experts)


def expert_utilization_balance(fractions: Tensor) -> float:
    """Compute the balance score (normalized entropy) of expert usage.

    Score of 1.0 means perfectly uniform usage across all E experts.
    Score of 0.0 means complete collapse (all items routed to 1 expert).

    Args:
        fractions: Tensor of shape (E,) summing to 1.0.

    Returns:
        Float score in [0, 1].

    Raises:
        ValueError: If ``fractions`` is empty or contains negative values.
    """
    E = fractions.size(0)
    if E <= 1:
        return 1.0
    if fractions.numel() == 0:
        raise ValueError("fractions must not be empty")
    if (fractions < 0).any():
        raise ValueError("fractions must be non-negative")
    if not torch.isfinite(fractions).all():
        raise ValueError("fractions contains non-finite values")
    if float(fractions.sum()) <= 0.0:
        raise ValueError("fractions must sum to a positive value")

    # Clamp for numerical stability
    probs = fractions.clamp(min=1e-8)

    # Entropy: -sum(p * log(p))
    entropy = -torch.sum(probs * torch.log(probs))

    # Normalize by log(E)
    denom = torch.log(torch.tensor(E, dtype=torch.float32, device=fractions.device))
    normalized_entropy = entropy / denom

    return normalized_entropy.item()


def collapse_rate(fractions: Tensor, threshold_factor: float = 5.0) -> float:
    """Calculate the percentage of "collapsed" (unused or rarely used) experts.

    An expert is considered collapsed if its usage fraction is less than
    1/(threshold_factor * E).

    Args:
        fractions: Tensor of shape (E,) summing to 1.0.
        threshold_factor: Factor controlling strictness of collapse definition.

    Returns:
        Float rate in [0, 1]. E.g., if 3 out of 6 experts are collapsed, returns 0.5.

    Raises:
        ValueError: If ``threshold_factor <= 0``, ``fractions`` is empty
            or contains negative/non-finite values.
    """
    if float(threshold_factor) <= 0.0:
        raise ValueError(f"threshold_factor must be > 0, got {threshold_factor}")
    E = fractions.size(0)
    if E == 0:
        return 0.0
    if (fractions < 0).any() or not torch.isfinite(fractions).all():
        raise ValueError("fractions must be finite and non-negative")

    threshold = 1.0 / (threshold_factor * E)
    collapsed_count = (fractions < threshold).sum().item()

    return collapsed_count / E

"""Phase-Expert alignment metrics.

Terminology: the NMI score and the contingency
matrix use only the **top-1** routing assignment per item
(``expert_indices[..., 0]``), whereas :func:`expert_utilization.expert_utilization`
counts all **top-k** assignments. Both choices are deliberate.
"""

from __future__ import annotations

import torch
from sklearn.metrics import normalized_mutual_info_score
from torch import Tensor


def _validate_labels(phases: Tensor, expert_indices: Tensor) -> tuple[Tensor, Tensor]:
    """Common input checks for the phase-expert metrics.

    Raises:
        ValueError: On empty inputs, shape mismatch after top-1 selection,
            or negative/non-finite label values.
    """
    if phases.numel() == 0 or expert_indices.numel() == 0:
        raise ValueError("phases and expert_indices must not be empty")
    p_flat = phases.view(-1)
    if expert_indices.ndim == phases.ndim + 1:
        e_top1 = expert_indices[..., 0].view(-1)
    else:
        e_top1 = expert_indices.view(-1)
    if p_flat.numel() != e_top1.numel():
        raise ValueError(
            f"Shape mismatch: phases has {p_flat.numel()} elements, experts has {e_top1.numel()}"
        )
    if p_flat.min() < 0 or e_top1.min() < 0:
        raise ValueError(
            "phases and expert_indices must contain non-negative labels; "
            f"got min phase={p_flat.min().item()}, min expert={e_top1.min().item()}"
        )
    return p_flat, e_top1


def phase_expert_nmi(phases: Tensor, expert_indices: Tensor) -> float:
    """Calculate Normalized Mutual Information (NMI) between phases and experts.

    NMI measures the alignment between the assigned phase labels and the
    router's expert choices.
    1.0 = Perfect one-to-one mapping (experts specialize perfectly by phase).
    0.0 = Completely independent (router ignores phase structure).

    Uses only the **top-1** expert assignment per item (module docstring).

    Args:
        phases: Tensor of shape (B,) or (B, T) containing integer phase labels.
        expert_indices: Tensor of shape (B, K) or (B, T, K) containing top-K experts.
            Only the top-1 expert (K=0) is used for NMI calculation.

    Returns:
        Float NMI score in [0, 1].

    Raises:
        ValueError: On empty inputs, shape mismatch, or negative labels.
    """
    p_flat, e_top1 = _validate_labels(phases, expert_indices)
    p_np = p_flat.cpu().numpy()
    e_np = e_top1.cpu().numpy()

    # We use 'arithmetic' average method as standard
    return float(normalized_mutual_info_score(p_np, e_np, average_method="arithmetic"))


def build_contingency_matrix(
    phases: Tensor,
    expert_indices: Tensor,
    num_phases: int,
    num_experts: int,
) -> Tensor:
    """Build a P x E contingency matrix (heatmap) of phase-expert assignments.

    Args:
        phases: Tensor of shape (B,) containing integer phase labels.
        expert_indices: Tensor of shape (B, K) containing top-K experts. Uses top-1.
        num_phases: Total number of phases (P).
        num_experts: Total number of experts (E).

    Returns:
        Tensor of shape (num_phases, num_experts) containing normalized counts.
        Each row (phase) sums to 1.0.

    Raises:
        ValueError: If ``num_phases``/``num_experts`` are < 1, inputs are
            empty, shapes mismatch, or any label is out of range (invalid
            labels are NEVER silently discarded).
    """
    if int(num_phases) < 1:
        raise ValueError(f"num_phases must be >= 1, got {num_phases}")
    if int(num_experts) < 1:
        raise ValueError(f"num_experts must be >= 1, got {num_experts}")
    p_flat, e_top1 = _validate_labels(phases, expert_indices)
    if p_flat.max() >= num_phases:
        raise ValueError(f"phase label {p_flat.max().item()} out of range [0, {num_phases})")
    if e_top1.max() >= num_experts:
        raise ValueError(f"expert index {e_top1.max().item()} out of range [0, {num_experts})")

    matrix = torch.zeros((num_phases, num_experts), dtype=torch.float32, device=phases.device)

    for p, e in zip(p_flat, e_top1):
        matrix[p, e] += 1.0

    row_sums = matrix.sum(dim=1, keepdim=True)

    # Avoid division by zero
    row_sums = torch.clamp(row_sums, min=1.0)

    return matrix / row_sums

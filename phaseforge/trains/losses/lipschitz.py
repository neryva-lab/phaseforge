"""Local contraction (Lipschitz) regularization.

Biases each impedance expert toward locally stabilizing behavior by
penalizing target maps that move faster than the task state::

    L_lip = E_{(i,j)} [ max(0, ‖T(x_i) − T(x_j)‖₂ / (‖y_i − y_j‖₂ + ε) − ρ)² ]

Pairs ``(i, j)`` share a predicted expert (same-regime pairs); the target
is ``‖∂T/∂y‖ ≤ ρ < 1``.

> The contraction term provides a local stabilizing bias and a useful
> diagnostic. It does not by itself certify global closed-loop stability
> in robosuite.
"""

from __future__ import annotations

import torch
from torch import Tensor


def lip_penalty(
    targets: Tensor,
    task_states: Tensor,
    expert_index: Tensor,
    trajectory_ids: Tensor | None = None,
    rho: float = 0.8,
    eps: float = 1e-6,
    num_pairs: int = 256,
    max_ratio: float = 10.0,
    min_delta_state: float = 1e-4,
    coord_slice: tuple[int, int] | None = None,
) -> Tensor:
    """Lipschitz penalty over deterministic within-expert pairs.

    Args:
        targets: Per-sample expert targets ``(B, Dy)``.
        task_states: Per-sample task states ``(B, Dy)``.
        expert_index: Per-sample predicted expert ``(B,)`` long.
        trajectory_ids: Optional per-sample trajectory index ``(B,)``. When provided,
            pairs are formed strictly within the same episode to avoid evaluating
            finite differences across diverging environmental contexts.
        rho: Contraction target (must satisfy ``0 <= rho < 1``).
        eps: Floor avoiding division by zero.
        num_pairs: Cap on evaluated pairs.
        max_ratio: Maximum clamped ratio to prevent gradient explosions.
        min_delta_state: Minimum task-state displacement to avoid singular division.
        coord_slice: Optional slice ``(start, end)`` of coordinates over which to
            evaluate contraction. Defaults to ``(0, 3)`` (Cartesian position)
            when ``Dy >= 8`` to avoid non-Euclidean quaternion double-cover and
            discrete gripper switching artifacts.
    """
    if float(rho) < 0.0 or float(rho) >= 1.0:
        raise ValueError(f"rho must satisfy 0 <= rho < 1, got {rho}.")
    if targets.ndim != 2 or task_states.ndim != 2:
        raise ValueError("targets and task_states must be (B, D) tensors.")
    if targets.shape != task_states.shape:
        raise ValueError("targets and task_states must share their shape.")
    flat_experts = expert_index.reshape(-1).long()
    if flat_experts.numel() != targets.size(0):
        raise ValueError("expert_index must have one entry per sample.")
    cap = max(0, int(num_pairs))
    if cap == 0:
        return torch.zeros((), device=targets.device, dtype=targets.dtype)

    if coord_slice is not None:
        s0, s1 = coord_slice
        eval_targets = targets[:, s0:s1]
        eval_states = task_states[:, s0:s1]
    elif targets.shape[-1] >= 8:
        eval_targets = targets[:, :3]
        eval_states = task_states[:, :3]
    else:
        eval_targets = targets
        eval_states = task_states

    flat_trajs: Tensor | None = None
    if trajectory_ids is not None:
        flat_trajs = trajectory_ids.reshape(-1).long()

    first_list: list[Tensor] = []
    second_list: list[Tensor] = []

    for regime in torch.unique(flat_experts).tolist():
        expert_mask = flat_experts == int(regime)
        if flat_trajs is not None:
            for tid in torch.unique(flat_trajs[expert_mask]).tolist():
                positions = (expert_mask & (flat_trajs == int(tid))).nonzero().flatten()
                if positions.numel() >= 2:
                    k = (positions.numel() // 2) * 2
                    first_list.append(positions[:k:2])
                    second_list.append(positions[1:k:2])
        else:
            positions = expert_mask.nonzero().flatten()
            if positions.numel() >= 2:
                k = (positions.numel() // 2) * 2
                first_list.append(positions[:k:2])
                second_list.append(positions[1:k:2])

    if not first_list:
        return torch.zeros((), device=targets.device, dtype=targets.dtype)

    first_idx = torch.cat(first_list)
    second_idx = torch.cat(second_list)

    # Batched vector difference & norms across all candidate pairs
    delta_state = torch.linalg.vector_norm(eval_states[first_idx] - eval_states[second_idx], dim=-1)
    valid_mask = delta_state >= float(min_delta_state)

    if not bool(valid_mask.any()):
        return torch.zeros((), device=targets.device, dtype=targets.dtype)

    first_valid = first_idx[valid_mask][:cap]
    second_valid = second_idx[valid_mask][:cap]
    valid_delta_state = delta_state[valid_mask][:cap]

    delta_target = torch.linalg.vector_norm(
        eval_targets[first_valid] - eval_targets[second_valid], dim=-1
    )
    ratios = torch.clamp(delta_target / (valid_delta_state + float(eps)), max=float(max_ratio))

    if not torch.isfinite(ratios).all():
        return torch.zeros((), device=targets.device, dtype=targets.dtype)

    excess = torch.clamp(ratios - float(rho), min=0.0)
    return (excess * excess).mean()


def gain_penalty(
    gains: Tensor,
    kappa_nominal: float = 1.0,
) -> Tensor:
    """Gain regularization ``E‖κ − κ_nom‖²`` against pathological stiffness."""
    nominal = float(kappa_nominal)
    if nominal <= 0.0:
        raise ValueError(f"kappa_nominal must be > 0.0, got {nominal}.")
    if not torch.isfinite(gains).all():
        raise ValueError("Non-finite gains in gain regularization.")
    return ((gains - nominal) ** 2).mean()


__all__ = ["gain_penalty", "lip_penalty"]

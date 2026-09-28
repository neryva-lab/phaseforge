"""Impedance action adapter: (target, gains) -> environment action.

Each expert predicts a target task state ``T_k`` and positive gains
``κ_k``; with the current task state ``y_t``::

    e_k  = TaskError(T_k, y_t) = [p* − p, RotErr(R*, R), g* − g]   (De = 7)
    u_k  = κ_k ⊙ e_k
    a_t  = tanh(u_k / s)                                           (A = 7)

preserving the ``[-1, 1]`` robosuite action contract while changing the
expert output from "absolute action" to "local feedback command".

Orientation: quaternions are never subtracted. The rotation error is the
quaternion error mapped to a 3D tangent (axis-angle) vector, with the
double cover fixed (``w ≥ 0``) and a small-angle guard.

Top-2 blending (ablation only) combines *controller parameters*, not raw
actions::

    K_eff = Σ_i w_i K_i        T_eff = K_eff⁻¹ Σ_i w_i K_i T_i
    u     = K_eff (T_eff − y_t)

Position/gripper targets blend linearly with stiffness weights; the
quaternion target blends by sign-aligned normalized lerp (documented
approximation — exact vector-space blending does not exist for
quaternions). The primary top-1 path is exact.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from phaseforge.models.components.task_state import TASK_ERROR_DIM


def quat_conjugate(quat: Tensor) -> Tensor:
    """Conjugate of ``(..., 4)`` quaternions in ``(w, x, y, z)`` order."""
    out = quat.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def quat_multiply(left: Tensor, right: Tensor) -> Tensor:
    """Hamilton product of ``(..., 4)`` quaternions in ``(w, x, y, z)`` order."""
    w1, v1 = left[..., :1], left[..., 1:]
    w2, v2 = right[..., :1], right[..., 1:]
    w = w1 * w2 - (v1 * v2).sum(dim=-1, keepdim=True)
    v = w1 * v2 + w2 * v1 + torch.linalg.cross(v1, v2, dim=-1)
    return torch.cat([w, v], dim=-1)


def quat_log_map(quat: Tensor) -> Tensor:
    """Map unit quaternions to axis-angle tangent vectors ``(..., 3)``.

    Identity maps to exactly zero; the 180° case stays finite via the
    small-angle guard (any unit axis is valid there up to sign).
    """
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    sign = torch.where(quat[..., 0:1] < 0.0, -1.0, 1.0)
    quat = quat * sign
    w = quat[..., 0:1].clamp(-1.0, 1.0)
    xyz = quat[..., 1:]
    sin_half = torch.sqrt((1.0 - w * w).clamp(min=0.0))
    # rotvec = axis * angle with axis = xyz/sin_half, angle = 2*atan2(sin_half, w).
    # As sin_half -> 0 the ratio atan2(sin_half, w)/sin_half -> 1/w ~= 1.
    scale = torch.where(
        sin_half > 1e-6, 2.0 * torch.atan2(sin_half, w) / sin_half, torch.full_like(w, 2.0)
    )
    return xyz * scale


def rotation_error(target_quat: Tensor, current_quat: Tensor) -> Tensor:
    """3D rotation error taking ``current`` toward ``target``."""
    if target_quat.shape[-1] != 4 or current_quat.shape[-1] != 4:
        raise ValueError("Quaternion inputs must have width 4.")
    t_norm = target_quat / target_quat.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    c_norm = current_quat / current_quat.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    error_quat = quat_multiply(t_norm, quat_conjugate(c_norm))
    return quat_log_map(error_quat)


def task_error(target: Tensor, task_state: Tensor) -> Tensor:
    """Task error ``e = TaskError(T, y)`` of shape ``(..., 7)``.

    Layouts: ``T``/``y`` are ``(..., 8)`` = [pos 3, quat 4, gripper 1];
    ``e`` is ``(..., 7)`` = [pos_err 3, rotvec_err 3, gripper_err 1].
    """
    if target.shape[-1] != 8 or task_state.shape[-1] != 8:
        raise ValueError(
            f"Target/task-state width must be 8, got {target.shape[-1]} and "
            f"{task_state.shape[-1]}."
        )
    pos_err = target[..., 0:3] - task_state[..., 0:3]
    rot_err = rotation_error(target[..., 3:7], task_state[..., 3:7])
    grip_err = target[..., 7:8] - task_state[..., 7:8]
    return torch.cat([pos_err, rot_err, grip_err], dim=-1)


def impedance_action(
    target: Tensor,
    gains: Tensor,
    task_state: Tensor,
    scale: float | Sequence[float] | Tensor = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Map one expert's ``(T, κ)`` and ``y`` to a clipped action.

    Returns ``(action (..., 7), {"task_error": e, "pre_clip_u": u})`` with
    ``u = κ ⊙ e`` and ``a = tanh(u / s)``. Scale ``s`` may be a positive
    scalar or a per-dimension positive scale vector (e.g. matching the OSC
    controller's physical command bounds). Gains must be finite;
    non-positive gains fail closed (they would flip the feedback sign).
    """
    if isinstance(scale, (int, float)):
        if float(scale) <= 0.0:
            raise ValueError(f"Action scale must be > 0.0, got {scale}.")
        scale_factor: float | Tensor = float(scale)
    else:
        scale_tensor = torch.as_tensor(scale, dtype=target.dtype, device=target.device)
        if bool((scale_tensor <= 0.0).any()):
            raise ValueError(f"Action scale elements must be strictly positive, got {scale}.")
        scale_factor = scale_tensor

    error = task_error(target, task_state)
    if gains.shape != error.shape:
        raise ValueError(
            f"Gains shape {tuple(gains.shape)} must match task-error shape "
            f"{tuple(error.shape)}."
        )
    if not torch.isfinite(gains).all():
        raise ValueError("Non-finite impedance gains.")
    if bool((gains <= 0.0).any()):
        raise ValueError("Impedance gains must be strictly positive.")
    command = gains * error
    action = torch.tanh(command / scale_factor)
    return action, {"task_error": error, "pre_clip_u": command}


def blend_impedance(
    targets: Tensor,
    gains: Tensor,
    weights: Tensor,
) -> tuple[Tensor, Tensor]:
    """Blend per-expert ``(T, κ)`` into effective ``(T_eff, K_eff)``.

    Args:
        targets: ``(B, K, 8)`` expert target task states.
        gains: ``(B, K, 7)`` expert gains (diagonal stiffnesses).
        weights: ``(B, K)`` routing weights (renormalized defensively).

    Position/gripper targets blend with stiffness weights
    ``α ∝ w·κ``; quaternions blend by sign-aligned normalized lerp with
    rotation-averaged weights; gains blend linearly.
    """
    if targets.ndim != 3 or gains.ndim != 3 or weights.ndim != 2:
        raise ValueError("Expected targets (B, K, 8), gains (B, K, 7), weights (B, K).")
    if targets.size(-1) != 8 or gains.size(-1) != TASK_ERROR_DIM:
        raise ValueError("Impedance blend expects Dy=8 targets and De=7 gains.")
    if targets.size(1) != gains.size(1) or targets.size(1) != weights.size(1):
        raise ValueError("Expert count K must agree across targets/gains/weights.")
    denom = weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    if bool((weights.sum(dim=-1) <= 0.0).any()):
        raise ValueError("Routing weights must sum to a positive value per sample.")
    normed = weights / denom

    pos_w = normed.unsqueeze(-1) * gains[..., 0:3]
    pos_norm = pos_w.sum(dim=1, keepdim=True).clamp(min=1e-12)
    pos_eff = (pos_w * targets[..., 0:3]).sum(dim=1, keepdim=True) / pos_norm
    grip_w = (normed * gains[..., 6:7].squeeze(-1)).unsqueeze(-1)
    grip_norm = grip_w.sum(dim=1, keepdim=True).clamp(min=1e-12)
    grip_eff = (grip_w * targets[..., 7:8]).sum(dim=1, keepdim=True) / grip_norm

    quats = targets[..., 3:7]
    ref = quats[:, 0:1, :]
    aligned = torch.where(
        (quats * ref).sum(dim=-1, keepdim=True) < 0.0, -quats, quats
    )
    rot_w = normed * gains[..., 3:6].mean(dim=-1)
    quat_eff = (aligned * rot_w.unsqueeze(-1)).sum(dim=1)
    quat_eff = quat_eff / quat_eff.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    target_eff = torch.cat(
        [pos_eff.squeeze(1), quat_eff, grip_eff.squeeze(1)], dim=-1
    )
    gains_eff = (normed.unsqueeze(-1) * gains).sum(dim=1)
    return target_eff, gains_eff


__all__ = [
    "blend_impedance",
    "impedance_action",
    "quat_conjugate",
    "quat_log_map",
    "quat_multiply",
    "rotation_error",
    "task_error",
]

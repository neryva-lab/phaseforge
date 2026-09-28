"""Task-space state extraction for impedance-parameterized experts.

Defines the controllable/task-relevant part of the state::

    y_t = ψ(x_t) = [eef_pos (3), eef_quat (4), gripper_aperture (1)]   (Dy = 8)

Object information stays available to the encoder and experts through the
full state ``x_t``; only the feedback error is computed in task space.

Single-arm layout convention (slices match the phase-labeler defaults and
the ``_LIFT_KEYS`` registry order — ``robot0_eef_pos`` 0:3,
``robot0_eef_quat`` 3:7, ``robot0_gripper_qpos`` 7:9; the object block is
everything from index 9 on). Accepted widths are the single-arm
benchmarks (19/23/53); anything else (e.g. two-arm Transport) fails
closed.

A note on coordinates: rollout and training both feed the policy
*normalized* states (see ``RolloutEvaluator._policy_action`` and the
ingestion normalizer), so ``ψ`` operates in normalized task space and the
expert targets live there too. Raw states are unavailable at inference,
and normalized-space feedback is self-consistent: the adapter maps the
feedback command back into the ``[-1, 1]`` action contract.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

#: Task-state width: [pos 3, quat 4, aperture 1].
TASK_STATE_DIM = 8

#: Task-error width: [pos_err 3, rotvec_err 3, gripper_err 1] == action dim.
TASK_ERROR_DIM = 7

#: Action width this parameterization serves (single-arm Panda OSC_POSE).
ACTION_DIM = 7

#: Accepted flat state widths (Lift 19 / Can+Square 23 / ToolHang 53).
_SINGLE_ARM_WIDTHS: tuple[int, ...] = (19, 23, 53)

_EEF_SLICE: tuple[int, int] = (0, 3)
_QUAT_SLICE: tuple[int, int] = (3, 7)
_GRIP_SLICE: tuple[int, int] = (7, 9)


def _check_width(width: int) -> None:
    if width not in _SINGLE_ARM_WIDTHS:
        raise ValueError(
            f"Impedance task state needs a single-arm layout {list(_SINGLE_ARM_WIDTHS)}, "
            f"got width {width}."
        )


def gripper_aperture(gripper_qpos: Tensor) -> Tensor:
    """Finger excursion magnitude ``max(|q0|, |q1|)``, shape ``(..., 1)``."""
    return gripper_qpos.abs().amax(dim=-1, keepdim=True)


def extract_task_state(
    state: Tensor,
    mean: Tensor | None = None,
    std: Tensor | None = None,
) -> Tensor:
    """Extract the task state ``y = ψ(x)`` of shape ``(..., 8)`` in physical units.

    If ``mean`` and ``std`` are provided, ``state`` is first denormalized back
    to physical space (meters, unit quaternions, joint positions) so that
    Cartesian displacement and $SO(3)$ rotation errors are physically meaningful.
    """
    if state.shape[-1] not in _SINGLE_ARM_WIDTHS:
        _check_width(int(state.shape[-1]))
    if mean is not None and std is not None:
        if isinstance(mean, np.ndarray):
            mean = torch.from_numpy(mean)
        if isinstance(std, np.ndarray):
            std = torch.from_numpy(std)
        raw_state = (
            state * std.to(device=state.device, dtype=state.dtype)
            + mean.to(device=state.device, dtype=state.dtype)
        )
    else:
        raw_state = state
    eef_pos = raw_state[..., _EEF_SLICE[0] : _EEF_SLICE[1]]
    raw_quat = raw_state[..., _QUAT_SLICE[0] : _QUAT_SLICE[1]]
    quat_norm = raw_quat.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    quat = raw_quat / quat_norm
    sign = torch.where(quat[..., 0:1] < 0.0, -1.0, 1.0)
    quat = quat * sign
    aperture = gripper_aperture(raw_state[..., _GRIP_SLICE[0] : _GRIP_SLICE[1]])
    return torch.cat([eef_pos, quat, aperture], dim=-1)


def extract_task_state_numpy(
    state: np.ndarray,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> np.ndarray:
    """Numpy twin of :func:`extract_task_state` (offline diagnostics)."""
    arr = np.asarray(state, dtype=np.float64)
    if arr.shape[-1] not in _SINGLE_ARM_WIDTHS:
        _check_width(int(arr.shape[-1]))
    if mean is not None and std is not None:
        raw_arr = arr * np.asarray(std, dtype=np.float64) + np.asarray(mean, dtype=np.float64)
    else:
        raw_arr = arr
    eef_pos = raw_arr[..., 0:3]
    raw_quat = raw_arr[..., 3:7]
    norm = np.linalg.norm(raw_quat, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    quat = raw_quat / norm
    sign = np.where(quat[..., 0:1] < 0.0, -1.0, 1.0)
    quat = quat * sign
    aperture = np.max(np.abs(raw_arr[..., 7:9]), axis=-1, keepdims=True)
    return np.concatenate([eef_pos, quat, aperture], axis=-1)


__all__ = [
    "ACTION_DIM",
    "TASK_ERROR_DIM",
    "TASK_STATE_DIM",
    "extract_task_state",
    "extract_task_state_numpy",
    "gripper_aperture",
]

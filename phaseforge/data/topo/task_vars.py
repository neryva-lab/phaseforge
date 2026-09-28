"""Task-space variable extraction for topological regime discovery.

Derives candidate task-space variables from the instantaneous
low-dimensional state using only the configured ``data.state_keys`` layout
(resolved via cumulative dims, never hardcoded slices):

* ``eef_pos`` (3): end-effector position.
* ``eef_quat`` (4): end-effector orientation quaternion.
* ``gripper`` (2): raw finger positions.
* ``object`` (Do): object observation block (task-dependent width).
* ``rel_ee_obj`` (3): ``eef_pos - object[0:3]``. The object block's leading
  three dims are used as the object-position proxy and documented per run
  in the topo manifest (see :mod:`phaseforge.data.topo.artifacts`); no
  future-transition information enters.
* ``gripper_aperture`` (1): ``max(|q0|, |q1|)`` finger excursion magnitude,
  mirroring :class:`RuleBasedPhaseLabeler` (the finger mean is ~constant
  for parallel-jaw grippers and cannot separate open from closed).

All outputs are in the input's own space (the discovery pipeline runs on
train-normalized states, so the variables inherit train z-scoring). Pure
numpy, CPU-only, deterministic.
"""

from __future__ import annotations

import numpy as np

#: Canonical output order (also the PELT input column order).
TASK_VAR_ORDER: tuple[str, ...] = (
    "eef_pos",
    "eef_quat",
    "gripper",
    "object",
    "rel_ee_obj",
    "gripper_aperture",
)

_REQUIRED_KEYS: tuple[str, ...] = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")


def _offsets(state_keys: list[str], state_dims: list[int]) -> dict[str, tuple[int, int]]:
    """Map each state key to its ``(start, stop)`` slice of the flat vector."""
    if len(state_keys) != len(state_dims):
        raise ValueError(
            f"state_keys ({len(state_keys)}) and state_dims ({len(state_dims)}) "
            "must have equal length."
        )
    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for key, dim in zip(state_keys, state_dims):
        dim_int = int(dim)
        if dim_int <= 0:
            raise ValueError(f"state dim for {key!r} must be positive, got {dim_int}.")
        offsets[str(key)] = (cursor, cursor + dim_int)
        cursor += dim_int
    return offsets


def extract_task_vars(
    state: np.ndarray,
    state_keys: list[str],
    state_dims: list[int],
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Extract topological task variables from instantaneous states in physical coordinates.

    If ``mean`` and ``std`` are provided, ``state`` is first denormalized back to
    physical units (meters, canonical $S^3$ quaternions, joint angles).

    Args:
        state: Array of shape ``(T, S)`` (one trajectory) or ``(S,)``.
        state_keys: Ordered observation keys (``data.state_keys``).
        state_dims: Per-key dimensions (``data.state_keys`` dims).
        mean: Optional normalizer mean for denormalization to physical units.
        std: Optional normalizer std for denormalization to physical units.

    Returns:
        Dict with one ``(T, D)`` (or ``(D,)``) float64 array per name in
        :data:`TASK_VAR_ORDER`, plus ``var_info`` metadata describing the
        object-position proxy slice used.
    """
    arr = np.asarray(state, dtype=np.float64)
    if arr.ndim not in (1, 2):
        raise ValueError(f"Expected state shape (S,) or (T, S), got {arr.shape}.")
    single = arr.ndim == 1
    if single:
        arr = arr[np.newaxis, :]
    if mean is not None and std is not None:
        arr = arr * np.asarray(std, dtype=np.float64) + np.asarray(mean, dtype=np.float64)
    offsets = _offsets(list(state_keys), list(state_dims))
    total = sum(int(d) for d in state_dims)
    if arr.shape[-1] != total:
        raise ValueError(
            f"State width {arr.shape[-1]} does not match the declared layout "
            f"sum {total} (keys={list(state_keys)})."
        )
    for key in _REQUIRED_KEYS:
        if key not in offsets:
            raise ValueError(
                f"State layout is missing required key {key!r} "
                f"(available: {sorted(offsets)})."
            )
    if "object" not in offsets:
        raise ValueError(
            f"State layout is missing the 'object' key (available: {sorted(offsets)})."
        )

    def _slice(key: str) -> np.ndarray:
        start, stop = offsets[key]
        return arr[..., start:stop]

    eef_pos = _slice("robot0_eef_pos")
    raw_quat = _slice("robot0_eef_quat")
    quat_norm = np.maximum(np.linalg.norm(raw_quat, axis=-1, keepdims=True), 1e-12)
    eef_quat = raw_quat / quat_norm
    sign = np.where(eef_quat[..., 0:1] < 0.0, -1.0, 1.0)
    eef_quat = eef_quat * sign
    gripper = _slice("robot0_gripper_qpos")
    obj = _slice("object")
    if eef_pos.shape[-1] != 3 or eef_quat.shape[-1] != 4 or gripper.shape[-1] != 2:
        raise ValueError(
            "Unexpected robot slice widths "
            f"(eef_pos {eef_pos.shape[-1]}, eef_quat {eef_quat.shape[-1]}, "
            f"gripper {gripper.shape[-1]}); expected (3, 4, 2)."
        )
    if obj.shape[-1] < 3:
        raise ValueError(
            f"Object block width {obj.shape[-1]} is too narrow for a position proxy."
        )
    rel_ee_obj = eef_pos - obj[..., 0:3]
    aperture = np.max(np.abs(gripper), axis=-1, keepdims=True)

    out: dict[str, np.ndarray] = {
        "eef_pos": np.ascontiguousarray(eef_pos),
        "eef_quat": np.ascontiguousarray(eef_quat),
        "gripper": np.ascontiguousarray(gripper),
        "object": np.ascontiguousarray(obj),
        "rel_ee_obj": np.ascontiguousarray(rel_ee_obj),
        "gripper_aperture": np.ascontiguousarray(aperture),
        "var_info": np.asarray(["object_pos_proxy=object[0:3]"], dtype=object),
    }
    for key in TASK_VAR_ORDER:
        if not np.isfinite(out[key]).all():
            raise ValueError(f"Non-finite values in task variable {key!r}.")
        if single:
            out[key] = np.ascontiguousarray(out[key][0])
    return out


def concat_task_matrix(vars_dict: dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate the canonical task variables into one ``(T, Ds)`` matrix."""
    parts = [np.asarray(vars_dict[key], dtype=np.float64) for key in TASK_VAR_ORDER]
    if any(part.ndim != 2 for part in parts):
        raise ValueError("concat_task_matrix expects per-variable (T, D) arrays.")
    lengths = {part.shape[0] for part in parts}
    if len(lengths) != 1:
        raise ValueError(f"Task variables have inconsistent lengths: {sorted(lengths)}.")
    return np.concatenate(parts, axis=-1)


__all__ = ["TASK_VAR_ORDER", "concat_task_matrix", "extract_task_vars"]

"""StateOnlyDataset: flat (state, action, phase, task_id) dataset."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def corrupt_phase_labels(
    phase: Tensor,
    corruption_rate: float,
    num_phases: int = 6,
    seed: int = 42,
    shuffle_control: bool = False,
) -> tuple[Tensor, Tensor]:
    """Corrupt phase labels with exact mathematical definition.

    For corruption rate p in [0, 1]:
    z'_i = (z_i + Uniform({1, ..., P-1})) mod P with probability p.
    This guarantees forced-different replacement without self-collisions.

    If shuffle_control is True (p=1.0 null control), permutes the phase labels
    across all steps while preserving marginal class counts.

    Args:
        phase: 1D Tensor of integer phase labels.
        corruption_rate: Fraction of labels to corrupt in [0.0, 1.0].
        num_phases: Total number of phase classes P.
        seed: Deterministic random seed.
        shuffle_control: If True, permutes phase labels across the array.

    Returns:
        (corrupted_phase, clean_phase) Tensors.
    """
    clean_phase = phase.clone()
    if corruption_rate <= 0.0:
        return clean_phase, clean_phase

    if num_phases < 2:
        return clean_phase, clean_phase

    corrupted_phase = clean_phase.clone()
    n = clean_phase.numel()

    rng = np.random.default_rng(seed)

    if shuffle_control or (corruption_rate >= 1.0 and shuffle_control):
        # Cross-sample phase permutation preserving class marginal counts
        perm_indices = rng.permutation(n)
        corrupted_phase = clean_phase[perm_indices]
        logger.info(f"Phase corruption: 100% permutation control across {n} steps (seed={seed}).")
        return corrupted_phase, clean_phase

    # Standard forced-different corruption
    u = rng.uniform(0.0, 1.0, size=n)
    corrupt_mask = u < corruption_rate

    if corrupt_mask.any():
        num_corrupted = int(corrupt_mask.sum())
        # Sample offset in [1, num_phases - 1] to guarantee z' != z
        offsets = rng.integers(1, num_phases, size=num_corrupted)
        orig_labels = clean_phase[corrupt_mask].cpu().numpy()
        new_labels = (orig_labels + offsets) % num_phases
        corrupted_phase[corrupt_mask] = torch.from_numpy(new_labels).to(dtype=clean_phase.dtype)
        logger.info(
            f"Phase corruption: corrupted {num_corrupted}/{n} labels "
            f"({num_corrupted / n:.2%}, target {corruption_rate:.2%}, seed={seed})."
        )

    return corrupted_phase, clean_phase


class StateOnlyDataset(Dataset):
    """Dataset of pre-normalized (state, action, phase, task_id) tuples.

    All trajectories are already normalized and converted to tensors
    before this class is constructed.

    Args:
        trajectories: List of trajectory dicts, each containing:
            - ``"state"``:   Tensor (T, state_dim)
            - ``"action"``:  Tensor (T, action_dim)
            - ``"phase"`` or configured ``phase_field``: Tensor (T,) int64
            - ``"task_id"``: int
        sequence_length: Number of consecutive timesteps per sample.
        stride: Step between consecutive samples within a trajectory.
        phase_corruption_rate: Fraction of phase labels to corrupt for Stage 1 training.
        phase_corruption_seed: RNG seed for deterministic corruption.
        num_phases: Total number of phase classes.
        phase_shuffle_control: Whether to run 100% phase permutation control.
        phase_field: Key in trajectory dicts to read phase/regime labels from (default 'phase').
    """

    def __init__(
        self,
        trajectories: list[dict[str, Any]],
        sequence_length: int = 1,
        stride: int = 1,
        phase_corruption_rate: float = 0.0,
        phase_corruption_seed: int = 42,
        num_phases: int = 6,
        phase_shuffle_control: bool = False,
        phase_field: str = "phase",
    ) -> None:
        super().__init__()
        if int(sequence_length) < 1:
            raise ValueError("sequence_length must be a positive integer")
        if int(stride) < 1:
            raise ValueError("stride must be a positive integer")
        if not 0.0 <= phase_corruption_rate <= 1.0:
            raise ValueError(
                f"phase_corruption_rate must be in [0, 1], got {phase_corruption_rate}"
            )

        self.sequence_length = int(sequence_length)
        self.stride = int(stride)
        self.phase_corruption_rate = float(phase_corruption_rate)
        self.phase_corruption_seed = int(phase_corruption_seed)
        self.num_phases = int(num_phases)
        self.phase_shuffle_control = bool(phase_shuffle_control)
        self.phase_field = str(phase_field)

        # Apply phase corruption if configured — strict on the requested field.
        # If `phase_field` is missing we fail closed instead of silently
        # corrupting a fallback field (which would mislabel dynamic vs rule).
        if self.phase_corruption_rate > 0.0:
            processed_trajectories = []
            for traj_idx, traj in enumerate(trajectories):
                traj_copy = dict(traj)
                if self.phase_field not in traj:
                    # Only allow implicit fallback when the requested field is
                    # the canonical "phase" (legacy trajectories).
                    if self.phase_field == "phase" and "phase" in traj:
                        clean_p = traj["phase"]
                    else:
                        raise KeyError(
                            f"Trajectory {traj_idx} missing required phase field "
                            f"{self.phase_field!r} (available: "
                            f"{sorted(k for k in traj.keys() if 'phase' in k.lower())}). "
                            "For PhaseForge 2.0 dynamic training ensure the cache was built with "
                            "data.dynamics.enabled=true and re-ingested if stale."
                        )
                else:
                    clean_p = traj[self.phase_field]
                corrupted_p, clean_p = corrupt_phase_labels(
                    clean_p,
                    corruption_rate=self.phase_corruption_rate,
                    num_phases=self.num_phases,
                    seed=self.phase_corruption_seed + traj_idx,
                    shuffle_control=self.phase_shuffle_control,
                )
                traj_copy[self.phase_field] = corrupted_p
                traj_copy["phase_gt_clean"] = clean_p
                processed_trajectories.append(traj_copy)
            self.trajectories = processed_trajectories
        else:
            self.trajectories = trajectories

        self._index_map = self._build_index_map()

    def _build_index_map(self) -> list[tuple[int, int]]:
        """Build (traj_idx, start_t) index pairs for all valid windows."""
        index_map = []
        for traj_idx, traj in enumerate(self.trajectories):
            T = traj["state"].shape[0]
            for start_t in range(0, T - self.sequence_length + 1, self.stride):
                index_map.append((traj_idx, start_t))
        return index_map

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        traj_idx, start_t = self._index_map[idx]
        traj = self.trajectories[traj_idx]
        end_t = start_t + self.sequence_length

        state = traj["state"][start_t:end_t]  # (seq_len, S)
        action = traj["action"][start_t:end_t]  # (seq_len, A)
        if self.phase_field not in traj:
            if self.phase_field == "phase" and "phase" in traj:
                raw_phase = traj["phase"]
            else:
                raise KeyError(
                    f"Trajectory {traj_idx} missing required phase field {self.phase_field!r} "
                    f"(available: {sorted(k for k in traj.keys() if 'phase' in k.lower())})"
                )
        else:
            raw_phase = traj[self.phase_field]
        phase = raw_phase[start_t:end_t]  # (seq_len,)
        phase_gt = traj.get("phase_gt_clean", raw_phase)[start_t:end_t]
        task_id = torch.tensor(traj["task_id"], dtype=torch.long)

        trajectory_id = torch.tensor(traj_idx, dtype=torch.long)
        trajectory_position = torch.tensor(start_t, dtype=torch.long)

        # Expose alternate regime labels for bootstrap selection.
        # When trajectories carry `phase_dynamic` / `phase_rule` (dynamics enabled)
        # we return them alongside the primary `phase` so `bootstrap_moe` can
        # select `prototype_source` without requiring a second data pass.
        # `phase_topo` is added when topo discovery is enabled. Per-step topo
        # confidence is intentionally not collated: segment-level confidence
        # lives in the topo artifact manifest, keeping this batch a long
        # tensor like the other regime labels.
        extras: dict[str, Tensor] = {}
        for extra_key in ("phase_dynamic", "phase_rule", "phase_topo"):
            extra_raw = traj.get(extra_key)
            if extra_raw is None:
                continue
            extra_slice = extra_raw[start_t:end_t]
            if self.sequence_length == 1:
                extra_slice = extra_slice.squeeze(0)
            extras[extra_key] = extra_slice

        if self.sequence_length == 1:
            out: dict[str, Tensor] = {
                "state": state.squeeze(0),  # (S,)
                "action": action.squeeze(0),  # (A,)
                "phase": phase.squeeze(0),  # scalar
                "phase_gt_clean": phase_gt.squeeze(0),  # scalar
                "task_id": task_id,
                "trajectory_id": trajectory_id,
                "trajectory_position": trajectory_position,
            }
            out.update(extras)
            return out
        out_multi: dict[str, Tensor] = {
            "state": state,
            "action": action,
            "phase": phase,
            "phase_gt_clean": phase_gt,
            "task_id": task_id,
            "trajectory_id": trajectory_id,
            "trajectory_position": trajectory_position,
        }
        out_multi.update(extras)
        return out_multi

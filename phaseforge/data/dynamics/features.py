"""Feature construction for dynamics-aware phase discovery.

Constructs action-conditioned transition features:
    phi_t = [x_t, a_t, Delta x_t], where Delta x_t = x_{t+1} - x_t

All inputs are assumed to be pre-normalized according to train-split statistics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TransitionBatch:
    """Container for aligned transition features across demonstrations.

    Attributes:
        x_t: State at time t, shape (N, state_dim).
        a_t: Action at time t, shape (N, action_dim).
        delta_x: Next state delta (x_{t+1} - x_t), shape (N, state_dim).
        x_next: Next state x_{t+1}, shape (N, state_dim).
        features: Concatenated [x_t, a_t, delta_x] vector, shape
            (N, 2*state_dim + action_dim).
        trajectory_indices: Trajectory index corresponding to each step, shape (N,).
        timesteps: Timestep within trajectory for each step, shape (N,).
        phase_semantic: Semantic/rule-based phase label if present, shape (N,).
    """

    x_t: Tensor
    a_t: Tensor
    delta_x: Tensor
    x_next: Tensor
    features: Tensor
    trajectory_indices: Tensor
    timesteps: Tensor
    phase_semantic: Tensor | None = None

    @property
    def num_samples(self) -> int:
        return self.x_t.size(0)

    @property
    def state_dim(self) -> int:
        return self.x_t.size(-1)

    @property
    def action_dim(self) -> int:
        return self.a_t.size(-1)

    @property
    def feature_dim(self) -> int:
        return self.features.size(-1)


def extract_trajectory_transitions(
    traj: dict[str, Any],
    traj_idx: int = 0,
) -> TransitionBatch:
    """Extract valid (t, t+1) transition tuples from a single trajectory dict.

    Args:
        traj: Trajectory dictionary with 'state', 'action', and optional 'phase'.
        traj_idx: Integer identifier for provenance tracking.

    Returns:
        TransitionBatch containing valid one-step transitions.
    """
    state = traj["state"]
    action = traj["action"]
    phase = traj.get("phase")

    if isinstance(state, np.ndarray):
        state = torch.from_numpy(state).float()
    if isinstance(action, np.ndarray):
        action = torch.from_numpy(action).float()
    if phase is not None and isinstance(phase, np.ndarray):
        phase = torch.from_numpy(phase).long()

    if state.ndim != 2:
        raise ValueError(f"Expected 2D state tensor (T, D), got shape {tuple(state.shape)}")
    if action.ndim != 2:
        raise ValueError(f"Expected 2D action tensor (T, A), got shape {tuple(action.shape)}")

    T = state.size(0)
    if T < 2:
        raise ValueError(f"Trajectory {traj_idx} has length {T} < 2; cannot form transition pairs.")

    if action.size(0) != T:
        raise ValueError(
            f"State length {T} does not match action length {action.size(0)} in traj {traj_idx}"
        )

    # Valid transition pairs are t = 0 ... T-2
    x_t = state[:-1]
    x_next = state[1:]
    a_t = action[:-1]
    delta_x = x_next - x_t

    # Check finite
    if (
        not torch.isfinite(x_t).all()
        or not torch.isfinite(a_t).all()
        or not torch.isfinite(delta_x).all()
    ):
        raise ValueError(f"Non-finite values detected in trajectory {traj_idx}")

    features = torch.cat([x_t, a_t, delta_x], dim=-1)
    traj_indices = torch.full((T - 1,), traj_idx, dtype=torch.long)
    timesteps = torch.arange(T - 1, dtype=torch.long)

    phase_semantic = phase[:-1] if phase is not None else None

    return TransitionBatch(
        x_t=x_t,
        a_t=a_t,
        delta_x=delta_x,
        x_next=x_next,
        features=features,
        trajectory_indices=traj_indices,
        timesteps=timesteps,
        phase_semantic=phase_semantic,
    )


def extract_dataset_transitions(
    trajectories: list[dict[str, Any]],
) -> TransitionBatch:
    """Extract and concatenate transition features across all trajectories in a dataset.

    Args:
        trajectories: List of trajectory dictionaries.

    Returns:
        Concatenated TransitionBatch.
    """
    if not trajectories:
        raise ValueError("Cannot extract transitions from an empty trajectory list.")

    batches = [
        extract_trajectory_transitions(traj, traj_idx=idx) for idx, traj in enumerate(trajectories)
    ]

    x_t_cat = torch.cat([b.x_t for b in batches], dim=0)
    a_t_cat = torch.cat([b.a_t for b in batches], dim=0)
    delta_x_cat = torch.cat([b.delta_x for b in batches], dim=0)
    x_next_cat = torch.cat([b.x_next for b in batches], dim=0)
    features_cat = torch.cat([b.features for b in batches], dim=0)
    traj_indices_cat = torch.cat([b.trajectory_indices for b in batches], dim=0)
    timesteps_cat = torch.cat([b.timesteps for b in batches], dim=0)

    has_phase = all(b.phase_semantic is not None for b in batches)
    phase_semantic_cat = (
        torch.cat([b.phase_semantic for b in batches], dim=0)  # type: ignore[misc]
        if has_phase
        else None
    )

    logger.debug(
        f"Extracted {x_t_cat.size(0)} total transition steps from {len(trajectories)} trajectories."
    )

    return TransitionBatch(
        x_t=x_t_cat,
        a_t=a_t_cat,
        delta_x=delta_x_cat,
        x_next=x_next_cat,
        features=features_cat,
        trajectory_indices=traj_indices_cat,
        timesteps=timesteps_cat,
        phase_semantic=phase_semantic_cat,
    )

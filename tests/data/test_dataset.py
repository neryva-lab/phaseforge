"""CPU-only tests for StateOnlyDataset sequence-length handling."""

from __future__ import annotations

import torch

from phaseforge.data.common.dataset import StateOnlyDataset


def _traj(T: int = 4, state_dim: int = 3, action_dim: int = 2) -> dict:
    return {
        "state": torch.randn(T, state_dim),
        "action": torch.randn(T, action_dim),
        "phase": torch.randint(0, 2, (T,)),
        "task_id": 0,
    }


def test_sequence_length_one_yields_flat_samples() -> None:
    dataset = StateOnlyDataset([_traj(T=4)], sequence_length=1)
    assert len(dataset) == 4
    sample = dataset[0]
    assert sample["state"].ndim == 1
    assert sample["action"].ndim == 1
    assert sample["phase"].ndim == 0


def test_sequence_length_returns_temporal_windows() -> None:
    dataset = StateOnlyDataset([_traj(T=5)], sequence_length=3, stride=2)
    assert len(dataset) == 2
    sample = dataset[1]
    assert sample["state"].shape == (3, 3)
    assert sample["action"].shape == (3, 2)
    assert sample["phase"].shape == (3,)
    assert int(sample["trajectory_position"]) == 2


def test_invalid_sequence_parameters_rejected() -> None:
    import pytest

    with pytest.raises(ValueError, match="positive"):
        StateOnlyDataset([_traj()], sequence_length=0)
    with pytest.raises(ValueError, match="positive"):
        StateOnlyDataset([_traj()], stride=0)

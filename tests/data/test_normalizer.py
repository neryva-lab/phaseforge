"""Normalizer behavior on ignored (mask) dims.

The object-slot occupancy mask dims must be excluded from z-score
statistics and frozen to identity (mean=0, std=1) — they are already
binary by construction.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from phaseforge.data.common.normalizer import RunningStatNormalizer


def _finalized(batches: list[np.ndarray], ignore: set[int] | None):
    norm = RunningStatNormalizer(ignore_dims=ignore)
    for b in batches:
        norm.update(b)
    return norm.finalize()


def test_ignored_dims_frozen_to_identity() -> None:
    """Ignored dims get mean=0/std=1; other dims match manual statistics."""
    # dim 2 is the "mask": always 1.0 in the data (would poison stats).
    batches = [
        np.array([[1.0, 2.0, 1.0, 4.0, 6.0], [2.0, 3.0, 1.0, 5.0, 7.0]]),
        np.array([[3.0, 4.0, 1.0, 6.0, 8.0]]),
    ]
    frozen = _finalized(batches, ignore={2})

    mean = frozen.mean.numpy()
    std = frozen.std.numpy()
    assert mean[2] == 0.0
    assert std[2] == 1.0

    # Manual Welford over the non-ignored dims.
    data = np.concatenate(batches, axis=0)[:, [0, 1, 3, 4]]
    np.testing.assert_allclose(mean[[0, 1, 3, 4]], data.mean(axis=0), atol=1e-6)
    np.testing.assert_allclose(
        std[[0, 1, 3, 4]],
        torch.from_numpy(np.sqrt(data.var(axis=0, ddof=1)) + 1e-6).float().numpy(),
        atol=1e-6,
    )


def test_mask_dims_survive_normalization_as_binary() -> None:
    """Normalizing a 0/1 mask with identity stats leaves it unchanged."""
    frozen = _finalized([np.array([[0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])], ignore={0})
    mask = np.array([[0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    out = frozen.normalize(torch.from_numpy(mask)).numpy()
    np.testing.assert_allclose(out[:, [0]], mask[:, [0]], atol=1e-6)


def test_no_ignore_behaves_identically_to_plain_stats() -> None:
    batches = [np.random.default_rng(0).normal(size=(8, 5)) for _ in range(3)]
    with_ignore = _finalized(batches, ignore=set())
    data = np.concatenate(batches, axis=0)
    np.testing.assert_allclose(with_ignore.mean.numpy(), data.mean(axis=0), atol=1e-6)
    np.testing.assert_allclose(
        with_ignore.std.numpy(),
        torch.from_numpy(np.sqrt(data.var(axis=0, ddof=1)) + 1e-6).float().numpy(),
        atol=1e-6,
    )


def test_ignore_dims_out_of_range_raises_on_update() -> None:
    norm = RunningStatNormalizer(ignore_dims={9})
    with pytest.raises(ValueError, match="out of range"):
        norm.update(np.zeros((2, 5)))


def test_ignore_dims_negative_raises_on_update() -> None:
    norm = RunningStatNormalizer(ignore_dims={-1})
    with pytest.raises(ValueError, match="out of range"):
        norm.update(np.zeros((2, 5)))

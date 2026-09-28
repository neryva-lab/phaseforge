"""Change-point segmentation for topological regime discovery.

Solves the offline segmentation objective exactly by optimal-partitioning
dynamic programming::

    min_τ Σ_j C(s_{τ_j:τ_{j+1}}) + β · |τ|

where ``s_t`` is the task-space signal, ``C(·)`` is the segment cost
(Gaussian negative log-likelihood with identity covariance, i.e. the
sum of squared errors up to constants), ``β`` penalizes over-segmentation,
and ``|τ|`` is the number of change points. A minimum segment length
suppresses noise-induced switching.

Pure numpy, CPU-only, deterministic. Trajectory lengths in this project
are a few hundred steps, so the exact ``O(T²·D)`` DP is cheap; no
approximate pruning is used (the optimum of the stated objective is
returned exactly).
"""

from __future__ import annotations

import numpy as np


def _segment_costs_l2(signal: np.ndarray, min_len: int) -> np.ndarray:
    """Pairwise Gaussian-NLL segment costs ``C[i, j]`` for ``s[i:j]``.

    Only entries with ``j - i >= min_len`` are finite; the rest are +inf
    so the DP can never select an under-length segment.
    """
    length, _dim = signal.shape
    cumsum = np.zeros((length + 1, signal.shape[1]), dtype=np.float64)
    cumsum[1:] = np.cumsum(signal, axis=0)
    cumsq = np.zeros(length + 1, dtype=np.float64)
    cumsq[1:] = np.cumsum(np.sum(signal * signal, axis=1))
    costs = np.full((length + 1, length + 1), np.inf, dtype=np.float64)
    for start in range(length):
        earliest_end = start + min_len
        if earliest_end > length:
            break
        counts = np.arange(earliest_end, length + 1) - start
        sums = cumsum[earliest_end:] - cumsum[start]
        sqs = cumsq[earliest_end:] - cumsq[start]
        # Σ‖x - mean‖² = Σ‖x‖² - ‖Σx‖² / n (clamped against fp noise).
        sse = sqs - np.sum(sums * sums, axis=1) / counts
        costs[start, earliest_end:] = np.maximum(sse, 0.0)
    return costs


def run_pelt(
    signal: np.ndarray,
    penalty_beta: float,
    min_segment_len: int = 5,
    cost: str = "l2",
) -> np.ndarray:
    """Segment a task-space signal into constant-geometry intervals.

    Args:
        signal: Array of shape ``(T, Ds)`` (one demonstration).
        penalty_beta: ``β`` penalty per change point (>= 0). Larger values
            yield fewer segments.
        min_segment_len: Minimum admissible segment length (>= 1).
        cost: Segment cost; only ``"l2"`` (Gaussian NLL) is implemented.

    Returns:
        Integer boundaries of shape ``(M+1,)`` with ``bounds[0] == 0`` and
        ``bounds[-1] == T``; segment ``j`` is ``signal[bounds[j]:bounds[j+1]]``.
    """
    if cost != "l2":
        raise ValueError(f"Unknown PELT cost {cost!r}; expected 'l2'.")
    beta = float(penalty_beta)
    if beta < 0.0:
        raise ValueError(f"penalty_beta must be >= 0.0, got {beta}.")
    min_len = int(min_segment_len)
    if min_len < 1:
        raise ValueError(f"min_segment_len must be >= 1, got {min_len}.")
    sig = np.asarray(signal, dtype=np.float64)
    if sig.ndim != 2:
        raise ValueError(f"Expected signal shape (T, Ds), got {sig.shape}.")
    if not np.isfinite(sig).all():
        raise ValueError("Non-finite values in the PELT input signal.")
    length = sig.shape[0]
    if length == 0:
        raise ValueError("Cannot segment an empty signal.")
    if length <= min_len:
        return np.array([0, length], dtype=np.int64)

    costs = _segment_costs_l2(sig, min_len)
    best = np.full(length + 1, np.inf, dtype=np.float64)
    prev = np.full(length + 1, -1, dtype=np.int64)
    best[0] = 0.0
    for end in range(1, length + 1):
        # Only starts leaving a valid-length final segment are feasible.
        candidates = np.arange(0, end - min_len + 1)
        if candidates.size == 0:
            continue
        seg = costs[candidates, end]
        feasible = np.isfinite(seg) & np.isfinite(best[candidates])
        if not np.any(feasible):
            continue
        total = best[candidates] + seg
        total[candidates > 0] += beta
        total[~feasible] = np.inf
        choice = int(np.argmin(total))
        prev[end] = candidates[choice]
        best[end] = float(total[choice])

    if not np.isfinite(best[length]):
        # Unreachable only for degenerate inputs; fall back to one segment
        # rather than failing the whole discovery run.
        return np.array([0, length], dtype=np.int64)
    boundaries: list[int] = []
    cursor = length
    while cursor > 0:
        boundaries.append(cursor)
        cursor = int(prev[cursor])
        if cursor < 0:  # pragma: no cover - guarded by the finite check above
            return np.array([0, length], dtype=np.int64)
    boundaries.append(0)
    return np.asarray(boundaries[::-1], dtype=np.int64)


__all__ = ["run_pelt"]

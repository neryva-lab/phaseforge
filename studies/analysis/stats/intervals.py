"""Confidence intervals for proportions and seed-level summaries."""

from __future__ import annotations

import math


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (n > 0)."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not 0 <= successes <= n:
        raise ValueError(f"successes {successes} outside [0, {n}]")
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def seed_mean_and_wilson(
    successes_per_seed: list[int], n_per_seed: int
) -> tuple[float, float, float]:
    """Pooled-proportion mean with the Wilson interval over all episodes."""
    total_successes = sum(successes_per_seed)
    lo, hi = wilson_interval(total_successes, n_per_seed * len(successes_per_seed))
    return total_successes / (n_per_seed * len(successes_per_seed)), lo, hi


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("mean of empty list")
    return sum(values) / len(values)


def sample_std(values: list[float]) -> float:
    """Sample standard deviation (n-1); 0.0 for a single value."""
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))

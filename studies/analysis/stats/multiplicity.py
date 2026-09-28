"""Multiplicity correction for the pre-declared primary comparison family."""

from __future__ import annotations


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (family-wise control).

    Returns adjusted values in the input order, monotonically enforced and
    capped at 1.0.
    """
    if not all(0.0 <= p <= 1.0 for p in p_values):
        raise ValueError(f"p-values must be in [0, 1], got {p_values}")
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [0.0] * len(p_values)
    running_max = 0.0
    m = len(p_values)
    for rank, idx in enumerate(order):
        adj = (m - rank) * p_values[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(1.0, running_max)
    return adjusted

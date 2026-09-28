"""Trajectory alignment helpers for overlay plots."""

from __future__ import annotations

import math


def resample(series: list[tuple[float, float]], grid: list[float]) -> list[float]:
    """Linear interpolation of (x, y) points onto ``grid`` (x monotone)."""
    if len(series) < 2:
        return [series[0][1] if series else math.nan] * len(grid)
    xs = [p[0] for p in series]
    ys = [p[1] for p in series]
    out: list[float] = []
    j = 0
    for x in grid:
        while j < len(xs) - 2 and xs[j + 1] < x:
            j += 1
        x0, x1 = xs[j], xs[j + 1]
        y0, y1 = ys[j], ys[j + 1]
        if x1 == x0:
            out.append(y0)
            continue
        t = (x - x0) / (x1 - x0)
        t = max(0.0, min(1.0, t))
        out.append(y0 + t * (y1 - y0))
    return out


def common_grid(series_list: list[list[tuple[float, float]]], points: int = 100) -> list[float]:
    """A shared grid spanning the union of all series' x-ranges."""
    if not series_list:
        return []
    lo = min(s[0][0] for s in series_list if s)
    hi = max(s[-1][0] for s in series_list if s)
    if hi <= lo:
        return [lo]
    step = (hi - lo) / (points - 1)
    return [lo + i * step for i in range(points)]


def minmax_band(values_per_series: list[list[float]]) -> tuple[list[float], list[float]]:
    """Per-column min and max across aligned series (envelope band)."""
    if not values_per_series:
        return [], []
    n = len(values_per_series[0])
    lows, highs = [], []
    for i in range(n):
        col = [row[i] for row in values_per_series if not math.isnan(row[i])]
        if not col:
            lows.append(math.nan)
            highs.append(math.nan)
        else:
            lows.append(min(col))
            highs.append(max(col))
    return lows, highs

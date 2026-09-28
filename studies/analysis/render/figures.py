"""Matplotlib render engine: figure factory + shared publication primitives.

Every figure asset composes these primitives inside ``paper_style()`` and
saves through ``save()`` so styling stays in one place (plan section 5).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt

from studies.analysis.common.config import paper_root
from studies.analysis.common.style import (
    OKABE_ITO,
    save_figure,
)


def new_figure(width_kind: str = "text", height_in: float = 2.8, ncols: int = 1):
    """Create a styled figure; call within the generator (style applied on save)."""
    from studies.analysis.common.style import column_width

    width = column_width(width_kind) * ncols
    fig, ax = plt.subplots(figsize=(width, height_in))
    return fig, ax


def new_figure_grid(width_kind: str, height_in: float, nrows: int, ncols: int):
    from studies.analysis.common.style import column_width

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(column_width(width_kind), height_in),
        squeeze=False,
    )
    return fig, axes


def save(fig: plt.Figure, relative: str) -> list[Path]:
    """Save into paper_root/<relative-without-suffix> as PDF + PNG.

    Generators draw inside ``paper_style()``; this only persists the figure.
    """
    base = paper_root() / relative
    return save_figure(fig, base)


def forest(
    ax,
    labels: Sequence[str],
    points: Sequence[float],
    lows: Sequence[float],
    highs: Sequence[float],
    colors: Sequence[str] | None = None,
    seed_points: Sequence[Sequence[float]] | None = None,
    xlabel: str = "",
    show_zero: bool = True,
    capsize: float = 3.0,
) -> None:
    """Dot-and-interval rows for forest plots (paired deltas, CIs).

    Features:
    - Bold mean point estimate + horizontal range bar with caps.
    - Vertically jittered individual seed points to avoid overplotting on small sample sizes (n=3).
    - Prominent dashed zero-reference line.
    """
    import numpy as np

    y_indices = np.arange(len(labels))
    if show_zero:
        ax.axvline(0.0, color="#777777", linewidth=0.9, linestyle="--", zorder=1)

    for i, (p, lo, hi) in enumerate(zip(points, lows, highs)):
        color = colors[i] if colors else OKABE_ITO["blue"]
        ax.plot([lo, hi], [i, i], color=color, linewidth=2.0, zorder=2)
        if capsize > 0 and lo != hi:
            ax.plot([lo, lo], [i - 0.12, i + 0.12], color=color, linewidth=1.4, zorder=2)
            ax.plot([hi, hi], [i - 0.12, i + 0.12], color=color, linewidth=1.4, zorder=2)
        ax.scatter([p], [i], color=color, s=36, zorder=4, edgecolor="white", linewidth=0.6)

        # Draw individual seed points jittered vertically
        if seed_points is not None and i < len(seed_points):
            seeds = seed_points[i]
            if len(seeds) > 1:
                offsets = np.linspace(-0.14, 0.14, len(seeds))
            else:
                offsets = [0.0]
            for s_val, y_off in zip(seeds, offsets):
                ax.scatter(
                    [s_val],
                    [i + y_off],
                    facecolors="none",
                    edgecolors=color,
                    s=20,
                    linewidths=1.0,
                    alpha=0.85,
                    zorder=3,
                )

    ax.set_yticks(list(y_indices))
    ax.set_yticklabels(list(labels))
    ax.invert_yaxis()
    if xlabel:
        ax.set_xlabel(xlabel)


def plot_seed_trajectories(
    ax,
    per_seed_series: list[list[tuple[float, float]]],
    color: str,
    label: str | None = None,
    xlabel: str = "",
    ylabel: str = "",
    linestyle: str = "-",
    show_ribbon: bool = True,
    t0_marker: float | None = None,
) -> None:
    """Trajectory plot with shaded error envelopes and t0 markers."""
    import numpy as np
    from studies.analysis.stats.trajectories import common_grid, resample

    grid = common_grid(per_seed_series)
    if grid and per_seed_series:
        resampled = [resample(s, grid) for s in per_seed_series]
        matrix = np.asarray(resampled, dtype=float)  # (n_seeds, len(grid))
        means = np.nanmean(matrix, axis=0)
        mins = np.nanmin(matrix, axis=0)
        maxs = np.nanmax(matrix, axis=0)

        if show_ribbon and len(per_seed_series) > 1:
            ax.fill_between(grid, mins, maxs, color=color, alpha=0.15, zorder=2)

        ax.plot(grid, means, color=color, linestyle=linestyle, linewidth=2.0, label=label, zorder=4)

        if t0_marker is not None and not np.isnan(t0_marker):
            ax.scatter(
                [0],
                [t0_marker],
                color=color,
                marker="D",
                s=28,
                edgecolor="black",
                linewidth=0.6,
                zorder=5,
            )
    else:
        # Fallback if no grid
        for series in per_seed_series:
            xs = [p[0] for p in series]
            ys = [p[1] for p in series]
            ax.plot(xs, ys, color=color, alpha=0.35, linewidth=0.9, linestyle=linestyle)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def heatmap(
    ax,
    matrix,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    cmap: str = "Blues",
    vmin=None,
    vmax=None,
    annotate: bool = True,
    fmt: str = "{:.2f}",
) -> None:
    """High-contrast heatmap with clean cell borders and auto text contrast."""
    import numpy as np

    data = np.asarray(matrix, dtype=float)
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=0, ha="center")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    # Minor ticks for clean grid dividers
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    if annotate:
        norm_min = vmin if vmin is not None else np.nanmin(data)
        norm_max = vmax if vmax is not None else np.nanmax(data)
        range_span = max(1e-9, norm_max - norm_min)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                value = data[i, j]
                if np.isnan(value):
                    continue
                lum = (value - norm_min) / range_span
                text_color = "white" if lum > 0.55 else "black"
                ax.text(
                    j,
                    i,
                    fmt.format(value),
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold" if lum > 0.4 else "normal",
                    color=text_color,
                )
    return im


def ecdf(
    ax,
    values: Sequence[float],
    color: str,
    label: str | None = None,
    linestyle: str = "-",
    linewidth: float = 1.8,
) -> None:
    import numpy as np

    data = np.sort(np.asarray(values, dtype=float))
    if data.size == 0:
        return
    ys = np.arange(1, data.size + 1) / data.size
    ax.step(
        np.concatenate([data, data[-1:]]),
        np.concatenate([ys, [1.0]]),
        where="post",
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        label=label,
    )


def stacked_bars(
    ax,
    row_labels: Sequence[str],
    shares: dict[str, list[float]],
    colors: dict[str, str] | None = None,
) -> None:
    """100% stacked horizontal bars (outcome/failure categories)."""
    import numpy as np

    categories = list(shares)
    colors = colors or {}
    palette = [colors.get(c, OKABE_ITO["grey"]) for c in categories]
    data = np.asarray([shares[c] for c in categories], dtype=float)
    left = np.zeros(len(row_labels))
    y_positions = np.arange(len(row_labels))
    for row, cat in enumerate(categories):
        ax.barh(
            y_positions,
            data[row],
            left=left,
            color=palette[row],
            label=cat,
            height=0.65,
            edgecolor="white",
            linewidth=0.9,
        )
        left += data[row]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(list(row_labels))
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda x, _pos: f"{x:.0%}")

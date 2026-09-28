"""A16 — Stage-2 normalized routing-entropy trajectories (appendix record).

Entropy with the same data, bounds, and styling as F3 (which shows NMI /
switch rate only). Descriptive only; entropy is not interpreted as a
performance measure anywhere.
"""

from __future__ import annotations

from pathlib import Path

from studies.analysis.common import registry
from studies.analysis.common.style import method_color, paper_style
from studies.analysis.dataset import AnalysisDataset
from studies.analysis.render.figures import save

METHODS = (
    "precision_residual_phaseforge",
    "final_aligned_softmax_top1",
    "precision_residual_phase_random_router",
    "precision_residual_plain_encoder",
)

FIELD = "routing_entropy"
YLABEL = "Normalized Routing\nEntropy ($H / \\ln K$)"


def generate(dataset: AnalysisDataset) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np
    from studies.analysis.stats.trajectories import common_grid, resample

    tasks = [t for t in ("Can", "Square") if t in registry.tasks()]
    method_names = [m for m in METHODS if m in registry.matrix_method_names() or any(m == em.name for em in registry.methods("final"))]

    with paper_style():
        fig, axes = plt.subplots(
            1,
            len(tasks),
            figsize=(7.2, 2.7),
            squeeze=False,
            sharex="col",
            sharey="row",
        )

        # Global min/max across both tasks and all methods (same rule as F3)
        all_vals = []
        for task in tasks:
            for method in method_names:
                for seed in registry.seeds("final"):
                    key = (task, method, seed, 2)
                    if key in dataset.curves:
                        series = dataset.curves[key].series(FIELD)
                        if series:
                            all_vals.extend([pt[1] for pt in series if not np.isnan(pt[1])])
                    init = dataset.init_routing.get(key)
                    if init is not None and init.t0_normalized_routing_entropy is not None:
                        val = float(init.t0_normalized_routing_entropy)
                        if not np.isnan(val):
                            all_vals.append(val)

        y_low, y_high = 0.0, 1.02
        if all_vals:
            v_min, v_max = float(np.min(all_vals)), float(np.max(all_vals))
            span = v_max - v_min
            pad = 0.06 * span
            y_low, y_high = v_min - pad, min(1.02, v_max + pad)

        # Render plots
        for col, task in enumerate(tasks):
            ax = axes[0][col]
            for method in method_names:
                color = method_color(method)
                per_seed_series = []
                t0_markers = []
                for seed in registry.seeds("final"):
                    key = (task, method, seed, 2)
                    if key in dataset.curves:
                        series = dataset.curves[key].series(FIELD)
                        if series:
                            per_seed_series.append(series)
                    init = dataset.init_routing.get(key)
                    if init is not None and init.t0_normalized_routing_entropy is not None:
                        val = float(init.t0_normalized_routing_entropy)
                        if not np.isnan(val):
                            t0_markers.append(val)

                if not per_seed_series:
                    continue

                for s_data in per_seed_series:
                    xs = [pt[0] for pt in s_data]
                    ys = [pt[1] for pt in s_data]
                    ax.plot(xs, ys, color=color, alpha=0.35, linewidth=0.9, linestyle="--")

                grid = common_grid(per_seed_series)
                if grid:
                    resampled = [resample(s, grid) for s in per_seed_series]
                    matrix = np.asarray(resampled, dtype=float)
                    means = np.nanmean(matrix, axis=0)
                    ax.plot(grid, means, color=color, linewidth=2.0, alpha=1.0,
                            label=registry.display_name(method) if col == 0 else None,
                            zorder=4)

                for t0_val in t0_markers:
                    ax.scatter([0], [t0_val], facecolors="none", edgecolors=color,
                               marker="D", s=22, linewidth=1.0, zorder=5)

            ax.set_ylim((y_low, y_high))

            ax.grid(axis="y", linestyle="-", color="#E5E5EA", linewidth=0.7, alpha=0.8)
            ax.grid(axis="x", visible=False)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#8E8E93")
            ax.spines["left"].set_linewidth(0.8)
            ax.spines["bottom"].set_color("#8E8E93")
            ax.spines["bottom"].set_linewidth(0.8)

            ax.tick_params(axis="both", colors="#3A3A3C", labelsize=8, length=3)

            if col == 0:
                ax.set_ylabel(YLABEL, fontsize=8.0, fontweight="500", color="#1C1C1E", multialignment="center")
            ax.set_xlabel("Stage-2 Epoch", fontsize=8.5, fontweight="500", color="#1C1C1E")
            ax.set_xticks([0, 50, 100, 150, 200])
            ax.set_title(task, fontsize=10.5, fontweight="bold", pad=8, color="#000000")

        # Two-tier legend
        from matplotlib.lines import Line2D
        h_methods, l_methods = axes[0][0].get_legend_handles_labels()
        h_marks = [
            Line2D([0], [0], color="#555555", lw=2.0, label="Seed mean"),
            Line2D([0], [0], color="#888888", lw=0.9, linestyle="--", label="Individual seed trajectory"),
            Line2D([0], [0], marker="D", color="w", markerfacecolor="none", markeredgecolor="#555555", markersize=5, label="t=0 init point"),
        ]

        leg1 = fig.legend(
            h_methods,
            l_methods,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=4,
            frameon=False,
            fontsize=8.0,
        )
        leg2 = fig.legend(
            h_marks,
            [m.get_label() for m in h_marks],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.945),
            ncol=3,
            frameon=False,
            fontsize=7.5,
        )
        fig.add_artist(leg1)
        fig.subplots_adjust(top=0.68, bottom=0.20, left=0.15, right=0.96, hspace=0.28, wspace=0.18)

    return save(fig, "figures/appendix/A16_routing_entropy")

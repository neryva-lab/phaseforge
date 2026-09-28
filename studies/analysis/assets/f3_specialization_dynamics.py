"""F3 — Stage-2 specialization dynamics: NMI and switch rate.

Two-row main-text figure (routing entropy is archived separately in appendix
asset A16). Dynamic data-driven limits with 6% offset padding and physical
boundary clamping. Shared y-scales across tasks ensure faithful comparison
without wasted whitespace.
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

FIELDS = (
    ("nmi", "Phase–Expert NMI"),
    ("switch_rate", "Routing Switch Rate"),
)


def generate(dataset: AnalysisDataset) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np
    from studies.analysis.stats.trajectories import common_grid, resample

    tasks = [t for t in ("Can", "Square") if t in registry.tasks()]
    method_names = [m for m in METHODS if m in registry.matrix_method_names() or any(m == em.name for em in registry.methods("final"))]

    with paper_style():
        # Share y across columns (tasks) for each metric row
        fig, axes = plt.subplots(
            len(FIELDS),
            len(tasks),
            figsize=(7.2, 4.5),
            squeeze=False,
            sharex="col",
            sharey="row",
        )

        # 1. Pre-compute global min/max per field across both tasks and all methods
        field_bounds = {}
        for field, _ in FIELDS:
            all_vals = []
            for task in tasks:
                for method in method_names:
                    for seed in registry.seeds("final"):
                        key = (task, method, seed, 2)
                        # Collect trajectory points
                        if key in dataset.curves:
                            series = dataset.curves[key].series(field)
                            if series:
                                all_vals.extend([pt[1] for pt in series if not np.isnan(pt[1])])
                        # Collect t=0 initialization markers
                        init = dataset.init_routing.get(key)
                        if init is not None:
                            val = None
                            if field == "nmi" and init.t0_nmi is not None:
                                val = float(init.t0_nmi)
                            elif field == "routing_entropy" and init.t0_normalized_routing_entropy is not None:
                                val = float(init.t0_normalized_routing_entropy)
                            if val is not None and not np.isnan(val):
                                all_vals.append(val)

            if all_vals:
                v_min, v_max = float(np.min(all_vals)), float(np.max(all_vals))
                span = v_max - v_min
                pad = 0.06 * span  # 6% breathing room

                y_low = v_min - pad
                y_high = v_max + pad

                # Physical boundary clamping
                if field == "switch_rate":
                    y_low = max(0.0, y_low)  # Switch rate cannot be negative
                if field in ("nmi", "routing_entropy"):
                    y_high = min(1.02, y_high)  # Normalized ceiling at 1.0

                field_bounds[field] = (y_low, y_high)

        # 2. Render plots
        for col, task in enumerate(tasks):
            for row, (field, ylabel) in enumerate(FIELDS):
                ax = axes[row][col]
                for method in method_names:
                    color = method_color(method)
                    per_seed_series = []
                    t0_markers = []
                    for seed in registry.seeds("final"):
                        key = (task, method, seed, 2)
                        if key in dataset.curves:
                            series = dataset.curves[key].series(field)
                            if series:
                                per_seed_series.append(series)
                        init = dataset.init_routing.get(key)
                        if init is not None:
                            val = None
                            if field == "nmi" and init.t0_nmi is not None:
                                val = float(init.t0_nmi)
                            elif field == "routing_entropy" and init.t0_normalized_routing_entropy is not None:
                                val = float(init.t0_normalized_routing_entropy)
                            if val is not None and not np.isnan(val):
                                t0_markers.append(val)

                    if not per_seed_series:
                        continue

                    # Individual seed trajectories (thin dashed lines)
                    for s_data in per_seed_series:
                        xs = [pt[0] for pt in s_data]
                        ys = [pt[1] for pt in s_data]
                        ax.plot(xs, ys, color=color, alpha=0.35, linewidth=0.9, linestyle="--")

                    # Seed mean (heavy line)
                    grid = common_grid(per_seed_series)
                    if grid:
                        resampled = [resample(s, grid) for s in per_seed_series]
                        matrix = np.asarray(resampled, dtype=float)
                        means = np.nanmean(matrix, axis=0)
                        ax.plot(grid, means, color=color, linewidth=2.0, alpha=1.0,
                                label=registry.display_name(method) if (row == 0 and col == 0) else None,
                                zorder=4)

                    # t=0 seed markers (diamonds)
                    for t0_val in t0_markers:
                        ax.scatter([0], [t0_val], facecolors="none", edgecolors=color,
                                   marker="D", s=22, linewidth=1.0, zorder=5)

                if field in field_bounds:
                    ax.set_ylim(field_bounds[field])

                # Minimal grid and spine styling
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
                    ax.set_ylabel(ylabel, fontsize=8.0, fontweight="500", color="#1C1C1E", multialignment="center")
                if row == len(FIELDS) - 1:
                    ax.set_xlabel("Stage-2 Epoch", fontsize=8.5, fontweight="500", color="#1C1C1E")
                    ax.set_xticks([0, 50, 100, 150, 200])
                if row == 0:
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
        fig.subplots_adjust(top=0.80, bottom=0.10, left=0.15, right=0.96, hspace=0.30, wspace=0.18)

    return save(fig, "figures/main/F3_specialization")
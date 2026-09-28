"""A8 — expert load balance score trajectories over Stage-2 training."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from studies.analysis.common import registry
from studies.analysis.common.style import method_color, paper_style
from studies.analysis.dataset import AnalysisDataset
from studies.analysis.render.figures import save
from studies.analysis.stats.trajectories import common_grid, resample

TASKS = ("Lift", "Can", "Square")

# Layering: solid base at bottom (zorder=2), broken patterns on top (zorders 3, 4, 5),
# so co-located methods on the plateau stay distinct and visible.
MOE_METHODS = (
    ("precision_residual_phaseforge", "PhaseForge", "-", 2.4, 2),
    ("final_aligned_softmax_top1", "Softmax Top-1", ":", 1.8, 3),
    ("precision_residual_phase_random_router", "Phase-Random", "-.", 1.5, 4),
    ("precision_residual_scratch_moe", "Scratch MoE", "--", 1.5, 5),
)


def generate(dataset: AnalysisDataset) -> list[Path]:
    import matplotlib.pyplot as plt

    tasks = [t for t in TASKS if t in registry.tasks()]
    
    with paper_style():
        fig, axes = plt.subplots(
            1, len(tasks), figsize=(7.2, 2.7), squeeze=True, sharey=True
        )

        # 1. First pass: collect all data and determine the global minimum
        task_data = {}
        all_vals = []
        for task in tasks:
            task_data[task] = {}
            for method_id, _, _, _, _ in MOE_METHODS:
                per_seed_series = []
                for seed in registry.seeds("final"):
                    for m_cand in (method_id, "phaseforge" if method_id == "precision_residual_phaseforge" else method_id):
                        key = (task, m_cand, seed, 2)
                        if key in dataset.curves:
                            series = dataset.curves[key].series("top1_balance")
                            if series:
                                per_seed_series.append(series)
                                all_vals.extend([pt[1] for pt in series if not np.isnan(pt[1])])
                            break
                if per_seed_series:
                    task_data[task][method_id] = per_seed_series

        # Floor the y-axis just below the lowest observed value
        if all_vals:
            v_min = float(np.min(all_vals))
            # Pad below and round the floor to the nearest 0.1
            y_floor = max(0.0, np.floor((v_min - 0.04) * 10) / 10.0)
        else:
            y_floor = 0.40

        # Tick marks from the floor up to 1.0
        step = 0.10 if (1.0 - y_floor) <= 0.6 else 0.20
        yticks = np.arange(y_floor, 1.01, step)

        # 2. Render subplots
        for col, task in enumerate(tasks):
            ax = axes[col]

            # Reference line for ideal balance
            ax.axhline(
                1.0,
                color="#888888",
                linestyle="--",
                linewidth=1.0,
                zorder=1,
            )

            for method_id, display, linestyle, linewidth, z_order in MOE_METHODS:
                if method_id not in task_data[task]:
                    continue

                color = method_color(method_id)
                per_seed_series = task_data[task][method_id]

                # Faint individual seed lines
                for s_data in per_seed_series:
                    xs = [pt[0] for pt in s_data]
                    ys = [pt[1] for pt in s_data]
                    ax.plot(xs, ys, color=color, alpha=0.18, linewidth=0.6, linestyle=":", zorder=1)

                # Resampled arithmetic mean with layered linestyles
                grid = common_grid(per_seed_series)
                if grid:
                    resampled = [resample(s, grid) for s in per_seed_series]
                    matrix = np.asarray(resampled, dtype=float)
                    means = np.nanmean(matrix, axis=0)
                    ax.plot(
                        grid,
                        means,
                        color=color,
                        linestyle=linestyle,
                        linewidth=linewidth,
                        alpha=1.0,
                        label=display if col == 0 else None,
                        zorder=z_order,
                    )

            # Axis limits from the computed floor; no data clipped
            ax.set_ylim(y_floor - 0.02, 1.03)
            ax.set_yticks(yticks)
            ax.set_yticklabels([f"{t:.1f}" for t in yticks])

            # Minimal grid and borders
            ax.grid(axis="y", linestyle=":", color="#CCCCCC", alpha=0.6)
            ax.grid(axis="x", visible=False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            ax.set_title(task, fontsize=10.0, fontweight="bold", pad=8)
            ax.set_xlabel("Stage-2 Epoch", fontsize=8.5)
            if col == 0:
                ax.set_ylabel("Expert Balance Score", fontsize=8.5, fontweight="bold")

        # 3. Unified single-row legend: Methods first, Reference line last
        h, l = axes[0].get_legend_handles_labels()
        from matplotlib.lines import Line2D
        h.append(Line2D([0], [0], color="#888888", linestyle="--", linewidth=1.0))
        l.append("Ideal (1.0)")

        fig.legend(
            h,
            l,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=5,
            frameon=False,
            fontsize=8.0,
            columnspacing=1.5,
            handlelength=2.2,
            handletextpad=0.5,
        )

        fig.subplots_adjust(top=0.78, bottom=0.18, left=0.10, right=0.97, wspace=0.15)

    return save(fig, "figures/appendix/A8_balance")
"""A6 — steps-to-completion ECDFs on solvable tasks (Lift, Can, Square).

Conditional completion-step distributions among successful episodes.
Unified top legend for method styles with zero title collisions.
Compact lower-right stats blocks report exact n and median without obstructing curves.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from studies.analysis.common import registry
from studies.analysis.common.style import method_color, paper_style
from studies.analysis.dataset import AnalysisDataset
from studies.analysis.render.figures import ecdf, save

SOLVABLE_TASKS = ("Lift", "Can", "Square")
METHODS_TO_COMPARE = (
    ("precision_residual_phaseforge", "-", 2.0, "PhaseForge"),
    ("bc", "--", 1.5, "BC"),
    ("final_aligned_softmax_top1", ":", 1.5, "Softmax Top-1"),
    ("precision_residual_plain_encoder", "-.", 1.5, "Plain Encoder"),
    ("precision_residual_phase_random_router", (0, (3, 1, 1, 1)), 1.5, "Phase-Random"),
    ("precision_residual_scratch_moe", (0, (1, 1)), 1.5, "Scratch MoE"),
)

# Compact acronyms for in-panel statistics
SHORT_NAMES = {
    "precision_residual_phaseforge": "PhaseForge",
    "bc": "BC",
    "final_aligned_softmax_top1": "Softmax",
    "precision_residual_plain_encoder": "Plain Enc",
    "precision_residual_phase_random_router": "Random",
    "precision_residual_scratch_moe": "Scratch",
}


def generate(dataset: AnalysisDataset) -> list[Path]:
    import matplotlib.pyplot as plt

    with paper_style():
        # Height 3.2 leaves room for the top legend and x-labels
        fig, axes = plt.subplots(
            1, len(SOLVABLE_TASKS), figsize=(7.2, 3.2), squeeze=True, sharey=True
        )

        legend_handles = []
        legend_labels = []

        for col, task in enumerate(SOLVABLE_TASKS):
            ax = axes[col]
            task_stats = []

            for method, linestyle, linewidth, display_name in METHODS_TO_COMPARE:
                if method not in registry.matrix_method_names():
                    continue

                success_steps = []
                for seed in registry.seeds("final"):
                    for ep in dataset.episodes.get((task, method, seed), []):
                        if ep.valid and ep.success and ep.steps > 0:
                            success_steps.append(ep.steps)

                if success_steps:
                    color = method_color(method)
                    med = float(np.median(success_steps))
                    n_succ = len(success_steps)

                    # Collect per-method stats for the in-panel annotation block
                    task_stats.append(f"{SHORT_NAMES[method]}: $n$={n_succ}, med={med:.0f}")

                    # ECDF step curve of completion steps
                    line = ecdf(
                        ax,
                        success_steps,
                        color=color,
                        linestyle=linestyle,
                        linewidth=linewidth,
                        label=display_name if col == 0 else None,
                    )

                    # No-op: legend handles are collected from axes[0] after the loop
                    if col == 0 and line is not None:
                        pass

            # Subplot axes formatting
            ax.set_title(task, fontsize=10, fontweight="bold", pad=8)
            ax.set_xlabel("Completion Steps", fontsize=8.5)
            ax.set_ylim(0.0, 1.05)
            ax.grid(True, linestyle=":", alpha=0.35)
            if col == 0:
                ax.set_ylabel("Empirical CDF", fontsize=8.5)

            # Hide top and right spines
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Compact stats block in the lower-right corner of the panel
            # (anchored at 0.96, 0.04, right/bottom, so it cannot overlap the curves)
            if task_stats:
                stats_text = "\n".join(task_stats)
                ax.text(
                    0.96,
                    0.04,
                    stats_text,
                    transform=ax.transAxes,
                    fontsize=5.8,
                    ha="right",
                    va="bottom",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        alpha=0.85,
                        edgecolor="#E5E5EA",
                        linewidth=0.6,
                    ),
                    zorder=2,
                )

        # Unified top legend (method names only, matching the rest of the paper)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(
            h,
            l,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=3,
            frameon=False,
            fontsize=7.8,
            handlelength=2.0,
            columnspacing=1.8,
        )

        # top=0.76 leaves a vertical buffer so the legend never touches the titles
        fig.subplots_adjust(top=0.76, bottom=0.16, left=0.09, right=0.97, wspace=0.18)

    return save(fig, "figures/appendix/A6_steps_ecdf")
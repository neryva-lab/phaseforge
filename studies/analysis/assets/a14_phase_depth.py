"""A14 — phase-depth analysis: distribution of deepest phase reached (max_phase)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from studies.analysis.common import registry
from studies.analysis.common.style import method_color, paper_style
from studies.analysis.dataset import AnalysisDataset
from studies.analysis.render.figures import save


def generate(dataset: AnalysisDataset) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    tasks = registry.tasks()
    target_methods = (
        "precision_residual_phaseforge",
        "bc",
        "final_aligned_softmax_top1",
        "precision_residual_plain_encoder",
        "precision_residual_phase_random_router",
    )
    methods = [m for m in target_methods if m in registry.matrix_method_names()]

    # Derive phase range dynamically from validated episode artifacts
    all_observed_phases = [
        ep.max_phase
        for eps in dataset.episodes.values()
        for ep in eps
        if ep.max_phase is not None
    ]
    max_depth = max(all_observed_phases) if all_observed_phases else 5

    with paper_style():
        fig, axes = plt.subplots(
            1, len(tasks), figsize=(7.2, 2.8), squeeze=True, sharey=True
        )
        for col, task in enumerate(tasks):
            ax = axes[col]
            depths_by_method: dict[str, Counter[int]] = {}
            for method in methods:
                counter: Counter[int] = Counter()
                for seed in registry.seeds("final"):
                    for ep in dataset.episodes.get((task, method, seed), []):
                        if ep.max_phase is not None:
                            counter[ep.max_phase] += 1
                if counter:
                    depths_by_method[method] = counter
            if not depths_by_method:
                ax.set_visible(False)
                continue

            width = 0.8 / len(depths_by_method)
            for i, (method, counter) in enumerate(depths_by_method.items()):
                total = sum(counter.values())
                xs = np.arange(max_depth + 1) + (i - len(depths_by_method) / 2 + 0.5) * width
                ax.bar(
                    xs,
                    [counter.get(d, 0) / total for d in range(max_depth + 1)],
                    width=width,
                    color=method_color(method),
                    label=registry.display_name(method) if col == 0 else None,
                    edgecolor="white",
                    linewidth=0.4,
                )
            ax.set_ylim(0.0, 1.08)
            ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xticks(range(max_depth + 1))
            ax.set_xticklabels([f"P{d}" for d in range(max_depth + 1)], fontsize=7.0)
            ax.set_title(task, fontsize=9.5, fontweight="bold", pad=7)
            ax.grid(axis="y", linestyle=":", alpha=0.3)
            if col == 0:
                ax.set_ylabel("Share of Episodes", fontsize=8.5)

        # Single centered xlabel at bottom
        fig.text(0.53, 0.03, "Deepest Phase Reached", ha="center", fontsize=8.5, fontweight="bold")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=len(methods),
            frameon=False,
            fontsize=7.5,
        )
        fig.subplots_adjust(top=0.80, bottom=0.18, left=0.08, right=0.98, wspace=0.16)
    return save(fig, "figures/appendix/A14_phase_depth")

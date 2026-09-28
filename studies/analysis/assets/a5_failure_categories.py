"""A5 — episode outcome / failure-category breakdown per method across all 5 tasks."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from studies.analysis.common import registry
from studies.analysis.common.style import OKABE_ITO, paper_style
from studies.analysis.dataset import AnalysisDataset
from studies.analysis.render.figures import save, stacked_bars

CATEGORY_COLORS = {
    "success": OKABE_ITO["green"],
    "task_timeout": OKABE_ITO["vermillion"],
    "other/invalid": OKABE_ITO["grey"],
}


def generate(dataset: AnalysisDataset) -> list[Path]:
    import matplotlib.pyplot as plt

    methods = [m for m in registry.matrix_method_names() if m != "precision_residual_oracle"]
    method_labels = [registry.display_name(m) for m in methods]
    tasks = list(registry.tasks())

    # Dynamically discover all termination categories present across all evaluated episodes
    present_categories = Counter()
    for key, eps in dataset.episodes.items():
        if key[0] in tasks and key[1] in methods:
            for ep in eps:
                if not ep.valid:
                    present_categories["invalid"] += 1
                elif ep.success:
                    present_categories["success"] += 1
                else:
                    reason = ep.termination_reason or "other"
                    present_categories[reason] += 1

    # Order categories: success first, then the remaining present failure
    # categories (zero-count categories suppressed)
    ordered_cats = ["success"] + [c for c in sorted(present_categories.keys()) if c != "success" and present_categories[c] > 0]
    cat_colors = {
        "success": OKABE_ITO["green"],
        "task_timeout": OKABE_ITO["vermillion"],
        "invalid": OKABE_ITO["grey"],
        "other": OKABE_ITO["purple"],
    }
    cat_display = {
        "success": "Success",
        "task_timeout": "Task Timeout",
        "invalid": "Invalid Attempt",
        "other": "Other Failure",
    }

    with paper_style():
        fig, axes = plt.subplots(1, len(tasks), figsize=(8.8, 3.4), sharey=False)

        for col, task in enumerate(tasks):
            ax = axes[col]
            shares = {cat: [] for cat in ordered_cats}

            for method in methods:
                cats: Counter[str] = Counter()
                total = 0
                for seed in registry.seeds("final"):
                    key = (task, method, seed)
                    if key not in dataset.episodes:
                        continue
                    for ep in dataset.episodes[key]:
                        total += 1
                        if not ep.valid:
                            cats["invalid"] += 1
                        elif ep.success:
                            cats["success"] += 1
                        else:
                            reason = ep.termination_reason or "other"
                            cats[reason] += 1

                if total == 0:
                    for cat in ordered_cats:
                        shares[cat].append(0.0)
                else:
                    for cat in ordered_cats:
                        shares[cat].append(cats.get(cat, 0) / total)

            stacked_bars(
                ax,
                method_labels if col == 0 else ["" for _ in method_labels],
                shares,
                colors={c: cat_colors.get(c, OKABE_ITO["grey"]) for c in ordered_cats},
            )
            # Semantic isolation for privileged Teacher-Forced (last in MATRIX_ORDER)
            # inverted axis: y=0 top, teacher_forced at bottom -> separator at idx-0.5
            try:
                tf_idx = methods.index("precision_residual_teacher_forced")
                ax.axhline(y=tf_idx - 0.5, color="#888888", linestyle="--", linewidth=0.8, alpha=0.7, zorder=5)
            except ValueError:
                pass
            ax.set_title(task, fontsize=9.5, fontweight="bold", pad=6)
            ax.set_xticks([0.0, 0.5, 1.0])
            ax.set_xticklabels(["0", "50", "100"], fontsize=7.5)
            # Inward tick alignment by index (not by string match):
            # 0 tucked left, 100 tucked right
            labels = ax.get_xticklabels()
            if len(labels) >= 3:
                labels[0].set_horizontalalignment("left")
                labels[-1].set_horizontalalignment("right")
            ax.set_yticks(range(len(method_labels)))
            if col == 0:
                ax.set_yticklabels(method_labels, fontsize=8.0)
            if col == 2:
                ax.set_xlabel("Episode Outcome Share (%)", fontsize=8.5)

        # Single top legend for active categories only
        legend_labels = [cat_display.get(c, c.replace("_", " ").title()) for c in ordered_cats]
        handles, _ = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=len(ordered_cats),
            frameon=False,
            fontsize=8.0,
        )
        # Inward 0%/100% alignment (ha=left/right) keeps labels inside spines; wspace 0.22 gives clean separation
        fig.subplots_adjust(top=0.86, bottom=0.14, left=0.22, right=0.97, wspace=0.22)
    return save(fig, "figures/appendix/A5_failure_categories")

"""A3 — training curves across all 5 tasks (validation action loss and expert balance loss)."""

from __future__ import annotations

from pathlib import Path

from studies.analysis.common import registry
from studies.analysis.common.style import method_color, paper_style
from studies.analysis.dataset import AnalysisDataset
from studies.analysis.render.figures import plot_seed_trajectories, save

PANELS = (
    ("val_loss_action", "Validation Action MSE"),
    ("train_loss_balance", "Expert Balance Loss"),
)
METHODS_TO_PLOT = (
    "precision_residual_phaseforge",
    "bc",
    "final_aligned_softmax_top1",
    "precision_residual_phase_random_router",
    "precision_residual_plain_encoder",
    "precision_residual_scratch_moe",
)


def generate(dataset: AnalysisDataset) -> list[Path]:
    import matplotlib.pyplot as plt

    tasks = registry.tasks()
    with paper_style():
        fig, axes = plt.subplots(
            len(tasks), len(PANELS), figsize=(7.2, 7.8), squeeze=False, sharex=True
        )
        for row, task in enumerate(tasks):
            for col, (field, title) in enumerate(PANELS):
                ax = axes[row][col]
                for method in METHODS_TO_PLOT:
                    if method not in registry.matrix_method_names():
                        continue
                    if field == "train_loss_balance" and method == "bc":
                        continue  # BC has no router or balance loss -> log(0) would fail
                    per_seed = []
                    for seed in registry.seeds("final"):
                        key = (task, method, seed, _final_stage(method))
                        if key in dataset.curves:
                            series = dataset.curves[key].series(field)
                            if series:
                                per_seed.append(series)
                    if not per_seed:
                        continue
                    plot_seed_trajectories(
                        ax,
                        per_seed,
                        method_color(method),
                        label=registry.display_name(method) if (row == 0 and col == 0) else None,
                        show_ribbon=True,
                    )

                # Log scale to resolve 10× (MSE) and 50× (balance) compression
                ax.set_yscale("log")
                ax.grid(True, which="both", linestyle=":", alpha=0.25, color="#CCCCCC")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                if row == 0:
                    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
                if col == 0:
                    ax.set_ylabel(task, fontsize=9.5, fontweight="bold", labelpad=8)
                    # Left col spans a narrow range on log scale: label a few
                    # plain-decimal majors and leave minors unlabeled. Labeling
                    # minors drew every 2..9 sub-tick as text, which collided
                    # with the majors in these short panels.
                    from matplotlib.ticker import (
                        FixedLocator,
                        NullFormatter,
                        ScalarFormatter,
                    )

                    _lo, _hi = ax.get_ylim()
                    _cands = (0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
                    _ticks = [t for t in _cands if _lo <= t <= _hi]
                    if len(_ticks) < 2:
                        _ticks = [
                            t
                            for t in (0.01, 0.1, 1.0, 10.0)
                            if _lo <= t <= _hi
                        ]
                    ax.yaxis.set_major_locator(FixedLocator(_ticks))
                    _fmt = ScalarFormatter()
                    _fmt.set_scientific(False)
                    _fmt.set_useOffset(False)
                    ax.yaxis.set_major_formatter(_fmt)
                    ax.yaxis.set_minor_formatter(NullFormatter())
                if row == len(tasks) - 1:
                    ax.set_xlabel("Epoch", fontsize=8.5)

        # Single outside legend
        h, l = axes[0][0].get_legend_handles_labels()
        fig.legend(
            h,
            l,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=3,
            frameon=False,
            fontsize=8,
        )
        fig.subplots_adjust(top=0.88, bottom=0.06, left=0.12, right=0.96, hspace=0.22, wspace=0.20)
    return save(fig, "figures/appendix/A3_training_curves")


def _final_stage(method: str) -> int:
    for m in registry.methods("final"):
        if m.name == method:
            return m.stages[-1]
    return 1

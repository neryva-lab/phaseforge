"""A11 — router-init family dynamics on Can and Square (NMI / entropy / switch rate / collapse)."""

from __future__ import annotations

from pathlib import Path

from studies.analysis.common import registry
from studies.analysis.common.style import method_color, paper_style
from studies.analysis.dataset import AnalysisDataset
from studies.analysis.render.figures import plot_seed_trajectories, save

FAMILY = (
    ("router_init_topology", "Topology Init (PF)"),
    ("router_init_phase", "Phase-Rule Init"),
    ("router_init_random", "Random Init"),
    ("routing_softmax_top1", "Softmax Top-1"),
    ("representation_bc", "BC Latent"),
)
FIELDS = (
    ("nmi", "Phase–Expert NMI"),
    ("routing_entropy", "Normalized Routing\nEntropy ($H / \\ln K$)"),
    ("switch_rate", "Switch Rate"),
    ("top1_collapse", "Top-1 Collapse"),
)
TASKS = ("Can", "Square")

FIELD_LIMITS = {
    "nmi": (0.05, 0.92),
    "routing_entropy": (0.935, 1.005),
    "switch_rate": (0.02, 0.125),
    "top1_collapse": (-0.005, 0.20),
}


def generate(dataset: AnalysisDataset) -> list[Path]:
    import matplotlib.pyplot as plt

    with paper_style():
        fig, axes = plt.subplots(
            len(FIELDS), len(TASKS), figsize=(7.2, 7.2), squeeze=False, sharex=True, sharey="row"
        )
        for col, task in enumerate(TASKS):
            for row, (field, ylabel) in enumerate(FIELDS):
                ax = axes[row][col]
                for name, display in FAMILY:
                    per_seed = []
                    for seed in registry.seeds("ablation"):
                        curve = dataset.curves.get((task, name, seed, 2))
                        if curve is not None:
                            series = curve.series(field)
                            if series:
                                per_seed.append(series)
                    if not per_seed:
                        continue
                    plot_seed_trajectories(
                        ax,
                        per_seed,
                        method_color(name),
                        label=display if (row == 0 and col == 0) else None,
                        show_ribbon=True,
                    )
                if field in FIELD_LIMITS:
                    ax.set_ylim(FIELD_LIMITS[field])
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                if col == 0:
                    ax.set_ylabel(ylabel, fontsize=8.5)
                ax.grid(True, linestyle=":", alpha=0.3)
                if row == len(FIELDS) - 1:
                    ax.set_xlabel("Stage-2 Epoch", fontsize=8.5)
                if row == 0:
                    ax.set_title(task, fontsize=9.5, fontweight="bold", pad=6)

        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=5,
            frameon=False,
            fontsize=8.0,
        )
        fig.subplots_adjust(top=0.91, bottom=0.07, left=0.14, right=0.96, hspace=0.22, wspace=0.14)
    return save(fig, "figures/appendix/A11_router_family")

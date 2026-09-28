"""F2 — per-task paired success deltas (PhaseForge vs key baselines/controls).

2x2 grid of forest plots (one panel per comparator); rows are tasks.
The solid point is the paired seed-mean delta on identical reset cases;
the horizontal bar spans the min-max seed deltas and the open seed
markers double as end-caps; individual seed deltas are exactly on the
line (no jitter) for clean visualization.
"""

from __future__ import annotations

import math
from pathlib import Path

from studies.analysis.common import registry
from studies.analysis.common.style import method_color, paper_style
from studies.analysis.dataset import AnalysisDataset
from studies.analysis.render.figures import save
from studies.analysis.stats.paired import pair_episodes

COMPARISONS = (
    "bc",
    "final_aligned_softmax_top1",
    "precision_residual_phase_random_router",
    "precision_residual_plain_encoder",
)

PANEL_TITLES = {
    "bc": "vs. BC",
    "final_aligned_softmax_top1": "vs. Softmax Top-1",
    "precision_residual_phase_random_router": "vs. Phase-Random",
    "precision_residual_plain_encoder": "vs. Plain Encoder",
}

MEAN_AREA, SEED_AREA = 15.0, 45.0
RANGE_LW, SEED_EDGE_LW = 2.5, 1.2
INK = "#444444"
LEFT, RIGHT, TOP, BOTTOM = 0.15, 0.96, 0.86, 0.15
HSPACE, WSPACE = 0.32, 0.12
X_CENTER = (LEFT + RIGHT) / 2.0


def _diameter(area: float) -> float:
    return 2.0 * math.sqrt(area / math.pi)


def _resolve_phaseforge_name(dataset: AnalysisDataset) -> str:
    full = "precision_residual_phaseforge"
    return full if any(k[1] == full for k in dataset.evals) else "phaseforge"


def _collect_deltas(dataset, tasks, seeds, pf_name, comparator):
    out = {}
    for task in tasks:
        deltas = []
        for seed in seeds:
            key_a = (task, pf_name, seed)
            key_b = (task, comparator, seed)
            if key_a not in dataset.episodes or key_b not in dataset.episodes:
                continue
            bank_a = dataset.evals[key_a].reset_bank
            bank_b = dataset.evals[key_b].reset_bank
            if bank_a != bank_b:
                continue  # pairing invalid across different banks
            deltas.append(
                pair_episodes(task, seed,
                              dataset.episodes[key_a], dataset.episodes[key_b],
                              bank_a=bank_a, bank_b=bank_b).delta)
        if deltas:
            out[task] = deltas
    return out


def generate(dataset: AnalysisDataset) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    tasks = list(reversed(registry.tasks()))   # y index ascends -> Lift on top
    seeds = tuple(registry.seeds("final"))
    pf_name = _resolve_phaseforge_name(dataset)

    stats = {c: _collect_deltas(dataset, tasks, seeds, pf_name, c) for c in COMPARISONS}

    extreme = 0.0
    for per_task in stats.values():
        for d in per_task.values():
            extreme = max(extreme, abs(min(d)), abs(max(d)))
    x_lim = max(0.30, math.ceil((extreme + 0.08) / 0.05) * 0.05)
    x_ticks = [t for t in (-0.4, -0.2, 0.0, 0.2, 0.4) if abs(t) <= x_lim - 0.05]

    y_positions = {task: i for i, task in enumerate(tasks)}

    with paper_style():
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.0), sharex=True, sharey=True)

        for idx, comparator in enumerate(COMPARISONS):
            ax = axes[idx // 2, idx % 2]
            color = method_color(comparator)

            ax.axvline(0, color="#888888", linestyle="--", linewidth=1.0, zorder=1)
            ax.set_axisbelow(True)

            for task, deltas in stats[comparator].items():
                y = y_positions[task]
                mean_delta = float(np.mean(deltas))
                ax.hlines(y, min(deltas), max(deltas), colors=color,
                          linewidth=RANGE_LW, zorder=2)
                ax.scatter(deltas, [y] * len(deltas), marker="o", s=SEED_AREA,
                           facecolors="white", edgecolors=color,
                           linewidth=SEED_EDGE_LW, zorder=3)
                ax.scatter([mean_delta], [y], marker="o", s=MEAN_AREA,
                           color=color, zorder=4)

            ax.set_title(PANEL_TITLES[comparator], fontsize=9.5, fontweight="bold", pad=6)
            ax.set_xlim(-x_lim, x_lim)
            ax.set_xticks(x_ticks)
            ax.yaxis.grid(True, linestyle=":", color="#d0d0d0", alpha=0.9)
            ax.xaxis.grid(True, linestyle=":", color="#d0d0d0", alpha=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        axes[0, 0].set_yticks(list(y_positions.values()))
        axes[0, 0].set_yticklabels(list(y_positions.keys()))

        fig.text(X_CENTER, 0.03,
                 r"Paired Difference: PhaseForge − Comparator $\Delta$ (Observed Seed Range)",
                 ha="center", fontsize=8.5, fontweight="bold")

        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=INK,
                   markersize=_diameter(MEAN_AREA), label="Seed mean paired Δ"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
                   markeredgecolor=INK, markeredgewidth=SEED_EDGE_LW,
                   markersize=_diameter(SEED_AREA),
                   label=f"Individual seed paired Δ (n={len(seeds)})"),
            Line2D([0], [0], color=INK, lw=RANGE_LW,
                   label="Observed seed range [min, max]"),
        ]
        fig.legend(handles=legend_elements, loc="upper center",
                   bbox_to_anchor=(X_CENTER, 0.99), ncol=3, frameon=False,
                   fontsize=8.0, handletextpad=0.4, columnspacing=1.5)

        fig.subplots_adjust(top=TOP, bottom=BOTTOM, left=LEFT, right=RIGHT,
                            hspace=HSPACE, wspace=WSPACE)

    return save(fig, "figures/main/F2_paired_deltas")
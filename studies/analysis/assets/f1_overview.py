"""F1 — PhaseForge architectural overview and methodology schematic.

Renders a publication-grade vector overview showing:
1. Stage 1: Phase Pretraining (State Encoder + Phase Head + Action Head)
2. Bootstrap Instant (t=0): Topological Changepoints, Prototype Router Init (K=6),
   Direct Expert Allocation
3. Stage 2: Top-1 MoE Policy (Hard Top-1 Router over 6 Specialized Experts)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt

from studies.analysis.common.style import OKABE_ITO, paper_style
from studies.analysis.dataset import AnalysisDataset
from studies.analysis.render.figures import save


def generate(dataset: AnalysisDataset | None = None) -> list[Path]:
    with paper_style():
        fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=300)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 50)
        ax.axis("off")

        # Color definitions
        c_stage1 = "#EBF4FA"
        c_stage1_border = OKABE_ITO["blue"]
        c_boot = "#FEF6E9"
        c_boot_border = OKABE_ITO["orange"]
        c_stage2 = "#F2F9F4"
        c_stage2_border = OKABE_ITO["green"]

        # Background Containers
        box_s1 = patches.FancyBboxPatch(
            (1, 2), 30, 45, boxstyle="round,pad=0.5,rounding_size=1.5",
            facecolor=c_stage1, edgecolor=c_stage1_border, linewidth=1.2, linestyle="--"
        )
        ax.add_patch(box_s1)
        ax.text(16, 44.5, "Stage 1: Phase Pretraining", ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=c_stage1_border)

        box_boot = patches.FancyBboxPatch(
            (34, 2), 29, 45, boxstyle="round,pad=0.5,rounding_size=1.5",
            facecolor=c_boot, edgecolor=c_boot_border, linewidth=1.2, linestyle="--"
        )
        ax.add_patch(box_boot)
        ax.text(48.5, 44.5, "Bootstrap Instant (t=0)", ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=c_boot_border)

        box_s2 = patches.FancyBboxPatch(
            (66, 2), 33, 45, boxstyle="round,pad=0.5,rounding_size=1.5",
            facecolor=c_stage2, edgecolor=c_stage2_border, linewidth=1.2, linestyle="--"
        )
        ax.add_patch(box_s2)
        ax.text(82.5, 44.5, "Stage 2: Top-1 MoE Policy", ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=c_stage2_border)

        # --- Stage 1 Components ---
        _draw_node(ax, 16, 39.0, 22, 4.5, r"State Input $\mathbf{s}_t \in \mathbb{R}^d$" + "\n(robot + object state)", "#FFFFFF", "#555555")
        _draw_node(ax, 16, 29.5, 22, 5.5, r"State Encoder $f_\theta$" + "\n(3-layer MLP + Residual)", "#D9EAF7", c_stage1_border)
        _draw_arrow(ax, (16, 36.5), (16, 32.5))

        _draw_node(ax, 16, 21.5, 18, 3.8, r"Latent $\mathbf{z}_t \in \mathbb{R}^{128}$", "#FFFFFF", c_stage1_border)
        _draw_arrow(ax, (16, 26.5), (16, 23.5))

        _draw_node(ax, 8.5, 11.5, 12.5, 6.5, r"Phase Head $g_\psi$" + "\n" + r"$\mathcal{L}_\mathrm{phase}$ (Cross-Ent)", "#FFFFFF", OKABE_ITO["vermillion"])
        _draw_node(ax, 23.5, 11.5, 12.5, 6.5, r"Action Head $h_\phi$" + "\n" + r"$\mathcal{L}_\mathrm{action}$ (MSE)", "#FFFFFF", OKABE_ITO["purple"])
        _draw_arrow(ax, (12.5, 19.5), (8.5, 15.0))
        _draw_arrow(ax, (19.5, 19.5), (23.5, 15.0))

        # --- Transition Arrow 1 -> Boot ---
        _draw_arrow(ax, (31.5, 25), (33.5, 25), lw=1.8, color="#555555")

        # --- Bootstrap Components ---
        _draw_node(ax, 48.5, 36, 25, 7.0, r"1. Topological Changepoints" + "\n" + r"PELT / Manifold clustering $\to \mathcal{D}_k$", "#FFFFFF", c_boot_border)

        _draw_node(ax, 48.5, 24, 25, 6.5, r"2. Prototype Router Init ($K=6$)" + "\n" + r"$\mathbf{c}_k = \frac{1}{|\mathcal{D}_k|} \sum_{i \in \mathcal{D}_k} f_\theta(\mathbf{s}_i)$", "#FFF2DE", OKABE_ITO["vermillion"])
        _draw_arrow(ax, (48.5, 32.2), (48.5, 27.5))

        _draw_node(ax, 48.5, 11.5, 25, 8.0, r"3. Direct Expert Allocation" + "\n" + r"Residual Experts ($\beta = 0.0$, inactive)" + "\n" + r"Initialized from action head $h_\phi$", "#FFF2DE", OKABE_ITO["purple"])
        _draw_arrow(ax, (48.5, 20.5), (48.5, 15.8))

        # --- Transition Arrow Boot -> 2 ---
        _draw_arrow(ax, (63.5, 25), (65.5, 25), lw=1.8, color="#555555")

        # --- Stage 2 Components ---
        _draw_node(ax, 82.5, 37.5, 26, 5.5, r"Encoder $\mathbf{z}_t = f_\theta(\mathbf{s}_t)$" + "\n" + r"(Trainable, $\mathrm{lr\_scale}=0.1$)", "#E2F0D9", c_stage2_border)

        _draw_node(ax, 74.0, 26.0, 14, 6.5, r"Hard Top-1 Router" + "\n" + r"$r_t = \arg\min_k \|\mathbf{z}_t - \mathbf{c}_k\|_2$", "#FFFFFF", OKABE_ITO["vermillion"])
        _draw_arrow(ax, (78.0, 34.5), (74.0, 29.5))

        _draw_node(ax, 91.0, 26.0, 14, 6.5, r"6 Specialized Experts" + "\n" + r"Direct Action $f_{r_t}(\mathbf{z}_t)$", "#FFFFFF", OKABE_ITO["purple"])
        _draw_arrow(ax, (87.0, 34.5), (91.0, 29.5))

        _draw_arrow(ax, (74.0, 22.5), (82.5, 17.5))
        _draw_arrow(ax, (91.0, 22.5), (82.5, 17.5))

        _draw_node(ax, 82.5, 11.5, 27, 8.0, r"Direct Action $\mathbf{a}_t = f_{r_t}(\mathbf{z}_t)$" + "\n" + r"$\mathcal{L}_\mathrm{total} = \mathcal{L}_\mathrm{action} + \lambda_\mathrm{bal}\mathcal{L}_\mathrm{bal} + \lambda_\mathrm{mar}\mathcal{L}_\mathrm{margin}$" + "\n(Subject to switch discontinuity)", "#E2F0D9", c_stage2_border)

        fig.tight_layout(pad=0.2)
    return save(fig, "figures/main/F1_overview")


def _draw_node(ax, cx, cy, w, h, text, facecolor, edgecolor):
    box = patches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.3,rounding_size=0.8",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=1.0, zorder=3
    )
    ax.add_patch(box)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=7.2, zorder=4, color="#111111")


def _draw_arrow(ax, start, end, lw=1.2, color="#333333"):
    ax.annotate(
        "", xy=end, xytext=start,
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, shrinkA=1, shrinkB=1),
        zorder=5
    )


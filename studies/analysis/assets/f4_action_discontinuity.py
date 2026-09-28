"""F4 — Commanded Action Discontinuity at Expert Switches.

Demonstrates the physical mechanism underlying the decoupling of routing organization
and closed-loop success:
Panel A: Two-task small-multiple layout of per-timestep action jumps ||Δa_t||_2 at
         expert switches vs. non-switch steps across ablation arms (Can and Square).
         Box-and-whisker summaries (median, IQR box, 5th-95th percentile whiskers)
         with overlaid per-seed means and switch-step sample counts.
Panel B: Expert switch rate across ablation arms (including BC as zero-switch reference)
         showing seed-level rings, descriptive mean (bullseye), and observed range.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from studies.analysis.common import io as cio
from studies.analysis.common import registry
from studies.analysis.common.style import OKABE_ITO, paper_style
from studies.analysis.dataset import AnalysisDataset
from studies.analysis.render.figures import save

METHODS_MOE = (
    "router_init_topology",
    "router_init_phase",
    "router_init_random",
    "routing_softmax_top1",
)
METHODS_ALL = METHODS_MOE + ("representation_bc",)
TASKS = ("Can", "Square")

INK = "#444444"          # neutral ink for per-seed rings (matches legend handle)
GREY = "#666666"         # annotation ink
SW_COLOR = OKABE_ITO["vermillion"]
NON_COLOR = OKABE_ITO["sky"]
TASK_COLORS = {"Can": "#333333", "Square": "#CC79A7"}
TASK_MARKERS = {"Can": "o", "Square": "s"}
TASK_OFFSETS = {"Can": -0.14, "Square": 0.14}

BOX_HALF = 0.128         # half box width  (width*0.4 with width=0.32)
CAP_HALF = 0.064         # half whisker-cap width
PAIR_OFFSET = 0.16       # ± width/2 for sw / non boxes
RING_A, RING_LW = 30.0, 1.0      # Panel B seed rings (larger ...)
MEAN_A, MEAN_LW = 14.0, 0.5      # ... than the solid mean -> visible bullseye
A_RING_A, A_RING_LW = 22.0, 1.1  # Panel A seed-mean rings (neutral ink)

# Text-size budget for Panel A (axis ≈114 pt wide, ≈36 pt per group slot):
#   count annotation "N=9999" at 6.2 pt  ≈ 22 pt  -> ≥9 pt gap between neighbours
#   tick names are vertical (90°)        -> cannot overlap by construction
COUNT_FS, NOTE_FS, TICK_FS = 6.2, 6.0, 6.8


def _extract_trace_metrics(dataset: AnalysisDataset):
    """Extract action jumps and switch rates directly from validated trace.jsonl files."""
    data = {
        task: {
            m: {
                "sw_jumps": [],
                "non_jumps": [],
                "seed_sw_means": [],
                "seed_non_means": [],
                "seed_sw_rates": [],
            }
            for m in METHODS_ALL
        }
        for task in TASKS
    }

    for task in TASKS:
        for m in METHODS_ALL:
            for s in registry.seeds("ablation"):
                key = (task, m, s)
                if key not in dataset.eval_runs:
                    continue
                tp = dataset.eval_runs[key].path / "trace.jsonl"
                if not tp.is_file():
                    continue

                cur_ep = None
                prev_act = None
                prev_exp = None
                s_sw_jumps = []
                s_non_jumps = []

                for rec in cio.iter_jsonl(tp):
                    ep = rec.get("episode_id")
                    exp = rec.get("selected_expert")
                    act = rec.get("final_action")
                    if act is None or len(act) == 0:
                        continue
                    act = np.array(act, dtype=float)

                    if ep != cur_ep:
                        cur_ep = ep
                        prev_act = act
                        prev_exp = exp
                        continue

                    if prev_act is not None and len(act) == len(prev_act):
                        jump = float(np.linalg.norm(act - prev_act))
                        if exp != prev_exp:
                            s_sw_jumps.append(jump)
                        else:
                            s_non_jumps.append(jump)

                    prev_act = act
                    prev_exp = exp

                total_steps = len(s_sw_jumps) + len(s_non_jumps)
                sw_rate = len(s_sw_jumps) / total_steps if total_steps > 0 else 0.0

                data[task][m]["seed_sw_rates"].append(sw_rate)
                data[task][m]["sw_jumps"].extend(s_sw_jumps)
                data[task][m]["non_jumps"].extend(s_non_jumps)
                if s_sw_jumps:
                    data[task][m]["seed_sw_means"].append(float(np.mean(s_sw_jumps)))
                if s_non_jumps:
                    data[task][m]["seed_non_means"].append(float(np.mean(s_non_jumps)))

    return data


def _box_stats(jumps):
    """IQR box + 5-95 whiskers; None when empty."""
    if not jumps:
        return None
    q1, med, q3 = np.percentile(jumps, [25, 50, 75])
    w_lo, w_hi = np.percentile(jumps, [5, 95])
    return float(q1), float(med), float(q3), float(w_lo), float(w_hi)


def generate(dataset: AnalysisDataset) -> list[Path]:
    trace_data = _extract_trace_metrics(dataset)
    n_seeds = len(registry.seeds("ablation"))
    short = {m: registry.display_name(m).replace(" Init", "").replace(" (PF)", "")
             for m in METHODS_ALL}

    # ---- pre-compute every glyph so limits are data-driven (no clipping) ----
    stats = {}
    y_max = 0.0
    for task in TASKS:
        for m in METHODS_MOE:
            sw = _box_stats(trace_data[task][m]["sw_jumps"])
            non = _box_stats(trace_data[task][m]["non_jumps"])
            stats[(task, m)] = (sw, non)
            for st in (sw, non):
                if st:
                    y_max = max(y_max, st[4])
            for v in trace_data[task][m]["seed_sw_means"] + trace_data[task][m]["seed_non_means"]:
                y_max = max(y_max, v)
    # +20% headroom so N=... row (axes y=0.985) clears 5-95% whiskers (≈0.88)
    y_top = max(0.26, np.ceil((y_max * 1.20) / 0.05) * 0.05)

    x_max = 0.0
    for task in TASKS:
        for m in METHODS_ALL:
            rates = trace_data[task][m]["seed_sw_rates"]
            if rates:
                x_max = max(x_max, max(rates))
    x_hi = np.ceil((x_max + 0.008) / 0.005) * 0.005

    with paper_style():
        fig = plt.figure(figsize=(7.2, 4.2))
        # Give Panels A1/A2 ~15% more physical width to relax 31pt→~36pt slot
        # without touching data scale — pure typographic budget, not x-axis rescaling.
        gs = fig.add_gridspec(1, 3, width_ratios=[1.24, 1.24, 1.08], wspace=0.44)
        ax_can = fig.add_subplot(gs[0])
        ax_sq = fig.add_subplot(gs[1], sharey=ax_can)
        ax_b = fig.add_subplot(gs[2])

        # ---------------- PANEL A: small-multiple box plots ----------------
        for ax, task, title in (
            (ax_can, "Can", "Panel A1: Can Jumps"),
            (ax_sq, "Square", "Panel A2: Square Jumps"),
        ):
            positions = np.arange(len(METHODS_MOE))

            for i, m in enumerate(METHODS_MOE):
                sw, non = stats[(task, m)]
                groups = ((sw, positions[i] + PAIR_OFFSET, SW_COLOR),
                          (non, positions[i] - PAIR_OFFSET, NON_COLOR))
                means = (trace_data[task][m]["seed_sw_means"],
                         trace_data[task][m]["seed_non_means"])

                for (st, pos, color), mu in zip(groups, means):
                    if st is None:
                        continue
                    q1, med, q3, w_lo, w_hi = st
                    # true closed box: RGBA face (alpha 0.30) + solid edge
                    ax.add_patch(Rectangle(
                        (pos - BOX_HALF, q1), 2 * BOX_HALF, max(q3 - q1, 1e-6),
                        facecolor=to_rgba(color, 0.30), edgecolor=color,
                        linewidth=1.0, zorder=2))
                    ax.plot([pos - BOX_HALF, pos + BOX_HALF], [med, med],
                            color=color, lw=1.6, zorder=3)
                    ax.plot([pos, pos], [q1, w_lo], color=color, lw=1.0, zorder=2)
                    ax.plot([pos, pos], [q3, w_hi], color=color, lw=1.0, zorder=2)
                    ax.plot([pos - CAP_HALF, pos + CAP_HALF], [w_lo, w_lo],
                            color=color, lw=1.0, zorder=2)
                    ax.plot([pos - CAP_HALF, pos + CAP_HALF], [w_hi, w_hi],
                            color=color, lw=1.0, zorder=2)

                # per-seed means: neutral ink rings (exact match to legend handle)
                for pos, mu in ((positions[i] + PAIR_OFFSET, means[0]),
                                (positions[i] - PAIR_OFFSET, means[1])):
                    if mu:
                        ax.scatter([pos] * len(mu), mu, facecolors="none",
                                   edgecolors=INK, s=A_RING_A, linewidths=A_RING_LW,
                                   zorder=5)

                # sample count: short string ("N=1948", ~22 pt) so the four
                # group-centred annotations cannot collide (slot ≈36 pt).
                n_sw = len(trace_data[task][m]["sw_jumps"])
                ax.text(positions[i], 0.985, f"N={n_sw}", fontsize=COUNT_FS,
                        color=GREY, ha="center", va="top",
                        transform=ax.get_xaxis_transform(), zorder=6)

            # one compact key for the count row, top-right, clear of every
            # whisker cap on the right half and of the count row above it
            ax.text(0.99, 0.93, "N = switch steps", fontsize=NOTE_FS, color=GREY,
                    ha="right", va="top", transform=ax.transAxes, zorder=6)

            ax.set_xticks(positions)
            # Vertical labels: group slots (~36pt) are narrower than the full
            # terms ("Trajectory-Derived" ~65pt), so any angled setting
            # collides across slots; vertical text cannot overlap by
            # construction. (Panel B keeps its wrapped horizontal y-labels.)
            ax.set_xticklabels([short[m] for m in METHODS_MOE], rotation=90,
                               ha="center", fontsize=TICK_FS)
            # Give N=... row (y=0.985 in axes coords, x=positions[0]=0) breathing room
            # from the y-axis spine — without this the first count straddles the spine.
            ax.set_xlim(-0.62, 3.62)
            ax.set_title(title, fontsize=9.0, fontweight="bold", pad=6)
            ax.grid(axis="y", linestyle=":", alpha=0.35)
            ax.set_axisbelow(True)

        ax_can.set_ylim(0.0, y_top)
        ax_can.set_ylabel(r"Commanded Action Jump $\|\Delta \mathbf{a}_t\|_2$", fontsize=8.5)
        ax_sq.tick_params(labelleft=False)

        # ---------------- PANEL B: expert switch rate ----------------
        y_pos = np.arange(len(METHODS_ALL))
        for task in TASKS:
            c = TASK_COLORS[task]
            mkr = TASK_MARKERS[task]
            for i, m in enumerate(METHODS_ALL):
                rates = trace_data[task][m]["seed_sw_rates"]
                if not rates:
                    continue
                y = y_pos[i] + TASK_OFFSETS[task]
                mean_r, min_r, max_r = float(np.mean(rates)), min(rates), max(rates)
                if max_r > min_r:
                    ax_b.plot([min_r, max_r], [y, y], color=c, lw=1.5, zorder=3)
                ax_b.scatter([mean_r], [y], color=c, s=MEAN_A, zorder=4,
                             edgecolor="white", linewidth=MEAN_LW, marker=mkr)
                ax_b.scatter(rates, [y] * len(rates), facecolors="none", edgecolors=c,
                             s=RING_A, linewidths=RING_LW, zorder=5, marker=mkr)

        ax_b.axvline(0.0, color="#888888", linestyle="--", linewidth=0.8, zorder=1)
        ax_b.set_yticks(y_pos)
        # Wrap longest label to avoid left overflow — keeps "Softmax Top-1" inside wspace
        b_labels = []
        for m in METHODS_ALL:
            lab = short[m]
            if lab == "Softmax Top-1":
                lab = "Softmax\nTop-1"
            b_labels.append(lab)
        ax_b.set_yticklabels(b_labels, fontsize=7.5, va="center")
        ax_b.invert_yaxis()
        ax_b.set_ylim(len(METHODS_ALL) - 0.45, -0.45)
        ax_b.set_xlim(-0.006, x_hi)
        ax_b.set_xlabel("Routing Switch Rate", fontsize=8.5)
        ax_b.set_title("Panel B: Switch Rate", fontsize=9.0, fontweight="bold", pad=6)
        ax_b.grid(axis="x", linestyle=":", alpha=0.35)
        ax_b.set_axisbelow(True)

        # ---------------- legend (matplotlib fills columns first!) --------
        # order [sw, Can, non, Square, ring] with ncol=3 ->
        #   row 1: switch box | non-switch box | seed ring   (Panel A semantics)
        #   row 2: Can mean+range+seeds | Square mean+range+seeds (Panel B semantics)
        # Can/Square now use charcoal vs reddish-purple + circle vs square
        # so they cannot be mistaken for Panel A's vermillion/sky switch encoding.
        legend_handles = [
            Patch(facecolor=to_rgba(SW_COLOR, 0.30), edgecolor=SW_COLOR, linewidth=1.0,
                  label="Switch Step Jump (IQR, 5-95%)"),
            Line2D([0], [0], marker=TASK_MARKERS["Can"], color=TASK_COLORS["Can"], lw=1.5,
                   markerfacecolor=TASK_COLORS["Can"], markeredgecolor=TASK_COLORS["Can"], markersize=4.5,
                   label="Can (mean + range; ○ seed, n=3)"),
            Patch(facecolor=to_rgba(NON_COLOR, 0.30), edgecolor=NON_COLOR, linewidth=1.0,
                  label="Non-Switch Step Jump (IQR, 5-95%)"),
            Line2D([0], [0], marker=TASK_MARKERS["Square"], color=TASK_COLORS["Square"], lw=1.5,
                   markerfacecolor=TASK_COLORS["Square"], markeredgecolor=TASK_COLORS["Square"], markersize=4.5,
                   label="Square (mean + range; ○/□ seed, n=3)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                   markeredgecolor=INK, markeredgewidth=A_RING_LW, markersize=5.0,
                    label=f"Seed mean jump (Panels A1/A2, n={n_seeds})"),
        ]
        fig.legend(handles=legend_handles, loc="upper center",
                   bbox_to_anchor=(0.5, 0.995), ncol=3, frameon=False,
                   fontsize=7.2, handletextpad=0.5, columnspacing=1.6)

        fig.subplots_adjust(top=0.86, bottom=0.20, left=0.09, right=0.97)

    return save(fig, "figures/main/F4_action_discontinuity")
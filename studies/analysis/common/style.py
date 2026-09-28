"""Publication style contract (figures_tables_plan.md section 5).

Okabe-Ito colorblind-safe palette, one fixed method->color mapping across
every figure, PhaseForge accent-first ordering, NeurIPS column widths,
vector PDF + 300 DPI PNG exports with embedded fonts.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # analysis runs headless; set before pyplot import
import matplotlib as mpl
import matplotlib.pyplot as plt

from studies.analysis.common.config import get_config

# Okabe-Ito colorblind-safe palette.
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#999999",
    "dark_yellow": "#a08600",  # legible yellow substitute for lines on white
}

# Fixed method -> color mapping: PhaseForge carries the accent; controls are
# muted; the BC family shares cool neutrals; ablation groups have distinct palettes.
METHOD_COLORS: dict[str, str] = {
    "precision_residual_phaseforge": OKABE_ITO["vermillion"],
    "phaseforge": OKABE_ITO["vermillion"],
    "bc": OKABE_ITO["orange"],
    "final_aligned_bc": OKABE_ITO["orange"],
    "final_aligned_softmax_top1": OKABE_ITO["sky"],
    "precision_residual_phase_random_router": OKABE_ITO["green"],
    "precision_residual_plain_encoder": OKABE_ITO["blue"],
    "precision_residual_scratch_moe": OKABE_ITO["purple"],
    "final_aligned_static_rule": OKABE_ITO["dark_yellow"],
    "precision_residual_factorial_floor": OKABE_ITO["grey"],
    "precision_residual_teacher_forced": OKABE_ITO["black"],
    "precision_residual_oracle": OKABE_ITO["grey"],
    # Router initialization ablation (Can & Square)
    "router_init_topology": OKABE_ITO["vermillion"],
    "router_init_phase": OKABE_ITO["purple"],
    "router_init_random": OKABE_ITO["green"],
    "representation_bc": OKABE_ITO["orange"],
    "routing_softmax_top1": OKABE_ITO["sky"],
}
ABLATION_COLOR = OKABE_ITO["grey"]
SEED_POINT_COLOR = OKABE_ITO["black"]

FONT_SIZES = {"title": 10, "label": 9, "tick": 8, "legend": 8, "annot": 7.5}


def method_color(name: str) -> str:
    return METHOD_COLORS.get(name, ABLATION_COLOR)


METHOD_MARKERS: dict[str, str] = {
    "precision_residual_phaseforge": "o",
    "phaseforge": "o",
    "bc": "s",
    "final_aligned_bc": "s",
    "final_aligned_softmax_top1": "^",
    "precision_residual_phase_random_router": "v",
    "precision_residual_plain_encoder": "D",
    "precision_residual_scratch_moe": "P",
    "final_aligned_static_rule": "X",
    "precision_residual_factorial_floor": "p",
    "precision_residual_teacher_forced": "*",
    "precision_residual_oracle": "+",
    # Ablation
    "router_init_topology": "o",
    "router_init_phase": "s",
    "router_init_random": "v",
    "representation_bc": "D",
    "routing_softmax_top1": "^",
}


def method_marker(name: str) -> str:
    return METHOD_MARKERS.get(name, "o")


def column_width(kind: str = "text") -> float:
    cfg = get_config().style
    if kind == "text":
        return float(cfg.text_width_in)
    if kind == "margin":
        return float(cfg.margin_width_in)
    raise ValueError(f"Unknown column kind {kind!r}")


@contextmanager
def paper_style() -> Iterator[None]:
    """Apply the publication rcParams for the enclosed figure block."""
    with mpl.rc_context(
        {
            "figure.dpi": 100,
            "savefig.dpi": int(get_config().style.dpi),
            "font.size": FONT_SIZES["tick"],
            "axes.titlesize": FONT_SIZES["title"],
            "axes.labelsize": FONT_SIZES["label"],
            "xtick.labelsize": FONT_SIZES["tick"],
            "ytick.labelsize": FONT_SIZES["tick"],
            "legend.fontsize": FONT_SIZES["legend"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    ):
        yield


def save_figure(fig: plt.Figure, out_base: Path) -> list[Path]:
    """Save one figure as vector PDF + raster PNG next to out_base (no suffix)."""
    out_base.parent.mkdir(parents=True, exist_ok=True)
    pdf = out_base.with_suffix(".pdf")
    png = out_base.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    return [pdf, png]

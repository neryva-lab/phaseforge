"""A4 — full ablation table (controlled router-initialization ablation on Can & Square)."""

from __future__ import annotations

from pathlib import Path

from studies.analysis.common import registry
from studies.analysis.dataset import AnalysisDataset
from studies.analysis.render.tables import Table, save_table


def _cell_summary(dataset: AnalysisDataset, task: str, name: str) -> list[str]:
    rates, nmis, collapses, switches = [], [], [], []
    seeds = sorted(registry.seeds("ablation"))
    for seed in seeds:
        ev = dataset.evals.get((task, name, seed))
        if ev is not None:
            rates.append(ev.success_rate)
        curve = dataset.curves.get((task, name, seed, 2))
        if curve is not None:
            nmi = curve.last("nmi")
            col = curve.last("top1_collapse")
            sw = curve.last("switch_rate")
            if nmi is not None:
                nmis.append(nmi)
            if col is not None:
                collapses.append(col)
            if sw is not None:
                switches.append(sw)
    sr = f"{sum(rates) / len(rates):.2f}" if rates else "--"
    seeds_str = ", ".join(f"{r:.2f}" for r in rates) if rates else "--"
    nmi = f"{sum(nmis) / len(nmis):.3f}" if nmis else "--"
    sw = f"{sum(switches) / len(switches):.3f}" if switches else "--"
    col = f"{sum(collapses) / len(collapses):.2f}" if collapses else "--"
    return [sr, seeds_str, nmi, sw, col]


def generate(dataset: AnalysisDataset) -> list[Path]:
    rows = []
    for method in registry.methods("ablation"):
        if not method.evaluate:
            continue
        name = method.name
        task = method.task or "Can"
        display = registry.display_name(name)
        if "topology" in name:
            display = rf"\textbf{{{display}}}"
        # Display layer only: render the manifest's "topology" role vocabulary with
        # the paper's canonical "trajectory-derived" term. Manifests stay frozen.
        role = (method.role or "ablation").replace("topology", "trajectory-derived")
        rows.append(
            [task, display, role]
            + _cell_summary(dataset, task, name)
        )
    table = Table(
        headers=[
            "Task",
            "Method",
            "Role",
            "SR (mean)",
            "SR (per seed)",
            "NMI final",
            "Switch rate",
            "Collapse",
        ],
        rows=rows,
        caption="Full router-initialization ablation suite across Can and Square "
        f"(seeds {list(registry.seeds('ablation'))}).",
        notes=(
            "All Stage~2 runs use identical phase-aware Stage~1 representations (except BC latent) "
            "and direct-action beta=0 residual experts with margin loss disabled.",
        ),
    )
    return save_table(table, "tables/A4_ablation_full")

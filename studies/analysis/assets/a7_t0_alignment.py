"""A7 — t=0 routing alignment across router initializations (H1 initial condition)."""

from __future__ import annotations

from pathlib import Path

from studies.analysis.common import registry
from studies.analysis.dataset import AnalysisDataset
from studies.analysis.render.tables import Table, save_table

ARMS = (
    ("router_init_topology", "Topology Init (PF)"),
    ("router_init_phase", "Phase-Rule Init"),
    ("router_init_random", "Random Init"),
    ("representation_bc", "BC Latent"),
    ("routing_softmax_top1", "Softmax Top-1"),
)
TASKS = ("Can", "Square")


def generate(dataset: AnalysisDataset) -> list[Path]:
    rows = []
    seeds = registry.seeds("ablation")
    first_seed = seeds[0] if seeds else 42
    for task in TASKS:
        for name, display in ARMS:
            init = dataset.init_routing.get((task, name, first_seed, 2))
            if init is None:
                continue
            freqs = init.t0_top1_expert_frequencies
            dead = init.t0_dead_expert_count
            top_share = max(freqs) if freqs else float("nan")
            rows.append(
                [
                    task,
                    display,
                    f"{init.t0_nmi:.3f}" if init.t0_nmi is not None else "--",
                    f"{init.t0_routing_entropy:.3f}" if init.t0_routing_entropy is not None else "--",
                    f"{init.t0_normalized_routing_entropy:.3f}"
                    if init.t0_normalized_routing_entropy is not None
                    else "--",
                    f"{top_share:.3f}" if top_share == top_share else "--",
                    str(dead) if dead is not None else "--",
                ]
            )
    table = Table(
        headers=[
            "Task",
            "Router init",
            "t=0 NMI",
            "Entropy",
            "Norm. entropy",
            "Max top-1 share",
            "Dead experts",
        ],
        rows=rows,
        caption="Bootstrap-instant routing diagnostics per router initialization on Can and Square "
        "(metadata/init\\_routing.json; seed 42).",
        notes=(
            "Topological initialization starts highly structured and phase-aligned (high t=0 NMI, "
            "zero dead experts); random prototype initialization starts with dead experts.",
        ),
    )
    return save_table(table, "tables/A7_t0_alignment")

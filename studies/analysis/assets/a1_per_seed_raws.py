"""A1 — per-seed raw success rates for every matrix cell."""

from __future__ import annotations

from pathlib import Path

from studies.analysis.common import registry
from studies.analysis.dataset import AnalysisDataset
from studies.analysis.render.tables import Table, save_table


def generate(dataset: AnalysisDataset) -> list[Path]:
    seeds = sorted(registry.seeds("final"))
    headers = ["Task", "Method"] + [f"seed {s}" for s in seeds] + ["Mean"]
    rows = []
    for task in registry.tasks():
        for method in registry.matrix_method_names():
            cells, rates = [], []
            for seed in seeds:
                ev = dataset.evals.get((task, method, seed))
                if ev is None:
                    cells.append("--")
                    continue
                cells.append(f"{ev.success_rate:.2f} ({ev.successes}/{ev.valid_episodes})")
                rates.append(ev.success_rate)
            mean = f"{sum(rates) / len(rates):.2f}" if rates else "--"
            rows.append([task, registry.display_name(method)] + cells + [mean])
    table = Table(
        headers=headers,
        rows=rows,
        caption="Per-seed rollout success rates for every five-task matrix cell.",
    )
    return save_table(table, "tables/A1_per_seed_raws")

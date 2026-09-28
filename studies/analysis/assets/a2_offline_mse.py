"""A2 — offline action MSE matrix (eval-time MSE; training val loss secondary)."""

from __future__ import annotations

from pathlib import Path

from studies.analysis.common import registry
from studies.analysis.dataset import AnalysisDataset
from studies.analysis.render.tables import Table, save_table


def generate(dataset: AnalysisDataset) -> list[Path]:
    tasks = registry.tasks()
    headers = ["Method"] + [f"{t} (eval)" for t in tasks] + [f"{t} (val)" for t in tasks]
    rows = []
    for method in registry.matrix_method_names():
        eval_cells, val_cells = [], []
        for task in tasks:
            mses = [
                dataset.evals[(task, method, s)].action_mse
                for s in registry.seeds("final")
                if (task, method, s) in dataset.evals
                and dataset.evals[(task, method, s)].action_mse is not None
            ]
            eval_cells.append(f"{sum(mses) / len(mses):.4f}" if mses else "--")
            vals = []
            for s in registry.seeds("final"):
                curve = dataset.curves.get((task, method, s, _final_stage(method)))
                if curve is not None:
                    v = curve.last("val_loss_action")
                    if v is not None:
                        vals.append(v)
            val_cells.append(f"{sum(vals) / len(vals):.4f}" if vals else "--")
        rows.append([registry.display_name(method)] + eval_cells + val_cells)
    table = Table(
        headers=headers,
        rows=rows,
        caption="Offline action MSE (eval-time primary; final-epoch validation loss secondary).",
        notes=(
            "Offline metrics are diagnostics only — they never substitute for rollout "
            "success (research\\_definition §4).",
        ),
    )
    return save_table(table, "tables/A2_offline_mse")


def _final_stage(method: str) -> int:
    for m in registry.methods("final"):
        if m.name == method:
            return m.stages[-1]
    return 1

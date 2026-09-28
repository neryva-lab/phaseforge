"""A9 — compute cost: wall-clock per phase, throughput, peak memory."""

from __future__ import annotations

from pathlib import Path

from studies.analysis.common import io as cio
from studies.analysis.common import registry
from studies.analysis.dataset import AnalysisDataset
from studies.analysis.render.tables import Table, save_table


def _avg(values: list[float]) -> str:
    return f"{sum(values) / len(values):.1f}" if values else "--"


def generate(dataset: AnalysisDataset) -> list[Path]:
    rows = []
    for method in registry.matrix_method_names():
        wall_s1, wall_s2, wall_eval, sps, mem = [], [], [], [], []
        for task in registry.tasks():
            for seed in registry.seeds("final"):
                t1 = dataset.timings.get((task, method, seed, 1))
                if t1 is not None and "wall_seconds" in t1.raw:
                    wall_s1.append(float(t1.raw["wall_seconds"]))
                t2 = dataset.timings.get((task, method, seed, 2))
                if t2 is not None and "wall_seconds" in t2.raw:
                    wall_s2.append(float(t2.raw["wall_seconds"]))
                ev_run = dataset.eval_runs.get((task, method, seed))
                if ev_run is not None:
                    ev_timings = ev_run.path / "timings.json"
                    if ev_timings.is_file():
                        raw = cio.read_json(ev_timings)
                        if "wall_seconds" in raw:
                            wall_eval.append(float(raw["wall_seconds"]))
                curve = dataset.curves.get((task, method, seed, _stage(method)))
                if curve is not None:
                    sps += [
                        p.steps_per_second for p in curve.points if p.steps_per_second is not None
                    ]
                    mem += [
                        p.peak_gpu_memory_mb
                        for p in curve.points
                        if p.peak_gpu_memory_mb is not None
                    ]
        rows.append(
            [
                registry.display_name(method),
                _avg(wall_s1),
                _avg(wall_s2),
                _avg(sps),
                _avg(mem),
            ]
        )
    table = Table(
        headers=["Method", "Stage~1 wall (s)", "Stage~2 wall (s)", "Steps/s", "Peak GPU MB"],
        rows=rows,
        caption="Compute cost per cell (mean across tasks and seeds; timings.json + "
        "training\\_curves efficiency fields).",
    )
    return save_table(table, "tables/A9_compute_cost")


def _stage(method: str) -> int:
    for m in registry.methods("final"):
        if m.name == method:
            return m.stages[-1]
    return 1

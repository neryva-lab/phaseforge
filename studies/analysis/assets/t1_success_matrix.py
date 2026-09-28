"""T1 — five-task success matrix (the result table).

Mean SR over seeds with the pooled Wilson interval; macro-average column
marked secondary; PhaseForge row bolded via LaTeX markup; paired Δ vs BC
reported in a footer band (full tests in A15).
"""

from __future__ import annotations

from pathlib import Path

from studies.analysis.common import registry
from studies.analysis.dataset import AnalysisDataset
from studies.analysis.render.tables import Table, save_table
from studies.analysis.stats.intervals import mean, sample_std, seed_mean_and_wilson


def generate(dataset: AnalysisDataset) -> list[Path]:
    tasks = list(registry.tasks())
    headers = ["Method"] + tasks + ["Macro-avg"]
    deployable_rows: list[list[str]] = []
    diagnostic_rows: list[list[str]] = []

    # Exclude offline oracle entirely; isolate privileged teacher-forced diagnostic
    all_methods = [
        m for m in registry.matrix_method_names()
        if m != "precision_residual_oracle"
    ]

    for method in all_methods:
        is_diagnostic = (method == "precision_residual_teacher_forced")
        cells: list[str] = []
        all_rates: list[float] = []
        total_denom = 0
        for task in tasks:
            per_seed_successes, per_seed_rates = [], []
            task_denom = 0
            for seed in registry.seeds("final"):
                key = (task, method, seed)
                if key not in dataset.evals:
                    continue
                ev = dataset.evals[key]
                per_seed_successes.append(ev.successes)
                per_seed_rates.append(ev.success_rate)
                task_denom += ev.valid_episodes

            if not per_seed_rates or task_denom == 0:
                cells.append("--")
                continue

            first_ev = dataset.evals.get((task, method, registry.seeds("final")[0]))
            ep_per_seed = first_ev.valid_episodes if first_ev else 50
            p, lo, hi = seed_mean_and_wilson(
                per_seed_successes,
                ep_per_seed,
            )
            all_rates.extend(per_seed_rates)
            total_denom += task_denom
            cells.append(f"{p:.2f} [{lo:.2f}, {hi:.2f}]")

        display = registry.display_name(method)
        if method in ("precision_residual_phaseforge", "phaseforge"):
            display = r"\textbf{PhaseForge}"

        if is_diagnostic:
            # Excluded from deployable macro-average; clearly marked as non-deployable diagnostic
            display = f"{display} (Privileged Diagnostic)"
            diagnostic_rows.append([display] + cells + ["-- [Diagnostic]"])
        else:
            macro = f"{mean(all_rates):.2f} ± {sample_std(all_rates):.2f}" if all_rates else "--"
            deployable_rows.append([display] + cells + [macro])

    rows = deployable_rows + diagnostic_rows

    table = Table(
        headers=headers,
        rows=rows,
        caption="Rollout success rate per task (mean over seeds; brackets: pooled Wilson 95\\% "
        "CI; macro-average ± sample std across deployable task-seeds, secondary aggregate; "
        "N=300 per cell for Can/Square (six seeds 42--47); N=150 per cell for "
        "Lift/ToolHang/Transport (three seeds 42--44)).",
        notes=(
            "Offline oracle is excluded from rollout success evaluation. Teacher-Forced is a "
            "privileged label-and-dispatch-mismatch diagnostic, excluded from "
            "deployable rankings and macro-average. Paired Δ vs BC with Holm-adjusted tests: asset A15.",
        ),
    )
    return save_table(table, "tables/T1_success_matrix")

"""A15 — paired statistical tests (Wilcoxon signed-rank + Holm-adjusted p-values).

Reads the sweep's own ``_summaries/paired_wilcoxon.csv`` when present; when
absent, computes paired per-task seed-mean deltas from episodes and applies
the Holm adjustment over the declared primary family (PhaseForge vs BC /
Warm-Start MoE / H1 / H2 controls, per task).
"""

from __future__ import annotations

from pathlib import Path

from studies.analysis.common import registry
from studies.analysis.dataset import AnalysisDataset
from studies.analysis.render.tables import Table, save_table
from studies.analysis.stats.intervals import sample_std
from studies.analysis.stats.multiplicity import holm_adjust
from studies.analysis.stats.paired import pair_episodes

PRIMARY_COMPARATORS = (
    "bc",
    "final_aligned_softmax_top1",
    "precision_residual_plain_encoder",
    "precision_residual_phase_random_router",
    "precision_residual_scratch_moe",
    "final_aligned_static_rule",
)


def generate(dataset: AnalysisDataset) -> list[Path]:
    wilcoxon = dataset.paired_wilcoxon.get("final")
    if wilcoxon is not None and wilcoxon.rows:
        rows = [
            [
                r.get("task", "--"),
                r.get("method_a", "--"),
                r.get("method_b", "--"),
                r.get("statistic", "--"),
                r.get("p_value", "--"),
                r.get("p_adjusted", "--") if "p_adjusted" in r else "--",
            ]
            for r in wilcoxon.rows
        ]
        table = Table(
            headers=["Task", "A", "B", "W", "p", "p (Holm)"],
            rows=rows,
            caption="Paired Wilcoxon signed-rank tests (from the sweep's "
            "\\_summaries/paired\\_wilcoxon.csv).",
        )
        return save_table(table, "tables/A15_paired_tests")

    # Fallback: seed-mean paired deltas + Holm over the primary family.
    entries: list[tuple[str, str, list[float]]] = []
    pf_name = "precision_residual_phaseforge"
    for comparator in PRIMARY_COMPARATORS:
        for task in registry.tasks():
            deltas = []
            for seed in registry.seeds("final"):
                key_a = (task, pf_name, seed) if (task, pf_name, seed) in dataset.episodes else (task, "phaseforge", seed)
                key_b = (task, comparator, seed)
                if key_a not in dataset.episodes or key_b not in dataset.episodes:
                    continue
                bank_a = dataset.evals[key_a].reset_bank
                bank_b = dataset.evals[key_b].reset_bank
                if bank_a != bank_b:
                    continue
                deltas.append(
                    pair_episodes(
                        task, seed, dataset.episodes[key_a], dataset.episodes[key_b]
                    ).delta
                )
            if deltas:
                entries.append((task, comparator, deltas))

    # Exact two-sided binomial sign test on seed means, Holm-adjusted.
    from math import comb

    def sign_p(deltas: list[float]) -> float:
        plus = sum(1 for d in deltas if d > 0)
        minus = sum(1 for d in deltas if d < 0)
        n = plus + minus
        if n == 0:
            return 1.0
        # Two-sided exact sign test: 2 * min(tail)
        tail_plus = sum(comb(n, k) for k in range(plus, n + 1)) / 2**n
        tail_minus = sum(comb(n, k) for k in range(0, plus + 1)) / 2**n
        # For plus > n/2 tail_plus is small, else tail_minus is small; take min
        # Equivalent to 2* min, capped at 1.0
        p = 2 * min(tail_plus, tail_minus)
        return min(1.0, p)

    ps = [sign_p(d) for _, _, d in entries]
    adjusted = holm_adjust(ps) if ps else []
    rows = []
    for (task, comparator, deltas), p, adj in zip(entries, ps, adjusted):
        rows.append(
            [
                task,
                "PhaseForge",
                registry.display_name(comparator),
                f"{sum(deltas) / len(deltas):+.3f}",
                f"{sample_std(deltas):.3f}",
                f"{p:.3f}",
                f"{adj:.3f}",
            ]
        )
    table = Table(
        headers=["Task", "A", "B", "Mean Δ", "Δ std (seeds)", "p (sign)", "p (Holm)"],
        rows=rows,
        caption="Paired per-episode deltas (identical reset cases) with Holm-adjusted "
        "exact sign tests over the pre-declared primary family.",
        notes=(
            "Can/Square rows use six seeds (42--47); all other tasks use three "
            "seeds (42--44). These comparisons are descriptive; no population-level "
            "claim is permitted (research\\_definition §4). Generate "
            "\\_summaries/paired\\_wilcoxon.csv for the full Wilcoxon variant.",
        ),
    )
    return save_table(table, "tables/A15_paired_tests")

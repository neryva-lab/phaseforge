"""Rollout report builder.

Aggregates every completed rollout eval run under the outputs base into the
paper tables:

* ``outputs/_results/rollout_success.csv`` — per ``(task, model, tag,
  training_seed)``: valid/success counts, success rate, Wilson CI,
  policy-failure counts, invalid attempts (per :func:`summarize_episodes`).
* ``outputs/_results/rollout_comparisons.csv`` — per ``(task,
  training_seed)``: paired rate differences, Newcombe CI, exact McNemar
  p-values over case-level discordant pairs, plus Holm-Bonferroni adjusted
  p-values across comparisons per seed.
* ``outputs/_results/rollout_report.json`` — the same data as JSON.

Only runs with a ``rollout_summary.json`` (i.e. real rollout runs, not
offline eval runs) are included; runs without it are reported as skipped.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from phaseforge.outputs_writer.episodes import (
    holm_bonferroni,
    paired_rollout_comparisons,
    read_episode_records,
    summarize_episodes,
)

SUCCESS_CSV = "rollout_success.csv"
COMPARISONS_CSV = "rollout_comparisons.csv"
REPORT_JSON = "rollout_report.json"

# Only matched, deployable comparators belong to the primary five-way
# hypothesis family. Privileged routing and robot-only controls remain in the
# report, but their p-values are descriptive and are not included in Holm.
PRIMARY_COMPARATORS = frozenset(
    {
        "bc",
        "precision_residual_plain_encoder",
        "precision_residual_phase_random_router",
        "precision_residual_scratch_moe",
    }
)


def _scan_rollout_runs(outputs_base: Path) -> list[dict[str, Any]]:
    """Collect ``(run_dir, meta)`` for every rollout eval run under the base."""
    runs: list[dict[str, Any]] = []
    eval_root = outputs_base / "eval"
    if not eval_root.is_dir():
        return runs
    for model_dir in sorted(eval_root.iterdir()):
        if not model_dir.is_dir():
            continue
        for seed_dir in sorted(model_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            for run_dir in sorted(seed_dir.iterdir()):
                if not (run_dir / "rollout_summary.json").is_file():
                    continue
                if not (run_dir / "episodes.jsonl").is_file():
                    continue
                # A summary can exist after a crash that happened before the
                # lifecycle marker was written. Such a run is not an eligible
                # paper input and must not enter the aggregate report.
                if not run_dir.with_name(run_dir.name + ".completed").is_file():
                    continue
                meta_path = run_dir / "run_meta.json"
                meta: dict[str, Any] = {}
                if meta_path.is_file():
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        meta = {}
                runs.append(
                    {
                        "run_dir": run_dir,
                        "model": model_dir.name,
                        "seed": seed_dir.name,
                        "meta": meta,
                        "summary": json.loads(
                            (run_dir / "rollout_summary.json").read_text(encoding="utf-8")
                        ),
                    }
                )
    return runs


def build_rollout_report(
    outputs_base: str | Path, out_dir: str | Path | None = None
) -> dict[str, Any]:
    """Aggregate all rollout runs into the CSV/JSON report files.

    Returns the report dict. ``out_dir`` defaults to ``{outputs_base}/_results``.
    """
    base = Path(outputs_base)
    out = Path(out_dir) if out_dir is not None else base / "_results"
    out.mkdir(parents=True, exist_ok=True)

    runs = _scan_rollout_runs(base)
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for entry in runs:
        try:
            rows.extend(read_episode_records(entry["run_dir"]))
        except Exception as exc:  # noqa: BLE001 — one bad run must not kill the report
            skipped.append(f"{entry['run_dir'].name}: {type(exc).__name__}: {exc}")

    success_rows = summarize_episodes(rows)
    comparison_rows = paired_rollout_comparisons(rows)

    _apply_holm(comparison_rows)

    with open(out / SUCCESS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(success_rows[0]) if success_rows else [])
        if success_rows:
            writer.writeheader()
            writer.writerows(success_rows)

    if comparison_rows:
        with open(out / COMPARISONS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0]))
            writer.writeheader()
            writer.writerows(comparison_rows)

    report: dict[str, Any] = {
        "run_count": len(runs),
        "episode_count": len(rows),
        "skipped_runs": skipped,
        "success_rows": success_rows,
        "comparison_rows": comparison_rows,
    }
    (out / REPORT_JSON).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _apply_holm(comparison_rows: list[dict[str, Any]]) -> None:
    """Adjust the five primary comparisons separately per task and seed.

    Diagnostic and negative-control rows retain their raw McNemar p-value but
    receive no adjusted p-value because they are outside the predeclared
    primary hypothesis family.
    """
    for row in comparison_rows:
        row["mcnemar_holm_p"] = None
    by_task_seed: dict[tuple[str, int], list[int]] = {}
    for index, row in enumerate(comparison_rows):
        if row["model"] not in PRIMARY_COMPARATORS or row.get("tag") is not None:
            continue
        key = (str(row["task"]), int(row["training_seed"]))
        by_task_seed.setdefault(key, []).append(index)
    for indices in by_task_seed.values():
        ps = [float(comparison_rows[i]["mcnemar_exact_p"]) for i in indices]
        adjusted = holm_bonferroni(ps)
        for i, p in zip(indices, adjusted):
            comparison_rows[i]["mcnemar_holm_p"] = p


__all__ = [
    "SUCCESS_CSV",
    "COMPARISONS_CSV",
    "REPORT_JSON",
    "build_rollout_report",
    "_scan_rollout_runs",
]

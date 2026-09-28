"""Top-level ``summarize_all()`` entry point.

Reads ``<outputs>/_results/results.jsonl``, validates each row, and
rebuilds the paper-ready artifacts under ``<outputs>/_summaries/``:

* ``aggregates.csv``          — per (model, stage) mean/std/n over seeds
* ``bootstrap_ci.csv``        — bootstrap 95% CIs per (model, stage, metric)
* ``paired_wilcoxon.csv``     — paired Wilcoxon vs the baseline method
* ``metrics.json``            — per (model, stage) per-metric means for the paper text

Idempotent and safe to re-run; no destructive filesystem effects beyond
overwriting the four artifacts.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from phaseforge.outputs_writer.results import read_result_rows
from phaseforge.outputs_writer.schema import ResultRow, validate_row
from phaseforge.outputs_writer.tables import (
    METRIC_COLUMNS,
    write_aggregates_csv,
    write_bootstrap_csv,
    write_paired_wilcoxon_csv,
)


def build_metrics_summary(rows: list[ResultRow]) -> dict[str, Any]:
    """Per (model, tag, stage) per-metric means for ``metrics.json``.

    Output shape::

        {
          "summary": {
              (model, tag, stage): {
                  <metric>: <mean over finite>,
                  "n_nan_<metric>": <count of NaN seeds>,
              },
              ...
          },
          "n_seeds": <distinct (seed,) pairs across all rows>,
          "rows": <row count>,
        }

    Methods that lack a particular metric surface as ``{"<metric>": null,
    "n_nan_<metric>": N}`` — the paper appendix can render this as "n/a"
    rather than dropping the cell.
    """
    groups: dict[tuple[str, str, int], list[ResultRow]] = {}
    for row in rows:
        groups.setdefault((row.model, row.tag or "", row.stage), []).append(row)
    summary: dict[str, dict[str, Any]] = {}
    for (model, tag, stage), group in groups.items():
        entry: dict[str, Any] = {"model": model, "tag": tag, "stage": stage}
        for metric in METRIC_COLUMNS:
            values = [
                float(getattr(row, metric))
                for row in group
                if math.isfinite(float(getattr(row, metric)))
            ]
            if values:
                entry[metric] = float(sum(values) / len(values))
            else:
                entry[metric] = None
            entry[f"n_nan_{metric}"] = len(group) - len(values)
        summary[f"{model}__stage{stage}__tag{tag or 'default'}"] = entry
    return {
        "summary": summary,
        "n_seeds": len({(row.seed,) for row in rows}),
        "rows": len(rows),
    }


def summarize_all(
    outputs_base: str | Path,
    *,
    baseline: str = "precision_residual_phaseforge",
) -> dict[str, Path]:
    """Rebuild paper-ready artifacts under ``<outputs_base>/_summaries/``.

    Args:
        outputs_base: The ``outputs`` directory (containing ``_results/``
            and ``_summaries/``).
        baseline: Method used as the reference in the paired Wilcoxon CSV
            (default ``"precision_residual_phaseforge"`` — the proposed method).

    Raises:
        FileNotFoundError: ``results.jsonl`` does not exist (no evals
            have been run yet).
        phaseforge.outputs.schema.SchemaError: a row in the ledger fails
            schema validation — a real corruption event that the caller
            must investigate (e.g. a hand-edited ``results.jsonl``).
    """
    outputs_base = Path(outputs_base)
    results_path = outputs_base / "_results" / "results.jsonl"
    if not results_path.exists():
        raise FileNotFoundError(
            f"No results ledger at {results_path}. "
            "Run `phaseforge-eval` for at least one (model, stage, seed) "
            "before summarising."
        )

    raw = read_result_rows(outputs_base / "_results")
    rows: list[ResultRow] = []
    for raw_row in raw:
        validate_row(raw_row)
        rows.append(ResultRow(**raw_row))

    summaries = outputs_base / "_summaries"
    summaries.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {
        "aggregates": write_aggregates_csv(rows, summaries / "aggregates.csv"),
        "bootstrap": write_bootstrap_csv(rows, summaries / "bootstrap_ci.csv"),
        "wilcoxon": write_paired_wilcoxon_csv(
            rows, summaries / "paired_wilcoxon.csv", baseline=baseline
        ),
    }

    metrics_path = summaries / "metrics.json"
    metrics_path.write_text(
        json.dumps(build_metrics_summary(rows), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    paths["metrics"] = metrics_path
    return paths


__all__ = ["build_metrics_summary", "summarize_all"]

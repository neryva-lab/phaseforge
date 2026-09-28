"""Training-side and rollout summary artifacts under ``_summaries/``.

The training side rebuilds three paper-ready
CSVs from the validated training ledger + each run's curves:

* ``training_aggregates.csv`` — per ``(model, stage)``: mean ± std over
  seeds of the ``final_val`` scalars, ``best_val_monitor``, ``epochs_run``,
  ``trainable_params`` and ``total_params``;
* ``training_cost.csv`` — per ``(model, stage)``: wall time (mean ± std),
  epochs run, total optimizer steps, parameter counts — the appendix cost
  table; wall time is the training-loop wall clock written to each
  ``summary.json`` (``wall_seconds``), **not** the full-lifecycle value in
  ``timings.json`` (which also spans dependency install, data prep and
  checkout), so the two are not comparable;
* ``training_curves.csv`` — per ``(model, stage, epoch)``: mean ± std over
  seeds of every curve metric — the plot source.

The rollout side reads ``eval/.../episodes.jsonl`` records and writes
``rollout_success.csv`` (per task/model/training-seed success rates with
Wilson intervals) and ``rollout_comparisons.csv`` (paired
PhaseForge-minus-baseline differences).

All aggregations are derived and idempotent — regenerated from the ledgers
by this tooling, never hand-edited.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from phaseforge.outputs_writer.curves import validate_curve_row, validate_summary
from phaseforge.outputs_writer.episodes import (
    paired_rollout_comparisons,
    summarize_episodes,
    validate_episode_record,
)
from phaseforge.outputs_writer.training_summary import read_training_summary_rows

#: Summary-level scalar fields aggregated over seeds alongside ``final_val``.
_SUMMARY_SCALARS = (
    "best_val_monitor",
    "epochs_run",
    "trainable_params",
    "total_params",
    "global_steps",
)


def _finite(values: list[float]) -> list[float]:
    return [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]


def _to_float(value: Any) -> float:
    """Schema-valid rows may carry ``None`` (e.g. ``wall_seconds``/``global_steps``
    are nullable); NaN is the honest aggregate for a missing value."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return float("nan")


def _nanmean(values: list[float]) -> float:
    finite = _finite(values)
    if not finite:
        return float("nan")
    return float(sum(finite) / len(finite))


def _nanstd(values: list[float]) -> float:
    finite = _finite(values)
    if len(finite) < 2:
        return float("nan")
    mean = sum(finite) / len(finite)
    var = sum((v - mean) ** 2 for v in finite) / (len(finite) - 1)
    return float(var**0.5)


def _summary_scalar(row: dict[str, Any], key: str) -> float | None:
    """Extract ``key`` from ``final_val`` (final scalars) or the row itself."""
    if key in row.get("final_val", {}):
        return row["final_val"][key]
    return row.get(key)


def training_aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per ``(model, tag, stage)`` mean/std/n over seeds of the final scalars."""
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["model"], row.get("tag") or "", int(row["stage"])), []).append(row)

    final_val_keys: list[str] = []
    for row in rows:
        for key in row.get("final_val", {}):
            if key not in final_val_keys:
                final_val_keys.append(key)

    scalar_keys: list[str] = []
    for k in _SUMMARY_SCALARS:
        if any(k in r for r in rows) and k not in scalar_keys:
            scalar_keys.append(k)
    for k in sorted(final_val_keys):
        if k not in scalar_keys:
            scalar_keys.append(k)

    out: list[dict[str, Any]] = []
    for (model, tag, stage), group in grouped.items():
        seeds = {r.get("seed") for r in group}
        entry: dict[str, Any] = {
            "model": model,
            "tag": tag,
            "stage": stage,
            "n_seeds": len(seeds),
            "n_rows": len(group),
        }
        for key in scalar_keys:
            values: list[float] = []
            for r in group:
                value = _summary_scalar(r, key)
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue
                if math.isfinite(float(value)):
                    values.append(float(value))
            entry[f"{key}_mean"] = _nanmean(values)
            entry[f"{key}_std"] = _nanstd(values)
            entry[f"{key}_n"] = len(values)
        out.append(entry)
    out.sort(key=lambda e: (e["model"], e["tag"], e["stage"]))
    return out


def training_cost_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per ``(model, tag, stage)`` appendix cost table rows."""
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["model"], row.get("tag") or "", int(row["stage"])), []).append(row)

    out: list[dict[str, Any]] = []
    for (model, tag, stage), group in grouped.items():
        wall = [_to_float(r.get("wall_seconds")) for r in group]
        epochs = [_to_float(r.get("epochs_run")) for r in group]
        steps = [_to_float(r.get("global_steps")) for r in group]
        trainable = [_to_float(r.get("trainable_params")) for r in group]
        total = [_to_float(r.get("total_params")) for r in group]
        seeds = {r.get("seed") for r in group}
        entry: dict[str, Any] = {
            "model": model,
            "tag": tag,
            "stage": stage,
            "n_seeds": len(seeds),
            "wall_seconds_mean": _nanmean(wall),
            "wall_seconds_std": _nanstd(wall),
            "epochs_run_mean": _nanmean(epochs),
            "epochs_run_std": _nanstd(epochs),
            "total_global_steps": float(sum(_finite(steps))),
            "global_steps_mean": _nanmean(steps),
            "global_steps_std": _nanstd(steps),
            "trainable_params_mean": _nanmean(trainable),
            "total_params_mean": _nanmean(total),
        }
        out.append(entry)
    out.sort(key=lambda e: (e["model"], e["tag"], e["stage"]))
    return out


def read_training_curves(
    outputs_base: str | Path, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Collect every validated curve row, tagged with model/stage/seed."""
    outputs_base = Path(outputs_base)
    out: list[dict[str, Any]] = []
    for row in rows:
        run_dir = row.get("run_dir")
        if not run_dir:
            continue
        path = outputs_base / run_dir / "metrics" / "training_curves.jsonl"
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        ends_with_newline = text.endswith(("\n", "\r"))
        lines = text.splitlines()
        for index, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            is_last = index == len(lines) - 1
            try:
                curve = json.loads(stripped)
            except json.JSONDecodeError:
                if is_last and not ends_with_newline:
                    continue
                raise
            validate_curve_row(curve)
            out.append(
                {
                    "model": row["model"],
                    "tag": row.get("tag"),
                    "stage": int(row["stage"]),
                    "seed": row.get("seed"),
                    **curve,
                }
            )
    return out


def write_training_aggregates_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    aggregates = training_aggregate_rows(rows)
    if not aggregates:
        path.write_text("", encoding="utf-8")
        return path
    import csv

    keys = list(aggregates[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for entry in aggregates:
            writer.writerow(entry)
    return path


def write_training_cost_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    costs = training_cost_rows(rows)
    if not costs:
        path.write_text("", encoding="utf-8")
        return path
    import csv

    keys = list(costs[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for entry in costs:
            writer.writerow(entry)
    return path


def write_training_curves_csv(curve_rows: list[dict[str, Any]], path: Path) -> Path:
    """Per ``(model, stage, epoch)`` mean ± std over seeds of curve metrics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    import csv

    metric_keys: list[str] = []
    for row in curve_rows:
        for key, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if key not in metric_keys and key not in {
                    "model",
                    "stage",
                    "epoch",
                    "global_step",
                    "seed",
                }:
                    metric_keys.append(key)

    grouped: dict[tuple[str, str, int, int], list[dict[str, Any]]] = {}
    for row in curve_rows:
        grouped.setdefault(
            (row["model"], row.get("tag") or "", int(row["stage"]), int(row["epoch"])),
            [],
        ).append(row)

    out: list[dict[str, Any]] = []
    for (model, tag, stage, epoch), group in grouped.items():
        entry: dict[str, Any] = {
            "model": model,
            "tag": tag,
            "stage": stage,
            "epoch": epoch,
        }
        for key in metric_keys:
            values: list[float] = []
            for r in group:
                value = r.get(key)
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue
                if math.isfinite(float(value)):
                    values.append(float(value))
            entry[f"{key}_mean"] = _nanmean(values)
            entry[f"{key}_std"] = _nanstd(values)
            entry[f"{key}_n"] = len(values)
        out.append(entry)
    out.sort(key=lambda e: (e["model"], e["tag"], e["stage"], e["epoch"]))

    if not out:
        path.write_text("", encoding="utf-8")
        return path
    keys = list(out[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for entry in out:
            writer.writerow(entry)
    return path


def write_rollout_success_csv(episode_rows: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    import csv

    summaries = summarize_episodes(episode_rows)
    if not summaries:
        path.write_text("", encoding="utf-8")
        return path
    keys = list(summaries[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for entry in summaries:
            writer.writerow(entry)
    return path


def write_rollout_comparisons_csv(
    episode_rows: list[dict[str, Any]],
    path: Path,
    *,
    baseline: str = "precision_residual_phaseforge",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    import csv

    comparisons = paired_rollout_comparisons(episode_rows, baseline=baseline)
    if not comparisons:
        path.write_text("", encoding="utf-8")
        return path
    keys = list(comparisons[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for entry in comparisons:
            writer.writerow(entry)
    return path


def summarize_training(outputs_base: str | Path) -> dict[str, Path]:
    """Rebuild the training summary artifacts under ``_summaries/``.

    Reads every schema-validated row in ``_results/training_summary.jsonl``
    plus each run's ``metrics/training_curves.jsonl`` (located via the
    ledger's ``run_dir``). Idempotent and safe to re-run.
    """
    outputs_base = Path(outputs_base)
    rows = read_training_summary_rows(outputs_base / "_results")
    for row in rows:
        validate_summary(row)

    summaries = outputs_base / "_summaries"
    summaries.mkdir(parents=True, exist_ok=True)
    paths = {
        "training_aggregates": write_training_aggregates_csv(
            rows, summaries / "training_aggregates.csv"
        ),
        "training_cost": write_training_cost_csv(rows, summaries / "training_cost.csv"),
    }
    curve_rows = read_training_curves(outputs_base, rows)
    paths["training_curves"] = write_training_curves_csv(
        curve_rows, summaries / "training_curves.csv"
    )
    return paths


def summarize_rollout(
    outputs_base: str | Path, *, baseline: str = "precision_residual_phaseforge"
) -> dict[str, Path]:
    """Rebuild the rollout summary artifacts from every ``episodes.jsonl``.

    Scans ``eval/`` run directories for episode records, validates each
    record, and writes ``rollout_success.csv`` + ``rollout_comparisons.csv``
    under ``_summaries/``. Idempotent.
    """
    outputs_base = Path(outputs_base)
    episode_rows: list[dict[str, Any]] = []
    for path in sorted(outputs_base.glob("eval/**/episodes.jsonl")):
        text = path.read_text(encoding="utf-8")
        ends_with_newline = text.endswith(("\n", "\r"))
        lines = text.splitlines()
        for index, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            is_last = index == len(lines) - 1
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                if is_last and not ends_with_newline:
                    continue
                raise
            validate_episode_record(record)
            episode_rows.append(record)

    summaries = outputs_base / "_summaries"
    summaries.mkdir(parents=True, exist_ok=True)
    return {
        "rollout_success": write_rollout_success_csv(
            episode_rows, summaries / "rollout_success.csv"
        ),
        "rollout_comparisons": write_rollout_comparisons_csv(
            episode_rows, summaries / "rollout_comparisons.csv", baseline=baseline
        ),
    }


__all__ = [
    "training_aggregate_rows",
    "training_cost_rows",
    "read_training_curves",
    "write_training_aggregates_csv",
    "write_training_cost_csv",
    "write_training_curves_csv",
    "write_rollout_success_csv",
    "write_rollout_comparisons_csv",
    "summarize_training",
    "summarize_rollout",
]

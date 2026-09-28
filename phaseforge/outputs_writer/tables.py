"""Aggregation, bootstrap CIs, and paired comparisons for PhaseForge.

Paper-ready CSVs produced from :file:`outputs/_results/results.jsonl`:

* ``_summaries/aggregates.csv``      — per (model, tag, stage) mean/std/n over seeds
* ``_summaries/bootstrap_ci.csv``    — bootstrap summaries per (model, tag, stage, metric)
* ``_summaries/paired_wilcoxon.csv`` — paired Wilcoxon vs the baseline method

Adaptations from the csd reference:

* grouping is (model, tag, stage) not (system, method) — the ``tag`` axis
  keeps data-variant runs that share a model name (e.g. ``bc`` vs
  ``bc``/``robot_only``) in separate rows;
* the paired Wilcoxon pairing key is (seed, tag) — PhaseForge's multi-seed
  protocol shares seeds across all model variants and every method is
  evaluated at its own final stage on the identical frozen reset bank, so
  the per-seed pairing is exact across stages and tasks (the tag keeps
  tasks/variants separate);
* ``min_pairs`` defaults to 3 (PhaseForge's SEEDS=[42, 43, 44]); scipy's
  Wilcoxon needs at least one pair to compute but n=3 has very low power,
  so the paper should report n alongside p;
* NaN values are skipped in every aggregation (per-metric ``n`` is
  reported so methods that lack a metric — e.g. ``bc`` lacks all routing
  metrics — surface honestly instead of dropping the cell);
* the metric list is sourced from :data:`phaseforge.outputs.schema.OPTIONAL_METRIC_FIELDS`
  plus ``action_mse`` so adding a new metric in one place extends the
  paper tables without editing this file.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from phaseforge.outputs_writer.schema import OPTIONAL_METRIC_FIELDS, ResultRow

METRIC_COLUMNS: tuple[str, ...] = ("action_mse",) + OPTIONAL_METRIC_FIELDS

_BOOT_SEED = 20240811
_BOOT_ITERS = 1000
_MIN_PAIRS_DEFAULT = 3


@dataclass
class Aggregate:
    """Per (model, tag, stage) mean / std / n over seeds, for every metric."""

    model: str
    tag: str
    stage: int
    n_seeds: int
    n_rows: int
    action_mse_mean: float
    action_mse_std: float
    action_mse_n: int
    action_l2_threshold_rate_mean: float
    action_l2_threshold_rate_std: float
    action_l2_threshold_rate_n: int
    boundary_action_smoothness_mean: float
    boundary_action_smoothness_std: float
    boundary_action_smoothness_n: int
    routing_entropy_mean: float
    routing_entropy_std: float
    routing_entropy_n: int
    routing_entropy_variance_mean: float
    routing_entropy_variance_std: float
    routing_entropy_variance_n: int
    time_to_stable_routing_mean: float
    time_to_stable_routing_std: float
    time_to_stable_routing_n: int
    routing_stability_fraction_mean: float
    routing_stability_fraction_std: float
    routing_stability_fraction_n: int
    topk_balance_score_mean: float
    topk_balance_score_std: float
    topk_balance_score_n: int
    top1_balance_score_mean: float
    top1_balance_score_std: float
    top1_balance_score_n: int
    topk_collapse_rate_mean: float
    topk_collapse_rate_std: float
    topk_collapse_rate_n: int
    top1_collapse_rate_mean: float
    top1_collapse_rate_std: float
    top1_collapse_rate_n: int
    phase_expert_nmi_mean: float
    phase_expert_nmi_std: float
    phase_expert_nmi_n: int

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


def _finite_or_nan(value: float) -> float:
    """Pass through ``NaN``; reject ``inf``."""
    if math.isnan(value):
        return float("nan")
    return float(value)


def _nanmean(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.nanmean(values))


def _nanstd(values: np.ndarray) -> float:
    # Sample standard deviation (ddof=1) — the paper reports mean +/- std
    # over N training seeds, so the conventional small-sample correction
    # applies (fewer than 2 samples yields NaN, which is honest rather
    # than a false 0 and avoids numpy's dof warning).
    if values.size < 2:
        return float("nan")
    return float(np.nanstd(values, ddof=1))


def _values_for(rows: Sequence[ResultRow], metric: str) -> np.ndarray:
    return np.array(
        [float(getattr(row, metric)) for row in rows if math.isfinite(float(getattr(row, metric)))],
        dtype=float,
    )


def aggregate_rows(rows: Sequence[ResultRow]) -> list[Aggregate]:
    """Group rows by ``(model, tag, stage)`` and compute mean/std/n per metric.

    ``tag`` separates data-variant runs that share a ``model`` name (e.g.
    the ``bc`` floor and the ``bc``/``robot_only`` negative control), so the
    two are never merged. ``n_seeds`` is the number of distinct seeds in the
    group; ``n_rows`` is the total row count (matches ``n_seeds`` when each
    seed is run once, which is the current protocol). Per-metric ``n`` is the
    count of finite (non-NaN) values — methods without a router (e.g. ``bc``)
    report ``n=0`` for routing metrics rather than dropping the cell.
    """
    grouped: dict[tuple[str, str, int], list[ResultRow]] = {}
    for row in rows:
        grouped.setdefault((row.model, row.tag or "", row.stage), []).append(row)
    out: list[Aggregate] = []
    for (model, tag, stage), group in grouped.items():
        seeds = {row.seed for row in group}
        per_metric = {}
        for metric in METRIC_COLUMNS:
            values = _values_for(group, metric)
            per_metric[metric] = (
                _nanmean(values),
                _nanstd(values),
                int(values.size),
            )
        out.append(
            Aggregate(
                model=model,
                tag=tag,
                stage=stage,
                n_seeds=len(seeds),
                n_rows=len(group),
                action_mse_mean=per_metric["action_mse"][0],
                action_mse_std=per_metric["action_mse"][1],
                action_mse_n=per_metric["action_mse"][2],
                action_l2_threshold_rate_mean=per_metric["action_l2_threshold_rate"][0],
                action_l2_threshold_rate_std=per_metric["action_l2_threshold_rate"][1],
                action_l2_threshold_rate_n=per_metric["action_l2_threshold_rate"][2],
                boundary_action_smoothness_mean=per_metric["boundary_action_smoothness"][0],
                boundary_action_smoothness_std=per_metric["boundary_action_smoothness"][1],
                boundary_action_smoothness_n=per_metric["boundary_action_smoothness"][2],
                routing_entropy_mean=per_metric["routing_entropy"][0],
                routing_entropy_std=per_metric["routing_entropy"][1],
                routing_entropy_n=per_metric["routing_entropy"][2],
                routing_entropy_variance_mean=per_metric["routing_entropy_variance"][0],
                routing_entropy_variance_std=per_metric["routing_entropy_variance"][1],
                routing_entropy_variance_n=per_metric["routing_entropy_variance"][2],
                time_to_stable_routing_mean=per_metric["time_to_stable_routing"][0],
                time_to_stable_routing_std=per_metric["time_to_stable_routing"][1],
                time_to_stable_routing_n=per_metric["time_to_stable_routing"][2],
                routing_stability_fraction_mean=per_metric["routing_stability_fraction"][0],
                routing_stability_fraction_std=per_metric["routing_stability_fraction"][1],
                routing_stability_fraction_n=per_metric["routing_stability_fraction"][2],
                topk_balance_score_mean=per_metric["topk_balance_score"][0],
                topk_balance_score_std=per_metric["topk_balance_score"][1],
                topk_balance_score_n=per_metric["topk_balance_score"][2],
                top1_balance_score_mean=per_metric["top1_balance_score"][0],
                top1_balance_score_std=per_metric["top1_balance_score"][1],
                top1_balance_score_n=per_metric["top1_balance_score"][2],
                topk_collapse_rate_mean=per_metric["topk_collapse_rate"][0],
                topk_collapse_rate_std=per_metric["topk_collapse_rate"][1],
                topk_collapse_rate_n=per_metric["topk_collapse_rate"][2],
                top1_collapse_rate_mean=per_metric["top1_collapse_rate"][0],
                top1_collapse_rate_std=per_metric["top1_collapse_rate"][1],
                top1_collapse_rate_n=per_metric["top1_collapse_rate"][2],
                phase_expert_nmi_mean=per_metric["phase_expert_nmi"][0],
                phase_expert_nmi_std=per_metric["phase_expert_nmi"][1],
                phase_expert_nmi_n=per_metric["phase_expert_nmi"][2],
            )
        )
    out.sort(key=lambda a: (a.model, a.tag, a.stage))
    return out


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_iters: int = _BOOT_ITERS,
    seed: int = _BOOT_SEED,
) -> tuple[float, float, float]:
    """Return ``(mean, ci95_low, ci95_high)`` via percentile bootstrap.

    Non-finite inputs are filtered before resampling. Returns
    ``(NaN, NaN, NaN)`` when no finite values are present.
    """
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = arr.size
    means = np.empty(int(n_iters), dtype=float)
    for i in range(int(n_iters)):
        idx = rng.integers(0, n, size=n)
        means[i] = arr[idx].mean()
    return (
        float(means.mean()),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def bootstrap_ci_per_row(
    rows: Sequence[ResultRow],
    *,
    metric: str,
) -> list[dict[str, object]]:
    """Exploratory bootstrap CI per (model, tag, stage) for a single metric.

    The current rows are seed-level summaries. With the protocol minimum of
    three seeds, these intervals are not the primary inferential result; the
    paper must report seed-level variation and episode-level Wilson intervals
    from rollout records separately.
    """
    grouped: dict[tuple[str, str, int], list[ResultRow]] = {}
    for row in rows:
        grouped.setdefault((row.model, row.tag or "", row.stage), []).append(row)
    out: list[dict[str, object]] = []
    for (model, tag, stage), group in grouped.items():
        values = [float(getattr(row, metric)) for row in group]
        m, lo, hi = bootstrap_ci(values)
        out.append(
            {
                "model": model,
                "tag": tag,
                "stage": stage,
                "metric": metric,
                "n": len(values),
                "mean": m,
                "ci95_low": lo,
                "ci95_high": hi,
            }
        )
    out.sort(key=lambda r: (r["model"], r["tag"], r["stage"], r["metric"]))
    return out


def paired_wilcoxon(
    rows: Sequence[ResultRow],
    *,
    method_a: tuple[str, str | None],
    method_b: tuple[str, str | None],
    metric: str,
    min_pairs: int = _MIN_PAIRS_DEFAULT,
) -> dict[str, object] | None:
    """Paired Wilcoxon signed-rank on same ``(seed, tag)`` pairs.

    ``method_a``/``method_b`` are ``(model, tag)`` identities, so data-variant
    runs that share a model name (e.g. ``("bc", None)`` vs
    ``("bc", "robot_only")``) never cross-pair. The pairing key deliberately
    excludes ``stage``: every method is evaluated at its own final stage
    (MoE family stage 2, BC family stage 1) on the identical frozen reset
    bank, so the per-seed paired difference is the deployment comparison the
    paper reports — a stage-locked key would silently drop every
    PhaseForge-vs-BC pair. Returns ``None`` when fewer than ``min_pairs``
    pairs are available or when scipy is not installed (the paper can still
    be built from ``aggregates.csv`` + ``bootstrap_ci.csv``; this function is
    best effort).
    """
    try:
        from scipy.stats import wilcoxon
    except Exception:
        return None
    pa: dict[tuple[int, str | None], float] = {}
    pb: dict[tuple[int, str | None], float] = {}
    for row in rows:
        if math.isnan(float(getattr(row, metric))):
            continue
        identity = (row.model, row.tag)
        key = (int(row.seed), row.tag)
        if identity == method_a:
            pa[key] = float(getattr(row, metric))
        elif identity == method_b:
            pb[key] = float(getattr(row, metric))
    keys = sorted(set(pa) & set(pb))
    if len(keys) < int(min_pairs):
        return None
    a = np.array([pa[k] for k in keys], dtype=float)
    b = np.array([pb[k] for k in keys], dtype=float)
    try:
        stat, p = wilcoxon(a, b, zero_method="zsplit", alternative="two-sided")
    except ValueError:
        return None
    return {
        "method_a": method_a[0],
        "tag_a": method_a[1],
        "method_b": method_b[0],
        "tag_b": method_b[1],
        "metric": metric,
        "n_pairs": len(keys),
        "mean_diff": float(np.mean(a - b)),
        "median_diff": float(np.median(a - b)),
        "statistic": float(stat),
        "p_value": float(p),
    }


def write_aggregates_csv(rows: Sequence[ResultRow], path: Path) -> Path:
    """Write ``_summaries/aggregates.csv`` (one row per model+stage)."""
    aggregates = aggregate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not aggregates:
        path.write_text("", encoding="utf-8")
        return path
    keys = list(aggregates[0].to_dict().keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for agg in aggregates:
            writer.writerow(agg.to_dict())
    return path


def write_bootstrap_csv(rows: Sequence[ResultRow], path: Path) -> Path:
    """Write exploratory seed-row bootstrap summaries.

    This file is useful for diagnostics, but is not the final paper CI under
    the three-seed protocol. Rollout episodes require their own Wilson or
    paired episode-level analysis.
    """
    out: list[dict[str, object]] = []
    for metric in METRIC_COLUMNS:
        out.extend(bootstrap_ci_per_row(rows, metric=metric))
    path.parent.mkdir(parents=True, exist_ok=True)
    if not out:
        path.write_text("", encoding="utf-8")
        return path
    keys = list(out[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for record in out:
            writer.writerow(record)
    return path


def write_paired_wilcoxon_csv(
    rows: Sequence[ResultRow],
    path: Path,
    *,
    baseline: str,
    min_pairs: int = _MIN_PAIRS_DEFAULT,
) -> Path:
    """Write ``_summaries/paired_wilcoxon.csv`` (baseline vs every other method).

    The baseline identity is resolved **per comparison** as
    ``(baseline, tag_of_method_b)``. Five-task rows carry the protocol task
    as their tag (``Can``/``Square``/…), so a single hardcoded
    ``(baseline, None)`` identity would match no row and the CSV would be
    silently empty; with per-tag resolution each comparison stays within one
    task while untagged (single-task) namespaces keep their exact previous
    pairing. Tagged variants of the baseline model itself (e.g.
    ``robot_only``) are skipped — they pair against nothing by design.
    """
    identities = sorted({(row.model, row.tag) for row in rows}, key=lambda i: (i[0], i[1] or ""))
    out: list[dict[str, object]] = []
    for identity in identities:
        if identity[0] == baseline:
            continue
        baseline_identity = (baseline, identity[1])
        for metric in METRIC_COLUMNS:
            record = paired_wilcoxon(
                rows,
                method_a=baseline_identity,
                method_b=identity,
                metric=metric,
                min_pairs=min_pairs,
            )
            if record is not None:
                out.append(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not out:
        path.write_text("", encoding="utf-8")
        return path
    keys = list(out[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for record in out:
            writer.writerow(record)
    return path


__all__ = [
    "Aggregate",
    "METRIC_COLUMNS",
    "aggregate_rows",
    "bootstrap_ci",
    "bootstrap_ci_per_row",
    "paired_wilcoxon",
    "write_aggregates_csv",
    "write_bootstrap_csv",
    "write_paired_wilcoxon_csv",
]

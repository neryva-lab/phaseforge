"""Optional loaders for cross-cell summary artifacts (post-sweep _summaries/)."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from studies.analysis.common import io as cio


@dataclass(frozen=True)
class StratifiedStats:
    """stratified_stats.json — per-task seed means, bootstrap CIs, P(X>Y)."""

    raw: dict[str, Any]


@dataclass(frozen=True)
class PairedWilcoxon:
    """paired_wilcoxon.csv rows as list[dict]."""

    rows: tuple[dict[str, str], ...]


def load_stratified(root: Path) -> StratifiedStats | None:
    path = root / "_summaries" / "stratified_stats.json"
    if not path.is_file():
        return None
    data = cio.read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return StratifiedStats(raw=data)


def load_paired_wilcoxon(root: Path) -> PairedWilcoxon | None:
    path = root / "_summaries" / "paired_wilcoxon.csv"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8", newline="") as f:
        rows = [dict(row) for row in csv.DictReader(f)]
    return PairedWilcoxon(rows=tuple(rows))

"""Global append-only training summary ledger.

``<outputs>/_results/training_summary.jsonl`` is the training-side analog
of ``results.jsonl``: one schema-validated row per completed training run,
appended by the CLI after ``trainer.fit()`` returns. The row content is the
run-local ``metrics/summary.json`` (same schema — see
:func:`phaseforge.outputs_writer.curves.validate_summary`), so the run-local
summary is authoritative for recovery if the global append fails.

Contract: the append is **idempotent** (a row
for a given ``run_id`` is written at most once), and a failed append is
never silently dropped — the caller writes a reconciliation record next to
the run-local summary so a reconciliation tool can rebuild the ledger from
``metrics/summary.json``. :func:`reconcile_training_ledger` is that tool.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from filelock import FileLock

from phaseforge.outputs_writer.curves import validate_summary
from phaseforge.outputs_writer.schema import SchemaError
from phaseforge.outputs_writer.writer import parse_run_dir

_RECONCILIATION_FILENAME = "training_summary_pending.json"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def append_training_summary_row(results_dir: str | Path, summary: dict[str, Any]) -> Path:
    """Validate ``summary`` and append it to ``training_summary.jsonl``.

    Idempotent per ``run_id``: if a row for the same run already exists,
    nothing is written. Raises :class:`SchemaError` for malformed rows
    (no file write occurs) and :class:`OSError` on filesystem failure.

    Returns:
        The ledger file path.
    """
    validate_summary(summary)

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    target = results_dir / "training_summary.jsonl"
    lock = FileLock(str(results_dir / ".training_summary.lock"))
    payload = json.dumps(summary, default=_json_default) + "\n"
    with lock:
        existing = read_training_summary_rows(results_dir)
        if any(row.get("run_id") == summary["run_id"] for row in existing):
            return target
        with open(target, "a", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
    return target


def read_training_summary_rows(results_dir: str | Path) -> list[dict[str, Any]]:
    """Read every row from ``training_summary.jsonl`` (tolerant).

    A crash-truncated trailing line is skipped; corruption anywhere else
    raises :class:`json.JSONDecodeError`.
    """
    results_dir = Path(results_dir)
    target = results_dir / "training_summary.jsonl"
    if not target.exists():
        return []
    text = target.read_text(encoding="utf-8")
    ends_with_newline = text.endswith(("\n", "\r"))
    lines = text.splitlines()
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        is_last = index == len(lines) - 1
        try:
            rows.append(json.loads(stripped))
        except json.JSONDecodeError:
            if is_last and not ends_with_newline:
                continue
            raise
    return rows


def write_reconciliation_record(run_dir: str | Path, error: BaseException) -> Path:
    """Persist a reconciliation marker for a run whose ledger append failed.

    The record is advisory only — the run-local ``metrics/summary.json`` is
    the authoritative recovery source for :func:`reconcile_training_ledger`.
    """
    run_dir = Path(run_dir)
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    _ts, _tag, run_id = parse_run_dir(run_dir.name)
    record = {
        "run_id": run_id or run_dir.name,
        "written_at": datetime.now(UTC).isoformat(),
        "error": repr(error),
    }
    path = metrics_dir / _RECONCILIATION_FILENAME
    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return path


def reconciliation_record_path(run_dir: str | Path) -> Path:
    """The on-disk location of a run's reconciliation record."""
    return Path(run_dir) / "metrics" / _RECONCILIATION_FILENAME


def has_reconciliation_record(run_dir: str | Path) -> bool:
    return reconciliation_record_path(run_dir).is_file()


def reconcile_training_ledger(
    results_dir: str | Path,
    outputs_base: str | Path,
) -> dict[str, int]:
    """Rebuild the training ledger from every run-local ``summary.json``.

    Scans ``<outputs_base>/**/metrics/summary.json`` (training runs only —
    eval run dirs never write that file), validates each summary, and
    appends any row whose ``run_id`` is missing from the ledger. Returns
    ``{"scanned": ..., "appended": ..., "duplicates": ..., "invalid": ...}``.

    Raises:
        SchemaError: a run-local summary fails schema validation — a real
            corruption event that must be investigated, not silently skipped.
    """
    results_dir = Path(results_dir)
    outputs_base = Path(outputs_base)

    existing = read_training_summary_rows(results_dir)
    present_ids = {row.get("run_id") for row in existing}

    scanned = appended = duplicates = invalid = 0
    for summary_path in sorted(outputs_base.rglob("metrics/summary.json")):
        scanned += 1
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            validate_summary(summary)
        except (OSError, json.JSONDecodeError) as exc:
            invalid += 1
            raise SchemaError(
                f"Run-local summary {summary_path} is unreadable or not JSON: {exc}"
            ) from exc
        run_id = summary["run_id"]
        if run_id in present_ids:
            duplicates += 1
            continue
        # Record the run directory relative to the outputs base so the
        # summarize tooling can locate the curves file.
        run_dir = summary_path.parent.parent
        try:
            summary["run_dir"] = run_dir.relative_to(outputs_base).as_posix()
        except ValueError:
            summary["run_dir"] = str(run_dir)
        append_training_summary_row(results_dir, summary)
        present_ids.add(run_id)
        appended += 1

    return {
        "scanned": scanned,
        "appended": appended,
        "duplicates": duplicates,
        "invalid": invalid,
    }


__all__ = [
    "append_training_summary_row",
    "read_training_summary_rows",
    "write_reconciliation_record",
    "reconciliation_record_path",
    "has_reconciliation_record",
    "reconcile_training_ledger",
]

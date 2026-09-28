"""Backfill ``tag``/``method`` identity fields into legacy result ledgers.

Legacy ``results.jsonl`` and ``training_summary.jsonl`` rows (written before
the schema gained the ``tag``/``method`` fields) only record ``model``, so
data-variant runs that share a model name — e.g. the ``bc`` floor and the
``bc``/``robot_only`` negative control — were merged by the summarizers.

This one-shot migration reconstructs the missing identity from each run's
``run_meta.json`` (which recorded ``tag`` all along):

* training rows: ``run_id`` -> run dir ``run_meta.json``
* eval rows: the row's own ``run_id`` (the eval run) first, falling back to
  the evaluated checkpoint's run id derived from ``ckpt_path``

Ledgers are rewritten in place, preserving row order; every modified row is
re-validated. Rows that already carry ``tag`` (post-fix runs) are untouched.
Rows with no matching ``run_meta.json`` keep ``tag: null`` and are reported.

The CLI entry point is ``scripts/backfill_tags.py``; calling the functions
directly is what the unit tests exercise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from phaseforge.outputs_writer.curves import validate_summary
from phaseforge.outputs_writer.results import read_result_rows
from phaseforge.outputs_writer.schema import validate_row
from phaseforge.outputs_writer.training_summary import read_training_summary_rows
from phaseforge.outputs_writer.writer import parse_run_dir


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _run_id_from_dir_name(name: str) -> str | None:
    _, _, run_id = parse_run_dir(name)
    return run_id


def collect_run_meta(outputs_base: Path) -> dict[str, dict[str, Any]]:
    """Map ``run_id`` -> metadata parsed from every ``run_meta.json``.

    ``run_meta.json`` lives in both the training tree
    (``outputs/<model>/stage<N>/seed<S>/<run>/``) and the eval tree
    (``outputs/eval/<model>/seed<S>/<run>/``); either is authoritative for
    the ``tag``.
    """
    index: dict[str, dict[str, Any]] = {}
    for meta_path in sorted(outputs_base.rglob("run_meta.json")):
        run_dir = meta_path.parent
        run_id = _run_id_from_dir_name(run_dir.name)
        if run_id is None:
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        index[run_id] = {
            "tag": meta.get("tag"),
            "method": meta.get("method"),
        }
    return index


def _eval_run_id_fallback(row: dict[str, Any]) -> str | None:
    """Derive the evaluated checkpoint's run id from ``ckpt_path``."""
    ckpt = row.get("ckpt_path") or ""
    if not ckpt:
        return None
    run_dir_name = Path(ckpt).parent.parent.name
    return _run_id_from_dir_name(run_dir_name)


def _lookup_meta(run_id: object, index: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Look up a run's metadata by id, tolerating a ``None``/non-str id."""
    if not isinstance(run_id, str) or not run_id:
        return None
    return index.get(run_id)


def backfill_results(results_dir: Path, index: dict[str, dict[str, Any]]) -> dict[str, int]:
    """Backfill ``tag``/``method`` on every results row missing them."""
    rows = read_result_rows(results_dir)
    changed = 0
    for row in rows:
        if row.get("tag") is not None:
            continue
        meta = _lookup_meta(row.get("run_id"), index)
        if not meta:
            meta = _lookup_meta(_eval_run_id_fallback(row), index)
        if not meta:
            continue
        row["tag"] = meta["tag"]
        if meta["method"] is not None and "method" not in row:
            row["method"] = meta["method"]
        validate_row(row)
        changed += 1
    if changed:
        _write_jsonl(results_dir / "results.jsonl", rows)
    return {"changed": changed, "rows": len(rows)}


def backfill_training_summary(
    results_dir: Path, index: dict[str, dict[str, Any]]
) -> dict[str, int]:
    """Backfill ``tag``/``method`` on every training row missing them."""
    rows = read_training_summary_rows(results_dir)
    changed = 0
    for row in rows:
        if row.get("tag") is not None:
            continue
        meta = index.get(row.get("run_id") or "")
        if not meta:
            continue
        row["tag"] = meta["tag"]
        if meta["method"] is not None and "method" not in row:
            row["method"] = meta["method"]
        validate_summary(row)
        changed += 1
    if changed:
        _write_jsonl(results_dir / "training_summary.jsonl", rows)
    return {"changed": changed, "rows": len(rows)}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(r, default=_json_default) + "\n" for r in rows)
    path.write_text(text, encoding="utf-8")


__all__ = [
    "collect_run_meta",
    "backfill_results",
    "backfill_training_summary",
]

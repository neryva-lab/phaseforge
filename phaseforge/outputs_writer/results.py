"""Global append-only results ledger for offline evaluations.

Every :func:`phaseforge-eval` run appends exactly one schema-validated
row to ``<outputs>/_results/results.jsonl``. The cross-process
:class:`filelock.FileLock` plus the OS-atomic ``O_APPEND`` semantics mean
that two concurrent evaluations can safely append rows without
corruption. Rows are validated by :func:`phaseforge.outputs_writer.schema.validate_row`
**before** any file write, so a malformed row never reaches disk.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from filelock import FileLock

from phaseforge.outputs_writer.schema import SchemaError, validate_row


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def append_result_row(results_dir: str | Path, row: dict[str, Any]) -> Path:
    """Validate ``row`` and append it to ``results.jsonl``.

    Raises:
        SchemaError: ``row`` is malformed; **no file write occurs**.
        OSError: filesystem error (disk full, permissions, ...).
    """
    try:
        validate_row(row)
    except SchemaError:
        raise

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    target = results_dir / "results.jsonl"
    lock = FileLock(str(results_dir / ".results.lock"))
    payload = json.dumps(row, default=_json_default) + "\n"
    with lock:
        with open(target, "a", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
    return target


def read_result_rows(results_dir: str | Path) -> list[dict[str, Any]]:
    """Read every row from ``results.jsonl``.

    Rows are returned as raw dicts (no schema validation here — callers
    that need it should pass each dict through
    :func:`phaseforge.outputs_writer.schema.validate_row`). A truncated
    **trailing** line (a crash mid-append) and empty lines are silently
    skipped; a real corruption event (incomplete JSON anywhere else in
    the file) raises :class:`json.JSONDecodeError` so the caller does not
    silently consume a half-written ledger.
    """
    results_dir = Path(results_dir)
    target = results_dir / "results.jsonl"
    if not target.exists():
        return []
    text = target.read_text(encoding="utf-8")
    # ``splitlines`` cannot tell us whether the file ended with a newline;
    # inspect the raw text to distinguish a truncated trailing line (crash
    # mid-append, no trailing newline) from genuine corruption.
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


__all__ = ["append_result_row", "read_result_rows"]

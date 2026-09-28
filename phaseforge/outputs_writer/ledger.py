"""Append-only run ledger.

Two views of the same append-only JSONL:

* ``outputs/_ledger/runs.jsonl`` — one row per run, atomic appends
* ``outputs/_ledger/index.json`` — rebuilt from ``runs.jsonl`` on demand

Rows mirror the lifecycle markers written by :class:`RunWriter` and carry
hashes + paths so analysis tooling never has to walk the filesystem. A
``status`` row is written when a run starts (pending) and updated on
completion or failure; if the cloud session is killed mid-run the ledger
correctly records ``pending`` until the next ``update_status`` call.

Schema note: the full ``config_hash`` and ``git_sha`` are the runtime
config hash and the actual git commit, not whatever values happen to
appear in the run's ``run_meta.json``. This is intentional — the ledger
row is the canonical provenance record.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from filelock import FileLock

_KIND_TRAIN = "train"
_KIND_EVAL = "eval"
_VALID_KINDS = frozenset({_KIND_TRAIN, _KIND_EVAL})

_STATUS_PENDING = "pending"
_STATUS_COMPLETED = "completed"
_STATUS_FAILED = "failed"
_VALID_STATUSES = frozenset({_STATUS_PENDING, _STATUS_COMPLETED, _STATUS_FAILED})


@dataclass
class LedgerRow:
    """One row in ``outputs/_ledger/runs.jsonl``."""

    run_id: str
    kind: str
    timestamp: str
    model: str
    config_hash: str
    git_sha: str
    status: str
    path: str
    stage: int | None = None
    seed: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class RunLedger:
    """Append-only JSONL ledger with throttled ``index.json`` mirror."""

    def __init__(
        self,
        ledger_dir: str | Path,
        *,
        index_throttle: int = 25,
    ) -> None:
        """Construct the ledger.

        ``index_throttle`` controls how often :meth:`append` rebuilds the
        ``index.json`` mirror. The mirror is **always** rebuilt after
        :meth:`update_status` and can be force-built with :meth:`flush`;
        between rebuilds the reader falls back to ``runs.jsonl`` so the
        cost of a full rebuild is amortised over ``index_throttle`` runs.
        """
        self.dir = Path(ledger_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.dir / "runs.jsonl"
        self.index_path = self.dir / "index.json"
        self.lock = FileLock(str(self.dir / ".ledger.lock"))
        self._index_throttle = max(1, int(index_throttle))
        self._appends_since_rebuild = 0

    def append(self, row: LedgerRow) -> None:
        if row.kind not in _VALID_KINDS:
            raise ValueError(
                f"Ledger row.kind must be one of {sorted(_VALID_KINDS)}, got {row.kind!r}"
            )
        if row.status not in _VALID_STATUSES:
            raise ValueError(
                f"Ledger row.status must be one of {sorted(_VALID_STATUSES)}, got {row.status!r}"
            )
        if not row.created_at:
            row.created_at = datetime.now(UTC).isoformat()
        payload = json.dumps(row.to_dict(), default=_json_default) + "\n"
        with self.lock:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            self._appends_since_rebuild += 1
            if self._appends_since_rebuild >= self._index_throttle:
                self._rebuild_index_locked()
                self._appends_since_rebuild = 0

    def flush(self) -> None:
        """Force a rebuild of the ``index.json`` mirror."""
        with self.lock:
            self._rebuild_index_locked()
            self._appends_since_rebuild = 0

    def update_status(self, run_id: str, status: str) -> None:
        if status not in _VALID_STATUSES:
            raise ValueError(f"status must be one of {sorted(_VALID_STATUSES)}, got {status!r}")
        with self.lock:
            rows = self._read_all_strict()
            found = False
            for r in rows:
                if r.run_id == run_id:
                    r.status = status
                    found = True
            if not found:
                raise KeyError(f"Unknown run_id: {run_id}")
            self._rewrite_all(rows)
            # A status change is a semantically significant event; rebuild
            # the mirror unconditionally so any subsequent reader sees it.
            self._rebuild_index_locked()
            self._appends_since_rebuild = 0

    def update_stage(self, run_id: str, stage: int) -> None:
        """Correct the recorded stage of a run (eval bookkeeping).

        Eval runs append their ledger row before the model is built, when
        only the default ``train.stage`` (stage 1) is known. The evaluated
        checkpoint's real stage is known once the model is restored; the
        eval flow calls this then so ``runs.jsonl`` agrees with
        ``run_meta.json`` and the results row. Mirrors :meth:`update_status`
        (strict rewrite + unconditional mirror rebuild).
        """
        if not isinstance(stage, int) or isinstance(stage, bool) or stage < 0:
            raise ValueError(f"stage must be a non-negative int, got {stage!r}")
        with self.lock:
            rows = self._read_all_strict()
            found = False
            for r in rows:
                if r.run_id == run_id:
                    r.stage = stage
                    found = True
            if not found:
                raise KeyError(f"Unknown run_id: {run_id}")
            self._rewrite_all(rows)
            self._rebuild_index_locked()
            self._appends_since_rebuild = 0

    def _read_all_strict(self) -> list[LedgerRow]:
        """Read every row, raising on corruption (unlike :meth:`read_all`).

        Called only before a rewrite so a corrupted row is surfaced to the
        caller instead of being silently dropped and lost forever. Only a
        truncated trailing line (crash mid-append) is tolerated.
        """
        if not self.jsonl_path.exists():
            return []
        text = self.jsonl_path.read_text(encoding="utf-8")
        ends_with_newline = text.endswith(("\n", "\r"))
        lines = text.splitlines()
        out: list[LedgerRow] = []
        for index, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            is_last = index == len(lines) - 1
            try:
                data = json.loads(stripped)
                out.append(LedgerRow(**data))
            except json.JSONDecodeError:
                if is_last and not ends_with_newline:
                    continue
                raise
            except (TypeError, ValueError):
                if is_last and not ends_with_newline:
                    continue
                raise
        return out

    def read_all(self) -> list[LedgerRow]:
        """Read every row.

        Tolerates a truncated trailing line (crash mid-append); the
        partial JSON line at EOF is silently skipped. Earlier truncated
        lines are also skipped — a complete row that was corrupted
        without a crash is a real corruption event that callers should
        investigate, but the ledger stays readable rather than crashing
        on the first bad line.
        """
        if not self.jsonl_path.exists():
            return []
        out: list[LedgerRow] = []
        for line in self.jsonl_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            try:
                out.append(LedgerRow(**data))
            except (TypeError, ValueError):
                continue
        return out

    def find_by_id(self, run_id: str) -> LedgerRow | None:
        for row in self.read_all():
            if row.run_id == run_id:
                return row
        return None

    def _rewrite_all(self, rows: list[LedgerRow]) -> None:
        """Rewrite ``runs.jsonl`` from scratch (caller holds the lock).

        Uses ``mkstemp`` + ``os.replace`` so a crash mid-rewrite leaves
        either the old file intact or the new file in place, never a
        truncated mix of both.
        """
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.jsonl_path.parent),
            prefix=".ledger_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row.to_dict(), default=_json_default) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.jsonl_path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def _rebuild_index_locked(self) -> None:
        """Rebuild ``index.json`` from ``runs.jsonl``.

        Caller MUST hold :attr:`lock`. Reads the JSONL from disk so the
        rebuild reflects the on-disk truth, not the in-memory state.
        """
        rows = self.read_all()
        index = {
            "rebuilt_at": datetime.now(UTC).isoformat(),
            "count": len(rows),
            "runs": [row.to_dict() for row in rows],
        }
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.index_path.parent),
            prefix=".index_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(index, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.index_path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass


__all__ = ["RunLedger", "LedgerRow"]

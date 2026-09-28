"""Runner state registry: resumable record of every step the sweep ran.

``outputs/_runner/state.json`` is the runner's own bookkeeping. It records,
per ``(method, seed, phase)``, whether the step completed (and its run
directory / checkpoint) or failed (with the error). It exists so that:

* a killed sweep resumes exactly where it stopped (skip completed steps),
* the evaluation always targets the *exact* checkpoint a stage produced
  (no fuzzy "newest match" that a later manual run could silently change),
* ``--force``/``--continue-on-error`` have a precise, inspectable effect.

The registry is authoritative only for what *this runner* did; it never
overrides the provenance ledgers (``_ledger`` / ``_results``) written by
the CLI runs themselves.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_STATE_VERSION = 1


class RegistryError(RuntimeError):
    """Raised on an unusable or corrupt runner state file."""


class RunnerState:
    """Thin JSON store for sweep step status, with atomic writes.

    ``expected_commit`` (when non-empty) gates every completion query: a
    step recorded by an *earlier* git commit is treated as not complete, so
    a sweep re-run after a code change re-executes stale cells instead of
    silently consuming pre-fix checkpoints. Entries written before commit
    tracking existed (no ``git_commit`` field) are stale under gating too.
    """

    def __init__(
        self,
        path: str | Path,
        expected_commit: str | None = None,
    ) -> None:
        self.path = Path(path)
        self._expected_commit = (expected_commit or "").strip() or None
        self._runs: dict[str, dict[str, dict[str, dict[str, Any]]]] = self._load()

    @classmethod
    def default_path(cls, outputs_base: str | Path) -> Path:
        return Path(outputs_base) / "_runner" / "state.json"

    def _load(self) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
        if not self.path.is_file():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RegistryError(
                f"Runner state {self.path} is corrupt ({exc.msg}). Refusing to "
                "overwrite it silently — move it aside or fix it, then re-run."
            ) from exc
        except OSError as exc:
            raise RegistryError(f"Cannot read runner state {self.path}: {exc}") from exc
        runs = raw.get("runs") if isinstance(raw, dict) else None
        if not isinstance(runs, dict):
            raise RegistryError(f"Runner state {self.path} has no 'runs' object.")
        return runs

    def get(self, method: str, seed: int, phase: str) -> dict[str, Any] | None:
        entry = self._runs.get(method, {}).get(str(seed), {}).get(phase)
        return dict(entry) if isinstance(entry, dict) else None

    def _entry_current(self, entry: dict[str, Any] | None) -> bool:
        """Return whether the entry was produced by the expected git commit.

        Under no gating (``expected_commit`` unset) every entry is current.
        Under gating, an entry with a missing or different ``git_commit`` is
        stale — the sweep must re-run it rather than reuse pre-fix artifacts.
        """
        if self._expected_commit is None:
            return True
        return entry is not None and entry.get("git_commit") == self._expected_commit

    def is_complete(self, method: str, seed: int, phase: str) -> bool:
        entry = self.get(method, seed, phase)
        return (
            entry is not None
            and entry.get("status") == "completed"
            and self._entry_current(entry)
        )

    def mark(self, method: str, seed: int, phase: str, **fields: Any) -> None:
        fields["git_commit"] = self._expected_commit or ""
        self._set(method, seed, phase, status="completed", **fields)

    def mark_failed(self, method: str, seed: int, phase: str, error: str) -> None:
        self._set(method, seed, phase, status="failed", error=str(error)[:2000])

    def get_ckpt(self, method: str, seed: int, stage: int) -> str | None:
        entry = self.get(method, seed, f"stage{stage}")
        if (
            entry
            and entry.get("status") == "completed"
            and self._entry_current(entry)
        ):
            ckpt = entry.get("ckpt")
            return ckpt if isinstance(ckpt, str) else None
        return None

    def _set(self, method: str, seed: int, phase: str, **fields: Any) -> None:
        self._runs.setdefault(method, {}).setdefault(str(seed), {})[phase] = fields
        self.save()

    def save(self) -> None:
        payload = {
            "version": _STATE_VERSION,
            "updated_at": datetime.now(UTC).isoformat(),
            "runs": self._runs,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), prefix=".state_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

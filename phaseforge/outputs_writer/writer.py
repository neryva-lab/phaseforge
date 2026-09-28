"""RunWriter — lifecycle markers and auxiliary artifacts for one run.

A :class:`RunWriter` wraps an existing run directory (the
``outputs/{model}/stage{N}/{ts}_{runid}/`` paths produced by
:func:`phaseforge.utils.config.get_output_dir`) with:

* sibling lifecycle markers (``<run_dir>.pending``, ``<run_dir>.completed``,
  ``<run_dir>.failed``) — sibling files (not in the run dir) so the marker
  survives a future rename or move of the run dir;
* ``metadata/environment.json`` — the dep / git / dataset-cache fingerprint;
* ``timings.json`` — ``started_at``, ``finished_at``, ``wall_seconds`` for
  the paper appendix.

The class does not create or own the run directory itself — that is
:func:`phaseforge.utils.config.get_output_dir`'s job — and it does not
write the canonical config / checkpoints / eval-results files, which the
CLI owns and writes before/after the writer.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_STATUS_PENDING = "pending"
_STATUS_COMPLETED = "completed"
_STATUS_FAILED = "failed"


def parse_run_dir(name: str) -> tuple[str, str | None, str]:
    """Parse a run directory name into ``(timestamp, tag, run_id)``.

    Accepted formats (produced by ``get_output_dir``):

    * no tag:    ``YYYY-MM-DD_HH-MM-SS_XXXXXXXX``
    * with tag:  ``YYYY-MM-DD_HH-MM-SS_<tag>_XXXXXXXX``

    where ``XXXXXXXX`` is always an 8-char hex run_id. Non-matching names
    return ``(name, None, "")`` so the legacy fallback in
    :func:`phaseforge.utils.config.scan_checkpoints` and the new ledger
    share the same tolerance.
    """
    tail = name.rsplit("_", 1)
    if len(tail) == 2 and len(tail[1]) == 8:
        run_id = tail[1]
        head_parts = tail[0].split("_", 2)
        if len(head_parts) >= 2:
            timestamp = f"{head_parts[0]}_{head_parts[1]}"
            tag = "_".join(head_parts[2:]) if len(head_parts) > 2 else None
            return timestamp, tag, run_id
        return tail[0], None, run_id
    return name, None, ""


def _iso_now() -> str:
    return datetime.now(UTC).isoformat()


class RunWriter:
    """Lifecycle markers + auxiliary artifacts for ONE run directory."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        started_at: str | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        if not self.run_dir.is_dir():
            raise FileNotFoundError(
                f"RunWriter needs an existing run directory; {self.run_dir} does not exist"
            )
        self.started_at = started_at or _iso_now()
        self._closed = False
        self._status = _STATUS_PENDING
        self._write_lifecycle_marker(_STATUS_PENDING)
        (self.run_dir / "metadata").mkdir(exist_ok=True)

    @property
    def status(self) -> str:
        return self._status

    def mark_completed(self) -> Path:
        """Mark the run as successfully completed. Writes ``timings.json``.

        Raises ``RuntimeError`` if called twice or after :meth:`mark_failed`.
        """
        if self._closed:
            raise RuntimeError("RunWriter already closed")
        self._write_timings(finished=True)
        self._closed = True
        self._status = _STATUS_COMPLETED
        return self._write_lifecycle_marker(_STATUS_COMPLETED)

    def mark_failed(self, exc: BaseException | None = None) -> Path:
        """Mark the run as failed. Writes ``exception.txt`` and ``timings.json``.

        Raises ``RuntimeError`` if called twice or after :meth:`mark_completed`.
        """
        if self._closed:
            raise RuntimeError("RunWriter already closed")
        self._write_timings(finished=False)
        self._closed = True
        self._status = _STATUS_FAILED
        path = self._write_lifecycle_marker(_STATUS_FAILED)
        if exc is not None:
            (self.run_dir / "logs").mkdir(exist_ok=True)
            (self.run_dir / "logs" / "exception.txt").write_text(repr(exc), encoding="utf-8")
        return path

    def _write_lifecycle_marker(self, status: str) -> Path:
        """Write ``<run_dir>.<status>`` sibling marker; drop the prior one.

        Sibling (not inside the run dir) so the marker survives renames
        or moves of the run dir.
        """
        marker = self.run_dir.with_name(self.run_dir.name + "." + status)
        if status != _STATUS_PENDING:
            pending = self.run_dir.with_name(self.run_dir.name + ".pending")
            if pending.exists():
                try:
                    pending.unlink()
                except OSError:
                    pass
        marker.write_text(
            json.dumps(
                {
                    "run_dir": self.run_dir.name,
                    "path": str(self.run_dir),
                    "status": status,
                    "stamped_at": _iso_now(),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return marker

    def write_environment(self, env: dict[str, Any]) -> Path:
        """Persist :func:`phaseforge.outputs_writer.metadata.collect_environment` output."""
        path = self.run_dir / "metadata" / "environment.json"
        path.write_text(
            json.dumps(env, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        return path

    def _write_timings(self, *, finished: bool) -> Path:
        finished_at = _iso_now()
        try:
            started = datetime.fromisoformat(self.started_at)
            wall = (datetime.fromisoformat(finished_at) - started).total_seconds()
        except ValueError:
            wall = float("nan")
        path = self.run_dir / "timings.json"
        payload = {
            "started_at": self.started_at,
            "finished_at": finished_at,
            "wall_seconds": float(wall),
            "status": _STATUS_COMPLETED if finished else _STATUS_FAILED,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path


__all__ = ["RunWriter", "parse_run_dir"]

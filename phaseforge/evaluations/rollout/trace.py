"""Per-step rollout tracing.

``trace.jsonl`` holds one JSON row per executed environment step (only when
``eval.episodes.trace_level=full``; ``minimal`` keeps the
``episodes.jsonl``-only behavior). Each row carries the 22 spec fields in
spec order; values a model cannot provide are explicit ``null`` (never
fabricated). Rows are schema-validated before append; the file uses the
same append-only + filelock discipline as ``episodes.jsonl``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from filelock import FileLock

#: The 22 trace fields, in spec order.
TRACE_FIELDS: tuple[str, ...] = (
    "episode_id",
    "case_id",
    "timestep",
    "termination_reason",
    "raw_obs_summary",
    "normalized_state_norm",
    "task_vars",
    "latent_norm",
    "dists",
    "selected_expert",
    "top2_expert",
    "router_margin",
    "router_entropy",
    "expert_target",
    "expert_gains",
    "task_error",
    "pre_clip_command",
    "final_action",
    "nearest_train_dist",
    "expert_disagreement",
    "lip_diagnostic",
    "done",
)

#: Steps between Lipschitz finite-difference samples (every 10th step).
LIP_SAMPLE_PERIOD = 10

#: Cap on reference states for the nearest-train-distance proxy.
REFERENCE_STATE_CAP = 2048


class TraceSchemaError(ValueError):
    """Raised when a trace row is malformed; no file write occurs."""


def to_jsonable(value: Any) -> Any:
    """Convert tensors/arrays/scalars to plain JSON values (None passes through)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return tolist()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return str(value)


def validate_trace_record(row: dict[str, Any]) -> None:
    """Strict validator for one ``trace.jsonl`` row."""
    if not isinstance(row, dict):
        raise TraceSchemaError(f"Trace record must be a dict, got {type(row).__name__}.")
    missing = [k for k in TRACE_FIELDS if k not in row]
    if missing:
        raise TraceSchemaError(f"Trace record missing required keys: {missing}.")
    unknown = sorted(set(row) - set(TRACE_FIELDS))
    if unknown:
        raise TraceSchemaError(f"Trace record has unknown top-level keys: {unknown}.")
    for key in ("episode_id", "case_id", "timestep"):
        if not isinstance(row[key], int) or isinstance(row[key], bool):
            raise TraceSchemaError(f"Trace record[{key!r}] must be int.")
    if not isinstance(row["done"], bool):
        raise TraceSchemaError("Trace record['done'] must be bool.")
    for key in ("termination_reason",):
        if not isinstance(row[key], str):
            raise TraceSchemaError(f"Trace record[{key!r}] must be str.")
    for key in (
        "normalized_state_norm",
        "latent_norm",
        "router_margin",
        "router_entropy",
        "nearest_train_dist",
        "expert_disagreement",
        "lip_diagnostic",
    ):
        value = row[key]
        if value is not None and not isinstance(value, (int, float)):
            raise TraceSchemaError(f"Trace record[{key!r}] must be a number or null.")
    for key in ("selected_expert", "top2_expert"):
        value = row[key]
        if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
            raise TraceSchemaError(f"Trace record[{key!r}] must be int or null.")
    for key in (
        "raw_obs_summary",
        "task_vars",
        "dists",
        "expert_target",
        "expert_gains",
        "task_error",
        "pre_clip_command",
        "final_action",
    ):
        value = row[key]
        if value is not None and not isinstance(value, (list, dict)):
            raise TraceSchemaError(f"Trace record[{key!r}] must be a list/dict or null.")


class TraceWriter:
    """Append-only validated writer for ``trace.jsonl`` (one per eval run)."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.target = self.output_dir / "trace.jsonl"
        self.lock = FileLock(str(self.output_dir / ".trace.lock"))

    def append_episode_rows(self, rows: list[dict[str, Any]]) -> Path:
        """Validate and append one episode's step rows (no-op for zero rows)."""
        for row in rows:
            validate_trace_record(row)
        if not rows:
            return self.target
        payload = "".join(json.dumps(row) + "\n" for row in rows)
        with self.lock:
            with open(self.target, "a", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        return self.target


def read_trace_rows(output_dir: str | Path) -> list[dict[str, Any]]:
    """Read every trace row from an eval run directory (tolerant)."""
    target = Path(output_dir) / "trace.jsonl"
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            rows.append(json.loads(stripped))
    return rows


def load_reference_states(cache_dir: str | Path, cap: int = REFERENCE_STATE_CAP) -> Any | None:
    """Load a deterministic reference subsample of train states or None.

    Reads the first ``cap`` states (file order, truncated) from the
    processed cache's trajectories. Any failure returns None (the runner
    records null OOD scores instead of breaking the rollout).
    """
    try:
        import torch

        files = sorted((Path(cache_dir) / "trajectories").glob("*.pt"))
        if not files:
            return None
        states: list[Any] = []
        total = 0
        for path in files:
            traj = torch.load(str(path), map_location="cpu", weights_only=False)
            chunk = traj["state"]
            states.append(chunk)
            total += chunk.shape[0]
            if total >= cap:
                break
        merged = torch.cat(states, dim=0)[:cap]
        return np.asarray(merged.numpy(), dtype=np.float64)
    except Exception:
        return None


__all__ = [
    "LIP_SAMPLE_PERIOD",
    "REFERENCE_STATE_CAP",
    "TRACE_FIELDS",
    "TraceSchemaError",
    "TraceWriter",
    "load_reference_states",
    "read_trace_rows",
    "to_jsonable",
    "validate_trace_record",
]

"""Typed loader for per-episode rollout records (episodes.jsonl)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from studies.analysis.common import io as cio


@dataclass(frozen=True)
class EpisodeRecord:
    """One rollout episode; ``episode_index`` is the pairing key across methods."""

    episode_index: int
    success: bool
    valid: bool
    steps: int
    timed_out: bool
    termination_reason: str
    max_phase: int | None
    task: str | None
    training_seed: int | None
    run_id: str | None

    @classmethod
    def from_row(cls, row: dict) -> EpisodeRecord:
        try:
            extra = row.get("extra") or {}
            return cls(
                episode_index=int(row["episode_index"]),
                success=bool(row["success"]),
                valid=bool(row.get("valid_episode", True)),
                steps=int(row.get("steps", 0)),
                timed_out=bool(row.get("timed_out", False)),
                termination_reason=str(row.get("termination_reason", "")),
                max_phase=int(extra["max_phase"]) if "max_phase" in extra else None,
                task=row.get("task"),
                training_seed=row.get("training_seed"),
                run_id=row.get("run_id"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed episode record ({exc}): {row!r}") from exc


def load_episodes(run_dir: Path) -> list[EpisodeRecord]:
    records = [EpisodeRecord.from_row(r) for r in cio.iter_jsonl(run_dir / "episodes.jsonl")]
    if not records:
        raise ValueError(f"{run_dir / 'episodes.jsonl'}: no episode records")
    return records

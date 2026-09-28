"""Per-episode paired comparisons on identical reset cases.

The frozen reset bank guarantees every method sees the same reset-case
indices; pairing on ``episode_index`` is therefore the strongest allowed
comparison form (figures_tables_plan.md section 1).
"""

from __future__ import annotations

from dataclasses import dataclass

from studies.analysis.loaders.episodes import EpisodeRecord
from studies.analysis.stats.intervals import wilson_interval


@dataclass(frozen=True)
class PairedOutcome:
    task: str | None
    seed: int
    n_cases: int
    successes_a: int
    successes_b: int
    only_a: int
    only_b: int
    neither: int
    bank_a: str | None
    bank_b: str | None

    @property
    def delta(self) -> float:
        """Success-rate difference A - B over the shared cases."""
        return (self.successes_a - self.successes_b) / self.n_cases

    @property
    def discordant(self) -> int:
        return self.only_a + self.only_b

    @property
    def wilson_on_delta(self) -> tuple[float, float]:
        """Wilson interval for the per-case 'A succeeds' proportion minus a
        constant 0.5 shift is not defined; we report the Wilson interval of
        the *net* advantage proportion among discordant pairs (paired sign
        style) when discordance exists, else (0, 0)."""
        if self.discordant == 0:
            return 0.0, 0.0
        lo, hi = wilson_interval(self.only_a, self.discordant)
        return lo - 0.5, hi - 0.5


def pair_episodes(
    task: str | None,
    seed: int,
    episodes_a: list[EpisodeRecord],
    episodes_b: list[EpisodeRecord],
    bank_a: str | None = None,
    bank_b: str | None = None,
) -> PairedOutcome:
    """Join two episode lists on reset-case index; fail closed on mismatched case sets."""
    valid_a = {e.episode_index: e for e in episodes_a if e.valid}
    valid_b = {e.episode_index: e for e in episodes_b if e.valid}
    shared = sorted(set(valid_a) & set(valid_b))
    if not shared:
        raise ValueError(
            f"No shared reset cases for task={task!r} seed={seed}: "
            f"{len(valid_a)} vs {len(valid_b)} valid episodes"
        )
    only_in_a = set(valid_a) - set(valid_b)
    only_in_b = set(valid_b) - set(valid_a)
    if only_in_a or only_in_b:
        raise ValueError(
            f"Reset-case mismatch for task={task!r} seed={seed}: cases only in A="
            f"{sorted(only_in_a)[:5]}... only in B={sorted(only_in_b)[:5]}... — "
            "banks differ; paired comparison is invalid"
        )
    sa = sum(1 for i in shared if valid_a[i].success)
    sb = sum(1 for i in shared if valid_b[i].success)
    only_a = sum(1 for i in shared if valid_a[i].success and not valid_b[i].success)
    only_b = sum(1 for i in shared if valid_b[i].success and not valid_a[i].success)
    return PairedOutcome(
        task=task,
        seed=seed,
        n_cases=len(shared),
        successes_a=sa,
        successes_b=sb,
        only_a=only_a,
        only_b=only_b,
        neither=len(shared) - only_a - only_b,
        bank_a=bank_a,
        bank_b=bank_b,
    )

"""Tests for the rollout episode statistics (McNemar, Newcombe, Holm, summaries)."""

from __future__ import annotations

import math

import pytest

from phaseforge.outputs_writer.episodes import (
    POLICY_FAILURE_CATEGORIES,
    holm_bonferroni,
    mcnemar_exact_p,
    newcombe_ci,
    paired_rollout_comparisons,
    summarize_episodes,
    wilson_interval,
)


def _row(
    index: int,
    model: str,
    seed: int = 42,
    *,
    valid: bool = True,
    success: bool = True,
    failure_category: str | None = None,
    tag: str | None = None,
) -> dict:
    row = {
        "run_id": "r1",
        "model": model,
        "checkpoint_sha256": "abc",
        "task": "Lift",
        "training_seed": seed,
        "reset_seed": 2026,
        "episode_index": index,
        "valid_episode": valid,
        "steps": 100,
    }
    if tag is not None:
        row["tag"] = tag
    if valid:
        row["success"] = success
        row["timed_out"] = not success
        if not success:
            row["failure_category"] = failure_category or "task_timeout"
    return row


class TestMcNemar:
    def test_no_discordant_pairs(self) -> None:
        assert mcnemar_exact_p(0, 0) == 1.0

    def test_balanced_discordants(self) -> None:
        # b=c=5, n=10: p = 2 * P(Bin(10,.5) <= 5) = 2 * 0.623046875 = 1.246… → 1.0
        assert mcnemar_exact_p(5, 5) == 1.0

    def test_known_small_case(self) -> None:
        # b=10, c=0, n=10: p = 2 * P(Bin(10,.5) <= 0) = 2 * (1/1024) ≈ 0.001953125
        assert mcnemar_exact_p(10, 0) == pytest.approx(2 / 2**10)

    def test_symmetric(self) -> None:
        assert mcnemar_exact_p(4, 2) == mcnemar_exact_p(2, 4)

    def test_clamped_to_one(self) -> None:
        assert mcnemar_exact_p(25, 25) == 1.0


class TestNewcombe:
    def test_equal_rates(self) -> None:
        low, high = newcombe_ci(5, 10, 5, 10)
        # wilson(0.5, 10) = (0.2366, 0.7634); Newcombe = ±0.2634*sqrt(2)
        assert low == pytest.approx(-0.3725, abs=1e-3)
        assert high == pytest.approx(0.3725, abs=1e-3)

    def test_perfect_vs_zero(self) -> None:
        # 10/10 vs 0/10: CI must exclude 0.
        low, high = newcombe_ci(10, 10, 0, 10)
        assert low > 0
        assert high <= 1.0

    def test_nan_on_zero_denominator(self) -> None:
        low, high = newcombe_ci(0, 0, 5, 10)
        assert math.isnan(low) and math.isnan(high)


class TestHolm:
    def test_adjustment_monotone(self) -> None:
        adjusted = holm_bonferroni([0.01, 0.02, 0.04])
        # m=3, sorted p = [0.01, 0.02, 0.04]; adjusted values are the
        # monotone non-decreasing cumulative maximum of (m-rank) * p:
        #   3*0.01=0.03, max(0.03, 2*0.02=0.04)=0.04, max(0.04, 1*0.04)=0.04.
        assert adjusted == [0.03, 0.04, 0.04]
        assert all(a <= b for a, b in zip(adjusted, adjusted[1:]))

    def test_empty(self) -> None:
        assert holm_bonferroni([]) == []

    def test_never_exceeds_one(self) -> None:
        assert all(p <= 1.0 for p in holm_bonferroni([0.5, 0.6, 0.9]))


class TestSummarize:
    def test_policy_failures_labeled(self) -> None:
        rows = [
            _row(0, "m", success=True),
            _row(1, "m", success=False, failure_category="policy_invalid_action"),
            _row(2, "m", success=False, failure_category="task_timeout"),
            _row(3, "m", valid=False),
        ]
        summary = summarize_episodes(rows)[0]
        assert summary["valid_episodes"] == 3
        assert summary["successes"] == 1
        assert summary["policy_failures"] == 1
        assert summary["invalid_attempts"] == 1
        assert summary["success_rate"] == pytest.approx(1 / 3)

    def test_wilson_interval_zero_and_full(self) -> None:
        assert wilson_interval(0, 10) == (0.0, 0.0)
        assert wilson_interval(10, 10) == (1.0, 1.0)
        assert math.isnan(wilson_interval(0, 0)[0])


class TestPairedComparisons:
    def test_case_level_mcnemar(self) -> None:
        base = [_row(i, "precision_residual_phaseforge", success=True) for i in range(5)]
        other = [_row(i, "bc", success=False, failure_category="task_timeout") for i in range(5)]
        rows = base + other
        rows.append(_row(5, "bc", success=False, failure_category="task_timeout"))
        comparisons = paired_rollout_comparisons(rows)
        assert len(comparisons) == 1
        comp = comparisons[0]
        assert comp["model"] == "bc"
        # 5 cases paired: baseline won all 5 → b=5, c=0
        assert comp["discordant_baseline_wins"] == 5
        assert comp["discordant_model_wins"] == 0
        assert comp["mcnemar_exact_p"] == pytest.approx(2 / 2**5)
        # Effect direction is PhaseForge minus comparator.
        assert comp["diff"] == pytest.approx(1.0)

    def test_invalid_episodes_excluded_from_pairing(self) -> None:
        base = [_row(i, "precision_residual_phaseforge", success=True) for i in range(3)]
        other = [_row(i, "bc", valid=False) for i in range(3)]
        comparisons = paired_rollout_comparisons(base + other)
        # no valid episodes on the model side → no pair row at all
        assert comparisons == []

    def test_tagged_variant_not_paired(self) -> None:
        rows = [_row(i, "precision_residual_phaseforge", success=True) for i in range(2)]
        rows += [
            _row(i, "precision_residual_phaseforge", success=True, tag="robot_only")
            for i in range(2)
        ]
        assert paired_rollout_comparisons(rows) == []

    def test_policy_failure_categories_frozen(self) -> None:
        assert POLICY_FAILURE_CATEGORIES == {
            "policy_invalid_action",
            "policy_exception",
        }

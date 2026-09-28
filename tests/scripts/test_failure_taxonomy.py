"""CPU-only tests for the failure taxonomy + confirmation gate."""

from __future__ import annotations

import pytest

from scripts.analysis.classify_failures import Thresholds, classify_episode, summarize_runs
from scripts.analysis.confirm_gate import decide


def _row(timestep: int, termination: str = "task_timeout", **overrides):
    row = {
        "episode_id": 0,
        "case_id": 0,
        "timestep": timestep,
        "termination_reason": termination,
        "raw_obs_summary": {"state_norm": 1.0, "eef_pos": [0.0, 0.0, 1.0]},
        "normalized_state_norm": 1.0,
        "task_vars": [0.0] * 8,
        "latent_norm": 1.0,
        "dists": [0.5, 0.6],
        "selected_expert": 0,
        "top2_expert": 1,
        "router_margin": 0.5,
        "router_entropy": 0.1,
        "expert_target": [0.0] * 8,
        "expert_gains": [1.0] * 7,
        "task_error": [0.0] * 7,
        "pre_clip_command": [0.1] * 7,
        "final_action": [0.0] * 7,
        "nearest_train_dist": 1.0,
        "expert_disagreement": 0.01,
        "lip_diagnostic": None,
        "done": timestep == 4,
    }
    row.update(overrides)
    return row


def _episode(n: int = 5, **overrides):
    return [_row(t, **overrides) for t in range(n)]


def test_outcomes_pass_through() -> None:
    assert classify_episode(_episode(termination="success")) == "success"
    assert classify_episode(_episode(termination="infrastructure")) == "infrastructure"
    assert (
        classify_episode(_episode(termination="policy_invalid_action"))
        == "policy_invalid_action"
    )
    assert classify_episode([]) == "timeout_unclassified"


def test_gain_collapse_fires() -> None:
    rows = _episode(expert_gains=[1e-4] * 7)
    assert classify_episode(rows) == "gain_collapse"


def test_action_saturation_fires() -> None:
    rows = _episode(pre_clip_command=[5.0] * 7)
    assert classify_episode(rows) == "action_saturation"


def test_routing_ambiguity_soft_and_hard() -> None:
    soft = _episode(router_margin=0.01, router_entropy=0.9)
    assert classify_episode(soft) == "routing_ambiguity"
    flipping = [
        _row(t, router_margin=0.01, router_entropy=None, selected_expert=t % 2)
        for t in range(5)
    ]
    assert classify_episode(flipping) == "routing_ambiguity"


def test_conflict_ood_chasing_controller_order() -> None:
    assert classify_episode(_episode(expert_disagreement=0.9)) == "expert_conflict"
    assert classify_episode(_episode(nearest_train_dist=9.0)) == "ood_drift"
    chasing = [
        _row(t, expert_target=[float(t)] + [0.0] * 7, task_vars=[0.0] * 8) for t in range(5)
    ]
    assert classify_episode(chasing) == "target_chasing"
    still = [_row(t) for t in range(5)]
    assert classify_episode(still) == "controller_limit"


def test_plain_timeout_is_unclassified() -> None:
    rows = [
        _row(t, raw_obs_summary={"state_norm": 1.0, "eef_pos": [float(t), 0.0, 1.0]})
        for t in range(5)
    ]
    assert classify_episode(rows) == "timeout_unclassified"


def test_reset_geometry_overrides_unanimous_failures() -> None:
    run_a = [_row(t, episode_id=0, case_id=0) for t in range(5)]
    run_b = [_row(t, episode_id=0, case_id=0) for t in range(5)]
    report = summarize_runs({"a": run_a, "b": run_b})
    assert report["reset_geometry_cases"] == [0]
    assert all(r["class"] == "reset_geometry" for r in report["episodes"])
    assert report["counts"]["reset_geometry"] == 2


def test_no_reset_geometry_without_unanimity() -> None:
    good = [_row(t, episode_id=0, case_id=0, termination="success") for t in range(5)]
    bad = [_row(t, episode_id=0, case_id=0) for t in range(5)]
    report = summarize_runs({"a": good, "b": bad})
    assert report["reset_geometry_cases"] == []
    assert report["counts"]["success"] == 1


def test_confirm_gate_decisive_and_hold() -> None:
    def _results(rate: float, low: float, high: float) -> dict:
        return {
            "eval/rollout/success_rate": rate,
            "eval/rollout/wilson_ci95_low": low,
            "eval/rollout/wilson_ci95_high": high,
        }

    proceed, report = decide(
        _results(0.7, 0.62, 0.78),
        [_results(0.5, 0.4, 0.58), _results(0.45, 0.3, 0.55)],
    )
    assert proceed is True
    assert report["best_control_wilson_high"] == pytest.approx(0.58)
    hold, _report = decide(_results(0.55, 0.42, 0.68), [_results(0.5, 0.4, 0.6)])
    assert hold is False
    with pytest.raises(KeyError):
        decide({"eval/rollout/success_rate": 0.5}, [])


def test_thresholds_defaults_exist() -> None:
    assert Thresholds().clip_frac == pytest.approx(0.8)

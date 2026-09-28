"""Tests for the rollout runner: strict-metric episode classification."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from phaseforge.data.common.normalizer import FrozenNormalizer
from phaseforge.data.robomimic.phase_labeler import CausalPhaseStepLabeler
from phaseforge.evaluations.envs.errors import EnvParityError, InfrastructureError
from phaseforge.evaluations.rollout.runner import (
    FAILURE_POLICY_EXCEPTION,
    FAILURE_POLICY_INVALID_ACTION,
    FAILURE_TIMEOUT,
    RolloutEvaluator,
    RolloutRunInvalid,
    require_rollout_eval_schema,
    resolve_pinned_metadata,
    resolve_robosuite_requirement,
    run_rollout_evaluation,
)
from phaseforge.outputs_writer.episodes import read_episode_records
from tests.rollout_helpers import FakeAdapter, make_bank


def test_task_source_robosuite_pin_overrides_eval_default() -> None:
    from omegaconf import DictConfig

    cfg = DictConfig(
        {
            "data": {"source": {"robosuite_requirement": "==1.5.0"}},
            "eval": {"env": {"robosuite_requirement": "==1.5.1"}},
        }
    )
    assert resolve_robosuite_requirement(cfg) == "==1.5.0"


def test_legacy_eval_robosuite_pin_is_fallback() -> None:
    from omegaconf import DictConfig

    cfg = DictConfig(
        {
            "data": {"source": {}},
            "eval": {"env": {"robosuite_requirement": "==1.5.1"}},
        }
    )
    assert resolve_robosuite_requirement(cfg) == "==1.5.1"


class TestResolvePinnedMetadata:
    """Fail-closed env metadata recovery.

    Regression: when neither the processed cache nor the raw HDF5 exists
    on the machine, ``resolve_pinned_metadata`` silently fell back to
    documented dev metadata — a real rollout could then run against the
    wrong task's environment and produce plausible-looking but invalid
    results. The fallback must be a hard error unless the caller opts in
    explicitly (``eval.env.allow_dev_fallback=true``, local gates only).
    """

    def _cfg(self, task: str = "Lift", allow_fallback: bool = False):
        from omegaconf import DictConfig

        return DictConfig(
            {
                "data": {
                    "source": {"dir": "/nonexistent/raw", "task_name": task},
                },
                "eval": {"env": {"allow_dev_fallback": allow_fallback}},
            }
        )

    def test_no_cache_no_raw_raises_without_opt_in(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(
            "phaseforge.data.paths.processed_cache_root",
            lambda: tmp_path / "no_such_cache",
        )

        cfg = self._cfg(task="Can")
        with pytest.raises(RuntimeError, match="Can"):
            resolve_pinned_metadata(cfg)

    def test_opt_in_dev_fallback_still_works(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(
            "phaseforge.data.paths.processed_cache_root",
            lambda: tmp_path / "no_such_cache",
        )
        monkeypatch.setattr(
            "phaseforge.evaluations.envs.env_metadata.dev_fallback_metadata",
            lambda task: {"task": task, "dev": True},
        )

        meta = resolve_pinned_metadata(self._cfg(task="Lift", allow_fallback=True))
        assert meta == {"task": "Lift", "dev": True}


class TestRequireRolloutEvalSchema:
    """Defensive guard at the rollout entry points.

    Regression: ``eval.mode=rollout`` alone leaves the default
    (``metrics``) config group in place, and the metrics schema has no
    ``bank``/``env``/``episodes`` sections. Without this guard the
    rollout path raises an obscure ``ConfigAttributeError`` deep in
    ``load_or_generate_bank``. The guard turns it into a single
    actionable ``EnvParityError`` before any expensive work runs.
    """

    def test_missing_sections_raises_actionable_error(self) -> None:
        from omegaconf import DictConfig

        # Only ``mode`` is set, just like the buggy runner argv.
        cfg = DictConfig({"eval": {"mode": "rollout"}})
        with pytest.raises(EnvParityError) as excinfo:
            require_rollout_eval_schema(cfg)
        message = str(excinfo.value)
        assert "eval=rollout" in message
        assert "bank" in message
        assert "env" in message
        assert "episodes" in message

    def test_missing_eval_group_raises_actionable_error(self) -> None:
        from omegaconf import DictConfig

        cfg = DictConfig({})
        with pytest.raises(EnvParityError, match="cfg.eval is missing entirely"):
            require_rollout_eval_schema(cfg)

    def test_partial_sections_lists_only_missing(self) -> None:
        from omegaconf import DictConfig

        cfg = DictConfig({"eval": {"bank": {}, "mode": "rollout"}})
        with pytest.raises(EnvParityError) as excinfo:
            require_rollout_eval_schema(cfg)
        message = str(excinfo.value)
        assert "env" in message
        assert "episodes" in message
        assert "bank" not in message.split("is missing section(s):", 1)[1]

    def test_full_rollout_schema_passes(self) -> None:
        from omegaconf import DictConfig

        cfg = DictConfig(
            {
                "eval": {
                    "mode": "rollout",
                    "bank": {"seed": 2026, "num_cases": 50},
                    "env": {"mujoco_requirement": ">=3.2.7"},
                    "episodes": {"action_tolerance": 1e-4},
                    "gates": {},
                }
            }
        )
        # Must not raise.
        require_rollout_eval_schema(cfg)

    def test_run_rollout_evaluation_raises_actionable_before_simulator(
        self, tmp_path
    ) -> None:
        """End-to-end: the guard fires inside ``run_rollout_evaluation``
        before any robosuite / simulator work."""
        from omegaconf import DictConfig

        cfg = DictConfig({"eval": {"mode": "rollout"}})
        dummy_model = torch.nn.Module()
        with pytest.raises(EnvParityError, match="eval=rollout"):
            run_rollout_evaluation(
                cfg,
                dummy_model,
                output_dir=tmp_path,
                run_id="test-run",
            )

    def test_run_all_gates_raises_actionable_before_simulator(self) -> None:
        """Same guard, wired into the gates entry point. The gates CLI
        composes ``main.yaml`` with ``eval=rollout`` per docs; this test
        pins that the gates path also fails loudly without it."""
        from omegaconf import DictConfig

        from phaseforge.evaluations.rollout.gates import run_all_gates

        cfg = DictConfig({"eval": {"mode": "rollout"}})
        with pytest.raises(EnvParityError, match="eval=rollout"):
            run_all_gates(cfg)


class _ZeroModel(torch.nn.Module):
    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        return torch.zeros(state.shape[0], 7)


class _NanModel(torch.nn.Module):
    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        return torch.full((state.shape[0], 7), float("nan"))


class _RaisingModel(torch.nn.Module):
    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("boom in the policy")


class _RangeModel(torch.nn.Module):
    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        return torch.full((state.shape[0], 7), 2.0)


class _Ticker:
    """Policy hook wrapper that advances an underlying policy's step counter."""

    def __init__(self, policy):
        self.policy = policy
        self.t = 0

    def reset(self) -> None:
        self.policy.reset()
        self.t = 0

    def __call__(self, state: np.ndarray) -> np.ndarray:
        action = self.policy.act(state, self.t)
        self.t += 1
        return action


class _MarkerPolicy:
    """Policy hook that delegates to a ticker unless ``state[2]`` is a marker."""

    def __init__(self, ticker: _Ticker):
        self._ticker = ticker

    def reset(self) -> None:
        self._ticker.reset()

    def __call__(self, state: np.ndarray) -> np.ndarray:
        if state[2] > 4.0:
            return np.full(7, np.nan)
        return self._ticker(state)


def _evaluator(
    tmp_path,
    adapter,
    model=None,
    normalizer=None,
    horizon=500,
    policy=None,
    phase_labeler=None,
    router_mode="learned",
):
    return RolloutEvaluator(
        cfg=None,  # type: ignore[arg-type]
        policy=policy,
        adapter=adapter,  # type: ignore[arg-type]
        bank=make_bank(),
        normalizer=normalizer or FrozenNormalizer(torch.zeros(19), torch.ones(19)),
        model=model,
        output_dir=tmp_path,
        run_id="testrun",
        model_name="precision_residual_phaseforge",
        training_seed=42,
        task="Lift",
        checkpoint_sha256="deadbeef",
        horizon=horizon,
        phase_labeler=phase_labeler,
        router_mode=router_mode,
    )


def _phase_calibration() -> dict:
    """Calibration artifact matching the fake Lift state layout (eef 0-2,
    quat 3-6, gripper qpos 7-8, object 9-18)."""
    return {
        "closed_level": 0.02,
        "open_level": 0.04,
        "mirror": False,
        "mirror_bounds": None,
        "velocity_threshold": 0.01,
        "min_duration": 5,
        "filter_size": 7,
        "eef_pos_slice": [0, 3],
        "gripper_qpos_slice": [7, 9],
        "num_phases": 6,
    }


def _phase_labeler() -> CausalPhaseStepLabeler:
    return CausalPhaseStepLabeler(_phase_calibration())


def _lift_controller():
    """Trivial valid-action policy used by the runner unit tests.

    Returns a small callable that emits a valid 7-dim neutral action every
    step. It is intentionally simple — these runner tests only need a
    policy that produces in-range actions, not one that solves the task.
    The fake simulator never reaches success with this policy (the cube
    does not move and the gripper does not close), so the affected tests
    assert ``successes == 0``.
    """
    return _ZeroPolicy()


class _ZeroPolicy:
    """A trivial valid-action policy (zero OSC delta, gripper stays open).

    Exposes ``act(state, t)`` and ``reset()`` so the ``_Ticker`` wrapper
    can drive it without changes.
    """

    def reset(self) -> None:  # noqa: D401 — runner calls reset() if present
        pass

    def act(self, state: np.ndarray, t: int) -> np.ndarray:  # noqa: ARG002
        action = np.zeros(7, dtype=np.float64)
        action[-1] = -1.0  # gripper open (matches FakeAdapter convention)
        return action

    def __call__(self, state: np.ndarray) -> np.ndarray:
        return self.act(state, 0)


class _SuccessAdapter(FakeAdapter):
    """Adapter wrapper that teleports the cube above the success threshold.

    Forces the simulator's success predicate (``cube.z > SUCCESS_Z``) to
    fire on the first step of every episode. Used to exercise the runner's
    success path without a hand-coded controller. The runner's success
    reporting is what we want to test; the geometric mechanism of *how*
    success is reached is the fake's responsibility, not the policy's.
    """

    def reset_to(self, states, *, xml=None, ep_meta=None) -> np.ndarray:
        super().reset_to(states, xml=xml, ep_meta=ep_meta)
        # Lift the cube above the success threshold (0.84 + epsilon).
        self.sim.cube = np.array([self.sim.cube[0], self.sim.cube[1], 0.90])
        return self.extract_state(self.sim.state)


class TestEpisodeOutcomes:
    def test_zero_policy_times_out(self, tmp_path) -> None:
        evaluator = _evaluator(tmp_path, FakeAdapter(horizon=500), model=_ZeroModel(), horizon=500)
        results = evaluator.run()
        assert results["eval/rollout/successes"] == 0
        assert results["eval/rollout/valid_episodes"] == 3
        rows = read_episode_records(tmp_path)
        assert all(r["timed_out"] for r in rows)
        assert all(r["failure_category"] == FAILURE_TIMEOUT for r in rows)

    def test_policy_nan_is_valid_failure(self, tmp_path) -> None:
        evaluator = _evaluator(tmp_path, FakeAdapter(horizon=500), model=_NanModel(), horizon=500)
        results = evaluator.run()
        rows = read_episode_records(tmp_path)
        assert all(r["valid_episode"] for r in rows)
        assert all(r["success"] is False for r in rows)
        assert all(r["failure_category"] == FAILURE_POLICY_INVALID_ACTION for r in rows)
        assert results["eval/rollout/policy_failures"] == 3
        assert results["eval/rollout/success_rate"] == 0.0

    def test_policy_out_of_range_is_valid_failure(self, tmp_path) -> None:
        evaluator = _evaluator(tmp_path, FakeAdapter(horizon=500), model=_RangeModel(), horizon=500)
        evaluator.run()
        rows = read_episode_records(tmp_path)
        assert all(r["failure_category"] == FAILURE_POLICY_INVALID_ACTION for r in rows)

    def test_policy_exception_is_valid_failure(self, tmp_path) -> None:
        evaluator = _evaluator(
            tmp_path, FakeAdapter(horizon=500), model=_RaisingModel(), horizon=500
        )
        evaluator.run()
        rows = read_episode_records(tmp_path)
        assert all(r["valid_episode"] for r in rows)
        assert all(r["failure_category"] == FAILURE_POLICY_EXCEPTION for r in rows)
        assert "boom in the policy" in rows[0]["exception"]

    def test_infrastructure_failure_invalidates_run(self, tmp_path) -> None:
        adapter = FakeAdapter(fail_step_with=InfrastructureError("sim exploded"))
        evaluator = _evaluator(tmp_path, adapter, model=_ZeroModel(), horizon=500)
        with pytest.raises(RolloutRunInvalid, match="infrastructure"):
            evaluator.run()
        rows = read_episode_records(tmp_path)
        assert all(not r["valid_episode"] for r in rows)
        assert all("sim exploded" in r["exception"] for r in rows)
        assert all("success" not in r for r in rows)
        assert all("failure_category" not in r for r in rows)

    def test_reset_failure_invalidates_run(self, tmp_path) -> None:
        adapter = FakeAdapter(fail_reset_with=InfrastructureError("no reset"))
        evaluator = _evaluator(tmp_path, adapter, model=_ZeroModel(), horizon=500)
        with pytest.raises(RolloutRunInvalid):
            evaluator.run()
        rows = read_episode_records(tmp_path)
        assert all(not r["valid_episode"] for r in rows)

    def test_mixed_outcomes(self, tmp_path) -> None:
        """Case 0 (marker eef z=5) → NaN → policy failure; others time out cleanly."""
        controller = _lift_controller()
        ticker = _Ticker(controller)
        policy = _MarkerPolicy(ticker)

        evaluator = _evaluator(
            tmp_path,
            FakeAdapter(horizon=500),
            policy=policy,
            horizon=500,
        )
        results = evaluator.run()
        assert results["eval/rollout/successes"] == 0
        assert results["eval/rollout/policy_failures"] == 1
        rows = read_episode_records(tmp_path)
        by_index = {r["episode_index"]: r for r in rows}
        assert by_index[0]["failure_category"] == FAILURE_POLICY_INVALID_ACTION
        assert by_index[1]["failure_category"] == FAILURE_TIMEOUT
        assert by_index[2]["failure_category"] == FAILURE_TIMEOUT

    def test_success_recorded_with_correct_metadata(self, tmp_path) -> None:
        """A successful episode records ``success=True``, ``termination_reason='success'``,
        ``timed_out=False``, and the evaluator stops at the success step rather than
        running to the horizon. Uses a ``_SuccessAdapter`` that forces the fake's
        success predicate to fire on the first step; no hand-coded controller is
        involved.
        """
        evaluator = _evaluator(
            tmp_path,
            _SuccessAdapter(horizon=500),
            policy=_Ticker(_lift_controller()),
            horizon=500,
        )
        results = evaluator.run()
        assert results["eval/rollout/successes"] == 3
        assert results["eval/rollout/valid_episodes"] == 3
        assert results["eval/rollout/success_rate"] == 1.0
        rows = read_episode_records(tmp_path)
        assert len(rows) == 3
        for row in rows:
            assert row["valid_episode"] is True
            assert row["success"] is True
            assert row["timed_out"] is False
            assert row["termination_reason"] == "success"
            assert "failure_category" not in row
            # Stops at success (step 1), well before the 500-step horizon.
            assert row["steps"] <= 2

    def test_summary_json_written(self, tmp_path) -> None:
        _evaluator(
            tmp_path,
            FakeAdapter(horizon=500),
            policy=_Ticker(_lift_controller()),
            horizon=500,
        ).run()
        summary = json.loads((tmp_path / "rollout_summary.json").read_text())
        assert summary["run_id"] == "testrun"
        assert summary["reset_bank"] == "testbank"
        assert "metrics" in summary
        assert summary["metrics"]["eval/rollout/valid_episodes"] == 3
        assert summary["metrics"]["eval/rollout/successes"] == 0

    def test_run_id_and_tag_in_rows(self, tmp_path) -> None:
        adapter = FakeAdapter(horizon=500)
        evaluator = RolloutEvaluator(
            cfg=None,  # type: ignore[arg-type]
            policy=_Ticker(_lift_controller()),
            adapter=adapter,  # type: ignore[arg-type]
            bank=make_bank(1),
            normalizer=None,
            model=None,
            output_dir=tmp_path,
            run_id="runabc",
            model_name="bc",
            training_seed=43,
            task="Lift",
            checkpoint_sha256="cafe",
            tag="robot_only",
            horizon=500,
        )
        evaluator.run()
        row = read_episode_records(tmp_path)[0]
        assert row["run_id"] == "runabc"
        assert row["model"] == "bc"
        assert row["training_seed"] == 43
        assert row["checkpoint_sha256"] == "cafe"
        assert row["tag"] == "robot_only"
        assert row["reset_seed"] == 2026


# ---------------------------------------------------------------------------
# Per-phase success tracking + router-mode recording
# ---------------------------------------------------------------------------


class TestPerPhaseTracking:
    """The causal phase labeler classifies the policy's own states during
    rollout; per-phase SR = P(success | episode reached phase p)."""

    def test_timeouts_reach_phase_1_and_fail_there(self, tmp_path) -> None:
        evaluator = _evaluator(
            tmp_path,
            FakeAdapter(horizon=500),
            model=_ZeroModel(),
            horizon=500,
            phase_labeler=_phase_labeler(),
        )
        results = evaluator.run()
        per_phase = results["eval/rollout/per_phase_sr"]
        # The zero policy stays open and motionless: it never grasps, so it
        # settles in the causal pre-grasp phase (1) and times out there.
        assert per_phase["0"] == pytest.approx(0.0)
        assert per_phase["1"] == pytest.approx(0.0)
        assert per_phase["2"] is None
        assert results["eval/rollout/phase_tracking"] == "aggregated"

        rows = read_episode_records(tmp_path)
        assert all(r["extra"]["max_phase"] == 1 for r in rows)

    def test_successful_episodes_count_per_phase(self, tmp_path) -> None:
        evaluator = _evaluator(
            tmp_path,
            _SuccessAdapter(horizon=500),
            model=_ZeroModel(),
            horizon=500,
            phase_labeler=_phase_labeler(),
        )
        results = evaluator.run()
        per_phase = results["eval/rollout/per_phase_sr"]
        # Success fires on the first step, still in the approach phase (0).
        assert per_phase["0"] == pytest.approx(1.0)
        assert per_phase["1"] is None

    def test_missing_labeler_reports_phase_tracking_missing(self, tmp_path) -> None:
        evaluator = _evaluator(
            tmp_path, FakeAdapter(horizon=500), model=_ZeroModel(), horizon=500
        )
        results = evaluator.run()
        assert results["eval/rollout/phase_tracking"] == "missing"
        assert "eval/rollout/per_phase_sr" not in results
        rows = read_episode_records(tmp_path)
        assert all("extra" not in r for r in rows)

    def test_router_mode_recorded_in_rows_and_summary(self, tmp_path) -> None:
        evaluator = _evaluator(
            tmp_path,
            FakeAdapter(horizon=500),
            model=_ZeroModel(),
            horizon=500,
            router_mode="sticky",
        )
        results = evaluator.run()
        assert results["eval/rollout/router_mode"] == "sticky"
        rows = read_episode_records(tmp_path)
        assert all(r["router_mode"] == "sticky" for r in rows)
        summary = json.loads((tmp_path / "rollout_summary.json").read_text())
        assert summary["router_mode"] == "sticky"

    def test_learned_mode_records_no_router_mode(self, tmp_path) -> None:
        evaluator = _evaluator(
            tmp_path, FakeAdapter(horizon=500), model=_ZeroModel(), horizon=500
        )
        results = evaluator.run()
        assert results["eval/rollout/router_mode"] == "learned"
        rows = read_episode_records(tmp_path)
        assert all("router_mode" not in r for r in rows)


class TestPhaseLabelerFromAggregate:
    def test_aggregate_builds_valid_labeler(self) -> None:
        from phaseforge.evaluations.rollout.runner import _phase_labeler_from_aggregate

        demo = _phase_calibration()
        thresholds = {
            "data_config_hash": "abc",
            "n_demos": 3,
            "closed_level": 0.019,
            "open_level": 0.041,
            "mirror": False,
            "per_demo": [demo, demo, demo],
        }
        labeler = _phase_labeler_from_aggregate(thresholds)
        assert labeler.num_phases == 6
        # The median levels win over the per-demo values.
        assert labeler.closed_level == pytest.approx(0.019)
        assert labeler.open_level == pytest.approx(0.041)

    def test_empty_per_demo_raises(self) -> None:
        from phaseforge.evaluations.rollout.runner import _phase_labeler_from_aggregate

        with pytest.raises(ValueError, match="per_demo"):
            _phase_labeler_from_aggregate(
                {"closed_level": 0.02, "open_level": 0.04, "per_demo": []}
            )


class TestRequirePhaseTracking:
    def test_require_tracking_fails_closed_when_artifact_missing(
        self, monkeypatch, tmp_path
    ) -> None:
        """require_phase_tracking=true + a cache without phase_thresholds.json
        must abort before the simulator runs (a pre-calibration cache cannot
        be silently evaluated without per-phase SR)."""
        from omegaconf import DictConfig

        from phaseforge.evaluations.rollout.runner import run_rollout_evaluation

        cfg = DictConfig(
            {
                "data": {"source": {"task_name": "Lift"}},
                "project": {"seed": 42},
                "models": {
                    "name": "precision_residual_phaseforge",
                    "_target_": "phaseforge.models.phase_moe.PhaseBootstrappedMoE",
                },
                "eval": {
                    "mode": "rollout",
                    "bank": {"seed": 2026, "num_cases": 50},
                    "env": {"mujoco_requirement": ">=3.2.7"},
                    "episodes": {
                        "require_phase_tracking": True,
                        "router_mode": "learned",
                    },
                },
            }
        )
        monkeypatch.setattr(
            "phaseforge.evaluations.rollout.runner.resolve_pinned_metadata",
            lambda cfg: object(),
        )
        monkeypatch.setattr(
            "phaseforge.evaluations.rollout.runner.load_or_generate_bank",
            lambda cfg, meta: make_bank(1),
        )
        monkeypatch.setattr(
            "phaseforge.evaluations.rollout.runner._adapter_from_config",
            lambda cfg, meta: FakeAdapter(horizon=500),
        )
        monkeypatch.setattr(
            "phaseforge.evaluations.rollout.runner.FrozenNormalizer.load",
            lambda path: FrozenNormalizer(torch.zeros(19), torch.ones(19)),
        )
        monkeypatch.setattr(
            "phaseforge.data.ingestion.cache_manager.load_phase_thresholds",
            lambda cache_dir: None,
        )
        with pytest.raises(EnvParityError, match="require_phase_tracking"):
            run_rollout_evaluation(
                cfg,
                torch.nn.Module(),
                output_dir=tmp_path,
                run_id="test-run",
            )

    def test_require_tracking_passes_with_artifact(
        self, monkeypatch, tmp_path
    ) -> None:
        from omegaconf import DictConfig

        from phaseforge.evaluations.rollout.runner import run_rollout_evaluation

        cfg = DictConfig(
            {
                "data": {"source": {"task_name": "Lift"}},
                "project": {"seed": 42},
                "models": {
                    "name": "precision_residual_phaseforge",
                    "_target_": "phaseforge.models.phase_moe.PhaseBootstrappedMoE",
                },
                "eval": {
                    "mode": "rollout",
                    "bank": {"seed": 2026, "num_cases": 50},
                    "env": {"mujoco_requirement": ">=3.2.7"},
                    "episodes": {
                        "require_phase_tracking": True,
                        "router_mode": "learned",
                    },
                },
            }
        )
        thresholds = {
            "data_config_hash": "abc",
            "n_demos": 1,
            "closed_level": 0.02,
            "open_level": 0.04,
            "mirror": False,
            "per_demo": [_phase_calibration()],
        }
        monkeypatch.setattr(
            "phaseforge.evaluations.rollout.runner.resolve_pinned_metadata",
            lambda cfg: object(),
        )
        monkeypatch.setattr(
            "phaseforge.evaluations.rollout.runner.load_or_generate_bank",
            lambda cfg, meta: make_bank(1),
        )
        monkeypatch.setattr(
            "phaseforge.evaluations.rollout.runner._adapter_from_config",
            lambda cfg, meta: FakeAdapter(horizon=500),
        )
        monkeypatch.setattr(
            "phaseforge.evaluations.rollout.runner.FrozenNormalizer.load",
            lambda path: FrozenNormalizer(torch.zeros(19), torch.ones(19)),
        )
        monkeypatch.setattr(
            "phaseforge.data.ingestion.cache_manager.load_phase_thresholds",
            lambda cache_dir: thresholds,
        )
        results = run_rollout_evaluation(
            cfg,
            _ZeroModel(),
            output_dir=tmp_path,
            run_id="test-run",
        )
        assert results["eval/rollout/phase_tracking"] == "aggregated"
        assert results["eval/rollout/per_phase_sr"]["0"] == pytest.approx(0.0)

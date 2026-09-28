"""Tests for the rollout validation gates (all with fakes, no robosuite)."""

from __future__ import annotations

import numpy as np

from phaseforge.evaluations.envs.errors import InfrastructureError
from phaseforge.evaluations.rollout.gates import (
    gate_action_contract,
    gate_demo_replay,
    gate_env_schema,
    gate_native_predicate,
    gate_random_noop_sanity,
)
from tests.rollout_helpers import (
    FakeAdapter,
    FakeLiftSim,
    make_bank,
)


class TestEnvSchema:
    def test_passes(self) -> None:
        result = gate_env_schema(
            FakeAdapter(horizon=500),  # type: ignore[arg-type]
            make_bank(3),
            expected_state_dim=19,
            expected_action_dim=7,
        )
        assert result.status == "PASS", result.detail

    def test_wrong_state_dim_fails(self) -> None:
        result = gate_env_schema(
            FakeAdapter(horizon=500),  # type: ignore[arg-type]
            make_bank(3),
            expected_state_dim=23,
            expected_action_dim=7,
        )
        assert result.status == "FAIL"


class TestDemoReplay:
    def test_skipped_when_hdf5_missing(self, tmp_path) -> None:
        result = gate_demo_replay(FakeAdapter(), tmp_path / "missing.hdf5", num_demos=1)  # type: ignore[arg-type]
        assert result.status == "SKIPPED"
        assert "not present" in result.detail

    def test_passes_on_matching_demo(self, tmp_path) -> None:
        import h5py

        adapter = FakeAdapter(horizon=500)  # type: ignore[arg-type]
        states, actions = _linear_demo(adapter, steps=5)
        path = tmp_path / "lift.hdf5"
        with h5py.File(path, "w") as h5:
            data = h5.create_group("data")
            demo = data.create_group("demo_0")
            demo.create_dataset("states", data=states)
            demo.create_dataset("actions", data=actions)

        result = gate_demo_replay(adapter, path, num_demos=1, tolerance=1e-6)  # type: ignore[arg-type]
        assert result.status == "PASS", result.detail

    def test_passes_on_v15_equal_length_demo(self, tmp_path) -> None:
        """The v1.5 PH files store T states and T actions.

        The published v1.5 states are post-action states, so state[0] is
        already the result of action[0]. The gate must replay actions[1:]
        against states[1:].
        """
        import h5py

        adapter = FakeAdapter(horizon=500)  # type: ignore[arg-type]
        states, actions = _linear_demo(adapter, steps=5)
        path = tmp_path / "lift_v15.hdf5"
        with h5py.File(path, "w") as h5:
            data = h5.create_group("data")
            demo = data.create_group("demo_0")
            demo.create_dataset("states", data=states[1:])
            demo.create_dataset("actions", data=actions)

        result = gate_demo_replay(adapter, path, num_demos=1, tolerance=1e-6)  # type: ignore[arg-type]
        assert result.status == "PASS", result.detail
        assert result.metrics["compared"] == 4

    def test_fails_on_divergent_demo(self, tmp_path) -> None:
        import h5py

        adapter = FakeAdapter(horizon=500)  # type: ignore[arg-type]
        states, actions = _linear_demo(adapter, steps=5)
        states[2, 1] += 0.01  # perturb the stored post-step state
        path = tmp_path / "lift.hdf5"
        with h5py.File(path, "w") as h5:
            data = h5.create_group("data")
            demo = data.create_group("demo_0")
            demo.create_dataset("states", data=states)
            demo.create_dataset("actions", data=actions)

        result = gate_demo_replay(adapter, path, num_demos=1, tolerance=1e-6)  # type: ignore[arg-type]
        assert result.status == "SKIPPED", result.detail
        assert result.metrics["diagnostic_only"] is True
        assert result.metrics["mismatches"] >= 1


def _linear_demo(adapter: FakeAdapter, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """A fake demo the kinematic sim reproduces exactly.

    World-state layout: ``[time, eef(3), cube(3)]``. Actions move the eef
    linearly; the cube never moves (gripper stays open).
    """
    states = []
    adapter.sim.reset()
    for i in range(steps):
        state = np.asarray(adapter.sim.get_state(), dtype=np.float64)
        states.append(state)
        action = np.zeros(7, dtype=np.float64)
        action[0:3] = np.array([0.01, 0.02, -0.01]) / 0.05
        adapter.sim.step(action)
    states.append(np.asarray(adapter.sim.get_state(), dtype=np.float64))
    actions = np.tile(np.array([0.2, 0.4, -0.2, 0.0, 0.0, 0.0, 1.0]), (steps, 1))
    return np.stack(states), actions


class TestActionContract:
    def test_passes(self) -> None:
        result = gate_action_contract(FakeAdapter(), trials=5)  # type: ignore[arg-type]
        assert result.status == "PASS", result.detail


class TestNativePredicate:
    def test_passes_when_predicate_returns_bool(self) -> None:
        # The default FakeAdapter returns bool from both check_success and
        # the underlying env._check_success() probe; Gate 4 must PASS.
        result = gate_native_predicate(
            FakeAdapter(horizon=500),  # type: ignore[arg-type]
            make_bank(3),
            num_cases=3,
        )
        assert result.status == "PASS", result.detail
        assert result.metrics["probed"] == 3

    def test_fails_when_predicate_raises(self) -> None:
        # Adapter raises from check_success -> Gate 4 must FAIL on the first case.
        adapter = FakeAdapter(
            FakeLiftSim(), fail_check_success_with=InfrastructureError("predicate boom")
        )
        result = gate_native_predicate(
            adapter,  # type: ignore[arg-type]
            make_bank(3),
            num_cases=3,
        )
        assert result.status == "FAIL"
        assert "check_success" in result.detail

    def test_fails_on_reset_error(self) -> None:
        # Adapter fails to reset -> Gate 4 must FAIL before any probe.
        adapter = FakeAdapter(FakeLiftSim(), fail_reset_with=InfrastructureError("reset boom"))
        result = gate_native_predicate(
            adapter,  # type: ignore[arg-type]
            make_bank(3),
            num_cases=3,
        )
        assert result.status == "FAIL"
        assert "reset failed" in result.detail

    def test_limits_to_num_cases(self) -> None:
        # 5-case bank, num_cases=2 -> only 2 probes counted.
        result = gate_native_predicate(
            FakeAdapter(horizon=500),  # type: ignore[arg-type]
            make_bank(5),
            num_cases=2,
        )
        assert result.status == "PASS", result.detail
        assert result.metrics["probed"] == 2
        assert result.metrics["cases"] == 2

    def test_fails_on_empty_bank(self) -> None:
        # Empty bank -> FAIL (refuse a vacuous 0/0 PASS).
        class _EmptyBank:
            task = "Lift"
            cases: list = []

        result = gate_native_predicate(
            FakeAdapter(horizon=500),  # type: ignore[arg-type]
            _EmptyBank(),  # type: ignore[arg-type]
            num_cases=5,
        )
        assert result.status == "FAIL"
        assert "empty" in result.detail

    def test_fails_on_zero_num_cases(self) -> None:
        # num_cases <= 0 -> FAIL (refuse a vacuous gate).
        result = gate_native_predicate(
            FakeAdapter(horizon=500),  # type: ignore[arg-type]
            make_bank(3),
            num_cases=0,
        )
        assert result.status == "FAIL"
        assert "must be > 0" in result.detail

    def test_clamps_num_cases_to_bank_size(self) -> None:
        # 3-case bank, num_cases=10 -> only 3 probes counted.
        result = gate_native_predicate(
            FakeAdapter(horizon=500),  # type: ignore[arg-type]
            make_bank(3),
            num_cases=10,
        )
        assert result.status == "PASS", result.detail
        assert result.metrics["probed"] == 3
        assert result.metrics["cases"] == 3

    def test_fails_on_dict_without_task_key(self) -> None:
        # Predicate returns dict without "task" -> FAIL on raw probe.
        class _DictNoTaskAdapter(FakeAdapter):
            def _check_success_on_env(self) -> dict:
                return {"success": True}  # missing "task"

        adapter = _DictNoTaskAdapter(horizon=500)
        # Override the underlying sim's _check_success to return the bad dict.
        adapter.env._check_success = lambda: {"success": True}  # type: ignore[assignment]
        result = gate_native_predicate(
            adapter,  # type: ignore[arg-type]
            make_bank(3),
            num_cases=1,
        )
        assert result.status == "FAIL"
        assert "task" in result.detail

    def test_accepts_dict_with_task_key(self) -> None:
        # Predicate returns {"task": True} -> PASS (matches robosuite's
        # multi-stage convention where each key is a sub-predicate).
        adapter = FakeAdapter(horizon=500)
        adapter.env._check_success = lambda: {"task": True}  # type: ignore[assignment]
        result = gate_native_predicate(
            adapter,  # type: ignore[arg-type]
            make_bank(3),
            num_cases=3,
        )
        assert result.status == "PASS", result.detail
        assert result.metrics["probed"] == 3

    def test_accepts_numpy_bool_scalar(self) -> None:
        """Real Lift ``env._check_success`` returns ``np.bool_`` scalars; the
        probe must accept them so every state-only rollout pipeline does
        not fail the gates on a numpy-vs-python-bool type mismatch."""
        import numpy as np

        adapter = FakeAdapter(horizon=500)
        adapter.env._check_success = lambda: np.bool_(False)  # type: ignore[assignment]
        result = gate_native_predicate(
            adapter,  # type: ignore[arg-type]
            make_bank(3),
            num_cases=3,
        )
        assert result.status == "PASS", result.detail
        assert result.metrics["probed"] == 3

    def test_rejects_non_boolean_numpy_scalar(self) -> None:
        """An ``np.int_`` or ``np.float_`` is not a boolean predicate,
        even if truthy — it would silently corrupt the success metric."""
        import numpy as np

        adapter = FakeAdapter(horizon=500)
        adapter.env._check_success = lambda: np.int_(1)  # type: ignore[assignment]
        result = gate_native_predicate(
            adapter,  # type: ignore[arg-type]
            make_bank(3),
            num_cases=1,
        )
        assert result.status == "FAIL"
        assert "unsupported" in result.detail


class TestRandomNoop:
    def test_no_successes_no_infra(self) -> None:
        # Tight grasp window: random actions can never align well enough.
        sim = FakeLiftSim(grasp_z_window=0.0005)
        result = gate_random_noop_sanity(
            FakeAdapter(sim, horizon=500),  # type: ignore[arg-type]
            make_bank(3),
            num_cases=3,
            horizon=200,
            max_success_rate=0.05,
        )
        assert result.status == "PASS", result.detail
        assert result.metrics["successes"] == 0
        assert result.metrics["infra_failures"] == 0

    def test_rejects_empty_bank(self) -> None:
        class _EmptyBank:
            cases: list = []

        result = gate_random_noop_sanity(
            FakeAdapter(),  # type: ignore[arg-type]
            _EmptyBank(),  # type: ignore[arg-type]
            num_cases=3,
            horizon=10,
        )
        assert result.status == "FAIL"
        assert "empty" in result.detail

    def test_rejects_nonpositive_limits(self) -> None:
        bank = make_bank(3)
        adapter = FakeAdapter()  # type: ignore[arg-type]
        assert (
            gate_random_noop_sanity(adapter, bank, num_cases=0, horizon=10).status
            == "FAIL"
        )
        assert (
            gate_random_noop_sanity(adapter, bank, num_cases=3, horizon=0).status
            == "FAIL"
        )


class TestCheckpointSmoke:
    def test_skipped_without_checkpoint(self) -> None:
        from phaseforge.evaluations.rollout.gates import gate_checkpoint_smoke

        class _Cfg:
            train = {"stage1_ckpt_path": ""}
            project = {"device": "cpu"}

        result = gate_checkpoint_smoke(_Cfg(), FakeAdapter(), make_bank(3), num_episodes=2)  # type: ignore[arg-type]
        assert result.status == "SKIPPED"


class TestCLIExitCode:
    """The CLI rule: a diagnostic FAIL must not block the protocol (exit 0);
    a required FAIL must block (exit 1)."""

    def _result(self, *, status: str, diagnostic: bool = False) -> object:
        from phaseforge.evaluations.rollout.gates import GateResult

        return GateResult(gate="probe", status=status, detail="probe", diagnostic=diagnostic)

    def test_diagnostic_fail_does_not_block(self) -> None:
        from phaseforge.evaluations.rollout.gates_cli import _compute_exit_code

        results = [
            self._result(status="PASS"),
            self._result(status="FAIL", diagnostic=True),
            self._result(status="SKIPPED"),
        ]
        assert _compute_exit_code(results) == 0

    def test_required_fail_blocks(self) -> None:
        from phaseforge.evaluations.rollout.gates_cli import _compute_exit_code

        results = [
            self._result(status="PASS"),
            self._result(status="FAIL", diagnostic=False),
        ]
        assert _compute_exit_code(results) == 1

    def test_all_pass_returns_zero(self) -> None:
        from phaseforge.evaluations.rollout.gates_cli import _compute_exit_code

        results = [
            self._result(status="PASS"),
            self._result(status="SKIPPED"),
        ]
        assert _compute_exit_code(results) == 0

    def test_diagnostic_marker_rendering(self) -> None:
        from phaseforge.evaluations.rollout.gates_cli import _result_marker

        assert _result_marker(self._result(status="PASS")) == "[PASS]"
        assert _result_marker(self._result(status="FAIL")) == "[FAIL]"
        assert _result_marker(self._result(status="SKIPPED")) == "[SKIP]"
        assert (
            _result_marker(self._result(status="PASS", diagnostic=True))
            == "[DIAG-PASS]"
        )
        assert (
            _result_marker(self._result(status="FAIL", diagnostic=True))
            == "[DIAG-FAIL]"
        )
        # SKIPPED never gets a diagnostic marker even if the flag is set.
        assert (
            _result_marker(self._result(status="SKIPPED", diagnostic=True))
            == "[SKIP]"
        )

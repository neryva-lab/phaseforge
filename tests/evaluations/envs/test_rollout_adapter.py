"""Tests for the robosuite state adapter (observation extraction + action contract)."""

from __future__ import annotations

import numpy as np
import pytest

from phaseforge.evaluations.envs.errors import (
    EnvParityError,
    PolicyInvalidActionError,
    StateSchemaError,
)
from phaseforge.evaluations.envs.robosuite_adapter import (
    RobosuiteStateAdapter,
    StateSpec,
)
from tests.rollout_helpers import make_meta


class TestStateSpec:
    def test_dims_and_index(self) -> None:
        spec = StateSpec(keys=("a", "b"), dims=(3, 4))
        assert spec.dim == 7
        assert spec.index_of("b") == (3, 7)

    def test_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError):
            StateSpec(keys=("a",), dims=(3, 4))

    def test_unknown_key(self) -> None:
        spec = StateSpec(keys=("a",), dims=(3,))
        with pytest.raises(KeyError):
            spec.index_of("nope")


class TestExtractState:
    def setup_method(self) -> None:
        self.spec = StateSpec(keys=("robot0_eef_pos", "object"), dims=(3, 10))

    def test_concatenation_and_order(self) -> None:
        adapter = _adapter_with_spec(self.spec)
        obs = {
            "robot0_eef_pos": np.ones(3),
            "object-state": np.arange(10, dtype=np.float32),
        }
        state = adapter.extract_state(obs)
        assert state.shape == (13,)
        assert np.allclose(state[:3], 1.0)
        assert np.allclose(state[3:], np.arange(10))

    def test_object_alias(self) -> None:
        adapter = _adapter_with_spec(self.spec)
        state = adapter.extract_state({"robot0_eef_pos": np.zeros(3), "object": np.zeros(10)})
        assert state.shape == (13,)

    def test_missing_key(self) -> None:
        adapter = _adapter_with_spec(self.spec)
        with pytest.raises(StateSchemaError, match="robot0_eef_pos"):
            adapter.extract_state({"object": np.zeros(10)})

    def test_wrong_dim(self) -> None:
        adapter = _adapter_with_spec(self.spec)
        with pytest.raises(StateSchemaError, match="dimension"):
            adapter.extract_state({"robot0_eef_pos": np.zeros(5), "object": np.zeros(10)})

    def test_joint_pos_cos_sin_fallback(self) -> None:
        spec = StateSpec(keys=("robot0_joint_pos", "robot0_eef_pos"), dims=(7, 3))
        adapter = _adapter_with_spec(spec)
        angles = np.array([0.1, -0.5, 1.2, -1.0, 0.0, 0.7, -0.3], dtype=np.float32)
        obs = {
            "robot0_joint_pos_cos": np.cos(angles),
            "robot0_joint_pos_sin": np.sin(angles),
            "robot0_eef_pos": np.zeros(3, dtype=np.float32),
        }
        state = adapter.extract_state(obs)
        assert state.shape == (10,)
        assert np.allclose(state[:7], angles, atol=1e-5)
        assert np.allclose(state[7:], 0.0)


class TestValidateAction:
    def setup_method(self) -> None:
        self.spec = StateSpec(keys=("robot0_eef_pos",), dims=(3,))
        self.adapter = _adapter_with_spec(self.spec, action_dim=7)

    def test_valid_action_passthrough(self) -> None:
        out = self.adapter.validate_action(np.ones(7) * 0.5)
        assert out.dtype == np.float64

    def test_batch_shape_accepted(self) -> None:
        out = self.adapter.validate_action(np.zeros((1, 7)))
        assert out.shape == (7,)

    def test_nan_rejected(self) -> None:
        with pytest.raises(PolicyInvalidActionError, match="non-finite"):
            self.adapter.validate_action(np.array([np.nan] * 7))

    def test_inf_rejected(self) -> None:
        with pytest.raises(PolicyInvalidActionError, match="non-finite"):
            self.adapter.validate_action(np.array([np.inf] * 7))

    def test_out_of_range_rejected(self) -> None:
        with pytest.raises(PolicyInvalidActionError, match="outside"):
            self.adapter.validate_action(np.ones(7) * 1.5)

    def test_wrong_shape_rejected(self) -> None:
        with pytest.raises(PolicyInvalidActionError, match="shape"):
            self.adapter.validate_action(np.zeros(6))

    def test_tolerance_extension(self) -> None:
        out = self.adapter.validate_action(np.ones(7) * 1.00005, tolerance=1e-4)
        assert out.shape == (7,)


class TestConstruction:
    def test_action_dim_mismatch_fails_closed(self, monkeypatch) -> None:
        import sys

        from phaseforge.evaluations.envs import robosuite_adapter as mod

        class _FakeRobosuite:
            def make(self, *args, **kwargs):
                class _Env:
                    action_spec = (np.zeros((5,)), np.zeros((5,)))
                    horizon = 500

                return _Env()

        monkeypatch.setitem(sys.modules, "robosuite", _FakeRobosuite())
        monkeypatch.setattr(mod, "robosuite", _FakeRobosuite(), raising=False)
        with pytest.raises(EnvParityError, match="Action dimension mismatch"):
            RobosuiteStateAdapter(
                make_meta(),
                StateSpec(keys=("a",), dims=(3,)),
                action_dim=7,
            )

    def test_missing_robosuite(self, monkeypatch) -> None:
        import builtins

        original = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "robosuite":
                raise ImportError("no robosuite")
            return original(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(EnvParityError, match="not installed"):
            RobosuiteStateAdapter(make_meta(), StateSpec(keys=("a",), dims=(3,)), action_dim=3)


def _adapter_with_spec(spec: StateSpec, action_dim: int = 7) -> RobosuiteStateAdapter:
    """Build an adapter without touching robosuite (bypass __init__)."""
    adapter = object.__new__(RobosuiteStateAdapter)
    adapter.meta = make_meta()
    adapter.state_spec = spec
    adapter.action_dim = action_dim
    adapter.action_low = -1.0
    adapter.action_high = 1.0
    # Matches the __init__ default; the bypass skips attribute initialization.
    adapter.action_tolerance = 1e-4
    return adapter

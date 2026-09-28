"""Tests for the five-task registry and dev fallback mapping."""

from __future__ import annotations

import pytest

from phaseforge.evaluations.envs.env_metadata import dev_fallback_metadata
from phaseforge.evaluations.envs.task_registry import (
    ENV_NAME_TO_PROTOCOL,
    PROTOCOL_ACTION_DIM,
    PROTOCOL_HORIZON,
    PROTOCOL_STATE_DIM,
    PROTOCOL_TO_ENV_NAME,
    TaskSpec,
    env_name_for_protocol,
    is_known_task,
    known_protocol_tasks,
    protocol_for_env_name,
)


class TestMappings:
    def test_all_five_tasks_are_known(self) -> None:
        assert known_protocol_tasks() == ["Can", "Lift", "Square", "ToolHang", "Transport"]

    def test_protocol_to_env_name_round_trip(self) -> None:
        for proto, env in PROTOCOL_TO_ENV_NAME.items():
            assert protocol_for_env_name(env) == proto
            assert env_name_for_protocol(proto) == env

    def test_expected_dims_for_each_task(self) -> None:
        assert PROTOCOL_STATE_DIM == {
            "Lift": 19,
            "Can": 23,
            "Square": 23,
            "ToolHang": 53,
            "Transport": 59,
        }
        assert PROTOCOL_ACTION_DIM == {
            "Lift": 7,
            "Can": 7,
            "Square": 7,
            "ToolHang": 7,
            "Transport": 14,
        }
        assert PROTOCOL_HORIZON == {
            "Lift": 500,
            "Can": 500,
            "Square": 500,
            "ToolHang": 500,
            "Transport": 700,
        }

    def test_unknown_protocol_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown protocol task"):
            TaskSpec.from_protocol("Hopper")
        with pytest.raises(KeyError, match="Unknown robosuite env_name"):
            protocol_for_env_name("Cartpole")

    def test_is_known_task(self) -> None:
        assert is_known_task("Lift")
        assert is_known_task("Transport")
        assert not is_known_task("PickPlaceCan")
        assert not is_known_task("")


class TestTaskSpec:
    @pytest.mark.parametrize(
        "protocol_name, env_name",
        [
            ("Lift", "Lift"),
            ("Can", "PickPlaceCan"),
            ("Square", "NutAssemblySquare"),
            ("ToolHang", "ToolHang"),
            ("Transport", "TwoArmTransport"),
        ],
    )
    def test_spec_resolves_to_controller(self, protocol_name: str, env_name: str) -> None:
        spec = TaskSpec.from_protocol(protocol_name)
        assert spec.protocol_name == protocol_name
        assert spec.robosuite_env_name == env_name
        assert spec.state_dim == PROTOCOL_STATE_DIM[protocol_name]
        assert spec.action_dim == PROTOCOL_ACTION_DIM[protocol_name]
        assert spec.horizon == PROTOCOL_HORIZON[protocol_name]

    def test_env_name_to_protocol_inverse(self) -> None:
        for env, proto in ENV_NAME_TO_PROTOCOL.items():
            assert env_name_for_protocol(proto) == env


class TestDevFallbackMetadata:
    @pytest.mark.parametrize(
        "protocol_name, env_name",
        [
            ("Lift", "Lift"),
            ("Can", "PickPlaceCan"),
            ("Square", "NutAssemblySquare"),
            ("ToolHang", "ToolHang"),
            ("Transport", "TwoArmTransport"),
        ],
    )
    def test_dev_fallback_resolves_each_protocol(self, protocol_name: str, env_name: str) -> None:
        meta = dev_fallback_metadata(protocol_name)
        assert meta.env_name == env_name
        assert meta.env_version == "1.5.1"
        assert meta.env_type == "robosuite"
        assert meta.horizon == PROTOCOL_HORIZON[protocol_name]

    def test_dev_fallback_default_is_lift(self) -> None:
        meta = dev_fallback_metadata()
        assert meta.env_name == "Lift"

    def test_dev_fallback_unknown_protocol_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown protocol task"):
            dev_fallback_metadata("Stack")


class TestDataConfigParity:
    """The per-task data YAMLs must agree with the task registry.

    A regression here would silently feed the wrong state layout into the
    ingest/normalize pipeline and surface only at training time. This test
    guards the cross-file contract by parsing each YAML and checking
    ``state_dim``, ``state_keys`` sums, and ``expected_env_name`` against
    the registry.
    """

    @pytest.mark.parametrize(
        "data_name, expected_dim",
        [
            ("lift", 19),
            ("can", 23),
            ("square", 23),
            ("tool_hang", 53),
            ("transport", 59),
        ],
    )
    def test_state_keys_sum_matches_declared(self, data_name: str, expected_dim: int) -> None:
        import yaml

        path = f"phaseforge/config/data/{data_name}.yaml"
        with open(path) as f:
            cfg = yaml.safe_load(f)
        computed = sum(entry["dim"] for entry in cfg["state_keys"])
        assert cfg["state_dim"] == computed == expected_dim

    @pytest.mark.parametrize(
        "data_name, protocol_name",
        [
            ("lift", "Lift"),
            ("can", "Can"),
            ("square", "Square"),
            ("tool_hang", "ToolHang"),
            ("transport", "Transport"),
        ],
    )
    def test_expected_env_name_matches_registry(self, data_name: str, protocol_name: str) -> None:
        import yaml

        path = f"phaseforge/config/data/{data_name}.yaml"
        with open(path) as f:
            cfg = yaml.safe_load(f)
        expected = cfg["source"]["expected_env_name"]
        assert PROTOCOL_TO_ENV_NAME[protocol_name] == expected

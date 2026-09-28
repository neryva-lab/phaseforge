"""Per-task data-config composition guards for the locked final matrix."""

from __future__ import annotations

import pytest
from hydra import compose, initialize
from omegaconf import DictConfig

from phaseforge.evaluations.envs.task_registry import (
    PROTOCOL_TO_ENV_NAME,
    TaskSpec,
    validate_task_schema,
)

#: Every per-task data config: the five structured schemas and their robot-only
#: negative controls.
PER_TASK_DATA_CONFIGS: tuple[str, ...] = (
    "lift",
    "can",
    "square",
    "tool_hang",
    "transport",
    "robot_only_lift",
    "robot_only_can",
    "robot_only_square",
    "robot_only_tool_hang",
    "robot_only_transport",
)


def _compose_data(data_config: str) -> DictConfig:
    with initialize(version_base="1.3", config_path="../../phaseforge/config"):
        return compose(config_name="main", overrides=[f"data={data_config}"])


def _key_offsets(cfg: DictConfig) -> dict[str, tuple[int, int]]:
    """Cumulative [start, end) offset of every declared state key."""
    offsets: dict[str, tuple[int, int]] = {}
    start = 0
    for entry in cfg.data.state_keys:
        offsets[str(entry.key)] = (start, start + int(entry.dim))
        start += int(entry.dim)
    return offsets


@pytest.mark.parametrize("data_config", PER_TASK_DATA_CONFIGS)
def test_per_task_config_validates_against_registry(data_config: str) -> None:
    cfg = _compose_data(data_config)
    data = cfg.data
    task = str(data.source.task_name)
    keys = [str(entry.key) for entry in data.state_keys]
    dims = [int(entry.dim) for entry in data.state_keys]

    validate_task_schema(task, keys, dims, int(data.action_dim))

    spec = TaskSpec.from_protocol(task)
    assert sum(dims) == int(data.state_dim)
    assert int(data.action_dim) == spec.action_dim
    assert str(data.source.expected_env_name) == PROTOCOL_TO_ENV_NAME[task]
    assert str(data.source.task_name) == task
    # The configured dataset pin matches the canonical registry horizon doc.
    assert spec.robosuite_env_name == PROTOCOL_TO_ENV_NAME[task]


@pytest.mark.parametrize("data_config", PER_TASK_DATA_CONFIGS)
def test_phase_labeler_slices_align_with_key_layout(data_config: str) -> None:
    """Labeler slices must land exactly on the declared key boundaries.

    The robot-only layouts reorder the state vector (joint state first), so
    the labeler's ``eef_pos_slice`` / ``gripper_qpos_slice`` must be
    re-pinned per layout — a wrong slice silently labels phases from the
    wrong signals (e.g. joint velocities read as gripper aperture).
    """
    cfg = _compose_data(data_config)
    offsets = _key_offsets(cfg)
    labeler = cfg.data.phase_labeler
    eef_start, eef_end = [int(v) for v in labeler.eef_pos_slice]
    grip_start, grip_end = [int(v) for v in labeler.gripper_qpos_slice]

    assert (eef_start, eef_end) == offsets["robot0_eef_pos"], data_config
    assert (grip_start, grip_end) == offsets["robot0_gripper_qpos"], data_config
    assert int(labeler.num_phases) == 6


def test_registry_schema_strings_match_structured_configs() -> None:
    """Registry ``schema_version`` strings must equal the data configs' pins.

    The registry field is documentation (the functional pin is the data
    config's ``schema_version`` read by the ingestion state machine), but a
    mismatch would silently mislead readers — this guard keeps the registry
    and the data configs locked together.
    """
    for task in ("lift", "can", "square", "tool_hang", "transport"):
        cfg = _compose_data(task)
        spec = TaskSpec.from_protocol(str(cfg.data.source.task_name))
        assert spec.schema_version == str(cfg.data.schema_version), task

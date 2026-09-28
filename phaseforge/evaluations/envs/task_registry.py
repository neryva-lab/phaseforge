"""Canonical registry of the five robomimic state-based benchmark tasks.

The five-task protocol is the scientific scope of the evaluation (the
final evaluation plan). Lift is the first debugging task because its geometry
is the simplest, but the reported results must cover all five tasks. Each
task is evaluated as a single-task policy (not a multi-task policy).

This module is the single source of truth for per-task metadata. The
ingestion layer (``data/robomimic/ingester.py``) already supports per-task
``env_args``; the rollout layer reads the cached ``env_metadata`` from
ingestion and validates it against this registry's ``robosuite_env_name``
in the parity gate.

Robosuite v1.5 environment names (matching the published
robomimic-v0.1/v1.5-track PH low-dim datasets):

* ``Lift``                -> 19-dim state, 7-dim action, horizon 500
* ``PickPlaceCan``         -> 23-dim state, 7-dim action, horizon 500
* ``NutAssemblySquare``   -> 23-dim state, 7-dim action, horizon 500
                             (referred to as "Square" in the protocol)
* ``ToolHang``            -> 53-dim state, 7-dim action, horizon 500
* ``TwoArmTransport``     -> 59-dim state, 14-dim action, horizon 700
                             (referred to as "Transport" in the protocol)

The "protocol name" is the short identifier used throughout the configs
and experiment manifests (``Lift``, ``Can``, ``Square``, ``ToolHang``,
``Transport``). The "robosuite_env_name" is the actual robosuite env class
name. ``Square`` and ``Transport`` use robosuite's longer names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Protocol name -> robosuite env_name. Used by the parity gate.
PROTOCOL_TO_ENV_NAME: dict[str, str] = {
    "Lift": "Lift",
    # robosuite 1.5.1 registers the benchmark as PickPlaceCan; ``Can`` is
    # only the short protocol label used by robomimic documentation.
    "Can": "PickPlaceCan",
    "Square": "NutAssemblySquare",
    "ToolHang": "ToolHang",
    "Transport": "TwoArmTransport",
}

#: Inverse mapping (for log/error messages).
ENV_NAME_TO_PROTOCOL: dict[str, str] = {v: k for k, v in PROTOCOL_TO_ENV_NAME.items()}

#: Protocol name -> expected state_dim. The values match the published PH
#: low-dimensional observation schemas.
PROTOCOL_STATE_DIM: dict[str, int] = {
    "Lift": 19,
    "Can": 23,
    "Square": 23,
    "ToolHang": 53,
    "Transport": 59,
}

#: Protocol name -> expected action_dim. Single-arm tasks use one Panda
#: OSC_POSE action (7 dimensions); TwoArmTransport uses concatenated
#: OSC_POSE actions for both arms (14 dimensions).
PROTOCOL_ACTION_DIM: dict[str, int] = {
    "Lift": 7,
    "Can": 7,
    "Square": 7,
    "ToolHang": 7,
    "Transport": 14,
}

#: Protocol name -> default horizon. The dataset's ``env_kwargs['horizon']``
#: is the authoritative value; this is the documented fallback.
PROTOCOL_HORIZON: dict[str, int] = {
    "Lift": 500,
    "Can": 500,
    "Square": 500,
    "ToolHang": 500,
    "Transport": 700,
}


@dataclass(frozen=True)
class TaskSpec:
    """Static, frozen description of a single robomimic benchmark task.

    Attributes
    ----------
    protocol_name:
        Short identifier used in configs, experiment manifests, and
        ``ResetBank.task``. One of the five protocol names listed in
        :data:`PROTOCOL_TO_ENV_NAME`.
    robosuite_env_name:
        The robosuite environment class name (e.g. ``"NutAssemblySquare"``
        for ``Square``). The parity gate compares this against the dataset's
        ``env_args['env_name']``.
    state_keys:
        Ordered observation keys in the published state vector.
    state_dims:
        Per-key dimensions. Total must equal ``state_dim``.
    state_dim:
        Convenience property == ``sum(state_dims)``.
    action_dim:
        Action vector dimensionality (Panda OSC_POSE = 7 for single-arm).
    horizon:
        Default episode horizon if the dataset's ``env_kwargs`` does not
        specify one (the dataset's value always wins via
        ``PinnedEnvMetadata.horizon``).
    schema_version:
        The schema identifier embedded in the cache manifest. Used by the
        cache manager to detect mismatched schemas on reuse.
    default_env_kwargs:
        Fallback ``env_args['env_kwargs']`` for environments that cannot
        reach the dataset (only used by the documented dev fallback path).
    hf_path:
        HuggingFace path on the ``amandlek/robomimic`` repo for the v1.5
        PH low-dim artifact.
    """

    protocol_name: str
    robosuite_env_name: str
    state_keys: tuple[str, ...]
    state_dims: tuple[int, ...]
    action_dim: int
    horizon: int
    schema_version: str
    default_env_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def state_dim(self) -> int:
        return sum(self.state_dims)

    @classmethod
    def from_protocol(cls, protocol_name: str) -> TaskSpec:
        """Construct the canonical :class:`TaskSpec` for a protocol name.

        Raises
        ------
        KeyError
            If ``protocol_name`` is not one of the five benchmark tasks.
        """
        if protocol_name not in PROTOCOL_TO_ENV_NAME:
            raise KeyError(
                f"Unknown protocol task {protocol_name!r}. "
                f"Expected one of {sorted(PROTOCOL_TO_ENV_NAME)}."
            )
        return _BUILD_SPECS[protocol_name]


#: Canonical state_keys / state_dims for each task. Lift uses a 10-D object
#: observation; Can and Square use 14-D object observations; ToolHang uses
#: three 14-D object poses plus two success indicators (44-D total).
#:
#: Transport publishes both arms' end-effector poses and grippers plus the
#: 41-dimensional object observation, summing to 59 dimensions.
_TRANSPORT_KEYS = (
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "robot1_eef_pos",
    "robot1_eef_quat",
    "robot1_gripper_qpos",
    "object",
)
_TRANSPORT_DIMS = (3, 4, 2, 3, 4, 2, 41)

_LIFT_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object")
_LIFT_DIMS = (3, 4, 2, 10)
_CAN_SQUARE_DIMS = (3, 4, 2, 14)
_TOOL_HANG_DIMS = (3, 4, 2, 44)

_DEFAULT_PANDA_KWARGS: dict[str, Any] = {
    "robots": "Panda",
    "controller_configs": None,
    "reward_shaping": True,
}
_DEFAULT_TWO_ARM_PANDA_KWARGS: dict[str, Any] = {
    **_DEFAULT_PANDA_KWARGS,
    "robots": ["Panda", "Panda"],
}

_BUILD_SPECS: dict[str, TaskSpec] = {
    "Lift": TaskSpec(
        protocol_name="Lift",
        robosuite_env_name="Lift",
        state_keys=_LIFT_KEYS,
        state_dims=_LIFT_DIMS,
        action_dim=7,
        horizon=500,
        schema_version="robomimic-lift-structured-v1",
        default_env_kwargs={**_DEFAULT_PANDA_KWARGS, "horizon": 500},
    ),
    "Can": TaskSpec(
        protocol_name="Can",
        robosuite_env_name="PickPlaceCan",
        state_keys=_LIFT_KEYS,
        state_dims=_CAN_SQUARE_DIMS,
        action_dim=7,
        horizon=500,
        schema_version="robomimic-can-structured-v2",
        default_env_kwargs={**_DEFAULT_PANDA_KWARGS, "horizon": 500},
    ),
    "Square": TaskSpec(
        protocol_name="Square",
        robosuite_env_name="NutAssemblySquare",
        state_keys=_LIFT_KEYS,
        state_dims=_CAN_SQUARE_DIMS,
        action_dim=7,
        horizon=500,
        schema_version="robomimic-square-structured-v2",
        default_env_kwargs={**_DEFAULT_PANDA_KWARGS, "horizon": 500},
    ),
    "ToolHang": TaskSpec(
        protocol_name="ToolHang",
        robosuite_env_name="ToolHang",
        state_keys=_LIFT_KEYS,
        state_dims=_TOOL_HANG_DIMS,
        action_dim=7,
        horizon=500,
        schema_version="robomimic-tool-hang-structured-v2",
        default_env_kwargs={**_DEFAULT_PANDA_KWARGS, "horizon": 500},
    ),
    "Transport": TaskSpec(
        protocol_name="Transport",
        robosuite_env_name="TwoArmTransport",
        # 59-dim two-arm layout per the robomimic v1.5 PH low-dim dataset.
        state_keys=_TRANSPORT_KEYS,
        state_dims=_TRANSPORT_DIMS,
        action_dim=14,
        horizon=700,
        schema_version="robomimic-transport-structured-v1",
        default_env_kwargs={**_DEFAULT_TWO_ARM_PANDA_KWARGS, "horizon": 700},
    ),
}


def known_protocol_tasks() -> list[str]:
    """Return the sorted list of supported protocol task names."""
    return sorted(_BUILD_SPECS)


def is_known_task(protocol_name: str) -> bool:
    """True if ``protocol_name`` is one of the five benchmark tasks."""
    return protocol_name in _BUILD_SPECS


def env_name_for_protocol(protocol_name: str) -> str:
    """Map a protocol name to its robosuite env_name. Raises KeyError."""
    return PROTOCOL_TO_ENV_NAME[protocol_name]


def protocol_for_env_name(env_name: str) -> str:
    """Map a robosuite env_name to its protocol name. Raises KeyError."""
    if env_name not in ENV_NAME_TO_PROTOCOL:
        raise KeyError(
            f"Unknown robosuite env_name {env_name!r}. "
            f"Expected one of {sorted(ENV_NAME_TO_PROTOCOL)}."
        )
    return ENV_NAME_TO_PROTOCOL[env_name]


def validate_task_schema(
    protocol_name: str,
    state_keys: tuple[str, ...] | list[str],
    state_dims: tuple[int, ...] | list[int],
    action_dim: int,
) -> None:
    """Validate a configured state/action contract against the task registry.

    Full structured-state configs must match the published key order and
    dimensions exactly. Robot-only negative controls are allowed as a
    deliberate alternate schema, but still have an explicit, task-specific
    contract and action dimension. This prevents a config from silently
    using another task's object state while preserving the negative-control
    experiment.
    """
    spec = TaskSpec.from_protocol(protocol_name)
    keys = tuple(str(key) for key in state_keys)
    dims = tuple(int(dim) for dim in state_dims)
    if len(keys) != len(dims):
        raise ValueError("state_keys and state_dims must have equal length")
    if int(action_dim) != spec.action_dim:
        raise ValueError(
            f"{protocol_name} action_dim={action_dim} does not match the "
            f"registered action_dim={spec.action_dim}"
        )
    if keys == spec.state_keys and dims == spec.state_dims:
        return

    robot_only = {
        "Lift": (
            (
                "robot0_joint_pos",
                "robot0_joint_vel",
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
            ),
            (7, 7, 3, 4, 2),
        ),
        "Can": (
            (
                "robot0_joint_pos",
                "robot0_joint_vel",
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
            ),
            (7, 7, 3, 4, 2),
        ),
        "Square": (
            (
                "robot0_joint_pos",
                "robot0_joint_vel",
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
            ),
            (7, 7, 3, 4, 2),
        ),
        "ToolHang": (
            (
                "robot0_joint_pos",
                "robot0_joint_vel",
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
            ),
            (7, 7, 3, 4, 2),
        ),
        "Transport": (
            (
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
                "robot1_eef_pos",
                "robot1_eef_quat",
                "robot1_gripper_qpos",
            ),
            (3, 4, 2, 3, 4, 2),
        ),
    }
    allowed_keys, allowed_dims = robot_only[protocol_name]
    if keys != allowed_keys or dims != allowed_dims:
        raise ValueError(
            f"{protocol_name} state schema does not match the canonical full "
            "schema or its registered robot-only negative-control schema: "
            f"keys={keys!r}, dims={dims!r}"
        )

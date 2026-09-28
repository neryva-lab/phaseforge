"""Simulator adapters for the state-only rollout protocol."""

from phaseforge.evaluations.envs.env_metadata import (
    PinnedEnvMetadata,
    dev_fallback_metadata,
    env_args_to_metadata,
    env_metadata_from_cache,
    env_metadata_from_hdf5,
    installed_versions,
    verify_environment_parity,
)
from phaseforge.evaluations.envs.errors import (
    EnvParityError,
    InfrastructureError,
    PolicyInvalidActionError,
    StateSchemaError,
)
from phaseforge.evaluations.envs.robosuite_adapter import (
    RobosuiteStateAdapter,
    StateSpec,
)
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
    validate_task_schema,
)

__all__ = [
    "PinnedEnvMetadata",
    "RobosuiteStateAdapter",
    "StateSpec",
    "EnvParityError",
    "InfrastructureError",
    "PolicyInvalidActionError",
    "StateSchemaError",
    "TaskSpec",
    "PROTOCOL_ACTION_DIM",
    "PROTOCOL_HORIZON",
    "PROTOCOL_STATE_DIM",
    "PROTOCOL_TO_ENV_NAME",
    "ENV_NAME_TO_PROTOCOL",
    "dev_fallback_metadata",
    "env_args_to_metadata",
    "env_metadata_from_cache",
    "env_metadata_from_hdf5",
    "env_name_for_protocol",
    "installed_versions",
    "is_known_task",
    "known_protocol_tasks",
    "protocol_for_env_name",
    "validate_task_schema",
    "verify_environment_parity",
]

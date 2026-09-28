"""Dynamic regime discovery sub-package."""

from __future__ import annotations

from phaseforge.data.dynamics.artifacts import (
    DISCOVERY_ARTIFACT_VERSION,
    load_discovery_artifact,
    save_discovery_artifact,
)
from phaseforge.data.dynamics.diagnostics import (
    DiscoveryQualityReport,
    evaluate_discovery_quality,
)
from phaseforge.data.dynamics.features import (
    TransitionBatch,
    extract_dataset_transitions,
    extract_trajectory_transitions,
)
from phaseforge.data.dynamics.switching_linear import (
    SingleDynamicsModel,
    SLDSParameters,
    StickySLDS,
)

__all__ = [
    "DISCOVERY_ARTIFACT_VERSION",
    "DiscoveryQualityReport",
    "SLDSParameters",
    "SingleDynamicsModel",
    "StickySLDS",
    "TransitionBatch",
    "evaluate_discovery_quality",
    "extract_dataset_transitions",
    "extract_trajectory_transitions",
    "load_discovery_artifact",
    "save_discovery_artifact",
]

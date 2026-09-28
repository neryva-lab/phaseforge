"""Pipeline state enum."""

from enum import Enum, auto


class PipelineState(Enum):
    CHECK_PERSISTENT_CACHE = auto()  # Entry state: check if pre-processed data exists
    VALIDATE_SOURCE = auto()  # Verify raw files exist + correct count
    PROVISION_SOURCE = auto()  # Optional: download missing raw data from the mirror
    INGEST_AND_STRIP = auto()  # Parse HDF5, strip vision, extract state + phases
    NORMALIZE_AND_SAVE = auto()  # Compute stats, normalize, persist to cache
    READY = auto()  # Terminal: DataLoader is ready to yield
    ERROR = auto()  # Terminal: unrecoverable failure

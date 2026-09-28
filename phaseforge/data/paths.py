"""Project-wide data path resolution.

This module is the SINGLE source of truth for where data lives on disk.
No other module should hard-code a data path. Resolve everything through
:func:`get_data_root` and the path constants below.

Directory contract
------------------
The data root is resolved from, in priority order:

1. The ``--data-root`` CLI flag (when a script passes it in).
2. The ``PHASEFORGE_DATA_DIR`` environment variable.
3. A default of ``./data`` relative to the current working directory.

This makes the code identical on local dev (``./data``) and on a cloud
box where the data root is a mounted volume (e.g. ``/mnt/data``) set via
the environment variable. The per-run ``outputs/`` directory (checkpoints,
logs, configs) is deliberately separate and never holds raw or processed
data.

Layout under the data root::

    {data_root}/
    ├── raw/                      # write-once, never modified after download
    │   └── {source}/             # one dir per dataset source (e.g. robomimic)
    └── processed/                # config-hash-keyed normalized cache (FSM output)
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Environment variable name and default
# ---------------------------------------------------------------------------

#: Name of the environment variable that overrides the data root.
DATA_DIR_ENV_VAR = "PHASEFORGE_DATA_DIR"

#: Default data root when neither the env var nor an explicit arg is set.
#: Relative to the current working directory.
DEFAULT_DATA_DIR = "data"

#: Subdirectory under the data root holding immutable downloaded sources.
RAW_SUBDIR = "raw"

#: Subdirectory under the data root holding normalized, cache-keyed output.
PROCESSED_SUBDIR = "processed"


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def get_data_root(override: str | os.PathLike | None = None) -> Path:
    """Resolve and return the project data root as an absolute ``Path``.

    Priority (highest first):

    1. ``override`` — an explicit path passed by a caller (e.g. a CLI flag).
    2. ``PHASEFORGE_DATA_DIR`` environment variable.
    3. ``./data`` (the default).

    The returned path is absolute but NOT necessarily existing; callers that
    need the directory to exist should call :func:`ensure_data_dirs`.

    Args:
        override: Optional explicit path. If given, it takes precedence over
            the environment variable and the default.

    Returns:
        Absolute :class:`~pathlib.Path` to the data root.
    """
    if override is not None and str(override).strip() != "":
        root = Path(override)
    else:
        env_val = os.environ.get(DATA_DIR_ENV_VAR)
        if env_val and env_val.strip() != "":
            root = Path(env_val)
        else:
            root = Path(DEFAULT_DATA_DIR)
    return root.resolve()


def ensure_data_dirs(data_root: str | os.PathLike | None = None) -> dict[str, Path]:
    """Create the standard data subdirectories if missing and return them.

    Creates (idempotently)::

        {data_root}/raw
        {data_root}/processed

    Args:
        data_root: Optional override forwarded to :func:`get_data_root`.

    Returns:
        Dict with keys ``"data_root"``, ``"raw"``, ``"processed"`` mapping
        to absolute :class:`~pathlib.Path` objects.
    """
    root = get_data_root(data_root)
    raw = root / RAW_SUBDIR
    processed = root / PROCESSED_SUBDIR
    raw.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    return {"data_root": root, "raw": raw, "processed": processed}


def processed_cache_root(data_root: str | os.PathLike | None = None) -> Path:
    """Return the shared, run-agnostic processed-cache root.

    Resolves to ``{data_root}/processed/cache``. This is intentionally
    UNDER THE DATA ROOT (not under the per-run ``outputs/`` directory) so
    that the config-hash-keyed cache built by :class:`CacheManager` is
    reused across training runs.

    Args:
        data_root: Optional override forwarded to :func:`get_data_root`.

    Returns:
        Absolute :class:`~pathlib.Path`. Not guaranteed to exist; the
        caller (the cache manager) creates it on first write.
    """
    return get_data_root(data_root) / PROCESSED_SUBDIR / "cache"

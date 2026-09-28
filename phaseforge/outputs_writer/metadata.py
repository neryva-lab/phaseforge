"""Environment fingerprint for one run (adapted from ``csd_observer``).

Captures the dependency versions, git state, host info, GPU silicon identity,
and dataset-cache identity that together make a single run reproducible from
its output directory alone. ``run_writer.write_environment()`` persists the dict
returned here as ``<run_dir>/metadata/environment.json``.

Version resolution never imports the fingerprinted packages: importing
sklearn/wandb/scipy just to read a version string costs seconds per training
process. Instead the version is read from already-imported modules when
available (free, and byte-identical to the historical fingerprint — e.g.
torch reports ``2.13.0+cpu`` from the module but ``2.13.0`` from
distribution metadata) and otherwise from a single-pass
``importlib.metadata.distributions()`` lookup.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import re
import socket
import sys
from pathlib import Path
from typing import Any

from phaseforge.utils.config import git_info

#: module import name -> installed-distribution name for the fingerprinted
#: packages whose two names differ.
_MODULE_TO_DIST: dict[str, str] = {
    "torch": "torch",
    "numpy": "numpy",
    "hydra": "hydra-core",
    "omegaconf": "omegaconf",
    "sklearn": "scikit-learn",
    "scipy": "scipy",
    "h5py": "h5py",
    "filelock": "filelock",
    "wandb": "wandb",
}

_INSTALLED_VERSIONS: dict[str, str] | None = None


def _normalize_dist_name(name: str) -> str:
    """PEP 503 name normalization (case and ``-``/``_``/``.`` runs)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _installed_versions() -> dict[str, str]:
    """``{normalized dist name: version}`` from ONE distributions() pass.

    Repeated ``importlib.metadata.version()`` calls rescan the environment
    each time; one pass + dict lookups is the robust form
    (importlib_metadata#95). Corrupted individual distributions are skipped,
    never fatal — a fingerprint must not crash a run.
    """
    global _INSTALLED_VERSIONS
    if _INSTALLED_VERSIONS is None:
        versions: dict[str, str] = {}
        try:
            for dist in importlib.metadata.distributions():
                try:
                    name = dist.metadata["Name"]
                    if name:
                        versions[_normalize_dist_name(str(name))] = str(dist.version)
                except Exception:  # noqa: BLE001 - one bad dist must not kill the pass
                    continue
        except Exception:  # noqa: BLE001
            pass
        _INSTALLED_VERSIONS = versions
    return _INSTALLED_VERSIONS


def _safe_version(modname: str) -> str | None:
    """Version of the distribution providing ``modname``, never importing it.

    Resolution order (cheapest and most faithful first):

    1. Already-imported module: read ``sys.modules[modname].__version__``.
    2. Metadata: the module->dist name map into the single-pass
       :func:`_installed_versions` table.

    Returns ``None`` when neither source has the version; never raises.
    """
    module = sys.modules.get(modname)
    if module is not None:
        version = getattr(module, "__version__", None)
        if isinstance(version, str) and version:
            return version
    dist_name = _MODULE_TO_DIST.get(modname, modname)
    return _installed_versions().get(_normalize_dist_name(dist_name))


def _cuda_device_identity() -> dict[str, Any] | None:
    """GPU silicon identity for cross-machine reproducibility forensics.

    Never imports torch: when the training process already imported it (the
    normal case — the model lives on CUDA), read the already-imported module
    from :data:`sys.modules`; otherwise return ``None`` rather than paying
    the import cost (same contract as :func:`_safe_version`). Any failure
    (CPU-only build, no device, driver issues) yields ``None``, never raises:
    a fingerprint must not crash a run.
    """
    module = sys.modules.get("torch")
    if module is None:
        return None
    try:
        cuda = getattr(module, "cuda", None)
        if cuda is None or not bool(cuda.is_available()):
            return None
        name = str(cuda.get_device_name(0))
        try:
            major, minor = cuda.get_device_capability(0)
            capability = f"{int(major)}.{int(minor)}"
        except Exception:  # noqa: BLE001 - capability is nice-to-have
            capability = None
        info: dict[str, Any] = {"name": name}
        if capability is not None:
            info["capability"] = capability
        return info
    except Exception:  # noqa: BLE001 - fingerprint must not crash a run
        return None


def collect_environment(
    *,
    data_config_hash: str | None = None,
    config_hash: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a single ``environment.json`` payload.

    Args:
        data_config_hash: Pipeline-level data config hash (the value
            printed by the state machine). The data hash is the cache
            identity; the full ``config_hash`` only matches when the
            data subset is included.
        config_hash: Full ``cfg`` config hash, written by
            :func:`phaseforge.utils.config.config_hash`.
        extra: Caller-supplied extras (e.g. resolved device, git branch).
    """
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cwd": str(Path.cwd()),
        "user": os.environ.get("USER") or os.environ.get("USERNAME") or "",
        "packages": {
            "torch": _safe_version("torch"),
            "numpy": _safe_version("numpy"),
            "hydra-core": _safe_version("hydra"),
            "omegaconf": _safe_version("omegaconf"),
            "scikit-learn": _safe_version("sklearn"),
            "scipy": _safe_version("scipy"),
            "h5py": _safe_version("h5py"),
            "filelock": _safe_version("filelock"),
            "wandb": _safe_version("wandb"),
        },
        "git_sha": git_info()["commit"],
        "git_branch": git_info()["branch"],
        # GPU silicon identity (None on CPU-only runs or when torch is not
        # yet imported). Cross-architecture floating-point divergence is a
        # real result confounder even with cudnn deterministic mode, so the
        # device name travels with every run's fingerprint.
        "gpu": _cuda_device_identity(),
        "data_config_hash": data_config_hash,
        "config_hash": config_hash,
    }
    if extra:
        env["extra"] = extra
    return env


__all__ = ["collect_environment"]

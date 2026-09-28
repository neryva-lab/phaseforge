"""Pinned environment metadata and the simulator/version parity gate.

The rollout protocol is only valid against the *same* simulator and
environment the dataset was collected with. The dataset pins this contract
in ``data.attrs["env_args"]`` (the output of robosuite's
``env.serialize()``: ``{env_name, env_version, type, env_kwargs}``). The
ingester already persists that block verbatim as ``env_metadata`` inside
every cached trajectory, so the parity gate can run from the processed
cache alone — no raw HDF5 needed on the evaluation machine.

The gate fails CLOSED: any mismatch between the installed robosuite/MuJoCo
versions, the environment name, or the schema contract raises
:class:`~phaseforge.evaluations.envs.errors.EnvParityError` before a single
rollout step.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

from packaging.version import InvalidVersion, Version

from phaseforge.evaluations.envs.errors import EnvParityError

#: robosuite versions are matched exactly when the dataset records one
#: (the collection branch per robomimic v0.4's v1.5 track was 1.5.1).
VERSION_MATCH_MODE: str = "exact"


#: Dev fallback env_args keyed by the protocol task name. Used only when
#: the dataset is not available locally (documented exception in the plan;
#: the evaluation machine always reads the pinned metadata from the
#: cache, never this fallback). The robosuite env_name is the canonical
#: class name (e.g. ``NutAssemblySquare`` for Square, ``TwoArmTransport``
#: for Transport) per the task registry.
def _dev_fallback_env_args(protocol_name: str) -> dict[str, Any]:
    from phaseforge.evaluations.envs.task_registry import (
        TaskSpec,
        env_name_for_protocol,
    )

    spec: TaskSpec = TaskSpec.from_protocol(protocol_name)
    return {
        "env_name": env_name_for_protocol(protocol_name),
        "env_version": "1.5.1",
        "type": "robosuite",
        "env_kwargs": dict(spec.default_env_kwargs),
    }


#: Backward-compat default — Lift, since that was the original single-task
#: pilot. New callers should use :func:`dev_fallback_metadata` with an
#: explicit protocol name.
DEV_FALLBACK_ENV_ARGS: dict[str, Any] = _dev_fallback_env_args("Lift")


@dataclass(frozen=True)
class PinnedEnvMetadata:
    """Validated environment contract pinned by the dataset ``env_args``."""

    env_name: str
    env_version: str
    env_type: str
    env_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def horizon(self) -> int:
        """The environment's episode horizon (dataset value, fallback 500)."""
        raw = self.env_kwargs.get("horizon", 500)
        try:
            return int(raw)
        except (TypeError, ValueError) as exc:
            raise EnvParityError(f"env_kwargs['horizon']={raw!r} is not an integer") from exc

    def canonical_json(self) -> str:
        """Deterministic canonical JSON — the bank/gate identity of the env."""
        payload = {
            "env_name": self.env_name,
            "env_version": self.env_version,
            "env_type": self.env_type,
            "env_kwargs": _canonical_env_kwargs(self.env_kwargs),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _canonical_env_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Env kwargs normalized for identity comparisons.

    The renderer/observation flags are forced by the adapter and are
    environment identity, not dataset identity — a dataset collected with
    renderer flags on must not mismatch a headless adapter run.
    """
    ignored = {
        "has_renderer",
        "has_offscreen_renderer",
        "use_camera_obs",
        "camera_depths",
        "ignore_done",
        "use_object_obs",
    }
    normalized: dict[str, Any] = {}
    for key, value in sorted(kwargs.items()):
        if key in ignored:
            continue
        if isinstance(value, (dict, list)):
            normalized[key] = json.dumps(value, sort_keys=True, default=str)
        else:
            normalized[key] = value
    return normalized


def env_args_to_metadata(env_args: dict[str, Any]) -> PinnedEnvMetadata:
    """Validate a raw ``env_args`` block into :class:`PinnedEnvMetadata`.

    Raises:
        EnvParityError: the block is missing or malformed.
    """
    if not isinstance(env_args, dict):
        raise EnvParityError(f"env_args must be a dict, got {type(env_args).__name__}")
    missing = [k for k in ("env_name", "env_version", "type") if k not in env_args]
    if missing:
        raise EnvParityError(
            f"env_args is missing required keys {missing}; the dataset was "
            "not serialized by robosuite's env.serialize()"
        )
    env_kwargs = env_args.get("env_kwargs")
    if env_kwargs is None:
        env_kwargs = {}
    if not isinstance(env_kwargs, dict):
        raise EnvParityError(
            f"env_args['env_kwargs'] must be a dict, got {type(env_kwargs).__name__}"
        )
    return PinnedEnvMetadata(
        env_name=str(env_args["env_name"]),
        env_version=str(env_args["env_version"]),
        env_type=str(env_args["type"]),
        env_kwargs=env_kwargs,
    )


def env_metadata_from_cache(
    cache_dir: str | Path, *, source: str = "cache_trajectory"
) -> PinnedEnvMetadata:
    """Recover the pinned env metadata from the processed cache.

    Reads ``trajectories/000000.pt`` only (the ingester persisted the full
    ``env_args`` block as ``env_metadata`` on every trajectory), so the
    parity gate never loads the dataset into RAM.

    Raises:
        EnvParityError: no trajectory or no valid ``env_metadata``.
    """
    import torch

    cache_dir = Path(cache_dir)
    traj_dir = cache_dir / "trajectories"
    files = sorted(traj_dir.glob("*.pt"))
    if not files:
        raise EnvParityError(
            f"Cache {cache_dir} has no trajectories — cannot recover the "
            "pinned environment metadata."
        )
    trajectory = torch.load(files[0], weights_only=False)
    raw = trajectory.get("env_metadata")
    if raw is None:
        raise EnvParityError(
            f"Cache trajectory {files[0].name} carries no 'env_metadata' — "
            "the cache predates env_args persistence; re-ingest the dataset."
        )
    meta = env_args_to_metadata(raw)
    return PinnedEnvMetadata(
        env_name=meta.env_name,
        env_version=meta.env_version,
        env_type=meta.env_type,
        env_kwargs=meta.env_kwargs,
    )


def env_metadata_from_hdf5(path: str | Path) -> PinnedEnvMetadata:
    """Recover the pinned env metadata directly from a raw HDF5 file.

    Mirrors the ingester's reader (``data.attrs["env_args"]``) so the
    replay gate can run against the raw dataset without a separate code
    path.
    """
    import json as _json

    import h5py
    import numpy as np

    with h5py.File(path, "r") as h5:
        data = h5.get("data")
        if not isinstance(data, h5py.Group):
            raise EnvParityError(f"{path} has no HDF5 data group")
        raw = data.attrs.get("env_args")
    if raw is None:
        raise EnvParityError(f"{path} is missing data.attrs['env_args']")
    if isinstance(raw, np.ndarray) and raw.ndim == 0:
        raw = raw.item()
    if isinstance(raw, (bytes, np.bytes_)):
        raw = raw.decode("utf-8")
    try:
        env_args = _json.loads(str(raw))
    except (TypeError, _json.JSONDecodeError) as exc:
        raise EnvParityError(f"{path} has invalid data.attrs['env_args']") from exc
    if not isinstance(env_args, dict):
        raise EnvParityError(f"{path} env_args must decode to a JSON object")
    return env_args_to_metadata(env_args)


def installed_versions() -> dict[str, str]:
    """Installed robosuite/MuJoCo versions from package metadata.

    Uses ``importlib.metadata`` (never imports the packages), so the gate
    can report cleanly when the rollout extra is absent.
    """
    out: dict[str, str] = {}
    for package in ("robosuite", "mujoco"):
        try:
            out[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            out[package] = ""
    return out


def _version_tuple(version: str) -> tuple[int, ...]:
    try:
        parsed = Version(version)
    except InvalidVersion:
        raise EnvParityError(f"Unparseable version string {version!r}")
    return parsed.release


def verify_environment_parity(
    meta: PinnedEnvMetadata,
    *,
    expected_env_name: str,
    robosuite_requirement: str,
    mujoco_requirement: str,
) -> dict[str, str]:
    """Fail-closed parity check of installed simulator vs pinned metadata.

    Verifies, aggregating every mismatch into the raised error:

    * robosuite is installed and its version matches ``meta.env_version``
      (exact match; the dataset records the collection version);
    * MuJoCo satisfies ``mujoco_requirement``;
    * the environment name matches ``expected_env_name``.

    The ``robosuite_requirement`` argument is the project-wide default
    (used when the dataset's ``env_args`` omit ``env_version``). When the
    dataset records its own ``env_version``, the parity gate uses that
    exact value instead — the dataset's recorded version is authoritative
    and never bypassed. The five robomimic PH low-dim datasets split into
    two version pins:

    * Lift, Can, Square, Transport → robosuite 1.5.1;
    * ToolHang → robosuite 1.5.0 (use a separately pinned 1.5.0 venv).

    Returns the installed ``{"robosuite": ..., "mujoco": ...}`` versions
    on success (the values are also used for provenance records).

    Raises:
        EnvParityError: any check fails (fail closed — no rollouts).
    """
    installed = installed_versions()
    problems: list[str] = []

    robosuite = installed["robosuite"]
    if not robosuite:
        problems.append(
            "robosuite is not installed. Install the rollout extra "
            "(`uv sync --extra rollout`) on the evaluation machine."
        )
    else:
        dataset_version = meta.env_version or ""
        if dataset_version:
            if _version_tuple(robosuite) != _version_tuple(dataset_version):
                problems.append(
                    f"robosuite {robosuite} installed but the dataset was "
                    f"collected with robosuite {dataset_version} "
                    f"(env_args['env_version']). The dataset's recorded "
                    "version is authoritative — install the matching "
                    f"robosuite release (e.g. `uv pip install "
                    f"robosuite=={dataset_version}` in a dedicated venv) "
                    "or evaluate a different dataset. The project-wide "
                    f"default is {robosuite_requirement}; per-task pins "
                    "are declared in `data/<task>.yaml`."
                )
        elif not _satisfies(robosuite, robosuite_requirement):
            problems.append(
                f"robosuite {robosuite} does not satisfy the pinned "
                f"requirement {robosuite_requirement}."
            )

    mujoco = installed["mujoco"]
    if not mujoco:
        problems.append("mujoco is not installed (robosuite dependency).")
    elif not _satisfies(mujoco, mujoco_requirement):
        problems.append(
            f"mujoco {mujoco} does not satisfy the pinned requirement {mujoco_requirement}."
        )

    if meta.env_name != expected_env_name:
        problems.append(
            f"environment name mismatch: dataset records {meta.env_name!r} "
            f"but the protocol expects {expected_env_name!r}."
        )

    if problems:
        raise EnvParityError(
            "Environment parity gate FAILED (fail closed, no rollouts):\n  - "
            + "\n  - ".join(problems)
        )
    return installed


def _satisfies(version: str, requirement: str) -> bool:
    try:
        from packaging.specifiers import SpecifierSet

        return Version(version) in SpecifierSet(requirement)
    except InvalidVersion as exc:
        raise EnvParityError(
            f"Unparseable requirement {requirement!r} or version {version!r}"
        ) from exc


def dev_fallback_metadata(protocol_name: str = "Lift") -> PinnedEnvMetadata:
    """Documented local-dev fallback when no dataset/cache is available.

    Parameters
    ----------
    protocol_name:
        One of the five benchmark task protocol names (e.g. ``"Lift"``,
        ``"Can"``, ``"Square"``, ``"ToolHang"``, ``"Transport"``). The
        matching robosuite env_name is resolved through the task
        registry.

    The parity gate and eval path on the evaluation machine always use the
    cache-derived metadata; this fallback exists only for running the
    environment self-tests without the dataset present. It is never
    acceptable for a real rollout.
    """
    return env_args_to_metadata(_dev_fallback_env_args(protocol_name))


__all__ = [
    "PinnedEnvMetadata",
    "DEV_FALLBACK_ENV_ARGS",
    "env_args_to_metadata",
    "env_metadata_from_cache",
    "env_metadata_from_hdf5",
    "installed_versions",
    "verify_environment_parity",
    "dev_fallback_metadata",
]

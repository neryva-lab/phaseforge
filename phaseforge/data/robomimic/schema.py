"""Fail-closed schema verification for robomimic state-only HDF5 files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from phaseforge.evaluations.envs.task_registry import TaskSpec, validate_task_schema


@dataclass(frozen=True)
class DatasetSchemaReport:
    """Observed schema facts for one HDF5 artifact."""

    path: str
    protocol_name: str
    env_name: str
    state_keys: tuple[str, ...]
    state_dims: tuple[int, ...]
    action_dim: int
    demo_count: int


def _decode_env_args(raw: Any, path: Path) -> dict[str, Any]:
    if isinstance(raw, np.ndarray) and raw.ndim == 0:
        raw = raw.item()
    if isinstance(raw, (bytes, np.bytes_)):
        raw = raw.decode("utf-8")
    try:
        value = json.loads(str(raw))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path} has invalid data.attrs['env_args']") from exc
    if not isinstance(value, dict) or not isinstance(value.get("env_name"), str):
        raise ValueError(f"{path} env_args must contain a string env_name")
    return value


def inspect_hdf5_schema(path: str | Path, protocol_name: str) -> DatasetSchemaReport:
    """Inspect every demo's declared low-dimensional keys and actions.

    This does not load image observations and does not download anything. It
    rejects a file if any demo changes a key dimension, omits a required key,
    contains a different action width, or records a different robosuite task
    than the protocol registry.
    """
    path = Path(path)
    spec = TaskSpec.from_protocol(protocol_name)
    aliases = {"object": ("object", "object-state")}
    observed_dims: tuple[int, ...] | None = None
    observed_action_dim: int | None = None
    demo_count = 0
    with h5py.File(path, "r") as h5:
        data = h5.get("data")
        if not isinstance(data, h5py.Group):
            raise ValueError(f"{path} has no HDF5 data group")
        env_args = _decode_env_args(data.attrs.get("env_args"), path)
        env_name = str(env_args["env_name"])
        if env_name != spec.robosuite_env_name:
            raise ValueError(
                f"{path} records env_name={env_name!r}, but {protocol_name} "
                f"requires {spec.robosuite_env_name!r}"
            )
        for demo_name in sorted(name for name in data if name.startswith("demo_")):
            demo = data[demo_name]
            obs = demo.get("obs")
            if not isinstance(obs, h5py.Group):
                raise ValueError(f"{path}:{demo_name} has no obs group")
            dims: list[int] = []
            length: int | None = None
            for key, dim in zip(spec.state_keys, spec.state_dims):
                dataset = obs.get(key)
                if dataset is None and key in aliases:
                    dataset = next(
                        (obs.get(alias) for alias in aliases[key] if obs.get(alias) is not None),
                        None,
                    )
                if not isinstance(dataset, h5py.Dataset):
                    raise ValueError(f"{path}:{demo_name} missing observation key {key!r}")
                shape = tuple(int(x) for x in dataset.shape)
                if len(shape) != 2 or shape[1] != dim:
                    raise ValueError(
                        f"{path}:{demo_name}:{key} has shape {shape}, expected (T,{dim})"
                    )
                if length is None:
                    length = shape[0]
                elif length != shape[0]:
                    raise ValueError(f"{path}:{demo_name} has inconsistent observation lengths")
                dims.append(shape[1])
            actions = demo.get("actions")
            if not isinstance(actions, h5py.Dataset) or actions.ndim != 2:
                raise ValueError(f"{path}:{demo_name} actions must be a rank-2 dataset")
            if length != int(actions.shape[0]):
                raise ValueError(f"{path}:{demo_name} action/state lengths differ")
            if observed_action_dim is None:
                observed_action_dim = int(actions.shape[1])
            elif observed_action_dim != int(actions.shape[1]):
                raise ValueError(f"{path} changes action width across demonstrations")
            if observed_dims is None:
                observed_dims = tuple(dims)
            elif observed_dims != tuple(dims):
                raise ValueError(f"{path} changes state widths across demonstrations")
            demo_count += 1
    if demo_count == 0:
        raise ValueError(f"{path} contains no demo_* groups")
    assert observed_dims is not None and observed_action_dim is not None
    validate_task_schema(protocol_name, spec.state_keys, observed_dims, observed_action_dim)
    return DatasetSchemaReport(
        path=str(path),
        protocol_name=protocol_name,
        env_name=env_name,
        state_keys=spec.state_keys,
        state_dims=observed_dims,
        action_dim=observed_action_dim,
        demo_count=demo_count,
    )


__all__ = ["DatasetSchemaReport", "inspect_hdf5_schema"]

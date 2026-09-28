"""State-only robosuite adapter.

The adapter wraps a robosuite environment behind the exact contracts the
robomimic wrapper used to collect the dataset (``robomimic/envs/robosuite_env.py``
v1.5 track):

* actions are passed **directly** to ``env.step`` — no rescaling. The
  dataset actions are already normalized deltas in ``[-1, 1]``; the
  controller scales them internally via ``input_min/input_max/output_max``.
* state restore mirrors ``reset_to``: ``set_ep_meta`` →
  ``reset_from_xml_string(xml)`` → ``sim.set_state_from_flattened(states)``
  → ``sim.forward()``. A flat ``MjSimState`` is ``[time(1), qpos(nq), qvel(nv)]``.
* the success predicate is the environment's own ``_check_success()``
  (Lift: cube height above table + 0.04).

Failure classification is strict (see :mod:`errors`): the adapter never
swallows simulator errors into task outcomes, and it rejects invalid
policy actions with :class:`PolicyInvalidActionError` before touching the
simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from phaseforge.evaluations.envs.env_metadata import PinnedEnvMetadata
from phaseforge.evaluations.envs.errors import (
    EnvParityError,
    InfrastructureError,
    PolicyInvalidActionError,
    StateSchemaError,
)

#: Observation keys forced to match the dataset collection pipeline.
#: ``camera_depths`` is passed through because robosuite 1.5 Lift accepts
#: it as a constructor kwarg (the dataset records it in env_kwargs).
_FORCED_ENV_KWARGS: dict[str, Any] = {
    "has_renderer": False,
    "has_offscreen_renderer": False,
    "use_camera_obs": False,
    "camera_depths": False,
    "ignore_done": True,
    "use_object_obs": True,
}

#: Legacy observation-key alias: robomimic datasets store the object
#: observation under ``object-state`` (classic) or ``object`` (1.5).
_OBS_ALIASES: dict[str, str] = {"object": "object-state"}


@dataclass(frozen=True)
class StateSpec:
    """Declared state schema: ordered ``(key, dim)`` pairs in the contract."""

    keys: tuple[str, ...]
    dims: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.keys) != len(self.dims):
            raise ValueError(
                f"state_keys and state_dims must have equal length "
                f"(got {len(self.keys)} keys vs {len(self.dims)} dims)"
            )
        if not self.keys:
            raise ValueError("state schema must declare at least one key")
        if any(d <= 0 for d in self.dims):
            raise ValueError("state dims must be positive")

    @property
    def dim(self) -> int:
        return sum(self.dims)

    def index_of(self, key: str) -> tuple[int, int]:
        """Slice ``(start, stop)`` of a key in the concatenated state vector."""
        offset = 0
        for candidate, dim in zip(self.keys, self.dims):
            if candidate == key:
                return offset, offset + dim
            offset += dim
        raise KeyError(f"{key!r} is not part of the state schema {self.keys}")

    @classmethod
    def from_config(cls, state_keys: Any, state_dims: Any) -> StateSpec:
        keys = tuple(str(k) for k in state_keys)
        dims = tuple(int(d) for d in state_dims)
        return cls(keys=keys, dims=dims)


class RobosuiteStateAdapter:
    """Stateless policy-environment adapter for robosuite rollouts.

    The adapter is stateful only about its underlying environment; the
    reset distribution is supplied externally (the frozen reset bank), so
    every episode starts from a pinned simulator state.
    """

    def __init__(
        self,
        meta: PinnedEnvMetadata,
        state_spec: StateSpec,
        *,
        action_dim: int = 7,
        action_low: float = -1.0,
        action_high: float = 1.0,
        horizon: int | None = None,
        seed: int | None = None,
        action_tolerance: float = 1e-4,
    ) -> None:
        self.meta = meta
        self.state_spec = state_spec
        self.action_dim = int(action_dim)
        self.action_low = float(action_low)
        self.action_high = float(action_high)
        # Single source of truth for the action-contract tolerance. Both the
        # runner's pre-step check and ``step``'s own guard validate with this
        # value, so a nondefault ``eval.episodes.action_tolerance`` cannot be
        # silently overridden by the adapter's historical hardcoded default.
        self.action_tolerance = float(action_tolerance)
        # The v1.5.1 transport dataset does not serialize ``horizon`` in its
        # env_args, although the evaluation protocol declares a 700-step
        # Transport horizon.  The runner supplies the task-aware fallback;
        # explicit metadata remains authoritative when it is present.
        self._horizon = int(meta.horizon if horizon is None else horizon)
        if self._horizon <= 0:
            raise ValueError(f"horizon must be positive, got {self._horizon}")
        # Keep the protocol seed as adapter metadata, but do not forward it
        # to robosuite.make().  In robosuite 1.5.1 the base MujocoEnv accepts
        # ``seed`` while task classes such as Lift do not expose or forward
        # that argument, so passing it to the registered task raises a
        # TypeError.  Reset-bank generation seeds robosuite's actual global
        # NumPy sampler explicitly instead.
        self.seed = None if seed is None else int(seed)

        try:
            import robosuite
        except ImportError as exc:
            raise EnvParityError(
                "robosuite is not installed. Install the rollout extra "
                "(`uv sync --extra rollout`) — the rollout protocol cannot "
                "run without the simulator."
            ) from exc
        self._robosuite = robosuite

        kwargs = {**meta.env_kwargs, **_FORCED_ENV_KWARGS}
        # Some serialized metadata may carry a generic seed field even
        # though the concrete robosuite task constructor does not accept it.
        # The rollout protocol controls reset determinism in
        # ``generate_reset_bank`` instead of relying on this task-level kwarg.
        kwargs.pop("seed", None)
        try:
            self.env = robosuite.make(meta.env_name, **kwargs)
        except Exception as exc:
            raise EnvParityError(
                f"robosuite.make({meta.env_name!r}) failed with the pinned env_kwargs: {exc}"
            ) from exc

        try:
            spec = self.env.action_spec
            env_action_dim = int(np.asarray(spec[0]).shape[0])
        except Exception as exc:
            raise EnvParityError(
                f"Could not read the action specification of {meta.env_name!r}"
            ) from exc
        if env_action_dim != self.action_dim:
            raise EnvParityError(
                f"Action dimension mismatch: the environment reports "
                f"{env_action_dim} but the data contract declares "
                f"{self.action_dim}. Refusing to evaluate a different "
                "action space than the dataset."
            )

    # ------------------------------------------------------------------
    # State extraction
    # ------------------------------------------------------------------

    def extract_state(self, obs: dict[str, Any]) -> np.ndarray:
        """Concatenate the declared state keys from a robosuite obs dict."""
        parts: list[np.ndarray] = []
        for key, dim in zip(self.state_spec.keys, self.state_spec.dims):
            value = obs.get(key)
            if value is None:
                alias = _OBS_ALIASES.get(key)
                if alias is not None:
                    value = obs.get(alias)
            if value is None:
                for prefix in ("robot0_", "robot1_", ""):
                    if key == f"{prefix}joint_pos":
                        cos_k = f"{prefix}joint_pos_cos"
                        sin_k = f"{prefix}joint_pos_sin"
                        if cos_k in obs and sin_k in obs:
                            cos_val = np.asarray(obs[cos_k], dtype=np.float32).reshape(-1)
                            sin_val = np.asarray(obs[sin_k], dtype=np.float32).reshape(-1)
                            value = np.arctan2(sin_val, cos_val)
                            break
            if value is None:
                raise StateSchemaError(
                    f"Observation is missing declared key {key!r} (available: {sorted(obs)})."
                )
            arr = np.asarray(value).reshape(-1)
            if arr.shape[0] != dim:
                raise StateSchemaError(
                    f"Observation key {key!r} has dimension {arr.shape[0]}, declared {dim}."
                )
            parts.append(arr.astype(np.float32))
        state = np.concatenate(parts)
        if not np.isfinite(state).all():
            raise StateSchemaError(
                "Simulator observation contains non-finite values — the "
                "simulation is invalid; this is an infrastructure failure."
            )
        return state

    # ------------------------------------------------------------------
    # Reset (mirrors robomimic reset_to)
    # ------------------------------------------------------------------

    def reset_to(
        self,
        states: np.ndarray,
        *,
        xml: str | None = None,
        ep_meta: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Restore a pinned simulator state and return the initial obs state.

        Mirrors ``EnvRobosuite.reset_to`` step-for-step so the reset
        distribution is exactly the dataset's.
        """
        try:
            if ep_meta is not None and hasattr(self.env, "set_ep_meta"):
                self.env.set_ep_meta(ep_meta)
            if xml is not None:
                self.env.reset_from_xml_string(xml)
            else:
                self.env.reset()
            self.env.sim.set_state_from_flattened(np.asarray(states, dtype=np.float64))
            self.env.sim.forward()
        except Exception as exc:
            raise InfrastructureError(
                f"State restore failed for reset case (xml={xml is not None}): {exc}"
            ) from exc
        try:
            obs = self.env._get_observations(force_update=True)
        except Exception as exc:
            raise InfrastructureError(
                f"Observation extraction failed after state restore: {exc}"
            ) from exc
        return self.extract_state(obs)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def validate_action(
        self, action: np.ndarray, *, tolerance: float | None = None
    ) -> np.ndarray:
        """Validate a policy action against the data contract; return float64.

        ``tolerance=None`` resolves to the adapter's configured
        ``action_tolerance`` (from ``eval.episodes.action_tolerance``), so
        every validation site enforces the same contract.

        Raises:
            PolicyInvalidActionError: NaN, non-finite, wrong shape, or out
                of the declared ``[action_low, action_high]`` range (with
                ``tolerance``).
        """
        if tolerance is None:
            tolerance = self.action_tolerance
        arr = np.asarray(action)
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr.reshape(-1)
        if arr.shape != (self.action_dim,):
            raise PolicyInvalidActionError(
                f"Action has shape {arr.shape}, expected ({self.action_dim},)"
            )
        if not np.isfinite(arr).all():
            bad = int(np.count_nonzero(~np.isfinite(arr)))
            raise PolicyInvalidActionError(f"Action contains {bad} non-finite value(s) (NaN/Inf).")
        low = self.action_low - tolerance
        high = self.action_high + tolerance
        if float(arr.min()) < low or float(arr.max()) > high:
            raise PolicyInvalidActionError(
                f"Action outside the declared contract "
                f"[{self.action_low}, {self.action_high}] (+{tolerance} tol): "
                f"min={float(arr.min()):.4g}, max={float(arr.max()):.4g}."
            )
        return arr.astype(np.float64)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, bool, bool, dict[str, Any]]:
        """Execute one policy action; returns ``(state, done, success, info)``.

        ``done`` is always False by construction (``ignore_done=True`` —
        horizon termination is the runner's responsibility). ``success`` is
        the environment's own predicate. Simulator exceptions are wrapped
        as :class:`InfrastructureError`, never as task outcomes.
        """
        validated = self.validate_action(action)
        try:
            obs, _reward, _done, _info = self.env.step(validated)
        except Exception as exc:
            raise InfrastructureError(
                f"Simulator step failed: {type(exc).__name__}: {exc}"
            ) from exc
        state = self.extract_state(obs)
        success = self.check_success()
        return state, False, success, {}

    def check_success(self) -> bool:
        """The environment's own success predicate (Lift: ``task`` key)."""
        try:
            result = self.env._check_success()
        except Exception as exc:
            raise InfrastructureError(
                f"Success predicate failed: {type(exc).__name__}: {exc}"
            ) from exc
        if isinstance(result, dict):
            task_success = result.get("task")
            if task_success is None:
                raise InfrastructureError(
                    f"Success predicate dict has no 'task' key: {sorted(result)}"
                )
            return bool(task_success)
        return bool(result)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    @property
    def horizon(self) -> int:
        return self._horizon

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass


__all__ = [
    "StateSpec",
    "RobosuiteStateAdapter",
    "_FORCED_ENV_KWARGS",
]

"""Deterministic auxiliary phase labels for low-dimensional demonstrations.

These labels are a training signal for the PhaseForge routing study. They are
not environment annotations and are never supplied to the policy at inference.

Threshold calibration is adaptive: the closed/open gripper levels are derived
per demonstration from the observed aperture range (5-95 percentiles, with the
grasp/transport-dominated middle of the demo disambiguating the sign), so any
robosuite gripper convention labels correctly without per-env tuning — e.g.
the Panda finger qpos range [-0.04, 0.04] with its antisymmetric fingers,
or a magnitude convention like [0, 0.04]. The aperture feature is the finger excursion
magnitude ``max(|q0|, |q1|)``: the mean of the two finger qpos is ~constant
for parallel-jaw grippers and cannot separate open from closed. The
configured absolute thresholds are only fallbacks for degenerate
demonstrations with a (nearly) constant aperture. Labels are computed offline
during ingestion (like the train-split normalization statistics); the phase
state machine and its smoothing are causal conditional on those thresholds.
Because calibration uses the complete demonstration, these are offline
auxiliary annotations rather than causal online labels; they are never
supplied to PhaseForge at inference.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

_MIN_SAMPLES_FOR_CALIBRATION = 8
_MIN_SPAN_FOR_CALIBRATION = 1e-3


class RuleBasedPhaseLabeler:
    """Assign coarse approach/grasp/transport/place/retract phases.

    The labeler only reads the configured end-effector position and gripper
    slices from the unnormalized state. It deliberately does not inspect
    actions, object state, or task metadata. Its adaptive calibration is an
    offline full-trajectory operation, so its outputs are not causal online
    annotations.
    """

    def __init__(
        self,
        num_phases: int = 6,
        gripper_closed_threshold: float = 0.02,
        gripper_open_threshold: float = 0.04,
        eef_velocity_threshold: float = 0.01,
        min_phase_duration: int = 5,
        median_filter_size: int = 7,
        eef_pos_slice: tuple[int, int] = (0, 3),
        gripper_qpos_slice: tuple[int, int] = (7, 9),
    ) -> None:
        if num_phases != 6:
            raise ValueError(
                "The current phase vocabulary has exactly 6 phases; use a "
                "different labeler before changing num_phases."
            )
        if not 0 < min_phase_duration:
            raise ValueError("min_phase_duration must be positive")
        if median_filter_size < 1:
            raise ValueError("median_filter_size must be a positive integer")
        if not gripper_closed_threshold < gripper_open_threshold:
            raise ValueError("gripper_closed_threshold must be below gripper_open_threshold")
        self.num_phases = int(num_phases)
        self.closed_threshold = float(gripper_closed_threshold)
        self.open_threshold = float(gripper_open_threshold)
        self.velocity_threshold = float(eef_velocity_threshold)
        self.min_duration = int(min_phase_duration)
        self.filter_size = int(median_filter_size)
        self.eef_pos_slice = tuple(int(v) for v in eef_pos_slice)
        self.gripper_slice = tuple(int(v) for v in gripper_qpos_slice)

    def _calibrate_impl(
        self, aperture: np.ndarray
    ) -> tuple[float, float, bool, float | None, float | None]:
        """Return ``(closed_level, open_level, mirror, lo, hi)``.

        ``closed_level``/``open_level`` are the hysteresis bands in the
        (possibly mirrored) signal space; ``mirror`` says whether the
        hysteresis must mirror the aperture before comparing (closed at the
        high extreme); ``lo``/``hi`` are the mirror axis endpoints (``None``
        when not mirrored). Degenerate demonstrations (nearly constant or
        too short) fall back to the configured absolute thresholds with no
        mirroring.
        """
        if aperture.size < _MIN_SAMPLES_FOR_CALIBRATION:
            return self.closed_threshold, self.open_threshold, False, None, None
        lo = float(np.percentile(aperture, 5))
        hi = float(np.percentile(aperture, 95))
        span = hi - lo
        if span <= _MIN_SPAN_FOR_CALIBRATION:
            return self.closed_threshold, self.open_threshold, False, None, None
        middle = aperture[aperture.size // 4 : 3 * aperture.size // 4]
        mid = float(np.median(middle))
        mirror = mid >= lo + 0.5 * span
        return lo + 0.3 * span, hi - 0.3 * span, mirror, lo, hi

    def _calibrate_aperture(self, aperture: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Return (signal, closed_level, open_level) for the hysteresis.

        The levels are calibrated from the demonstration's observed aperture
        range so the closed/open detection is independent of the gripper's
        absolute scale and sign convention. The gripper is open at reset and
        mostly closed through grasp/transport/place, so:

        - ``lo``/``hi`` are the 5/95 percentiles of the aperture (robust to
          contact spikes at the extremes);
        - the middle-50% median tells which extreme the closed position is
          on; if closed is the high side, the signal is mirrored so the
          hysteresis can treat closed as low;
        - the hysteresis bands sit 30% inside the observed span, keeping
          contact noise out of the levels.

        Falls back to the configured absolute thresholds when the aperture is
        (nearly) constant or the demonstration is too short to calibrate.
        """
        closed_level, open_level, mirror, lo, hi = self._calibrate_impl(aperture)
        if mirror:
            aperture = (lo + hi) - aperture
        return aperture, closed_level, open_level

    def _features(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Validate the state array and extract the aperture + speed features."""
        if state.ndim != 2:
            raise ValueError(f"Expected state shape (T, D), got {state.shape}")
        T, D = state.shape
        e0, e1 = self.eef_pos_slice
        g0, g1 = self.gripper_slice
        if not (0 <= e0 < e1 <= D and 0 <= g0 < g1 <= D):
            raise ValueError(
                f"Phase slices eef={self.eef_pos_slice}, gripper={self.gripper_slice} "
                f"are invalid for state_dim={D}"
            )
        eef = state[:, e0:e1]
        fingers = state[:, g0:g1]
        # Parallel-jaw grippers (panda finger qpos spans [-0.04, 0.04] with the
        # two fingers antisymmetric: open = [+0.0208, -0.0208], closed =
        # [+0.04, -0.04]) make the MEAN of the two finger qpos ~constant, so
        # the closed/open signal is the finger excursion magnitude instead.
        aperture = np.max(np.abs(fingers), axis=1)
        speed = np.linalg.norm(np.diff(eef, axis=0, prepend=eef[:1]), axis=1)
        return aperture, speed

    def calibrate(self, traj: dict[str, Any]) -> dict[str, Any]:
        """Calibrate the phase thresholds from one full demonstration.

        Returns a serializable calibration artifact that fully determines
        the per-step phase classification, including every parameter the
        hysteresis and state machine need:

        * ``closed_level``/``open_level`` — the hysteresis bands (in the
          signal space the state machine compares against);
        * ``mirror`` — whether the aperture must be mirrored before the
          hysteresis compares (closed at the high extreme);
        * ``mirror_bounds`` — the ``[lo, hi]`` mirror axis endpoints, or
          ``None`` when not mirrored;
        * ``velocity_threshold``/``min_duration``/``filter_size``/
          ``eef_pos_slice``/``gripper_qpos_slice``/``num_phases`` — the
          state-machine and smoothing parameters.

        The artifact is what :class:`CausalPhaseStepLabeler` consumes at
        rollout time: the same demonstration's per-step labels then match
        :meth:`label` exactly (for trajectories longer than the filter
        size). It is persisted per demonstration during ingestion so a
        rollout never re-derives thresholds from the policy's own states.
        """
        state = np.asarray(traj["state"], dtype=np.float32)
        aperture, _speed = self._features(state)
        closed_level, open_level, mirror, lo, hi = self._calibrate_impl(aperture)
        return {
            "closed_level": float(closed_level),
            "open_level": float(open_level),
            "mirror": bool(mirror),
            "mirror_bounds": [float(lo), float(hi)] if mirror else None,
            "velocity_threshold": float(self.velocity_threshold),
            "min_duration": int(self.min_duration),
            "filter_size": int(self.filter_size),
            "eef_pos_slice": [int(v) for v in self.eef_pos_slice],
            "gripper_qpos_slice": [int(v) for v in self.gripper_slice],
            "num_phases": int(self.num_phases),
        }

    def label(self, traj: dict[str, Any]) -> np.ndarray:
        state = np.asarray(traj["state"], dtype=np.float32)
        if state.ndim != 2:
            raise ValueError(f"Expected state shape (T, D), got {state.shape}")
        T, D = state.shape
        e0, e1 = self.eef_pos_slice
        g0, g1 = self.gripper_slice
        if not (0 <= e0 < e1 <= D and 0 <= g0 < g1 <= D):
            raise ValueError(
                f"Phase slices eef={self.eef_pos_slice}, gripper={self.gripper_slice} "
                f"are invalid for state_dim={D}"
            )
        if T == 0:
            return np.zeros(0, dtype=np.int64)

        aperture, speed = self._features(state)

        aperture, closed_level, open_level = self._calibrate_aperture(aperture)

        closed = np.zeros(T, dtype=bool)
        previous = False
        for t, value in enumerate(aperture):
            if value < closed_level:
                previous = True
            elif value > open_level:
                previous = False
            closed[t] = previous

        transitions = np.diff(closed.astype(np.int8), prepend=0)
        grasp = transitions > 0
        release = transitions < 0

        # 0 approach, 1 pre-grasp, 2 grasp, 3 transport, 4 place, 5 retract.
        phases = np.zeros(T, dtype=np.int64)
        phase = 0
        entered = 0
        for t in range(T):
            held = t - entered >= self.min_duration
            if grasp[t]:
                phase, entered = 2, t
            elif release[t]:
                phase, entered = 4, t
            elif (
                phase == 0
                and t >= self.min_duration
                and not closed[t]
                and speed[t] < self.velocity_threshold
            ):
                # Causal pre-grasp state: the demonstrator has finished the
                # approach and is stationary while the gripper is open.
                phase, entered = 1, t
            elif phase == 2 and held and closed[t] and speed[t] > self.velocity_threshold:
                phase, entered = 3, t
            elif phase == 4 and held and not closed[t] and speed[t] > self.velocity_threshold:
                phase, entered = 5, t
            elif phase == 5 and speed[t] < self.velocity_threshold:
                phase, entered = 0, t
            phases[t] = phase

        if T > self.filter_size:
            # scipy.ndimage.median_filter uses a centered window by default.
            # Apply the same median idea causally conditional on the offline
            # calibrated thresholds, using only [t-size+1, t].
            causal = np.empty_like(phases, dtype=np.float32)
            for t in range(T):
                start = max(0, t - self.filter_size + 1)
                causal[t] = np.median(phases[start : t + 1])
            phases = causal.astype(np.int64)
        return np.clip(phases, 0, self.num_phases - 1).astype(np.int64)


class CausalPhaseStepLabeler:
    """Per-step phase labeler that exactly replicates :meth:`RuleBasedPhaseLabeler.label`.

    Consumes a calibration artifact from
    :meth:`RuleBasedPhaseLabeler.calibrate` (persisted per demonstration at
    ingestion) and classifies states one at a time, causally — only the
    current and past states are used. For trajectories longer than the
    median filter size, running :meth:`step` over a demonstration's states
    reproduces :meth:`RuleBasedPhaseLabeler.label` exactly: the hysteresis,
    the state machine, and the causal median window are the same
    computation. (``label()`` skips smoothing when ``T <= filter_size``; the
    step labeler always smooths, so the equality contract starts at
    ``T > filter_size``.)
    """

    _REQUIRED_KEYS = (
        "closed_level",
        "open_level",
        "mirror",
        "velocity_threshold",
        "min_duration",
        "filter_size",
        "eef_pos_slice",
        "gripper_qpos_slice",
        "num_phases",
    )

    def __init__(self, calibration: dict[str, Any]) -> None:
        missing = [key for key in self._REQUIRED_KEYS if key not in calibration]
        if missing:
            raise ValueError(f"Calibration artifact missing required key(s): {missing}")
        self.closed_level = float(calibration["closed_level"])
        self.open_level = float(calibration["open_level"])
        if not self.closed_level < self.open_level:
            raise ValueError(
                "Calibration artifact has closed_level >= open_level; "
                "the hysteresis bands are invalid."
            )
        self.mirror = bool(calibration["mirror"])
        bounds = calibration.get("mirror_bounds")
        if self.mirror:
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ValueError(
                    "Mirrored calibration artifacts must carry mirror_bounds=[lo, hi]."
                )
            self._mirror_lo = float(bounds[0])
            self._mirror_hi = float(bounds[1])
        self.velocity_threshold = float(calibration["velocity_threshold"])
        self.min_duration = int(calibration["min_duration"])
        self.filter_size = int(calibration["filter_size"])
        self.eef_pos_slice = tuple(int(v) for v in calibration["eef_pos_slice"])
        self.gripper_slice = tuple(int(v) for v in calibration["gripper_qpos_slice"])
        self.num_phases = int(calibration["num_phases"])
        if self.min_duration < 1 or self.filter_size < 1:
            raise ValueError("Calibration artifact has invalid duration/filter parameters.")
        self.reset()

    def reset(self) -> None:
        """Reset all state-machine state (start of a new episode)."""
        self._phase = 0
        self._entered = 0
        self._t = 0
        self._closed_prev = False
        self._prev_eef: np.ndarray | None = None
        self._history: deque[int] = deque()

    def step(self, state: np.ndarray) -> int:
        """Classify one raw (unnormalized) state row; causal in step order.

        ``state`` may be a single row ``(D,)`` or a ``(1, D)`` batch.
        """
        state = np.asarray(state, dtype=np.float32)
        if state.ndim == 2:
            if state.shape[0] != 1:
                raise ValueError(f"step() takes one state row, got {state.shape}")
            state = state[0]
        if state.ndim != 1:
            raise ValueError(f"Expected a state row (D,), got {state.shape}")
        e0, e1 = self.eef_pos_slice
        g0, g1 = self.gripper_slice
        if not (0 <= e0 < e1 <= state.shape[0] and 0 <= g0 < g1 <= state.shape[0]):
            raise ValueError(
                f"Phase slices eef={self.eef_pos_slice}, gripper={self.gripper_slice} "
                f"are invalid for state_dim={state.shape[0]}"
            )

        eef = state[e0:e1]
        aperture = float(np.max(np.abs(state[g0:g1])))
        if self._prev_eef is None:
            speed = 0.0
        else:
            speed = float(np.linalg.norm(eef - self._prev_eef))
        self._prev_eef = eef
        if self.mirror:
            aperture = (self._mirror_lo + self._mirror_hi) - aperture

        if aperture < self.closed_level:
            closed = True
        elif aperture > self.open_level:
            closed = False
        else:
            closed = self._closed_prev
        grasp = closed and not self._closed_prev
        release = (not closed) and self._closed_prev
        self._closed_prev = closed

        t = self._t
        held = t - self._entered >= self.min_duration
        if grasp:
            self._phase, self._entered = 2, t
        elif release:
            self._phase, self._entered = 4, t
        elif (
            self._phase == 0
            and t >= self.min_duration
            and not closed
            and speed < self.velocity_threshold
        ):
            self._phase, self._entered = 1, t
        elif self._phase == 2 and held and closed and speed > self.velocity_threshold:
            self._phase, self._entered = 3, t
        elif self._phase == 4 and held and not closed and speed > self.velocity_threshold:
            self._phase, self._entered = 5, t
        elif self._phase == 5 and speed < self.velocity_threshold:
            self._phase, self._entered = 0, t
        self._t += 1

        self._history.append(self._phase)
        if len(self._history) > self.filter_size:
            self._history.popleft()
        smoothed = np.median(np.asarray(self._history, dtype=np.float32))
        return int(np.clip(int(smoothed), 0, self.num_phases - 1))

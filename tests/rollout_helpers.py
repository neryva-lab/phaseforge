"""Shared test fakes: a kinematic simulator and state adapter per task.

Enough physics for the controller/gates/runner tests: the end-effector
integrates toward its commanded delta target, the manipulated object is
grasped when the gripper is closed at the grasp height, and the object
then follows the end-effector. The success predicate and placement
geometry are task-specific:

* **Lift** -- object z > 0.84 above the table.
* **Can** -- object is in the receptacle bin (target xy near a fixed
  offset, z above table by 0.04).
* **Square** -- object is on the square peg (target xy near a fixed
  offset, z above table by 0.04).
* **ToolHang** -- object is hooked on the rack (target xy near a fixed
  offset, z above table by 0.04).
* **Transport** -- the fake remains single-arm and is only a controller unit
  test; it is not evidence that the real two-arm Transport oracle works.

The simulator is finite, deterministic, and fast.
"""

from __future__ import annotations

import numpy as np

from phaseforge.evaluations.envs.env_metadata import PinnedEnvMetadata
from phaseforge.evaluations.envs.robosuite_adapter import StateSpec
from phaseforge.evaluations.rollout.reset_bank import ResetBank, ResetCase

TABLE_HEIGHT = 0.8
SUCCESS_Z = TABLE_HEIGHT + 0.04
LIFT_GRASP_Z_OFFSET = 0.01
# Vertical clearance for placement tasks: OBJECT_HALF_SIZE (0.0215) +
# the 0.02 m clearance used by the kinematic fake's placement predicate.
PLACEMENT_GRASP_Z_OFFSET = 0.0415

#: Supported tasks for the kinematic fake simulator.
SUPPORTED_TASKS: tuple[str, ...] = ("Lift", "Can", "Square", "ToolHang", "Transport")


def lift_state_spec() -> StateSpec:
    return StateSpec(
        keys=(
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "object",
        ),
        dims=(3, 4, 2, 10),
    )


def state_from_parts(
    eef: np.ndarray,
    cube: np.ndarray,
    gripper_qpos: np.ndarray | None = None,
) -> np.ndarray:
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    if gripper_qpos is None:
        gripper_qpos = np.array([0.04, 0.04])
    rel = cube[:3] - eef
    object_part = np.concatenate([cube[:3], quat, rel])
    return np.concatenate([eef, quat, gripper_qpos, object_part]).astype(np.float32)


class FakeLiftSim:
    """Kinematic simulator for the five-task state-only protocol.

    Parameters
    ----------
    rng:
        Deterministic source of cube placement.
    grasp_z_window:
        Vertical alignment tolerance (metres) for the gripper to grasp
        the manipulated object. Tighter windows make the random-policy
        sanity gate fail (no random policy can grasp by chance).
    task:
        One of :data:`SUPPORTED_TASKS`. Drives the success predicate
        and the receptacle offset used in placement success.
    """

    def __init__(
        self,
        rng: np.random.Generator | None = None,
        *,
        grasp_z_window: float = 0.02,
        task: str = "Lift",
    ) -> None:
        if task not in SUPPORTED_TASKS:
            raise ValueError(f"task must be one of {SUPPORTED_TASKS}, got {task!r}")
        self.rng = rng or np.random.default_rng(0)
        self.grasp_z_window = grasp_z_window
        self.task = task
        self.grasp_z_offset = (
            LIFT_GRASP_Z_OFFSET if task == "Lift" else PLACEMENT_GRASP_Z_OFFSET
        )
        self._receptacle_offset = _receptacle_offset_for(task)
        self.reset()

    def reset(self) -> None:
        self.eef = np.array([0.0, 0.0, 1.00], dtype=np.float64)
        cube_x = float(self.rng.uniform(-0.15, 0.15))
        cube_y = float(self.rng.uniform(-0.15, 0.15))
        self.cube = np.array([cube_x, cube_y, TABLE_HEIGHT], dtype=np.float64)
        self.gripper_qpos = np.array([0.04, 0.04], dtype=np.float64)
        self.grasped = False
        self.t = 0

    @property
    def state(self) -> np.ndarray:
        return state_from_parts(self.eef, self.cube, self.gripper_qpos)

    def step(self, action: np.ndarray) -> None:
        """Integrate a normalized OSC delta; gripper +1 closes, -1 opens."""
        pos = np.asarray(action[0:3], dtype=np.float64) * 0.05
        self.eef = np.clip(self.eef + pos, [-0.6, -0.6, 0.76], [0.6, 0.6, 1.25])
        gripper = float(action[6]) if len(action) > 6 else -1.0
        # Close (gripper_action = +1) decreases the finger gap (qpos -> 0),
        # matching the robosuite PandaGripper convention.
        self.gripper_qpos = np.clip(self.gripper_qpos - gripper * 0.02, 0.0, 0.04)
        if (
            not self.grasped
            and gripper >= 0.9
            and abs(self.eef[0] - self.cube[0]) < 0.04
            and abs(self.eef[1] - self.cube[1]) < 0.04
            and abs(self.eef[2] - (self.cube[2] + self.grasp_z_offset))
            < self.grasp_z_window
        ):
            self.grasped = True
        elif self.grasped and gripper <= -0.5:
            self.grasped = False
        if self.grasped:
            self.cube = self.eef - np.array([0.0, 0.0, self.grasp_z_offset])
        self.t += 1

    @property
    def success(self) -> bool:
        """Task-specific success predicate mirrored from ``env._check_success``.

        * Lift: cube z above table + 0.04.
        * Can/Square/ToolHang/Transport: cube is in the receptacle (xy
          alignment with the per-task offset) AND lifted above the table.
        """
        if self.task == "Lift":
            return bool(self.cube[2] > SUCCESS_Z)
        target_x, target_y = self._receptacle_offset
        xy_ok = abs(self.cube[0] - target_x) < 0.05 and abs(self.cube[1] - target_y) < 0.05
        return xy_ok and bool(self.cube[2] > SUCCESS_Z)

    def _check_success(self) -> bool:
        """Mirror of the robosuite native predicate (``_check_success``)."""
        return self.success

    @property
    def sim(self) -> FakeLiftSim:
        """Mirror of the real adapter's ``env.sim`` (flat state accessor)."""
        return self

    def get_state(self) -> np.ndarray:
        """Flat 7-dim 'world' state: ``[time, eef(3), cube(3)]``."""
        return np.array([0.0, *self.eef, *self.cube], dtype=np.float64)


def _receptacle_offset_for(task: str) -> tuple[float, float]:
    """Per-task absolute receptacle xy used by the kinematic fake's success predicate."""
    return {
        "Can": (0.15, 0.15),
        "Square": (-0.12, -0.08),
        "ToolHang": (0.20, -0.05),
        "Transport": (-0.18, 0.12),
        "Lift": (0.0, 0.0),
    }[task]


class FakeAdapter:
    """Minimal adapter implementing the runner/gates interface."""

    def __init__(
        self,
        sim: FakeLiftSim | None = None,
        *,
        action_dim: int = 7,
        horizon: int = 500,
        state_spec: StateSpec | None = None,
        fail_step_with: Exception | None = None,
        fail_reset_with: Exception | None = None,
        fail_check_success_with: Exception | None = None,
        obs_key_override: str | None = None,
        task: str = "Lift",
    ) -> None:
        self.sim = sim or FakeLiftSim(task=task)
        self.action_dim = action_dim
        self._horizon = horizon
        self.state_spec = state_spec or lift_state_spec()
        self.fail_step_with = fail_step_with
        self.fail_reset_with = fail_reset_with
        self.fail_check_success_with = fail_check_success_with
        self.obs_key_override = obs_key_override
        self.task = task
        self.step_calls = 0
        self.reset_calls = 0

    @property
    def horizon(self) -> int:
        return self._horizon

    @property
    def env(self) -> FakeLiftSim:
        return self.sim

    def validate_action(self, action, *, tolerance: float = 1e-4) -> np.ndarray:
        from phaseforge.evaluations.envs.errors import PolicyInvalidActionError

        arr = np.asarray(action)
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr.reshape(-1)
        if arr.shape != (self.action_dim,):
            raise PolicyInvalidActionError(
                f"Action has shape {arr.shape}, expected ({self.action_dim},)"
            )
        if not np.isfinite(arr).all():
            raise PolicyInvalidActionError("Action contains non-finite values")
        if float(arr.min()) < -1.0 - tolerance or float(arr.max()) > 1.0 + tolerance:
            raise PolicyInvalidActionError("Action outside the declared range")
        return arr.astype(np.float64)

    def reset_to(self, states, *, xml=None, ep_meta=None) -> np.ndarray:
        if self.fail_reset_with is not None:
            raise self.fail_reset_with
        self.sim.reset()
        if states is not None and states.size >= 12:
            self.sim.eef = np.asarray(states[:3], dtype=np.float64)
            self.sim.cube = np.asarray(states[9:12], dtype=np.float64)
        elif states is not None and states.size >= 7:
            self.sim.eef = np.asarray(states[1:4], dtype=np.float64)
            self.sim.cube = np.asarray(states[4:7], dtype=np.float64)
        self.reset_calls += 1
        return self.extract_state(self.sim.state)

    def step(self, action) -> tuple[np.ndarray, bool, bool, dict]:
        # Mirror RobosuiteStateAdapter.step: validate the action BEFORE the
        # simulator sees it. Policy-invalid actions are surface errors, never
        # simulator errors; the runner classifies them as policy failures.
        self.validate_action(action)
        if self.fail_step_with is not None:
            raise self.fail_step_with
        self.sim.step(action)
        self.step_calls += 1
        return self.extract_state(self.sim.state), False, self.sim.success, {}

    def extract_state(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(obs, dtype=np.float32)

    def check_success(self) -> bool:
        if self.fail_check_success_with is not None:
            raise self.fail_check_success_with
        return self.sim.success

    def close(self) -> None:
        pass


def make_meta(
    env_name: str = "Lift", version: str = "1.5.1", horizon: int = 500
) -> PinnedEnvMetadata:
    return PinnedEnvMetadata(
        env_name=env_name,
        env_version=version,
        env_type="robosuite",
        env_kwargs={"robots": "Panda", "horizon": horizon},
    )


def make_bank(num_cases: int = 3, *, task: str = "Lift") -> ResetBank:
    """Deterministic bank: cube on the table, eef above the approach threshold
    so the controller traverses APPROACH→DESCEND→GRASP→LIFT in order (case 0
    uses an unreachable eef z marker for policy-failure tests)."""
    cases: list[ResetCase] = []
    for i in range(num_cases):
        rng = np.random.default_rng(i)
        states = rng.uniform(0, 1, 19).astype(np.float32)
        x, y = float(rng.uniform(-0.1, 0.1)), float(rng.uniform(-0.1, 0.1))
        # case 0: unreachable eef z marker for policy-failure tests.
        states[2] = 5.0 if i == 0 else 0.95
        states[0:3] = [x, y, states[2]]
        states[9:12] = [x, y, 0.8]
        cases.append(ResetCase(index=i, states=states))
    return ResetBank(
        task=task,
        bank_id="testbank",
        seed=2026,
        num_cases=num_cases,
        env_canonical=make_meta().canonical_json(),
        robosuite_version="1.5.1",
        git_commit="",
        generated_at="2026-01-01T00:00:00Z",
        cases=cases,
    )


def make_sim(task: str = "Lift", **kwargs) -> FakeLiftSim:
    """Factory for a kinematic simulator configured for ``task``."""
    return FakeLiftSim(task=task, **kwargs)

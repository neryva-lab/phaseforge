"""Unit tests for the adaptive rule-based phase labeler.

The core regression: with fixed absolute thresholds (closed < 0.02,
open > 0.04) the robomimic Lift PH panda gripper signal (finger qpos in
[-0.02, 0.02], closed at the positive end) degenerated to only phases
2/3 — grasp fired at t=0 and release never fired. The labeler must now
populate all six phases for every common gripper convention.
"""

import numpy as np
import pytest

from phaseforge.data.robomimic.phase_labeler import (
    CausalPhaseStepLabeler,
    RuleBasedPhaseLabeler,
)


def _build_demo(
    open_aperture: float,
    closed_aperture: float,
    segments,
    antisymmetric: bool = False,
) -> dict:
    """Build a 9-dim state (eef_pos(3) + quat(4) + gripper_qpos(2)) demo.

    ``segments`` is a list of (num_samples, aperture, eef_speed) tuples; the
    eef position accumulates the per-sample speed so the velocity threshold
    is exercised. With ``antisymmetric`` the two finger qpos are +a/-a (the
    panda convention), otherwise both fingers read ``aperture``.
    """
    states = []
    pos = 0.0
    for n, aperture, speed in segments:
        for _ in range(n):
            pos += speed
            states.append((pos, aperture))
    T = len(states)
    state = np.zeros((T, 9), dtype=np.float32)
    for t, (p, aperture) in enumerate(states):
        state[t, 0] = p
        if antisymmetric:
            state[t, 7] = aperture
            state[t, 8] = -aperture
        else:
            state[t, 7:9] = aperture
    return {"state": state}


CANONICAL_SEGMENTS = [
    (60, None, 0.02),  # approach
    (20, None, 0.0),  # pre-grasp hold (open, stationary)
    (10, None, 0.002),  # grasp close
    (15, None, 0.0),  # grip hold
    (40, None, 0.03),  # transport
    (20, None, 0.003),  # place
    (10, None, 0.02),  # release + retract start
    (20, None, 0.02),  # retract
    (15, None, 0.0),  # settle
]


def _canonical(open_aperture: float, closed_aperture: float) -> dict:
    segments = [
        (n, closed_aperture if i in (2, 3, 4, 5) else open_aperture, speed)
        for i, (n, _unused, speed) in enumerate(CANONICAL_SEGMENTS)
    ]
    return _build_demo(open_aperture, closed_aperture, segments)


def _canonical_panda(open_aperture: float, closed_aperture: float) -> dict:
    segments = [
        (n, closed_aperture if i in (2, 3, 4, 5) else open_aperture, speed)
        for i, (n, _unused, speed) in enumerate(CANONICAL_SEGMENTS)
    ]
    return _build_demo(open_aperture, closed_aperture, segments, antisymmetric=True)


def _first_occurrence(phases: np.ndarray) -> dict[int, int]:
    first: dict[int, int] = {}
    for t, p in enumerate(phases):
        if p not in first:
            first[int(p)] = t
    return first


@pytest.mark.parametrize(
    ("open_aperture", "closed_aperture"),
    [
        (0.04, 0.0),  # magnitude convention, closed at the low end
        (0.0, 0.04),  # magnitude convention, closed at the high end
    ],
)
def test_all_six_phases_populated(open_aperture, closed_aperture) -> None:
    """Regression: fixed thresholds degenerated to {2, 3} for panda qpos."""
    labeler = RuleBasedPhaseLabeler()
    phases = labeler.label(_canonical(open_aperture, closed_aperture))

    assert set(phases.tolist()) == {0, 1, 2, 3, 4, 5}


@pytest.mark.parametrize(
    ("open_aperture", "closed_aperture"),
    [
        (0.04, 0.0),
        (0.0, 0.04),
    ],
)
def test_phase_order_is_approach_to_retract(open_aperture, closed_aperture) -> None:
    labeler = RuleBasedPhaseLabeler()
    phases = labeler.label(_canonical(open_aperture, closed_aperture))
    first = _first_occurrence(phases)

    assert first[0] == 0
    for p in range(5):
        assert first[p] < first[p + 1], f"phase {p} first occurs after phase {p + 1}"


@pytest.mark.parametrize(
    ("open_aperture", "closed_aperture"),
    [
        (0.04, 0.0),
        (0.0, 0.04),
    ],
)
def test_post_grasp_phases_dominate_approach(open_aperture, closed_aperture) -> None:
    labeler = RuleBasedPhaseLabeler()
    phases = labeler.label(_canonical(open_aperture, closed_aperture))
    counts = np.bincount(phases, minlength=6)

    assert counts[2:].sum() > counts[0]


def test_panda_antisymmetric_fingers_all_six_phases() -> None:
    """Regression: the mean of antisymmetric finger qpos (panda: open =
    [+0.0208, -0.0208], closed = [+0.04, -0.04]) is ~constant zero, so the
    aperture must be the finger excursion magnitude; all six phases must
    populate (the cloud failure produced only phases {2, 3})."""
    labeler = RuleBasedPhaseLabeler()
    phases = labeler.label(_canonical_panda(open_aperture=0.020833, closed_aperture=0.04))

    assert set(phases.tolist()) == {0, 1, 2, 3, 4, 5}


def test_panda_antisymmetric_fingers_phase_order() -> None:
    labeler = RuleBasedPhaseLabeler()
    phases = labeler.label(_canonical_panda(open_aperture=0.020833, closed_aperture=0.04))
    first = _first_occurrence(phases)

    assert first[0] == 0
    for p in range(5):
        assert first[p] < first[p + 1]


def test_constant_aperture_falls_back_to_absolute_thresholds() -> None:
    labeler = RuleBasedPhaseLabeler()
    segments = [(n, 0.04, speed) for n, _unused, speed in CANONICAL_SEGMENTS]
    phases = labeler.label(_build_demo(0.04, 0.04, segments))

    assert set(phases.tolist()).issubset({0, 1})


def test_short_trajectory_no_crash() -> None:
    labeler = RuleBasedPhaseLabeler()
    state = np.zeros((3, 9), dtype=np.float32)
    state[:, 7:9] = 0.04
    phases = labeler.label({"state": state})
    assert phases.shape == (3,)
    assert set(phases.tolist()).issubset({0, 1})


def test_empty_trajectory() -> None:
    labeler = RuleBasedPhaseLabeler()
    phases = labeler.label({"state": np.zeros((0, 9), dtype=np.float32)})
    assert phases.shape == (0,)


def test_invalid_slices_raise() -> None:
    labeler = RuleBasedPhaseLabeler(eef_pos_slice=(0, 20))
    with pytest.raises(ValueError, match="invalid for state_dim"):
        labeler.label({"state": np.zeros((10, 9), dtype=np.float32)})


def test_num_phases_vocabulary_guard() -> None:
    with pytest.raises(ValueError, match="exactly 6 phases"):
        RuleBasedPhaseLabeler(num_phases=4)


def test_threshold_ordering_guard() -> None:
    with pytest.raises(ValueError, match="below gripper_open_threshold"):
        RuleBasedPhaseLabeler(gripper_closed_threshold=0.04, gripper_open_threshold=0.02)


# ---------------------------------------------------------------------------
# Calibration artifacts + causal step labeler
# ---------------------------------------------------------------------------


class TestCalibrationArtifacts:
    def test_calibrate_artifact_has_all_keys(self) -> None:
        labeler = RuleBasedPhaseLabeler()
        artifact = labeler.calibrate(_canonical(0.04, 0.0))
        for key in (
            "closed_level",
            "open_level",
            "mirror",
            "mirror_bounds",
            "velocity_threshold",
            "min_duration",
            "filter_size",
            "eef_pos_slice",
            "gripper_qpos_slice",
            "num_phases",
        ):
            assert key in artifact
        assert artifact["closed_level"] < artifact["open_level"]
        assert artifact["num_phases"] == 6

    def test_calibrate_degenerate_falls_back(self) -> None:
        labeler = RuleBasedPhaseLabeler()
        segments = [(n, 0.04, speed) for n, _unused, speed in CANONICAL_SEGMENTS]
        artifact = labeler.calibrate(_build_demo(0.04, 0.04, segments))
        assert artifact["mirror"] is False
        assert artifact["mirror_bounds"] is None
        assert artifact["closed_level"] == pytest.approx(labeler.closed_threshold)
        assert artifact["open_level"] == pytest.approx(labeler.open_threshold)

    def test_mirrored_convention_marks_mirror_and_bounds(self) -> None:
        labeler = RuleBasedPhaseLabeler()
        artifact = labeler.calibrate(_canonical(0.0, 0.04))
        assert artifact["mirror"] is True
        assert len(artifact["mirror_bounds"]) == 2
        assert artifact["mirror_bounds"][0] < artifact["mirror_bounds"][1]

    @pytest.mark.parametrize(
        ("open_aperture", "closed_aperture", "antisymmetric"),
        [
            (0.04, 0.0, False),
            (0.0, 0.04, False),
            (0.020833, 0.04, True),
        ],
    )
    def test_causal_step_labeler_matches_label(
        self, open_aperture, closed_aperture, antisymmetric
    ) -> None:
        labeler = RuleBasedPhaseLabeler()
        if antisymmetric:
            demo = _canonical_panda(open_aperture, closed_aperture)
        else:
            demo = _canonical(open_aperture, closed_aperture)
        expected = labeler.label(demo)
        assert len(expected) > labeler.filter_size

        artifact = labeler.calibrate(demo)
        step_labeler = CausalPhaseStepLabeler(artifact)
        actual = np.asarray(
            [step_labeler.step(row) for row in demo["state"]], dtype=np.int64
        )
        assert (actual == expected).all(), (
            f"step labels diverge at indices: {np.flatnonzero(actual != expected).tolist()}"
        )

    def test_causal_step_labeler_reset_restarts_state(self) -> None:
        labeler = RuleBasedPhaseLabeler()
        demo = _canonical(0.04, 0.0)
        artifact = labeler.calibrate(demo)
        step_labeler = CausalPhaseStepLabeler(artifact)
        first = np.asarray([step_labeler.step(row) for row in demo["state"]], dtype=np.int64)
        step_labeler.reset()
        second = np.asarray([step_labeler.step(row) for row in demo["state"]], dtype=np.int64)
        assert (first == second).all()

    def test_causal_step_labeler_missing_keys_raise(self) -> None:
        with pytest.raises(ValueError, match="missing required key"):
            CausalPhaseStepLabeler({"closed_level": 0.0})

    def test_causal_step_labeler_invalid_levels_raise(self) -> None:
        artifact = {
            "closed_level": 0.04,
            "open_level": 0.02,
            "mirror": False,
            "mirror_bounds": None,
            "velocity_threshold": 0.01,
            "min_duration": 5,
            "filter_size": 7,
            "eef_pos_slice": [0, 3],
            "gripper_qpos_slice": [7, 9],
            "num_phases": 6,
        }
        with pytest.raises(ValueError, match="closed_level >= open_level"):
            CausalPhaseStepLabeler(artifact)

    def test_causal_step_labeler_mirror_requires_bounds(self) -> None:
        artifact = {
            "closed_level": 0.02,
            "open_level": 0.04,
            "mirror": True,
            "mirror_bounds": None,
            "velocity_threshold": 0.01,
            "min_duration": 5,
            "filter_size": 7,
            "eef_pos_slice": [0, 3],
            "gripper_qpos_slice": [7, 9],
            "num_phases": 6,
        }
        with pytest.raises(ValueError, match="mirror_bounds"):
            CausalPhaseStepLabeler(artifact)

    def test_causal_step_labeler_batch_row(self) -> None:
        labeler = RuleBasedPhaseLabeler()
        demo = _canonical(0.04, 0.0)
        artifact = labeler.calibrate(demo)
        step_labeler = CausalPhaseStepLabeler(artifact)
        single = step_labeler.step(demo["state"][0])
        step_labeler.reset()
        batched = step_labeler.step(demo["state"][:1])
        assert single == batched

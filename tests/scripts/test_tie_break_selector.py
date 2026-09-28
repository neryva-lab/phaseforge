"""CPU-only tests for the stage-1 tie-break checkpoint re-selection logic."""

from __future__ import annotations

import pytest

from scripts.analysis.tie_break_selector import select_tie_break_epoch


def _rows(epochs: list[int], actions: list[float], phases: list[float]) -> list[dict]:
    return [
        {"epoch": e, "val/loss_action": a, "val/loss_phase": p}
        for e, a, p in zip(epochs, actions, phases)
    ]


def test_no_plateau_keeps_monitor_epoch() -> None:
    rows = _rows(
        [0, 1, 2, 3],
        [3.0, 2.0, 1.0, 0.5],  # strictly improving
        [0.9, 1.0, 1.1, 1.2],
    )
    decision = select_tie_break_epoch(rows, 0.02)
    assert decision["monitor_epoch"] == 3
    assert decision["tie_break_epoch"] == 3
    assert decision["changed"] is False


def test_plateau_picks_lowest_phase_loss() -> None:
    rows = _rows(
        [0, 1, 2, 3, 4],
        [1.00, 0.99, 1.00, 0.98, 1.01],  # plateau within 2% of 0.98
        [1.5, 0.9, 1.1, 1.3, 2.5],  # epoch 1 has the best phase loss
    )
    decision = select_tie_break_epoch(rows, 0.02)
    assert decision["monitor_epoch"] == 3
    assert decision["tie_break_epoch"] == 1
    assert decision["changed"] is True
    assert decision["plateau_size"] >= 2


def test_plateau_includes_only_epochs_within_tolerance() -> None:
    rows = _rows(
        [0, 1, 2, 3],
        [0.980, 0.995, 1.030, 0.990],
        [2.0, 0.1, 1.0, 2.2],
    )
    # 2% of 0.980 = 0.9996: epoch 1 (0.995) is inside, epoch 2 (1.030) is not.
    decision = select_tie_break_epoch(rows, 0.02)
    assert decision["plateau_epochs"] == [0, 1, 3]
    assert decision["tie_break_epoch"] == 1


def test_missing_phase_column_refuses() -> None:
    rows = [{"epoch": 0, "val/loss_action": 1.0}]
    with pytest.raises(SystemExit, match="val/loss_phase"):
        select_tie_break_epoch(rows, 0.02)


def test_non_finite_action_refuses() -> None:
    rows = _rows([0, 1], [1.0, float("nan")], [0.5, 0.6])
    with pytest.raises(SystemExit, match="val/loss_action"):
        select_tie_break_epoch(rows, 0.02)

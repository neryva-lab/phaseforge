"""CPU-only tests for the offline evaluator and its decisive action-MSE metric.

``eval/action_mse`` is the fast diagnostic that distinguishes "the model
cannot reproduce the demonstration actions" (rollout success is
expected) from "the eval path is broken" — and it is logged immediately so
an interrupted run (e.g. inside the slower routing metrics) still shows the
decisive number.
"""

from __future__ import annotations

import logging
import math

import pytest
import torch
from omegaconf import OmegaConf

from phaseforge.evaluations.metrics.task_metrics import (
    action_mse,
    phase_error_by_transition_distance,
)
from phaseforge.evaluations.runners.offline_evaluator import OfflineEvaluator
from phaseforge.models.base import BaseManipulationModel, ModelOutput


class FakeModel(BaseManipulationModel):
    """Deterministic policy: action = first 7 state dims doubled."""

    def forward(self, batch: dict[str, torch.Tensor]) -> ModelOutput:
        return ModelOutput(action_pred=batch["state"][:, :7] * 2.0)

    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        return state[:, :7] * 2.0

    def num_parameters(self) -> int:
        return 0


class PhaseHeadModel(BaseManipulationModel):
    """Model with a phase head whose logits are the phase labels one-hot."""

    def forward(self, batch: dict[str, torch.Tensor]) -> ModelOutput:
        n_classes = batch["phase"].max().item() + 1
        one_hot = torch.zeros(batch["phase"].numel(), n_classes)
        one_hot[torch.arange(batch["phase"].numel()), batch["phase"]] = 10.0
        return ModelOutput(
            action_pred=torch.zeros(batch["state"].shape[0], 7),
            phase_logits=one_hot.reshape(*batch["phase"].shape, n_classes),
        )

    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        return torch.zeros(state.shape[0], 7)

    def num_parameters(self) -> int:
        return 0


class NoisyPhaseHeadModel(PhaseHeadModel):
    """Phase head that is wrong exactly at phase boundaries (distance 0)."""

    def forward(self, batch: dict[str, torch.Tensor]) -> ModelOutput:
        out = super().forward(batch)
        logits = out.phase_logits.clone()
        labels = batch["phase"].flatten()
        wrong = torch.zeros_like(labels, dtype=torch.bool)
        if labels.numel() > 1:
            wrong[1:] = labels[1:] != labels[:-1]
        wrong_idx = wrong.nonzero(as_tuple=True)[0]
        if wrong_idx.numel():
            flip = (labels[wrong_idx] + 1) % logits.shape[-1]
            logits = logits.reshape(-1, logits.shape[-1])
            logits[wrong_idx, labels[wrong_idx]] = 0.0
            logits[wrong_idx, flip] = 10.0
            logits = logits.reshape(*batch["phase"].shape, -1)
        return ModelOutput(
            action_pred=out.action_pred, phase_logits=logits
        )


def _make_cfg() -> OmegaConf:
    return OmegaConf.create(
        {
            "project": {"device": "cpu"},
            "eval": {
                "task": {
                    "action_l2_threshold_rate": {
                        "enabled": True,
                        "l2_threshold": 0.05,
                    },
                    "boundary_action_smoothness": {"enabled": False},
                },
            },
        }
    )


# ---------------------------------------------------------------------------
# action_mse
# ---------------------------------------------------------------------------


def test_action_mse_exact_match() -> None:
    preds = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert action_mse(preds, preds) == 0.0


def test_action_mse_known_value() -> None:
    preds = torch.zeros(2, 2)
    targets = torch.tensor([[1.0, 3.0], [1.0, 3.0]])
    # (1^2 + 3^2 + 1^2 + 3^2) / 4 = 20 / 4 = 5
    assert action_mse(preds, targets) == pytest.approx(5.0)


def test_action_mse_respects_padding_mask() -> None:
    preds = torch.zeros(2, 3, 2)  # (B, T, A)
    targets = torch.ones(2, 3, 2)
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])  # (B, T): 3 valid entries
    assert action_mse(preds, targets, mask) == pytest.approx(1.0)


def test_action_mse_empty_returns_nan() -> None:
    preds = torch.zeros(0, 2)
    targets = torch.zeros(0, 2)
    assert torch.isnan(torch.tensor(action_mse(preds, targets)))


# ---------------------------------------------------------------------------
# OfflineEvaluator.run()
# ---------------------------------------------------------------------------


def test_run_reports_and_logs_action_mse_immediately(caplog) -> None:
    cfg = _make_cfg()
    states = torch.randn(8, 23)
    actions = torch.randn(8, 7)
    dataloader = [
        {"state": states[:4], "action": actions[:4], "phase": torch.zeros(4, dtype=torch.long)},
        {"state": states[4:], "action": actions[4:], "phase": torch.zeros(4, dtype=torch.long)},
    ]

    evaluator = OfflineEvaluator(cfg=cfg, model=FakeModel(), dataloader=dataloader)
    expected_mse = float(((states[:, :7] * 2.0 - actions) ** 2).mean())

    with caplog.at_level(logging.INFO, logger="phaseforge.evaluations.runners.offline_evaluator"):
        results = evaluator.run()

    assert results["eval/action_mse"] == pytest.approx(expected_mse)
    assert "eval/action_mse" in caplog.text
    assert "eval/action_l2_threshold_rate" in caplog.text
    assert "eval/action_l2_threshold_rate" in results
    assert "eval/success_rate" not in results
    assert results["eval/action_mse"] > 0.0  # the fake model is wrong on purpose


# ---------------------------------------------------------------------------
# phase_error_by_transition_distance
# ---------------------------------------------------------------------------


def _trajectory_with_boundaries(
    segments: list[int], t_per_segment: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """One trajectory of repeated phase ids; returns (labels, n_classes)."""
    labels = []
    for i, seg in enumerate(segments):
        labels.extend([seg] * t_per_segment)
    return torch.tensor(labels, dtype=torch.long), max(segments) + 1


def _perfect_logits(labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    logits = torch.full((labels.numel(), n_classes), -5.0)
    logits[torch.arange(labels.numel()), labels] = 5.0
    return logits


def test_phase_error_perfect_predictions_all_zero() -> None:
    labels, n_classes = _trajectory_with_boundaries([0, 1, 2], 10)
    out = phase_error_by_transition_distance(_perfect_logits(labels, n_classes), labels)
    for key, value in out.items():
        if key.startswith("dist_") and out[f"n_{key}"] > 0:
            assert value == 0.0
    assert out["n_boundaries"] == 2.0
    assert out["n_samples"] == 30.0
    assert out["any_boundary"] == 1.0


def test_phase_error_errors_cluster_at_boundary() -> None:
    labels, n_classes = _trajectory_with_boundaries([0, 1, 2], 10)
    logits = _perfect_logits(labels, n_classes)
    # Corrupt exactly the transition positions (first step of each segment
    # after the first): distances 0 must show error 1.0.
    for t in (10, 20):
        logits[t, labels[t]] = -5.0
        logits[t, (labels[t] + 1) % n_classes] = 5.0
    out = phase_error_by_transition_distance(logits, labels)
    assert out["dist_0_1"] == pytest.approx(1.0)
    assert out["dist_1_3"] == 0.0
    assert out["dist_3_6"] == 0.0
    assert out["dist_6_11"] == 0.0
    assert math.isnan(out["dist_11_plus"])  # max distance 10 -> bucket empty


def test_phase_error_uniform_errors_spread_across_buckets() -> None:
    labels, n_classes = _trajectory_with_boundaries([0, 1, 2], 10)
    logits = _perfect_logits(labels, n_classes)
    # Wrong everywhere (argmax shifted to another class): every non-empty
    # bucket must show error 1.0.
    logits = torch.roll(logits, shifts=1, dims=1)
    out = phase_error_by_transition_distance(logits, labels)
    for key, value in out.items():
        if key.startswith("dist_") and out[f"n_{key}"] > 0:
            assert value == pytest.approx(1.0)


def test_phase_error_no_transitions_lands_in_final_bucket() -> None:
    labels, n_classes = _trajectory_with_boundaries([0], 10)
    logits = _perfect_logits(labels, n_classes)
    out = phase_error_by_transition_distance(logits, labels)
    assert out["dist_11_plus"] == 0.0
    assert out["n_boundaries"] == 0.0
    assert out["any_boundary"] == 0.0
    assert out["n_samples"] == 10.0


def test_phase_error_validates_inputs() -> None:
    labels, n_classes = _trajectory_with_boundaries([0, 1], 5)
    logits = _perfect_logits(labels, n_classes)
    with pytest.raises(ValueError, match="(T, C)"):
        phase_error_by_transition_distance(logits.unsqueeze(0), labels)
    with pytest.raises(ValueError, match="same timesteps"):
        phase_error_by_transition_distance(logits[:-1], labels)
    with pytest.raises(ValueError, match="empty"):
        phase_error_by_transition_distance(torch.zeros(0, n_classes), labels[:0])
    with pytest.raises(ValueError, match="non-finite"):
        bad = logits.clone()
        bad[0, 0] = float("nan")
        phase_error_by_transition_distance(bad, labels)
    with pytest.raises(ValueError, match="bucket_edges"):
        phase_error_by_transition_distance(logits, labels, bucket_edges=(3, 1))
    with pytest.raises(ValueError, match="bucket_edges"):
        phase_error_by_transition_distance(logits, labels, bucket_edges=(0, 2))


# ---------------------------------------------------------------------------
# OfflineEvaluator: phase_boundary_error integration
# ---------------------------------------------------------------------------


def _make_cfg_with_boundary_error(enabled: bool) -> OmegaConf:
    cfg = _make_cfg()
    cfg.eval.task.phase_boundary_error = {"enabled": enabled, "bucket_edges": [1, 3, 6, 11]}
    return cfg


def _trajectory_loader(segments: list[list[int]]) -> list[dict]:
    """Two batches from per-trajectory phase streams with identity keys."""
    batches: list[dict] = []
    for start in (0, 6):
        rows = []
        for tid, seg in enumerate(segments):
            for pos, phase in enumerate(seg[start : start + 6], start=start):
                rows.append(
                    {
                        "state": torch.randn(23),
                        "action": torch.randn(7),
                        "phase": torch.tensor(phase, dtype=torch.long),
                        "task_id": torch.tensor(0),
                        "trajectory_id": torch.tensor(tid),
                        "trajectory_position": torch.tensor(pos),
                    }
                )
        batches.append({k: torch.stack([r[k] for r in rows]) for k in rows[0]})
    return batches


def test_boundary_error_reported_when_enabled() -> None:
    # Trajectory 0: phases 0 x6 then 1 x6; trajectory 1: phases 2 x12.
    segments = [[0] * 6 + [1] * 6, [2] * 12]
    loader = _trajectory_loader(segments)
    evaluator = OfflineEvaluator(
        cfg=_make_cfg_with_boundary_error(True), model=PhaseHeadModel(), dataloader=loader
    )
    results = evaluator.run()
    assert results["eval/phase_err_dist_0_1"] == 0.0
    assert results["eval/phase_err_n_dist_0_1"] > 0.0
    assert results["eval/phase_err_any_boundary"] == 0.5  # traj 1 has none
    assert results["eval/phase_err_n_boundaries"] == 1.0


def test_boundary_error_respects_custom_bucket_edges() -> None:
    # Custom single edge: only distance-0 and distance-1+ buckets exist.
    cfg = _make_cfg()
    cfg.eval.task.phase_boundary_error = {"enabled": True, "bucket_edges": [1]}
    segments = [[0] * 6 + [1] * 6, [2] * 12]
    loader = _trajectory_loader(segments)
    evaluator = OfflineEvaluator(
        cfg=cfg, model=PhaseHeadModel(), dataloader=loader
    )
    results = evaluator.run()
    assert "eval/phase_err_dist_0_1" in results
    assert "eval/phase_err_dist_1_plus" in results
    assert not any("dist_1_3" in k for k in results)


def test_boundary_error_not_reported_when_disabled() -> None:
    segments = [[0] * 6 + [1] * 6, [2] * 12]
    loader = _trajectory_loader(segments)
    evaluator = OfflineEvaluator(
        cfg=_make_cfg_with_boundary_error(False), model=PhaseHeadModel(), dataloader=loader
    )
    results = evaluator.run()
    assert not any(k.startswith("eval/phase_err") for k in results)


def test_boundary_error_skipped_without_trajectory_ids() -> None:
    loader = [
        {
            "state": torch.randn(4, 23),
            "action": torch.randn(4, 7),
            "phase": torch.zeros(4, dtype=torch.long),
        },
        {
            "state": torch.randn(4, 23),
            "action": torch.randn(4, 7),
            "phase": torch.zeros(4, dtype=torch.long),
        },
    ]
    evaluator = OfflineEvaluator(
        cfg=_make_cfg_with_boundary_error(True), model=PhaseHeadModel(), dataloader=loader
    )
    results = evaluator.run()
    assert not any(k.startswith("eval/phase_err") for k in results)

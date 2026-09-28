"""Rollout protocol: frozen reset bank, simulator adapter, runners, gates."""

from phaseforge.evaluations.rollout.report import build_rollout_report
from phaseforge.evaluations.rollout.reset_bank import (
    ResetBank,
    ResetCase,
    bank_dir,
    compute_bank_id,
    generate_reset_bank,
)
from phaseforge.evaluations.rollout.runner import (
    RolloutEvaluator,
    RolloutOutcome,
    RolloutRunInvalid,
    resolve_pinned_metadata,
    run_rollout_evaluation,
)

__all__ = [
    "ResetBank",
    "ResetCase",
    "RolloutEvaluator",
    "RolloutOutcome",
    "RolloutRunInvalid",
    "bank_dir",
    "build_rollout_report",
    "compute_bank_id",
    "generate_reset_bank",
    "resolve_pinned_metadata",
    "run_rollout_evaluation",
]

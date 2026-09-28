"""Typed loaders for evaluation artifacts (eval_results.json, rollout_summary.json)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from studies.analysis.common import io as cio

_REQUIRED = (
    "eval/rollout/success_rate",
    "eval/rollout/successes",
    "eval/rollout/valid_episodes",
    "eval/rollout/wilson_ci95_low",
    "eval/rollout/wilson_ci95_high",
)


@dataclass(frozen=True)
class EvalResult:
    """One (task, method, seed) rollout evaluation."""

    path: Path
    success_rate: float
    successes: int
    valid_episodes: int
    wilson_low: float
    wilson_high: float
    policy_failures: int = 0
    invalid_attempts: int = 0
    horizon: int | None = None
    reset_bank: str | None = None
    action_mse: float | None = None
    # From rollout_summary.json (optional file; None when absent).
    router_mode: str | None = None
    checkpoint_sha256: str | None = None
    reset_seed: int | None = None
    failure_categories: dict[str, int] = field(default_factory=dict)

    @property
    def wilson_recomputed_disagrees(self) -> bool:  # cross-check hook
        from studies.analysis.stats.intervals import wilson_interval

        lo, hi = wilson_interval(self.successes, self.valid_episodes)
        return abs(lo - self.wilson_low) > 1e-6 or abs(hi - self.wilson_high) > 1e-6


def load_eval_result(run_dir: Path) -> EvalResult:
    data: dict[str, Any] = cio.read_json(run_dir / "eval_results.json")
    sr = data.get("eval/rollout/success_rate")
    if sr is None:
        return EvalResult(
            path=run_dir,
            success_rate=0.0,
            successes=0,
            valid_episodes=0,
            wilson_low=0.0,
            wilson_high=0.0,
            action_mse=data.get("eval/action_mse"),
            horizon=data.get("eval/rollout/horizon"),
            reset_bank=data.get("eval/rollout/reset_bank"),
        )

    missing = [k for k in _REQUIRED if k not in data]
    if missing:
        raise ValueError(f"{run_dir / 'eval_results.json'}: missing required keys {missing}")

    summary: dict[str, Any] = {}
    summary_path = run_dir / "rollout_summary.json"
    if summary_path.is_file():
        loaded = cio.read_json(summary_path)
        if isinstance(loaded, dict):
            summary = loaded

    def opt(key: str, source: dict[str, Any]) -> Any:
        return source.get(key)

    return EvalResult(
        path=run_dir,
        success_rate=float(data["eval/rollout/success_rate"]),
        successes=int(data["eval/rollout/successes"]),
        valid_episodes=int(data["eval/rollout/valid_episodes"]),
        wilson_low=float(data["eval/rollout/wilson_ci95_low"]),
        wilson_high=float(data["eval/rollout/wilson_ci95_high"]),
        policy_failures=int(data.get("eval/rollout/policy_failures", 0)),
        invalid_attempts=int(data.get("eval/rollout/invalid_attempts", 0)),
        horizon=opt("eval/rollout/horizon", data),
        reset_bank=opt("eval/rollout/reset_bank", data),
        action_mse=opt("eval/action_mse", data),
        router_mode=opt("router_mode", summary),
        checkpoint_sha256=opt("checkpoint_sha256", summary),
        reset_seed=opt("reset_seed", summary),
        failure_categories=dict(summary.get("failure_categories", {})),
    )

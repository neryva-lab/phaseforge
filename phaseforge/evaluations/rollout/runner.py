"""Rollout evaluation runner.

Drives one evaluation run over the frozen reset bank with the strict
metric: infrastructure failures invalidate the run; policy failures
(NaN/invalid actions, policy exceptions) are valid failed episodes with a
``failure_category``. Every episode is appended to ``episodes.jsonl`` via
the schema validator, and per-run summary statistics land in
``rollout_summary.json`` plus the results dict the CLI writes to
``eval_results.json``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from phaseforge.data.common.normalizer import FrozenNormalizer
from phaseforge.data.robomimic.phase_labeler import CausalPhaseStepLabeler
from phaseforge.evaluations.envs.env_metadata import (
    PinnedEnvMetadata,
    env_metadata_from_cache,
    verify_environment_parity,
)
from phaseforge.evaluations.envs.errors import (
    EnvParityError,
    InfrastructureError,
    PolicyInvalidActionError,
)
from phaseforge.evaluations.envs.robosuite_adapter import (
    RobosuiteStateAdapter,
    StateSpec,
)
from phaseforge.evaluations.rollout.reset_bank import (
    ResetBank,
    bank_dir,
    compute_bank_id,
    generate_reset_bank,
)
from phaseforge.evaluations.rollout.trace import LIP_SAMPLE_PERIOD, TraceWriter, to_jsonable
from phaseforge.outputs_writer.episodes import (
    append_episode_record,
    summarize_episodes,
    wilson_interval,
)

logger = logging.getLogger(__name__)

#: Failure categories emitted by the runner (frozen taxonomy).
FAILURE_TIMEOUT = "task_timeout"
FAILURE_POLICY_INVALID_ACTION = "policy_invalid_action"
FAILURE_POLICY_EXCEPTION = "policy_exception"

#: Rollout trace verbosity.
#: ``minimal`` = episode records + per-run summary (episodes.jsonl + rollout_summary.json).
#: ``full`` = 22-field per-step trace.jsonl written next to them.
TRACE_LEVELS: frozenset[str] = frozenset({"minimal", "full"})


class RolloutRunInvalid(RuntimeError):
    """Raised when the run contains infrastructure-failure episodes.

    The run is marked failed in the ledger; the episode rows remain on
    disk (append-only, audit-ready) and the run must be rerun.
    """


@dataclass
class RolloutOutcome:
    """Outcome of one episode, before it is turned into a record row."""

    valid: bool
    success: bool = False
    timed_out: bool = False
    steps: int = 0
    termination_reason: str | None = None
    failure_category: str | None = None
    exception: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class RolloutEvaluator:
    """Runs one policy over the reset bank."""

    def __init__(
        self,
        *,
        cfg: DictConfig,
        policy: Callable[[np.ndarray], np.ndarray] | None,
        adapter: RobosuiteStateAdapter,
        bank: ResetBank,
        normalizer: FrozenNormalizer | None,
        model: torch.nn.Module | None,
        output_dir: str | Path,
        run_id: str,
        model_name: str,
        training_seed: int,
        task: str,
        checkpoint_sha256: str,
        tag: str | None = None,
        horizon: int | None = None,
        action_tolerance: float = 1e-4,
        phase_labeler: CausalPhaseStepLabeler | None = None,
        router_mode: str = "learned",
        trace_level: str = "minimal",
        trace_every_n_steps: int = 1,
        reference_states: np.ndarray | None = None,
    ) -> None:
        self.cfg = cfg
        self.policy = policy
        self.adapter = adapter
        self.bank = bank
        self.normalizer = normalizer
        self.model = model
        self.output_dir = Path(output_dir)
        self.run_id = run_id
        self.model_name = model_name
        self.training_seed = int(training_seed)
        self.task = task
        self.checkpoint_sha256 = checkpoint_sha256
        self.tag = tag
        self.horizon = int(horizon or adapter.horizon)
        self.action_tolerance = float(action_tolerance)
        self.phase_labeler = phase_labeler
        self.router_mode = str(router_mode).lower()
        self.trace_level = str(trace_level).lower()
        if self.trace_level not in TRACE_LEVELS:
            raise EnvParityError(
                f"eval.episodes.trace_level={trace_level!r} is not one of "
                + ", ".join(sorted(TRACE_LEVELS))
                + "."
            )
        self.trace_every_n_steps = int(trace_every_n_steps)
        if self.trace_every_n_steps < 1:
            raise EnvParityError(
                f"eval.episodes.trace_every_n_steps must be >= 1, got {trace_every_n_steps!r}."
            )
        self.reference_states = (
            np.asarray(reference_states, dtype=np.float64)
            if reference_states is not None
            else None
        )
        self.num_phases = phase_labeler.num_phases if phase_labeler is not None else None
        # Active full-trace buffer, owned by _run_episode (one episode at a
        # time, sequential execution): _episode consumes it when present.
        # Always None outside an episode so direct _episode calls stay clean.
        self._active_trace_rows: list[dict[str, Any]] | None = None

        # These values are invariant for the whole rollout. Resolving them
        # once avoids a parameter walk, an eval-mode traversal, and repeated
        # normalizer device transfers at every simulator step.
        self._model_device: torch.device | None = None
        self._normalizer_mean: torch.Tensor | None = None
        self._normalizer_std: torch.Tensor | None = None
        if self.model is not None:
            self.model.eval()
            try:
                self._model_device = next(self.model.parameters()).device
            except (StopIteration, AttributeError):
                self._model_device = torch.device("cpu")
            if self.normalizer is not None:
                self._normalizer_mean = self.normalizer.mean.to(self._model_device)
                self._normalizer_std = self.normalizer.std.to(self._model_device)
                if hasattr(self.model, "set_normalizer_stats"):
                    self.model.set_normalizer_stats(self._normalizer_mean, self._normalizer_std)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Evaluate every bank case; returns the results dict for eval_results.json.

        Raises:
            RolloutRunInvalid: any episode was an infrastructure failure
                (invalid episode rows stay on disk for audit).
        """
        rows: list[dict[str, Any]] = []
        had_infrastructure_failure = False

        for case in self.bank.cases:
            outcome, row = self._run_episode(case)
            append_episode_record(self.output_dir, row)
            rows.append(row)
            if not outcome.valid:
                had_infrastructure_failure = True
            logger.info(
                "case %02d: valid=%s success=%s steps=%d reason=%s",
                case.index,
                outcome.valid,
                outcome.success,
                outcome.steps,
                outcome.termination_reason,
            )

        results = self._summarize(rows)
        self._write_run_summary(results, rows)

        if had_infrastructure_failure:
            raise RolloutRunInvalid(
                f"Rollout run {self.run_id} contains infrastructure-failure "
                "episodes (see episodes.jsonl). The run is invalid and must "
                "be rerun; no success rate is reported from it."
            )
        return results

    # ------------------------------------------------------------------
    # Per-episode execution
    # ------------------------------------------------------------------

    def _run_episode(self, case) -> tuple[RolloutOutcome, dict[str, Any]]:
        """Run one bank case; returns ``(outcome, schema-valid record row)``."""
        try:
            state = self.adapter.reset_to(case.states, xml=case.xml, ep_meta=case.ep_meta)
        except InfrastructureError as exc:
            return self._episode(case, RolloutOutcome(valid=False, exception=str(exc)))
        except Exception as exc:  # noqa: BLE001 — simulator misuse is infra
            return self._episode(case, RolloutOutcome(valid=False, exception=str(exc)))

        # Both externally supplied policies and stateful model objects may
        # carry episode history. Reset each independently; swallowing a
        # broken reset would leak one episode into the next and invalidate a
        # recurrent baseline, so surface it as a policy failure.
        reset_targets = []
        if self.policy is not None:
            reset_targets.append(("policy", self.policy))
        if self.model is not None and self.model is not self.policy:
            reset_targets.append(("model", self.model))
        for label, target in reset_targets:
            reset = getattr(target, "reset", None)
            if callable(reset):
                try:
                    reset()
                except Exception as exc:  # noqa: BLE001 — policy state bug
                    return self._episode(
                        case,
                        RolloutOutcome(
                            valid=True,
                            steps=0,
                            termination_reason=FAILURE_POLICY_EXCEPTION,
                            failure_category=FAILURE_POLICY_EXCEPTION,
                            exception=f"{label}.reset failed: {type(exc).__name__}: {exc}",
                        ),
                    )

        steps = 0
        if self.phase_labeler is not None:
            self.phase_labeler.reset()
        max_phase: int | None = None
        # Full per-step tracing: rows cover simulator-returned
        # steps only (state -> action pairs); failed/policy-rejected steps
        # are captured by the episode outcome, not by a trace row.
        trace_rows: list[dict[str, Any]] | None = (
            [] if self.trace_level == "full" else None
        )
        self._active_trace_rows = trace_rows
        prev_params: tuple[torch.Tensor, torch.Tensor] | None = None
        episode_seq = int(case.index)
        while steps < self.horizon:
            try:
                if self.policy is not None:
                    action = self.policy(state)
                else:
                    action = self._policy_action(state)
            except PolicyInvalidActionError as exc:
                outcome = RolloutOutcome(
                    valid=True,
                    steps=steps,
                    termination_reason=FAILURE_POLICY_INVALID_ACTION,
                    failure_category=FAILURE_POLICY_INVALID_ACTION,
                    exception=str(exc),
                )
                return self._episode(case, outcome, max_phase=max_phase)
            except Exception as exc:  # noqa: BLE001 — model raised
                outcome = RolloutOutcome(
                    valid=True,
                    steps=steps,
                    termination_reason=FAILURE_POLICY_EXCEPTION,
                    failure_category=FAILURE_POLICY_EXCEPTION,
                    exception=f"{type(exc).__name__}: {exc}",
                )
                return self._episode(case, outcome, max_phase=max_phase)

            # V2-E per-phase success tracking: classify the raw state with the
            # causal phase labeler (calibrated on the training demonstrations)
            # as the episode unfolds. Policy failures after at least one state
            # carry the phase reached so far.
            if self.phase_labeler is not None:
                try:
                    phase = self.phase_labeler.step(state)
                    if max_phase is None or phase > max_phase:
                        max_phase = phase
                except Exception as exc:  # noqa: BLE001 — labeler misuse
                    raise RuntimeError(
                        f"Per-phase tracking failed on step {steps}: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc

            # Pre-action observation for full tracing (the state the policy
            # acted on; `adapter.step` rebinds `state` to the next obs below).
            state_before_action = state
            try:
                state, _done, success, _info = self.adapter.step(action)
            except PolicyInvalidActionError as exc:
                return self._episode(
                    case,
                    RolloutOutcome(
                        valid=True,
                        steps=steps,
                        termination_reason=FAILURE_POLICY_INVALID_ACTION,
                        failure_category=FAILURE_POLICY_INVALID_ACTION,
                        exception=str(exc),
                    ),
                    max_phase=max_phase,
                )
            except InfrastructureError as exc:
                return self._episode(
                    case,
                    RolloutOutcome(valid=False, exception=str(exc)),
                    max_phase=max_phase,
                )
            except Exception as exc:  # noqa: BLE001 — simulator misuse
                return self._episode(
                    case,
                    RolloutOutcome(valid=False, exception=str(exc)),
                    max_phase=max_phase,
                )
            done_now = bool(success or steps + 1 >= self.horizon)
            if trace_rows is not None and (
                steps % self.trace_every_n_steps == 0 or done_now
            ):
                row, prev_params = self._trace_step(
                    episode_id=episode_seq,
                    case_id=int(case.index),
                    timestep=steps,
                    state=state_before_action,
                    action=action,
                    done=done_now,
                    prev_params=prev_params,
                )
                trace_rows.append(row)
            steps += 1
            if success:
                return self._episode(
                    case,
                    RolloutOutcome(
                        valid=True,
                        success=True,
                        steps=steps,
                        termination_reason="success",
                    ),
                    max_phase=max_phase,
                )

        return self._episode(
            case,
            RolloutOutcome(
                valid=True,
                timed_out=True,
                steps=steps,
                termination_reason=FAILURE_TIMEOUT,
                failure_category=FAILURE_TIMEOUT,
            ),
            max_phase=max_phase,
        )

    def _normalized_tensor(self, state: np.ndarray) -> tuple[torch.Tensor, torch.device]:
        """Normalized state tensor on the model device (shared helper)."""
        device = self._model_device or torch.device("cpu")
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
        if self._normalizer_mean is None or self._normalizer_std is None:
            # This fallback keeps the method safe for a custom normalizer
            # implementation while the standard FrozenNormalizer takes the
            # cached path above.
            assert self.normalizer is not None
            normalized_state = self.normalizer.normalize(state_tensor)
        else:
            normalized_state = (state_tensor - self._normalizer_mean) / self._normalizer_std
        return normalized_state, device

    def _policy_action(self, state: np.ndarray) -> np.ndarray:
        """Normalize + infer one action from the model (policy failures surface)."""
        if self.normalizer is None:
            raise PolicyInvalidActionError("No normalizer available for policy inference.")
        if self.model is None:
            raise PolicyInvalidActionError("No policy available.")
        normalized_state, _device = self._normalized_tensor(state)
        normalized = normalized_state.unsqueeze(0)
        with torch.inference_mode():
            action_tensor = self.model.get_action(normalized)  # type: ignore[operator]
        action = action_tensor.squeeze(0).cpu().numpy()
        return self.adapter.validate_action(action, tolerance=self.action_tolerance)

    def _describe_for_trace(self, state: np.ndarray) -> dict[str, Any] | None:
        """Model introspection snapshot for tracing; None when unavailable.

        Never raises: any failure degrades the affected fields to null so
        diagnostics can never convert a rollout outcome into an error.
        """
        describe = getattr(self.model, "describe_step", None)
        if self.model is None or not callable(describe):
            return None
        try:
            normalized_state, _device = self._normalized_tensor(state)
            with torch.inference_mode():
                snapshot = describe(normalized_state.unsqueeze(0))
            return dict(snapshot) if isinstance(snapshot, dict) else None
        except Exception:
            return None

    @staticmethod
    def _trace_json(value: Any) -> Any:
        """Single value/tensor/array to a plain JSON scalar or list."""
        if value is None:
            return None
        if torch.is_tensor(value):
            flat = value.detach().cpu().flatten()
            if flat.numel() == 1:
                return flat.item()
            return [float(v) for v in flat.tolist()]
        return to_jsonable(value)

    def _trace_step(
        self,
        *,
        episode_id: int,
        case_id: int,
        timestep: int,
        state: np.ndarray,
        action: np.ndarray,
        done: bool,
        prev_params: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[dict[str, Any], tuple[torch.Tensor, torch.Tensor] | None]:
        """Build one validated-shape trace row (22 spec fields).

        Returns ``(row, prev_params)`` where the second element carries the
        current ``(target, task_state)`` clones forward for the periodic
        Lipschitz finite-difference diagnostic.
        """
        state_np = np.asarray(state, dtype=np.float64)
        action_list = [float(v) for v in np.asarray(action, dtype=np.float64).reshape(-1)]
        try:
            normalized, _device = self._normalized_tensor(state_np)
            normalized_np = np.asarray(normalized.detach().cpu().numpy(), dtype=np.float64)
        except Exception:
            normalized_np = state_np
        snapshot = self._describe_for_trace(state_np) or {}

        def _field(name: str) -> Any:
            return self._trace_json(snapshot.get(name))

        nearest: float | None = None
        if self.reference_states is not None:
            try:
                nearest = float(
                    np.linalg.norm(self.reference_states - normalized_np, axis=-1).min()
                )
            except Exception:
                nearest = None

        lip: float | None = None
        current_params: tuple[torch.Tensor, torch.Tensor] | None = None
        target = snapshot.get("target")
        task_vars = snapshot.get("task_vars")
        if (
            torch.is_tensor(target)
            and torch.is_tensor(task_vars)
            and target.ndim == 2
            and task_vars.ndim == 2
        ):
            current_params = (target.detach().cpu().clone(), task_vars.detach().cpu().clone())
        if (
            timestep % LIP_SAMPLE_PERIOD == 0
            and prev_params is not None
            and current_params is not None
        ):
            prev_target, prev_task = prev_params
            cur_target, cur_task = current_params
            delta_target = float((cur_target - prev_target).norm().item())
            delta_state = float((cur_task - prev_task).norm().item())
            lip = delta_target / (delta_state + 1e-6)

        eef_pos = state_np[0:3].tolist() if state_np.size >= 3 else None
        row: dict[str, Any] = {
            "episode_id": int(episode_id),
            "case_id": int(case_id),
            "timestep": int(timestep),
            # Finalized by _episode (the outcome is known only there).
            "termination_reason": "running",
            "raw_obs_summary": {
                "state_norm": float(np.linalg.norm(state_np)),
                "eef_pos": eef_pos,
            },
            "normalized_state_norm": float(np.linalg.norm(normalized_np)),
            "task_vars": _field("task_vars"),
            "latent_norm": _field("latent_norm"),
            "dists": _field("dists"),
            "selected_expert": _field("selected_expert"),
            "top2_expert": _field("top2_expert"),
            "router_margin": _field("router_margin"),
            "router_entropy": _field("router_entropy"),
            "expert_target": _field("expert_target"),
            "expert_gains": _field("expert_gains"),
            "task_error": _field("task_error"),
            "pre_clip_command": _field("pre_clip_u"),
            "final_action": action_list,
            "nearest_train_dist": nearest,
            "expert_disagreement": _field("expert_disagreement"),
            "lip_diagnostic": lip,
            "done": bool(done),
        }
        return row, (current_params if current_params is not None else prev_params)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def _episode(
        self,
        case,
        outcome: RolloutOutcome,
        *,
        max_phase: int | None = None,
    ) -> tuple[RolloutOutcome, dict[str, Any]]:
        row: dict[str, Any] = {
            "run_id": self.run_id,
            "model": self.model_name,
            "checkpoint_sha256": self.checkpoint_sha256,
            "task": self.task,
            "training_seed": self.training_seed,
            "reset_seed": self.bank.seed,
            "episode_index": int(case.index),
            "valid_episode": outcome.valid,
            "steps": int(outcome.steps),
        }
        if self.tag is not None:
            row["tag"] = self.tag
        if self.router_mode != "learned":
            row["router_mode"] = self.router_mode
        if max_phase is not None:
            row["extra"] = {**(row.get("extra") or {}), "max_phase": int(max_phase)}
        if outcome.valid:
            row["success"] = outcome.success
            row["timed_out"] = outcome.timed_out
            row["termination_reason"] = outcome.termination_reason
            if not outcome.success:
                row["failure_category"] = outcome.failure_category
                if outcome.exception is not None:
                    row["exception"] = outcome.exception
        else:
            row["termination_reason"] = "infrastructure"
            row["exception"] = outcome.exception or "unknown infrastructure failure"
        if outcome.extra:
            row["extra"] = {**(row.get("extra") or {}), **outcome.extra}
        # Full-trace finalization: stamp the now-known termination
        # reason, close the last row, and append. Validation errors propagate
        # (they indicate a tracer bug, caught by tests); filesystem errors
        # must not invalidate an otherwise valid rollout.
        trace_rows = self._active_trace_rows
        self._active_trace_rows = None
        if trace_rows is not None:
            termination = outcome.termination_reason or "infrastructure"
            for trace_row in trace_rows:
                trace_row["termination_reason"] = termination
            if trace_rows:
                trace_rows[-1]["done"] = True
            try:
                TraceWriter(self.output_dir).append_episode_rows(trace_rows)
            except OSError:
                logger.exception("Trace append failed; continuing with episode record.")
        return outcome, row

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def _summarize(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        summaries = summarize_episodes(rows)
        summary = summaries[0] if summaries else {}
        valid = int(summary.get("valid_episodes", 0))
        successes = int(summary.get("successes", 0))
        low, high = wilson_interval(successes, valid)
        policy_failures = int(summary.get("policy_failures", 0))
        results: dict[str, Any] = {
            "eval/action_mse": float("nan"),
            "eval/rollout/success_rate": (successes / valid if valid else float("nan")),
            "eval/rollout/valid_episodes": valid,
            "eval/rollout/successes": successes,
            "eval/rollout/policy_failures": policy_failures,
            "eval/rollout/invalid_attempts": int(summary.get("invalid_attempts", 0)),
            "eval/rollout/wilson_ci95_low": low,
            "eval/rollout/wilson_ci95_high": high,
            "eval/rollout/horizon": self.horizon,
            "eval/rollout/reset_bank": self.bank.bank_id,
            "eval/rollout/reset_seed": self.bank.seed,
            "eval/rollout/task": self.task,
            "eval/rollout/router_mode": self.router_mode,
            "eval/rollout/trace_level": self.trace_level,
        }
        if self.num_phases is not None:
            results["eval/rollout/per_phase_sr"] = _per_phase_success_rates(
                rows, num_phases=self.num_phases
            )
            results["eval/rollout/phase_tracking"] = "aggregated"
        else:
            results["eval/rollout/phase_tracking"] = "missing"
        return results

    def _write_run_summary(self, results: dict[str, Any], rows: list[dict[str, Any]]) -> None:
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "model": self.model_name,
            "tag": self.tag,
            "task": self.task,
            "training_seed": self.training_seed,
            "reset_seed": self.bank.seed,
            "reset_bank": self.bank.bank_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "horizon": self.horizon,
            "router_mode": self.router_mode,
            "episodes": len(rows),
            "failure_categories": _failure_breakdown(rows),
            "metrics": {
                key: value for key, value in results.items() if key.startswith("eval/rollout/")
            },
        }
        (self.output_dir / "rollout_summary.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )


def _phase_labeler_from_aggregate(thresholds: dict[str, Any]) -> CausalPhaseStepLabeler:
    """Build the causal step labeler from the aggregated calibration artifact.

    ``phase_thresholds.json`` stores the median hysteresis levels
    (``closed_level``/``open_level``/``mirror``/``mirror_bounds``) plus the
    full per-demo artifacts. The labeler needs one complete calibration; the
    median levels are authoritative, and the labeler parameters that are
    constant across demos (thresholds, slices, filter size) come from the
    first per-demo artifact.
    """
    demos = thresholds.get("per_demo") or []
    if not demos:
        raise ValueError(
            "Aggregated phase_thresholds artifact has no per_demo entries; "
            "cannot build a phase labeler."
        )
    calibration = dict(demos[0])
    calibration["closed_level"] = thresholds["closed_level"]
    calibration["open_level"] = thresholds["open_level"]
    calibration["mirror"] = bool(thresholds.get("mirror", demos[0]["mirror"]))
    if calibration["mirror"]:
        bounds = thresholds.get("mirror_bounds") or demos[0].get("mirror_bounds")
        calibration["mirror_bounds"] = bounds
    return CausalPhaseStepLabeler(calibration)


def _per_phase_success_rates(
    rows: list[dict[str, Any]], num_phases: int
) -> dict[str, float | None]:
    """Per-phase success rate: P(success | episode reached phase p).

    An episode "reaches" phase p when its recorded max phase is at least p
    (the causal phase labeler classifies every observed state); only valid
    episodes count in the denominator (infra failures carry no states).
    Each entry is a float or null when no valid episode reached the phase.
    """
    rates: dict[str, float | None] = {}
    for phase in range(num_phases):
        reached: list[dict[str, Any]] = [
            row
            for row in rows
            if row.get("valid_episode")
            and (row.get("extra") or {}).get("max_phase") is not None
            and int(row["extra"]["max_phase"]) >= phase
        ]
        successes = [row for row in reached if row.get("success")]
        rates[str(phase)] = (
            len(successes) / len(reached) if reached else None
        )
    return rates


def _failure_breakdown(rows: list[dict[str, Any]]) -> dict[str, int]:
    breakdown: dict[str, int] = {}
    for row in rows:
        if row["valid_episode"] and not row.get("success"):
            category = row.get("failure_category") or "unknown"
            breakdown[category] = breakdown.get(category, 0) + 1
        elif not row["valid_episode"]:
            breakdown["infrastructure"] = breakdown.get("infrastructure", 0) + 1
    return breakdown


# ---------------------------------------------------------------------------
# CLI-facing orchestration
# ---------------------------------------------------------------------------


def state_spec_from_config(cfg: DictConfig) -> StateSpec:
    """Build the declared StateSpec from ``data.state_keys`` (list of dicts)."""
    entries = list(cfg.data.state_keys)
    keys: list[str] = []
    dims: list[int] = []
    for entry in entries:
        keys.append(str(entry["key"]))
        dims.append(int(entry["dim"]))
    spec = StateSpec(keys=tuple(keys), dims=tuple(dims))
    declared = int(cfg.data.get("state_dim", -1))
    if declared != spec.dim:
        raise EnvParityError(
            f"data.state_keys sum to {spec.dim} but data.state_dim declares {declared}."
        )
    data_source = cfg.data.get("source")
    task_name = data_source.get("task_name") if data_source is not None else None
    if task_name is not None:
        from phaseforge.evaluations.envs.task_registry import validate_task_schema

        validate_task_schema(
            str(task_name),
            spec.keys,
            spec.dims,
            int(cfg.data.action_dim),
        )
    return spec


def resolve_cache_dir(cfg: DictConfig) -> Path:
    """Resolve the processed dataset cache directory for evaluation.

    The cache contains ``norm_stats.pt``, ``trajectories/``, and optionally
    ``phase_thresholds.json``. Because ``CacheManager.compute_hash`` includes
    the git commit, checking out a fix or moving commits across training and
    evaluation causes the newly computed hash to differ from the cache directory
    created during training.

    This function recovers the authentic cache directory by checking:
    1. The exact directory ``processed_cache_root() / compute_hash(cfg.data)``.
    2. The training run's recorded ``data_config_hash`` from ``run_meta.json``
       or ``environment.json`` sibling to the evaluation checkpoint.
    3. The repo-relative cache root ``PROJECT_ROOT / data / processed / cache``.
    4. Any valid cache directory in the cache root whose ``task_index.json`` or
       ``manifest.json`` matches the task name.
    5. Falls back to ``processed_cache_root() / compute_hash(cfg.data)``.
    """
    from phaseforge.data.ingestion.cache_manager import CacheManager
    from phaseforge.data.paths import DEFAULT_DATA_DIR, processed_cache_root

    data_cfg = cfg.get("data") if hasattr(cfg, "get") else getattr(cfg, "data", None)
    primary_root = processed_cache_root()
    if data_cfg is None:
        return primary_root

    current_hash = CacheManager.compute_hash(data_cfg)
    roots = [primary_root]

    # If primary root is the default ./data/processed/cache and doesn't exist,
    # also search repo-relative data root if it exists.
    default_root = (Path(DEFAULT_DATA_DIR) / "processed" / "cache").resolve()
    repo_cache_root = Path(__file__).resolve().parents[3] / "data" / "processed" / "cache"
    if (
        primary_root.resolve() == default_root
        and not primary_root.exists()
        and repo_cache_root.exists()
        and repo_cache_root.resolve() != primary_root.resolve()
    ):
        roots.append(repo_cache_root)

    # 1. Exact match for current hash
    for root in roots:
        cand = root / current_hash
        if (cand / "norm_stats.pt").is_file() or (cand / "trajectories").is_dir():
            return cand

    # 2. Checkpoint provenance: read data_config_hash recorded by training
    train_cfg = cfg.get("train") if hasattr(cfg, "get") else getattr(cfg, "train", None)
    ckpt_path = (
        train_cfg.get("stage1_ckpt_path")
        if train_cfg is not None and hasattr(train_cfg, "get")
        else None
    )
    if ckpt_path:
        ckpt = Path(str(ckpt_path))
        if not ckpt.is_absolute():
            for base in [Path.cwd(), Path(__file__).resolve().parents[3]]:
                if (base / ckpt).is_file():
                    ckpt = base / ckpt
                    break
        search_dirs = [ckpt.parent, ckpt.parent.parent]
        for sdir in search_dirs:
            for meta_name in (
                "run_meta.json",
                "metadata/environment.json",
                "metadata/data_provenance.json",
            ):
                meta_file = sdir / meta_name
                if meta_file.is_file():
                    try:
                        data = json.loads(meta_file.read_text(encoding="utf-8"))
                        train_hash = (
                            data.get("data_config_hash")
                            or data.get("config_hash")
                            or (data.get("provenance") or {}).get("config_hash")
                        )
                        if train_hash:
                            for root in roots:
                                cand = root / str(train_hash)
                                if (
                                    (cand / "norm_stats.pt").is_file()
                                    or (cand / "trajectories").is_dir()
                                ):
                                    logger.info(
                                        "Resolved cache dir from checkpoint provenance (%s): %s",
                                        meta_name,
                                        cand,
                                    )
                                    return cand
                    except Exception:
                        pass

    # 3. Task matching across available caches
    source_cfg = data_cfg.get("source") if hasattr(data_cfg, "get") else None
    project_cfg = cfg.get("project") if hasattr(cfg, "get") else getattr(cfg, "project", None)

    task_name = ""
    if source_cfg is not None and hasattr(source_cfg, "get"):
        task_name = str(source_cfg.get("task_name") or "")
    if not task_name and project_cfg is not None and hasattr(project_cfg, "get"):
        task_name = str(project_cfg.get("tag") or "")
    if not task_name and hasattr(data_cfg, "get"):
        task_name = str(data_cfg.get("task") or "")
    task_name = task_name.strip().lower()

    if task_name:
        for root in roots:
            if not root.is_dir():
                continue
            for subdir in sorted(root.iterdir()):
                if not subdir.is_dir() or subdir.name.endswith("_tmp"):
                    continue
                ti_file = subdir / "task_index.json"
                if ti_file.is_file():
                    try:
                        ti = json.loads(ti_file.read_text(encoding="utf-8"))
                        if any(task_name in k.lower() or k.lower() in task_name for k in ti.keys()):
                            logger.info(
                                "Resolved cache dir by task match (%r in task_index.json): %s",
                                task_name,
                                subdir,
                            )
                            return subdir
                    except Exception:
                        pass
                man_file = subdir / "manifest.json"
                if man_file.is_file():
                    try:
                        man = json.loads(man_file.read_text(encoding="utf-8"))
                        prov = man.get("provenance") or {}
                        split_names = prov.get("split_task_names") or {}
                        all_names = [n.lower() for names in split_names.values() for n in names]
                        if any(task_name in n or n in task_name for n in all_names):
                            logger.info(
                                "Resolved cache dir by task match in manifest.json: %s",
                                subdir,
                            )
                            return subdir
                        raw_files = prov.get("raw_files") or []
                        if any(task_name in str(rf.get("name", "")).lower() for rf in raw_files):
                            logger.info(
                                "Resolved cache dir by raw file match in manifest.json: %s",
                                subdir,
                            )
                            return subdir
                    except Exception:
                        pass

    return primary_root / current_hash


def resolve_rollout_normalizer(
    cfg: DictConfig,
    model: torch.nn.Module | None = None,
    cache_dir: Path | None = None,
) -> FrozenNormalizer:
    """Resolve the FrozenNormalizer for rollout evaluation.

    Resolution order:
    1. Model persistent buffers (`model.get_normalizer_stats()`).
       When the model already has ``normalizer_mean`` and ``normalizer_std``
       buffers (restored from the checkpoint), this returns the exact statistics
       the model was trained on, completely independent of disk cache state or git commit.
    2. Explicit ``cache_dir / "norm_stats.pt"`` if provided and present.
    3. Resolved cache directory from :func:`resolve_cache_dir`.
    """
    from phaseforge.data.common.normalizer import FrozenNormalizer

    # 1. Model buffers (highest authority: exact training statistics)
    target_model = getattr(model, "module", model)
    if target_model is not None and hasattr(target_model, "get_normalizer_stats"):
        mean, std = target_model.get_normalizer_stats()
        if mean is not None and std is not None:
            logger.info("Using normalizer statistics cached in model checkpoint buffers.")
            return FrozenNormalizer(mean=mean.cpu().detach(), std=std.cpu().detach())

    # 2. Provided cache_dir
    if cache_dir is not None:
        norm_file = Path(cache_dir) / "norm_stats.pt"
        if norm_file.is_file():
            return FrozenNormalizer.load(norm_file)
        try:
            return FrozenNormalizer.load(norm_file)
        except (FileNotFoundError, OSError):
            pass

    # 3. Resolved cache directory
    resolved = resolve_cache_dir(cfg)
    norm_file = resolved / "norm_stats.pt"
    if norm_file.is_file():
        return FrozenNormalizer.load(norm_file)
    try:
        return FrozenNormalizer.load(norm_file)
    except (FileNotFoundError, OSError):
        pass

    raise FileNotFoundError(
        f"Could not load normalizer statistics: model has no registered "
        f"normalizer buffers, and norm_stats.pt was not found at {norm_file} "
        f"(nor in any resolved cache directory for data config)."
    )


def resolve_pinned_metadata(
    cfg: DictConfig, cache_dir: Path | None = None
) -> PinnedEnvMetadata:
    """Recover the pinned env metadata from the cache (or raw HDF5, or dev).

    Priority (fail-closed): processed cache trajectory -> raw HDF5
    ``env_args`` -> documented dev fallback (only when the data source is
    genuinely absent locally; the evaluation machine always hits the
    cache path because training ingests the same dataset).

    The dev fallback is a *hard fail* for real runs: silently rolling out
    against the wrong task's environment (e.g. Lift metadata for a Can
    run whose cache is missing on the machine) would produce plausible-
    looking but invalid results. Callers that genuinely want the fallback
    (local self-tests / gates) must opt in via ``eval.env.allow_dev_fallback``.
    """
    if cache_dir is None:
        cache_dir = resolve_cache_dir(cfg)
    if (cache_dir / "trajectories").is_dir():
        meta = env_metadata_from_cache(cache_dir)
        logger.info("Pinned env metadata recovered from cache %s", cache_dir.name)
        return meta

    raw_dir = cfg.data.source.get("dir")
    hdf5_files = []
    if raw_dir:
        raw_path = Path(str(raw_dir))
        if raw_path.is_dir():
            hdf5_files = sorted(raw_path.glob("*.hdf5"))
    if hdf5_files:
        from phaseforge.evaluations.envs.env_metadata import env_metadata_from_hdf5

        meta = env_metadata_from_hdf5(hdf5_files[0])
        logger.info("Pinned env metadata recovered from raw HDF5 %s", hdf5_files[0])
        return meta

    allow_dev = bool(cfg.eval.env.get("allow_dev_fallback", False))
    if not allow_dev:
        task_name = str(cfg.data.source.get("task_name") or "Lift")
        raise RuntimeError(
            f"No processed cache (hash {cache_dir.name}) and no raw HDF5 found for "
            f"task {task_name!r} — the pinned environment metadata cannot be "
            "recovered, so a rollout would silently run against the wrong "
            "environment. Copy the dataset/cache to this machine (or re-ingest "
            "the raw HDF5) and re-run, or set eval.env.allow_dev_fallback=true "
            "ONLY for local self-tests/gates."
        )

    from phaseforge.evaluations.envs.env_metadata import dev_fallback_metadata

    logger.warning(
        "No processed cache or raw HDF5 found — using the documented dev "
        "fallback env metadata (allow_dev_fallback=true). This is only "
        "acceptable for local self-tests/gates; a real rollout requires the "
        "dataset."
    )
    task_name = str(cfg.data.source.get("task_name") or "Lift")
    return dev_fallback_metadata(task_name)


def resolve_robosuite_requirement(cfg: DictConfig) -> str:
    """Resolve the effective robosuite requirement for this task.

    Per-task data configuration is authoritative when present.  The
    evaluation-level value remains a compatibility fallback for legacy
    configs that predate the five-task source pins.
    """
    source_requirement = cfg.data.source.get("robosuite_requirement")
    if source_requirement is not None and str(source_requirement).strip():
        return str(source_requirement)
    return str(cfg.eval.env.get("robosuite_requirement", "==1.5.1"))


def resolve_trace_level(cfg: DictConfig) -> str:
    """Resolve the rollout trace verbosity.

    Reads ``eval.episodes.trace_level`` with a ``minimal`` default so
    legacy configs and unit-test ``DictConfig`` objects without the key
    keep today's behavior. Unknown values fail closed with
    :class:`EnvParityError`.
    """
    try:
        episodes = cfg.eval.episodes
    except Exception:
        return "minimal"
    try:
        get = getattr(episodes, "get", None)
        raw = get("trace_level", "minimal") if callable(get) else "minimal"
    except Exception:
        return "minimal"
    level = str(raw).lower()
    if level not in TRACE_LEVELS:
        raise EnvParityError(
            f"eval.episodes.trace_level={raw!r} is not one of "
            + ", ".join(sorted(TRACE_LEVELS))
            + "."
        )
    return level


# Sections the rollout evaluator requires from the ``eval=rollout``
# config group. ``metrics.yaml`` does not define any of them; a Hydra
# composition that leaves the default group in place (e.g. only setting
# ``eval.mode=rollout`` without the ``eval=rollout`` group selector)
# triggers the schema guard below rather than the obscure
# ``ConfigAttributeError`` deep in ``load_or_generate_bank``.
_ROLLOUT_EVAL_SECTIONS = ("bank", "env", "episodes")


def require_rollout_eval_schema(cfg: DictConfig) -> None:
    """Fail fast with an actionable message when the rollout schema is missing.

    ``cfg.eval.mode=rollout`` alone is not enough — Hydra composes the
    ``eval`` group from the default (``metrics``) config unless the
    group selector ``eval=rollout`` is also passed, and the metrics
    schema has no ``bank``/``env``/``episodes`` sections. This guard
    turns that ``ConfigAttributeError`` into a single actionable error
    at the public entry points (``run_rollout_evaluation``,
    ``run_all_gates``) before any expensive work runs.
    """
    eval_cfg = cfg.get("eval") if hasattr(cfg, "get") else cfg.eval
    if eval_cfg is None:
        raise EnvParityError(
            "Rollout evaluation requires the eval=rollout config group, "
            "but cfg.eval is missing entirely. Pass 'eval=rollout' on the "
            "command line (the phaseforge-sweep runner does this "
            "automatically) — 'eval.mode=rollout' alone leaves the "
            "default metrics group in place."
        )
    missing = [
        section
        for section in _ROLLOUT_EVAL_SECTIONS
        if section not in eval_cfg
    ]
    if missing:
        raise EnvParityError(
            "Rollout evaluation requires the eval=rollout config group, "
            f"but cfg.eval is missing section(s): {', '.join(missing)}. "
            "Pass 'eval=rollout' on the command line (the phaseforge-sweep "
            "runner does this automatically) — 'eval.mode=rollout' alone "
            "leaves the default metrics group in place."
        )


def load_or_generate_bank(cfg: DictConfig, meta: PinnedEnvMetadata) -> ResetBank:
    """Load the frozen bank for the pinned env; generate if configured.

    ``cfg.eval.bank.auto_generate`` (default true) creates the bank on
    first use; the artifact is then frozen and verified by SHA-256 on
    every later load. Regeneration of an existing, hash-verified bank is
    impossible (its ``bank_id`` is content-derived), so existing banks are
    always reused.
    """
    from phaseforge.data.paths import get_data_root

    task = str(cfg.data.source.get("task_name") or meta.env_name)
    seed = int(cfg.eval.bank.get("seed", 2026))
    num_cases = int(cfg.eval.bank.get("num_cases", 50))
    configured_expected_env = cfg.eval.env.get("expected_env_name")
    expected_env_name = (
        str(configured_expected_env) if configured_expected_env is not None else meta.env_name
    )
    versions = verify_environment_parity(
        meta,
        expected_env_name=expected_env_name,
        robosuite_requirement=resolve_robosuite_requirement(cfg),
        mujoco_requirement=str(cfg.eval.env.get("mujoco_requirement", "==3.2.7")),
    )
    bank_id = compute_bank_id(
        meta,
        task=task,
        seed=seed,
        num_cases=num_cases,
        robosuite_version=versions["robosuite"],
    )
    directory = bank_dir(get_data_root(), task, bank_id)
    if directory.is_dir():
        bank = ResetBank.load(directory)
        mismatches: list[str] = []
        if bank.bank_id != bank_id:
            mismatches.append(f"bank_id={bank.bank_id!r} (expected {bank_id!r})")
        if bank.task != task:
            mismatches.append(f"task={bank.task!r} (expected {task!r})")
        if bank.seed != seed:
            mismatches.append(f"seed={bank.seed} (expected {seed})")
        if bank.num_cases != num_cases:
            mismatches.append(f"num_cases={bank.num_cases} (expected {num_cases})")
        if bank.env_canonical != meta.canonical_json():
            mismatches.append("env_canonical does not match the pinned dataset metadata")
        if bank.robosuite_version != versions["robosuite"]:
            mismatches.append(
                f"robosuite_version={bank.robosuite_version!r} (expected {versions['robosuite']!r})"
            )
        if mismatches:
            raise EnvParityError(
                f"Reset bank {directory} identity mismatch; refusing to use it: "
                + "; ".join(mismatches)
            )
        logger.info("Loaded frozen reset bank %s (%d cases)", bank_id, bank.num_cases)
        return bank

    auto = bool(cfg.eval.bank.get("auto_generate", True))
    if not auto:
        raise EnvParityError(
            f"Reset bank {bank_id} does not exist under {directory} and "
            "auto_generate is disabled. Generate the bank explicitly "
            "(documented exception: one-time artifact, then frozen)."
        )
    logger.warning(
        "Reset bank %s does not exist — generating it now (one-time "
        "artifact, then frozen and verified on every load).",
        bank_id,
    )
    bank = generate_reset_bank(
        lambda: _adapter_from_config(cfg, meta, seed=seed),
        meta,
        task=task,
        seed=seed,
        num_cases=num_cases,
        max_attempts_per_case=int(cfg.eval.bank.get("max_attempts_per_case", 40)),
        robosuite_version=versions["robosuite"],
        git_commit=_git_commit(),
    )
    directory.parent.mkdir(parents=True, exist_ok=True)
    bank.save(directory)
    logger.info("Reset bank %s generated and saved to %s", bank_id, directory)
    return bank


def _git_commit() -> str:
    try:
        from phaseforge.data.ingestion.cache_manager import git_commit

        return git_commit()
    except Exception:  # noqa: BLE001
        return ""


def _adapter_from_config(
    cfg: DictConfig, meta: PinnedEnvMetadata, *, seed: int | None = None
) -> RobosuiteStateAdapter:
    from phaseforge.evaluations.envs.robosuite_adapter import RobosuiteStateAdapter
    from phaseforge.evaluations.envs.task_registry import TaskSpec

    spec = state_spec_from_config(cfg)
    action_contract = cfg.data.action_contract
    task_name = str(cfg.data.source.get("task_name") or meta.env_name)
    # The v1.5.1 PH Transport metadata omits ``horizon``.  In that case use
    # the protocol registry's task-specific default (700 for Transport), not
    # PinnedEnvMetadata's generic legacy fallback of 500.  An explicitly
    # serialized horizon still wins.
    horizon = meta.horizon
    if "horizon" not in meta.env_kwargs:
        horizon = TaskSpec.from_protocol(task_name).horizon
    return RobosuiteStateAdapter(
        meta,
        spec,
        action_dim=int(cfg.data.action_dim),
        action_low=float(action_contract.range[0]),
        action_high=float(action_contract.range[1]),
        horizon=horizon,
        seed=seed,
        # The adapter owns the action-contract tolerance so every validation
        # site (runner pre-step check and step's own guard) enforces the
        # same configured contract.
        action_tolerance=float(cfg.eval.episodes.get("action_tolerance", 1e-4)),
    )


def run_rollout_evaluation(
    cfg: DictConfig,
    model: torch.nn.Module,
    output_dir: Path,
    *,
    run_id: str,
    checkpoint_sha256: str = "",
) -> dict[str, Any]:
    """Full rollout evaluation for one model checkpoint (called by the CLI).

    Order: pinned env metadata -> environment parity gate
    (fail closed) -> frozen reset bank -> normalized rollout over all bank
    cases -> strict-metric episode rows -> per-run summary.
    """
    require_rollout_eval_schema(cfg)
    from phaseforge.data.ingestion.cache_manager import load_phase_thresholds

    meta = resolve_pinned_metadata(cfg)
    bank = load_or_generate_bank(cfg, meta)
    adapter = _adapter_from_config(cfg, meta)
    cache_dir = resolve_cache_dir(cfg)
    normalizer = resolve_rollout_normalizer(cfg, model=model, cache_dir=cache_dir)

    # V2-E: evaluation-time routing intervention (learned/sticky/uniform/
    # oracle). Baselines that are not PhaseBootstrappedMoE have no eval_mode
    # and always route learned.
    router_mode = str(cfg.eval.episodes.get("router_mode", "learned")).lower()
    from phaseforge.models.phase_moe import EVAL_ROUTER_MODES

    if router_mode not in EVAL_ROUTER_MODES:
        raise EnvParityError(
            f"eval.episodes.router_mode={router_mode!r} is not one of "
            + ", ".join(sorted(EVAL_ROUTER_MODES))
        )
    if hasattr(model, "eval_mode"):
        # setattr because nn.Module.__getattr__ types any unknown attribute
        # as Tensor | Module and mypy would reject a plain str assignment.
        setattr(model, "eval_mode", router_mode)

    # Trace verbosity + contract log.
    # `minimal` preserves the episodes.jsonl-only behavior; `full` writes
    # the 22-field per-step trace.jsonl alongside episodes.jsonl.
    # A non-learned router_mode is logged (not failed): sticky /
    # uniform / oracle are legitimate eval ablations, but only `learned`
    # counts as the primary memoryless-contract result.
    trace_level = resolve_trace_level(cfg)
    try:
        trace_every = int(cfg.eval.episodes.get("trace_every_n_steps", 1))
    except (TypeError, ValueError):
        raise EnvParityError(
            "eval.episodes.trace_every_n_steps must be an int >= 1."
        ) from None
    if trace_every < 1:
        raise EnvParityError(
            f"eval.episodes.trace_every_n_steps must be >= 1, got {trace_every}."
        )
    contract = None
    contract_fn = getattr(model, "deployment_contract", None)
    if callable(contract_fn):
        try:
            contract = contract_fn()
        except Exception:
            logger.warning("deployment_contract() raised; continuing.", exc_info=True)
            contract = None
    if contract is not None:
        logger.info("Deployment contract: %s (router_mode=%s).", contract, router_mode)
    if trace_level == "full":
        logger.info("trace_level=full: per-step trace.jsonl will be written.")
    if router_mode != "learned":
        logger.info("router_mode=%s is an eval ablation, not the primary result.", router_mode)

    # V2-E per-phase success tracking: the training cache's aggregated
    # phase-threshold calibration (median hysteresis + shared labeler
    # params) classifies the policy's own states causally during rollout.
    # Caches that predate the artifact fail closed when tracking is
    # required; otherwise per-phase SR is null.
    thresholds = load_phase_thresholds(cache_dir)
    require_tracking = bool(cfg.eval.episodes.get("require_phase_tracking", False))
    phase_labeler: CausalPhaseStepLabeler | None = None
    if thresholds is not None:
        phase_labeler = _phase_labeler_from_aggregate(thresholds)
        logger.info(
            "Per-phase success tracking active (aggregated calibration over "
            "%d demos).",
            int(thresholds.get("n_demos", 0)),
        )
    elif require_tracking:
        raise EnvParityError(
            f"eval.episodes.require_phase_tracking=true but cache {cache_dir.name} "
            "has no phase_thresholds.json (it predates per-demo phase "
            "calibration). Re-ingest the raw HDF5 before per-phase success "
            "tracking can run."
        )

    model_name = getattr(cfg.models, "name", cfg.models._target_.split(".")[-1])
    training_seed = cfg.project.get("seed")
    if not isinstance(training_seed, int):
        raise EnvParityError(
            "Rollout evaluation requires an integer project seed (shared "
            "across variants for the paired comparisons)."
        )

    # Nearest-train-distance proxy: deterministic subsample of
    # normalized train states from the same cache the normalizer came from.
    # Unreadable cache -> null OOD scores (never breaks the rollout).
    reference_states: np.ndarray | None = None
    if trace_level == "full":
        from phaseforge.evaluations.rollout.trace import load_reference_states

        reference_states = load_reference_states(cache_dir)
        if reference_states is None:
            logger.info("No reference states for OOD scores; recording nulls.")

    evaluator = RolloutEvaluator(
        cfg=cfg,
        policy=None,
        adapter=adapter,
        bank=bank,
        normalizer=normalizer,
        model=model,
        output_dir=output_dir,
        run_id=run_id,
        model_name=model_name,
        training_seed=training_seed,
        task=str(cfg.data.source.get("task_name") or meta.env_name),
        checkpoint_sha256=checkpoint_sha256,
        tag=cfg.project.get("tag"),
        horizon=cfg.eval.episodes.get("horizon"),
        action_tolerance=float(cfg.eval.episodes.get("action_tolerance", 1e-4)),
        phase_labeler=phase_labeler,
        router_mode=router_mode,
        trace_level=trace_level,
        trace_every_n_steps=trace_every,
        reference_states=reference_states,
    )
    try:
        return evaluator.run()
    finally:
        adapter.close()


__all__ = [
    "RolloutEvaluator",
    "RolloutOutcome",
    "RolloutRunInvalid",
    "TRACE_LEVELS",
    "resolve_trace_level",
    "run_rollout_evaluation",
    "resolve_cache_dir",
    "resolve_rollout_normalizer",
    "resolve_pinned_metadata",
    "resolve_robosuite_requirement",
    "require_rollout_eval_schema",
    "load_or_generate_bank",
    "state_spec_from_config",
]

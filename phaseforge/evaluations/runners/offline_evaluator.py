"""Offline evaluator runner.

Computes offline metrics over a validation/test dataset. Static metrics
(routing entropy, expert utilization, phase-expert NMI, action MSE) pool
all samples; **temporal** metrics (``routing_entropy_variance``,
``time_to_stable_routing``, ``boundary_action_smoothness``) are computed
PER TRAJECTORY and then averaged — never across task/trajectory
boundaries.

Trajectory boundaries come from the collator keys ``trajectory_id`` /
``trajectory_position`` (``PhaseAwareCollator`` / ``StateOnlyDataset``).
When a dataloader does not provide them, the temporal metrics are
skipped with a loud warning and recorded in ``eval/skipped_metrics`` —
they are NEVER computed on a merged step stream.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from phaseforge.evaluations.metrics import (
    expert_utilization,
    phase_alignment,
    routing_stability,
    task_metrics,
)
from phaseforge.models.base import BaseManipulationModel

logger = logging.getLogger(__name__)


def _regroup_by_trajectory(
    trajectory_ids: torch.Tensor,
    trajectory_positions: torch.Tensor,
    values: torch.Tensor,
) -> list[torch.Tensor]:
    """Split a pooled tensor back into per-trajectory, time-ordered tensors.

    ``values`` is (N, ...) aligned row-wise with ``trajectory_ids`` (N,).
    Returns one tensor per trajectory (rows sorted by ``trajectory_position``),
    so shuffled loaders still reconstruct the exact episode order.
    """
    ids = trajectory_ids.tolist()
    positions = trajectory_positions.tolist()
    groups: dict[int, list[tuple[int, torch.Tensor]]] = {}
    for row_idx, (tid, pos) in enumerate(zip(ids, positions)):
        groups.setdefault(tid, []).append((pos, values[row_idx]))
    out: list[torch.Tensor] = []
    for tid in sorted(groups):
        ordered = sorted(groups[tid], key=lambda item: item[0])
        out.append(torch.stack([v for _, v in ordered]))
    return out


class OfflineEvaluator:
    """Runs a comprehensive offline evaluation over a validation/test dataset.

    Computes all enabled metrics defined in config/eval/metrics.yaml.
    """

    def __init__(
        self, cfg: DictConfig, model: BaseManipulationModel, dataloader: DataLoader
    ) -> None:
        self.cfg = cfg
        self.metrics_cfg = cfg.eval
        requested_device = str(cfg.project.get("device", "cuda"))
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning(
                "Device '%s' requested but CUDA is unavailable. Falling back to CPU.",
                requested_device,
            )
            requested_device = "cpu"
        self.device = torch.device(requested_device)
        self.model = model.to(self.device)
        self.dataloader = dataloader

    def _metric_enabled(self, group: str, name: str) -> bool:
        """Read an enabled flag defensively (missing group -> metric off)."""
        section = self.metrics_cfg.get(group, None)
        if section is None:
            return False
        entry = section.get(name, None)
        return bool(entry.get("enabled", False)) if entry is not None else False

    @torch.no_grad()
    def run(self) -> dict[str, Any]:
        """Execute the evaluation loop and return aggregated metrics."""
        self.model.eval()

        all_action_preds = []
        all_action_targets = []
        all_phases = []
        all_routing_weights = []
        all_expert_indices = []
        all_gate_logits = []
        all_phase_logits = []
        all_masks = []
        all_traj_ids = []
        all_traj_positions = []
        has_trajectory_ids = True
        boundary_error_enabled = self._metric_enabled("task", "phase_boundary_error")

        # 1. Collect all outputs
        for batch in self.dataloader:
            has_trajectory_ids = has_trajectory_ids and (
                "trajectory_id" in batch and "trajectory_position" in batch
            )
            # Trajectory identity is evaluation bookkeeping, not model input:
            # pop it before the forward pass so no model sees these keys.
            if "trajectory_id" in batch:
                all_traj_ids.append(batch.pop("trajectory_id").cpu())
            if "trajectory_position" in batch:
                all_traj_positions.append(batch.pop("trajectory_position").cpu())

            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            out = self.model(batch)

            all_action_preds.append(out.action_pred.detach().cpu())
            all_action_targets.append(batch["action"].cpu())
            all_phases.append(batch["phase"].cpu())

            mask = batch.get("padding_mask")
            if mask is not None:
                all_masks.append(mask.cpu())

            if out.routing_weights is not None:
                all_routing_weights.append(out.routing_weights.detach().cpu())
            if out.expert_indices is not None:
                all_expert_indices.append(out.expert_indices.detach().cpu())
            if out.gate_logits is not None:
                all_gate_logits.append(out.gate_logits.detach().cpu())
            if out.phase_logits is not None and boundary_error_enabled:
                all_phase_logits.append(out.phase_logits.detach().cpu())

        # 2. Fail clearly on an empty loader (never return a hollow results dict).
        if not all_action_preds:
            raise RuntimeError(
                "Offline evaluation produced NO batches — the validation/test "
                "dataloader is empty. Refusing to report metrics over zero data."
            )
        # Inconsistent output lists (batch counts must match for the required
        # outputs; per-batch concatenation must never misalign).
        if len(all_phases) != len(all_action_preds):
            raise RuntimeError(
                f"Inconsistent output collection: {len(all_action_preds)} action "
                f"batches vs {len(all_phases)} phase batches."
            )

        # 2b. Concatenate (flattening batch dim if needed, or keeping it depending on metrics)
        action_preds = torch.cat(all_action_preds, dim=0)
        action_targets = torch.cat(all_action_targets, dim=0)
        phases = torch.cat(all_phases, dim=0)

        # 2c. Decisive signal first, logged IMMEDIATELY: raw action error. If a
        #     later metric is slow (e.g. the routing metrics) and the run gets
        #     interrupted, this line is what distinguishes "the model cannot
        #     reproduce the training actions" (rollout success would be
        #     expected to remain near zero in that case)
        #     from "the eval path is broken" (low MSE but 0% rollouts).
        mask = torch.cat(all_masks, dim=0) if all_masks else None
        mse = task_metrics.action_mse(action_preds, action_targets, mask)
        logger.info(
            "  eval/action_mse: %.6f (RMSE %.4f)",
            mse,
            math.sqrt(max(mse, 0.0)),
        )
        # The payload intentionally mixes floats with the definitions record.
        results: dict[str, Any] = {"eval/action_mse": mse}

        is_moe = len(all_expert_indices) > 0
        expert_indices: torch.Tensor | None
        gate_logits: torch.Tensor | None
        if is_moe:
            expert_indices = torch.cat(all_expert_indices, dim=0)
            gate_logits = torch.cat(all_gate_logits, dim=0)
        else:
            expert_indices = None
            gate_logits = None

        traj_ids: torch.Tensor | None
        traj_positions: torch.Tensor | None
        if has_trajectory_ids:
            traj_ids = torch.cat(all_traj_ids, dim=0)
            traj_positions = torch.cat(all_traj_positions, dim=0)
        else:
            traj_ids = None
            traj_positions = None

        # 3. Compute Metrics
        # (results already contains eval/action_mse from step 2c)

        # Task Metrics.  Offline action agreement is deliberately not called
        # "success": true task success requires a closed-loop rollout.
        task_cfg = self.metrics_cfg.get("task", {})
        action_threshold_cfg = task_cfg.get("action_l2_threshold_rate")
        # Read the legacy config key so old experiment files remain runnable,
        # but always emit the unambiguous result key below.
        if action_threshold_cfg is None:
            action_threshold_cfg = task_cfg.get("success_rate")
        if action_threshold_cfg is not None and bool(action_threshold_cfg.get("enabled", False)):
            threshold = action_threshold_cfg.get("l2_threshold", 0.05)
            results["eval/action_l2_threshold_rate"] = task_metrics.action_l2_threshold_rate(
                action_preds, action_targets, threshold
            )
            logger.info(
                "  eval/action_l2_threshold_rate (L2 <= %.2f): %.4f",
                threshold,
                results["eval/action_l2_threshold_rate"],
            )

        boundary_key = "boundary_action_smoothness"
        if self._metric_enabled("task", boundary_key):
            window = self.metrics_cfg.task.boundary_action_smoothness.boundary_window
            val = self._boundary_smoothness_trajectory_wise(
                action_preds, phases, window, traj_ids, traj_positions
            )
            if val is not None:
                results["eval/boundary_action_smoothness"] = val

        # Phase-head error bucketed by distance to the nearest phase
        # transition (diagnostic: boundary label noise vs uniform auxiliary
        # overfitting). Per-trajectory temporal metric, gated on the
        # task.phase_boundary_error config (off by default).
        if boundary_error_enabled and all_phase_logits:
            phase_logits = torch.cat(all_phase_logits, dim=0)
            bucket_edges = self.metrics_cfg.task.phase_boundary_error.get(
                "bucket_edges", (1, 3, 6, 11)
            )
            pooled = self._phase_boundary_error_trajectory_wise(
                phase_logits, phases, traj_ids, traj_positions, bucket_edges=bucket_edges
            )
            if pooled is not None:
                results.update({f"eval/phase_err_{k}": v for k, v in pooled.items()})

        # MoE Metrics
        if is_moe:
            # mypy narrowing + runtime guard: these are only set when is_moe.
            assert expert_indices is not None and gate_logits is not None
            num_experts = gate_logits.size(-1)

            if self._metric_enabled("mechanism", "routing_entropy"):
                results["eval/routing_entropy"] = routing_stability.routing_entropy(
                    gate_logits, normalize=True
                ).item()

            # Temporal routing metrics: per-trajectory, never across boundaries.
            if self._metric_enabled("mechanism", "routing_entropy_variance"):
                window = self.metrics_cfg.mechanism.routing_entropy_variance.get("window_size", 100)
                val = self._per_trajectory_routing_variance(
                    gate_logits, window, traj_ids, traj_positions
                )
                if val is not None:
                    results["eval/routing_entropy_variance"] = val

            if self._metric_enabled("mechanism", "time_to_stable_routing"):
                window = self.metrics_cfg.mechanism.time_to_stable_routing.get("window_size", 100)
                var_threshold = self.metrics_cfg.mechanism.time_to_stable_routing.get(
                    "variance_threshold", 0.001
                )
                consecutive = self.metrics_cfg.mechanism.time_to_stable_routing.get(
                    "consecutive_windows", 5
                )
                val, stability_fraction = self._per_trajectory_time_to_stable(
                    gate_logits,
                    window,
                    var_threshold,
                    consecutive,
                    traj_ids,
                    traj_positions,
                )
                if stability_fraction is not None:
                    results["eval/routing_stability_fraction"] = stability_fraction
                if val is not None:
                    results["eval/time_to_stable_routing"] = val

            if self._metric_enabled("mechanism", "expert_utilization"):
                topk_fractions = expert_utilization.expert_utilization(expert_indices, num_experts)
                top1_fractions = expert_utilization.expert_utilization_top1(
                    expert_indices, num_experts
                )

                results["eval/topk_balance_score"] = expert_utilization.expert_utilization_balance(
                    topk_fractions
                )
                results["eval/top1_balance_score"] = expert_utilization.expert_utilization_balance(
                    top1_fractions
                )

                if self._metric_enabled("mechanism", "collapse_rate"):
                    factor = self.metrics_cfg.mechanism.collapse_rate.threshold_factor
                    results["eval/topk_collapse_rate"] = expert_utilization.collapse_rate(
                        topk_fractions, factor
                    )
                    results["eval/top1_collapse_rate"] = expert_utilization.collapse_rate(
                        top1_fractions, factor
                    )

            if self._metric_enabled("mechanism", "phase_expert_nmi"):
                results["eval/phase_expert_nmi"] = phase_alignment.phase_expert_nmi(
                    phases, expert_indices
                )

            # Direct Expert Specialization & Behavioral Diagnostics (M_{z,e} matrix)
            try:
                from phaseforge.evaluations.metrics.specialization import (
                    compute_specialization_matrix,
                )

                spec_data = compute_specialization_matrix(
                    self.model, self.dataloader, device=self.device
                )
                if spec_data:
                    results["eval/expert_phase_mse_matrix"] = (
                        spec_data["expert_phase_mse_matrix"]
                    )
                    results["eval/best_expert_per_phase"] = (
                        spec_data["best_expert_per_phase"]
                    )
                    results["eval/theoretical_best_phase_mse"] = (
                        spec_data["theoretical_best_phase_mse"]
                    )
                    results["eval/routed_phase_mse"] = (
                        spec_data["routed_phase_mse"]
                    )
                    results["eval/theoretical_mean_mse"] = (
                        spec_data["theoretical_mean_mse"]
                    )
                    results["eval/expert_pairwise_divergence"] = (
                        spec_data["expert_pairwise_divergence"]
                    )
                    results["eval/mean_expert_divergence"] = (
                        spec_data["mean_expert_divergence"]
                    )
                    results["eval/phase_expert_contingency"] = (
                        spec_data["phase_expert_contingency"]
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not compute specialization matrix: %s", exc)

        # 4. Metric-definition record: every reported routing metric is
        #    explicitly labeled so results cannot be misread downstream.
        results["eval/metric_definitions"] = {
            "eval/routing_entropy": (
                "Shannon entropy of the PRE-TOP-K softmax gate probabilities "
                "over all experts (normalized by log(E)); 1 = uniform, 0 = "
                "fully peaked."
            ),
            "eval/routing_entropy_variance": (
                "Mean over trajectories of the within-window variance of "
                "per-step pre-top-k routing entropy; computed per trajectory "
                "from collator trajectory_id, never across episode boundaries."
            ),
            "eval/time_to_stable_routing": (
                "Mean one-based observed-step count among trajectories that reach the requested "
                "stability run; trajectories that do not reach it are right-"
                "censored and summarized separately by routing_stability_fraction."
            ),
            "eval/routing_stability_fraction": (
                "Fraction of trajectories for which the requested consecutive "
                "stable windows were observed before the trajectory ended."
            ),
            "eval/topk_balance_score": (
                "Normalized entropy of expert usage counted over ALL top-k "
                "routing assignments; 1 = uniform usage."
            ),
            "eval/top1_balance_score": (
                "Normalized entropy of expert usage counted from only the "
                "top-1 assignment per item; 1 = uniform usage."
            ),
            "eval/topk_collapse_rate": (
                "Fraction of experts below the collapse threshold using ALL top-k assignments."
            ),
            "eval/top1_collapse_rate": (
                "Fraction of experts below the collapse threshold using only top-1 assignments."
            ),
            "eval/action_l2_threshold_rate": (
                "Fraction of individual action vectors within the configured "
                "L2 threshold of the demonstration action; offline action "
                "agreement only, NOT rollout task success."
            ),
            "eval/phase_expert_nmi": (
                "NMI between phase labels and the TOP-1 routing assignment "
                "(top-1 only; distinct from the top-k balance counts)."
            ),
            "eval/expert_phase_mse_matrix": (
                "P x E behavioral matrix M_{z,e} = MSE(pi_e(x_z), a_z) running "
                "each expert e independently across validation samples of phase z."
            ),
            "eval/mean_expert_divergence": (
                "Mean pairwise L2 distance between individual expert outputs on shared inputs."
            ),
            "eval/theoretical_mean_mse": (
                "Mean MSE over phases under optimal theoretical phase-to-expert selection e*(z)."
            ),
            "eval/routed_mean_mse": (
                "Mean MSE over phases of the actual routed policy (per-phase routed MSE averaged)."
            ),
            "eval/theoretical_best_phase_mse": (
                "Per-phase dict: MSE of the best expert e*(z) = argmin_e M_{z,e} on validation "
                "samples of phase z (in-sample selection; diagnostic only, not out-of-sample)."
            ),
            "eval/best_expert_per_phase": (
                "Per-phase dict: expert index minimizing M_{z,e} on validation samples of phase z."
            ),
            "eval/routed_phase_mse": (
                "Per-phase dict: MSE of the routed policy on validation samples of phase z."
            ),
            "eval/phase_expert_contingency": (
                "P x E matrix: P(top-1 expert = e | phase = z) normalized per phase row."
            ),
            "eval/boundary_action_smoothness": (
                "Mean L2 temporal change of PREDICTED actions in a window "
                "around phase transitions (NOT an error vs target actions); "
                "computed per trajectory."
            ),
            "eval/phase_err_dist_*": (
                "Phase-head classification error rate restricted to timesteps "
                "whose distance to the nearest phase transition falls in the "
                "named interval [lo, hi) (final bucket open-ended); computed "
                "per trajectory and pooled across trajectories. Diagnoses "
                "boundary label noise (errors cluster near transitions) vs "
                "uniform auxiliary-head overfitting. Gated on "
                "task.phase_boundary_error (off by default)."
            ),
        }

        # 5. Honest skip record: temporal metrics that COULD not be computed
        #    because the loader carried no trajectory identity.
        if traj_ids is None:
            skipped = [
                name
                for name in (
                    "routing_entropy_variance",
                    "time_to_stable_routing",
                    "boundary_action_smoothness",
                )
                if name in results or self._enabled_any(name)
            ]
            if skipped:
                logger.warning(
                    "Dataloader provides no trajectory_id/trajectory_position "
                    "keys — trajectory-based metrics SKIPPED: %s. Use "
                    "StateOnlyDataset + PhaseAwareCollator for offline eval.",
                    skipped,
                )
                results["eval/skipped_metrics"] = skipped

        return results

    def _enabled_any(self, name: str) -> bool:
        if name == "boundary_action_smoothness":
            return self._metric_enabled("task", name)
        return self._metric_enabled("mechanism", name)

    def _trajectory_sequences(
        self,
        values: torch.Tensor,
        traj_ids: torch.Tensor | None,
        traj_positions: torch.Tensor | None,
    ) -> list[torch.Tensor] | None:
        """Per-trajectory row-sorted sequences, or None when ids are absent."""
        if traj_ids is None or traj_positions is None:
            return None
        return _regroup_by_trajectory(traj_ids, traj_positions, values)

    def _per_trajectory_routing_variance(
        self,
        gate_logits: torch.Tensor,
        window: int,
        traj_ids: torch.Tensor | None,
        traj_positions: torch.Tensor | None,
    ) -> float | None:
        sequences = self._trajectory_sequences(gate_logits, traj_ids, traj_positions)
        if sequences is None:
            return None
        variances = [
            routing_stability.routing_entropy_variance(seq, window_size=window) for seq in sequences
        ]
        return float(torch.stack(variances).mean().item())

    def _per_trajectory_time_to_stable(
        self,
        gate_logits: torch.Tensor,
        window: int,
        var_threshold: float,
        consecutive: int,
        traj_ids: torch.Tensor | None,
        traj_positions: torch.Tensor | None,
    ) -> tuple[float | None, float | None]:
        sequences = self._trajectory_sequences(gate_logits, traj_ids, traj_positions)
        if sequences is None:
            return None, None
        observations = [
            routing_stability.time_to_stable_routing_result(
                seq,
                window_size=window,
                variance_threshold=var_threshold,
                consecutive_windows=consecutive,
            )
            for seq in sequences
        ]
        stabilized_steps = [r.step for r in observations if r.stabilized]
        stability_fraction = len(stabilized_steps) / len(observations)
        if not stabilized_steps:
            return None, stability_fraction
        return float(sum(stabilized_steps) / len(stabilized_steps)), stability_fraction

    def _boundary_smoothness_trajectory_wise(
        self,
        action_preds: torch.Tensor,
        phases: torch.Tensor,
        window: int,
        traj_ids: torch.Tensor | None,
        traj_positions: torch.Tensor | None,
    ) -> float | None:
        """Per-trajectory boundary-action smoothness (mean of valid values).

        Requires trajectory identity; returns None (-> metric skipped) when
        the loader carries no trajectory keys.
        """
        sequences = self._trajectory_sequences(action_preds, traj_ids, traj_positions)
        phase_sequences = self._trajectory_sequences(phases, traj_ids, traj_positions)
        if sequences is None or phase_sequences is None:
            return None
        values = [
            task_metrics.boundary_action_smoothness(
                seq.unsqueeze(0), phase_seq.unsqueeze(0), window
            )
            for seq, phase_seq in zip(sequences, phase_sequences)
        ]
        valid = [v for v in values if not math.isnan(v)]
        if not valid:
            logger.warning(
                "boundary_action_smoothness: no phase boundaries found in any "
                "trajectory — metric omitted (not fabricated)."
            )
            return None
        return float(sum(valid) / len(valid))

    def _phase_boundary_error_trajectory_wise(
        self,
        phase_logits: torch.Tensor,
        phases: torch.Tensor,
        traj_ids: torch.Tensor | None,
        traj_positions: torch.Tensor | None,
        bucket_edges: tuple[int, ...] = (1, 3, 6, 11),
    ) -> dict[str, float] | None:
        """Phase-head error rate by distance-to-transition, pooled per bucket.

        Regroups the pooled phase logits/labels into per-trajectory, time-
        ordered sequences, computes the distance-bucketed error breakdown
        per trajectory (``phase_error_by_transition_distance``), then pools
        error rates across trajectories weighted by per-bucket sample counts.
        Returns ``None`` (-> metric skipped) when trajectory identity is
        absent or the logits are not per-step ``(N, C)``.
        """
        if phase_logits.ndim != 2:
            logger.warning(
                "phase_boundary_error: expected per-step (N, C) phase logits, "
                "got shape %s — metric omitted.",
                tuple(phase_logits.shape),
            )
            return None
        logit_sequences = self._trajectory_sequences(phase_logits, traj_ids, traj_positions)
        phase_sequences = self._trajectory_sequences(phases, traj_ids, traj_positions)
        if logit_sequences is None or phase_sequences is None:
            return None

        per_traj = [
            task_metrics.phase_error_by_transition_distance(
                logits, labels, bucket_edges=bucket_edges
            )
            for logits, labels in zip(logit_sequences, phase_sequences)
        ]
        if not per_traj:
            return None

        # Pool rates per bucket weighted by their per-trajectory sample
        # counts; per-bucket counts and totals are exact sums.
        pooled: dict[str, float] = {}
        bucket_keys = [k for k in per_traj[0] if k.startswith("dist_")]
        for key in bucket_keys:
            total_n = float(sum(d[f"n_{key}"] for d in per_traj))
            weighted_err = float(
                sum(d[key] * d[f"n_{key}"] for d in per_traj if d[f"n_{key}"] > 0)
            )
            pooled[key] = weighted_err / total_n if total_n > 0 else float("nan")
            pooled[f"n_{key}"] = total_n
        pooled["n_samples"] = float(sum(d["n_samples"] for d in per_traj))
        pooled["n_boundaries"] = float(sum(d["n_boundaries"] for d in per_traj))
        pooled["any_boundary"] = float(
            sum(d["any_boundary"] for d in per_traj) / len(per_traj)
        )
        return pooled

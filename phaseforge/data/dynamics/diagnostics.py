"""Quality check diagnostics for dynamics-aware phase discovery.

Implements the mandatory discovery quality checks before policy training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from phaseforge.data.dynamics.switching_linear import SingleDynamicsModel, StickySLDS


@dataclass
class DiscoveryQualityReport:
    """Structured report of discovery quality checks."""

    passed_all: bool
    num_regimes: int
    total_train_steps: int
    total_train_trajs: int
    occupancy: dict[int, float]  # regime_id -> fraction
    min_occupancy: float
    single_regime_trajs: int
    single_regime_fraction: float
    mean_switch_rate: float
    held_out_nll_slds: float
    held_out_nll_single_dynamics: float
    nll_improvement: float
    within_regime_residual_var: float
    within_rule_residual_var: float | None
    failure_reasons: list[str]
    # Observability-audit attachments. None when
    # the audit was not run (e.g. SLDS-only reports); populated by
    # the topo discovery path via `report.to_dict()`-compatible values.
    audit_macro_f1: float | None = None
    audit_merge_candidates: list[list[int]] = field(default_factory=list)


def evaluate_discovery_quality(
    slds: StickySLDS,
    train_trajectories: list[dict[str, Any]],
    val_trajectories: list[dict[str, Any]],
    min_occupancy_threshold: float = 0.02,
    max_single_regime_fraction: float = 0.5,
    max_switch_rate: float = 0.6,
    min_nll_improvement: float = 0.0,
    max_within_rule_residual_ratio: float | None = None,
) -> DiscoveryQualityReport:
    """Run all mandatory discovery quality checks on train and validation splits.

    Args:
        slds: Fitted StickySLDS instance.
        train_trajectories: List of training trajectory dicts.
        val_trajectories: List of validation trajectory dicts.
        min_occupancy_threshold: Minimum required fraction of steps per regime.
        max_single_regime_fraction: Maximum fraction of single-regime trajectories.
        max_switch_rate: Maximum allowed mean switch rate.
        min_nll_improvement: Minimum required positive NLL improvement over the
            single-dynamics baseline.
        max_within_rule_residual_ratio: Optional maximum ratio of dynamic to
            rule-label residual variance. ``None`` disables this comparison.

    Returns:
        DiscoveryQualityReport summarizing test outcomes.
    """
    assert slds.params is not None
    K = slds.num_regimes
    failures: list[str] = []

    # 1. Decode training trajectories
    decoded_train_labels: list[np.ndarray] = [
        slds.decode_trajectory(traj) for traj in train_trajectories
    ]
    all_train_labels = np.concatenate(decoded_train_labels)
    total_steps = len(all_train_labels)

    if not train_trajectories:
        raise ValueError("Discovery quality checks require at least one training trajectory.")

    # Check 1: Minimum occupancy per regime
    counts = np.bincount(all_train_labels, minlength=K)
    occupancy = {k: float(counts[k] / total_steps) for k in range(K)}
    min_occ = min(occupancy.values())

    if min_occ < min_occupancy_threshold:
        failures.append(
            f"Regime collapse: min occupancy {min_occ:.2%} is below "
            f"threshold {min_occupancy_threshold:.2%}. "
            f"Occupancies: {occupancy}"
        )

    # Check 2: No excessive single-regime trajectories
    single_regime_count = 0
    total_switches = 0
    for labels in decoded_train_labels:
        unique_k = np.unique(labels)
        if len(unique_k) == 1:
            single_regime_count += 1
        total_switches += int(np.sum(labels[1:] != labels[:-1]))

    single_frac = single_regime_count / len(train_trajectories)
    if single_frac > max_single_regime_fraction:
        failures.append(
            f"Over-persistence: {single_regime_count}/{len(train_trajectories)} "
            f"({single_frac:.1%}) trajectories were assigned a single regime "
            f"for their entire duration "
            f"(limit {max_single_regime_fraction:.1%})."
        )

    # Check 3: Plausible switch rate
    mean_switch_rate = total_switches / max(1, total_steps - len(train_trajectories))
    if mean_switch_rate > max_switch_rate:
        failures.append(
            f"High phase flickering: mean switch rate is {mean_switch_rate:.3f} "
            f"switches/step (limit {max_switch_rate:.3f})."
        )

    # Check 4: Out-of-sample NLL vs Single Dynamics Model on validation set
    single_model = SingleDynamicsModel().fit(train_trajectories)

    val_nll_slds = 0.0
    val_nll_single = 0.0
    val_steps = 0
    for trajectory in val_trajectories:
        transitions = max(0, len(trajectory["state"]) - 1)
        val_steps += transitions
        val_nll_slds -= slds.score_trajectory(trajectory)
        val_nll_single -= single_model.score_trajectory(trajectory)

    if val_steps == 0:
        raise ValueError(
            "Discovery quality checks require validation trajectories with transitions."
        )

    # Normalize by transition count so trajectory length cannot dominate the
    # comparison when validation demonstrations have different lengths.
    mean_nll_slds = float(val_nll_slds / val_steps)
    mean_nll_single = float(val_nll_single / val_steps)
    nll_diff = mean_nll_single - mean_nll_slds  # Positive means SLDS is better

    if not np.isfinite(mean_nll_slds):
        failures.append(f"SLDS validation NLL is non-finite: {mean_nll_slds}")
    elif not np.isfinite(mean_nll_single):
        failures.append(f"Single-dynamics validation NLL is non-finite: {mean_nll_single}")
    elif nll_diff < min_nll_improvement:
        failures.append(
            f"SLDS validation NLL improvement {nll_diff:.6f} is below the required "
            f"minimum {min_nll_improvement:.6f}."
        )

    # Check 5: Within-regime transition residual variance vs rule-based labels (if available)
    all_x_t = []
    all_a_t = []
    all_x_next = []
    all_rule_phases = []

    for traj in train_trajectories:
        s = traj["state"]
        a = traj["action"]
        p = traj.get("phase")
        if isinstance(s, torch.Tensor):
            s = s.cpu().numpy()
        if isinstance(a, torch.Tensor):
            a = a.cpu().numpy()
        if p is not None and isinstance(p, torch.Tensor):
            p = p.cpu().numpy()

        all_x_t.append(s[:-1])
        all_a_t.append(a[:-1])
        all_x_next.append(s[1:])
        if p is not None:
            all_rule_phases.append(p[:-1])

    x_t_cat = np.concatenate(all_x_t, axis=0)
    a_t_cat = np.concatenate(all_a_t, axis=0)
    x_next_cat = np.concatenate(all_x_next, axis=0)

    # Dynamic regime residuals
    dynamic_residuals = []
    labels_minus_1 = np.concatenate([lbl[:-1] for lbl in decoded_train_labels])

    for k in range(K):
        mask = labels_minus_1 == k
        if np.any(mask):
            pred = (
                x_t_cat[mask] @ slds.params.A[k].T
                + a_t_cat[mask] @ slds.params.B[k].T
                + slds.params.b[k]
            )
            res = x_next_cat[mask] - pred
            dynamic_residuals.append(np.mean(np.var(res, axis=0)))

    within_regime_res_var = float(np.mean(dynamic_residuals)) if dynamic_residuals else 0.0

    within_rule_res_var = None
    if all_rule_phases:
        rule_cat = np.concatenate(all_rule_phases, axis=0)
        rule_residuals = []
        for p_idx in np.unique(rule_cat):
            mask = rule_cat == p_idx
            if np.sum(mask) > 10:
                # Fit standard linear regression for rule phase
                u_p = np.concatenate(
                    [x_t_cat[mask], a_t_cat[mask], np.ones((int(np.sum(mask)), 1))], axis=-1
                )
                y_p = x_next_cat[mask]
                W_p = np.linalg.lstsq(u_p, y_p, rcond=None)[0]
                res_p = y_p - (u_p @ W_p)
                rule_residuals.append(np.mean(np.var(res_p, axis=0)))
        if rule_residuals:
            within_rule_res_var = float(np.mean(rule_residuals))

    if max_within_rule_residual_ratio is not None and within_rule_res_var is not None:
        if not np.isfinite(within_rule_res_var):
            failures.append("Within-rule residual variance is non-finite.")
        elif within_regime_res_var > within_rule_res_var * max_within_rule_residual_ratio:
            failures.append(
                "Dynamic within-regime residual variance "
                f"({within_regime_res_var:.6f}) exceeds the rule-label variance "
                f"({within_rule_res_var:.6f}) by the allowed ratio "
                f"{max_within_rule_residual_ratio:.3f}."
            )

    passed_all = len(failures) == 0

    return DiscoveryQualityReport(
        passed_all=passed_all,
        num_regimes=K,
        total_train_steps=total_steps,
        total_train_trajs=len(train_trajectories),
        occupancy=occupancy,
        min_occupancy=min_occ,
        single_regime_trajs=single_regime_count,
        single_regime_fraction=single_frac,
        mean_switch_rate=float(mean_switch_rate),
        held_out_nll_slds=mean_nll_slds,
        held_out_nll_single_dynamics=mean_nll_single,
        nll_improvement=float(nll_diff),
        within_regime_residual_var=within_regime_res_var,
        within_rule_residual_var=within_rule_res_var,
        failure_reasons=failures,
    )

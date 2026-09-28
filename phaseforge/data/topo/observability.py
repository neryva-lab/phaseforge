"""Mandatory observability audit for regime labels.

Before any discovered regime label trains the router, it must prove itself
inferable from the instantaneous state ``x_t`` alone — with a
trajectory-aware split (GroupKFold over trajectories, never a shuffled
split that leaks adjacent steps across folds).

Required checks:

* macro-F1 from ``x_t`` to regime (instantaneous inferability),
* confusion matrix (aliased regime pairs),
* per-regime duration distribution (pathological fragmentation),
* regime occupancy (dead regimes),
* action residual reduction (regimes explain behavior; optional, needs
  per-step actions).

Regime pairs that are strongly confused from ``x_t`` are reported as
merge candidates and must be merged or redefined — never routed on.
CPU-only (scikit-learn), deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ObservabilityReport:
    """Structured result of :func:`audit_regimes` (JSON-serializable via :meth:`to_dict`)."""

    passed: bool
    num_regimes: int
    num_samples: int
    num_trajectories: int
    macro_f1: float
    confusion: list[list[int]]
    occupancy: dict[int, float]
    min_occupancy: float
    mean_duration: float
    duration_per_regime: dict[int, float]
    action_residual_reduction: float | None
    merge_candidates: list[list[int]] = field(default_factory=list)
    failure_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": bool(self.passed),
            "num_regimes": int(self.num_regimes),
            "num_samples": int(self.num_samples),
            "num_trajectories": int(self.num_trajectories),
            "macro_f1": float(self.macro_f1),
            "confusion": [[int(v) for v in row] for row in self.confusion],
            "occupancy": {int(k): float(v) for k, v in self.occupancy.items()},
            "min_occupancy": float(self.min_occupancy),
            "mean_duration": float(self.mean_duration),
            "duration_per_regime": {int(k): float(v) for k, v in self.duration_per_regime.items()},
            "action_residual_reduction": (
                None if self.action_residual_reduction is None
                else float(self.action_residual_reduction)
            ),
            "merge_candidates": [[int(a) for a in pair] for pair in self.merge_candidates],
            "failure_reasons": [str(r) for r in self.failure_reasons],
        }


def _durations(labels: np.ndarray, traj_ids: np.ndarray) -> tuple[float, dict[int, float]]:
    """Mean run length overall and per regime (inputs in time order per traj)."""
    order = np.argsort(traj_ids, kind="stable")
    sorted_labels = labels[order]
    sorted_trajs = traj_ids[order]
    run_lengths: list[int] = []
    per_regime: dict[int, list[int]] = {}
    current_label = int(sorted_labels[0])
    current_traj = int(sorted_trajs[0])
    run = 1
    for idx in range(1, len(sorted_labels)):
        label = int(sorted_labels[idx])
        traj = int(sorted_trajs[idx])
        if traj != current_traj:
            run_lengths.append(run)
            per_regime.setdefault(current_label, []).append(run)
            current_label, current_traj, run = label, traj, 1
        elif label != current_label:
            run_lengths.append(run)
            per_regime.setdefault(current_label, []).append(run)
            current_label, run = label, 1
        else:
            run += 1
    run_lengths.append(run)
    per_regime.setdefault(current_label, []).append(run)
    mean_duration = float(np.mean(run_lengths))
    per_regime_mean = {k: float(np.mean(v)) for k, v in per_regime.items()}
    return mean_duration, per_regime_mean


def _action_reduction(
    actions: np.ndarray | None, labels: np.ndarray, num_regimes: int
) -> float | None:
    """Fractional action-variance reduction vs a single global head."""
    if actions is None:
        return None
    acts = np.asarray(actions, dtype=np.float64)
    if acts.ndim != 2 or acts.shape[0] != labels.shape[0]:
        raise ValueError("actions must have shape (N, Da) matching labels.")
    if not np.isfinite(acts).all():
        raise ValueError("Non-finite values in audit actions.")
    global_var = float(np.mean(np.var(acts, axis=0)))
    if global_var <= 0.0:
        return 0.0
    parts = []
    for regime in range(num_regimes):
        mask = labels == regime
        if np.sum(mask) > 1:
            parts.append(float(np.mean(np.var(acts[mask], axis=0))) * float(np.sum(mask)))
    if not parts:
        return 0.0
    within = float(np.sum(parts)) / float(labels.shape[0])
    return float((global_var - within) / global_var)


def audit_regimes(
    states: np.ndarray,
    labels: np.ndarray,
    traj_ids: np.ndarray,
    num_regimes: int,
    *,
    actions: np.ndarray | None = None,
    min_macro_f1: float = 0.6,
    min_occupancy: float = 0.01,
    merge_f1_threshold: float = 0.5,
    seed: int = 42,
) -> ObservabilityReport:
    """Audit whether regime labels are inferable from ``x_t`` alone.

    Args:
        states: Normalized instantaneous states, shape ``(N, S)``.
        labels: Discovered regime labels, shape ``(N,)`` in ``[0, K)``.
        traj_ids: Trajectory index per sample, shape ``(N,)``. Samples
            must appear in time order within each trajectory (duration
            statistics rely on it).
        num_regimes: Regime count K.
        actions: Optional per-step actions ``(N, Da)`` for the residual
            reduction check.
        min_macro_f1: Pass threshold on cross-validated macro-F1.
        min_occupancy: Minimum allowed per-regime occupancy fraction.
        merge_f1_threshold: Regime pairs whose pairwise F1 falls below
            this become merge candidates.
        seed: Deterministic seed for the probe classifier.

    Returns:
        :class:`ObservabilityReport` (``passed`` is False with populated
        ``failure_reasons`` when any gate fails).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix, f1_score
    from sklearn.model_selection import GroupKFold

    states_arr = np.asarray(states, dtype=np.float64)
    labels_arr = np.asarray(labels).reshape(-1)
    traj_arr = np.asarray(traj_ids).reshape(-1)
    regimes = int(num_regimes)
    if regimes < 2:
        raise ValueError(f"num_regimes must be >= 2, got {regimes}.")
    if states_arr.ndim != 2:
        raise ValueError(f"Expected states shape (N, S), got {states_arr.shape}.")
    count = states_arr.shape[0]
    if labels_arr.shape[0] != count or traj_arr.shape[0] != count:
        raise ValueError("states, labels, and traj_ids must have equal length.")
    if count == 0:
        raise ValueError("Cannot audit an empty sample set.")
    if not np.isfinite(states_arr).all():
        raise ValueError("Non-finite values in audit states.")
    if np.any(labels_arr < 0) or np.any(labels_arr >= regimes):
        raise ValueError("Audit labels are out of range.")
    unique_trajs = np.unique(traj_arr)
    failures: list[str] = []

    counts = np.bincount(labels_arr, minlength=regimes)
    occupancy = {k: float(counts[k] / count) for k in range(regimes)}
    min_occ = min(occupancy.values())
    if min_occ < min_occupancy:
        failures.append(
            f"Dead regime: min occupancy {min_occ:.4f} is below {min_occupancy:.4f}."
        )

    # Trajectory-aware cross-validation: groups are trajectories, so no
    # adjacent-step leakage across folds. Few trajectories -> fewer splits.
    n_groups = len(unique_trajs)
    n_splits = int(min(3, n_groups))
    if n_splits < 2:
        predicted = np.full(count, -1, dtype=np.int64)
        # Degenerate single-trajectory input: resubstitution probe only.
        probe = LogisticRegression(max_iter=500, random_state=int(seed))
        probe.fit(states_arr, labels_arr)
        predicted = np.asarray(probe.predict(states_arr), dtype=np.int64)
    else:
        predicted = np.full(count, -1, dtype=np.int64)
        splitter = GroupKFold(n_splits=n_splits)
        for train_idx, test_idx in splitter.split(states_arr, labels_arr, traj_arr):
            probe = LogisticRegression(max_iter=500, random_state=int(seed))
            probe.fit(states_arr[train_idx], labels_arr[train_idx])
            predicted[test_idx] = np.asarray(probe.predict(states_arr[test_idx]))
    macro_f1 = float(f1_score(labels_arr, predicted, average="macro", zero_division=0))
    if macro_f1 < min_macro_f1:
        failures.append(
            f"Unobservable regimes: macro-F1 {macro_f1:.4f} is below {min_macro_f1:.4f}."
        )
    conf = confusion_matrix(labels_arr, predicted, labels=list(range(regimes)))
    confusion = [[int(v) for v in row] for row in conf.tolist()]

    # Pairwise confusion -> merge candidates:
    # A pair (a, b) is an aliasing candidate only if there is direct mutual confusion
    # between them (samples of a predicted as b or vice versa) and the pairwise F1
    # on the restricted decision {a, b} drops below threshold with significant mutual confusion.
    merge_candidates: list[list[int]] = []
    for first in range(regimes):
        for second in range(first + 1, regimes):
            n_first = counts[first]
            n_second = counts[second]
            if n_first == 0 or n_second == 0:
                continue
            misclass_first_as_second = conf[first, second]
            misclass_second_as_first = conf[second, first]
            mutual_misclass = misclass_first_as_second + misclass_second_as_first
            if mutual_misclass == 0:
                continue
            pair_mask = ((labels_arr == first) | (labels_arr == second)) & (
                (predicted == first) | (predicted == second)
            )
            if np.sum(pair_mask) > 0:
                pair_f1 = float(
                    f1_score(
                        labels_arr[pair_mask],
                        predicted[pair_mask],
                        labels=[first, second],
                        average="macro",
                        zero_division=0,
                    )
                )
                confusion_rate = mutual_misclass / float(n_first + n_second)
                if pair_f1 < merge_f1_threshold and confusion_rate > 0.15:
                    merge_candidates.append([first, second])
    if merge_candidates:
        failures.append(
            "Aliased regime pairs (merge or redefine): "
            + ", ".join(f"({a},{b})" for a, b in merge_candidates)
            + "."
        )

    mean_duration, duration_per_regime = _durations(labels_arr, traj_arr)
    full_durations = {k: float(duration_per_regime.get(k, 0.0)) for k in range(regimes)}
    reduction = _action_reduction(actions, labels_arr, regimes)

    return ObservabilityReport(
        passed=not failures,
        num_regimes=regimes,
        num_samples=int(count),
        num_trajectories=int(n_groups),
        macro_f1=macro_f1,
        confusion=confusion,
        occupancy=occupancy,
        min_occupancy=float(min_occ),
        mean_duration=mean_duration,
        duration_per_regime=full_durations,
        action_residual_reduction=reduction,
        merge_candidates=merge_candidates,
        failure_reasons=failures,
    )


__all__ = ["ObservabilityReport", "audit_regimes"]

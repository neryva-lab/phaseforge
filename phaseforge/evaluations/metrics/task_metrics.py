"""Task-level offline performance metrics."""

from __future__ import annotations

import torch
from torch import Tensor


def action_l2_threshold_rate(
    predicted_actions: Tensor, target_actions: Tensor, l2_threshold: float = 0.05
) -> float:
    """Fraction of individual action vectors within an L2 error threshold.

    This is an offline action-reproduction diagnostic, not task success. True
    success requires a closed-loop environment rollout and must be reported by
    the rollout evaluator under a separate metric name.

    Args:
        predicted_actions: Tensor of shape (B, A) or (B, T, A).
        target_actions: Tensor of shape (B, A) or (B, T, A).
        l2_threshold: Maximum allowed L2 distance to be considered "successful".

    Returns:
        Float action-threshold rate in [0, 1].
    """
    if predicted_actions.numel() == 0:
        return 0.0

    l2_errors = torch.norm(predicted_actions - target_actions, p=2, dim=-1)

    successes = (l2_errors <= l2_threshold).sum().item()
    total = l2_errors.numel()

    return successes / total


def success_rate(
    predicted_actions: Tensor, target_actions: Tensor, l2_threshold: float = 0.05
) -> float:
    """Deprecated compatibility alias for :func:`action_l2_threshold_rate`.

    The old name was misleading because offline action agreement is not task
    success. New evaluation code must use ``action_l2_threshold_rate``.
    """
    return action_l2_threshold_rate(predicted_actions, target_actions, l2_threshold)


def action_mse(
    predicted_actions: Tensor, target_actions: Tensor, mask: Tensor | None = None
) -> float:
    """Mean squared error between predicted and target actions.

    The single most informative offline diagnostic: if a policy that scores
    0% rollout success cannot even reproduce the demonstration actions
    (high MSE), the rollout result reflects model capability rather than an
    evaluation bug.

    Args:
        predicted_actions: (B, A) or (B, T, A).
        target_actions: Same shape as ``predicted_actions``.
        mask: Optional (B, T) (or (B,)) boolean/0-1 padding mask. When given,
            only the masked-in entries contribute (variable-length batches).

    Returns:
        Scalar MSE over the (masked) action entries, or ``float('nan')`` when
        no entries remain.
    """
    if mask is not None:
        m = mask.bool()
        while m.ndim < predicted_actions.ndim:
            m = m.unsqueeze(-1)
        m = m.expand_as(predicted_actions)
        predicted_actions = predicted_actions[m]
        target_actions = target_actions[m]

    if predicted_actions.numel() == 0:
        return float("nan")
    return float(((predicted_actions - target_actions) ** 2).mean())


def phase_error_by_transition_distance(
    logits: Tensor,
    labels: Tensor,
    bucket_edges: tuple[int, ...] = (1, 3, 6, 11),
) -> dict[str, float]:
    """Phase-head classification error rate bucketed by distance to transition.

    Distinguishes *boundary label noise* (errors cluster at the labeler's
    decision boundaries) from *uniform auxiliary overfitting* (errors spread
    across all distances). Per-trajectory temporal metric: transitions are
    detected from the label stream itself, and each timestep is assigned the
    distance to its nearest transition (0 = the transition timestep itself).

    Args:
        logits: (T, C) phase-head logits for ONE trajectory.
        labels: (T,) integer phase labels for the same trajectory.
        bucket_edges: ascending exclusive upper edges for the distance
            buckets. Bucket i covers ``[lo, hi)`` with ``lo = 0`` for the
            first bucket and ``lo = edges[i-1]`` afterwards; the final bucket
            is ``[edges[-1], inf)``. Defaults ``(1, 3, 6, 11)`` produce
            distances 0 / 1-2 / 3-5 / 6-10 / 11+.

    Returns:
        A flat dict with one ``dist_<lo>_<hi>`` error-rate key per bucket
        (``dist_11_plus`` for the open-ended bucket), matching ``n_dist_*``
        per-bucket sample counts, plus ``n_samples`` (valid timesteps),
        ``n_boundaries`` (detected transitions) and ``any_boundary``
        (1.0/0.0). When a trajectory has no transitions every step falls
        into the final bucket.
    """
    if logits.ndim != 2 or labels.ndim != 1:
        raise ValueError(
            "phase_error_by_transition_distance expects (T, C) logits and "
            f"(T,) labels, got {tuple(logits.shape)} and {tuple(labels.shape)}"
        )
    if logits.size(0) != labels.size(0):
        raise ValueError("logits and labels must cover the same timesteps")
    if not logits.numel() or not labels.numel():
        raise ValueError("logits/labels must not be empty")
    if not torch.isfinite(logits).all():
        raise ValueError("logits contains non-finite values")
    edges = tuple(int(e) for e in bucket_edges)
    if not edges or edges != tuple(sorted(edges)) or edges[0] < 1:
        raise ValueError(
            f"bucket_edges must be a strictly ascending tuple of ints >= 1, got {bucket_edges}"
        )

    T = logits.size(0)
    preds = logits.argmax(dim=-1)
    errors = (preds != labels).float()  # (T,)

    # Transition positions: first timestep of each new phase segment.
    changed = labels[1:] != labels[:-1]
    boundary_positions = torch.nonzero(changed, as_tuple=False).flatten() + 1  # (K,)

    if boundary_positions.numel():
        positions = boundary_positions.unsqueeze(0)  # (1, K)
        steps = torch.arange(T, device=logits.device).unsqueeze(1)  # (T, 1)
        distances = (steps - positions).abs().min(dim=1).values  # (T,)
    else:
        distances = torch.full((T,), float("inf"), device=logits.device)

    # Bucket assignment: [0, e0) -> 0, [e0, e1) -> 1, ..., [e_last, inf) -> K.
    # right=True keeps the buckets LEFT-closed / RIGHT-open, so distance 0 is
    # isolated in its own bucket and distance e0 falls into the next bucket.
    edges_t = torch.tensor(edges, device=logits.device)
    bucket_idx = torch.bucketize(distances, edges_t, right=True).clamp(max=len(edges))

    out: dict[str, float] = {}
    for i in range(len(edges) + 1):
        lo = 0 if i == 0 else edges[i - 1]
        hi = "plus" if i == len(edges) else edges[i]
        bucket_mask = bucket_idx == i
        n = int(bucket_mask.sum().item())
        rate = float(errors[bucket_mask].mean().item()) if n else float("nan")
        out[f"dist_{lo}_{hi}"] = rate
        out[f"n_dist_{lo}_{hi}"] = float(n)
    out["n_samples"] = float(T)
    out["n_boundaries"] = float(boundary_positions.numel())
    out["any_boundary"] = 1.0 if boundary_positions.numel() else 0.0
    return out


def boundary_action_smoothness(
    predicted_actions: Tensor, phases: Tensor, boundary_window: int = 3
) -> float:
    """Measure the temporal smoothness of predicted actions at phase boundaries.

    .. note::
       Despite its historical name ("boundary smoothness"), this metric
       does NOT measure an error against target actions. It reports the
       mean per-step temporal change (L2 of ``pred[t] - pred[t-1]``) of
       the *predicted* actions restricted to a window around detected
       phase transitions. A target-based boundary error requires
       trajectory-aligned targets and is not implemented.

    Phase transitions are the hardest parts of long-horizon tasks. This metric
    isolates the temporal change of the predicted actions immediately
    surrounding a phase boundary.

    Args:
        predicted_actions: Tensor of shape (B, T, A).
        phases: Tensor of shape (B, T).
        boundary_window: Number of timesteps before and after the boundary to include.

    Returns:
        Mean L2 temporal difference at the boundaries, or float('nan') if no
        boundaries exist.

    Raises:
        ValueError: If ``boundary_window < 0``, inputs are empty, or the
            tensors contain non-finite values.
    """
    if int(boundary_window) < 0:
        raise ValueError(f"boundary_window must be >= 0, got {boundary_window}")
    if predicted_actions.ndim != 3 or phases.ndim != 2:
        # Require sequence dimension to detect boundaries
        return float("nan")
    if predicted_actions.numel() == 0 or phases.numel() == 0:
        raise ValueError("predicted_actions/phases must not be empty")
    if not torch.isfinite(predicted_actions).all():
        raise ValueError("predicted_actions contains non-finite values")
    if not torch.isfinite(phases.to(torch.float32)).all():
        raise ValueError("phases contains non-finite values")

    B, T, _ = predicted_actions.shape
    if T < 2:
        return float("nan")

    # mask is (B, T-1)
    transitions = phases[:, 1:] != phases[:, :-1]

    boundary_mask = torch.zeros((B, T), dtype=torch.bool, device=predicted_actions.device)

    has_boundary = False

    for b in range(B):
        # Indices where a transition occurs (offset by 1 because diff)
        transition_idxs = torch.where(transitions[b])[0] + 1

        for t_idx in transition_idxs:
            has_boundary = True
            start = max(0, int(t_idx.item()) - boundary_window)
            end = min(T, int(t_idx.item()) + boundary_window + 1)
            boundary_mask[b, start:end] = True

    if not has_boundary:
        return float("nan")

    # We don't have the ground truth actions here directly to compute error against,
    # so we measure the smoothness (temporal difference) of the predicted actions
    # at the boundaries.
    # High smoothness means the model doesn't jerk violently at transitions.

    diffs = predicted_actions[:, 1:] - predicted_actions[:, :-1]
    diffs_l2 = torch.norm(diffs, p=2, dim=-1)  # (B, T-1)

    # Apply mask (excluding the last element which we diff'd against)
    valid_diffs = diffs_l2[boundary_mask[:, :-1]]

    if valid_diffs.numel() == 0:
        return float("nan")

    return valid_diffs.mean().item()

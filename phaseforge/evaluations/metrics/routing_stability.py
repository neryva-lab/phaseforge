"""Routing stability metrics: entropy, variance, and convergence.

Terminology: all entropy values here are computed on
the **pre-top-k softmax gate probabilities** — i.e. the full softmax over
all ``E`` experts before any top-k selection. They describe the router's
certainty, not the entropy of the discrete top-k assignment.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor


class RoutingStabilityResult(NamedTuple):
    """Result of a per-trajectory routing-stability analysis.

    ``step`` is the number of observed timesteps at which the requested
    stability run is complete, or the trajectory length when the run is not
    observed. The
    explicit ``stabilized`` flag is essential: a trajectory ending at its
    horizon is censored, not evidence that stability was reached at the last
    step.
    """

    step: int
    stabilized: bool


def _validate_logits(gate_logits: Tensor, name: str = "gate_logits") -> None:
    """Reject empty or non-finite routing tensors."""
    if gate_logits.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if not torch.isfinite(gate_logits).all():
        raise ValueError(f"{name} contains non-finite values")


def routing_entropy(gate_logits: Tensor, normalize: bool = True) -> Tensor:
    """Calculate the Shannon entropy of the routing distribution.

    Entropy measures how "certain" the router is. It is computed on the
    **pre-top-k** softmax probabilities over all experts (see module
    docstring).
    High entropy = uniform routing (uncertain).
    Low entropy = peaked routing (certain).

    Args:
        gate_logits: Raw gating logits of shape (..., E).
        normalize: If True, divides by log(E) so output is in [0, 1].

    Returns:
        Scalar tensor containing the mean entropy across the batch.

    Raises:
        ValueError: If ``gate_logits`` is empty or non-finite.
    """
    _validate_logits(gate_logits)
    E = gate_logits.size(-1)
    if E < 1:
        raise ValueError(f"gate_logits must have at least 1 expert, got E={E}")

    probs = F.softmax(gate_logits, dim=-1)

    # log_probs, clamped for numerical stability
    log_probs = torch.log(probs.clamp(min=1e-8))

    # Entropy: -sum(p * log(p)) over experts
    entropy = -torch.sum(probs * log_probs, dim=-1)  # (...,)

    mean_entropy = entropy.mean()

    if normalize and E > 1:
        denom = torch.log(torch.tensor(E, dtype=torch.float32, device=gate_logits.device))
        mean_entropy = mean_entropy / denom

    return mean_entropy


def routing_switch_rate(
    top1_indices: Tensor, trajectory_id: Tensor, trajectory_position: Tensor
) -> float:
    """Fraction of adjacent in-trajectory step pairs whose top-1 expert changes.

    A pair is adjacent iff the samples share a ``trajectory_id`` and the
    later ``trajectory_position`` is exactly the earlier one plus one — the
    same in-batch two-pass rule the V2-C router uses to resolve previous
    steps (samples from different batches are never paired). Returns 0.0
    when no adjacent pairs exist.

    Args:
        top1_indices: (N,) long tensor of per-sample top-1 expert choices.
        trajectory_id: (N,) long tensor identifying each sample's trajectory.
        trajectory_position: (N,) long tensor of step positions.
    """
    if top1_indices.numel() == 0:
        return 0.0
    order = torch.argsort(
        trajectory_id * (trajectory_position.max() + 1) + trajectory_position,
        stable=True,
    )
    sorted_ids = trajectory_id[order]
    sorted_pos = trajectory_position[order]
    sorted_top1 = top1_indices[order]
    adjacent = (sorted_ids[1:] == sorted_ids[:-1]) & (
        sorted_pos[1:] == sorted_pos[:-1] + 1
    )
    if not adjacent.any():
        return 0.0
    switches = (sorted_top1[1:] != sorted_top1[:-1])[adjacent]
    return float(switches.float().mean().item())


def routing_entropy_variance(gate_logits: Tensor, window_size: int = 100) -> Tensor:
    """Mean variance of per-step routing entropy over sliding windows.

    Measures how much the router's certainty fluctuates over time. A low
    value indicates stable routing; a high value indicates flickering.

    .. note::
       Temporal metrics are only meaningful WITHIN a single trajectory;
       computing them across task/trajectory boundaries merges unrelated
       step streams. Callers must group ``gate_logits`` by trajectory
       (the offline evaluator uses the collator's ``trajectory_id``).

    Args:
        gate_logits: (T, E) gating logits over the (flattened) step sequence.
        window_size: Sliding-window length used to compute local variance.

    Returns:
        Scalar tensor: mean over all windows of the within-window variance
        of per-step routing entropy. Falls back to the full-sequence
        variance when the sequence is shorter than ``window_size``.

    Raises:
        ValueError: If ``window_size < 1`` or ``gate_logits`` is empty or
            non-finite.
    """
    if int(window_size) < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")
    _validate_logits(gate_logits)
    probs = F.softmax(gate_logits, dim=-1)
    log_probs = torch.log(probs.clamp(min=1e-8))
    entropy = -(probs * log_probs).sum(dim=-1)  # (T,)

    if entropy.numel() < 2:
        return torch.tensor(0.0, dtype=torch.float32, device=gate_logits.device)
    if entropy.numel() <= window_size:
        return entropy.var()

    windows = entropy.unfold(0, window_size, 1)  # (T - window_size + 1, window_size)
    return windows.var(dim=-1).mean()


def time_to_stable_routing(
    gate_logits: Tensor,
    window_size: int = 100,
    variance_threshold: float = 0.001,
    consecutive_windows: int = 5,
) -> Tensor:
    """Step index at which routing first becomes stable.

    Routing is considered stable when the variance of per-step routing
    entropy within a ``window_size`` window stays below
    ``variance_threshold`` for ``consecutive_windows`` consecutive windows.

    .. note::
       Like :func:`routing_entropy_variance`, this is a within-trajectory
       temporal metric — see its note about trajectory grouping.

    Args:
        gate_logits: (T, E) gating logits over the (flattened) step sequence.
        window_size: Sliding-window length for local entropy variance.
        variance_threshold: Maximum within-window entropy variance for a
            window to count as stable.
        consecutive_windows: Number of consecutive stable windows required.

    Returns:
        Int tensor: one-based observed-step count at the end of the window
        at which stability was reached. Returns ``T`` (the sequence length) if routing never
        becomes stable within the available steps. Use
        :func:`time_to_stable_routing_result` when the caller must distinguish
        a true stabilization from right-censoring at the trajectory horizon.

    Raises:
        ValueError: If ``window_size < 1``, ``consecutive_windows < 1``,
            ``variance_threshold < 0``, or ``gate_logits`` is empty /
            non-finite.
    """
    result = time_to_stable_routing_result(
        gate_logits,
        window_size=window_size,
        variance_threshold=variance_threshold,
        consecutive_windows=consecutive_windows,
    )
    return torch.tensor(result.step, dtype=torch.int64, device=gate_logits.device)


def time_to_stable_routing_result(
    gate_logits: Tensor,
    window_size: int = 100,
    variance_threshold: float = 0.001,
    consecutive_windows: int = 5,
) -> RoutingStabilityResult:
    """Return routing stabilization time together with its censoring status.

    The legacy scalar metric returns ``T`` both when stability is reached at
    the final available step and when it is never observed.  This structured
    result removes that ambiguity for aggregate evaluation: ``stabilized`` is
    true only when the complete run of stable windows was observed.
    """
    if int(window_size) < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")
    if int(consecutive_windows) < 1:
        raise ValueError(f"consecutive_windows must be >= 1, got {consecutive_windows}")
    if float(variance_threshold) < 0.0:
        raise ValueError(f"variance_threshold must be >= 0, got {variance_threshold}")
    _validate_logits(gate_logits)
    probs = F.softmax(gate_logits, dim=-1)
    log_probs = torch.log(probs.clamp(min=1e-8))
    entropy = -(probs * log_probs).sum(dim=-1)  # (T,)
    T = entropy.numel()

    # A run of N windows needs one full window plus N-1 additional steps,
    # not N full windows. This matters for short demonstrations.
    if T < window_size + consecutive_windows - 1:
        return RoutingStabilityResult(step=T, stabilized=False)

    windows = entropy.unfold(0, window_size, 1)
    variances = windows.var(dim=-1)
    stable = (variances < variance_threshold).to(torch.int64)
    if stable.numel() >= consecutive_windows:
        counts = stable.unfold(0, consecutive_windows, 1).sum(dim=-1)
        hits = (counts == consecutive_windows).nonzero()
        if hits.numel() > 0:
            return RoutingStabilityResult(
                # Window index j covers [j, j + window_size - 1]. The run
                # beginning at hit completes after the final window's last
                # timestep, reported as a one-based observed-step count.
                step=(int(hits[0].item()) + window_size + consecutive_windows - 1),
                stabilized=True,
            )

    return RoutingStabilityResult(step=T, stabilized=False)


class RoutingEntropyTracker:
    """Tracks routing entropy over a sliding window to compute variance."""

    def __init__(self, window_size: int = 100) -> None:
        self.window_size = window_size
        self.history: list[float] = []

    def update(self, entropy_val: float) -> None:
        """Add a new entropy value to the history."""
        self.history.append(entropy_val)
        if len(self.history) > self.window_size:
            self.history.pop(0)

    def current_variance(self) -> float:
        """Compute the variance of the current window."""
        if len(self.history) < 2:
            return 0.0
        # Sample variance
        mean = sum(self.history) / len(self.history)
        var = sum((x - mean) ** 2 for x in self.history) / (len(self.history) - 1)
        return var


class TimeToStableRouting:
    """Detects when routing has converged to a stable state."""

    def __init__(self, variance_threshold: float = 0.001, consecutive_windows: int = 5) -> None:
        self.variance_threshold = variance_threshold
        self.consecutive_windows = consecutive_windows
        self.stable_count = 0
        self.is_stable = False
        self.stable_step = -1

    def update(self, step: int, variance: float) -> bool:
        """Update stability status.

        Returns:
            True if newly stabilized on this step, False otherwise.
        """
        if self.is_stable:
            return False

        if variance < self.variance_threshold:
            self.stable_count += 1
        else:
            self.stable_count = 0

        if self.stable_count >= self.consecutive_windows:
            self.is_stable = True
            self.stable_step = step
            return True

        return False

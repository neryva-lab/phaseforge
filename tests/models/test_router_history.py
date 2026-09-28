"""Tests for the router history path (two-pass in-batch prev-top1
embedding + stickiness loss)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from phaseforge.models.components.router import TopKRouter


def _router(use_history: bool = True, seed: int = 0) -> TopKRouter:
    torch.manual_seed(seed)
    return TopKRouter(
        latent_dim=8,
        num_experts=4,
        top_k=2,
        noise_std=0.0,
        balance_coeff=0.01,
        use_history=use_history,
    )


def _latent(B: int = 6, seed: int = 1) -> torch.Tensor:
    return torch.randn(B, 8, generator=torch.Generator().manual_seed(seed))


def test_history_off_emits_zero_sticky() -> None:
    router = _router(use_history=False).eval()
    out = router(_latent())
    assert out.sticky_loss.item() == 0.0
    assert router.history_embedding is None
    assert router.history_proj is None


def test_history_on_without_trajectory_info_degrades_to_plain() -> None:
    router = _router(use_history=True).eval()
    latent = _latent()
    out_plain = router(latent)
    # With trajectory args every sample gets a history bias (at minimum the
    # "no previous step" sentinel), so the logits must change.
    traj_id = torch.tensor([0, 0, 0, 1, 1, 1])
    traj_pos = torch.tensor([0, 1, 2, 0, 1, 2])
    out_hist = router(latent, trajectory_id=traj_id, trajectory_position=traj_pos)
    assert not torch.equal(out_hist.gate_logits, out_plain.gate_logits)
    # With no adjacent pairs (every sample its own trajectory) the sticky loss
    # is 0 and every sample gets the same "no previous step" sentinel bias.
    out_nopair = router(
        latent,
        trajectory_id=torch.arange(6),
        trajectory_position=torch.zeros(6, dtype=torch.long),
    )
    assert out_nopair.sticky_loss.item() == 0.0
    assert not torch.equal(out_nopair.gate_logits, out_plain.gate_logits)
    # Paired samples get a DIFFERENT bias than the sentinel -> logits differ.
    assert not torch.equal(out_hist.gate_logits, out_nopair.gate_logits)


def test_history_changes_logits_and_sticky_matches_manual() -> None:
    router = _router(use_history=True).eval()
    latent = _latent()
    # Two trajectories: t0 steps 0..2 (all adjacent), t1 steps 0..2.
    traj_id = torch.tensor([0, 0, 0, 1, 1, 1])
    traj_pos = torch.tensor([0, 1, 2, 0, 1, 2])

    # First pass = the same router without trajectory info (no bias).
    first_pass_logits = router(latent).gate_logits
    first_pass_top1 = first_pass_logits.argmax(dim=-1)

    out = router(latent, trajectory_id=traj_id, trajectory_position=traj_pos)
    assert not torch.equal(out.gate_logits, first_pass_logits)

    # Expected stickiness by hand: valid pairs are (t0,0)->(t0,1),
    # (t0,1)->(t0,2), (t1,0)->(t1,1), (t1,1)->(t1,2) — 4 pairs.
    # For pair (prev, cur), the term is -log p_cur[top1_prev].
    expected_pairs = [(0, 1), (1, 2), (3, 4), (4, 5)]
    log_probs = F.log_softmax(out.gate_logits, dim=-1)
    expected_sticky = torch.stack(
        [-log_probs[j, first_pass_top1[i]] for i, j in expected_pairs]
    ).mean()
    assert out.sticky_loss == expected_sticky


def test_previous_top1_resolution_adjacent_only() -> None:
    router = _router(use_history=True).eval()
    top1 = torch.tensor([3, 0, 2, 1, 2, 0])  # per-sample first-pass choices
    traj_id = torch.tensor([0, 0, 0, 1, 1, 1])
    traj_pos = torch.tensor([0, 1, 2, 0, 2, 3])
    prev_top1, prev_valid = router._resolve_previous_top1(top1, traj_id, traj_pos)

    # Pairs: (0->1): prev 3, (1->2): prev 0, (4->5): prev 2.
    assert torch.equal(prev_valid, torch.tensor([False, True, True, False, False, True]))
    assert torch.equal(prev_top1, torch.tensor([0, 3, 0, 0, 0, 2]))


def test_no_valid_pairs_gives_zero_sticky() -> None:
    router = _router(use_history=True).eval()
    latent = _latent(B=3)
    traj_id = torch.tensor([0, 1, 2])
    traj_pos = torch.tensor([0, 0, 0])
    out = router(latent, trajectory_id=traj_id, trajectory_position=traj_pos)
    assert out.sticky_loss.item() == 0.0


def test_mismatched_trajectory_shapes_rejected() -> None:
    router = _router(use_history=True).eval()
    with pytest.raises(ValueError, match="use_history"):
        router(
            _latent(B=4),
            trajectory_id=torch.tensor([0, 0]),
            trajectory_position=torch.tensor([0, 1]),
        )


# ---------------------------------------------------------------------------
# evaluation-time routing interventions (sticky/uniform/oracle)
# ---------------------------------------------------------------------------


def test_uniform_selection_averages_all_experts() -> None:
    router = _router(use_history=False).eval()
    latent = _latent(B=3)
    weights, indices = router.uniform_selection(latent)
    assert weights.shape == (3, 4)
    assert indices.shape == (3, 4)
    assert torch.allclose(weights, torch.full((3, 4), 0.25))
    assert torch.equal(indices, torch.arange(4).unsqueeze(0).expand(3, 4))


def test_sticky_ema_updates_and_resets() -> None:
    router = _router(use_history=False).eval()
    latent = _latent(B=1)
    gate = router.gate_linear(latent)
    first_choice = torch.softmax(gate, dim=-1).argmax(dim=-1)

    # First step: EMA initializes to the current one-hot -> learned choice.
    w1, i1 = router.sticky_selection(gate)
    assert router._sticky_ema is not None
    assert i1[0, 0].item() == first_choice.item()
    assert torch.allclose(w1.sum(dim=-1), torch.ones(1))

    # A later step must move the EMA toward the new choice (beta=0.9).
    next_choice = (first_choice.item() + 1) % 4
    other = torch.full_like(gate, -10.0)
    other[0, first_choice.item()] = 0.0
    other[0, next_choice] = 10.0
    ema_before = router._sticky_ema.clone()
    w2, i2 = router.sticky_selection(other)
    next_onehot = torch.zeros(4)
    next_onehot[next_choice] = 1.0
    expected_ema = 0.9 * ema_before + 0.1 * next_onehot
    assert torch.allclose(router._sticky_ema, expected_ema)
    # The sticky EMA resists the switch: after one update the old choice
    # still leads the EMA scores (0.9 vs 0.1), so it keeps the old top-1.
    assert i2[0, 0].item() == first_choice.item()
    assert i2[0, 1].item() == next_choice

    # reset_sticky_ema clears the state for the next episode.
    router.reset_sticky_ema()
    assert router._sticky_ema is None


def test_oracle_selection_routes_through_mapping() -> None:
    router = _router(use_history=False).eval()
    # P=2 phases, E=4 experts, M maps phase 0 -> experts {0,1}, phase 1 ->
    # experts {2,3}.
    mapping = torch.zeros(2, 4)
    mapping[0, :2] = 0.5
    mapping[1, 2:] = 0.5
    phase_logits = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    weights, indices = router.oracle_selection(phase_logits, mapping)
    assert torch.equal(indices[0], torch.tensor([0, 1]))
    assert torch.equal(indices[1], torch.tensor([2, 3]))
    # Weights are the normalized top-k of M^T softmax(phase_logits).
    phase_probs = F.softmax(phase_logits, dim=-1)
    expert_probs = torch.einsum("pe,bp->be", mapping, phase_probs)
    assert torch.allclose(
        weights[0], F.softmax(expert_probs[0][indices[0]], dim=-1)
    )
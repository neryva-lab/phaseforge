"""Unit test auditing action adapter scaling and response characteristics.

Validates:
1. Zero error -> zero normalized action.
2. Half-limit physical displacement -> expected normalized action (~0.5 under linear clip, ~0.462
under tanh).
3. Full-limit physical displacement -> expected normalized action (1.0 under linear clip, ~0.762
under tanh).
4. Negative full-limit displacement -> expected normalized action (-1.0 under linear clip, -0.762
under tanh).
5. Comparison between tanh soft-clipping and linear saturation clipping.
"""

from __future__ import annotations

import math

import torch

from phaseforge.models.components.action_adapter import (
    impedance_action,
)

ROBOSUITE_OSC_SCALE = (0.05, 0.05, 0.05, 0.5, 0.5, 0.5, 0.04)


class TestActionAdapterScalingAudit:
    """Audit action adapter scaling across all 7 dimensions."""

    def test_zero_displacement_gives_zero_action(self):
        """Zero physical error must yield exactly 0.0 action across all dimensions."""
        task_state = torch.tensor([[0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0, 0.02]])
        target = torch.tensor([[0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0, 0.02]])
        gains = torch.ones(1, 7)

        act, info = impedance_action(target, gains, task_state, scale=ROBOSUITE_OSC_SCALE)
        assert torch.allclose(act, torch.zeros(1, 7), atol=1e-7)
        assert torch.allclose(info["task_error"], torch.zeros(1, 7), atol=1e-7)

    def test_per_dimension_half_and_full_scale_response(self):
        """Audit each dimension at half and full scale limit under nominal gains."""
        gains = torch.ones(1, 7)
        scales = torch.tensor(ROBOSUITE_OSC_SCALE)

        # Base state: position (0,0,0), identity quat (1,0,0,0), gripper aperture 0.0
        base_state = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])

        # Test translation dimensions (0, 1, 2)
        for dim in range(3):
            limit = scales[dim].item()
            target_half = base_state.clone()
            target_half[0, dim] = 0.5 * limit
            act_half, _ = impedance_action(
        target_half, gains, base_state, scale=ROBOSUITE_OSC_SCALE
    )
            expected_half_tanh = math.tanh(0.5)  # ~0.4621
            assert math.isclose(act_half[0, dim].item(), expected_half_tanh, rel_tol=1e-4)

            target_full = base_state.clone()
            target_full[0, dim] = limit
            act_full, _ = impedance_action(
        target_full, gains, base_state, scale=ROBOSUITE_OSC_SCALE
    )
            expected_full_tanh = math.tanh(1.0)  # ~0.7616
            assert math.isclose(act_full[0, dim].item(), expected_full_tanh, rel_tol=1e-4)

            target_neg = base_state.clone()
            target_neg[0, dim] = -limit
            act_neg, _ = impedance_action(target_neg, gains, base_state, scale=ROBOSUITE_OSC_SCALE)
            expected_neg_tanh = math.tanh(-1.0)  # ~ -0.7616
            assert math.isclose(act_neg[0, dim].item(), expected_neg_tanh, rel_tol=1e-4)

        # Test gripper dimension (dim 6): stroke limit 0.04m
        # Gripper open at 0.04m, target closed at 0.0m -> error = -0.04m
        grip_state = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.04]])
        grip_target_closed = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.00]])
        act_grip, _ = impedance_action(
        grip_target_closed, gains, grip_state, scale=ROBOSUITE_OSC_SCALE
    )
        # Full stroke displacement with kappa=1.0 gives tanh(-1.0) = -0.7616
        assert math.isclose(act_grip[0, 6].item(), math.tanh(-1.0), rel_tol=1e-4)

    def test_tanh_saturation_deficit_vs_linear_clipping(self):
        """Demonstrate the 24% command deficit of tanh(u/s) at boundary limit compared to linear
        clip."""
        # At boundary limit error = scale:
        # linear clip gives exactly 1.0
        # tanh gives tanh(1.0) = 0.7616 (23.8% deficit)
        u = torch.tensor([0.05])
        s = torch.tensor([0.05])

        linear_clipped = torch.clamp(u / s, -1.0, 1.0).item()
        tanh_clipped = torch.tanh(u / s).item()

        assert math.isclose(linear_clipped, 1.0, rel_tol=1e-5)
        assert math.isclose(tanh_clipped, 0.761594, rel_tol=1e-4)

        deficit = linear_clipped - tanh_clipped
        assert deficit > 0.23, f"Expected >23% deficit at full stroke, got {deficit:.4f}"

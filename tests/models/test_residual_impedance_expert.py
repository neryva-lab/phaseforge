"""Unit tests for ResidualImpedanceExpert.

Validates:
1. Zero residual scaling (beta=0) outputs bit-for-bit identical output to base_expert.
2. warm_start_experts_from_action_head copies weights into base_expert exactly.
3. partial_reinit_experts_from_action_head reinitializes dropped neurons consistently.
4. Separate direct gripper channel (gripper action is unchanged by impedance residual).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from phaseforge.models.components.action_head import ActionHead
from phaseforge.models.components.expert import (
    partial_reinit_experts_from_action_head,
    warm_start_experts_from_action_head,
)
from phaseforge.models.components.impedance_expert import ResidualImpedanceExpert


class TestResidualImpedanceExpert:
    """Test Precision-Residual Expert functionality and warm-start compatibility."""

    def test_beta_zero_is_exact_base_expert(self):
        """When beta=0, forward output must exactly equal base_expert output."""
        expert = ResidualImpedanceExpert(input_dim=16, hidden_dim=32, output_dim=7, beta=0.0)
        latent = torch.randn(8, 16)

        out_expert = expert(latent)
        out_base = expert.base_expert(latent)

        assert torch.allclose(out_expert, out_base, atol=1e-7)

    def test_warm_start_from_action_head(self):
        """ActionHead weights must transfer exactly into base_expert with jitter=0."""
        action_head = ActionHead(input_dim=16, output_dim=7, hidden_dim=32)
        expert = ResidualImpedanceExpert(input_dim=16, hidden_dim=32, output_dim=7, beta=0.0)
        experts = nn.ModuleList([expert])

        warm_start_experts_from_action_head(experts, action_head, jitter_std=0.0)

        latent = torch.randn(8, 16)
        act_head_out = action_head(latent)
        expert_out = expert(latent)

        assert torch.allclose(act_head_out, expert_out, atol=1e-6)

    def test_partial_reinit_from_action_head(self):
        """Partial reinitialization must drop exactly drop_rate fraction of neurons."""
        action_head = ActionHead(input_dim=16, output_dim=7, hidden_dim=32)
        expert1 = ResidualImpedanceExpert(input_dim=16, hidden_dim=32, output_dim=7, beta=0.0)
        expert2 = ResidualImpedanceExpert(input_dim=16, hidden_dim=32, output_dim=7, beta=0.0)
        experts = nn.ModuleList([expert1, expert2])

        dropped = partial_reinit_experts_from_action_head(
        experts, action_head, drop_rate=0.5, seed=42
    )
        assert len(dropped) == 16  # 50% of 32 hidden neurons

        # Kept neurons must match action_head exactly
        kept_indices = [i for i in range(32) if i not in dropped]
        for exp in experts:
            base = exp.base_expert
            assert torch.allclose(
                base.hidden[0].weight[kept_indices],
                action_head.trunk[0].weight[kept_indices],
            )

    def test_gripper_channel_is_unaffected_by_residual(self):
        """When beta > 0, residual impedance acts on dims 0:6 while gripper (dim 6) is
        unaffected."""
        expert = ResidualImpedanceExpert(input_dim=16, hidden_dim=32, output_dim=7, beta=0.5)
        # Set large delta and kappa to produce strong residual
        nn.init.constant_(expert.delta_head.weight, 1.0)
        nn.init.constant_(expert.gain_head.weight, 1.0)

        latent = torch.randn(8, 16)
        out_expert = expert(latent)
        out_base = expert.base_expert(latent)

        assert torch.allclose(out_expert[:, 6], out_base[:, 6], atol=1e-7)
        assert not torch.allclose(out_expert[:, :6], out_base[:, :6], atol=1e-4)

    def test_multi_arm_transport_dual_arm_geometry(self):
        """When output_dim=14 (Transport task), residual compliance affects both arms (0:6, 7:13)
        while leaving grippers (6, 13) unlagged."""
        expert = ResidualImpedanceExpert(input_dim=16, hidden_dim=32, output_dim=14, beta=0.5)
        assert expert.num_arms == 2
        assert expert.pose_dim == 12
        assert len(expert.action_scale) == 14

        nn.init.constant_(expert.delta_head.weight, 1.0)
        nn.init.constant_(expert.gain_head.weight, 1.0)

        latent = torch.randn(8, 16)
        out_expert = expert(latent)
        out_base = expert.base_expert(latent)

        assert out_expert.shape == (8, 14)
        assert torch.allclose(out_expert[:, 6], out_base[:, 6], atol=1e-7)
        assert torch.allclose(out_expert[:, 13], out_base[:, 13], atol=1e-7)
        assert not torch.allclose(out_expert[:, :6], out_base[:, :6], atol=1e-4)
        assert not torch.allclose(out_expert[:, 7:13], out_base[:, 7:13], atol=1e-4)


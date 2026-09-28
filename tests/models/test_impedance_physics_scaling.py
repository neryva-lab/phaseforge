"""Physical scaling and controller bound verification for IS-PhaseForge.

Validates:
1. Vector action_scale matches Robosuite OSC_POSE controller limits:
   - pos_delta max 0.05m -> tanh(1.0) ~ 0.7616 at nominal stiffness
   - ori_delta max 0.5rad -> tanh(1.0) ~ 0.7616 at nominal stiffness
   - gripper stroke 0.04m -> full clamping force (tanh >= 0.96 at kappa >= 2.0)
2. Cold-start zero-bias saturation prevention via task centroid bootstrapping:
   - Error at y ~ y_bar is ~ 0, tanh is in linear regime with unit gradient
   - Contrast with zero-bias baseline which saturates at tanh(-20) = -1.0
3. Cartesian position Lipschitz contraction (coord_slice=(0, 3)):
   - Gripper mode switches do not trigger false Lipschitz violations
   - Position expansions > rho are correctly penalized
4. End-to-end PhaseBootstrappedMoE bootstrapping with ImpedanceExperts.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from phaseforge.models.components.action_adapter import impedance_action
from phaseforge.models.components.action_head import ActionHead
from phaseforge.models.components.encoder import StateEncoder
from phaseforge.models.components.impedance_expert import ImpedanceExpert
from phaseforge.models.components.phase_head import PhaseClassificationHead
from phaseforge.models.components.prototype_router import PrototypeRouter
from phaseforge.models.phase_moe import PhaseBootstrappedMoE
from phaseforge.trains.losses.lipschitz import lip_penalty

ROBOSUITE_OSC_SCALE = (0.05, 0.05, 0.05, 0.5, 0.5, 0.5, 0.04)


class TestImpedancePhysicsScaling:
    """Test physical scale matching against Robosuite operational space controller."""

    def test_vector_scale_vs_scalar_scale_force_deficit(self):
        """Demonstrate that scalar s=1.0 causes a 15x force deficit while vector scale delivers full
        authority."""
        # 5 cm position displacement (typical manipulation reach)
        # Identity quaternion, 0 gripper aperture
        task_state = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
        target = torch.tensor([[0.05, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
        gains = torch.ones(1, 7)  # nominal stiffness kappa = 1.0

        # With scalar scale s = 1.0
        act_scalar, _ = impedance_action(target, gains, task_state, scale=1.0)
        # With vector scale s = ROBOSUITE_OSC_SCALE
        act_vector, _ = impedance_action(target, gains, task_state, scale=ROBOSUITE_OSC_SCALE)

        pos_scalar = act_scalar[0, 0].item()
        pos_vector = act_vector[0, 0].item()

        # Scalar scale only commands tanh(0.05 / 1.0) = 0.04996
        assert math.isclose(pos_scalar, math.tanh(0.05), rel_tol=1e-4)
        # Vector scale commands tanh(0.05 / 0.05) = tanh(1.0) = 0.76159
        assert math.isclose(pos_vector, math.tanh(1.0), rel_tol=1e-4)

        # Force/speed ratio: vector scale produces > 15x the command authority of scalar 1.0!
        ratio = pos_vector / pos_scalar
        assert ratio > 15.0, f"Expected > 15x command ratio, got {ratio:.2f}"

    def test_gripper_full_clamping_force(self):
        """Verify gripper can generate full clamping force (tanh >= 0.96) with vector scale."""
        # Gripper open: aperture 0.04m, target closed: 0.0m
        task_state = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.04]])
        target = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.00]])

        # At nominal stiffness kappa = 1.0
        gains_nom = torch.ones(1, 7)
        act_nom, _ = impedance_action(target, gains_nom, task_state, scale=ROBOSUITE_OSC_SCALE)
        grip_nom = act_nom[0, 6].item()
        assert math.isclose(grip_nom, math.tanh(-1.0), rel_tol=1e-4)

        # At grasp stiffness kappa = 2.0
        gains_grasp = torch.ones(1, 7)
        gains_grasp[0, 6] = 2.0
        act_grasp, _ = impedance_action(target, gains_grasp, task_state, scale=ROBOSUITE_OSC_SCALE)
        grip_grasp = act_grasp[0, 6].item()
        assert math.isclose(grip_grasp, math.tanh(-2.0), rel_tol=1e-4)
        # Clamping command magnitude > 0.96
        assert abs(grip_grasp) > 0.96

        # With scalar 1.0, even with max kappa=5.0, max command was
    # tanh(5 * 0.04) = tanh(0.20) = 0.197
        act_scalar_max, _ = impedance_action(target, torch.full((1, 7), 5.0), task_state, scale=1.0)
        assert abs(act_scalar_max[0, 6].item()) < 0.20, (
        "Scalar scale can never produce grasp force!"
    )

    def test_impedance_expert_with_vector_scale(self):
        """ImpedanceExpert propagates vector action_scale correctly through params and forward."""
        expert = ImpedanceExpert(
            input_dim=16,
            hidden_dim=32,
            action_scale=ROBOSUITE_OSC_SCALE,
            kappa_min=0.1,
            kappa_max=5.0,
        )
        assert expert.action_scale == ROBOSUITE_OSC_SCALE

        latent = torch.randn(4, 16)
        task_state = torch.zeros(4, 8)
        task_state[:, 2] = 1.0  # z = 1.0
        task_state[:, 3] = 1.0  # quat w = 1.0

        action = expert(latent, task_state=task_state)
        assert action.shape == (4, 7)
        assert (action >= -1.0).all() and (action <= 1.0).all()


class TestColdStartCentroidBootstrapping:
    """Verify task-space centroid initialization prevents initial saturation and vanishing
    gradients."""

    def test_nominal_stiffness_initialization(self):
        """Verify gain_head.bias init gives exact nominal stiffness kappa = 1.0 at step 0."""
        expert = ImpedanceExpert(input_dim=16, hidden_dim=32)
        # softplus(0.54132485) = ln(1 + e^0.54132485) = 1.0
        bias = expert.gain_head.bias
        stiffness = nn.functional.softplus(bias)
        torch.testing.assert_close(stiffness, torch.ones_like(stiffness), atol=1e-5, rtol=1e-5)

    def test_centroid_bias_init_prevents_saturation(self):
        """Verify expert target initialized at centroid has near-zero initial error and non-zero
        gradient."""
        # Can task: table at z ~ 0.8-1.0m, EEF at z ~ 1.0m
        centroid = torch.tensor([0.0, -0.2, 1.0, 1.0, 0.0, 0.0, 0.0, 0.04])
        expert = ImpedanceExpert(
            input_dim=16,
            hidden_dim=32,
            action_scale=ROBOSUITE_OSC_SCALE,
            target_init_bias=centroid,
        )

        # Step 0: latent is zero or small
        latent = torch.zeros(1, 16)
        target, gains = expert.params(latent)
        # Target at step 0 is very close to centroid
        torch.testing.assert_close(target[0], centroid, atol=1e-4, rtol=1e-4)

        # Given current state at centroid:
        current_state = centroid.unsqueeze(0).clone()
        action, parts = impedance_action(target, gains, current_state, scale=ROBOSUITE_OSC_SCALE)

        assert torch.allclose(action, torch.zeros_like(action), atol=1e-3)
        # Tanh derivative d(tanh(x))/dx = 1 - tanh^2(x) is ~ 1.0 (maximum gradient flow!)
        tanh_grad = 1.0 - action.pow(2)
        assert (tanh_grad > 0.99).all()

    def test_zero_bias_baseline_suffers_extreme_saturation(self):
        """Demonstrate that without centroid bias (target=0), the policy saturates at -1.0."""
        # Zero bias target
        expert_zero = ImpedanceExpert(
            input_dim=16,
            hidden_dim=32,
            action_scale=ROBOSUITE_OSC_SCALE,
            target_init_bias=None,  # defaults to 0
        )
        latent = torch.zeros(1, 16)
        target, gains = expert_zero.params(latent)

        # Physical state at z = 1.0m
        current_state = torch.tensor([[0.0, -0.2, 1.0, 1.0, 0.0, 0.0, 0.0, 0.04]])
        action, parts = impedance_action(target, gains, current_state, scale=ROBOSUITE_OSC_SCALE)

        # Z error is -1.0m / 0.05m = -20.0 -> tanh(-20) = -1.0
        z_action = action[0, 2].item()
        assert math.isclose(z_action, -1.0, abs_tol=1e-6)
        # Gradient is completely vanished: 1 - (-1.0)^2 = 0.0
        z_grad = 1.0 - z_action ** 2
        assert math.isclose(z_grad, 0.0, abs_tol=1e-8)


class TestCartesianPositionContraction:
    """Verify Lipschitz contraction regularization operates on Cartesian position without discrete
    gripper artifacts."""

    def test_gripper_switch_does_not_trigger_lipschitz_penalty(self):
        """A discrete gripper toggle (aperture 0.04 -> 0.0) with contractive position must NOT
        violate Lipschitz."""
        # 4 state pairs where position is strictly contractive (norm(dT) = 0.5 * norm(dy)),
        # but gripper snaps from 0.04 to 0.0 (discrete contact closure).
        y1 = torch.tensor([
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.04],
            [0.1, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.04],
        ])
        y2 = torch.tensor([
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.00],
            [0.1, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.00],
        ])
        # Targets contract position by 0.5:
        T1 = torch.tensor([
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.04],
            [0.05, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.04],
        ])
        T2 = torch.tensor([
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.00],
            [0.05, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.00],
        ])
        targets = torch.cat([T1, T2], dim=0)
        states = torch.cat([y1, y2], dim=0)
        experts = torch.zeros(4, dtype=torch.long)

        # coord_slice=(0, 3) restricts contraction check to Cartesian position
        loss = lip_penalty(targets, states, experts, rho=0.8, coord_slice=(0, 3), num_pairs=16)
        assert loss.item() == 0.0, (
        f"Expected 0 penalty for position-contractive states, got {loss.item()}"
    )

    def test_position_expansion_triggers_lipschitz_penalty(self):
        """When position expansion exceeds rho=0.8, lip_penalty correctly penalizes the
        violation."""
        # dT_pos = 2.0 * dy_pos
        y = torch.tensor([
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ])
        T = torch.tensor([
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # ratio = 0.2 / 0.1 = 2.0 > 0.8
        ])
        experts = torch.zeros(2, dtype=torch.long)
        loss = lip_penalty(T, y, experts, rho=0.8, coord_slice=(0, 3), num_pairs=4)
        # Violation: (2.0 - 0.8)^2 = 1.44
        assert loss.item() > 1.0, f"Expected violation penalty > 1.0, got {loss.item()}"


class TestEndToEndMoEBootstrapping:
    """Verify PhaseBootstrappedMoE bootstrapping initializes per-regime centroids into
    ImpedanceExperts."""

    def test_bootstrap_initializes_expert_centroids_from_regimes(self):
        """Verify each expert receives its regime's task centroid upon bootstrapping."""
        torch.manual_seed(42)
        state_dim = 23  # Can task state dim
        latent_dim = 16
        num_experts = 3

        encoder = StateEncoder(input_dim=state_dim, hidden_dims=[32], latent_dim=latent_dim)
        action_head = ActionHead(input_dim=latent_dim, output_dim=7, hidden_dim=32)
        phase_head = PhaseClassificationHead(latent_dim=latent_dim, num_phases=num_experts)
        router = PrototypeRouter(latent_dim=latent_dim, num_experts=num_experts, top_k=1)
        expert = ImpedanceExpert(
            input_dim=latent_dim,
            hidden_dim=32,
            action_scale=ROBOSUITE_OSC_SCALE,
        )

        model = PhaseBootstrappedMoE(
            encoder=encoder,
            action_head=action_head,
            phase_head=phase_head,
            router=router,
            expert=expert,
            router_init={"type": "centroid", "prototype_source": "topo"},
            expert_init={"type": "random"},
        )

        # Construct synthetic dataset with distinct task-space positions per regime:
        # Regime 0: pos z = 1.0
        # Regime 1: pos z = 0.8
        # Regime 2: pos z = 0.6
        samples = []
        for regime in range(num_experts):
            for _ in range(20):
                st = torch.zeros(state_dim)
                st[0:3] = torch.tensor([0.0, -0.2, 1.0 - regime * 0.2])  # EEF pos
                st[3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])  # EEF quat
                st[7:9] = torch.tensor([0.02, 0.02])  # gripper qpos
                samples.append({
                    "state": st,
                    "phase": torch.tensor(regime),
                    "phase_topo": torch.tensor(regime),
                })

        loader = DataLoader(samples, batch_size=10, shuffle=False)
        model.bootstrap_moe(loader, device="cpu")

        assert model.stage == 2
        # Check that expert biases were initialized to distinct regime centroids
        exp0 = model.moe_layer.experts[0]
        exp1 = model.moe_layer.experts[1]
        exp2 = model.moe_layer.experts[2]

        z0 = exp0.target_head.bias[2].item()
        z1 = exp1.target_head.bias[2].item()
        z2 = exp2.target_head.bias[2].item()

        assert math.isclose(z0, 1.0, abs_tol=0.05), f"Expert 0 z-centroid expected ~1.0, got {z0}"
        assert math.isclose(z1, 0.8, abs_tol=0.05), f"Expert 1 z-centroid expected ~0.8, got {z1}"
        assert math.isclose(z2, 0.6, abs_tol=0.05), f"Expert 2 z-centroid expected ~0.6, got {z2}"

        # Check nominal gains
        for exp in model.moe_layer.experts:
            stiff = nn.functional.softplus(exp.gain_head.bias)
            torch.testing.assert_close(stiff, torch.ones_like(stiff), atol=1e-4, rtol=1e-4)

        # Policy forward pass
        model.eval()
        with torch.no_grad():
            test_state = samples[0]["state"].unsqueeze(0)
            act = model.get_action(test_state)
            assert act.shape == (1, 7)
            # Action should be small since test_state matches regime 0 centroid
            assert (act.abs() < 0.5).all()

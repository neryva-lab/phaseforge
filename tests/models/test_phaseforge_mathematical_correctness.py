"""Exhaustive mathematical and physical correctness test suite for PhaseForge.

Tests the core mathematical foundations directly:
1. Physical coordinate denormalization in task_state.py (preserves meters, S^3 canonical
quaternions, aperture).
2. Action adapter quaternion error invariance under non-unit scaling.
3. Impedance expert projection to S^3 with canonical w >= 0.
4. Dynamic regime count K != 6 in PhaseBootstrappedMoE.bootstrap_moe (K=4, K=8).
5. Observability audit: well-separated regimes with 0 mutual confusion are never falsely flagged as
merge candidates.
6. Lipschitz contraction regularization: numerical stability, trajectory grouping, gradient
boundedness.
7. Stage 1 trainer: SupCon activation and out-of-range phase label resilience.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from phaseforge.data.topo.observability import audit_regimes
from phaseforge.data.topo.task_vars import extract_task_vars
from phaseforge.models.components.action_adapter import rotation_error
from phaseforge.models.components.action_head import ActionHead
from phaseforge.models.components.encoder import StateEncoder
from phaseforge.models.components.impedance_expert import ImpedanceExpert
from phaseforge.models.components.phase_head import PhaseClassificationHead
from phaseforge.models.components.router import TopKRouter
from phaseforge.models.components.task_state import extract_task_state, extract_task_state_numpy
from phaseforge.models.phase_moe import PhaseBootstrappedMoE
from phaseforge.trains.loops.base import _PhaseAccumulator
from phaseforge.trains.losses.lipschitz import lip_penalty


class TestPhysicalTaskStateExtraction:
    """Verify that task space extraction reconstructs true physical units."""

    def test_denormalization_reconstruction(self):
        # 19D state: 0:3 eef_pos, 3:7 eef_quat, 7:9 gripper_qpos, 9:19 object
        true_pos = torch.tensor([0.15, -0.25, 0.85], dtype=torch.float32)
        true_quat = torch.tensor([0.70710678, 0.0, 0.70710678, 0.0], dtype=torch.float32)
        true_grip = torch.tensor([-0.035, 0.035], dtype=torch.float32)
        true_obj = torch.zeros(10, dtype=torch.float32)

        raw_state = torch.cat([true_pos, true_quat, true_grip, true_obj], dim=0)

        # Realistic normalizer stats
        mean = torch.tensor(
        [0.1, -0.2, 0.8] + [0.5, 0.1, 0.5, 0.1] + [0.0, 0.0] + [0.0] * 10,
        dtype=torch.float32,
    )
        std = torch.tensor(
        [0.05, 0.05, 0.05] + [0.3, 0.2, 0.3, 0.2] + [0.02, 0.02] + [1.0] * 10,
        dtype=torch.float32,
    )

        normalized_state = (raw_state - mean) / std

        # Extract task state with normalizer stats
        y = extract_task_state(normalized_state, mean=mean, std=std)

        assert y.shape == (8,)
        # Pos in meters must match true physical pos
        torch.testing.assert_close(y[0:3], true_pos, atol=1e-5, rtol=1e-5)
        # Quat must be unit quaternion on S^3 with w >= 0
        torch.testing.assert_close(y[3:7], true_quat, atol=1e-5, rtol=1e-5)
        assert y[3].item() >= 0.0
        assert abs(y[3:7].norm().item() - 1.0) < 1e-6
        # Gripper aperture must be 0.035
        torch.testing.assert_close(y[7:8], torch.tensor([0.035]), atol=1e-5, rtol=1e-5)

    def test_numpy_twin_matches_torch(self):
        rng = np.random.default_rng(42)
        raw = rng.normal(size=(5, 23)).astype(np.float32)
        mean = rng.normal(size=(23,)).astype(np.float32)
        std = np.abs(rng.normal(size=(23,))).astype(np.float32) + 0.1

        norm_arr = (raw - mean) / std
        y_np = extract_task_state_numpy(norm_arr, mean=mean, std=std)

        norm_t = torch.from_numpy(norm_arr)
        mean_t = torch.from_numpy(mean)
        std_t = torch.from_numpy(std)
        y_t = extract_task_state(norm_t, mean=mean_t, std=std_t).numpy()

        np.testing.assert_allclose(y_np, y_t, atol=1e-5, rtol=1e-5)


class TestQuaternionErrorInvariance:
    """Verify rotation error is invariant to quaternion scaling."""

    def test_rotation_error_scale_invariance(self):
        # Base unit quaternions
        target_q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        # 90-degree rotation about z: [cos(pi/4), 0, 0, sin(pi/4)]
        current_q = torch.tensor([0.70710678, 0.0, 0.0, 0.70710678], dtype=torch.float32)

        err_unit = rotation_error(target_q, current_q)

        # Scale target and current by arbitrary non-zero factors (e.g. from neural net outputs)
        scaled_target = target_q * 3.7
        scaled_current = current_q * 0.42

        err_scaled = rotation_error(scaled_target, scaled_current)

        torch.testing.assert_close(err_unit, err_scaled, atol=1e-6, rtol=1e-6)

    def test_impedance_expert_s3_projection(self):
        expert = ImpedanceExpert(input_dim=16, hidden_dim=32)
        latent = torch.randn(4, 16)
        target, gains = expert.params(latent)

        assert target.shape == (4, 8)
        assert gains.shape == (4, 7)
        # Target quaternion slice (indices 3:7) must be on S^3 with w >= 0
        target_quats = target[:, 3:7]
        norms = target_quats.norm(dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-6, rtol=1e-6)
        assert (target_quats[:, 0] >= 0.0).all()


class TestDynamicRegimeBootstrapping:
    """Verify PhaseBootstrappedMoE cleanly bootstraps with arbitrary regime counts K != 6."""

    @pytest.mark.parametrize("num_regimes", [4, 8])
    def test_bootstrap_with_non_six_regimes(self, num_regimes: int):
        device = "cpu"
        latent_dim = 16
        state_dim = 19
        num_experts = num_regimes

        encoder = StateEncoder(input_dim=state_dim, hidden_dims=[32], latent_dim=latent_dim)
        action_head = ActionHead(input_dim=latent_dim, output_dim=7)
        phase_head = PhaseClassificationHead(latent_dim=latent_dim, num_phases=6)
        router = TopKRouter(latent_dim=latent_dim, num_experts=num_experts, top_k=1)
        expert = ImpedanceExpert(input_dim=latent_dim, hidden_dim=32)

        model = PhaseBootstrappedMoE(
            encoder=encoder,
            action_head=action_head,
            phase_head=phase_head,
            router=router,
            expert=expert,
            router_init={"type": "centroid", "prototype_source": "topo"},
        )

        # Create dummy dataloader with `num_regimes` distinct phase_topo labels
        n_samples = 40
        states = torch.randn(n_samples, state_dim)
        actions = torch.randn(n_samples, 7)
        # Distinct discovered regimes
        phases_topo = torch.randint(0, num_regimes, (n_samples,))
        # Ensure every regime has at least one sample
        for r in range(num_regimes):
            phases_topo[r] = r

        loader = DataLoader(
            [
        {"state": s, "action": a, "phase_topo": p}
        for s, a, p in zip(states, actions, phases_topo)
    ],
            batch_size=10,
        )

        # Bootstrapping must succeed without IndexError or zero-sample crash
        model.bootstrap_moe(loader, device=device)
        model.stage = 2

        # Test forward pass in Stage 2
        batch = {"state": torch.randn(2, state_dim)}
        out = model(batch)
        assert out.action_pred.shape == (2, 7)
        assert out.gate_logits.shape == (2, num_experts)


class TestObservabilityAuditPrecision:
    """Verify merge candidates are only reported for true mutual confusion."""

    def test_well_separated_regimes_not_flagged(self):
        # 3 regimes well separated in 2D space
        rng = np.random.default_rng(42)
        n_per_class = 60
        c0 = rng.normal(loc=[-5.0, 0.0], scale=0.5, size=(n_per_class, 2))
        c1 = rng.normal(loc=[0.0, 5.0], scale=0.5, size=(n_per_class, 2))
        c2 = rng.normal(loc=[5.0, 0.0], scale=0.5, size=(n_per_class, 2))

        states = np.vstack([c0, c1, c2])
        labels = np.array([0] * n_per_class + [1] * n_per_class + [2] * n_per_class)
        traj_ids = np.array([i // 20 for i in range(len(labels))])

        report = audit_regimes(
            states=states,
            labels=labels,
            traj_ids=traj_ids,
            num_regimes=3,
            min_macro_f1=0.7,
            merge_f1_threshold=0.5,
        )

        assert report.passed
        assert len(report.merge_candidates) == 0
        assert report.macro_f1 > 0.9


class TestLipschitzRegularizerStability:
    """Verify Lipschitz loss handles differing contexts and bounds gradients."""

    def test_lipschitz_cross_episode_stability(self):
        targets = torch.tensor([
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.04],
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.04],
            # Near identical state but different targets across different episodes
            [2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.04],
            [2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.04],
        ], requires_grad=True)

        states = torch.tensor([
            [0.1, 0.2, 0.8, 1.0, 0.0, 0.0, 0.0, 0.04],
            [0.15, 0.22, 0.81, 1.0, 0.0, 0.0, 0.0, 0.04],
            # State very close to sample 0
            [0.10001, 0.20001, 0.8, 1.0, 0.0, 0.0, 0.0, 0.04],
            [0.10002, 0.20002, 0.8, 1.0, 0.0, 0.0, 0.0, 0.04],
        ])

        experts = torch.tensor([0, 0, 0, 0])
        trajs = torch.tensor([0, 0, 1, 1])

        loss = lip_penalty(
            targets,
            states,
            experts,
            trajectory_ids=trajs,
            rho=0.8,
            max_ratio=10.0,
            min_delta_state=1e-4,
        )

        assert torch.isfinite(loss)
        loss.backward()
        assert targets.grad is not None
        assert torch.isfinite(targets.grad).all()
        # Gradient must not explode
        assert targets.grad.norm().item() < 100.0


class TestPhaseAccumulatorBounds:
    """Verify _PhaseAccumulator safely handles out-of-range labels."""

    def test_accumulator_out_of_range_handling(self):
        acc = _PhaseAccumulator(device=torch.device("cpu"), num_phases=6)
        logits = torch.randn(10, 6)
        # Phase labels with values exceeding 5 (e.g. 7)
        targets = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 1, 2])

        # Must not crash with scatter_add_ index error
        acc.update(logits, targets)
        assert acc.has_data
        micro, macro = acc.compute()
        assert 0.0 <= micro <= 1.0
        assert 0.0 <= macro <= 1.0

class TestTaskVarsPhysicalMetric:
    """Verify extract_task_vars computes exact physical relative positions when denormalized."""

    def test_rel_ee_obj_physical_correctness(self):
        # Keys: robot0_eef_pos(3), robot0_eef_quat(4), robot0_gripper_qpos(2), object(10)
        state_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
        state_dims = [3, 4, 2, 10]

        true_eef = np.array([0.2, 0.1, 0.8])
        true_quat = np.array([1.0, 0.0, 0.0, 0.0])
        true_grip = np.array([0.02, 0.02])
        true_obj = np.array([0.25, 0.12, 0.78, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        raw_state = np.concatenate([true_eef, true_quat, true_grip, true_obj])

        mean = np.random.randn(19)
        std = np.abs(np.random.randn(19)) + 0.1

        norm_state = (raw_state - mean) / std

        # Extract with denormalization
        vars_dict = extract_task_vars(norm_state, state_keys, state_dims, mean=mean, std=std)

        # rel_ee_obj should be exactly true_eef - true_obj[0:3]
        expected_rel = true_eef - true_obj[0:3]
        np.testing.assert_allclose(vars_dict["rel_ee_obj"], expected_rel, atol=1e-6)

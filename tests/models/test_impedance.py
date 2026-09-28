"""CPU-only tests for impedance experts + action adapter."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from phaseforge.models.components.action_adapter import (
    blend_impedance,
    impedance_action,
    quat_log_map,
    task_error,
)
from phaseforge.models.components.action_head import ActionHead
from phaseforge.models.components.encoder import StateEncoder
from phaseforge.models.components.impedance_expert import ImpedanceExpert
from phaseforge.models.components.phase_head import PhaseClassificationHead
from phaseforge.models.components.prototype_router import PrototypeRouter
from phaseforge.models.components.task_state import (
    TASK_ERROR_DIM,
    TASK_STATE_DIM,
    extract_task_state,
)
from phaseforge.models.phase_moe import PhaseBootstrappedMoE


def _identity_quats(n: int) -> torch.Tensor:
    quats = torch.zeros(n, 4)
    quats[:, 0] = 1.0
    return quats


def test_task_state_widths_and_guards() -> None:
    for width in (19, 23, 53):
        assert extract_task_state(torch.randn(5, width)).shape == (5, TASK_STATE_DIM)
    assert TASK_STATE_DIM == 8 and TASK_ERROR_DIM == 7
    with pytest.raises(ValueError):
        extract_task_state(torch.randn(5, 59))
    with pytest.raises(ValueError):
        extract_task_state(torch.randn(5, 10))


def test_quat_log_map_properties() -> None:
    assert quat_log_map(_identity_quats(3)).abs().max().item() == pytest.approx(0.0)
    # 90° about z: rotvec [0, 0, pi/2].
    half = float(np.cos(np.pi / 4))
    quat = torch.tensor([[half, 0.0, 0.0, half]])
    np.testing.assert_allclose(
        quat_log_map(quat).numpy(), [[0.0, 0.0, np.pi / 2]], atol=1e-5
    )
    # 180° stays finite; antipodal quats agree.
    straight = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    assert torch.isfinite(quat_log_map(straight)).all()
    assert torch.allclose(quat_log_map(quat), quat_log_map(-quat), atol=1e-5)


def test_task_error_layout() -> None:
    target = torch.zeros(4, 8)
    target[:, 0:3] = 1.0
    target[:, 0] = 1.0
    state = torch.zeros(4, 8)
    state[:, 0] = 1.0
    error = task_error(target, state)
    assert error.shape == (4, 7)
    assert error[0, 0].item() == pytest.approx(0.0)
    assert error[0, 3:6].abs().max().item() == pytest.approx(0.0)
    with pytest.raises(ValueError):
        task_error(torch.zeros(4, 7), state)


def test_impedance_action_bounds_and_guards() -> None:
    torch.manual_seed(0)
    target = torch.randn(8, 8)
    state = torch.randn(8, 23)
    task_y = extract_task_state(state)
    gains = torch.rand(8, 7) * 4.9 + 0.1
    action, parts = impedance_action(target, gains, task_y, scale=1.0)
    assert action.shape == (8, 7)
    assert bool((action >= -1.0).all()) and bool((action <= 1.0).all())
    assert parts["task_error"].shape == (8, 7)
    assert parts["pre_clip_u"].shape == (8, 7)
    with pytest.raises(ValueError):
        impedance_action(target, torch.zeros(8, 7), task_y)
    with pytest.raises(ValueError):
        impedance_action(target, gains, task_y, scale=0.0)
    with pytest.raises(ValueError):
        impedance_action(target, gains[:, :6], task_y)


def test_blend_top1_matches_direct_and_keff_formula() -> None:
    torch.manual_seed(0)
    targets = torch.randn(6, 1, 8)
    gains = torch.rand(6, 1) * 2.0 + 0.5
    gains = gains.unsqueeze(-1).expand(6, 1, 7).contiguous()
    weights = torch.ones(6, 1)
    target_eff, gains_eff = blend_impedance(targets, gains, weights)
    # K=1: pos/gripper exact; the quaternion is the normalized input.
    assert torch.allclose(target_eff[..., 0:3], targets[:, 0, 0:3])
    assert torch.allclose(target_eff[..., 7:8], targets[:, 0, 7:8])
    assert torch.allclose(
        target_eff[..., 3:7],
        targets[:, 0, 3:7] / targets[:, 0, 3:7].norm(dim=-1, keepdim=True),
    )
    assert torch.allclose(gains_eff, gains[:, 0])

    targets2 = torch.randn(4, 2, 8)
    gains2 = torch.rand(4, 2, 7) + 0.5
    weights2 = torch.tensor([[0.7, 0.3]] * 4)
    target_eff2, gains_eff2 = blend_impedance(targets2, gains2, weights2)
    expected_k = (weights2.unsqueeze(-1) * gains2).sum(1)
    assert torch.allclose(gains_eff2, expected_k)
    # Per-dim stiffness weights: alpha_d = w*K_d / sum(w*K_d).
    alpha = weights2.unsqueeze(-1) * gains2[..., 0:3]
    alpha = alpha / alpha.sum(dim=1, keepdim=True)
    expected_pos = (alpha * targets2[..., 0:3]).sum(1)
    assert torch.allclose(target_eff2[..., 0:3], expected_pos)
    with pytest.raises(ValueError):
        blend_impedance(targets2, gains2, torch.zeros(4, 2))


def test_impedance_expert_params_and_forward() -> None:
    torch.manual_seed(0)
    expert = ImpedanceExpert(input_dim=8, hidden_dim=16)
    assert expert.output_dim == 7
    latent = torch.randn(8, 8)
    target, gains = expert.params(latent)
    assert target.shape == (8, 8) and gains.shape == (8, 7)
    assert bool((gains > 0.0).all()) and bool((gains <= 5.0 + 1e-6).all())
    state = torch.randn(8, 23)
    action = expert(latent, extract_task_state(state))
    assert action.shape == (8, 7)
    with pytest.raises(ValueError):
        expert(latent)
    with pytest.raises(ValueError):
        ImpedanceExpert(input_dim=8, action_scale=0.0)
    with pytest.raises(ValueError):
        ImpedanceExpert(input_dim=8, kappa_min=0.0)


def test_moe_impedance_top1_matches_manual_adapter() -> None:
    from phaseforge.models.components.expert import ExpertMLP
    from phaseforge.models.components.moe_layer import MoELayer
    from phaseforge.models.components.router import TopKRouter

    torch.manual_seed(2)
    router = TopKRouter(latent_dim=8, num_experts=3, top_k=1, normalize_input=True)
    experts = torch.nn.ModuleList(
        [ImpedanceExpert(input_dim=8, hidden_dim=16) for _ in range(3)]
    )
    layer = MoELayer(router=router, experts=experts)
    latent = torch.randn(8, 8)
    task_state = extract_task_state(torch.randn(8, 23))
    out = layer(latent, task_state=task_state)
    assert out.combined_output.shape == (8, 7)
    assert out.info is not None
    for row in range(8):
        expert_idx = int(out.expert_indices[row, 0].item())
        target, gains = experts[expert_idx].params(latent[row : row + 1])
        expected, _parts = impedance_action(
            target, gains, task_state[row : row + 1], experts[expert_idx].action_scale
        )
        assert torch.allclose(out.combined_output[row : row + 1], expected)
    direct_layer = MoELayer(
        router=TopKRouter(latent_dim=8, num_experts=2, top_k=1),
        experts=[ExpertMLP(input_dim=8, hidden_dims=[16], output_dim=7) for _ in range(2)],
    )
    assert direct_layer(latent, task_state=None).info is None


def test_moe_flavor_mismatch_fails_closed() -> None:
    from phaseforge.models.components.expert import ExpertMLP
    from phaseforge.models.components.moe_layer import MoELayer
    from phaseforge.models.components.router import TopKRouter

    torch.manual_seed(0)
    router = TopKRouter(latent_dim=8, num_experts=2, top_k=1)
    imp_layer = MoELayer(
        router=router,
        experts=torch.nn.ModuleList(
            [ImpedanceExpert(input_dim=8, hidden_dim=8) for _ in range(2)]
        ),
    )
    with pytest.raises(RuntimeError, match="task_state"):
        imp_layer(torch.randn(4, 8))
    direct_layer = MoELayer(
        router=TopKRouter(latent_dim=8, num_experts=2, top_k=1),
        experts=torch.nn.ModuleList(
            [ExpertMLP(input_dim=8, hidden_dims=[8], output_dim=7) for _ in range(2)]
        ),
    )
    with pytest.raises(RuntimeError, match="Direct"):
        direct_layer(torch.randn(4, 8), task_state=torch.randn(4, 8))
    with pytest.raises(ValueError, match="mixing"):
        MoELayer(
            router=TopKRouter(latent_dim=8, num_experts=2, top_k=1),
            experts=torch.nn.ModuleList(
                [
                    ExpertMLP(input_dim=8, hidden_dims=[8], output_dim=7),
                    ImpedanceExpert(input_dim=8, hidden_dim=8),
                ]
            ),
        )


def test_bootstrap_installs_prototypes_for_impedance_template() -> None:
    """Template-cloned ImpedanceExperts + centroid install into prototypes."""
    torch.manual_seed(0)
    encoder = StateEncoder(input_dim=19, hidden_dims=[16], latent_dim=8)
    model = PhaseBootstrappedMoE(
        encoder=encoder,
        action_head=ActionHead(input_dim=8, output_dim=7, hidden_dim=16),
        phase_head=PhaseClassificationHead(latent_dim=8, num_phases=2),
        router=PrototypeRouter(latent_dim=8, num_experts=2, top_k=1),
        expert=ImpedanceExpert(input_dim=8, hidden_dim=16),
        router_init={"type": "centroid", "prototype_source": "rule", "seed": 0},
        expert_init={"type": "random"},
    )
    assert len(model.moe_layer.experts) == 2
    assert all(isinstance(e, ImpedanceExpert) for e in model.moe_layer.experts)
    gen = torch.Generator().manual_seed(0)
    rows = [
        {
            "state": torch.randn(19, generator=gen),
            "phase": torch.tensor(int(i % 2)),
            "phase_topo": torch.tensor(int(i % 2)),
        }
        for i in range(64)
    ]
    model.bootstrap_moe(dataloader=DataLoader(rows, batch_size=16), device="cpu")
    assert model.stage == 2
    assert torch.isfinite(model.moe_layer.router.prototypes).all()
    model.eval()
    with torch.no_grad():
        action = model.get_action(torch.randn(4, 19))
    assert action.shape == (4, 7)
    assert bool((action >= -1.0).all()) and bool((action <= 1.0).all())

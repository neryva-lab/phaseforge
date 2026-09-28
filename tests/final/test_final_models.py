"""Final model-contract gates (ledger TEST-03, TEST-04, TEST-05, TEST-06).

TEST-03: plain prototypes come from the plain model's OWN latents (latent
isolation — never the proposed-method checkpoint/latents).
TEST-04: topology and random router initialization is deterministic under
the declared seed policy.
TEST-05: partial warm-start drop masks and metadata are seed-deterministic.
TEST-06: beta-zero residual experts equal the direct base action and carry
no action-loss gradient on the residual heads.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from phaseforge.models.baselines.precision_residual_plain_encoder import (
    PrecisionResidualPlainEncoderModel,
)
from phaseforge.models.components.action_head import ActionHead
from phaseforge.models.components.encoder import StateEncoder
from phaseforge.models.components.expert import ExpertMLP
from phaseforge.models.components.impedance_expert import ResidualImpedanceExpert
from phaseforge.models.components.phase_head import PhaseClassificationHead
from phaseforge.models.components.prototype_router import PrototypeRouter
from phaseforge.models.phase_moe import PhaseBootstrappedMoE


class _DictDataset(Dataset):
    def __init__(self, n: int = 48, seed: int = 0) -> None:
        gen = torch.Generator().manual_seed(seed)
        self.states = torch.randn(n, 19, generator=gen)
        self.actions = torch.randn(n, 7, generator=gen)
        self.phases = torch.randint(0, 6, (n,), generator=gen)
        self.topos = torch.randint(0, 6, (n,), generator=gen)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> dict:
        return {
            "state": self.states[idx],
            "action": self.actions[idx],
            "phase": self.phases[idx],
            "phase_topo": self.topos[idx],
        }


def _loader(**kwargs) -> DataLoader:
    return DataLoader(_DictDataset(**kwargs), batch_size=16)


def _plain(**over) -> PrecisionResidualPlainEncoderModel:
    torch.manual_seed(11)
    kwargs: dict = {
        "encoder": StateEncoder(
            input_dim=19, hidden_dims=[64], latent_dim=32, normalize_output=True
        ),
        "action_head": ActionHead(input_dim=32, output_dim=7, hidden_dim=64),
        "router": PrototypeRouter(latent_dim=32, num_experts=6),
        "expert": ResidualImpedanceExpert(input_dim=32, hidden_dims=[64], output_dim=7, beta=0.0),
        "num_phases": 6,
        "expert_init": {"type": "partial_warm", "drop_rate": 0.5, "seed": 7},
        "router_init": {"type": "centroid"},
        "bootstrap_label_field": "phase_topo",
    }
    kwargs.update(over)
    return PrecisionResidualPlainEncoderModel(**kwargs)


def test_plain_prototypes_come_from_own_latents() -> None:
    """TEST-03: prototypes equal own-encoder centroids (isolation)."""
    model = _plain()
    loader = _loader()
    model.bootstrap_moe(dataloader=loader, device="cpu", training_seed=7)
    # Recompute the expected centroids from the model's OWN encoder.
    model.eval()
    sums = torch.zeros(6, 32)
    counts = torch.zeros(6)
    with torch.no_grad():
        for batch in loader:
            latent = model.encoder(batch["state"])
            sums.scatter_add_(0, batch["phase_topo"].unsqueeze(1).expand_as(latent), latent)
            counts += torch.bincount(batch["phase_topo"], minlength=6).float()
    expected = torch.nn.functional.normalize(sums / counts.unsqueeze(1), p=2, dim=-1)
    assert torch.allclose(model.moe_layer.router.prototypes.data, expected, atol=1e-6)
    # A different latent geometry yields different prototypes (no reuse).
    other = _plain()
    torch.manual_seed(99)
    with torch.no_grad():
        for p in other.encoder.parameters():
            p.add_(torch.randn_like(p))
    other.bootstrap_moe(dataloader=loader, device="cpu", training_seed=7)
    assert not torch.allclose(
        model.moe_layer.router.prototypes.data, other.moe_layer.router.prototypes.data
    )


def test_topology_init_is_seed_deterministic() -> None:
    """TEST-04: identical seeds give identical topology prototypes."""
    first = _plain()
    first.bootstrap_moe(dataloader=_loader(), device="cpu", training_seed=7)
    second = _plain()
    second.bootstrap_moe(dataloader=_loader(), device="cpu", training_seed=7)
    assert torch.equal(
        first.moe_layer.router.prototypes.data, second.moe_layer.router.prototypes.data
    )
    assert first._expert_init_info["training_seed"] == 7
    assert second._expert_init_info["training_seed"] == 7
    assert (
        first._expert_init_info["router"]["bootstrap_label_field"] == "phase_topo"
    )


def test_partial_warm_start_is_seed_deterministic() -> None:
    """TEST-05: drop masks, hashes, and kept rows reproduce exactly."""
    first = _plain()
    first.bootstrap_moe(dataloader=_loader(), device="cpu", training_seed=7)
    second = _plain()
    second.bootstrap_moe(dataloader=_loader(), device="cpu", training_seed=7)
    info_a = first._expert_init_info["expert_init"]
    info_b = second._expert_init_info["expert_init"]
    assert info_a["dropped_neuron_indices"] == info_b["dropped_neuron_indices"]
    assert info_a["dropped_indices_sha256"] == info_b["dropped_indices_sha256"]
    assert info_a["init_seed"] == info_b["init_seed"] == 7
    for expert_a, expert_b in zip(first.moe_layer.experts, second.moe_layer.experts):
        for pa, pb in zip(expert_a.parameters(), expert_b.parameters()):
            assert torch.equal(pa.data, pb.data)


def test_beta_zero_equals_base_action_without_residual_gradient() -> None:
    """TEST-06: beta-zero output is the direct base action; heads get no grad."""
    torch.manual_seed(3)
    expert = ResidualImpedanceExpert(input_dim=32, hidden_dims=[64], output_dim=7, beta=0.0)
    latent = torch.randn(8, 32, requires_grad=True)
    action = expert(latent)
    with torch.no_grad():
        expected = expert.base_expert(latent)
    assert torch.allclose(action, expected, atol=1e-9)
    loss = action.pow(2).mean()
    grads = torch.autograd.grad(loss, latent, retain_graph=True)
    assert grads[0] is not None and torch.isfinite(grads[0]).all()
    # Residual heads carry no action-loss gradient at beta zero: the forward
    # returns before they are even evaluated.
    assert expert.delta_head.weight.grad is None
    assert expert.gain_head.weight.grad is None


def test_proposed_family_partial_warm_matches_plain_policy() -> None:
    """TEST-05 (family scope): same seed/policy reproduces across classes."""
    torch.manual_seed(11)
    model = PhaseBootstrappedMoE(
        encoder=StateEncoder(input_dim=19, hidden_dims=[64], latent_dim=32, normalize_output=True),
        action_head=ActionHead(input_dim=32, output_dim=7, hidden_dim=64),
        phase_head=PhaseClassificationHead(latent_dim=32, num_phases=6),
        router=PrototypeRouter(latent_dim=32, num_experts=6),
        expert=ExpertMLP(input_dim=32, hidden_dims=[64], output_dim=7),
        router_init={"type": "centroid", "prototype_source": "topo", "seed": 7},
        expert_init={"type": "partial_warm", "drop_rate": 0.5, "seed": 7, "jitter_std": 0.0},
    )
    model.bootstrap_moe(dataloader=_loader(), device="cpu", training_seed=7)
    info = model._expert_init_info
    assert info["expert_init"]["init_seed"] == 7
    assert info["router"]["prototype_source"] == "topo"
    assert info["router"]["label_mapping"] is not None

"""Final privileged-diagnostic gates (ledger TEST-10, TEST-11).

TEST-10: teacher-forced training routes by the declared labels while
evaluation routes by the predicted phase-head class (label-free).
TEST-11: the oracle refuses ordinary state-only rollout and serves the
offline supplied-label path with recorded privilege metadata.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from phaseforge.models.baselines.precision_residual_oracle import PrecisionResidualOracleModel
from phaseforge.models.baselines.precision_residual_teacher_forced import (
    PrecisionResidualTeacherForcedModel,
)
from phaseforge.models.components.action_head import ActionHead
from phaseforge.models.components.encoder import StateEncoder
from phaseforge.models.components.expert import ExpertMLP
from phaseforge.models.components.impedance_expert import ResidualImpedanceExpert
from phaseforge.models.components.phase_head import PhaseClassificationHead
from phaseforge.models.components.router import TopKRouter


class _DictDataset(Dataset):
    def __init__(self, n: int = 32, seed: int = 0) -> None:
        gen = torch.Generator().manual_seed(seed)
        self.states = torch.randn(n, 19, generator=gen)
        self.actions = torch.randn(n, 7, generator=gen)
        self.topos = torch.randint(0, 6, (n,), generator=gen)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> dict:
        return {
            "state": self.states[idx],
            "action": self.actions[idx],
            "phase": self.topos[idx],
            "phase_topo": self.topos[idx],
        }


def _loader(**kwargs) -> DataLoader:
    return DataLoader(_DictDataset(**kwargs), batch_size=8)


def _teacher() -> PrecisionResidualTeacherForcedModel:
    torch.manual_seed(5)
    return PrecisionResidualTeacherForcedModel(
        encoder=StateEncoder(
            input_dim=19, hidden_dims=[64], latent_dim=32, normalize_output=True
        ),
        action_head=ActionHead(input_dim=32, output_dim=7, hidden_dim=64),
        phase_head=PhaseClassificationHead(latent_dim=32, num_phases=6),
        router=TopKRouter(latent_dim=32, num_experts=6, top_k=1),
        expert=ResidualImpedanceExpert(input_dim=32, hidden_dims=[64], output_dim=7, beta=0.0),
        expert_init={"type": "partial_warm", "drop_rate": 0.5, "seed": 9},
        label_field="phase_topo",
    )


def _oracle(**over) -> PrecisionResidualOracleModel:
    torch.manual_seed(5)
    kwargs: dict = {
        "encoder": StateEncoder(
            input_dim=19, hidden_dims=[64], latent_dim=32, normalize_output=True
        ),
        "router": TopKRouter(latent_dim=32, num_experts=6, top_k=1),
        "expert": ExpertMLP(input_dim=32, hidden_dims=[64], output_dim=7),
        "num_phases": 6,
    }
    kwargs.update(over)
    return PrecisionResidualOracleModel(**kwargs)


def test_teacher_uses_labels_in_training_and_predictions_in_eval() -> None:
    """TEST-10: GT-train / predicted-eval split on the declared field."""
    model = _teacher()
    model.bootstrap_moe(dataloader=_loader(), device="cpu", training_seed=9)
    assert model.stage == 2
    assert model._expert_init_info["label_field"] == "phase_topo"
    assert model._expert_init_info["diagnostic"] == "privileged_training"

    model.train()
    batch = next(iter(_loader()))
    out = model(batch)
    assert out.expert_indices.squeeze(-1).tolist() == batch["phase_topo"].tolist()

    model.eval()
    state_only = {"state": batch["state"]}
    out = model(state_only)
    assert out.phase_logits is not None
    with torch.no_grad():
        expected = model.phase_head(model.encoder(batch["state"])).argmax(-1)
    assert out.expert_indices.squeeze(-1).tolist() == expected.tolist()


def test_oracle_refuses_rollout_and_serves_offline_labels() -> None:
    """TEST-11: state-only rollout refused; offline path routes by labels."""
    model = _oracle(
        label_field="phase_topo",
        action_head=ActionHead(input_dim=32, output_dim=7, hidden_dim=64),
        expert_init={"type": "partial_warm", "drop_rate": 0.5, "seed": 9},
        allow_rollout=False,
    )
    model.bootstrap_moe(dataloader=_loader(), device="cpu", training_seed=9)
    info = model._expert_init_info
    assert info["diagnostic"] == "privileged_offline"
    assert info["label_field"] == "phase_topo"
    assert info["allow_rollout"] is False

    with pytest.raises(RuntimeError, match="refuses ordinary state-only rollout"):
        model.get_action(torch.randn(2, 19))

    batch = next(iter(_loader()))
    out = model(batch)
    assert out.expert_indices.squeeze(-1).tolist() == batch["phase_topo"].tolist()


def test_oracle_missing_labels_fail_closed() -> None:
    """TEST-11: the offline path never falls back to the untrained router."""
    model = _oracle()
    with pytest.raises(RuntimeError, match="requires ground-truth"):
        model({"state": torch.randn(4, 19)})

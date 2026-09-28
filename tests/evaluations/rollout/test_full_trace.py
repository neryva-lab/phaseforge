"""CPU-only tests for full-trace rollout integration.

Drives RolloutEvaluator with the kinematic FakeAdapter (no robosuite) in
both trace modes: `minimal` writes no trace file and behaves exactly as
before; `full` writes validated 22-field rows without changing outcomes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from phaseforge.data.common.normalizer import FrozenNormalizer
from phaseforge.evaluations.rollout.runner import RolloutEvaluator
from phaseforge.evaluations.rollout.trace import (
    TRACE_FIELDS,
    read_trace_rows,
    validate_trace_record,
)
from phaseforge.models.components.action_head import ActionHead
from phaseforge.models.components.encoder import StateEncoder
from phaseforge.models.components.impedance_expert import ImpedanceExpert
from phaseforge.models.components.phase_head import PhaseClassificationHead
from phaseforge.models.components.prototype_router import PrototypeRouter
from phaseforge.models.phase_moe import PhaseBootstrappedMoE
from tests.rollout_helpers import FakeAdapter, make_bank


def _impedance_model(seed: int = 0) -> PhaseBootstrappedMoE:
    torch.manual_seed(seed)
    encoder = StateEncoder(input_dim=19, hidden_dims=[16], latent_dim=8)
    model = PhaseBootstrappedMoE(
        encoder=encoder,
        action_head=ActionHead(input_dim=8, output_dim=7, hidden_dim=16),
        phase_head=PhaseClassificationHead(latent_dim=8, num_phases=2),
        router=PrototypeRouter(latent_dim=8, num_experts=2, top_k=1),
        expert=ImpedanceExpert(input_dim=8, hidden_dim=16),
        router_init={"type": "random"},
        expert_init={"type": "random"},
    )
    model.stage = 2
    return model


def _evaluator(tmp_path: Path, horizon: int = 5, **kwargs) -> RolloutEvaluator:
    defaults: dict = {
        "cfg": None,
        "policy": None,
        "adapter": FakeAdapter(),
        "bank": make_bank(),
        "normalizer": FrozenNormalizer(torch.zeros(19), torch.ones(19)),
        "model": _impedance_model(),
        "output_dir": tmp_path,
        "run_id": "trace-test",
        "model_name": "precision_residual_phaseforge",
        "training_seed": 42,
        "task": "Lift",
        "checkpoint_sha256": "deadbeef",
        "horizon": horizon,
        "trace_level": "full",
    }
    defaults.update(kwargs)
    return RolloutEvaluator(**defaults)  # type: ignore[arg-type]


def test_full_trace_rows_validate_and_close_episodes(tmp_path: Path) -> None:
    evaluator = _evaluator(tmp_path)
    results = evaluator.run()
    assert results["eval/rollout/trace_level"] == "full"
    rows = read_trace_rows(tmp_path)
    assert len(rows) == 3 * 5
    for row in rows:
        validate_trace_record(row)
    assert set(rows[0]) == set(TRACE_FIELDS)
    by_episode: dict[int, list[dict]] = {}
    for row in rows:
        by_episode.setdefault(row["episode_id"], []).append(row)
    assert len(by_episode) == 3
    for episode_rows in by_episode.values():
        assert [r["timestep"] for r in episode_rows] == [0, 1, 2, 3, 4]
        assert [r["done"] for r in episode_rows] == [False] * 4 + [True]
        assert {r["termination_reason"] for r in episode_rows} == {"task_timeout"}
        assert all(
            r["final_action"] is not None and len(r["final_action"]) == 7
            for r in episode_rows
        )
        assert all(r["expert_target"] is not None for r in episode_rows)


def test_minimal_mode_writes_no_trace_file(tmp_path: Path) -> None:
    evaluator = _evaluator(tmp_path, trace_level="minimal")
    evaluator.run()
    assert not (tmp_path / "trace.jsonl").exists()


def test_full_trace_matches_minimal_outcomes(tmp_path_factory) -> None:
    first = tmp_path_factory.mktemp("full") / "run"
    first.mkdir()
    second = tmp_path_factory.mktemp("minimal") / "run"
    second.mkdir()
    torch.manual_seed(0)
    full = _evaluator(first).run()
    torch.manual_seed(0)
    minimal = _evaluator(second, trace_level="minimal").run()
    assert full["eval/rollout/successes"] == minimal["eval/rollout/successes"]
    assert full["eval/rollout/success_rate"] == minimal["eval/rollout/success_rate"]


def test_full_trace_without_describe_records_nulls(tmp_path: Path) -> None:
    class _ZeroModel(torch.nn.Module):
        def get_action(self, state: torch.Tensor) -> torch.Tensor:
            return torch.zeros(state.shape[0], 7)

    evaluator = _evaluator(tmp_path, model=_ZeroModel())
    evaluator.run()
    rows = read_trace_rows(tmp_path)
    assert rows
    for row in rows:
        validate_trace_record(row)
        assert row["expert_target"] is None
        assert row["dists"] is None


def test_stride_records_final_step(tmp_path: Path) -> None:
    evaluator = _evaluator(tmp_path, horizon=5, trace_every_n_steps=2)
    evaluator.run()
    rows = read_trace_rows(tmp_path)
    by_episode: dict[int, list[dict]] = {}
    for row in rows:
        by_episode.setdefault(row["episode_id"], []).append(row)
    for episode_rows in by_episode.values():
        assert [r["timestep"] for r in episode_rows] == [0, 2, 4]
        assert episode_rows[-1]["done"] is True


def test_bad_stride_rejected(tmp_path: Path) -> None:
    from phaseforge.evaluations.envs.errors import EnvParityError

    with pytest.raises(EnvParityError):
        _evaluator(tmp_path, trace_every_n_steps=0)

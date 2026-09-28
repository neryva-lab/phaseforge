"""Phase 1 instrumentation tests (CPU-only).

Covers the additive Phase 1 scaffolding without touching behavior:
  * ``resolve_trace_level`` defaults/accepts/rejects (fail-closed).
  * ``RolloutEvaluator`` stores ``trace_level`` (default minimal).
  * ``deployment_contract()`` on the BC baseline and the switched MoE.
  * ``freeze_check.compare_resolved_configs`` frozen-key equality.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from phaseforge.data.common.normalizer import FrozenNormalizer
from phaseforge.evaluations.envs.errors import EnvParityError
from phaseforge.evaluations.rollout.runner import (
    TRACE_LEVELS,
    RolloutEvaluator,
    resolve_trace_level,
)
from tests.rollout_helpers import FakeAdapter, make_bank

REPO_ROOT = Path(__file__).resolve().parents[3]
FREEZE_CHECK = REPO_ROOT / "scripts" / "dev" / "freeze_check.py"


def _load_freeze_check():
    spec = importlib.util.spec_from_file_location("freeze_check", FREEZE_CHECK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trace_levels_contains_minimal_full() -> None:
    assert TRACE_LEVELS == frozenset({"minimal", "full"})


def test_resolve_trace_level_defaults_minimal() -> None:
    assert resolve_trace_level(DictConfig({})) == "minimal"
    assert resolve_trace_level(DictConfig({"eval": {}})) == "minimal"
    cfg = DictConfig({"eval": {"episodes": {}}})
    assert resolve_trace_level(cfg) == "minimal"


def test_resolve_trace_level_accepts_full_case_insensitive() -> None:
    cfg = DictConfig({"eval": {"episodes": {"trace_level": "Full"}}})
    assert resolve_trace_level(cfg) == "full"


def test_resolve_trace_level_rejects_unknown() -> None:
    cfg = DictConfig({"eval": {"episodes": {"trace_level": "verbose"}}})
    with pytest.raises(EnvParityError):
        resolve_trace_level(cfg)


def _evaluator(tmp_path: Path, **kwargs) -> RolloutEvaluator:
    defaults: dict = {
        "cfg": None,
        "policy": None,
        "adapter": FakeAdapter(),
        "bank": make_bank(),
        "normalizer": FrozenNormalizer(torch.zeros(19), torch.ones(19)),
        "model": None,
        "output_dir": tmp_path,
        "run_id": "phase1",
        "model_name": "precision_residual_phaseforge",
        "training_seed": 42,
        "task": "Lift",
        "checkpoint_sha256": "deadbeef",
    }
    defaults.update(kwargs)
    return RolloutEvaluator(**defaults)  # type: ignore[arg-type]


def test_evaluator_trace_level_defaults_minimal(tmp_path: Path) -> None:
    assert _evaluator(tmp_path).trace_level == "minimal"


def test_evaluator_trace_level_full_stored(tmp_path: Path) -> None:
    assert _evaluator(tmp_path, trace_level="full").trace_level == "full"


def test_evaluator_trace_level_rejects_unknown(tmp_path: Path) -> None:
    with pytest.raises(EnvParityError):
        _evaluator(tmp_path, trace_level="verbose")


def _small_bc():
    from phaseforge.models.baselines.bc import BehaviorCloningModel
    from phaseforge.models.components.action_head import ActionHead
    from phaseforge.models.components.encoder import StateEncoder

    encoder = StateEncoder(input_dim=10, hidden_dims=[16], latent_dim=8)
    head = ActionHead(input_dim=8, output_dim=7, hidden_dim=16)
    return BehaviorCloningModel(encoder=encoder, action_head=head)


def test_deployment_contract_bc_is_memoryless() -> None:
    assert _small_bc().deployment_contract() == {"memoryless": True}


def _small_phase_moe():
    from phaseforge.models.components.action_head import ActionHead
    from phaseforge.models.components.encoder import StateEncoder
    from phaseforge.models.components.expert import ExpertMLP
    from phaseforge.models.components.phase_head import PhaseClassificationHead
    from phaseforge.models.components.router import TopKRouter
    from phaseforge.models.phase_moe import PhaseBootstrappedMoE

    encoder = StateEncoder(input_dim=10, hidden_dims=[16], latent_dim=8)
    head = ActionHead(input_dim=8, output_dim=7, hidden_dim=16)
    phase_head = PhaseClassificationHead(latent_dim=8, num_phases=3)
    router = TopKRouter(latent_dim=8, num_experts=3, top_k=1, normalize_input=True)
    expert = ExpertMLP(input_dim=8, hidden_dims=[16], output_dim=7)
    return PhaseBootstrappedMoE(
        encoder=encoder, action_head=head, phase_head=phase_head,
        router=router, expert=expert,
    )


def test_deployment_contract_phase_moe_reports_router() -> None:
    contract = _small_phase_moe().deployment_contract()
    assert contract["memoryless"] is True
    assert contract["router_type"] == "TopKRouter"
    assert contract["expert_type"] == "direct"
    assert contract["top_k"] == 1
    assert contract["num_experts"] == 3


def test_phase_moe_get_action_stays_memoryless_cpu() -> None:
    model = _small_phase_moe()
    model.eval()
    state = torch.randn(2, 10)
    with torch.no_grad():
        first = model.get_action(state)
        second = model.get_action(state)
    assert first.shape == (2, 7)
    assert torch.equal(first, second)


def _write_yaml(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_freeze_check_passes_identical(tmp_path: Path) -> None:
    fc = _load_freeze_check()
    text = (
        "data:\n  source:\n    task_name: Can\n  state_dim: 23\n  action_dim: 7\n"
        "eval:\n  mode: rollout\n  bank:\n    seed: 2026\n    num_cases: 50\n"
        "  episodes:\n    horizon: null\nproject:\n  seed: 42\n"
    )
    a_path = _write_yaml(tmp_path / "a.yaml", text)
    b_path = _write_yaml(tmp_path / "b.yaml", text.replace("seed: 42", "seed: 43"))
    assert fc.compare_resolved_configs(a_path, b_path) == []


def test_freeze_check_fails_on_task_or_bank_mismatch(tmp_path: Path) -> None:
    fc = _load_freeze_check()
    base = (
        "data:\n  source:\n    task_name: Can\n  state_dim: 23\n  action_dim: 7\n"
        "eval:\n  mode: rollout\n  bank:\n    seed: 2026\n    num_cases: 50\n"
        "  episodes:\n    horizon: null\nproject:\n  seed: 42\n"
    )
    a_path = _write_yaml(tmp_path / "a.yaml", base)
    b_path = _write_yaml(tmp_path / "b.yaml", base.replace("task_name: Can", "task_name: Square"))
    mismatches = fc.compare_resolved_configs(a_path, b_path)
    assert any("task_name" in item for item in mismatches)
    c_path = _write_yaml(tmp_path / "c.yaml", base.replace("num_cases: 50", "num_cases: 20"))
    assert fc.compare_resolved_configs(a_path, c_path) != []


def test_freeze_check_skips_bank_pins_when_both_absent(tmp_path: Path) -> None:
    """Training configs (eval=metrics) carry no bank section: no pin to check."""
    fc = _load_freeze_check()
    text = (
        "data:\n  source:\n    task_name: Can\n  state_dim: 23\n  action_dim: 7\n"
        "eval:\n  mode: offline\nproject:\n  seed: 42\n"
    )
    a_path = _write_yaml(tmp_path / "a.yaml", text)
    b_path = _write_yaml(tmp_path / "b.yaml", text.replace("seed: 42", "seed: 43"))
    assert fc.compare_resolved_configs(a_path, b_path) == []


def test_real_rollout_yaml_has_trace_level() -> None:
    cfg = OmegaConf.load(str(REPO_ROOT / "phaseforge" / "config" / "eval" / "rollout.yaml"))
    assert cfg["episodes"]["trace_level"] == "minimal"
    assert cfg["episodes"]["trace_every_n_steps"] == 1

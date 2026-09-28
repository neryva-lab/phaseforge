"""Checkpoint contract gating.

Covers:
* resolver config-hash gating (only matching runs are eligible; runs
  without a recorded hash are rejected when the gate is on);
* ``verify_checkpoint_contract`` (expert count, model tree, stage,
  unreadable-artifact fail-closed);
* end-to-end fail-closed rejection of a pre-final 8-expert
  canonical-model artifact through both runner funnels (stage-2 prerequisite
  and evaluation target): the runner must never silently consume or fall
  back to a wrong-contract checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from phaseforge.runner import cli as runner_cli
from phaseforge.runner.protocol import Method, Step
from phaseforge.runner.registry import RunnerState
from phaseforge.runner.resolver import (
    CheckpointError,
    resolve_stage_ckpt,
    verify_checkpoint_contract,
)


def _make_contract_run(
    base: Path,
    model: str,
    stage: int,
    name: str,
    seed: int,
    *,
    num_experts: int | None = None,
    config_hash: str | None = None,
    model_name: str | None = None,
    tag: str | None = None,
    completed: bool = True,
) -> Path:
    """Fabricate a completed run with a REAL, loadable checkpoint payload."""
    run_dir = base / model / f"stage{stage}" / f"seed{seed}" / name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state: dict[str, object] = {
        "encoder.hidden.0.weight": torch.zeros(4, 4),
        "action_head.trunk.0.weight": torch.zeros(4, 4),
    }
    if num_experts is not None:
        for i in range(num_experts):
            state[f"moe_layer.experts.{i}.hidden.0.weight"] = torch.zeros(4, 4)
    torch.save({"model_state_dict": state, "stage": stage}, ckpt_dir / "checkpoint_best.pt")
    meta: dict[str, object] = {
        "kind": "train",
        "model_name": model_name or model,
        "stage": stage,
        "seed": seed,
        "git_commit": "deadbeef",
        "tag": tag,
        "method": None,
    }
    if config_hash is not None:
        meta["config_hash"] = config_hash
    (run_dir / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    if completed:
        (run_dir.with_name(name + ".completed")).write_text("{}", encoding="utf-8")
    return run_dir


# ---------------------------------------------------------------------------
# Resolver config-hash gating
# ---------------------------------------------------------------------------


def test_config_hash_gate_selects_only_matching_run(tmp_path: Path) -> None:
    _make_contract_run(
        tmp_path, "precision_residual_phaseforge", 1, "2026-08-01_10-00-00_aaaa0001", 42,
        num_experts=6, config_hash="hash_a",
    )
    _make_contract_run(
        tmp_path, "precision_residual_phaseforge", 1, "2026-08-02_10-00-00_aaaa0002", 42,
        num_experts=6, config_hash="hash_b",
    )
    got = resolve_stage_ckpt(
        tmp_path, "precision_residual_phaseforge", 1, seed=42, tag=None, expected_config_hash="hash_b"
    )
    assert "aaaa0002" in str(got)
    got = resolve_stage_ckpt(
        tmp_path, "precision_residual_phaseforge", 1, seed=42, tag=None, expected_config_hash="hash_a"
    )
    assert "aaaa0001" in str(got)
    # An absent hash matches nothing — fail closed, no fallback to a
    # mismatching (even newer) artifact.
    with pytest.raises(CheckpointError, match="config_hash 'hash_c'"):
        resolve_stage_ckpt(
            tmp_path, "precision_residual_phaseforge", 1, seed=42, tag=None, expected_config_hash="hash_c"
        )


def test_config_hash_gate_rejects_runs_without_recorded_hash(tmp_path: Path) -> None:
    _make_contract_run(tmp_path, "precision_residual_phaseforge", 1, "2026-08-01_10-00-00_aaaa0001", 42)
    with pytest.raises(CheckpointError, match="config_hash"):
        resolve_stage_ckpt(
            tmp_path, "precision_residual_phaseforge", 1, seed=42, tag=None, expected_config_hash="hash_a"
        )
    # Without the gate, backwards-compatible behaviour is unchanged.
    assert resolve_stage_ckpt(tmp_path, "precision_residual_phaseforge", 1, seed=42, tag=None).is_file()


# ---------------------------------------------------------------------------
# verify_checkpoint_contract
# ---------------------------------------------------------------------------


def test_verify_contract_accepts_canonical_six_expert(tmp_path: Path) -> None:
    run = _make_contract_run(
        tmp_path, "precision_residual_phaseforge", 2, "2026-08-01_10-00-00_aaaa0001", 42,
        num_experts=6, config_hash="hash_a",
    )
    ckpt = run / "checkpoints" / "checkpoint_best.pt"
    summary = verify_checkpoint_contract(
        ckpt,
        expected_model_name="precision_residual_phaseforge",
        expected_num_experts=6,
        expected_stage=2,
    )
    assert summary["num_experts"] == 6
    assert summary["model_name"] == "precision_residual_phaseforge"
    assert summary["config_hash"] == "hash_a"


def test_verify_contract_rejects_legacy_eight_expert(tmp_path: Path) -> None:
    run = _make_contract_run(
        tmp_path, "precision_residual_phaseforge", 1, "2026-08-01_10-00-00_aaaa0001", 42, num_experts=8
    )
    ckpt = run / "checkpoints" / "checkpoint_best.pt"
    with pytest.raises(CheckpointError, match="8 experts.*requires 6"):
        verify_checkpoint_contract(ckpt, expected_num_experts=6)


def test_verify_contract_rejects_wrong_model_tree(tmp_path: Path) -> None:
    run = _make_contract_run(
        tmp_path, "precision_residual_phaseforge", 1, "2026-08-01_10-00-00_aaaa0001", 42,
        num_experts=6, model_name="something_else",
    )
    ckpt = run / "checkpoints" / "checkpoint_best.pt"
    with pytest.raises(CheckpointError, match="belongs to model 'something_else'"):
        verify_checkpoint_contract(ckpt, expected_model_name="precision_residual_phaseforge")


def test_verify_contract_skips_expert_check_for_dense_checkpoints(tmp_path: Path) -> None:
    run = _make_contract_run(tmp_path, "bc", 1, "2026-08-01_10-00-00_aaaa0001", 42)
    ckpt = run / "checkpoints" / "checkpoint_best.pt"
    summary = verify_checkpoint_contract(
        ckpt, expected_model_name="bc", expected_num_experts=6, expected_stage=1
    )
    assert summary["num_experts"] is None


def test_verify_contract_rejects_unreadable_checkpoint(tmp_path: Path) -> None:
    run = _make_contract_run(tmp_path, "precision_residual_phaseforge", 1, "2026-08-01_10-00-00_aaaa0001", 42)
    ckpt = run / "checkpoints" / "checkpoint_best.pt"
    ckpt.write_text("not a torch file", encoding="utf-8")
    with pytest.raises(CheckpointError, match="cannot be loaded"):
        verify_checkpoint_contract(ckpt, expected_num_experts=6)


# ---------------------------------------------------------------------------
# Fail-closed through the runner funnels
# ---------------------------------------------------------------------------


def _proposed_method() -> Method:
    return Method(
        index=1,
        name="precision_residual_phaseforge",
        role="proposed method",
        model="precision_residual_phaseforge",
        data="lift",
        stages=(1, 2),
        stage2_source="self",
        evaluate=True,
        task="Lift",
    )


def test_stage2_prereq_fails_closed_on_wrong_expert_artifact(tmp_path: Path) -> None:
    """An 8-expert artifact for the canonical model must be rejected.

    The runner must fail closed instead of consuming a wrong-contract
    checkpoint from the canonical model namespace.
    """
    _make_contract_run(
        tmp_path, "precision_residual_phaseforge", 1, "2026-01-01_10-00-00_wrong0001", 42,
        num_experts=8, tag="Lift",
    )
    step = Step(kind="train", method=_proposed_method(), seed=42, stage=2)
    with pytest.raises(CheckpointError, match="experts"):
        runner_cli._require_stage2_prereq(step, tmp_path)


def test_eval_target_fails_closed_on_wrong_expert_artifact(tmp_path: Path) -> None:
    _make_contract_run(
        tmp_path, "precision_residual_phaseforge", 2, "2026-01-01_10-00-00_wrong0001", 42,
        num_experts=8, tag="Lift",
    )
    state = RunnerState(RunnerState.default_path(tmp_path))
    step = Step(kind="eval", method=_proposed_method(), seed=42)
    with pytest.raises(CheckpointError, match="experts"):
        runner_cli._eval_target(step, tmp_path, state)


def test_eval_target_accepts_canonical_artifact(tmp_path: Path) -> None:
    _make_contract_run(
        tmp_path, "precision_residual_phaseforge", 2, "2026-08-01_10-00-00_aaaa0001", 42,
        num_experts=6, tag="Lift",
    )
    state = RunnerState(RunnerState.default_path(tmp_path))
    step = Step(kind="eval", method=_proposed_method(), seed=42)
    ckpt = runner_cli._eval_target(step, tmp_path, state)
    assert ckpt.is_file()
    assert "aaaa0001" in str(ckpt)


def test_stage2_prereq_accepts_canonical_artifact(tmp_path: Path) -> None:
    _make_contract_run(
        tmp_path, "precision_residual_phaseforge", 1, "2026-08-01_10-00-00_aaaa0001", 42,
        num_experts=6, tag="Lift",
    )
    step = Step(kind="train", method=_proposed_method(), seed=42, stage=2)
    ckpt = runner_cli._require_stage2_prereq(step, tmp_path)
    assert ckpt is not None and "aaaa0001" in str(ckpt)


def test_method_expected_num_experts_property() -> None:
    assert _proposed_method().expected_num_experts == 6

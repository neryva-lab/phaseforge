"""Final provider gates (ledger TEST-12) and namespace freshness (DRY-01 helper).

TEST-12 pins every provider failure mode against fabricated provider
trees: missing, wrong-seed, wrong-task, wrong-commit, and wrong-hash
providers all fail before any subprocess launches, while the honest tree
resolves. The config-hash case runs through the runner's own
expected-hash composer against the REAL final manifest, so the wiring it
exercises is the production wiring.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from phaseforge.runner import cli as runner_cli
from phaseforge.runner.protocol import Method, ProtocolError, load_protocol, verify_output_namespace
from phaseforge.runner.resolver import CheckpointError

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "experiments" / "final_causal_matrix.json"


def _protocol():
    return load_protocol(MANIFEST)


def _write_tree(
    base: Path,
    model: str,
    run_name: str,
    *,
    seed: int,
    tag,
    task: str | None,
    config_hash: str | None,
    commit: str = "abc123",
    num_experts: int | None = None,
) -> Path:
    run_dir = base / model / "stage1" / f"seed{seed}" / run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    state: dict = {
        "encoder.hidden.0.weight": torch.zeros(2, 2),
        "action_head.trunk.0.weight": torch.zeros(2, 2),
    }
    if num_experts is not None:
        for i in range(num_experts):
            state[f"moe_layer.experts.{i}.hidden.0.weight"] = torch.zeros(2, 2)
    torch.save(
        {"model_state_dict": state, "stage": 1}, run_dir / "checkpoints" / "checkpoint_best.pt"
    )
    if task is not None:
        OmegaConf.save(
            {"data": {"source": {"task_name": task}}, "project": {"seed": seed}},
            str(run_dir / "resolved_config.yaml"),
        )
    meta: dict = {
        "kind": "train",
        "model_name": model,
        "stage": 1,
        "seed": seed,
        "git_commit": commit,
        "tag": tag,
        "method": None,
        "device": "cpu",
    }
    if config_hash is not None:
        meta["config_hash"] = config_hash
    (run_dir / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (run_dir.with_name(run_name + ".completed")).write_text("{}", encoding="utf-8")
    return run_dir


def _consumer() -> Method:
    return _protocol().method_by_name("precision_residual_plain_encoder", task="Lift")


def _provider() -> Method:
    return _protocol().method_by_name("bc", task="Lift")


def _honest_tree(base: Path) -> str:
    """Build a provider tree whose recorded hash equals the runner's expectation."""
    protocol = _protocol()
    provider = _provider()
    expected = runner_cli._expected_stage1_config_hash(
        protocol, provider, 42, base.resolve(), "cpu"
    )
    _write_tree(
        base, "final_aligned_bc", "2026-09-08_honest01", seed=42, tag="Lift",
        task="Lift", config_hash=expected,
    )
    return expected


def test_honest_provider_resolves(tmp_path: Path) -> None:
    base = tmp_path / "out"
    _honest_tree(base)
    step = runner_cli.Step(kind="train", method=_consumer(), seed=42, stage=2)
    ckpt = runner_cli._require_stage2_prereq(step, base, protocol=_protocol())
    assert ckpt is not None and "honest01" in str(ckpt)


def test_missing_provider_fails_before_subprocess(tmp_path: Path) -> None:
    step = runner_cli.Step(kind="train", method=_consumer(), seed=42, stage=2)
    with pytest.raises(CheckpointError, match="needs a final_aligned_bc stage 1 checkpoint"):
        runner_cli._require_stage2_prereq(step, tmp_path, protocol=_protocol())


def test_wrong_seed_provider_fails(tmp_path: Path) -> None:
    _write_tree(
        tmp_path, "final_aligned_bc", "2026-09-08_seed43", seed=43, tag="Lift",
        task="Lift", config_hash="x",
    )
    step = runner_cli.Step(kind="train", method=_consumer(), seed=42, stage=2)
    with pytest.raises(CheckpointError, match="seed 42"):
        runner_cli._require_stage2_prereq(step, tmp_path, protocol=_protocol())


def test_wrong_task_provider_fails(tmp_path: Path) -> None:
    _write_tree(
        tmp_path, "final_aligned_bc", "2026-09-08_can01", seed=42, tag="Can",
        task="Can", config_hash="x",
    )
    # A Lift consumer must never consume the Can provider, even seed-exact.
    _write_tree(
        tmp_path, "final_aligned_bc", "2026-09-08_liftag", seed=42, tag="Lift",
        task="Can", config_hash="y",
    )
    step = runner_cli.Step(kind="train", method=_consumer(), seed=42, stage=2)
    with pytest.raises(CheckpointError, match="Can.*Lift|Lift.*Can"):
        runner_cli._require_stage2_prereq(step, tmp_path, protocol=_protocol())


def test_wrong_commit_provider_fails(tmp_path: Path) -> None:
    _write_tree(
        tmp_path, "final_aligned_bc", "2026-09-08_old01", seed=42, tag="Lift",
        task="Lift", config_hash="x", commit="oldcommit",
    )
    step = runner_cli.Step(kind="train", method=_consumer(), seed=42, stage=2)
    with pytest.raises(CheckpointError):
        runner_cli._require_stage2_prereq(
            step, tmp_path, expected_commit="newcommit", protocol=_protocol()
        )


def test_tampered_config_hash_fails(tmp_path: Path) -> None:
    base = tmp_path / "out"
    expected = _honest_tree(base)
    assert expected
    run = base / "final_aligned_bc" / "stage1" / "seed42" / "2026-09-08_honest01"
    meta = json.loads((run / "run_meta.json").read_text(encoding="utf-8"))
    meta["config_hash"] = "0" * 16
    (run / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    step = runner_cli.Step(kind="train", method=_consumer(), seed=42, stage=2)
    with pytest.raises(CheckpointError, match="config mismatch"):
        runner_cli._require_stage2_prereq(step, base, protocol=_protocol())


def test_final_provider_without_sidecar_fails_strict(tmp_path: Path) -> None:
    _write_tree(
        tmp_path, "final_aligned_bc", "2026-09-08_noside", seed=42, tag="Lift",
        task=None, config_hash="x",
    )
    step = runner_cli.Step(kind="train", method=_consumer(), seed=42, stage=2)
    with pytest.raises(CheckpointError, match="resolved_config"):
        runner_cli._require_stage2_prereq(step, tmp_path, protocol=_protocol())


def test_output_namespace_gate(tmp_path: Path) -> None:
    """DRY-01 helper: absent passes, occupied fails with an itemized error."""
    fresh = tmp_path / "newns"
    assert verify_output_namespace(fresh).is_absolute()
    (fresh / "m" / "stage1").mkdir(parents=True)
    (fresh / "m" / "stage1" / "run.completed").write_text("{}", encoding="utf-8")
    with pytest.raises(ProtocolError, match="not fresh"):
        verify_output_namespace(fresh)

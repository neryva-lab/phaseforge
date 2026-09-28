"""Tests for robust normalizer and cache resolution across git revisions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from omegaconf import DictConfig

from phaseforge.evaluations.rollout.runner import (
    resolve_cache_dir,
    resolve_rollout_normalizer,
    run_rollout_evaluation,
)
from phaseforge.models.base import BaseManipulationModel


class _DummyModel(BaseManipulationModel):
    """Minimal manipulation model inheriting normalizer buffer management."""

    def __init__(self, obs_dim: int = 10, action_dim: int = 7) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(obs_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _base_cfg(tmp_path: Path, task: str = "Can") -> DictConfig:
    return DictConfig(
        {
            "project": {"seed": 42, "tag": task},
            "data": {
                "source": {
                    "task_name": task,
                    "dir": str(tmp_path / "raw"),
                },
                "action_dim": 7,
            },
            "train": {"stage": 2},
            "eval": {
                "mode": "rollout",
                "bank": {"seed": 2026, "num_cases": 1},
                "env": {"mujoco_requirement": ">=3.2.7", "allow_dev_fallback": True},
                "episodes": {
                    "horizon": 10,
                    "router_mode": "learned",
                    "action_tolerance": 1e-4,
                },
            },
            "models": {
                "name": "dummy_model",
                "_target_": "tests.evaluations.rollout.test_normalizer_resolution._DummyModel",
            },
        }
    )


def test_resolve_rollout_normalizer_from_model_buffers(tmp_path: Path) -> None:
    """Normalizer is obtained directly from model persistent buffers without disk access."""
    cfg = _base_cfg(tmp_path)
    model = _DummyModel(obs_dim=5)
    mean_expected = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    std_expected = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    model.set_normalizer_stats(mean_expected, std_expected)

    # Point to a nonexistent cache directory
    norm = resolve_rollout_normalizer(cfg, model=model, cache_dir=tmp_path / "nonexistent")
    assert torch.allclose(norm.mean, mean_expected)
    assert torch.allclose(norm.std, std_expected)


def test_resolve_rollout_normalizer_from_cache_dir(tmp_path: Path) -> None:
    """Falls back to norm_stats.pt on disk when model has no registered buffers."""
    cfg = _base_cfg(tmp_path)
    model = torch.nn.Linear(5, 7)  # Plain Module, no get_normalizer_stats

    cache_dir = tmp_path / "valid_cache"
    cache_dir.mkdir(parents=True)
    mean_disk = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
    std_disk = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
    torch.save({"mean": mean_disk, "std": std_disk}, cache_dir / "norm_stats.pt")

    norm = resolve_rollout_normalizer(cfg, model=model, cache_dir=cache_dir)
    assert torch.allclose(norm.mean, mean_disk)
    assert torch.allclose(norm.std, std_disk)


def test_resolve_cache_dir_recovers_from_checkpoint_run_meta(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When git revision causes hash mismatch, cache is recovered from checkpoint's
    run_meta.json."""
    cache_root = tmp_path / "processed_cache"
    cache_root.mkdir(parents=True)
    monkeypatch.setattr(
        "phaseforge.data.paths.processed_cache_root",
        lambda: cache_root,
    )

    # Simulate old training run at hash "train_hash_abc123"
    train_cache = cache_root / "train_hash_abc123"
    train_cache.mkdir()
    torch.save({"mean": torch.zeros(5), "std": torch.ones(5)}, train_cache / "norm_stats.pt")
    (train_cache / "trajectories").mkdir()

    # Create run output directory with run_meta.json
    run_dir = (
        tmp_path / "outputs" / "my_model" / "stage2" / "seed42" / "2026-09-05_06-00-00_Can_c8463149"
    )
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    ckpt_path = checkpoints_dir / "checkpoint_best.pt"
    ckpt_path.touch()

    run_meta = {
        "data_config_hash": "train_hash_abc123",
        "git_commit": "c8463149",
        "seed": 42,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(run_meta))

    cfg = _base_cfg(tmp_path, task="Can")
    cfg.train.stage1_ckpt_path = str(ckpt_path)

    resolved = resolve_cache_dir(cfg)
    assert resolved == train_cache
    assert (resolved / "norm_stats.pt").is_file()


def test_resolve_cache_dir_recovers_from_task_index(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When git hash differs and no checkpoint metadata is available, matches task_index.json."""
    cache_root = tmp_path / "processed_cache"
    cache_root.mkdir(parents=True)
    monkeypatch.setattr(
        "phaseforge.data.paths.processed_cache_root",
        lambda: cache_root,
    )

    can_cache = cache_root / "hash_can_9999"
    can_cache.mkdir()
    (can_cache / "task_index.json").write_text(json.dumps({"Can": 0, "can_pick": 0}))
    torch.save({"mean": torch.zeros(5), "std": torch.ones(5)}, can_cache / "norm_stats.pt")

    cfg = _base_cfg(tmp_path, task="Can")
    resolved = resolve_cache_dir(cfg)
    assert resolved == can_cache


def test_resolve_cache_dir_recovers_from_manifest_task_names(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Recovers cache directory matching task in manifest.json split_task_names."""
    cache_root = tmp_path / "processed_cache"
    cache_root.mkdir(parents=True)
    monkeypatch.setattr(
        "phaseforge.data.paths.processed_cache_root",
        lambda: cache_root,
    )

    square_cache = cache_root / "hash_square_8888"
    square_cache.mkdir()
    manifest_data = {
        "provenance": {
            "split_task_names": {
                "train": ["square_task_demo_0"],
                "val": ["square_task_demo_1"],
            }
        }
    }
    (square_cache / "manifest.json").write_text(json.dumps(manifest_data))
    torch.save({"mean": torch.zeros(5), "std": torch.ones(5)}, square_cache / "norm_stats.pt")

    cfg = _base_cfg(tmp_path, task="Square")
    resolved = resolve_cache_dir(cfg)
    assert resolved == square_cache


def test_resolve_rollout_normalizer_fails_closed_when_nothing_found(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Fails closed with FileNotFoundError when model has no stats and no cache exists."""
    empty_root = tmp_path / "empty_cache"
    empty_root.mkdir(parents=True)
    monkeypatch.setattr(
        "phaseforge.data.paths.processed_cache_root",
        lambda: empty_root,
    )

    cfg = _base_cfg(tmp_path, task="NonexistentTask")
    with pytest.raises(FileNotFoundError, match="Could not load normalizer statistics"):
        resolve_rollout_normalizer(cfg, model=None)


def test_run_rollout_evaluation_succeeds_with_missing_disk_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Full rollout evaluation succeeds even when disk cache is completely missing, using model
    buffers."""
    empty_root = tmp_path / "missing_cache"
    empty_root.mkdir(parents=True)
    monkeypatch.setattr(
        "phaseforge.data.paths.processed_cache_root",
        lambda: empty_root,
    )

    class _MockAdapter:
        action_dim = 7
        def __init__(self):
            self.closed = False
        def reset_to(self, states, xml=None, ep_meta=None):
            return torch.zeros(10).numpy()
        def step(self, action):
            return torch.zeros(10).numpy(), False, True, {}
        def validate_action(self, action, tolerance=1e-4):
            return action
        def close(self):
            self.closed = True

    class _MockCase:
        index = 0
        states = [torch.zeros(10).numpy()]
        xml = ""
        ep_meta = {}

    class _MockBank:
        bank_id = "mock_bank_123"
        seed = 2026
        cases = [_MockCase()]

    monkeypatch.setattr(
        "phaseforge.evaluations.rollout.runner._adapter_from_config",
        lambda cfg, meta: _MockAdapter(),
    )
    monkeypatch.setattr(
        "phaseforge.evaluations.rollout.runner.load_or_generate_bank",
        lambda cfg, meta: _MockBank(),
    )

    cfg = _base_cfg(tmp_path, task="Can")
    cfg.eval.env.allow_dev_fallback = True

    model = _DummyModel(obs_dim=10, action_dim=7)
    model.set_normalizer_stats(torch.ones(10) * 1.5, torch.ones(10) * 0.8)

    results = run_rollout_evaluation(
        cfg,
        model,
        output_dir=tmp_path / "eval_out",
        run_id="test-run-res",
    )
    assert results is not None
    assert results.get("eval/rollout/success_rate") == 1.0

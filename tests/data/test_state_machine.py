"""Tests for the data pipeline FSM's phase-label consumption guard.

The phase-count guard distinguishes models that consume phase labels
(phase_head cross-entropy / privileged oracle routing / centroid bootstrap)
from label-free rows (BC pilot, scratch/warm-start MoE). Degenerate labels
must fail loudly for the former and only warn for the latter.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

from phaseforge.data.ingestion.state_machine import (
    DataPipelineStateMachine,
    PipelineError,
)

PHASE_CONSUMING_MODELS = {
    "precision_residual_phaseforge",
    "precision_residual_teacher_forced",
    "precision_residual_oracle",
    "precision_residual_plain_encoder",
    "precision_residual_factorial_floor",
    "precision_residual_phase_random_router",
    "precision_residual_scratch_moe",
}
LABEL_FREE_MODELS = {
    "final_aligned_bc",
}


def _fsm_for_model(model: str) -> DataPipelineStateMachine:
    data_cfg = OmegaConf.load("phaseforge/config/data/common.yaml")
    path = f"phaseforge/config/models/{model}.yaml"
    return DataPipelineStateMachine(DictConfig({"models": OmegaConf.load(path), "data": data_cfg}))


def test_phase_consuming_models_are_detected() -> None:
    for name in PHASE_CONSUMING_MODELS:
        assert _fsm_for_model(name)._model_uses_phase_labels(), name


def test_soft_mapping_init_allows_experts_ne_phases() -> None:
    """A soft-mapping init with P=6 phases, E=8 experts passes the guard.

    The P x E soft mapping bridges phase index to expert index, so
    num_experts must be decoupled from num_phases for this router init.
    The canonical ``phaseforge`` config is centroid/E=6 (E == P), so the
    E != P case is constructed explicitly here from the canonical config.
    """
    data_cfg = OmegaConf.load("phaseforge/config/data/common.yaml")
    models_cfg = OmegaConf.load(
        "phaseforge/config/models/precision_residual_phaseforge.yaml"
    )
    models_cfg.router_init.type = "soft_mapping"
    models_cfg.soft_mapping.enabled = True
    models_cfg.router.num_experts = 8
    fsm = DataPipelineStateMachine(
        DictConfig({"models": models_cfg, "data": data_cfg})
    )
    assert int(fsm.cfg.models.router.num_experts) == 8
    assert int(fsm.cfg.data.phase_labeler.num_phases) == 6
    # Construction already ran the guard; reach it explicitly for the assert.
    fsm._check_num_phases_consistency()


def test_non_soft_mapping_init_still_requires_match() -> None:
    """Without soft_mapping/centroid/random init, E != P must fail loudly."""
    data_cfg = OmegaConf.load("phaseforge/config/data/common.yaml")
    models_cfg = OmegaConf.load(
        "phaseforge/config/models/precision_residual_phaseforge.yaml"
    )
    models_cfg.router_init.type = "banana"
    # The canonical config has E == P == 6, which the guard would accept for
    # any init type; force E != P so the strict-matching requirement fires.
    models_cfg.router.num_experts = 8
    with pytest.raises(PipelineError, match="num_phases inconsistency"):
        DataPipelineStateMachine(
            DictConfig({"models": models_cfg, "data": data_cfg})
        )


def test_label_free_models_are_not_phase_consuming() -> None:
    for name in LABEL_FREE_MODELS:
        assert not _fsm_for_model(name)._model_uses_phase_labels(), name


class _FakeIngester:
    """Minimal ingester returning degenerate (all-zero) phase labels."""

    def __init__(self, raw_dir: str | Path) -> None:
        self.raw_dir = Path(raw_dir)

    def ingest(self) -> tuple[list[dict[str, Any]], dict[str, int]]:
        traj: dict[str, Any] = {
            "state": np.zeros((10, 19), dtype=np.float32),
            "action": np.zeros((10, 7), dtype=np.float32),
            "phase": np.zeros(10, dtype=np.int64),
            "task_id": 0,
        }
        return [traj], {"lift": 0}


def _fsm_with_fake_ingester(model: str) -> DataPipelineStateMachine:
    fsm = _fsm_for_model(model)
    fsm.data_cfg.ingester = DictConfig({"_target_": _FakeIngester}, flags={"allow_objects": True})
    fsm._raw_dir = Path("fake-raw-dir")
    return fsm


def test_degenerate_phases_fail_loud_for_phase_consuming_models() -> None:
    fsm = _fsm_with_fake_ingester("precision_residual_phaseforge")
    with pytest.raises(PipelineError, match="no samples for phase"):
        fsm._ingest_source()


def test_degenerate_phases_only_warn_for_label_free_models(caplog) -> None:
    fsm = _fsm_with_fake_ingester("bc")
    with caplog.at_level(logging.WARNING, logger="phaseforge.data.ingestion.state_machine"):
        fsm._ingest_source()
    assert "no samples for phase" in caplog.text
    assert fsm._trajectories


def _fsm_with_missing_source(tmp_path: Path, auto_download: bool) -> DataPipelineStateMachine:
    data_cfg = OmegaConf.load("phaseforge/config/data/common.yaml")
    data_cfg.source.dir = str(tmp_path / "no_such_raw")
    data_cfg.source.auto_download = auto_download
    return DataPipelineStateMachine(
        DictConfig(
            {
                "models": OmegaConf.load(
                    "phaseforge/config/models/precision_residual_phaseforge.yaml"
                ),
                "data": data_cfg,
            }
        )
    )


def test_missing_source_without_auto_download_raises(tmp_path) -> None:
    fsm = _fsm_with_missing_source(tmp_path, auto_download=False)
    with pytest.raises(PipelineError, match="Raw source directory not found"):
        fsm._validate_source()


def test_missing_source_with_auto_download_enters_provision_state(tmp_path) -> None:
    from phaseforge.data.ingestion.states import PipelineState

    fsm = _fsm_with_missing_source(tmp_path, auto_download=True)
    fsm._validate_source()
    assert fsm._state == PipelineState.PROVISION_SOURCE
    assert fsm._raw_dir is not None


def test_provision_source_downloads_verified_file(tmp_path, monkeypatch) -> None:
    from phaseforge.data.ingestion.states import PipelineState

    fsm = _fsm_with_missing_source(tmp_path, auto_download=True)
    raw_dir = fsm._resolve_raw_dir()
    downloaded = raw_dir / "low_dim_v15.hdf5"
    downloaded.parent.mkdir(parents=True)
    downloaded.write_bytes(b"fake-hdf5")

    calls: list[tuple] = []

    def fake_download(repo_id, path, dest_dir, pinned_sha256=None):
        calls.append((repo_id, path, Path(dest_dir), pinned_sha256))
        return downloaded

    monkeypatch.setattr("phaseforge.data.ingestion.state_machine.download_hf_file", fake_download)
    fsm._provision_source()
    assert calls == [
        (
            "amandlek/robomimic",
            "v1.5/lift/ph/low_dim_v15.hdf5",
            raw_dir,
            None,
        )
    ]
    assert fsm._state == PipelineState.INGEST_AND_STRIP


def test_provision_source_requires_hf_config(tmp_path) -> None:
    fsm = _fsm_with_missing_source(tmp_path, auto_download=True)
    fsm.data_cfg.source.pop("huggingface")
    with pytest.raises(PipelineError, match="requires data.source.huggingface"):
        fsm._provision_source()


class TestTrainSamplerGenerator:
    """The train DataLoader must use an explicit CPU ``torch.Generator`` seeded
    from ``cfg.project.seed`` so the per-epoch sample order is reproducible
    from the project seed alone — without depending on the global torch RNG
    state at the time the DataLoader is built."""

    def _make_fsm(self, project_seed: int) -> DataPipelineStateMachine:
        data_cfg = OmegaConf.load("phaseforge/config/data/common.yaml")
        bc_target = "phaseforge.models.baselines.bc.BehaviorCloningModel"
        cfg = DictConfig(
            {
                "models": {"name": "bc", "_target_": bc_target},
                "data": data_cfg,
                "project": {"seed": project_seed},
            }
        )
        return DataPipelineStateMachine(cfg)

    def test_returns_cpu_generator_seeded_from_project_seed(self) -> None:
        import torch

        fsm = self._make_fsm(project_seed=42)
        gen = fsm._train_sampler_generator()
        assert isinstance(gen, torch.Generator)
        # CPU placement is required because DataLoader workers cannot share
        # CUDA generators across the multiprocessing boundary.
        assert gen.device.type == "cpu"
        # The initial seed matches the project seed exactly so the sample
        # order is reproducible from the project seed alone.
        assert int(gen.initial_seed()) == 42

    def test_cached_generator_returned_on_second_call(self) -> None:
        fsm = self._make_fsm(project_seed=7)
        first = fsm._train_sampler_generator()
        second = fsm._train_sampler_generator()
        # Identity check: caching avoids regenerating and re-seeding, which
        # matters for resume correctness — a fresh generator would consume
        # additional entropy and break determinism.
        assert first is second

    def test_different_project_seeds_produce_different_generators(self) -> None:
        fsm_42 = self._make_fsm(project_seed=42)
        fsm_99 = self._make_fsm(project_seed=99)
        assert int(fsm_42._train_sampler_generator().initial_seed()) == 42
        assert int(fsm_99._train_sampler_generator().initial_seed()) == 99

    def test_non_integer_project_seed_rejected(self) -> None:
        fsm = self._make_fsm(project_seed=42)
        fsm.cfg.project.seed = "not-an-int"  # type: ignore[assignment]
        with pytest.raises(PipelineError, match="must be an integer"):
            fsm._train_sampler_generator()


class TestSplitSeedIndependence:
    """The train/val split is intentionally seeded from ``data.split.seed``
    and **not** from ``cfg.project.seed``. Two FSMs sharing the same data
    config but different project seeds must produce identical train/val
    partitions so seed sweeps compare like-for-like."""

    @staticmethod
    def _make_fsm(project_seed: int) -> DataPipelineStateMachine:
        data_cfg = OmegaConf.load("phaseforge/config/data/common.yaml")
        bc_target = "phaseforge.models.baselines.bc.BehaviorCloningModel"
        cfg = DictConfig(
            {
                "models": {"name": "bc", "_target_": bc_target},
                "data": data_cfg,
                "project": {"seed": project_seed},
            }
        )
        return DataPipelineStateMachine(cfg)

    @staticmethod
    def _inject_trajectories(fsm: DataPipelineStateMachine) -> None:
        """Populate ``_trajectories`` with 30 single-task demos so
        ``_build_task_level_splits`` can run without the full ingest
        pipeline. Each demo is minimal — only ``task_id`` is required by
        the splitter."""
        fsm._trajectories = [{"task_id": 0} for _ in range(30)]

    def test_split_identical_across_project_seeds(self) -> None:
        fsm_42 = self._make_fsm(project_seed=42)
        fsm_99 = self._make_fsm(project_seed=99)
        self._inject_trajectories(fsm_42)
        self._inject_trajectories(fsm_99)
        splits_42 = fsm_42._build_task_level_splits()
        splits_99 = fsm_99._build_task_level_splits()
        # Identical train/val partition: a model-seed change does not move
        # the data boundary, so the val curves remain comparable across
        # the seed sweep.
        assert splits_42["train"] == splits_99["train"]
        assert splits_42["val"] == splits_99["val"]

    def test_split_seed_isolated_from_global_numpy_rng(self) -> None:
        """``np.random.default_rng(split_seed)`` is independent of the global
        numpy RNG, so polluting ``np.random`` state before the split must
        not change the partition."""
        fsm_a = self._make_fsm(project_seed=42)
        fsm_b = self._make_fsm(project_seed=42)
        self._inject_trajectories(fsm_a)
        self._inject_trajectories(fsm_b)
        # Pollute the global numpy state on one of them.
        np.random.seed(12345)
        _ = np.random.rand(1024)
        splits_b = fsm_b._build_task_level_splits()
        # The other FSM has a clean global state.
        splits_a = fsm_a._build_task_level_splits()
        assert splits_a["train"] == splits_b["train"]
        assert splits_a["val"] == splits_b["val"]


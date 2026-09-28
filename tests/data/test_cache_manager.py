"""Cache identity + provenance tests (generic source schema).

The raw-dataset identity is read from the generic ``data.source`` block::

    data:
      source:
        dir: {data_root}/raw/{source}      # dataset files
        manifest_path: null                # null -> {dir}/MANIFEST.json

- compute_hash folds in cheap provenance context (git commit, raw dataset
  names/sizes, pinned dataset commit from MANIFEST.json) so a stale cache
  can never be silently reused.
- save() persists a full audit manifest: per-file SHA-256, git commit,
  schema, split task names — and writes human-readable split files.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf


def _data_cfg(source_dir: Path | None = None) -> OmegaConf:
    cfg = {
        "state_dim": 23,
        "action_dim": 7,
        "state_keys": [{"key": "robot0_joint_pos", "dim": 7}],
    }
    if source_dir is not None:
        cfg["source"] = {"dir": str(source_dir)}
    return OmegaConf.create(cfg)


def _write_dummy_hdf5(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"dummy-hdf5-content")


def _write_source_manifest(src_dir: Path, commit: str, files: list[dict]) -> None:
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "MANIFEST.json").write_text(
        json.dumps(
            {"source": "robomimic", "commit_sha": commit, "files": files},
            indent=2,
        )
    )


@pytest.fixture
def fake_data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point PHASEFORGE_DATA_DIR at a tmp root so source paths are local."""
    monkeypatch.setenv("PHASEFORGE_DATA_DIR", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# Cache identity (compute_hash)
# ---------------------------------------------------------------------------


def test_hash_changes_when_raw_dataset_changes(
    fake_data_root: Path,
) -> None:
    src_dir = fake_data_root / "raw" / "robomimic" / "lift"
    _write_dummy_hdf5(src_dir / "demo_1.hdf5")
    cfg = _data_cfg(source_dir=src_dir)

    from phaseforge.data.ingestion.cache_manager import CacheManager

    h1 = CacheManager.compute_hash(cfg)
    _write_dummy_hdf5(src_dir / "demo_2.hdf5")
    h2 = CacheManager.compute_hash(cfg)

    assert h1 != h2, "raw dataset change must invalidate the cache"


def test_hash_portable_across_machine_mtimes_when_manifest_present(
    fake_data_root: Path,
) -> None:
    """With MANIFEST.json present the identity is the pinned dataset commit
    — different download mtimes must NOT change the cache key (this is what
    makes a cache transferred to another machine reusable)."""
    import os
    import time

    from phaseforge.data.ingestion.cache_manager import CacheManager

    src_dir = fake_data_root / "raw" / "robomimic" / "lift"
    demo = src_dir / "demo_1.hdf5"
    _write_dummy_hdf5(demo)
    _write_source_manifest(
        src_dir,
        commit="f13aa24a" * 5,
        files=[{"name": "demo_1.hdf5", "size_bytes": len(b"dummy-hdf5-content")}],
    )
    cfg = _data_cfg(source_dir=src_dir)

    h1 = CacheManager.compute_hash(cfg)
    # Simulate a re-download on another machine: same content, new mtime.
    os.utime(demo, (time.time() - 86400, time.time() - 86400))
    h2 = CacheManager.compute_hash(cfg)

    assert h1 == h2, "mtimes must not enter the key when the manifest exists"


def test_hash_sensitive_to_manifest_commit(fake_data_root: Path) -> None:
    """A different pinned dataset revision must invalidate the cache."""
    from phaseforge.data.ingestion.cache_manager import CacheManager

    src_dir = fake_data_root / "raw" / "robomimic" / "lift"
    _write_dummy_hdf5(src_dir / "demo_1.hdf5")
    _write_source_manifest(src_dir, commit="a" * 40, files=[])
    cfg = _data_cfg(source_dir=src_dir)

    h1 = CacheManager.compute_hash(cfg)
    _write_source_manifest(src_dir, commit="b" * 40, files=[])
    h2 = CacheManager.compute_hash(cfg)

    assert h1 != h2, "dataset revision change must invalidate the cache"


def test_hash_reconstructs_identity_from_manifest_inventory(
    fake_data_root: Path,
) -> None:
    """Cache transfers omit the raw HDF5 files; the identity must be
    reconstructed from MANIFEST.json's file inventory so a machine without
    the raw data still produces the SAME cache key."""
    from phaseforge.data.ingestion.cache_manager import CacheManager

    src_dir = fake_data_root / "raw" / "robomimic" / "lift"
    _write_dummy_hdf5(src_dir / "demo_1.hdf5")
    _write_source_manifest(
        src_dir,
        commit="c" * 40,
        files=[{"name": "demo_1.hdf5", "size_bytes": len(b"dummy-hdf5-content")}],
    )
    cfg = _data_cfg(source_dir=src_dir)

    h_with_files = CacheManager.compute_hash(cfg)

    # Machine B: same config + manifest, raw files deleted.
    for f in src_dir.glob("*.hdf5"):
        f.unlink()
    h_without_files = CacheManager.compute_hash(cfg)

    assert h_with_files == h_without_files


def test_hash_sensitive_to_mtime_without_manifest(fake_data_root: Path) -> None:
    """Without MANIFEST.json the fallback mtime fingerprint applies."""
    import os
    import time

    from phaseforge.data.ingestion.cache_manager import CacheManager

    src_dir = fake_data_root / "raw" / "robomimic" / "lift"
    demo = src_dir / "demo_1.hdf5"
    _write_dummy_hdf5(demo)
    cfg = _data_cfg(source_dir=src_dir)

    h1 = CacheManager.compute_hash(cfg)
    os.utime(demo, (time.time() - 86400, time.time() - 86400))
    h2 = CacheManager.compute_hash(cfg)

    assert h1 != h2, "no-manifest fallback must stay mtime-sensitive"


def test_hash_stable_without_data(tmp_path: Path) -> None:
    from phaseforge.data.ingestion.cache_manager import CacheManager

    cfg = _data_cfg(source_dir=tmp_path / "missing_source")

    assert CacheManager.compute_hash(cfg) == CacheManager.compute_hash(cfg)


# ---------------------------------------------------------------------------
# Manifest provenance (save)
# ---------------------------------------------------------------------------


def test_save_manifest_records_full_provenance(tmp_path: Path) -> None:
    from phaseforge.data.ingestion.cache_manager import CacheManager

    cfg = _data_cfg()
    mgr = CacheManager(tmp_path)
    config_hash = CacheManager.compute_hash(cfg)

    provenance = {
        "dataset_manifest": {"source": "robomimic", "commit_sha": "a" * 40},
        "code_git_commit": "deadbeef",
        "raw_files": [{"name": "demo_1.hdf5", "size": 123, "sha256": "a" * 64}],
        "state_schema": {
            "keys": [{"key": "robot0_joint_pos", "dim": 7}],
            "state_dim": 23,
            "action_dim": 7,
        },
        "split_task_names": {
            "train": ["lift"],
            "val": ["can"],
            "eval": [],
        },
    }
    mgr.save(
        config_hash=config_hash,
        trajectories=[],
        norm_stats={},
        splits={"train": [], "val": [], "eval": []},
        task_index={"lift": 0, "can": 1},
        provenance=provenance,
    )

    cache_dir = mgr.cache_dir(config_hash)
    assert mgr.cache_exists(config_hash)
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    assert manifest["complete"] is True
    assert manifest["provenance"]["raw_files"][0]["sha256"] == "a" * 64
    assert manifest["provenance"]["code_git_commit"] == "deadbeef"
    assert manifest["provenance"]["dataset_manifest"]["commit_sha"] == "a" * 40

    train_tasks = (cache_dir / "train_tasks.txt").read_text().splitlines()
    val_tasks = (cache_dir / "validation_tasks.txt").read_text().splitlines()
    assert train_tasks == ["lift"]
    assert val_tasks == ["can"]


def test_save_without_provenance_still_works(tmp_path: Path) -> None:
    from phaseforge.data.ingestion.cache_manager import CacheManager

    mgr = CacheManager(tmp_path)
    config_hash = CacheManager.compute_hash(_data_cfg())
    mgr.save(
        config_hash=config_hash,
        trajectories=[],
        norm_stats={},
        splits={"train": [], "val": [], "eval": []},
    )
    assert mgr.cache_exists(config_hash)


def test_git_commit_helper_returns_sha(tmp_path: Path) -> None:
    from phaseforge.data.ingestion.cache_manager import git_commit

    commit = git_commit()
    # Inside a git repo this is a 40-hex SHA; elsewhere it degrades to "".
    assert commit == "" or len(commit) == 40


def test_git_commit_is_cached_per_process(monkeypatch) -> None:
    """The lru_cache must collapse repeated lookups into one git subprocess.

    The data-config identity derives git_commit() several times per training
    run; without the cache each derivation spawns its own ~150 ms subprocess.
    """
    import phaseforge.data.ingestion.cache_manager as cm

    calls = []
    real_run = cm.subprocess.run

    def counting_run(*args, **kwargs):
        calls.append(args)
        return real_run(*args, **kwargs)

    cm.git_commit.cache_clear()
    monkeypatch.setattr(cm.subprocess, "run", counting_run)
    try:
        first = cm.git_commit()
        second = cm.git_commit()
        third = cm.git_commit()
    finally:
        monkeypatch.setattr(cm.subprocess, "run", real_run)
    assert first == second == third
    assert len(calls) == 1, f"expected exactly one git subprocess, got {len(calls)}"


def test_sha256_file_streams_large_content(tmp_path: Path) -> None:
    from phaseforge.data.ingestion.cache_manager import sha256_file

    data = b"phaseforge-provenance" * 1000
    path = tmp_path / "blob.bin"
    path.write_bytes(data)
    expected = hashlib.sha256(data).hexdigest()
    assert sha256_file(path) == expected

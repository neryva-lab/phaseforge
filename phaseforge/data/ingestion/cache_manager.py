"""Config-hash-based persistent cache manager with full provenance tracking."""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def git_commit() -> str:
    """Best-effort HEAD SHA of the repo containing this module ('' when N/A).

    Cached per process: a training run never moves HEAD, and the value feeds
    the data-config identity which is derived several times per run (each
    derivation used to spawn its own ~150 ms git subprocess).
    """
    try:
        repo = Path(__file__).resolve().parents[3]
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:  # noqa: BLE001
        return ""


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file (handles multi-GB HDF5 files)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


class CacheManager:
    """Manages the persistent on-disk cache for processed dataset.

    Cache layout::

        cache_root/
        └── {config_hash}/
            ├── manifest.json
            ├── norm_stats.pt
            ├── splits.json
            ├── task_index.json
            ├── train_tasks.txt
            ├── validation_tasks.txt
            └── trajectories/
                ├── 000000.pt   # {"state": (T,S), "action": (T,A), "phase": (T,), "task_id": int}
                └── ...

    The config_hash is the SHA-256 of the canonical serialized data config
    PLUS a cheap provenance context (git commit, raw dataset file names/sizes
    from ``data.source``). Any change to phase thresholds, state keys, split
    ratios, code revision, or the raw dataset invalidates the cache. Full
    content provenance (per-file SHA-256, schema, task names) is persisted in
    ``manifest.json`` so a result can be audited later.
    """

    def __init__(self, cache_root: Path) -> None:
        self.cache_root = Path(cache_root)

    # ------------------------------------------------------------------
    # Hash
    # ------------------------------------------------------------------

    @staticmethod
    def provenance_context(data_cfg: DictConfig) -> dict:
        """Cheap inputs to the cache identity (never reads file contents).

        Folds in code revision and data identity so a stale cache cannot be
        silently reused after the raw dataset changes. Content-level hashes
        (expensive) live in the manifest instead.

        The raw-dataset identity is read from the generic ``data.source``
        block::

            data:
              source:
                dir: {data_root}/raw/{source}      # dataset files
                manifest_path: null                # null -> {dir}/MANIFEST.json

        When ``data.source`` is absent the key depends only on the data
        config + git commit (no raw fingerprint). When ``MANIFEST.json``
        pins a ``commit_sha`` the identity is portable across machines
        (download mtimes never enter the key); without a manifest the
        mtime fingerprint is a same-machine-only identity.
        """
        ctx: dict[str, Any] = {"git_commit": git_commit()}

        # Raw dataset identity: names + sizes (cheap). Content hashes are
        # recorded in the manifest during save(). When the source's
        # MANIFEST.json is present, its pinned dataset-revision commit SHA
        # is the identity — reproducible on ANY machine, unlike per-file
        # mtimes (which differ between downloads). The mtime fingerprint is
        # only a last resort for raw data that arrived without a manifest; a
        # cache keyed that way is NOT portable across machines.
        try:
            source = data_cfg.get("source")
            if source is not None and source.get("dir"):
                src_dir = Path(str(source["dir"]))
                manifest_path = Path(str(source.get("manifest_path") or src_dir / "MANIFEST.json"))
                files = sorted(src_dir.glob("*.hdf5")) if src_dir.exists() else []
                ctx["raw_files"] = [{"name": p.name, "size": p.stat().st_size} for p in files]

                download_manifest: dict[str, Any] | None = None
                if manifest_path.exists():
                    try:
                        download_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        logger.warning(
                            "Could not read dataset manifest %s while computing "
                            "the cache identity.",
                            manifest_path,
                        )
                if download_manifest is not None:
                    commit = download_manifest.get("commit_sha")
                    if commit:
                        ctx["dataset_commit"] = str(commit)

                    # Cache transfers intentionally omit the raw HDF5 files.
                    # Reconstruct the same cheap identity used when those
                    # files were present from MANIFEST.json's file inventory.
                    if not files:
                        manifest_files = download_manifest.get("files") or []
                        if manifest_files:
                            ctx["raw_files"] = [
                                {
                                    "name": str(entry["name"]),
                                    "size": int(entry["size_bytes"]),
                                }
                                for entry in manifest_files
                                if "name" in entry and "size_bytes" in entry
                            ]
                if "dataset_commit" not in ctx:
                    # No manifest: keep the mtime-based fingerprint as the
                    # same-machine identity (the documented workflow writes
                    # MANIFEST.json, so the portable path is the default).
                    ctx["raw_files"] = [
                        {
                            "name": p.name,
                            "size": p.stat().st_size,
                            "mtime_ns": p.stat().st_mtime_ns,
                        }
                        for p in files
                    ]
                if not src_dir.exists():
                    logger.warning(
                        "Raw source dir %s not found while computing the "
                        "cache identity — the cache key will not include "
                        "the raw dataset unless MANIFEST.json contains its "
                        "file inventory. If the data exists elsewhere, set "
                        "PHASEFORGE_DATA_DIR.",
                        src_dir,
                    )
        except Exception:  # noqa: BLE001
            logger.warning(
                "Could not fingerprint the raw dataset for the cache "
                "identity — the cache key will not depend on it.",
                exc_info=True,
            )
        return ctx

    @staticmethod
    def compute_hash(data_cfg: DictConfig, extra_context: dict | None = None) -> str:
        """SHA-256 of the data config YAML + provenance context (first 16 chars).

        ``extra_context`` (if given) is merged over the automatic
        :meth:`provenance_context` so callers can pin additional identity
        inputs.
        """
        context = CacheManager.provenance_context(data_cfg)
        if extra_context:
            context = {**context, **extra_context}
        config = OmegaConf.to_container(data_cfg, resolve=True, throw_on_missing=True)
        # The source location is machine-specific and must not make an
        # otherwise identical cache unusable after moving the data root. Raw
        # file identity is already represented by ``context`` above.
        if isinstance(config, dict):
            source_config = config.get("source")
            if isinstance(source_config, dict):
                if "dir" in source_config:
                    source_config["dir"] = "<data.source.dir>"
                if source_config.get("manifest_path") is not None:
                    source_config["manifest_path"] = "<data.source.manifest_path>"
        payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
        if context:
            payload += "\n" + json.dumps(context, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def cache_dir(self, config_hash: str) -> Path:
        return self.cache_root / config_hash

    # ------------------------------------------------------------------
    # Existence check and Fallback
    # ------------------------------------------------------------------

    def _is_valid_cache(self, config_hash: str) -> bool:
        """Return True only if the cache directory and a valid manifest exist."""
        manifest = self.cache_dir(config_hash) / "manifest.json"
        if not manifest.exists():
            return False
        try:
            meta = json.loads(manifest.read_text())
            return meta.get("complete", False)
        except (json.JSONDecodeError, KeyError):
            return False

    def cache_exists(self, config_hash: str) -> bool:
        """Return True only if the cache directory and a valid manifest exist."""
        return self._is_valid_cache(config_hash)

    def find_cache(self, config_hash: str, enforce_strict: bool = True) -> str | None:
        """Find a valid cache to load.

        If enforce_strict is True, only an exact match for config_hash is valid.
        If False, falls back to ANY available valid cache in the cache_root if
        the exact match fails.
        Returns the hash of the matched cache, or None.
        """
        # 1. Try exact match first
        if self._is_valid_cache(config_hash):
            return config_hash

        # 2. Try fallback if allowed
        if not enforce_strict:
            if self.cache_root.exists():
                for d in self.cache_root.iterdir():
                    if d.is_dir() and not d.name.endswith("_tmp"):
                        if self._is_valid_cache(d.name):
                            logger.warning(
                                f"Strict cache mismatch. Using fallback cache '{d.name}' "
                                f"instead of expected '{config_hash}' because "
                                "data.enforce_strict_cache is False."
                            )
                            return d.name
        return None

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        config_hash: str,
        trajectories: list[dict[str, Any]],
        norm_stats: dict[str, torch.Tensor],
        splits: dict[str, list[int]],
        task_index: dict[str, int] | None = None,
        provenance: dict[str, Any] | None = None,
        phase_thresholds: dict[str, Any] | None = None,
        dynamics_artifact: tuple[Any, ...] | None = None,
        topo_artifact: dict[str, Any] | None = None,
    ) -> None:
        """Atomically write all processed data to the cache.

        Writes to a tmp directory first, then renames to the final path
        to prevent partial cache corruption.

        Args:
            task_index: Optional ``{task_name: int_id}`` mapping to persist
                alongside the cache for auditability.
            provenance: Optional dict with the audit trail recorded in
                ``manifest.json``: dataset source/commit, per-file SHA-256,
                code git commit, state schema, normalization method,
                phase-labeler config and split task names (the latter are
                also written as human-readable ``train_tasks.txt`` /
                ``validation_tasks.txt``).
            phase_thresholds: Optional aggregated per-demonstration phase
                calibration artifact, persisted as ``phase_thresholds.json``
                (the rollout layer's source of truth for per-phase success
                tracking; see
                :func:`phaseforge.data.robomimic.phase_labeler.calibrate`).
            dynamics_artifact: Optional pending dynamic discovery artifact
                tuple ``(slds, report, task_name, train_labels, val_labels)``.
                It is written inside the temporary cache before the cache is
                marked complete, preserving atomicity for dynamics-enabled runs.
            topo_artifact: Optional pending topological discovery payload
                dict with ``save_kwargs`` for
                :func:`phaseforge.data.topo.artifacts.save_topo_artifact`
                (``output_dir`` is filled in here as ``topo_artifact/``).
                Written atomically alongside the dynamics bundle.
        """
        final_dir = self.cache_dir(config_hash)
        tmp_dir = self.cache_root / f"{config_hash}_tmp"

        # Clean up any existing tmp dir
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)

        # Trajectories
        traj_dir = tmp_dir / "trajectories"
        traj_dir.mkdir()
        for idx, traj in enumerate(trajectories):
            torch.save(traj, traj_dir / f"{idx:06d}.pt")

        # Norm stats
        torch.save(norm_stats, tmp_dir / "norm_stats.pt")

        # Splits (indices into the trajectories list)
        (tmp_dir / "splits.json").write_text(json.dumps(splits, indent=2))

        # Task index (deterministic name -> id; auditable)
        if task_index is not None:
            (tmp_dir / "task_index.json").write_text(json.dumps(task_index, indent=2))

        # Aggregated phase calibration artifacts (per-phase success tracking)
        if phase_thresholds is not None:
            (tmp_dir / "phase_thresholds.json").write_text(
                json.dumps(phase_thresholds, indent=2), encoding="utf-8"
            )

        if dynamics_artifact is not None:
            from phaseforge.data.dynamics.artifacts import save_discovery_artifact

            slds, report, task_name, train_labels, val_labels = dynamics_artifact
            save_discovery_artifact(
                output_dir=tmp_dir / "dynamics_artifact",
                slds=slds,
                report=report,
                task_name=task_name,
                data_config_hash=config_hash,
                train_labels=train_labels,
                val_labels=val_labels,
                extra_metadata={
                    "num_train_trajs": len(splits.get("train", [])),
                    "num_val_trajs": len(splits.get("val", [])),
                },
            )

        if topo_artifact is not None:
            from phaseforge.data.topo.artifacts import save_topo_artifact

            topo_kwargs = dict(topo_artifact.get("save_kwargs", {}))
            topo_kwargs["output_dir"] = tmp_dir / "topo_artifact"
            topo_kwargs.setdefault("data_config_hash", config_hash)
            save_topo_artifact(**topo_kwargs)

        # Human-readable split task names for auditability
        split_task_names = (provenance or {}).get("split_task_names") if provenance else None
        if split_task_names:
            for fname, key in (
                ("train_tasks.txt", "train"),
                ("validation_tasks.txt", "val"),
            ):
                names = sorted(split_task_names.get(key, []))
                (tmp_dir / fname).write_text(
                    "\n".join(names) + ("\n" if names else ""), encoding="utf-8"
                )

        # Manifest
        manifest = {
            "config_hash": config_hash,
            "num_trajectories": len(trajectories),
            "splits": {k: len(v) for k, v in splits.items()},
            "num_tasks": len(task_index) if task_index is not None else None,
            "created_at": time.time(),
            "complete": True,
        }
        if provenance:
            manifest["provenance"] = provenance
        (tmp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        # Atomic rename
        if final_dir.exists():
            shutil.rmtree(final_dir)
        shutil.move(str(tmp_dir), str(final_dir))

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(
        self,
        config_hash: str,
    ) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor], dict[str, list[int]], dict[str, int]]:
        """Load all data from the cache.

        Returns:
            ``(trajectories, norm_stats, splits, task_index)``.
            ``task_index`` is ``{}`` if the cache predates task-index
            persistence (loaded from an old cache that lacks task_index.json).
        """
        d = self.cache_dir(config_hash)
        traj_dir = d / "trajectories"

        traj_files = sorted(traj_dir.glob("*.pt"))
        trajectories = [torch.load(f, weights_only=False) for f in traj_files]
        norm_stats = torch.load(d / "norm_stats.pt", weights_only=False)
        splits = json.loads((d / "splits.json").read_text())

        task_index_path = d / "task_index.json"
        if task_index_path.exists():
            task_index = json.loads(task_index_path.read_text())
        else:
            task_index = {}

        return trajectories, norm_stats, splits, task_index


def load_phase_thresholds(cache_dir: str | Path) -> dict[str, Any] | None:
    """Load the aggregated phase-threshold calibration, or None if absent.

    A cache that predates phase-threshold persistence has no
    ``phase_thresholds.json`` (and its trajectories carry no per-demo
    artifacts); callers that need the calibration for per-phase tracking
    must fail closed rather than fabricate thresholds. The returned dict
    contains ``data_config_hash``, ``n_demos``, the aggregated median
    ``closed_level``/``open_level`` and ``mirror`` convention, and the
    full ``per_demo`` artifact list.
    """
    path = Path(cache_dir) / "phase_thresholds.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

"""Data ingestion state machine.

This is the single entry point for all data loading. The trainer calls
``DataPipelineStateMachine(cfg).run()`` and receives a dict of DataLoaders.

States
------
CHECK_PERSISTENT_CACHE → (hit) READY
                       → (miss) VALIDATE_SOURCE → INGEST_AND_STRIP
                                               → NORMALIZE_AND_SAVE → READY

When the raw source is missing and ``data.source.auto_download=true``,
VALIDATE_SOURCE routes to PROVISION_SOURCE, which downloads the configured
HuggingFace artifact (SHA-256-verified) before INGEST_AND_STRIP.

Design notes
------------
- The FSM consumes pre-downloaded data from the env-var-aware data root
  (paths.py). VALIDATE_SOURCE checks the configured ``data.source``
  directory before ingesting; downloading from the mirror happens only in
  the explicit PROVISION_SOURCE state when ``data.source.auto_download=true``
  (off by default, so training runs stay fail-closed).
- The processed cache lives under {data_root}/processed/cache, NOT under
  the per-run outputs/ directory, so the config-hash cache is reused across
  runs.
- Splits are done at trajectory or task level according to the protocol,
  never by timestep; the primary single-task config uses trajectory splits
  so validation is available without trajectory leakage.
- task_id is deterministic (sorted-name -> int); the ingester builds it.

Benchmark contract (docs/plan/design/final_evaluation_plan.md)
----------------------------------------------------------------
The dataset-specific conversion (robomimic HDF5 -> trajectory dicts) lives in
a pluggable ``data.ingester`` instantiated by :meth:`_ingest_source`. The
former benchmark-specific ingestion is archived under ``legacy/`` and must
not be resurrected here; the robomimic adapter defines its own observation
schema, action contract, splits and phase labels.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from phaseforge.data.common.collator import PhaseAwareCollator
from phaseforge.data.common.dataset import StateOnlyDataset
from phaseforge.data.common.normalizer import RunningStatNormalizer
from phaseforge.data.ingestion.cache_manager import (
    CacheManager,
    git_commit,
    sha256_file,
)
from phaseforge.data.ingestion.hf_downloader import download_hf_file
from phaseforge.data.ingestion.states import PipelineState
from phaseforge.data.paths import processed_cache_root

# Dynamics-aware discovery (optional, train-only).
StickySLDS: Any = None
SingleDynamicsModel: Any = None
evaluate_discovery_quality: Any = None
save_discovery_artifact: Any = None
load_discovery_artifact: Any = None
_DYNAMICS_IMPORT_ERROR: ImportError | None = None
try:
    from phaseforge.data.dynamics.artifacts import load_discovery_artifact, save_discovery_artifact
    from phaseforge.data.dynamics.diagnostics import evaluate_discovery_quality
    from phaseforge.data.dynamics.switching_linear import StickySLDS

    _DYNAMICS_AVAILABLE = True
except ImportError as exc:
    _DYNAMICS_AVAILABLE = False
    _DYNAMICS_IMPORT_ERROR = exc

# Topological discovery (optional, train-only). Same lazy pattern:
# the topo package needs only numpy/sklearn/scipy/torch, but a broken
# install must degrade to a loud opt-in error, never a silent skip.
audit_regimes: Any = None
cluster_segments: Any = None
concat_task_matrix: Any = None
extract_task_vars: Any = None
load_topo_artifact: Any = None
run_pelt: Any = None
save_topo_artifact: Any = None
segment_features: Any = None
_TOPO_IMPORT_ERROR: ImportError | None = None
try:
    from phaseforge.data.topo.artifacts import load_topo_artifact, save_topo_artifact
    from phaseforge.data.topo.cluster import cluster_segments, segment_features
    from phaseforge.data.topo.observability import audit_regimes
    from phaseforge.data.topo.pelt import run_pelt
    from phaseforge.data.topo.task_vars import concat_task_matrix, extract_task_vars

    _TOPO_AVAILABLE = True
except ImportError as exc:
    _TOPO_AVAILABLE = False
    _TOPO_IMPORT_ERROR = exc

logger = logging.getLogger(__name__)

DEFAULT_VAL_RATIO = 0.1
DEFAULT_SPLIT_SEED = 42


class PipelineError(RuntimeError):
    """Raised when the pipeline reaches the ERROR state."""


class DataPipelineStateMachine:
    """Autonomous data pipeline implemented as a finite state machine.

    Args:
        cfg: Root Hydra config. The pipeline uses ``cfg.data`` (including
             the generic ``data.source`` and ``data.ingester`` blocks).
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.data_cfg = cfg.data

        # num_phases is repeated across data + model configs with no
        # validation: a mismatch would silently break cross_entropy /
        # scatter_add / bincount. Guard here.
        self._check_num_phases_consistency()
        self._check_dynamics_consistency()
        self._check_topo_consistency()

        # Cache lives under the shared data root, not under outputs/.
        cache_root = processed_cache_root()
        self.cache_manager = CacheManager(cache_root)
        self.config_hash = CacheManager.compute_hash(self.data_cfg)

        self._state = PipelineState.CHECK_PERSISTENT_CACHE
        self._error_msg: str = ""

        # Resolved during processing
        self._task_index: dict[str, int] = {}
        self._trajectories: list[dict[str, Any]] = []
        self._norm_stats: dict[str, torch.Tensor] = {}
        self._splits: dict[str, list[int]] = {}
        self._raw_dir: Path | None = None
        self._pending_dynamics_artifact: tuple | None = None
        self._pending_topo_artifact: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Config consistency
    # ------------------------------------------------------------------

    def _check_num_phases_consistency(self) -> None:
        """Guard against latent num_phases mismatches across configs.

        The integer num_phases is repeated in:
        - data.phase_labeler.num_phases   (label generation, when present)
        - models.phase_head.num_phases    (classifier width)
        - models.router.num_experts       (router width)
        - models.num_phases               (oracle model)

        Nothing validated they match. If they diverge, the labeler produces
        values the classifier/scatter/bincount cannot index. This guard makes
        the inconsistency a loud, early failure instead of a silent corruption.

        Only checks model fields that exist in the current config (so a
        config that omits the oracle's top-level num_phases doesn't trip it),
        and is skipped entirely when the data config carries no phase-labeler
        block.

        The router width is exempted for PhaseBootstrappedMoE prototype /
        unsupervised / random / soft-mapping inits (``centroid``,
        ``phase_centroid``, ``spherical_centroid``, ``spherical_kmeans``,
        ``kmeans``, ``random``, ``soft_mapping``): the hierarchical
        prototype construction supports E != P (super-prototypes E<P,
        intra-phase scaling E>P), unsupervised clustering never indexes
        phase labels, and the explicit P x E soft mapping bridges the two
        spaces — so the Wave-2 expert-scaling sweep (K=3/12 vs P=6) is
        legal. The 1:1 phase->expert mapping stays mandatory for the
        phase-head copy init (E rows copied from P rows) and for models
        without a ``router_init`` block (teacher-forced hard dispatch,
        oracle).
        """
        labeler_cfg = self.data_cfg.get("phase_labeler")
        if labeler_cfg is None or labeler_cfg.get("num_phases") is None:
            return

        data_phases = int(labeler_cfg.num_phases)

        models_cfg = self.cfg.get("models")
        if models_cfg is None:
            return  # model config not part of this run (e.g. eval-only)

        router_init_type: str | None = None
        router_init_cfg = models_cfg.get("router_init")
        if router_init_cfg is not None and router_init_cfg.get("type") is not None:
            router_init_type = str(router_init_cfg.type).lower()

        candidates: list[tuple[str, int]] = []
        phase_head = models_cfg.get("phase_head")
        if phase_head is not None and phase_head.get("num_phases") is not None:
            candidates.append(("models.phase_head.num_phases", int(phase_head.num_phases)))

        router = models_cfg.get("router")
        # Routers whose initial expert layout is derived from the phase
        # distribution (centroid/kmeans/random) or from an explicit P x E
        # soft mapping (V2-B) legitimately decouple expert count from phase
        # count — the mapping bridges the two spaces, so the strict
        # num_experts == num_phases equality does not apply.
        e_ne_p_capable = router_init_type in (
            "centroid",
            "phase_centroid",
            "spherical_centroid",
            "spherical_kmeans",
            "kmeans",
            "random",
            "soft_mapping",
        )
        if router is not None and router.get("num_experts") is not None and not e_ne_p_capable:
            candidates.append(("models.router.num_experts", int(router.num_experts)))

        # Oracle model carries a top-level num_phases; other models may not.
        if models_cfg.get("num_phases") is not None:
            candidates.append(("models.num_phases", int(models_cfg.num_phases)))

        mismatches = [(name, val) for name, val in candidates if val != data_phases]
        if mismatches:
            details = ", ".join(f"{n}={v}" for n, v in mismatches)
            raise PipelineError(
                f"num_phases inconsistency: data.phase_labeler.num_phases="
                f"{data_phases} but {details}. All must match, otherwise the "
                "phase labels cannot be indexed by the classifier/router/scatter."
            )

    def _check_dynamics_consistency(self) -> None:
        """Validate the primary six-regime dynamics contract early."""
        dyn_cfg = self._get_dynamics_cfg()
        enabled = bool(dyn_cfg.get("enabled", False)) if hasattr(dyn_cfg, "get") else False
        if not enabled:
            return

        num_regimes = int(dyn_cfg.get("num_regimes", 6))
        if num_regimes != 6:
            raise PipelineError(
                "PhaseForge 2.0 currently requires data.dynamics.num_regimes=6; "
                f"received {num_regimes}. Run a declared regime-count ablation separately."
            )

        models_cfg = self.cfg.get("models")
        if models_cfg is None:
            return
        router = models_cfg.get("router")
        router_experts = router.get("num_experts") if router is not None else None
        phase_head = models_cfg.get("phase_head")
        phase_count = phase_head.get("num_phases") if phase_head is not None else None
        mismatches = []
        if router_experts is not None and int(router_experts) != num_regimes:
            mismatches.append(f"models.router.num_experts={router_experts}")
        if phase_count is not None and int(phase_count) != num_regimes:
            mismatches.append(f"models.phase_head.num_phases={phase_count}")
        if mismatches:
            raise PipelineError(
                "Dynamic regime/expert vocabulary mismatch: "
                + ", ".join(mismatches)
                + f"; expected six values matching num_regimes={num_regimes}."
            )

    def _model_uses_phase_labels(self) -> bool:
        """True when the selected model consumes phase labels in training.

        Models with a ``phase_head`` (phase cross-entropy supervision) or a
        top-level ``num_phases`` (privileged oracle routing / centroid
        bootstrap) need every phase to be populated; degenerate labels would
        silently corrupt them. MoE rows that declare only a
        ``router.num_experts`` (scratch/warm-start MoE) train without labels
        and must not block the BC pilot on degenerate labels.
        """
        models_cfg = self.cfg.get("models")
        if models_cfg is None:
            return False
        phase_head = models_cfg.get("phase_head")
        if phase_head is not None and phase_head.get("num_phases") is not None:
            return True
        return models_cfg.get("num_phases") is not None

    def _get_dynamics_cfg(self) -> dict[str, Any] | Any:
        """Return dynamics config dict, supporting both `cfg.dynamics` and `cfg.data.dynamics`.

        Configs with no dynamics block are treated as disabled. Dynamics
        are normally carried under `cfg.data.dynamics` (the explicit
        dynamics group packaged at the global root). A top-level
        `cfg.dynamics` wrapper may be left by explicit overrides; this
        helper normalizes both forms and returns an empty disabled dict
        when absent.
        """
        # Top-level `dynamics` (legacy explicit override)
        dyn = self.cfg.get("dynamics")
        # Canonical `data.dynamics` (explicit dynamics group)
        data_dyn = None
        try:
            data_dyn = self.data_cfg.get("dynamics")  # type: ignore[union-attr]
        except Exception:
            data_dyn = None
        # Prefer canonical `data.dynamics` when present, else top-level.
        if data_dyn is not None and isinstance(data_dyn, (dict, DictConfig)):
            return data_dyn
        if dyn is not None:
            # A `+dynamics=...` append creates a top-level wrapper containing
            # the packaged `dynamics` block. Normalize it so an explicit
            # override cannot silently disable discovery.
            if isinstance(dyn, (dict, DictConfig)) and isinstance(
                dyn.get("dynamics"), (dict, DictConfig)
            ):
                return dyn["dynamics"]
            return dyn
        return {}

    def _is_dynamics_enabled(self) -> bool:
        """True when dynamic discovery is enabled."""
        dyn = self._get_dynamics_cfg()
        if dyn is None:
            return False
        try:
            enabled = (
                bool(dyn.get("enabled", False))
                if hasattr(dyn, "get")
                else bool(getattr(dyn, "enabled", False))
            )
        except Exception:
            enabled = False
        if enabled and not _DYNAMICS_AVAILABLE:
            raise PipelineError(
                "data.dynamics.enabled=true but phaseforge.data.dynamics package is unavailable. "
                "Check phaseforge/data/dynamics/__init__.py and dependencies "
                f"(torch, numpy, sklearn): {_DYNAMICS_IMPORT_ERROR}"
            )
        return enabled

    def _check_topo_consistency(self) -> None:
        """Validate the topological discovery contract early."""
        topo_cfg = self._get_topo_cfg()
        enabled = False
        try:
            enabled = (
                bool(topo_cfg.get("enabled", False))
                if hasattr(topo_cfg, "get")
                else bool(getattr(topo_cfg, "enabled", False))
            )
        except Exception:
            enabled = False
        if not enabled:
            return
        num_regimes = topo_cfg.get("num_regimes", 6) if hasattr(topo_cfg, "get") else 6
        try:
            num_regimes_int = int(num_regimes)
        except (TypeError, ValueError) as exc:
            raise PipelineError(
                f"data.topo.num_regimes={num_regimes!r} is not an integer."
            ) from exc
        if not 2 <= num_regimes_int <= 12:
            raise PipelineError(
                f"data.topo.num_regimes={num_regimes_int} is outside [2, 12]. "
                "Run a declared regime-count ablation separately."
            )
        method = str(topo_cfg.get("method", "pelt")) if hasattr(topo_cfg, "get") else "pelt"
        if method.lower() != "pelt":
            raise PipelineError(
                f"Unknown data.topo.method={method!r}; expected 'pelt'."
            )
        cost = str(topo_cfg.get("cost", "l2")) if hasattr(topo_cfg, "get") else "l2"
        if cost.lower() != "l2":
            raise PipelineError(f"Unknown data.topo.cost={cost!r}; expected 'l2'.")
        if self._is_dynamics_enabled():
            raise PipelineError(
                "data.topo and data.dynamics are both enabled; enable at most one "
                "discovery source so the primary supervision target stays unambiguous."
            )

    def _get_topo_cfg(self) -> dict[str, Any] | Any:
        """Return the topo config dict (``cfg.data.topo``); disabled when absent.

        Configs with no topo block are treated as disabled — mirroring
        :meth:`_get_dynamics_cfg` without the top-level wrapper (topo was
        never a top-level override).
        """
        try:
            data_topo = self.data_cfg.get("topo")  # type: ignore[union-attr]
        except Exception:
            data_topo = None
        if data_topo is not None and isinstance(data_topo, (dict, DictConfig)):
            return data_topo
        return {}

    def _is_topo_enabled(self) -> bool:
        """True when topological discovery is enabled."""
        topo = self._get_topo_cfg()
        if topo is None:
            return False
        try:
            enabled = (
                bool(topo.get("enabled", False))
                if hasattr(topo, "get")
                else bool(getattr(topo, "enabled", False))
            )
        except Exception:
            enabled = False
        if enabled and not _TOPO_AVAILABLE:
            raise PipelineError(
                "data.topo.enabled=true but phaseforge.data.topo package is unavailable. "
                "Check phaseforge/data/topo/__init__.py and dependencies "
                f"(numpy, sklearn, scipy, torch): {_TOPO_IMPORT_ERROR}"
            )
        return enabled

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> dict[str, DataLoader | None]:
        """Execute the FSM to the READY terminal state.

        Returns:
            Dict of ``{"train": DataLoader, "val": DataLoader}``.
            Splits that have 0 samples return None in their slot.
        """
        logger.info(f"Pipeline starting. Config hash: {self.config_hash}")

        while self._state not in (PipelineState.READY, PipelineState.ERROR):
            self._step()

        if self._state == PipelineState.ERROR:
            raise PipelineError(self._error_msg)

        return self._build_dataloaders()

    def _dynamic_cache_is_valid(self, cache_dir: Path) -> bool:
        """Validate dynamic labels and their matching discovery artifact."""
        if load_discovery_artifact is None:
            return False
        try:
            _, artifact_labels, metadata = load_discovery_artifact(cache_dir / "dynamics_artifact")
            expected_k = int(self._get_dynamics_cfg().get("num_regimes", 6))
            if metadata.get("data_config_hash") != self.config_hash:
                logger.warning("Dynamic artifact hash does not match cache hash.")
                return False
            if int(metadata.get("num_regimes", -1)) != expected_k:
                logger.warning("Dynamic artifact regime count does not match the active config.")
                return False

            expected_split_lengths = {
                "train": len(self._splits.get("train", [])),
                "val": len(self._splits.get("val", [])),
            }
            for split_name, expected_length in expected_split_lengths.items():
                labels = artifact_labels.get(split_name)
                if labels is None or len(labels) != expected_length:
                    logger.warning(
                        "Dynamic artifact %s labels do not match the cache split.", split_name
                    )
                    return False

            for trajectory_index, trajectory in enumerate(self._trajectories):
                labels = trajectory.get("phase_dynamic")
                state = trajectory.get("state")
                if labels is None or state is None:
                    return False
                label_array = (
                    labels.detach().cpu().numpy()
                    if isinstance(labels, torch.Tensor)
                    else np.asarray(labels)
                )
                state_length = int(state.shape[0])
                if label_array.shape != (state_length,) or label_array.size == 0:
                    return False
                if np.any(label_array < 0) or np.any(label_array >= expected_k):
                    return False
                artifact_split_name: str | None = next(
                    (name for name, indices in self._splits.items() if trajectory_index in indices),
                    None,
                )
                if artifact_split_name in ("train", "val"):
                    split_position = self._splits[artifact_split_name].index(trajectory_index)
                    artifact_array = (
                        artifact_labels[artifact_split_name][split_position].cpu().numpy()
                    )
                    if not np.array_equal(label_array, artifact_array):
                        logger.warning(
                            "Dynamic artifact labels do not match trajectory %d.",
                            trajectory_index,
                        )
                        return False
            return True
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            logger.warning("Dynamic cache validation failed: %s", exc)
            return False

    def _topo_cache_is_valid(self, cache_dir: Path) -> bool:
        """Validate topo labels and their matching discovery artifact."""
        if load_topo_artifact is None:
            return False
        try:
            topo_cfg = self._get_topo_cfg()
            expected_k = int(topo_cfg.get("num_regimes", 6)) if hasattr(topo_cfg, "get") else 6
            _, _, metadata = load_topo_artifact(cache_dir / "topo_artifact")
            if metadata.get("data_config_hash") != self.config_hash:
                logger.warning("Topo artifact hash does not match cache hash.")
                return False
            if int(metadata.get("num_regimes", -1)) != expected_k:
                logger.warning("Topo artifact regime count does not match the active config.")
                return False
            for trajectory_index, trajectory in enumerate(self._trajectories):
                labels = trajectory.get("phase_topo")
                state = trajectory.get("state")
                if labels is None or state is None:
                    return False
                label_array = (
                    labels.detach().cpu().numpy()
                    if isinstance(labels, torch.Tensor)
                    else np.asarray(labels)
                )
                state_length = int(state.shape[0])
                if label_array.shape != (state_length,) or label_array.size == 0:
                    return False
                if np.any(label_array < 0) or np.any(label_array >= expected_k):
                    return False
            return True
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            logger.warning("Topo cache validation failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def _step(self) -> None:
        try:
            if self._state == PipelineState.CHECK_PERSISTENT_CACHE:
                self._check_cache()
            elif self._state == PipelineState.VALIDATE_SOURCE:
                self._validate_source()
            elif self._state == PipelineState.PROVISION_SOURCE:
                self._provision_source()
            elif self._state == PipelineState.INGEST_AND_STRIP:
                self._ingest_source()
            elif self._state == PipelineState.NORMALIZE_AND_SAVE:
                self._normalize_and_save()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Pipeline error")
            self._error_msg = str(exc)
            self._state = PipelineState.ERROR

    def _check_cache(self) -> None:
        logger.info("CHECK_PERSISTENT_CACHE: looking for cached data…")
        enforce_strict = self.data_cfg.get("enforce_strict_cache", True)
        found_hash = self.cache_manager.find_cache(self.config_hash, enforce_strict)

        if found_hash:
            logger.info(f"Cache hit (hash: {found_hash}). Loading from disk.")
            (
                self._trajectories,
                self._norm_stats,
                self._splits,
                self._task_index,
            ) = self.cache_manager.load(found_hash)
            self._backfill_phase_thresholds(found_hash)

            # Use the loaded hash so it doesn't mismatch later if we need it
            self.config_hash = found_hash
            # Require both labels and a matching artifact: old caches are
            # upgraded lazily, but a partially written artifact is never
            # accepted as a valid cache hit.
            if self._is_dynamics_enabled():
                cache_dir = self.cache_manager.cache_dir(self.config_hash)
                if not self._dynamic_cache_is_valid(cache_dir):
                    logger.info(
                        "Dynamics cache labels/artifact are missing or stale — running discovery."
                    )
                    self._run_dynamics_discovery(persist_to_cache=True)
                    if not self._dynamic_cache_is_valid(cache_dir):
                        raise PipelineError(
                            "Dynamics discovery completed without a valid cache artifact."
                        )
                else:
                    logger.info(
                        "Cache hit includes validated dynamic labels and discovery artifact."
                    )
            if self._is_topo_enabled():
                cache_dir = self.cache_manager.cache_dir(self.config_hash)
                if not self._topo_cache_is_valid(cache_dir):
                    logger.info("Topo cache is missing or stale; running discovery.")
                    self._run_topo_discovery(persist_to_cache=True)
                    if not self._topo_cache_is_valid(cache_dir):
                        raise PipelineError(
                            "Topo discovery completed without a valid cache artifact."
                        )
                else:
                    logger.info("Cache hit includes validated topo labels and discovery artifact.")
            self._state = PipelineState.READY
        else:
            logger.info("Cache miss. Proceeding to validate source.")
            self._state = PipelineState.VALIDATE_SOURCE

    def _resolve_raw_dir(self) -> Path:
        """Resolve the configured raw dataset directory (``data.source.dir``)."""
        source = self.data_cfg.get("source")
        if source is None or not source.get("dir"):
            raise PipelineError(
                "data.source.dir is not configured. The robomimic protocol "
                "(docs/plan/design/final_evaluation_plan.md, Gate 0) requires a "
                "source block with the raw dataset directory before the "
                "pipeline can ingest. Legacy benchmark-specific data configs "
                "are archived under legacy/ and are not part of the protocol."
            )
        return Path(str(source["dir"]))

    def _auto_download_enabled(self) -> bool:
        """True when the FSM may provision missing raw data from the mirror."""
        source = self.data_cfg.get("source")
        if source is None:
            return False
        return bool(source.get("auto_download", False))

    def _validate_source(self) -> None:
        """Consume pre-downloaded data and verify it exists."""
        logger.info("VALIDATE_SOURCE: checking pre-downloaded raw data…")
        raw_suite_dir = self._resolve_raw_dir()

        if not raw_suite_dir.exists():
            if self._auto_download_enabled():
                logger.info(
                    "  Raw source directory missing — auto-download enabled, "
                    "provisioning from the configured HuggingFace mirror."
                )
                self._raw_dir = raw_suite_dir
                self._state = PipelineState.PROVISION_SOURCE
                return
            raise PipelineError(
                f"Raw source directory not found: {raw_suite_dir}. "
                "Provision the dataset first, or set "
                "data.source.auto_download=true to download it from the "
                "configured HuggingFace mirror."
            )

        hdf5_files = sorted(raw_suite_dir.glob("*.hdf5"))
        if not hdf5_files:
            if self._auto_download_enabled():
                logger.info(
                    "  Raw source has no .hdf5 files — auto-download enabled, "
                    "provisioning from the configured HuggingFace mirror."
                )
                self._raw_dir = raw_suite_dir
                self._state = PipelineState.PROVISION_SOURCE
                return
            raise PipelineError(
                f"No .hdf5 files found in {raw_suite_dir}. "
                "Provision the dataset first, or set "
                "data.source.auto_download=true to download it from the "
                "configured HuggingFace mirror."
            )
        logger.info(
            "  OK: source '%s' has %d .hdf5 files",
            raw_suite_dir,
            len(hdf5_files),
        )

        self._raw_dir = raw_suite_dir
        self._state = PipelineState.INGEST_AND_STRIP

    def _provision_source(self) -> None:
        """PROVISION_SOURCE: download the raw artifact from the configured mirror.

        Reads ``data.source.huggingface`` (``repo_id``, ``path``, optional
        pinned ``sha256``) and downloads into the raw directory with SHA-256
        verification (pinned value, else the mirror's LFS metadata, else an
        HDF5 sanity check). Proceeds to ingestion only after a verified
        artifact is on disk; an already-verified file is a no-op.
        """
        logger.info("PROVISION_SOURCE: downloading raw dataset artifact…")
        source = self.data_cfg.get("source")
        if source is None:
            raise PipelineError("data.source is not configured.")
        hf_cfg = source.get("huggingface")
        if hf_cfg is None or not hf_cfg.get("repo_id") or not hf_cfg.get("path"):
            raise PipelineError(
                "data.source.auto_download=true requires data.source.huggingface "
                "with repo_id and path (the HuggingFace mirror artifact)."
            )

        dest_dir = self._raw_dir or self._resolve_raw_dir()
        download_hf_file(
            repo_id=str(hf_cfg["repo_id"]),
            path=str(hf_cfg["path"]),
            dest_dir=dest_dir,
            pinned_sha256=str(hf_cfg["sha256"]) if hf_cfg.get("sha256") else None,
        )
        self._state = PipelineState.INGEST_AND_STRIP

    def _ingest_source(self) -> None:
        """Delegate dataset conversion to the pluggable ``data.ingester``.

        The ingester is a Hydra target receiving ``raw_dir`` and returning
        ``(trajectories, task_index)`` where each trajectory carries
        ``state`` (T, S), ``action`` (T, A), ``phase`` (T,) and ``task_id``.
        """
        from hydra.utils import instantiate

        ingester_cfg = self.data_cfg.get("ingester")
        if ingester_cfg is None:
            raise PipelineError(
                "data.ingester is not configured — the source adapter for the "
                "robomimic/robosuite protocol (docs/plan/design/final_evaluation_plan.md, "
                "Gate 0/1) is not implemented yet. The former benchmark-specific "
                "ingestion stack was archived to legacy/ and must not be used."
            )

        logger.info("INGEST_AND_STRIP: converting raw HDF5 to trajectories…")
        ingester = instantiate(ingester_cfg, raw_dir=self._raw_dir)
        trajectories, task_index = ingester.ingest()

        if not trajectories:
            raise PipelineError("The ingester produced no trajectories.")

        num_phases = None
        labeler_cfg = self.data_cfg.get("phase_labeler")
        if labeler_cfg is not None and labeler_cfg.get("num_phases") is not None:
            num_phases = int(labeler_cfg.num_phases)
        phase_counts: np.ndarray | None = None
        if num_phases is not None:
            phase_counts = np.zeros(num_phases, dtype=np.int64)
        for traj in trajectories:
            if "phase" not in traj:
                raise PipelineError(
                    f"(demo {traj.get('demo_key', '?')}): the ingester did not "
                    "produce phase labels. PhaseForge training requires labels "
                    "generated from permitted low-dimensional observations."
                )
            if num_phases is not None:
                self._validate_phase_labels(traj, traj["phase"], num_phases)
                counts = np.bincount(
                    np.asarray(traj["phase"], dtype=np.int64),
                    minlength=num_phases,
                )
                assert phase_counts is not None
                phase_counts += counts

        if phase_counts is not None and np.any(phase_counts == 0):
            missing = np.flatnonzero(phase_counts == 0).tolist()
            if self._model_uses_phase_labels():
                raise PipelineError(
                    f"Phase labels contain no samples for phase(s) {missing} "
                    f"(per-phase sample counts across all trajectories: "
                    f"{phase_counts.tolist()}). "
                    "The selected model consumes phase labels (phase_head or "
                    "top-level num_phases); degenerate labels would silently "
                    "corrupt phase cross-entropy and router centroid "
                    "initialization. Revise the state-only label thresholds "
                    "or the demonstrations before training."
                )
            logger.warning(
                "Phase labels contain no samples for phase(s) %s. "
                "The selected model does not consume phase labels, so the "
                "BC pilot may continue; PhaseForge centroid initialization "
                "must not proceed until Gate 3 is passed.",
                missing,
            )

        self._check_state_dim_consistency(trajectories)
        self._trajectories = trajectories
        self._task_index = task_index or {}
        self._state = PipelineState.NORMALIZE_AND_SAVE

    def _validate_phase_labels(
        self, traj: dict[str, Any], phases: np.ndarray, num_phases: int
    ) -> None:
        """Guard: phase labels must align with the state length and range.

        A length mismatch or out-of-range phase would corrupt training
        (cross-entropy / scatter / bincount) silently. This makes it a loud
        per-trajectory failure instead.
        """
        T = int(traj["state"].shape[0])
        if phases.ndim != 1:
            raise PipelineError(
                f"(demo {traj.get('demo_key', '?')}): phase labels must have "
                f"shape (T,), got {phases.shape}."
            )
        if phases.shape[0] != T:
            raise PipelineError(
                f"(demo {traj.get('demo_key', '?')}): phase labels have length "
                f"{phases.shape[0]} but state has T={T}. Labeler and ingester "
                "disagree on the trajectory length."
            )
        if phases.size and (int(phases.min()) < 0 or int(phases.max()) >= num_phases):
            raise PipelineError(
                f"(demo {traj.get('demo_key', '?')}): phase labels out of range "
                f"[0, {num_phases}): observed [{phases.min()}, {phases.max()}]."
            )

    def _provenance(self, splits: dict[str, list[int]]) -> dict[str, Any]:
        """Build the manifest audit trail for the final cache.

        Includes full-content SHA-256 of every raw HDF5 file (streamed, so
        multi-GB files are fine — one-time cost per data version), the code
        git commit, the state schema, the normalization method, the
        phase-labeler config and the split task names. Enough to prove
        exactly which dataset produced a result.
        """
        split_task_names: dict[str, list[str]] = {}
        split_demo_keys: dict[str, list[str]] = {}
        id_to_name = {v: k for k, v in self._task_index.items()}
        for split_name, indices in splits.items():
            split_task_names[split_name] = sorted(
                {
                    str(
                        self._trajectories[i].get(
                            "task_name",
                            id_to_name.get(
                                int(self._trajectories[i]["task_id"]),
                                self._trajectories[i]["task_id"],
                            ),
                        )
                    )
                    for i in indices
                }
            )
            split_demo_keys[split_name] = sorted(
                str(self._trajectories[i].get("demo_key", f"trajectory_{i}")) for i in indices
            )

        raw_files: list[dict[str, Any]] = []
        if self._raw_dir is not None and self._raw_dir.exists():
            for path in sorted(self._raw_dir.glob("*.hdf5")):
                logger.info("  Provenance: SHA-256 of %s …", path.name)
                stat = path.stat()
                raw_files.append(
                    {
                        "name": path.name,
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                        "sha256": sha256_file(path),
                    }
                )

        # Dataset revision: record the source's MANIFEST.json (repo id +
        # pinned commit SHA) verbatim when present.
        dataset_manifest: dict[str, Any] | None = None
        source = self.data_cfg.get("source")
        if source is not None and source.get("dir"):
            manifest_path = Path(
                str(source.get("manifest_path") or Path(str(source["dir"])) / "MANIFEST.json")
            )
            try:
                if manifest_path.exists():
                    dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Could not read the source manifest for provenance.",
                    exc_info=True,
                )

        state_schema = {
            "schema_version": str(self.data_cfg.get("schema_version", "")),
            "keys": [
                {"key": str(e["key"]), "dim": int(e["dim"])} for e in self.data_cfg.state_keys
            ],
            "state_dim": int(self.data_cfg.get("state_dim", 0)),
            "action_dim": int(self.data_cfg.get("action_dim", 7)),
            "action_contract": OmegaConf.to_container(
                self.data_cfg.get("action_contract", {}), resolve=True
            ),
        }

        labeler_cfg = self.data_cfg.get("phase_labeler")
        provenance: dict[str, Any] = {
            "dataset_manifest": dataset_manifest,
            "code_git_commit": git_commit(),
            "raw_files": raw_files,
            "state_schema": state_schema,
            "normalization": {
                "strategy": str(self.data_cfg.get("normalization_strategy", "zscore")),
                "ddof": 1,
                "train_split_only": True,
            },
            "sampling": {
                "sequence_length": int(self.data_cfg.get("sequence_length", 1)),
                "stride": int(self.data_cfg.get("stride", 1)),
            },
            "split_task_names": split_task_names,
            "split_demo_keys": split_demo_keys,
            "configuration": OmegaConf.to_yaml(self.data_cfg, resolve=True),
        }
        if labeler_cfg is not None:
            provenance["phase_labeler"] = OmegaConf.to_container(labeler_cfg, resolve=True)
        environment_metadata: list[dict[str, Any]] = []
        seen_metadata: set[str] = set()
        for traj in self._trajectories:
            metadata = traj.get("env_metadata")
            if not isinstance(metadata, dict):
                continue
            serialized = json.dumps(metadata, sort_keys=True)
            if serialized not in seen_metadata:
                seen_metadata.add(serialized)
                environment_metadata.append(metadata)
        provenance["environment_metadata"] = environment_metadata
        return provenance

    def _normalize_and_save(self) -> None:
        logger.info("NORMALIZE_AND_SAVE: computing statistics and persisting cache…")

        # Split before normalization so validation never contributes to the
        # fitted statistics.
        splits = self._build_task_level_splits()

        # Compute normalization stats on TRAIN split only.
        normalizer = RunningStatNormalizer()
        for idx in splits["train"]:
            traj = self._trajectories[idx]
            state_np = traj["state"]  # (T, S) numpy array
            normalizer.update(state_np)

        frozen_norm = normalizer.finalize()

        # Normalize all splits and convert to tensors
        for traj in self._trajectories:
            state_t = torch.from_numpy(traj["state"]).float()
            action_t = torch.from_numpy(traj["action"]).float()
            phase_t = torch.from_numpy(traj["phase"]).long()
            traj["state"] = frozen_norm.normalize(state_t)
            traj["action"] = action_t
            traj["phase"] = phase_t
            # Preserve canonical rule alias for factorial router selection
            if "phase_rule" not in traj:
                traj["phase_rule"] = phase_t

        norm_stats = {"mean": frozen_norm.mean, "std": frozen_norm.std}
        # Materialize splits/norm before dynamics so _run_dynamics_discovery can
        # use them (train-only fit). The trajectories are already normalized.
        self._norm_stats = norm_stats
        self._splits = splits

        # Train-only dynamics discovery (opt-in). When enabled,
        # fit StickySLDS on the normalized train split, decode every trajectory,
        # attach `phase_dynamic`, and stage the artifact for persistence.
        if self._is_dynamics_enabled():
            try:
                self._run_dynamics_discovery(persist_to_cache=False)
            except Exception as exc:
                # Surface as PipelineError so the FSM lands in ERROR with a
                # diagnosable message rather than silently training on bad regimes.
                raise PipelineError(f"Dynamics discovery failed: {exc}") from exc

        # Train-only topological discovery (opt-in). Independent of
        # dynamics (the consistency guard forbids enabling both at once).
        if self._is_topo_enabled():
            try:
                self._run_topo_discovery(persist_to_cache=False)
            except Exception as exc:
                raise PipelineError(f"Topo discovery failed: {exc}") from exc

        pending_dynamics_artifact = self._pending_dynamics_artifact
        pending_topo_artifact = self._pending_topo_artifact
        self.cache_manager.save(
            config_hash=self.config_hash,
            trajectories=self._trajectories,
            norm_stats=norm_stats,
            splits=splits,
            task_index=self._task_index,
            provenance=self._provenance(splits),
            phase_thresholds=self._aggregate_phase_thresholds(),
            dynamics_artifact=pending_dynamics_artifact,
            topo_artifact=pending_topo_artifact,
        )
        self._pending_dynamics_artifact = None
        self._pending_topo_artifact = None
        self._state = PipelineState.READY

    # ------------------------------------------------------------------
    # Phase calibration persistence (per-phase success tracking)
    # ------------------------------------------------------------------

    def _aggregate_phase_thresholds(self) -> dict[str, Any] | None:
        """Aggregate per-demonstration phase calibrations into one artifact.

        The rollout layer must classify the policy's own states into phases
        without re-deriving thresholds from the policy's trajectories. The
        per-demo artifacts persisted by the ingester are aggregated here:
        median hysteresis levels, the majority mirror convention (with its
        median bounds), and the full per-demo list for audit. Returns None
        when no trajectory carries a calibration artifact (e.g. a custom
        ingester without adaptive calibration) — per-phase tracking then
        fails closed downstream.
        """
        artifacts = [
            traj["phase_thresholds"]
            for traj in self._trajectories
            if isinstance(traj.get("phase_thresholds"), dict)
        ]
        if not artifacts:
            return None
        closed_levels = [float(a["closed_level"]) for a in artifacts]
        open_levels = [float(a["open_level"]) for a in artifacts]
        mirrors = [bool(a["mirror"]) for a in artifacts]
        mirror = max(set(mirrors), key=mirrors.count) if mirrors else False
        aggregated: dict[str, Any] = {
            "data_config_hash": self.config_hash,
            "n_demos": len(artifacts),
            "closed_level": float(np.median(closed_levels)),
            "open_level": float(np.median(open_levels)),
            "mirror": mirror,
            "per_demo": artifacts,
        }
        if mirror:
            bounds = [
                (float(a["mirror_bounds"][0]), float(a["mirror_bounds"][1]))
                for a in artifacts
                if a.get("mirror") and isinstance(a.get("mirror_bounds"), (list, tuple))
            ]
            if bounds:
                lo = float(np.median([b[0] for b in bounds]))
                hi = float(np.median([b[1] for b in bounds]))
                aggregated["mirror_bounds"] = [lo, hi]
        return aggregated

    def _backfill_phase_thresholds(self, found_hash: str) -> None:
        """Persist the aggregated calibration on cache hits when missing.

        Caches produced by this code carry per-demo artifacts in their
        trajectories, so a cache hit can backfill ``phase_thresholds.json``
        without re-ingesting. Caches that predate the persistence carry no
        artifacts: warn loudly (the rollout layer fails closed on the
        missing file) so the operator knows a re-ingest is required before
        per-phase success tracking can run.
        """
        cache_dir = self.cache_manager.cache_dir(found_hash)
        if (cache_dir / "phase_thresholds.json").exists():
            return
        aggregated = self._aggregate_phase_thresholds()
        if aggregated is None:
            logger.warning(
                "Cache %s predates per-demonstration phase-threshold "
                "persistence (no calibration artifacts in its trajectories). "
                "Per-phase success tracking will fail closed until the cache "
                "is re-ingested (delete %s and re-run).",
                found_hash,
                cache_dir,
            )
            return
        self.config_hash = found_hash
        (cache_dir / "phase_thresholds.json").write_text(
            json.dumps(aggregated, indent=2), encoding="utf-8"
        )
        logger.info(
            "Backfilled phase_thresholds.json from per-demo artifacts for cache %s (%d demos).",
            found_hash,
            aggregated["n_demos"],
        )

    def _run_dynamics_discovery(self, persist_to_cache: bool = False) -> None:
        """Fit Sticky SLDS train-only and decode every trajectory.

        Attaches ``phase_dynamic`` (Tensor[T] long) to every trajectory dict and
        persists a versioned artifact ``dynamics_artifact/`` under the cache dir
        for audit and reuse. Train-only fitting, deterministic for a fixed
        (task, fingerprint, seed), and validation decoding via the frozen model.

        Args:
            persist_to_cache: When True (cache-hit backfill path), overwrite
                the trajectory ``*.pt`` files on disk and write the artifact
                to the existing cache directory. When False (fresh ingest
                path), the caller (``_normalize_and_save``) will persist the
                augmented trajectories via ``CacheManager.save`` — we only write
                the artifact to a staging area that ``_normalize_and_save`` will
                move into the final cache.

        Raises:
            PipelineError: If discovery quality gates fail or if
                required fields are missing.
        """
        assert _DYNAMICS_AVAILABLE and StickySLDS is not None
        dyn_cfg = self._get_dynamics_cfg()
        # Read hyper-parameters with safe defaults matching
        # phaseforge/config/dynamics/switching_linear_k6.yaml
        num_regimes = int(dyn_cfg.get("num_regimes", 6)) if hasattr(dyn_cfg, "get") else 6
        sticky_kappa = float(dyn_cfg.get("sticky_kappa", 50.0)) if hasattr(dyn_cfg, "get") else 50.0
        dirichlet_alpha = (
            float(dyn_cfg.get("dirichlet_alpha", 1.0)) if hasattr(dyn_cfg, "get") else 1.0
        )
        ridge_lambda = float(dyn_cfg.get("ridge_lambda", 1e-4)) if hasattr(dyn_cfg, "get") else 1e-4
        min_variance = float(dyn_cfg.get("min_variance", 1e-4)) if hasattr(dyn_cfg, "get") else 1e-4
        max_em_iter = int(dyn_cfg.get("max_em_iter", 40)) if hasattr(dyn_cfg, "get") else 40
        em_tol = float(dyn_cfg.get("em_tol", 1e-3)) if hasattr(dyn_cfg, "get") else 1e-3
        min_duration = int(dyn_cfg.get("min_duration", 3)) if hasattr(dyn_cfg, "get") else 3
        seed = int(dyn_cfg.get("seed", 42)) if hasattr(dyn_cfg, "get") else 42

        if not hasattr(self, "_splits") or not self._splits or not self._trajectories:
            raise PipelineError(
                "Dynamics discovery requires splits and trajectories to be materialized."
            )

        train_trajs = [self._trajectories[i] for i in self._splits.get("train", [])]
        val_trajs = [self._trajectories[i] for i in self._splits.get("val", [])]

        if not train_trajs:
            raise PipelineError("Dynamics discovery: train split is empty — cannot fit SLDS.")

        logger.info(
            "Dynamics discovery: fitting StickySLDS "
            "(K=%d, kappa=%.1f, seed=%d) on %d train trajectories …",
            num_regimes,
            sticky_kappa,
            seed,
            len(train_trajs),
        )

        slds = StickySLDS(
            num_regimes=num_regimes,
            sticky_kappa=sticky_kappa,
            dirichlet_alpha=dirichlet_alpha,
            ridge_lambda=ridge_lambda,
            min_variance=min_variance,
            max_em_iter=max_em_iter,
            em_tol=em_tol,
            min_duration=min_duration,
            seed=seed,
        )
        slds.fit(train_trajs)

        # Quality gates — must pass before any MoE training
        min_occ_thresh = (
            float(dyn_cfg.get("min_occupancy_threshold", 0.02)) if hasattr(dyn_cfg, "get") else 0.02
        )
        max_single_regime_fraction = (
            float(dyn_cfg.get("max_single_regime_fraction", 0.5))
            if hasattr(dyn_cfg, "get")
            else 0.5
        )
        max_switch_rate = (
            float(dyn_cfg.get("max_switch_rate", 0.6)) if hasattr(dyn_cfg, "get") else 0.6
        )
        min_nll_improvement = (
            float(dyn_cfg.get("min_nll_improvement", 0.0)) if hasattr(dyn_cfg, "get") else 0.0
        )
        residual_ratio = (
            dyn_cfg.get("max_within_rule_residual_ratio") if hasattr(dyn_cfg, "get") else None
        )
        max_within_rule_residual_ratio = (
            float(residual_ratio) if residual_ratio is not None else None
        )
        report = evaluate_discovery_quality(
            slds,
            train_trajectories=train_trajs,
            val_trajectories=val_trajs if val_trajs else train_trajs,
            min_occupancy_threshold=min_occ_thresh,
            max_single_regime_fraction=max_single_regime_fraction,
            max_switch_rate=max_switch_rate,
            min_nll_improvement=min_nll_improvement,
            max_within_rule_residual_ratio=max_within_rule_residual_ratio,
        )
        logger.info(
            "Dynamics quality: passed=%s min_occ=%.2f%% "
            "single_reg_frac=%.1f%% switch_rate=%.3f NLL_improv=%.2f",
            report.passed_all,
            report.min_occupancy * 100,
            report.single_regime_fraction * 100,
            report.mean_switch_rate,
            report.nll_improvement,
        )
        enforce_quality_gates = bool(
            dyn_cfg.get("enforce_quality_gates", True) if hasattr(dyn_cfg, "get") else True
        )
        if not report.passed_all and enforce_quality_gates:
            raise PipelineError(
                "Dynamics discovery failed quality gates (Section 4.2): "
                + "; ".join(report.failure_reasons)
                + f" [occupancy={report.occupancy} nll_improv={report.nll_improvement:.2f}]"
            )
        if not report.passed_all:
            logger.warning(
                "Dynamics discovery quality gates failed, but discovery is continuing "
                "because data.dynamics.enforce_quality_gates=false. The failed "
                "quality report will be persisted with the discovery artifact: %s",
                "; ".join(report.failure_reasons),
            )

        # Decode every trajectory (train + val) with frozen model
        for traj in self._trajectories:
            labels_np = slds.decode_trajectory(traj)  # (T,) int
            labels_t = torch.from_numpy(labels_np).long()
            # Attach alongside canonical rule labels — never overwrite traj["phase"]
            traj["phase_dynamic"] = labels_t
            # Also expose as generic alias for downstream consumers that read
            # phase_field dynamically (e.g. dataset with phase_field=phase_dynamic)
            # The canonical `phase` key remains the rule label for backward compatibility.
            # Keep a convenience duplicate `phase_rule` for explicit factorial code
            if "phase_rule" not in traj:
                # traj["phase"] at this point is already Tensor (normalized path)
                # or ndarray (hit path) — unify to Tensor for the alias
                rule_phase = traj.get("phase")
                if isinstance(rule_phase, np.ndarray):
                    rule_phase = torch.from_numpy(rule_phase).long()
                traj["phase_rule"] = rule_phase

        # Derive per-split decoded labels for artifact persistence
        train_labels = [
            self._trajectories[i]["phase_dynamic"].cpu().numpy()
            for i in self._splits.get("train", [])
        ]
        val_labels = [
            self._trajectories[i]["phase_dynamic"].cpu().numpy()
            for i in self._splits.get("val", [])
        ]

        # Determine task name for artifact provenance (single-task primary)
        task_name = "unknown"
        try:
            source = self.data_cfg.get("source")
            if source is not None and source.get("task_name"):
                task_name = str(source.get("task_name")).lower()
            elif self._task_index:
                # Fallback: most frequent task id's name
                task_name = sorted(self._task_index.keys())[0].lower()
        except Exception:
            pass

        data_config_hash = self.config_hash
        if persist_to_cache:
            cache_dir = self.cache_manager.cache_dir(self.config_hash)
            artifact_dir = cache_dir / "dynamics_artifact"
            save_discovery_artifact(
                output_dir=artifact_dir,
                slds=slds,
                report=report,
                task_name=task_name,
                data_config_hash=data_config_hash,
                train_labels=train_labels,
                val_labels=val_labels,
                extra_metadata={
                    "num_train_trajs": len(train_trajs),
                    "num_val_trajs": len(val_trajs),
                },
            )
            # Overwrite trajectory files on disk so future cache hits are warm
            traj_dir = cache_dir / "trajectories"
            for idx, traj in enumerate(self._trajectories):
                torch.save(traj, traj_dir / f"{idx:06d}.pt")
            logger.info(
                "Persisted phase_dynamic augmentation and artifact to existing cache %s", cache_dir
            )
        else:
            # Fresh ingest path: stage artifact alongside the tmp cache dir.
            # _normalize_and_save will move it into the final cache after
            # CacheManager.save atomic rename. Store in instance for later.
            self._pending_dynamics_artifact = (slds, report, task_name, train_labels, val_labels)

    def _run_topo_discovery(self, persist_to_cache: bool = False) -> None:
        """Fit topological regimes train-only and label every trajectory.

        For each trajectory: extract task-space variables
        (:mod:`phaseforge.data.topo.task_vars`), segment with PELT
        (:mod:`phaseforge.data.topo.pelt`), featurize segments and cluster
        them across the training split
        (:mod:`phaseforge.data.topo.cluster`), then audit the resulting
        labels from ``x_t`` alone
        (:mod:`phaseforge.data.topo.observability`). Attaches
        ``phase_topo`` (Tensor[T] long) to every trajectory dict and stages
        a versioned ``topo_artifact/`` for persistence.

        Args:
            persist_to_cache: When True (cache-hit backfill path), write the
                artifact to the existing cache directory and overwrite the
                trajectory ``*.pt`` files on disk. When False (fresh ingest
                path), stage the artifact payload in
                ``self._pending_topo_artifact`` for ``CacheManager.save``.

        Raises:
            PipelineError: If the observability audit fails while
                ``data.topo.enforce_observability`` is true, or if required
                fields are missing.
        """
        assert _TOPO_AVAILABLE and run_pelt is not None
        topo_cfg = self._get_topo_cfg()
        get = topo_cfg.get if hasattr(topo_cfg, "get") else (lambda _k, _d=None: _d)
        num_regimes = int(get("num_regimes", 6))
        penalty_beta = float(get("penalty_beta", 10.0))
        min_segment_len = int(get("min_segment_len", 5))
        cost = str(get("cost", "l2"))
        method = str(get("clustering", "kmeans"))
        seed = int(get("seed", 42))
        enforce = bool(get("enforce_observability", True))
        audit_f1 = float(get("audit_min_macro_f1", 0.6))
        audit_occ = float(get("audit_min_occupancy", 0.01))

        if not hasattr(self, "_splits") or not self._splits or not self._trajectories:
            raise PipelineError(
                "Topo discovery requires splits and trajectories to be materialized."
            )
        try:
            state_keys = [str(e["key"]) for e in self.data_cfg.state_keys]
            state_dims = [int(e["dim"]) for e in self.data_cfg.state_keys]
        except Exception as exc:
            raise PipelineError(
                "Topo discovery needs data.state_keys as [{key, dim}] entries."
            ) from exc

        def _to_numpy(traj: dict[str, Any], key: str) -> np.ndarray:
            value = traj[key]
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().numpy()
            return np.asarray(value)

        norm_mean = None
        norm_std = None
        if hasattr(self, "_norm_stats") and self._norm_stats:
            m = self._norm_stats.get("mean")
            s = self._norm_stats.get("std")
            if m is not None:
                norm_mean = (
                    m.detach().cpu().numpy() if isinstance(m, torch.Tensor) else np.asarray(m)
                )
            if s is not None:
                norm_std = (
                    s.detach().cpu().numpy() if isinstance(s, torch.Tensor) else np.asarray(s)
                )

        # Per-trajectory PELT segmentation on task-space variables.
        boundaries_per_traj: list[np.ndarray] = []
        segments: list[np.ndarray] = []
        segment_traj: list[int] = []
        for traj_idx, traj in enumerate(self._trajectories):
            state_np = _to_numpy(traj, "state")
            if state_np.ndim != 2:
                raise PipelineError(f"Trajectory {traj_idx} state must be (T, S).")
            task_matrix = concat_task_matrix(
                extract_task_vars(
                    state_np, state_keys, state_dims, mean=norm_mean, std=norm_std
                )
            )
            bounds = run_pelt(
                task_matrix,
                penalty_beta=penalty_beta,
                min_segment_len=min_segment_len,
                cost=cost,
            )
            boundaries_per_traj.append(np.asarray(bounds, dtype=np.int64))
            for seg_idx in range(len(bounds) - 1):
                segments.append(task_matrix[bounds[seg_idx] : bounds[seg_idx + 1]])
                segment_traj.append(traj_idx)

        # Cluster segment prototypes across the training split only.
        train_idx_set = set(self._splits.get("train", []))
        train_positions = [i for i, t in enumerate(segment_traj) if t in train_idx_set]
        if not train_positions:
            raise PipelineError("Topo discovery: no training segments — cannot cluster.")
        train_feats = segment_features([segments[i] for i in train_positions])
        seg_labels_train = cluster_segments(
            train_feats, num_clusters=num_regimes, method=method, seed=seed
        )
        # Label every segment (train + val) by nearest training centroid.
        centroids = np.stack(
            [
                train_feats[seg_labels_train == k].mean(axis=0)
                if np.any(seg_labels_train == k)
                else train_feats.mean(axis=0)
                for k in range(num_regimes)
            ]
        )
        all_feats = segment_features(segments)
        dists = np.linalg.norm(all_feats[:, None, :] - centroids[None, :, :], axis=-1)
        seg_labels_all = np.argmin(dists, axis=1).astype(np.int64)

        # Scatter segment labels back to per-step trajectory labels.
        per_traj_labels: list[np.ndarray] = [
            np.zeros(_to_numpy(traj, "state").shape[0], dtype=np.int64)
            for traj in self._trajectories
        ]
        for pos, traj_idx in enumerate(segment_traj):
            bounds = boundaries_per_traj[traj_idx]
            # Position of this segment within its trajectory's boundary list.
            seg_idx_in_traj = sum(1 for t in segment_traj[:pos] if t == traj_idx)
            start, stop = int(bounds[seg_idx_in_traj]), int(bounds[seg_idx_in_traj + 1])
            per_traj_labels[traj_idx][start:stop] = int(seg_labels_all[pos])
        for traj, labels_np in zip(self._trajectories, per_traj_labels):
            traj["phase_topo"] = torch.from_numpy(labels_np).long()

        # Mandatory observability audit from x_t alone.
        flat_states = np.concatenate(
            [_to_numpy(traj, "state") for traj in self._trajectories], axis=0
        )
        flat_labels = np.concatenate(per_traj_labels, axis=0)
        flat_traj_ids = np.concatenate(
            [
                np.full(_to_numpy(traj, "state").shape[0], idx, dtype=np.int64)
                for idx, traj in enumerate(self._trajectories)
            ]
        )
        report = audit_regimes(
            flat_states,
            flat_labels,
            flat_traj_ids,
            num_regimes,
            min_macro_f1=audit_f1,
            min_occupancy=audit_occ,
        )
        logger.info(
            "Topo audit: passed=%s macro_f1=%.4f min_occ=%.4f mean_dur=%.1f",
            report.passed,
            report.macro_f1,
            report.min_occupancy,
            report.mean_duration,
        )
        if not report.passed and enforce:
            raise PipelineError(
                "Topo discovery failed the observability audit (Professor §4.4): "
                + "; ".join(report.failure_reasons)
                + ". Merge confused regimes or set "
                "data.topo.enforce_observability=false to persist labels anyway."
            )
        if not report.passed:
            logger.warning(
                "Topo audit failed but continuing because "
                "data.topo.enforce_observability=false: %s",
                "; ".join(report.failure_reasons),
            )

        try:
            source = self.data_cfg.get("source")
            task_name = str(source.get("task_name")).lower() if source is not None else "unknown"
        except Exception:
            task_name = "unknown"
        train_labels = [
            per_traj_labels[i] for i in self._splits.get("train", []) if i < len(per_traj_labels)
        ]
        val_labels = [
            per_traj_labels[i] for i in self._splits.get("val", []) if i < len(per_traj_labels)
        ]
        save_kwargs = {
            "method": "pelt",
            "task_name": task_name,
            "num_regimes": num_regimes,
            "hyper_params": {
                "penalty_beta": penalty_beta,
                "min_segment_len": min_segment_len,
                "cost": cost,
                "clustering": method,
                "seed": seed,
            },
            "train_labels": train_labels,
            "val_labels": val_labels,
            "train_boundaries": [
                boundaries_per_traj[i]
                for i in self._splits.get("train", [])
                if i < len(boundaries_per_traj)
            ],
            "val_boundaries": [
                boundaries_per_traj[i]
                for i in self._splits.get("val", [])
                if i < len(boundaries_per_traj)
            ],
            "report": report.to_dict(),
        }
        if persist_to_cache:
            cache_dir = self.cache_manager.cache_dir(self.config_hash)
            save_topo_artifact(output_dir=cache_dir / "topo_artifact", **save_kwargs)
            traj_dir = cache_dir / "trajectories"
            for idx, traj in enumerate(self._trajectories):
                torch.save(traj, traj_dir / f"{idx:06d}.pt")
            logger.info("Persisted phase_topo augmentation and artifact to cache %s", cache_dir)
        else:
            self._pending_topo_artifact = {"save_kwargs": save_kwargs}

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def _build_task_level_splits(self) -> dict[str, list[int]]:
        """Build the configured task-level or within-task trajectory split.

        ``strategy=task`` is appropriate for a multitask generalization test:
        every task is entirely in train or validation. The primary protocol is
        single-task, so ``strategy=trajectory`` holds out demonstrations from
        each task and prevents the validation loader from silently becoming
        empty.

        Reads ``data.split`` ({strategy, val_ratio, seed}) with safe defaults.

        Seed policy (intentional, documented for the paper): the split RNG
        is seeded from ``data.split.seed`` and **not** from
        ``cfg.project.seed``. A model-seed change must not re-shuffle the
        train/val boundary — otherwise every cell of an ablation seed sweep
        would see a different validation set, breaking direct comparison
        of val curves across cells and introducing a hidden coupling
        between the seed we report and the data we hold out. Across the
        protocol seeds [42, 43, 44] every cell therefore sees the *same*
        train/val trajectories; only the model initialisation and the
        per-epoch DataLoader shuffle (driven by
        :meth:`_train_sampler_generator`) differ.
        """
        split_cfg = self.data_cfg.get("split", {})
        strategy = str(split_cfg.get("strategy", "task"))
        val_ratio = float(split_cfg.get("val_ratio", DEFAULT_VAL_RATIO))
        split_seed = int(split_cfg.get("seed", DEFAULT_SPLIT_SEED))
        if strategy not in {"task", "trajectory"}:
            raise PipelineError(
                f"Unsupported data.split.strategy={strategy!r}; expected 'task' or 'trajectory'."
            )
        if not 0.0 <= val_ratio < 1.0:
            raise PipelineError("data.split.val_ratio must be in [0, 1).")

        # Group trajectory indices by task_id
        by_task: dict[int, list[int]] = defaultdict(list)
        for i, traj in enumerate(self._trajectories):
            by_task[int(traj["task_id"])].append(i)

        task_ids = sorted(by_task.keys())
        rng = np.random.default_rng(split_seed)

        if strategy == "trajectory":
            if bool(split_cfg.get("use_dataset_filters", True)):
                filtered = [
                    (i, traj.get("dataset_split")) for i, traj in enumerate(self._trajectories)
                ]
                if any(split is not None for _, split in filtered):
                    missing = [i for i, split in filtered if split not in {"train", "val"}]
                    if missing:
                        raise PipelineError(
                            "The HDF5 dataset contains an incomplete train/valid "
                            f"filter assignment; missing trajectories: {missing[:8]}"
                        )
                    filtered_train_idx = [i for i, split in filtered if split == "train"]
                    filtered_val_idx = [i for i, split in filtered if split == "val"]
                    if not filtered_train_idx or not filtered_val_idx:
                        raise PipelineError("The HDF5 train/valid filters must both be non-empty.")
                    logger.info(
                        "  Dataset-filter split: %d train trajectories, %d val "
                        "trajectories across %d tasks",
                        len(filtered_train_idx),
                        len(filtered_val_idx),
                        len(task_ids),
                    )
                    return {"train": filtered_train_idx, "val": filtered_val_idx}

            train_idx: list[int] = []
            val_idx: list[int] = []
            for task_id in task_ids:
                indices = np.asarray(by_task[task_id], dtype=np.int64)
                rng.shuffle(indices)
                if len(indices) < 2:
                    n_val = 0
                else:
                    n_val = (
                        min(
                            len(indices) - 1,
                            max(1, int(round(len(indices) * val_ratio))),
                        )
                        if val_ratio > 0
                        else 0
                    )
                val_idx.extend(int(i) for i in indices[:n_val])
                train_idx.extend(int(i) for i in indices[n_val:])
            logger.info(
                "  Trajectory-level split: %d train trajectories, %d val "
                "trajectories across %d tasks",
                len(train_idx),
                len(val_idx),
                len(task_ids),
            )
            if val_ratio > 0 and task_ids and not val_idx:
                raise PipelineError(
                    "The trajectory split produced no validation trajectories. "
                    "Provide at least two demonstrations for the single-task "
                    "protocol or set val_ratio=0 only for an explicitly "
                    "training-only pilot."
                )
            return {"train": train_idx, "val": val_idx}

        rng.shuffle(task_ids)

        # Hard floor so val never collapses to zero tasks when more than one
        # task exists (the single-task case below intentionally yields no
        # validation task).
        n_val_tasks = max(1, round(len(task_ids) * val_ratio)) if task_ids else 0
        # A task-level single-task split has no validation task. Callers that
        # need validation for a single task must select strategy=trajectory.
        if len(task_ids) <= 1:
            n_val_tasks = 0

        val_tasks = set(task_ids[:n_val_tasks])
        train_tasks = set(task_ids[n_val_tasks:])

        task_train_idx: list[int] = []
        task_val_idx: list[int] = []
        for tid in task_ids:
            if tid in val_tasks:
                task_val_idx.extend(by_task[tid])
            else:
                task_train_idx.extend(by_task[tid])

        logger.info(
            "  Task-level split: %d train tasks (%d trajs), %d val tasks (%d trajs)",
            len(train_tasks),
            len(task_train_idx),
            len(val_tasks),
            len(task_val_idx),
        )
        return {"train": task_train_idx, "val": task_val_idx}

    def phase_counts(self, split: str = "train") -> dict[int, int]:
        """Per-class phase step counts over one split (for class weighting).

        Counts each labelled step (trajectory timestep) in the requested
        split, keyed by phase index. Only meaningful after ``run()`` has
        reached READY (``self._trajectories``/``self._splits`` populated).
        """
        if split not in self._splits:
            raise PipelineError(f"Unknown split {split!r}; expected one of {sorted(self._splits)}")
        counts: dict[int, int] = defaultdict(int)
        for idx in self._splits[split]:
            phase = np.asarray(self._trajectories[idx]["phase"])
            for p, n in zip(*np.unique(phase, return_counts=True)):
                counts[int(p)] += int(n)
        return counts

    # ------------------------------------------------------------------
    # DataLoader construction
    # ------------------------------------------------------------------

    def _train_sampler_generator(self) -> torch.Generator:
        """Return a CPU ``torch.Generator`` for the train DataLoader sampler.

        The train DataLoader's shuffle uses an explicit generator seeded
        from ``cfg.project.seed`` so the per-epoch sample order is fully
        reproducible from the project seed alone. ``torch.Generator`` is
        placed on CPU because DataLoader worker processes cannot share a
        CUDA generator across the multiprocessing boundary.

        The generator is cached on the instance so a second call (e.g.
        after a cache miss + rerun) returns the same object and consumes
        no additional entropy — important for resume correctness.

        Returns:
            A ``torch.Generator`` on ``cpu`` seeded with the project seed.
        """
        cached = getattr(self, "_train_generator", None)
        if cached is not None:
            return cached
        project_seed = self.cfg.get("project", {}).get("seed", 0)
        try:
            seed_value = int(project_seed)
        except (TypeError, ValueError) as exc:
            raise PipelineError(
                f"cfg.project.seed must be an integer for DataLoader reproducibility, "
                f"got {project_seed!r}."
            ) from exc
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed_value)
        self._train_generator = generator
        return generator

    def _build_dataloaders(self) -> dict[str, DataLoader | None]:
        data_cfg = self.data_cfg
        result: dict[str, DataLoader | None] = {}

        for split_name, indices in self._splits.items():
            if not indices:
                result[split_name] = None
                continue

            split_trajs = [self._trajectories[i] for i in indices]
            is_train = split_name == "train"
            corruption_rate = float(data_cfg.get("phase_corruption_rate", 0.0)) if is_train else 0.0
            default_seed = int(self.cfg.get("project", {}).get("seed", 42))
            corruption_seed = (
                int(data_cfg.get("phase_corruption_seed", default_seed)) if is_train else 42
            )
            labeler_cfg = data_cfg.get("phase_labeler")
            # Dynamics: num_phases remains 6 for both vocabularies
            # (regime count == expert count in primary experiment). Dynamics may
            # carry its own num_regimes but the classifier head stays at 6.
            num_phases_val = int(labeler_cfg.get("num_phases", 6)) if labeler_cfg else 6
            phase_shuffle = (
                bool(data_cfg.get("phase_shuffle_control", False)) if is_train else False
            )

            # Select which label field Stage 1 trains on.
            # Configs without `data.dynamics` collapse to "phase".
            # When dynamics is enabled, `train_label_field` chooses the primary
            # supervision target ("phase"=rule, "phase_dynamic"=dynamic). The
            # alternate field is retained in the trajectory for router bootstrap.
            # Topo adds its own supervision source ("phase_topo"); dynamics
            # and topo are mutually exclusive (see _check_topo_consistency).
            phase_field = "phase"
            if self._is_dynamics_enabled():
                dyn_cfg = self._get_dynamics_cfg()
                try:
                    requested = (
                        str(dyn_cfg.get("train_label_field", "phase"))
                        if hasattr(dyn_cfg, "get")
                        else "phase"
                    )
                except Exception:
                    requested = "phase"
                if requested not in ("phase", "phase_dynamic", "phase_rule"):
                    raise PipelineError(
                        "Unknown data.dynamics.train_label_field="
                        f"{requested!r}; expected 'phase', 'phase_rule', or 'phase_dynamic'."
                    )
                if not all(requested in traj for traj in split_trajs):
                    raise PipelineError(
                        "Requested data.dynamics.train_label_field="
                        f"{requested!r} is missing from one or more trajectories; "
                        "refuse to fall back to rule labels. Re-ingest the cache."
                    )
                phase_field = requested
            if self._is_topo_enabled():
                topo_cfg = self._get_topo_cfg()
                try:
                    topo_requested = (
                        str(topo_cfg.get("train_label_field", "phase"))
                        if hasattr(topo_cfg, "get")
                        else "phase"
                    )
                except Exception:
                    topo_requested = "phase"
                if topo_requested not in ("phase", "phase_rule", "phase_topo"):
                    raise PipelineError(
                        "Unknown data.topo.train_label_field="
                        f"{topo_requested!r}; expected 'phase', 'phase_rule', or 'phase_topo'."
                    )
                if not all(topo_requested in traj for traj in split_trajs):
                    raise PipelineError(
                        "Requested data.topo.train_label_field="
                        f"{topo_requested!r} is missing from one or more trajectories; "
                        "refuse to fall back to rule labels. Re-ingest the cache."
                    )
                phase_field = topo_requested

            dataset = StateOnlyDataset(
                trajectories=split_trajs,
                sequence_length=int(data_cfg.sequence_length),
                stride=int(data_cfg.stride),
                phase_corruption_rate=corruption_rate,
                phase_corruption_seed=corruption_seed,
                num_phases=num_phases_val,
                phase_shuffle_control=phase_shuffle,
                phase_field=phase_field,
            )

            # Cap num_workers to os.cpu_count() to prevent warnings and slowdowns.
            # The dataset is fully materialized in memory, so zero remains a
            # valid choice for small CPU-only pilots where worker IPC costs
            # more than it saves.
            num_workers = max(0, int(data_cfg.num_workers))
            cpu_count = os.cpu_count()
            if cpu_count is not None:
                num_workers = min(num_workers, cpu_count)

            # These options are intentionally derived from the effective
            # runtime. Pinned memory is useful for CUDA host-to-device copies,
            # but adds overhead on CPU-only runs. prefetch_factor and
            # persistent_workers are only legal when workers are enabled.
            project_cfg = self.cfg.get("project")
            requested_device = str(
                project_cfg.get("device", "cuda") if project_cfg is not None else "cuda"
            )
            pin_memory = (
                bool(data_cfg.get("pin_memory", False))
                and requested_device.startswith("cuda")
                and torch.cuda.is_available()
            )
            loader_options: dict[str, Any] = {
                "pin_memory": pin_memory,
            }
            if num_workers > 0:
                loader_options["prefetch_factor"] = max(1, int(data_cfg.get("prefetch_factor", 2)))
                loader_options["persistent_workers"] = bool(
                    data_cfg.get("persistent_workers", False)
                )

            loader = DataLoader(
                dataset,
                batch_size=int(data_cfg.batch_size),
                shuffle=is_train,
                num_workers=num_workers,
                collate_fn=PhaseAwareCollator(),
                drop_last=is_train,
                generator=self._train_sampler_generator() if is_train else None,
                **loader_options,
            )
            result[split_name] = loader
            logger.info(f"  {split_name}: {len(dataset)} samples, {len(loader)} batches")

        return result

    def _check_state_dim_consistency(self, trajs: list[dict[str, Any]]) -> None:
        """Guard: every decoded state vector must match ``data.state_dim``.

        ``data.state_dim`` sizes the model encoder's ``input_dim``. A
        mismatch — e.g. the ingester producing a different width than the
        config — would otherwise surface only as a shape error at the first
        model forward pass, after a useless re-ingest. Failing loud here
        catches it at the source.
        """
        expected = int(self.data_cfg.get("state_dim", 0))
        if expected <= 0:
            return
        bad = sorted(
            {int(traj["state"].shape[-1]) for traj in trajs if traj["state"].shape[-1] != expected}
        )
        if bad:
            raise PipelineError(
                f"Decoded state dims {bad} do not match data.state_dim="
                f"{expected}. Fix data.source / data.ingester / data.state_dim, "
                "then re-ingest — the model encoder input_dim would disagree "
                "with the decoded states."
            )

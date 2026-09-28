"""Versioned persistence for topological discovery artifacts.

Mirrors :mod:`phaseforge.data.dynamics.artifacts`: atomic staging writes,
SHA-256 file checksums recorded in the manifest, and fail-closed loading.
The topo bundle is smaller (no dynamics matrices)::

    topo_artifact/
    ├── topo_labels.pt      # {"train": [Tensor[T] long], "val": [...]}
    ├── topo_segments.pt    # {"train": [int boundaries], "val": [...]}
    └── topo_manifest.json  # version, hyper-parameters, quality summary
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

TOPO_ARTIFACT_VERSION = "1.0.0"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_topo_artifact(
    output_dir: Path | str,
    *,
    method: str,
    task_name: str,
    data_config_hash: str,
    num_regimes: int,
    hyper_params: dict[str, Any],
    train_labels: list[np.ndarray],
    val_labels: list[np.ndarray],
    train_boundaries: list[np.ndarray],
    val_boundaries: list[np.ndarray],
    report: dict[str, Any],
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Save a versioned topo discovery artifact bundle.

    Args:
        output_dir: Directory receiving ``topo_artifact/`` contents.
        method: Discovery method id (``"pelt"``).
        task_name: Lowercase task name (e.g. ``"can"``).
        data_config_hash: Fingerprint of the data config that produced it.
        num_regimes: Regime count K (labels must lie in ``[0, K)``).
        hyper_params: JSON-serializable hyper-parameters (beta, min_len,
            cost, clustering method/seed, K).
        train_labels: Per-training-trajectory regime labels.
        val_labels: Per-validation-trajectory regime labels.
        train_boundaries: Per-training-trajectory PELT boundaries.
        val_boundaries: Per-validation-trajectory PELT boundaries.
        report: JSON-serializable quality/audit summary.
        extra_metadata: Optional additional metadata.

    Returns:
        Path to the saved artifact directory.
    """
    out_path = Path(output_dir)
    staging_path = out_path.with_name(f".{out_path.name}.tmp")
    if staging_path.exists():
        shutil.rmtree(staging_path)
    staging_path.mkdir(parents=True, exist_ok=True)
    try:
        for name, label_lists in (("train", train_labels), ("val", val_labels)):
            if not all(isinstance(lbl, np.ndarray) and lbl.ndim == 1 for lbl in label_lists):
                raise ValueError(f"{name} labels must be a list of 1-D arrays.")
        labels_path = staging_path / "topo_labels.pt"
        segments_path = staging_path / "topo_segments.pt"
        torch.save(
            {
                "train": [torch.from_numpy(np.asarray(lbl)).long() for lbl in train_labels],
                "val": [torch.from_numpy(np.asarray(lbl)).long() for lbl in val_labels],
            },
            labels_path,
        )
        torch.save(
            {
                "train": [torch.from_numpy(np.asarray(b)).long() for b in train_boundaries],
                "val": [torch.from_numpy(np.asarray(b)).long() for b in val_boundaries],
            },
            segments_path,
        )
        metadata = {
            "version": TOPO_ARTIFACT_VERSION,
            "method": str(method),
            "task_name": str(task_name),
            "data_config_hash": str(data_config_hash),
            "num_regimes": int(num_regimes),
            "hyper_params": dict(hyper_params),
            "quality_report": dict(report),
            "extra": dict(extra_metadata or {}),
            "checksums": {
                "topo_labels.pt": _sha256_file(labels_path),
                "topo_segments.pt": _sha256_file(segments_path),
            },
        }
        manifest_path = staging_path / "topo_manifest.json"
        manifest_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        if out_path.exists():
            shutil.rmtree(out_path)
        staging_path.replace(out_path)
    except Exception:
        if staging_path.exists():
            shutil.rmtree(staging_path)
        raise
    logger.info("Saved topo discovery artifact (v%s) to %s", TOPO_ARTIFACT_VERSION, out_path)
    return out_path


def load_topo_artifact(
    artifact_dir: Path | str,
) -> tuple[dict[str, list[torch.Tensor]], dict[str, list[torch.Tensor]], dict[str, Any]]:
    """Load a topo artifact bundle (fail-closed on version/checksum/shape).

    Returns:
        ``(labels_dict, boundaries_dict, metadata)`` with ``train``/``val``
        lists of 1-D long tensors.
    """
    path = Path(artifact_dir)
    manifest_file = path / "topo_manifest.json"
    labels_file = path / "topo_labels.pt"
    segments_file = path / "topo_segments.pt"
    if not manifest_file.exists() or not labels_file.exists() or not segments_file.exists():
        raise FileNotFoundError(f"Missing required topo artifact files in {path}.")
    metadata = json.loads(manifest_file.read_text(encoding="utf-8"))
    if metadata.get("version") != TOPO_ARTIFACT_VERSION:
        raise ValueError(
            f"Unsupported topo artifact version {metadata.get('version')!r}; "
            f"expected {TOPO_ARTIFACT_VERSION!r}."
        )
    checksums = metadata.get("checksums")
    if not isinstance(checksums, dict):
        raise ValueError("Topo artifact manifest has no file checksums.")
    for filename in ("topo_labels.pt", "topo_segments.pt"):
        expected = checksums.get(filename)
        if not isinstance(expected, str) or _sha256_file(path / filename) != expected:
            raise ValueError(f"Topo artifact checksum mismatch for {filename}.")
    labels_dict = torch.load(labels_file, map_location="cpu", weights_only=True)
    boundaries_dict = torch.load(segments_file, map_location="cpu", weights_only=True)
    if not isinstance(labels_dict, dict) or not isinstance(boundaries_dict, dict):
        raise ValueError("Topo artifact tensors have invalid container types.")
    num_regimes = int(metadata.get("num_regimes", -1))
    for split_name in ("train", "val"):
        for container, kind in ((labels_dict, "labels"), (boundaries_dict, "boundaries")):
            items = container.get(split_name)
            if not isinstance(items, list):
                raise ValueError(f"Topo artifact {split_name} {kind} are missing or invalid.")
            for tensor in items:
                if not isinstance(tensor, torch.Tensor) or tensor.ndim != 1:
                    raise ValueError(f"Topo artifact {split_name} {kind} have invalid shape.")
                if tensor.numel() == 0:
                    raise ValueError(f"Topo artifact {split_name} {kind} are empty.")
                if kind == "labels" and num_regimes > 0 and (
                    torch.any(tensor < 0) or torch.any(tensor >= num_regimes)
                ):
                    raise ValueError(f"Topo artifact {split_name} labels are out of range.")
    return labels_dict, boundaries_dict, metadata


__all__ = ["TOPO_ARTIFACT_VERSION", "load_topo_artifact", "save_topo_artifact"]

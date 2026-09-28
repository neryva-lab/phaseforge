"""Versioned serialization, provenance, and persistence for dynamic discovery artifacts."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from phaseforge.data.dynamics.diagnostics import DiscoveryQualityReport
from phaseforge.data.dynamics.switching_linear import SLDSParameters, StickySLDS

logger = logging.getLogger(__name__)

DISCOVERY_ARTIFACT_VERSION = "2.0.0"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_discovery_artifact(
    output_dir: Path | str,
    slds: StickySLDS,
    report: DiscoveryQualityReport,
    task_name: str,
    data_config_hash: str,
    train_labels: list[np.ndarray],
    val_labels: list[np.ndarray],
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Save versioned dynamic discovery artifact to disk.

    Args:
        output_dir: Directory where the artifact bundle will be stored.
        slds: Fitted StickySLDS model.
        report: DiscoveryQualityReport.
        task_name: Name of the task (e.g. 'can', 'square').
        data_config_hash: SHA-256 fingerprint of the data config.
        train_labels: List of decoded label arrays for training trajectories.
        val_labels: List of decoded label arrays for validation trajectories.
        extra_metadata: Optional additional metadata dict.

    Returns:
        Path to the saved artifact directory.
    """
    out_path = Path(output_dir)
    staging_path = out_path.with_name(f".{out_path.name}.tmp")
    if staging_path.exists():
        shutil.rmtree(staging_path)
    staging_path.mkdir(parents=True, exist_ok=True)

    try:
        assert slds.params is not None
        params = slds.params

        # 1. Save model parameters as PT/NPY
        param_dict = {
            "pi_0": torch.from_numpy(params.pi_0).float(),
            "transition_matrix": torch.from_numpy(params.transition_matrix).float(),
            "A": torch.from_numpy(params.A).float(),
            "B": torch.from_numpy(params.B).float(),
            "b": torch.from_numpy(params.b).float(),
            "covariances": torch.from_numpy(params.covariances).float(),
        }
        params_path = staging_path / "model_parameters.pt"
        labels_path = staging_path / "decoded_labels.pt"
        torch.save(param_dict, params_path)

        # 2. Save decoded labels
        labels_dict = {
            "train": [torch.from_numpy(lbl).long() for lbl in train_labels],
            "val": [torch.from_numpy(lbl).long() for lbl in val_labels],
        }
        torch.save(labels_dict, labels_path)

        # 3. Save report and metadata
        metadata = {
            "version": DISCOVERY_ARTIFACT_VERSION,
            "task_name": task_name,
            "data_config_hash": data_config_hash,
            "num_regimes": slds.num_regimes,
            "state_dim": params.state_dim,
            "action_dim": params.action_dim,
            "sticky_kappa": slds.sticky_kappa,
            "dirichlet_alpha": slds.dirichlet_alpha,
            "ridge_lambda": slds.ridge_lambda,
            "min_variance": slds.min_variance,
            "max_em_iter": slds.max_em_iter,
            "em_tol": slds.em_tol,
            "min_duration": slds.min_duration,
            "seed": slds.seed,
            "discovery_method": "sticky_slds",
            "quality_report": asdict(report),
            "extra": extra_metadata or {},
            "checksums": {
                "model_parameters.pt": _sha256_file(params_path),
                "decoded_labels.pt": _sha256_file(labels_path),
            },
        }

        manifest_path = staging_path / "discovery_manifest.json"
        manifest_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if out_path.exists():
            shutil.rmtree(out_path)
        staging_path.replace(out_path)
    except Exception:
        if staging_path.exists():
            shutil.rmtree(staging_path)
        raise

    logger.info(f"Saved dynamic discovery artifact (v{DISCOVERY_ARTIFACT_VERSION}) to {out_path}")
    return out_path


def load_discovery_artifact(
    artifact_dir: Path | str,
) -> tuple[StickySLDS, dict[str, list[torch.Tensor]], dict[str, Any]]:
    """Load discovery artifact from disk.

    Args:
        artifact_dir: Path to directory containing the artifact.

    Returns:
        (fitted_slds, decoded_labels_dict, metadata)
    """
    path = Path(artifact_dir)
    manifest_file = path / "discovery_manifest.json"
    params_file = path / "model_parameters.pt"
    labels_file = path / "decoded_labels.pt"

    if not manifest_file.exists() or not params_file.exists() or not labels_file.exists():
        raise FileNotFoundError(f"Missing required discovery artifact files in {path}")

    metadata = json.loads(manifest_file.read_text(encoding="utf-8"))
    if metadata.get("version") != DISCOVERY_ARTIFACT_VERSION:
        raise ValueError(
            f"Unsupported discovery artifact version {metadata.get('version')!r}; "
            f"expected {DISCOVERY_ARTIFACT_VERSION!r}."
        )
    checksums = metadata.get("checksums")
    if not isinstance(checksums, dict):
        raise ValueError("Discovery artifact manifest has no file checksums.")
    for filename in ("model_parameters.pt", "decoded_labels.pt"):
        expected = checksums.get(filename)
        if not isinstance(expected, str) or _sha256_file(path / filename) != expected:
            raise ValueError(f"Discovery artifact checksum mismatch for {filename}.")

    param_tensors = torch.load(params_file, map_location="cpu", weights_only=True)
    labels_dict = torch.load(labels_file, map_location="cpu", weights_only=True)

    if not isinstance(param_tensors, dict) or not isinstance(labels_dict, dict):
        raise ValueError("Discovery artifact tensors have invalid container types.")
    required_params = {"pi_0", "transition_matrix", "A", "B", "b", "covariances"}
    if set(param_tensors) != required_params:
        raise ValueError("Discovery artifact model parameters are incomplete or unexpected.")

    num_regimes = int(metadata["num_regimes"])
    pi_0 = param_tensors["pi_0"]
    trans = param_tensors["transition_matrix"]
    A = param_tensors["A"]
    B = param_tensors["B"]
    b = param_tensors["b"]
    covs = param_tensors["covariances"]
    if not all(isinstance(tensor, torch.Tensor) for tensor in param_tensors.values()):
        raise ValueError("Discovery artifact model parameters must all be tensors.")
    if pi_0.shape != (num_regimes,) or trans.shape != (num_regimes, num_regimes):
        raise ValueError("Discovery artifact initial or transition parameters have invalid shapes.")
    if A.ndim != 3 or A.shape[0] != num_regimes or A.shape[1] != A.shape[2]:
        raise ValueError("Discovery artifact A parameters have invalid shapes.")
    if B.ndim != 3 or B.shape[:2] != (num_regimes, A.shape[1]):
        raise ValueError("Discovery artifact B parameters have invalid shapes.")
    if b.shape != (num_regimes, A.shape[1]) or covs.shape != (num_regimes, A.shape[1]):
        raise ValueError("Discovery artifact bias or covariance parameters have invalid shapes.")
    if (
        int(metadata.get("state_dim", -1)) != A.shape[1]
        or int(metadata.get("action_dim", -1)) != B.shape[2]
    ):
        raise ValueError("Discovery artifact metadata dimensions do not match model parameters.")
    if any(not torch.isfinite(tensor).all() for tensor in param_tensors.values()) or torch.any(
        covs <= 0
    ):
        raise ValueError(
            "Discovery artifact contains non-finite parameters or non-positive variances."
        )

    for split_name in ("train", "val"):
        labels = labels_dict.get(split_name)
        if not isinstance(labels, list):
            raise ValueError(f"Discovery artifact {split_name} labels are missing or invalid.")
        for label_tensor in labels:
            if not isinstance(label_tensor, torch.Tensor) or label_tensor.ndim != 1:
                raise ValueError(f"Discovery artifact {split_name} labels have invalid shape.")
            if (
                label_tensor.numel() == 0
                or torch.any(label_tensor < 0)
                or torch.any(label_tensor >= num_regimes)
            ):
                raise ValueError(
                    f"Discovery artifact {split_name} labels are out of range or empty."
                )

    slds = StickySLDS(
        num_regimes=num_regimes,
        sticky_kappa=metadata.get("sticky_kappa", 50.0),
        dirichlet_alpha=metadata.get("dirichlet_alpha", 1.0),
        ridge_lambda=metadata.get("ridge_lambda", 1e-4),
        min_variance=metadata.get("min_variance", 1e-4),
        max_em_iter=metadata.get("max_em_iter", 40),
        em_tol=metadata.get("em_tol", 1e-3),
        min_duration=metadata.get("min_duration", 3),
        seed=metadata.get("seed", 42),
    )

    slds.params = SLDSParameters(
        num_regimes=num_regimes,
        state_dim=A.shape[1],
        action_dim=B.shape[2],
        pi_0=pi_0.numpy(),
        transition_matrix=trans.numpy(),
        A=A.numpy(),
        B=B.numpy(),
        b=b.numpy(),
        covariances=covs.numpy(),
        log_likelihood_history=[],
    )

    return slds, labels_dict, metadata

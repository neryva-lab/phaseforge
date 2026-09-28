"""Audit whether the fixed six-phase scaffold is action-relevant.

This is a read-only, simulator-free diagnostic for the structured-state
dataset. It is designed for an auditable cloud run before expensive training.
It verifies cache and raw-data provenance, validates every cached trajectory,
checks train/validation split integrity, and reports trajectory statistics,
descriptive action variance explained by the six labels, and a held-out
comparison between train-fitted global and phase-conditioned action means.

The held-out comparison is a screening diagnostic, not a policy-performance
metric. It can show whether phase labels contain action information, but it
cannot prove that a phase router or a particular PhaseForge architecture is
optimal.

The default run selects the newest complete, clean-or-legacy-clean,
sequence-length-1 cache
for each requested task and requires the corresponding raw HDF5 file to be
present and SHA-256 identical to the cache manifest. It fails closed on
missing tasks, malformed caches, split inconsistencies, non-finite values,
invalid phase IDs, or raw-file mismatches.

Examples::

    uv run python scripts/analysis/phase_suitability.py
    uv run python scripts/analysis/phase_suitability.py --tasks Lift Can Square
    PHASEFORGE_DATA_DIR=/content/data uv run python scripts/analysis/phase_suitability.py
    uv run python scripts/analysis/phase_suitability.py --all-caches --allow-missing-raw

The default JSON report is ``phase_suitability_report.json`` in the current
working directory. Use ``--json-out`` to choose another location.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ORDER = ("Lift", "Can", "Square", "ToolHang", "Transport")
TASK_ALIASES = {
    "Lift": "Lift",
    "PickPlaceCan": "Can",
    "Can": "Can",
    "NutAssemblySquare": "Square",
    "Square": "Square",
    "ToolHang": "ToolHang",
    "TwoArmTransport": "Transport",
    "Transport": "Transport",
}
TASK_RAW_DIR = {
    "Lift": "lift",
    "Can": "can",
    "Square": "square",
    "ToolHang": "tool_hang",
    "Transport": "transport",
}
NUM_PHASES = 6
REPORT_VERSION = 2


@dataclass(frozen=True)
class CacheRecord:
    """A validated manifest candidate and its cache directory."""

    task: str
    cache_dir: Path
    manifest: dict[str, Any]

    @property
    def created_at(self) -> float:
        return float(self.manifest.get("created_at", 0.0))

    @property
    def raw_files(self) -> list[dict[str, Any]]:
        raw_files = self.manifest.get("provenance", {}).get("raw_files", [])
        return [item for item in raw_files if isinstance(item, dict)]

    @property
    def raw_sha256(self) -> str:
        return str(self.raw_files[0].get("sha256", "")) if self.raw_files else ""

    @property
    def clean_status(self) -> str:
        """Return declared_clean, nonclean, or unspecified."""
        configuration = str(self.manifest.get("provenance", {}).get("configuration", ""))
        corruption = _parse_float_setting(configuration, "phase_corruption_rate")
        shuffled = _parse_bool_setting(configuration, "phase_shuffle_control")
        if corruption is not None and corruption != 0.0:
            return "nonclean"
        if shuffled is True:
            return "nonclean"
        if corruption is None or shuffled is None:
            return "unspecified"
        return "declared_clean"


def _default_data_root() -> Path:
    configured = os.environ.get("PHASEFORGE_DATA_DIR")
    return Path(configured).expanduser() if configured else PROJECT_ROOT / "data"


def _repository_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _parse_float_setting(configuration: str, key: str) -> float | None:
    match = re.search(rf"(?m)^\s*{re.escape(key)}:\s*([^#\n]+)", configuration)
    if match is None:
        return None
    try:
        return float(match.group(1).strip())
    except ValueError:
        return None


def _parse_bool_setting(configuration: str, key: str) -> bool | None:
    match = re.search(rf"(?m)^\s*{re.escape(key)}:\s*([^#\n]+)", configuration)
    if match is None:
        return None
    value = match.group(1).strip().lower()
    if value in {"true", "yes", "1"}:
        return True
    if value in {"false", "no", "0"}:
        return False
    return None


def _read_manifest(path: Path, include_nonclean: bool = False) -> CacheRecord | None:
    """Return a usable manifest, or ``None`` for unrelated cache entries."""
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict) or manifest.get("complete") is not True:
        return None

    provenance = manifest.get("provenance")
    metadata = provenance.get("environment_metadata") if isinstance(provenance, dict) else None
    schema = provenance.get("state_schema") if isinstance(provenance, dict) else None
    sampling = provenance.get("sampling") if isinstance(provenance, dict) else None
    phase_labeler = provenance.get("phase_labeler") if isinstance(provenance, dict) else None
    if not isinstance(metadata, list) or len(metadata) != 1 or not isinstance(metadata[0], dict):
        return None
    if not isinstance(schema, dict) or not isinstance(sampling, dict):
        return None
    if not isinstance(phase_labeler, dict):
        return None
    if sampling.get("sequence_length") != 1 or sampling.get("stride") != 1:
        return None

    task = TASK_ALIASES.get(str(metadata[0].get("env_name", "")))
    if task is None:
        return None
    try:
        state_dim = int(schema.get("state_dim", 0))
        action_dim = int(schema.get("action_dim", 0))
        num_phases = int(phase_labeler.get("num_phases", 0))
        num_tasks = int(manifest.get("num_tasks", 0))
        num_trajectories = int(manifest.get("num_trajectories", 0))
    except (TypeError, ValueError):
        return None
    if state_dim <= 0 or action_dim <= 0:
        return None
    if num_phases != NUM_PHASES:
        return None
    if num_tasks != 1:
        return None
    if num_trajectories <= 0:
        return None
    if not isinstance(manifest.get("splits"), dict):
        return None

    configuration = str(provenance.get("configuration", ""))
    corruption = _parse_float_setting(configuration, "phase_corruption_rate")
    shuffled = _parse_bool_setting(configuration, "phase_shuffle_control")
    if not include_nonclean and (corruption is not None and corruption != 0.0 or shuffled is True):
        return None

    return CacheRecord(task=task, cache_dir=path.parent, manifest=manifest)


def _select_records(
    cache_root: Path,
    tasks: set[str],
    all_caches: bool,
    include_nonclean: bool,
) -> list[CacheRecord]:
    records = [
        record
        for manifest_path in sorted(cache_root.glob("*/manifest.json"))
        if (record := _read_manifest(manifest_path, include_nonclean)) is not None
        and record.task in tasks
    ]
    if all_caches:
        return sorted(
            records,
            key=lambda record: (TASK_ORDER.index(record.task), record.cache_dir.name),
        )

    selected: list[CacheRecord] = []
    for task in TASK_ORDER:
        candidates = [record for record in records if record.task == task]
        if candidates:
            selected.append(
                max(
                    candidates,
                    key=lambda record: (record.created_at, record.cache_dir.name),
                )
            )
    return selected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_candidates(record: CacheRecord, data_root: Path) -> list[Path]:
    names = [str(item.get("name", "")) for item in record.raw_files]
    names = [name for name in names if name]
    task_dir = TASK_RAW_DIR[record.task]
    candidates: list[Path] = []
    for name in names:
        candidates.append(data_root / "raw" / "robomimic" / task_dir / name)
        candidates.append(data_root / "raw" / task_dir / name)
        candidates.append(record.cache_dir / name)
    return candidates


def _verify_raw_files(
    record: CacheRecord,
    data_root: Path,
    allow_missing_raw: bool,
    skip_raw_hash: bool,
) -> dict[str, Any]:
    if not record.raw_files:
        if allow_missing_raw:
            return {"status": "manifest_hash_missing", "verified": False}
        raise ValueError(f"{record.cache_dir}: manifest has no raw_files provenance")

    result: dict[str, Any] = {"status": "missing", "verified": False, "files": []}
    for raw in record.raw_files:
        name = str(raw.get("name", ""))
        expected_hash = str(raw.get("sha256", ""))
        try:
            expected_size = int(raw.get("size", -1))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{record.task}: invalid raw-file size for {name!r}") from exc
        if not expected_hash and not allow_missing_raw and not skip_raw_hash:
            raise ValueError(f"{record.task}: raw file {name!r} has no manifest SHA-256")
        path = next(
            (
                candidate
                for candidate in _raw_candidates(record, data_root)
                if candidate.name == name and candidate.is_file()
            ),
            None,
        )
        if path is None:
            if allow_missing_raw:
                result["files"].append({"name": name, "status": "missing"})
                continue
            raise ValueError(
                f"{record.task}: raw file {name!r} was not found under {data_root}. "
                "Use --allow-missing-raw only when the raw HDF5 is intentionally "
                "not available; that run is not raw-data verified."
            )
        actual_size = path.stat().st_size
        if expected_size >= 0 and actual_size != expected_size:
            raise ValueError(
                f"{record.task}: raw file size mismatch for {path}: "
                f"actual={actual_size}, manifest={expected_size}"
            )
        actual_hash = "" if skip_raw_hash else _sha256(path)
        if expected_hash and actual_hash and actual_hash != expected_hash:
            raise ValueError(
                f"{record.task}: raw SHA-256 mismatch for {path}: "
                f"actual={actual_hash}, manifest={expected_hash}"
            )
        result["files"].append(
            {
                "name": name,
                "path": str(path),
                "size": actual_size,
                "sha256": actual_hash or expected_hash,
                "status": "present_hash_skipped" if skip_raw_hash else "verified",
            }
        )
    if result["files"] and all(
        item["status"] in {"verified", "present_hash_skipped"}
        for item in result["files"]
    ):
        result["status"] = "present_hash_skipped" if skip_raw_hash else "verified"
        result["verified"] = not skip_raw_hash
    return result


def _mse(values: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.square(values.astype(np.float64) - prediction.astype(np.float64))))


def _limit_trajectory_files(trajectory_files: list[Path], limit: int) -> list[Path]:
    """Choose a small split-covered subset for debugging."""
    if limit < 2:
        raise ValueError("--max-trajectories must be at least 2 to cover train and validation")
    by_split: dict[str, list[Path]] = {"train": [], "val": []}
    for path in trajectory_files:
        item = torch.load(path, map_location="cpu", weights_only=False)
        split = str(item.get("dataset_split", ""))
        if split in by_split:
            by_split[split].append(path)
    if not by_split["train"] or not by_split["val"]:
        raise ValueError("--max-trajectories requires both train and validation trajectories")
    val_count = max(1, limit // 3)
    train_count = limit - val_count
    return by_split["train"][:train_count] + by_split["val"][:val_count]


def _phase_means(actions: np.ndarray, phases: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    action_dim = actions.shape[1]
    means = np.zeros((NUM_PHASES, action_dim), dtype=np.float64)
    counts = np.bincount(phases, minlength=NUM_PHASES).astype(np.int64)
    for phase_id in range(NUM_PHASES):
        if counts[phase_id]:
            means[phase_id] = actions[phases == phase_id].mean(axis=0)
    return means, counts


def _bootstrap_interval(
    values: list[float], seed: int, resamples: int
) -> tuple[float, float] | None:
    if not values:
        return None
    if len(values) == 1:
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(seed)
    samples = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, len(samples), size=(resamples, len(samples)))
    boot_means = samples[indices].mean(axis=1)
    low, high = np.quantile(boot_means, [0.025, 0.975])
    return float(low), float(high)


def _load_summary(
    record: CacheRecord,
    data_root: Path,
    max_trajectories: int | None,
    allow_missing_raw: bool,
    skip_raw_hash: bool,
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    manifest = record.manifest
    provenance = manifest["provenance"]
    schema = provenance["state_schema"]
    state_dim = int(schema["state_dim"])
    action_dim = int(schema["action_dim"])
    trajectory_files = sorted((record.cache_dir / "trajectories").glob("*.pt"))
    expected_trajectories = int(manifest["num_trajectories"])
    if len(trajectory_files) != expected_trajectories:
        raise ValueError(
            f"{record.task}: manifest declares {expected_trajectories} trajectories, "
            f"but found {len(trajectory_files)} files"
        )
    if max_trajectories is not None:
        trajectory_files = _limit_trajectory_files(trajectory_files, max_trajectories)
    if not trajectory_files:
        raise ValueError(f"{record.task}: cache has no trajectory files")

    try:
        expected_split_counts = {
            key: int(value) for key, value in manifest["splits"].items()
        }
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{record.task}: manifest split counts are malformed") from exc
    expected_demo_keys = provenance.get("split_demo_keys", {})
    if not isinstance(expected_demo_keys, dict):
        raise ValueError(f"{record.task}: provenance.split_demo_keys is missing or malformed")
    if any(not isinstance(expected_demo_keys.get(split), list) for split in ("train", "val")):
        raise ValueError(f"{record.task}: manifest must provide train and val demo-key lists")

    rows: list[dict[str, Any]] = []
    seen_demo_keys: dict[str, set[str]] = {"train": set(), "val": set()}
    all_actions: list[np.ndarray] = []
    all_phases: list[np.ndarray] = []
    transition_counts = np.zeros((NUM_PHASES, NUM_PHASES), dtype=np.int64)
    lengths: list[int] = []
    switch_counts: list[int] = []
    for trajectory_path in trajectory_files:
        try:
            item = torch.load(trajectory_path, map_location="cpu", weights_only=False)
        except Exception as exc:  # pragma: no cover - torch errors vary by version
            raise ValueError(f"{record.task}: failed to load {trajectory_path}: {exc}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"{trajectory_path}: trajectory payload is not a dictionary")
        required = {"state", "action", "phase", "dataset_split", "demo_key"}
        missing = sorted(required - set(item))
        if missing:
            raise ValueError(f"{trajectory_path}: missing required keys {missing}")

        state = np.asarray(item["state"], dtype=np.float32)
        action = np.asarray(item["action"], dtype=np.float32)
        phase = np.asarray(item["phase"], dtype=np.int64).reshape(-1)
        split = str(item["dataset_split"])
        demo_key = str(item["demo_key"])
        if split not in {"train", "val"}:
            raise ValueError(f"{trajectory_path}: unsupported dataset_split={split!r}")
        if state.ndim != 2 or state.shape[1] != state_dim:
            raise ValueError(f"{trajectory_path}: state shape {state.shape} != (*, {state_dim})")
        if action.ndim != 2 or action.shape != (state.shape[0], action_dim):
            raise ValueError(
                f"{trajectory_path}: action shape {action.shape} is incompatible with "
                f"state shape {state.shape} and action_dim={action_dim}"
            )
        if phase.size != state.shape[0] or np.any((phase < 0) | (phase >= NUM_PHASES)):
            raise ValueError(f"{trajectory_path}: invalid phase labels")
        if not (np.isfinite(state).all() and np.isfinite(action).all()):
            raise ValueError(f"{trajectory_path}: non-finite state or action")
        if np.any(action < -1.000001) or np.any(action > 1.000001):
            raise ValueError(f"{trajectory_path}: action is outside the declared [-1, 1] contract")

        expected_keys = {str(key) for key in expected_demo_keys.get(split, [])}
        if expected_keys and demo_key not in expected_keys:
            raise ValueError(
                f"{trajectory_path}: demo_key {demo_key!r} is absent from manifest {split} split"
            )
        seen_demo_keys[split].add(demo_key)
        if len(phase) == 0:
            raise ValueError(f"{trajectory_path}: empty trajectory")
        if len(phase) > 1:
            transition_counts += np.histogram2d(
                phase[:-1],
                phase[1:],
                bins=(NUM_PHASES, NUM_PHASES),
                range=((0, NUM_PHASES), (0, NUM_PHASES)),
            )[0].astype(np.int64)
        lengths.append(int(len(phase)))
        switch_counts.append(int(np.count_nonzero(phase[1:] != phase[:-1])))
        rows.append({"split": split, "action": action, "phase": phase})
        all_actions.append(action)
        all_phases.append(phase)

    if max_trajectories is None:
        actual_split_counts = {
            split: sum(row["split"] == split for row in rows)
            for split in {"train", "val"}
        }
        for split, expected in expected_split_counts.items():
            if actual_split_counts.get(split, 0) != expected:
                raise ValueError(
                    f"{record.task}: split count mismatch for {split}: "
                    f"actual={actual_split_counts.get(split, 0)}, manifest={expected}"
                )
        train_keys = seen_demo_keys["train"]
        val_keys = seen_demo_keys["val"]
        if train_keys & val_keys:
            raise ValueError(f"{record.task}: demo keys overlap between train and validation")

    actions = np.concatenate(all_actions, axis=0)
    phases = np.concatenate(all_phases, axis=0)
    total_phase_means, total_phase_counts = _phase_means(actions, phases)
    total_mean = actions.mean(axis=0)
    total_mse = _mse(actions, np.broadcast_to(total_mean, actions.shape))
    within_sse = 0.0
    for phase_id in range(NUM_PHASES):
        mask = phases == phase_id
        if mask.any():
            errors = actions[mask] - total_phase_means[phase_id]
            within_sse += float(np.square(errors.astype(np.float64)).sum())
    within_mse = within_sse / (len(actions) * action_dim)
    explained = 0.0 if total_mse <= 1e-12 else max(0.0, 1.0 - within_mse / total_mse)

    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    if not train_rows or not val_rows:
        raise ValueError(f"{record.task}: both train and validation splits are required")
    train_actions = np.concatenate([row["action"] for row in train_rows], axis=0)
    train_phases = np.concatenate([row["phase"] for row in train_rows], axis=0)
    val_actions = np.concatenate([row["action"] for row in val_rows], axis=0)
    val_phases = np.concatenate([row["phase"] for row in val_rows], axis=0)
    train_mean = train_actions.mean(axis=0)
    train_phase_means, train_phase_counts = _phase_means(train_actions, train_phases)
    missing_train_phases = [
        phase_id for phase_id in range(NUM_PHASES) if train_phase_counts[phase_id] == 0
    ]
    if missing_train_phases:
        raise ValueError(
            f"{record.task}: training split cannot fit phase-conditioned means; "
            f"missing phases={missing_train_phases}"
        )
    val_global_mse = _mse(val_actions, np.broadcast_to(train_mean, val_actions.shape))
    val_phase_mse = _mse(val_actions, train_phase_means[val_phases])
    val_reduction = (
        0.0 if val_global_mse <= 1e-12 else (val_global_mse - val_phase_mse) / val_global_mse
    )
    per_trajectory_reduction: list[float] = []
    for row in val_rows:
        trajectory_action = row["action"]
        trajectory_phase = row["phase"]
        global_error = _mse(
            trajectory_action,
            np.broadcast_to(train_mean, trajectory_action.shape),
        )
        phase_error = _mse(trajectory_action, train_phase_means[trajectory_phase])
        if global_error > 1e-12:
            per_trajectory_reduction.append((global_error - phase_error) / global_error)
    ci = _bootstrap_interval(per_trajectory_reduction, bootstrap_seed, bootstrap_resamples)

    def phase_fractions(values: np.ndarray) -> list[float]:
        counts = np.bincount(values, minlength=NUM_PHASES)
        return [float(value / len(values)) for value in counts]

    raw_verification = _verify_raw_files(record, data_root, allow_missing_raw, skip_raw_hash)
    metadata = provenance["environment_metadata"][0]
    return {
        "task": record.task,
        "cache": record.cache_dir.name,
        "created_at": record.created_at,
        "cache_config_hash": manifest.get("config_hash"),
        "code_git_commit": provenance.get("code_git_commit"),
        "raw_sha256_manifest": record.raw_sha256,
        "schema_version": schema.get("schema_version"),
        "environment": {"name": metadata.get("env_name"), "version": metadata.get("env_version")},
        "state_dim": state_dim,
        "action_dim": action_dim,
        "sequence_length": provenance["sampling"].get("sequence_length"),
        "clean_status": record.clean_status,
        "integrity": {
            "manifest_complete": manifest.get("complete") is True,
            "trajectory_files": len(trajectory_files),
            "manifest_trajectory_files": expected_trajectories,
            "manifest_split_counts": expected_split_counts,
            "raw_files": raw_verification,
            "partial_analysis": max_trajectories is not None,
        },
        "trajectories": {
            "count": len(trajectory_files),
            "steps": int(len(phases)),
            "length_min": min(lengths),
            "length_median": int(statistics.median(lengths)),
            "length_max": max(lengths),
            "mean_phase_switches": float(np.mean(switch_counts)),
        },
        "phase_distribution": {
            "counts_all": total_phase_counts.tolist(),
            "fractions_all": phase_fractions(phases),
            "fractions_train": phase_fractions(train_phases),
            "fractions_val": phase_fractions(val_phases),
            "transition_counts": transition_counts.tolist(),
            "label_entropy_normalized": float(
                -sum(
                    fraction * np.log(fraction)
                    for fraction in phase_fractions(phases)
                    if fraction > 0
                )
                / np.log(NUM_PHASES)
            ),
        },
        "action_phase_relationship": {
            "descriptive_all_data": {
                "global_action_mse": total_mse,
                "within_phase_action_mse": within_mse,
                "phase_action_variance_explained": explained,
            },
            "heldout_val_train_fitted": {
                "global_mean_action_mse": val_global_mse,
                "phase_mean_action_mse": val_phase_mse,
                "relative_mse_reduction": val_reduction,
                "per_trajectory_reduction_mean": float(np.mean(per_trajectory_reduction))
                if per_trajectory_reduction
                else None,
                "per_trajectory_reduction_bootstrap_ci95": list(ci) if ci is not None else None,
                "bootstrap_seed": bootstrap_seed,
                "bootstrap_resamples": bootstrap_resamples,
                "train_phase_counts": train_phase_counts.tolist(),
                "missing_train_phases": missing_train_phases,
            },
        },
    }


def _print_table(rows: list[dict[str, Any]]) -> None:
    print(
        "task       cache             demos steps  len[min/med/max]  "
        "all-phase-R2  heldout-MSE-reduction  bootstrap-CI95  raw-status"
    )
    for row in rows:
        relationship = row["action_phase_relationship"]
        descriptive = relationship["descriptive_all_data"]
        heldout = relationship["heldout_val_train_fitted"]
        ci = heldout["per_trajectory_reduction_bootstrap_ci95"]
        ci_text = "n/a" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        raw_status = row["integrity"]["raw_files"]["status"]
        trajectories = row["trajectories"]
        print(
            f"{row['task']:<10} {row['cache']:<17} {trajectories['count']:>5} "
            f"{trajectories['steps']:>5}  "
            f"{trajectories['length_min']}/{trajectories['length_median']}/"
            f"{trajectories['length_max']:<7} "
            f"{descriptive['phase_action_variance_explained']:.3f}        "
            f"{heldout['relative_mse_reduction']:.3f}                 "
            f"{ci_text:<18} {raw_status}"
        )
        fractions = ", ".join(
            f"{value:.3f}" for value in row["phase_distribution"]["fractions_all"]
        )
        print(f"  phase fractions [0..5]: {fractions}")
        print(f"  raw SHA-256: {row['raw_sha256_manifest']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Dataset root; defaults to PHASEFORGE_DATA_DIR or ./data.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Override cache directory; defaults to <data-root>/processed/cache.",
    )
    parser.add_argument("--tasks", nargs="+", choices=TASK_ORDER, default=list(TASK_ORDER))
    parser.add_argument(
        "--all-caches",
        action="store_true",
        help="Audit every matching cache instead of selecting the newest per task.",
    )
    parser.add_argument(
        "--include-nonclean",
        action="store_true",
        help="Include phase-corrupted/shuffled caches for explicit auditing.",
    )
    parser.add_argument(
        "--allow-missing-tasks",
        action="store_true",
        help="Do not fail when a requested task has no matching cache.",
    )
    parser.add_argument(
        "--allow-missing-raw",
        action="store_true",
        help="Allow analysis without raw HDF5; marks raw provenance unverified.",
    )
    parser.add_argument(
        "--skip-raw-hash",
        action="store_true",
        help="Check raw-file presence/size but skip SHA-256 calculation.",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="Debug-only partial analysis; omit for a decision run.",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("phase_suitability_report.json"),
    )
    args = parser.parse_args(argv)
    if args.max_trajectories is not None and args.max_trajectories < 1:
        parser.error("--max-trajectories must be positive")
    if args.bootstrap_resamples < 100:
        parser.error("--bootstrap-resamples must be at least 100")

    data_root = (args.data_root or _default_data_root()).expanduser().resolve()
    cache_root = (args.cache_root or data_root / "processed" / "cache").expanduser().resolve()
    requested_tasks = set(args.tasks)
    records = _select_records(
        cache_root,
        requested_tasks,
        args.all_caches,
        args.include_nonclean,
    )
    found_tasks = {record.task for record in records}
    missing_tasks = [
        task for task in TASK_ORDER if task in requested_tasks and task not in found_tasks
    ]
    if missing_tasks and not args.allow_missing_tasks:
        print(
            f"[FAIL] missing matching clean sequence-length-1 cache(s) for "
            f"{missing_tasks} under {cache_root}",
            file=sys.stderr,
        )
        return 2
    if not records:
        print(f"[FAIL] no matching caches found under {cache_root}", file=sys.stderr)
        return 2

    if not args.all_caches:
        for task in found_tasks:
            candidates = _select_records(
                cache_root, {task}, True, args.include_nonclean
            )
            raw_hashes = {record.raw_sha256 for record in candidates if record.raw_sha256}
            if len(raw_hashes) > 1:
                print(
                    f"[FAIL] {task} has multiple raw SHA-256 values across matching caches: "
                    f"{sorted(raw_hashes)}; use --all-caches to audit them explicitly",
                    file=sys.stderr,
                )
                return 2

    print(f"[phase-suitability] data_root={data_root}")
    print(f"[phase-suitability] cache_root={cache_root}")
    print(f"[phase-suitability] selected={len(records)} task/cache record(s)")
    rows: list[dict[str, Any]] = []
    try:
        for record in records:
            rows.append(
                _load_summary(
                    record,
                    data_root,
                    args.max_trajectories,
                    args.allow_missing_raw,
                    args.skip_raw_hash,
                    args.bootstrap_seed,
                    args.bootstrap_resamples,
                )
            )
    except ValueError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2

    report = {
        "report_version": REPORT_VERSION,
        "analysis": {
            "name": "phase_suitability",
            "report_version": REPORT_VERSION,
            "read_only": True,
            "simulator_used": False,
            "repository_git_commit": _repository_commit(),
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "data_root": str(data_root),
            "cache_root": str(cache_root),
            "command": sys.argv,
            "requested_tasks": list(args.tasks),
            "selected_records": len(rows),
            "missing_tasks": missing_tasks,
            "partial_analysis": args.max_trajectories is not None,
            "include_nonclean": args.include_nonclean,
            "raw_hash_skipped": args.skip_raw_hash,
        },
        "rows": rows,
    }
    output_path = args.json_out.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _print_table(rows)
    print(f"[phase-suitability] report={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

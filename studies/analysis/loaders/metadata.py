"""Typed loaders for run metadata (init diagnostics, environment, provenance, timings)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from studies.analysis.common import io as cio


@dataclass(frozen=True)
class InitRouting:
    """Bootstrap-instant routing diagnostics (metadata/init_routing.json)."""

    t0_nmi: float
    t0_routing_entropy: float
    t0_normalized_routing_entropy: float | None
    t0_collapse_rate: float | None
    t0_dead_expert_count: int | None
    t0_phase_head_accuracy: float | None
    t0_top1_expert_frequencies: tuple[float, ...] = ()


@dataclass(frozen=True)
class InitExpert:
    """Bootstrap fingerprint (metadata/init_expert.json)."""

    router_init_type: str
    num_experts: int
    top_k: int
    expert_init_type: str
    drop_rate: float | None
    init_seed: int | None
    dropped_indices_sha256: str | None
    training_seed: int | None


@dataclass(frozen=True)
class EnvironmentRecord:
    """metadata/environment.json — pinned stack for provenance (A10)."""

    git_branch: str | None
    git_sha: str | None
    platform: str | None
    python: str | None
    device: str | None
    packages: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TimingsRecord:
    """timings.json — wall-clock per phase (A9)."""

    raw: dict[str, Any] = field(default_factory=dict)


def _load_dict(path: Path) -> dict[str, Any]:
    data = cio.read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def load_init_routing(run_dir: Path) -> InitRouting | None:
    path = run_dir / "metadata" / "init_routing.json"
    if not path.is_file():
        return None
    d = _load_dict(path)
    for key in ("t0_nmi_phase_top1", "t0_mean_routing_entropy"):
        if key not in d:
            raise ValueError(f"{path}: missing {key!r}")
    return InitRouting(
        t0_nmi=float(d["t0_nmi_phase_top1"]),
        t0_routing_entropy=float(d["t0_mean_routing_entropy"]),
        t0_normalized_routing_entropy=(
            float(d["t0_normalized_routing_entropy"])
            if d.get("t0_normalized_routing_entropy") is not None
            else None
        ),
        t0_collapse_rate=(
            float(d["t0_collapse_rate"]) if d.get("t0_collapse_rate") is not None else None
        ),
        t0_dead_expert_count=(
            int(d["t0_dead_expert_count"]) if d.get("t0_dead_expert_count") is not None else None
        ),
        t0_phase_head_accuracy=(
            float(d["t0_phase_head_accuracy"])
            if d.get("t0_phase_head_accuracy") is not None
            else None
        ),
        t0_top1_expert_frequencies=tuple(float(x) for x in d.get("t0_top1_expert_frequencies", []) if x is not None),
    )


def load_init_expert(run_dir: Path) -> InitExpert | None:
    path = run_dir / "metadata" / "init_expert.json"
    if not path.is_file():
        return None
    d = _load_dict(path)
    router = d.get("router") or {}
    expert = d.get("expert_init") or {}
    if "num_experts" not in router or "type" not in expert:
        raise ValueError(f"{path}: missing router/expert_init identity")
    return InitExpert(
        router_init_type=str(router.get("init_type", "")),
        num_experts=int(router["num_experts"]),
        top_k=int(router.get("top_k", 0)),
        expert_init_type=str(expert["type"]),
        drop_rate=float(expert["drop_rate"]) if "drop_rate" in expert else None,
        init_seed=int(expert["init_seed"]) if "init_seed" in expert else None,
        dropped_indices_sha256=expert.get("dropped_indices_sha256"),
        training_seed=int(d["training_seed"]) if "training_seed" in d else None,
    )


def load_environment(run_dir: Path) -> EnvironmentRecord | None:
    path = run_dir / "metadata" / "environment.json"
    if not path.is_file():
        return None
    d = _load_dict(path)
    extra = d.get("extra") or {}
    packages = d.get("packages") or {}
    return EnvironmentRecord(
        git_branch=d.get("git_branch"),
        git_sha=d.get("git_sha"),
        platform=d.get("platform"),
        python=d.get("python"),
        device=extra.get("device"),
        packages={str(k): str(v) for k, v in packages.items()} if packages else {},
    )


def load_timings(run_dir: Path) -> TimingsRecord | None:
    path = run_dir / "timings.json"
    if not path.is_file():
        return None
    return TimingsRecord(raw=_load_dict(path))

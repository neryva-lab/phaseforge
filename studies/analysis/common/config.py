"""OmegaConf-backed configuration for the analysis pipeline."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"

#: Tests point this at a temporary config with synthetic namespaces.
CONFIG_ENV = "PHASEFORGE_ANALYSIS_CONFIG"


@lru_cache(maxsize=1)
def get_config():
    path = Path(os.environ.get(CONFIG_ENV, str(CONFIG_PATH)))
    if not path.is_file():
        raise FileNotFoundError(f"Analysis config not found: {path}")
    return OmegaConf.load(path)


def _resolve(base: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else base / p


def namespace_root(namespace: str) -> Path:
    cfg = get_config()
    if namespace not in cfg.namespaces:
        known = list(cfg.namespaces.keys())
        raise KeyError(f"Unknown namespace {namespace!r}; configured: {known}")
    return _resolve(REPO_ROOT, str(cfg.namespaces[namespace].root))


def namespace_manifest(namespace: str) -> Path:
    cfg = get_config()
    return _resolve(REPO_ROOT, str(cfg.namespaces[namespace].manifest))


def paper_root() -> Path:
    return _resolve(REPO_ROOT, str(get_config().output.paper_root))


def generation_manifest_path() -> Path:
    cfg = get_config()
    return paper_root().parent / str(cfg.output.manifest_name)

"""A12 — hyperparameters & configuration (from resolved_config.yaml per method)."""

from __future__ import annotations

from pathlib import Path

from studies.analysis.common import registry
from studies.analysis.dataset import AnalysisDataset
from studies.analysis.render.tables import Table, save_table

_ROWS = (
    (
        "Encoder hidden / latent",
        lambda cfg: (
            f"{list(cfg.get('models', {}).get('encoder', {}).get('hidden_dims', []))}"
            f" / {cfg.get('models', {}).get('encoder', {}).get('latent_dim', '—')}"
        ),
    ),
    (
        "Experts (top-k)",
        lambda cfg: (
            f"{cfg.get('models', {}).get('router', {}).get('num_experts', '—')} "
            f"(top-{cfg.get('models', {}).get('router', {}).get('top_k', '—')})"
            if "router" in cfg.get("models", {})
            else "—"
        ),
    ),
    (
        "Router init / expert init",
        lambda cfg: (
            f"{cfg.get('models', {}).get('router_init', {}).get('type', 'random')}"
            f" / {cfg.get('models', {}).get('expert_init', {}).get('type', 'random')}"
            if "router" in cfg.get("models", {}) or "expert_init" in cfg.get("models", {})
            else "—"
        ),
    ),
    (
        "Drop rate",
        lambda cfg: str(cfg.get("models", {}).get("expert_init", {}).get("drop_rate", "—")),
    ),
    (
        "Batch size",
        lambda cfg: str(cfg.get("data", {}).get("batch_size", "256")),
    ),
    (
        "Learning rate",
        lambda cfg: (
            str(cfg.get("train", {}).get("optimizer", {}).get("lr"))
            if "optimizer" in cfg.get("train", {})
            else str(cfg.get("train", {}).get("lr", "0.0001"))
        ),
    ),
    (
        "Epochs",
        lambda cfg: str(cfg.get("train", {}).get("epochs", "—")),
    ),
    (
        "Early stopping",
        lambda cfg: str(bool(cfg.get("train", {}).get("early_stopping", {}).get("enabled", False))),
    ),
)


def _resolved_config(dataset: AnalysisDataset, method: str) -> dict | None:
    cands = (method, "precision_residual_phaseforge") if method == "phaseforge" else (method,)
    for (task, name, seed, stage), run in dataset.train_runs.items():
        if name in cands:
            path = run.path / "resolved_config.yaml"
            if path.is_file():
                import yaml

                data = yaml.safe_load(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
    return None


def generate(dataset: AnalysisDataset) -> list[Path]:
    rows = []
    methods = list(registry.matrix_method_names())
    configs: dict[str, dict] = {}
    for method in methods:
        configs[method] = _resolved_config(dataset, method)
    for label, getter in _ROWS:
        row = [label]
        for method in methods:
            cfg = configs.get(method)
            try:
                row.append(getter(cfg) if cfg else "--")
            except (KeyError, TypeError):
                row.append("--")
        rows.append(row)
    table = Table(
        headers=["Setting"] + [registry.display_name(m) for m in methods],
        rows=rows,
        caption="Resolved hyperparameters per method (from the sweep's own "
        "resolved\\_config.yaml artifacts; values as run, not as documented).",
        notes=(
            "One representative run per method (seed shown in A10); settings that "
            "do not vary across seeds.",
        ),
    )
    return save_table(table, "tables/A12_hyperparameters")

"""T3 — capacity & fairness accounting (wraps scripts/analysis/fairness_accounting)."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from studies.analysis.common import registry
from studies.analysis.common.config import REPO_ROOT
from studies.analysis.dataset import AnalysisDataset
from studies.analysis.render.tables import Table, save_table


def _compose(method_model: str, data: str) -> OmegaConf:
    config_dir = str(REPO_ROOT / "phaseforge" / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(
            config_name="main",
            overrides=[f"models={method_model}", f"data={data}"],
        )


def generate(dataset: AnalysisDataset) -> list[Path]:
    from scripts.analysis.fairness_accounting import calculate_model_accounting

    rows = []
    lift = "lift"
    for method in registry.matrix_method_names():
        spec = next(m for m in registry.methods("final") if m.name == method and m.task == "Lift")
        cfg = _compose(spec.model, spec.data if spec.data != "common" else lift)
        record = calculate_model_accounting(cfg, spec.model, spec.stage2_source or "None")
        rows.append(
            [
                registry.display_name(method),
                f"{record.deployed_params:,}",
                f"{record.active_params_per_sample:,}",
                f"{record.forward_flops_approx:,}",
                f"{record.total_optimizer_steps:,}",
            ]
        )
    table = Table(
        headers=["Method", "Deployed params", "Active/sample", "~FLOPs/fwd", "Optimizer steps"],
        rows=rows,
        caption="Capacity and fairness accounting per method (Stage~2 deployed configuration).",
    )
    return save_table(table, "tables/T3_capacity")

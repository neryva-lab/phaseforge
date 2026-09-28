"""Asset registry: the publication contract (figures_tables_plan.md).

Each planned asset is an ``AssetSpec``. ``verify`` cross-checks the registry
against the plan; ``generate`` dispatches on it. F1 is a hand-drawn schematic
(``generator=None``): verify expects the file to be placed manually under
paper/figures/main/.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pathlib import Path

    from studies.analysis.dataset import AnalysisDataset

    Generator = Callable[[AnalysisDataset], list[Path]]


@dataclass(frozen=True)
class AssetSpec:
    id: str
    kind: str  # figure | table | schematic
    section: str  # main | appendix
    title: str
    priority: str  # P0 | P1 | P2
    module: str  # studies.analysis.assets.<name>
    outputs: tuple[str, ...]  # relative to paper_root, WITH suffix
    generator_attr: str = "generate"


def _spec(
    asset_id: str,
    kind: str,
    section: str,
    title: str,
    priority: str,
    module: str,
    outputs: tuple[str, ...],
) -> AssetSpec:
    return AssetSpec(
        id=asset_id,
        kind=kind,
        section=section,
        title=title,
        priority=priority,
        module=module,
        outputs=outputs,
    )


ASSET_REGISTRY: dict[str, AssetSpec] = {}
for spec in (
    _spec(
        "F1",
        "figure",
        "main",
        "Method overview schematic",
        "P0",
        "f1_overview",
        ("figures/main/F1_overview.pdf", "figures/main/F1_overview.png"),
    ),
    _spec(
        "F2",
        "figure",
        "main",
        "Paired success deltas per task",
        "P0",
        "f2_paired_deltas",
        ("figures/main/F2_paired_deltas.pdf", "figures/main/F2_paired_deltas.png"),
    ),
    _spec(
        "F3",
        "figure",
        "main",
        "Specialization dynamics (NMI / switch rate)",
        "P0",
        "f3_specialization_dynamics",
        ("figures/main/F3_specialization.pdf", "figures/main/F3_specialization.png"),
    ),
    _spec(
        "F4",
        "figure",
        "main",
        "Commanded action discontinuity at expert switches",
        "P0",
        "f4_action_discontinuity",
        ("figures/main/F4_action_discontinuity.pdf", "figures/main/F4_action_discontinuity.png"),
    ),
    _spec(
        "T1",
        "table",
        "main",
        "Five-task success matrix",
        "P0",
        "t1_success_matrix",
        ("tables/T1_success_matrix.tex", "tables/T1_success_matrix.md"),
    ),
    _spec(
        "T2",
        "table",
        "main",
        "Controlled router-initialization ablation (Can & Square)",
        "P0",
        "t2_causal_controls",
        ("tables/T2_causal_controls.tex", "tables/T2_causal_controls.md"),
    ),
    _spec(
        "T3",
        "table",
        "main",
        "Capacity & fairness accounting",
        "P0",
        "t3_capacity",
        ("tables/T3_capacity.tex", "tables/T3_capacity.md"),
    ),
    _spec(
        "A1",
        "table",
        "appendix",
        "Per-seed raw success rates",
        "P0",
        "a1_per_seed_raws",
        ("tables/A1_per_seed_raws.tex", "tables/A1_per_seed_raws.md"),
    ),
    _spec(
        "A2",
        "table",
        "appendix",
        "Offline action MSE matrix",
        "P1",
        "a2_offline_mse",
        ("tables/A2_offline_mse.tex", "tables/A2_offline_mse.md"),
    ),
    _spec(
        "A3",
        "figure",
        "appendix",
        "Training curves (all methods x tasks)",
        "P1",
        "a3_training_curves",
        ("figures/appendix/A3_training_curves.pdf", "figures/appendix/A3_training_curves.png"),
    ),
    _spec(
        "A4",
        "table",
        "appendix",
        "Full ablation table",
        "P0",
        "a4_ablation_full",
        ("tables/A4_ablation_full.tex", "tables/A4_ablation_full.md"),
    ),
    _spec(
        "A5",
        "figure",
        "appendix",
        "Episode outcome / failure categories",
        "P1",
        "a5_failure_categories",
        (
            "figures/appendix/A5_failure_categories.pdf",
            "figures/appendix/A5_failure_categories.png",
        ),
    ),
    _spec(
        "A6",
        "figure",
        "appendix",
        "Steps-to-outcome ECDFs",
        "P2",
        "a6_steps_ecdf",
        ("figures/appendix/A6_steps_ecdf.pdf", "figures/appendix/A6_steps_ecdf.png"),
    ),
    _spec(
        "A7",
        "table",
        "appendix",
        "t=0 routing alignment across router inits",
        "P1",
        "a7_t0_alignment",
        ("tables/A7_t0_alignment.tex", "tables/A7_t0_alignment.md"),
    ),
    _spec(
        "A8",
        "figure",
        "appendix",
        "Expert balance-score trajectories (Stage 2)",
        "P2",
        "a8_balance_trajectories",
        ("figures/appendix/A8_balance.pdf", "figures/appendix/A8_balance.png"),
    ),
    _spec(
        "A9",
        "table",
        "appendix",
        "Compute cost (wall-clock, throughput, memory)",
        "P2",
        "a9_compute_cost",
        ("tables/A9_compute_cost.tex", "tables/A9_compute_cost.md"),
    ),
    _spec(
        "A10",
        "table",
        "appendix",
        "Protocol & provenance",
        "P0",
        "a10_provenance",
        ("tables/A10_provenance.tex", "tables/A10_provenance.md"),
    ),
    _spec(
        "A11",
        "figure",
        "appendix",
        "Router-init family dynamics (Can & Square)",
        "P2",
        "a11_router_family_dynamics",
        ("figures/appendix/A11_router_family.pdf", "figures/appendix/A11_router_family.png"),
    ),
    _spec(
        "A12",
        "table",
        "appendix",
        "Hyperparameters & configuration",
        "P0",
        "a12_hyperparameters",
        ("tables/A12_hyperparameters.tex", "tables/A12_hyperparameters.md"),
    ),
    _spec(
        "A13",
        "table",
        "appendix",
        "Dataset & phase-label statistics",
        "P1",
        "a13_dataset_stats",
        ("tables/A13_dataset_stats.tex", "tables/A13_dataset_stats.md"),
    ),
    _spec(
        "A14",
        "figure",
        "appendix",
        "Phase-depth analysis (max_phase)",
        "P1",
        "a14_phase_depth",
        ("figures/appendix/A14_phase_depth.pdf", "figures/appendix/A14_phase_depth.png"),
    ),
    _spec(
        "A15",
        "table",
        "appendix",
        "Paired statistical tests (Holm-adjusted)",
        "P0",
        "a15_paired_tests",
        ("tables/A15_paired_tests.tex", "tables/A15_paired_tests.md"),
    ),
    _spec(
        "A16",
        "figure",
        "appendix",
        "Normalized routing-entropy trajectories (record only)",
        "P1",
        "a16_routing_entropy",
        ("figures/appendix/A16_routing_entropy.pdf", "figures/appendix/A16_routing_entropy.png"),
    ),
):
    ASSET_REGISTRY[spec.id] = spec


def load_generator(spec: AssetSpec) -> Generator | None:
    """Import the asset module and return its ``generate`` callable."""
    if spec.kind == "schematic":
        return None
    module = importlib.import_module(f"studies.analysis.assets.{spec.module}")
    generator = getattr(module, spec.generator_attr, None)
    if not callable(generator):
        raise AttributeError(f"{spec.module} does not define {spec.generator_attr}()")
    return generator


def specs_by_section(section: str | None = None) -> list[AssetSpec]:
    specs = [ASSET_REGISTRY[k] for k in sorted(ASSET_REGISTRY, key=_asset_sort_key)]
    if section is not None:
        specs = [s for s in specs if s.section == section]
    return specs


def _asset_sort_key(asset_id: str) -> tuple[int, int, str]:
    prefix = asset_id[0]
    order = {"F": 0, "T": 1, "A": 2}.get(prefix, 3)
    try:
        number = int(asset_id[1:])
    except ValueError:
        number = 999
    return order, number, asset_id

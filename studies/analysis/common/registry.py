"""Method/task registry — derived from the frozen protocol manifests.

The runner's ``load_protocol`` is the single source of truth; this module adds
the paper-facing display layer (names, ordering) on top, so the analysis can
never disagree with what actually ran.
"""

from __future__ import annotations

from functools import cache

from phaseforge.runner.protocol import Method, Protocol, load_protocol
from studies.analysis.common.config import namespace_manifest

#: Paper-facing display names (figures_tables_plan.md style contract).
DISPLAY_NAMES: dict[str, str] = {
    "precision_residual_phaseforge": "PhaseForge",
    "phaseforge": "PhaseForge",
    "bc": "BC",
    "final_aligned_bc": "BC",
    "final_aligned_softmax_top1": "Softmax Top-1",
    "precision_residual_phase_random_router": "Phase-Random",
    "precision_residual_plain_encoder": "Plain Encoder",
    "precision_residual_scratch_moe": "Scratch MoE",
    "final_aligned_static_rule": "Static Rule",
    "precision_residual_factorial_floor": "Factorial Floor",
    "precision_residual_teacher_forced": "Teacher-Forced",
    "precision_residual_oracle": "Oracle (Offline)",
    # Router Initialization Ablation (Can & Square)
    "router_init_topology": "Trajectory-Derived Init (PF)",
    "router_init_phase": "Phase-Rule Init",
    "router_init_random": "Random Init",
    "representation_bc": "BC Latent",
    "routing_softmax_top1": "Softmax Top-1",
}

#: T1/T2 row order: proposed method first, then controls and diagnostics.
MATRIX_ORDER: tuple[str, ...] = (
    "precision_residual_phaseforge",
    "bc",
    "final_aligned_softmax_top1",
    "precision_residual_phase_random_router",
    "precision_residual_plain_encoder",
    "precision_residual_scratch_moe",
    "final_aligned_static_rule",
    "precision_residual_factorial_floor",
    "precision_residual_teacher_forced",
    "precision_residual_oracle",
)

TASK_ORDER: tuple[str, ...] = ("Lift", "Can", "Square", "ToolHang", "Transport")


@cache
def protocol(namespace: str) -> Protocol:
    return load_protocol(namespace_manifest(namespace))


def methods(namespace: str) -> tuple[Method, ...]:
    return protocol(namespace).methods


def seeds(namespace: str) -> tuple[int, ...]:
    return protocol(namespace).seeds


def matrix_method_names() -> tuple[str, ...]:
    """The 9 five-task method names in canonical paper order."""
    present = {m.name for m in methods("final")}
    return tuple(name for name in MATRIX_ORDER if name in present)


def tasks() -> tuple[str, ...]:
    present = {m.task for m in methods("final") if m.task is not None}
    return tuple(t for t in TASK_ORDER if t in present)


def ablation_method_names() -> tuple[str, ...]:
    """Ablation-only cells (not part of the nine-method matrix), manifest order."""
    matrix = set(matrix_method_names())
    seen: list[str] = []
    for m in methods("ablation"):
        if m.name not in matrix and m.name not in seen:
            seen.append(m.name)
    return tuple(seen)


def display_name(method_name: str) -> str:
    return DISPLAY_NAMES.get(method_name, method_name)


def experiment_id(namespace: str, method_name: str) -> str | None:
    for m in methods(namespace):
        if m.name == method_name:
            return m.experiment_id
    return None


def expected_cells(namespace: str) -> tuple[tuple[str | None, str], ...]:
    """Every (task, method-name) cell the namespace must contain."""
    return tuple({(m.task, m.name) for m in methods(namespace)})

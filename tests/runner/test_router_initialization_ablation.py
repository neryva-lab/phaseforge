"""Contract tests for the focused router-initialization ablation protocol."""

from __future__ import annotations

from pathlib import Path

from hydra import compose
from hydra.initialize import initialize_config_dir
from omegaconf import OmegaConf

from phaseforge.runner.protocol import build_plan, load_protocol
from phaseforge.runner.verify import verify_command_contract, verify_plan_ordering


REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "experiments" / "router_initialization_ablation.json"
CONFIG_DIR = (REPO / "phaseforge" / "config").resolve()
TASKS = {"Lift", "Can", "Square"}
EVALUATED = {
    "router_init_random",
    "router_init_phase",
    "router_init_topology",
    "representation_bc",
    "routing_softmax_top1",
}


def _protocol():
    return load_protocol(MANIFEST)


def test_protocol_shape_and_task_scope() -> None:
    protocol = _protocol()

    assert protocol.name == "router_initialization_ablation"
    assert protocol.seeds == (42, 43, 44)
    assert set(protocol.known_tasks) == TASKS
    assert len(protocol.methods) == 21

    for task in TASKS:
        rows = [method for method in protocol.methods if method.task == task]
        assert len(rows) == 7
        assert {method.name for method in rows} == {
            "precision_residual_phaseforge",
            "bc",
            *EVALUATED,
        }

    assert not any(method.task in {"ToolHang", "Transport"} for method in protocol.methods)


def test_primary_arms_share_every_training_contract_except_initialization() -> None:
    protocol = _protocol()

    for task in TASKS:
        rows = {
            method.name: method
            for method in protocol.methods
            if method.task == task
        }
        primary = [rows[name] for name in ("router_init_random", "router_init_phase", "router_init_topology")]

        assert all(method.stage2_source == "precision_residual_phaseforge_stage1" for method in primary)
        assert all(method.stages == (2,) and method.evaluate for method in primary)

        common = []
        for method in primary:
            common.append(
                tuple(
                    override
                    for override in method.overrides
                    if not override.startswith("models.router_init.")
                )
            )
        assert common[0] == common[1] == common[2]

        assert "models.router_init.type=random" in rows["router_init_random"].overrides
        assert "models.router_init.prototype_source=phase" in rows["router_init_phase"].overrides
        assert "models.router_init.prototype_source=topo" in rows["router_init_topology"].overrides
        assert all("train.margin.enabled=false" in method.overrides for method in primary)


def test_primary_model_configs_match_outside_router_initialization() -> None:
    protocol = _protocol()
    rows = {
        method.name: method
        for method in protocol.methods
        if method.task == "Lift"
        and method.name in {"router_init_random", "router_init_phase", "router_init_topology"}
    }
    composed: dict[str, tuple[dict, dict]] = {}

    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        for name, method in rows.items():
            train_overrides = [
                override for override in method.overrides if not override.startswith("eval.")
            ]
            cfg = compose(
                config_name="main",
                overrides=[
                    f"models={method.model}",
                    "data=lift",
                    "train=stage2",
                ]
                + train_overrides,
            )
            model = OmegaConf.to_container(cfg.models, resolve=False)
            assert isinstance(model, dict)
            model.pop("name", None)
            model.pop("router_init", None)
            train = OmegaConf.to_container(cfg.train, resolve=False)
            assert isinstance(train, dict)
            composed[name] = (model, train)

    assert composed["router_init_random"] == composed["router_init_phase"]
    assert composed["router_init_phase"] == composed["router_init_topology"]


def test_secondary_arms_use_topology_reference_contract() -> None:
    protocol = _protocol()

    for task in TASKS:
        rows = {
            method.name: method
            for method in protocol.methods
            if method.task == task
        }
        topology = rows["router_init_topology"]
        representation = rows["representation_bc"]
        softmax = rows["routing_softmax_top1"]

        assert topology.stage2_source == "precision_residual_phaseforge_stage1"
        assert representation.stage2_source == "final_aligned_bc_stage1"
        assert softmax.stage2_source == "precision_residual_phaseforge_stage1"

        for method in (topology, representation, softmax):
            assert "models.expert.beta=0.0" in method.overrides
            assert "train.phase_label_field=phase" in method.overrides
            assert "train.margin.enabled=false" in method.overrides
            assert "topo@_global_=topo_pelt_k6" in method.overrides


def test_all_rows_compose_and_plan_contract_passes() -> None:
    protocol = _protocol()
    composed = 0

    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        for method in protocol.methods:
            train_overrides = [
                override for override in method.overrides if not override.startswith("eval.")
            ]
            for stage in method.stages:
                compose(
                    config_name="main",
                    overrides=[
                        f"models={method.model}",
                        f"data={method.data}",
                        f"train=stage{stage}",
                    ]
                    + train_overrides,
                )
                composed += 1

            if method.evaluate:
                compose(
                    config_name="main",
                    overrides=[
                        f"models={method.model}",
                        f"data={method.data}",
                        "eval=rollout",
                        "eval.mode=rollout",
                    ]
                    + list(method.overrides),
                )
                composed += 1

    # Six provider rows compose once plus fifteen evaluated rows each compose
    # for Stage 2 and rollout evaluation: 6 + (15 * 2) = 36.
    assert composed == 36

    plan = build_plan(protocol, list(protocol.methods), with_dependencies=True)
    assert len(plan) == 108
    assert verify_plan_ordering(protocol, plan) == []
    assert verify_command_contract(protocol, plan, REPO / "outputs_router_initialization_ablation") == []

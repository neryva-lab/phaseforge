"""Final manifest gates: composition, plan expansion, and dry-run checks.

Covers ledger TEST-01 (every final model composes for every task),
MAN-04/05/06/12 (explicit labels, beta, topology pin, loss switches),
DRY-02 (plan expansion), DRY-03 (provider ordering), DRY-04 (command
contract), DRY-05 (historical isolation), and the DRY-06 pilot readiness
(selection builds without training).

All assertion targets are the real ``experiments/final_causal_matrix.json``
and the shared :mod:`phaseforge.runner.verify` gates, so these tests pin
the exact evidence the dry-run report prints.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from hydra import compose, initialize

from phaseforge.runner.protocol import build_plan, load_protocol
from phaseforge.runner.selection import SelectionSpec, resolve_selection
from phaseforge.runner.verify import (
    run_final_gates,
    verify_command_contract,
    verify_historical_isolation,
    verify_plan_ordering,
)

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "experiments" / "final_causal_matrix.json"

TASKS = ["Lift", "Can", "Square", "ToolHang", "Transport"]
TOPO_ROWS = {
    "precision_residual_plain_encoder",
    "precision_residual_phase_random_router",
    "precision_residual_scratch_moe",
    "precision_residual_factorial_floor",
    "final_aligned_softmax_top1",
    "precision_residual_teacher_forced",
    "precision_residual_oracle",
}


def _protocol():
    return load_protocol(MANIFEST)


def test_final_manifest_has_locked_identities_and_seeds() -> None:
    """MAN-01/02/03: exactly the ten locked identities x five tasks x seeds."""
    protocol = _protocol()
    assert protocol.seeds == (42, 43, 44)
    assert {m.task for m in protocol.methods} == set(TASKS)
    assert len(protocol.methods) == 50
    names = sorted({m.name for m in protocol.methods})
    assert names == sorted(
        TOPO_ROWS | {"bc", "final_aligned_static_rule", "precision_residual_phaseforge"}
    )


def test_final_manifest_declares_labels_beta_topo_explicitly() -> None:
    """MAN-04/05/06/12: no inherited ambiguity in the manifest text itself."""
    raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for row in raw["methods"]:
        ov = list(row["overrides"])
        name = row["name"]
        if name == "bc":
            assert "train.lambda_phase=0.0" in ov
            assert "train.supcon.enabled=false" in ov
            assert not any(o.startswith("models.expert.beta") for o in ov)
            assert not any(o.startswith("topo@") for o in ov)
            continue
        if name == "precision_residual_phaseforge":
            # Preserve the locked proposed-method contract from the confirmed
            # v8 run; omission would silently change its supervision/loss.
            assert "train.phase_label_field=phase" in ov
            assert "train.supcon.label_field=phase" in ov
            assert "train.margin.label_field=phase" in ov
            assert "train.margin.lambda_margin=0.05" in ov
            assert "train.lipschitz.enabled=false" in ov
            assert "train.gain_reg.enabled=false" in ov
            assert "eval.episodes.trace_level=full" in ov
        # Every residual row pins beta zero explicitly (MAN-05).
        assert "models.expert.beta=0.0" in ov, name
        if name == "precision_residual_phaseforge":
            pass
        elif name == "final_aligned_static_rule":
            assert "train.phase_label_field=phase" in ov
            assert "train.supcon.label_field=phase" in ov
            assert "train.margin.label_field=phase" in ov
            assert not any(o.startswith("topo@") for o in ov)
        else:
            assert "train.phase_label_field=phase_topo" in ov, name
            assert "train.margin.label_field=phase_topo" in ov, name
            assert "topo@_global_=topo_pelt_k6" in ov, name
        # Loss enablement is declared, never inherited (MAN-12).
        assert any(o.startswith("train.supcon.enabled=") for o in ov), name
        assert any(o.startswith("train.margin.enabled=") for o in ov), name
    # Evaluation modes are separated (MAN-08).
    modes = {(r["name"], r["evaluate_mode"]) for r in raw["methods"]}
    assert ("precision_residual_oracle", "offline") in modes
    assert all(m == "rollout" for n, m in modes if n != "precision_residual_oracle")


def test_final_rows_compose_for_every_task_stage_and_eval() -> None:
    """TEST-01: Hydra composition passes for all 50 rows (train + eval)."""
    protocol = _protocol()
    count = 0
    with initialize(version_base="1.3", config_path="../../phaseforge/config"):
        for method in protocol.methods:
            train_overrides = [o for o in method.overrides if not o.startswith("eval.")]
            stages = (
                ["train=stage1"]
                if method.stages == (1,)
                else (
                    ["train=stage2"]
                    if method.stages == (2,)
                    else ["train=stage1", "train=stage2"]
                )
            )
            for train in stages:
                cfg = compose(
                    config_name="main",
                    overrides=[f"models={method.model}", f"data={method.data}", train]
                    + train_overrides,
                )
                assert cfg.data.source.task_name == method.task
                count += 1
            group = "rollout" if method.evaluate_mode == "rollout" else "metrics"
            cfg = compose(
                config_name="main",
                overrides=[
                    f"models={method.model}",
                    f"data={method.data}",
                    f"eval={group}",
                    f"eval.mode={method.evaluate_mode}",
                ]
                + list(method.overrides),
            )
            assert cfg.data.source.task_name == method.task
            count += 1
    # 60 train compositions (5 bc x stage1, 35 stage-2-only, 10 x both) + 50 eval.
    assert count == 110


def test_final_plan_expands_to_330_steps() -> None:
    """DRY-02: the runner expands the complete matrix without training."""
    protocol = _protocol()
    plan = build_plan(protocol, list(protocol.methods), with_dependencies=True)
    assert len(plan) == 330
    kinds = Counter((s.kind, s.stage) for s in plan)
    assert kinds[("train", 1)] == 45
    assert kinds[("train", 2)] == 135
    assert kinds[("eval", None)] == 150


def test_final_plan_provider_ordering() -> None:
    """DRY-03: Stage 1 providers precede every Stage 2 consumer per task/seed."""
    protocol = _protocol()
    plan = build_plan(protocol, list(protocol.methods), with_dependencies=True)
    assert verify_plan_ordering(protocol, plan) == []


def test_final_command_contract() -> None:
    """DRY-04: every command carries its resolved contract tokens."""
    protocol = _protocol()
    plan = build_plan(protocol, list(protocol.methods), with_dependencies=True)
    assert verify_command_contract(protocol, plan, REPO / "outputs_final") == []


def test_final_historical_isolation() -> None:
    """DRY-05: no final row resolves to a historical config or alias."""
    protocol = _protocol()
    assert verify_historical_isolation(protocol) == []


def test_final_gates_report_passes() -> None:
    """Aggregated DRY-02..05 report over the real manifest (no writes)."""
    report = run_final_gates(MANIFEST, REPO / "outputs_final")
    assert report["methods"] == 50 and report["steps"] == 330
    for gate, result in report["gates"].items():
        assert result["status"] == "pass", (gate, result)


def test_dry_run_pilot_selection_builds_without_training() -> None:
    """DRY-06 readiness: an implementation-only pilot selection expands into
    real plan commands.

    The live pilot itself (a short Stage 1 run) executes on the training
    host; this pins that the pilot scope resolves to real commands.
    """
    protocol = _protocol()
    selection = resolve_selection(
        protocol, SelectionSpec(method_tokens=("bc@Lift",), tasks=())
    )
    assert [m.name for m in selection.methods] == ["bc"]
    plan = build_plan(protocol, list(selection.methods), seeds=[42], stage=1)
    assert len(plan) == 1
    step = plan[0]
    assert (step.kind, step.stage, step.seed) == ("train", 1, 42)


def test_eval_only_plan_carries_no_ordering_constraint() -> None:
    """Eval-only selections reference pre-existing artifacts (no trains)."""
    protocol = _protocol()
    selection = resolve_selection(
        protocol, SelectionSpec(method_tokens=("bc@Lift",), tasks=())
    )
    plan = build_plan(protocol, list(selection.methods), seeds=[42], eval_only=True)
    assert len(plan) == 1 and plan[0].kind == "eval"
    assert verify_plan_ordering(protocol, plan) == []


def test_final_row_without_provider_fails_isolation(tmp_path) -> None:
    """MAN-07: a final stage-2 row with no declared provider is flagged."""
    import json

    doc = {
        "name": "bad",
        "task": "all",
        "description": "",
        "seeds": [42],
        "defaults": [],
        "methods": [
            {
                "index": 1,
                "name": "precision_residual_scratch_moe",
                "role": "x",
                "model": "precision_residual_scratch_moe",
                "data": "lift",
                "task": "Lift",
                "stages": [2],
                "stage2_source": None,
                "evaluate": False,
            }
        ],
    }
    manifest = tmp_path / "bad.json"
    manifest.write_text(json.dumps(doc), encoding="utf-8")
    violations = verify_historical_isolation(load_protocol(manifest))
    assert len(violations) == 1 and "MAN-07" in violations[0]

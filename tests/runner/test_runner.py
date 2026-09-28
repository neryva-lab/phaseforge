"""CPU-only tests for the experiment runner (protocol, plan, registry, resolver, CLI)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from phaseforge.runner import cli as runner_cli
from phaseforge.runner import executor as runner_executor
from phaseforge.runner.commands import eval_command, train_command
from phaseforge.runner.executor import CommandError
from phaseforge.runner.protocol import (
    Protocol,
    ProtocolError,
    Step,
    build_plan,
    load_protocol,
)
from phaseforge.runner.registry import RegistryError, RunnerState
from phaseforge.runner.resolver import (
    CheckpointError,
    checkpoint_exists,
    resolve_checkpoint_path,
    resolve_run_dir,
    stage_checkpoint_relative,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = PROJECT_ROOT / "experiments" / "final_causal_matrix.json"


def _protocol() -> Protocol:
    return load_protocol(MANIFEST)


def _step(method_name: str, seed: int, phase: str) -> Step:
    protocol = _protocol()
    method = protocol.method_by_name(method_name)
    assert method is not None
    if phase == "eval":
        return Step(kind="eval", method=method, seed=seed)
    return Step(kind="train", method=method, seed=seed, stage=int(phase[-1]))


# ---------------------------------------------------------------------------
# Protocol loading / validation
# ---------------------------------------------------------------------------


def test_load_real_protocol_matches_notebook_methods() -> None:
    protocol = _protocol()
    assert protocol.name == "final_causal_matrix"
    assert protocol.seeds == (42, 43, 44)
    assert len(protocol.methods) == 50
    names = {m.name for m in protocol.methods}
    assert names == {
        "precision_residual_phaseforge",
        "bc",
        "precision_residual_plain_encoder",
        "precision_residual_phase_random_router",
        "precision_residual_scratch_moe",
        "precision_residual_factorial_floor",
        "final_aligned_softmax_top1",
        "final_aligned_static_rule",
        "precision_residual_teacher_forced",
        "precision_residual_oracle",
    }
    phaseforge = protocol.method_by_name("precision_residual_phaseforge", task="Lift")
    assert phaseforge is not None
    assert phaseforge.stages == (1, 2)
    assert phaseforge.stage2_source == "self"
    assert phaseforge.evaluate
    assert phaseforge.model_name == "precision_residual_phaseforge"

    bc = protocol.method_by_name("bc", task="Lift")
    assert bc is not None
    assert bc.stages == (1,)
    assert bc.stage2_source is None
    assert bc.model_name == "final_aligned_bc"

    warmstart = protocol.method_by_name("precision_residual_scratch_moe", task="Lift")
    assert warmstart is not None
    assert warmstart.stage2_source == "precision_residual_phaseforge_stage1"

    teacher = protocol.method_by_name("precision_residual_teacher_forced", task="Lift")
    assert teacher is not None
    assert teacher.stage2_source == "precision_residual_phaseforge_stage1"


def _write_protocol(tmp_path: Path, doc: dict) -> Path:
    p = tmp_path / "protocol.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def _valid_doc() -> dict:
    return {
        "name": "mini",
        "task": "Mini",
        "seeds": [42],
        "defaults": ["train.early_stopping.enabled=false"],
        "methods": [
            {
                "index": 1,
                "name": "precision_residual_phaseforge",
                "role": "proposed",
                "model": "precision_residual_phaseforge",
                "data": "common",
                "stages": [1, 2],
                "stage2_source": "self",
                "evaluate": True,
            },
            {
                "index": 2,
                "name": "bc",
                "role": "floor",
                "model": "final_aligned_bc",
                "data": "common",
                "stages": [1],
                "evaluate": True,
            },
        ],
    }


def test_load_protocol_valid(tmp_path: Path) -> None:
    protocol = load_protocol(_write_protocol(tmp_path, _valid_doc()))
    assert len(protocol.methods) == 2
    assert protocol.methods[0].index == 1
    assert protocol.defaults == ("train.early_stopping.enabled=false",)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda d: d["methods"].append(dict(d["methods"][0], index=1)), "Duplicate method index"),
        (lambda d: d.update(seeds=[]), "seeds"),
        (lambda d: d.update(seeds=["x"]), "seed"),
        (lambda d: d["methods"][0].update(stages=[3]), "stage"),
        (lambda d: d["methods"][0].update(stages=[2, 1]), "ascending"),
        (lambda d: d["methods"][0].pop("model"), "model"),
        (lambda d: d["methods"][0].update(index=0), "index"),
        (lambda d: d["methods"][0].update(stage2_source="nope"), "stage2_source"),
        (
            lambda d: d["methods"][0].update(
                stage2_source="precision_residual_phaseforge", stages=[1]
            ),
            "stage 2",
        ),
    ],
)
def test_load_protocol_validation_errors(tmp_path: Path, mutate, match) -> None:
    doc = _valid_doc()
    mutate(doc)
    with pytest.raises(ProtocolError, match=match):
        load_protocol(_write_protocol(tmp_path, doc))


def test_load_protocol_rejects_unknown_provider(tmp_path: Path) -> None:
    doc = _valid_doc()
    # bc gains a stage 2 sourced from a provider that no longer exists.
    doc["methods"][1].update(
        stages=[1, 2], stage2_source="precision_residual_phaseforge"
    )
    doc["methods"] = [doc["methods"][1]]  # drop the phaseforge method entirely
    with pytest.raises(ProtocolError, match="not a method"):
        load_protocol(_write_protocol(tmp_path, doc))


def test_select_methods_by_index_and_name() -> None:
    protocol = _protocol()
    selected = protocol.select_methods(
        [
            "precision_residual_scratch_moe@Lift",
            "precision_residual_phaseforge@Lift",
        ]
    )
    assert [m.name for m in selected] == [
        "precision_residual_phaseforge",
        "precision_residual_scratch_moe",
    ]
    with pytest.raises(ProtocolError, match="Unknown method"):
        protocol.select_methods(["does_not_exist"])
    with pytest.raises(ProtocolError, match="Unknown method index"):
        protocol.select_methods(["99"])


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------


def test_build_plan_full_matrix_order_and_count() -> None:
    protocol = _protocol()
    plan = build_plan(
        protocol,
        [m for m in protocol.methods if m.task == "Lift"],
        seeds=[42],
    )
    # Ten final identities: two two-stage methods contribute three steps;
    # Seven stage-2-only methods contribute two each; BC contributes two.
    assert len(plan) == 22
    assert any(
        step.label == "precision_residual_phaseforge seed=42 stage2"
        for step in plan
    )
    eval_step = next(
        step
        for step in plan
        if step.kind == "eval" and step.method.name == "precision_residual_phaseforge"
    )
    assert eval_step.required_checkpoint() == ("precision_residual_phaseforge", 2)


def test_build_plan_multi_seed() -> None:
    protocol = _protocol()
    plan = build_plan(protocol, [protocol.method_by_name("bc")], seeds=[42, 44])
    assert [s.seed for s in plan] == [42, 42, 44, 44]


def test_build_plan_stage_filter() -> None:
    protocol = _protocol()
    plan = build_plan(
        protocol,
        [protocol.method_by_name("precision_residual_phaseforge", task="Lift")],
        seeds=[42],
        stage=1,
    )
    assert len(plan) == 1
    assert plan[0].stage == 1
    assert plan[0].kind == "train"


def test_build_plan_eval_only() -> None:
    protocol = _protocol()
    plan = build_plan(
        protocol,
        [protocol.method_by_name("precision_residual_phaseforge", task="Lift")],
        seeds=[42],
        eval_only=True,
    )
    assert len(plan) == 1
    assert plan[0].kind == "eval"


def test_build_plan_skip_eval() -> None:
    protocol = _protocol()
    plan = build_plan(
        protocol,
        [protocol.method_by_name("precision_residual_phaseforge", task="Lift")],
        seeds=[42],
        skip_eval=True,
    )
    assert [s.kind for s in plan] == ["train", "train"]


def test_build_plan_conflicting_filters() -> None:
    protocol = _protocol()
    with pytest.raises(ProtocolError, match="mutually exclusive"):
        build_plan(
            protocol,
            [protocol.method_by_name("bc")],
            seeds=[42],
            stage=1,
            eval_only=True,
        )
    with pytest.raises(ProtocolError, match="mutually exclusive"):
        build_plan(
            protocol,
            [protocol.method_by_name("bc")],
            seeds=[42],
            eval_only=True,
            skip_eval=True,
        )


def test_build_plan_injects_dependency() -> None:
    protocol = _protocol()
    teacher = protocol.method_by_name("precision_residual_teacher_forced", task="Lift")
    plan = build_plan(protocol, [teacher], seeds=[42], with_dependencies=True)
    assert len(plan) == 3
    dep = plan[0]
    assert dep.dependency
    assert dep.method.name == "precision_residual_phaseforge"
    assert dep.stage == 1
    assert dep.registry_phase == "stage1"
    assert plan[1].label == "precision_residual_teacher_forced seed=42 stage2"
    assert plan[2].kind == "eval"


def test_build_plan_no_injection_when_provider_selected() -> None:
    protocol = _protocol()
    selected = [
        protocol.method_by_name("precision_residual_phaseforge", task="Lift"),
        protocol.method_by_name("precision_residual_teacher_forced", task="Lift"),
    ]
    plan = build_plan(protocol, selected, seeds=[42], with_dependencies=True)
    assert not any(s.dependency for s in plan)


def test_build_plan_rejects_unknown_seed() -> None:
    protocol = _protocol()
    with pytest.raises(ProtocolError, match="Seed 99"):
        build_plan(protocol, [protocol.method_by_name("bc")], seeds=[99])


# ---------------------------------------------------------------------------
# Command building
# ---------------------------------------------------------------------------


def test_train_command_common_cell(tmp_path: Path) -> None:
    cmd = train_command(
        _step("bc", 42, "stage1"),
        outputs_base=tmp_path / "outputs",
        defaults=("train.early_stopping.enabled=false",),
    )
    assert cmd == [
        "phaseforge-train",
        "models=final_aligned_bc",
        "train=stage1",
        "project.seed=42",
        f"project.output_dir={tmp_path / 'outputs'}",
        "project.method=bc",
        "train.early_stopping.enabled=false",
    ]


def test_train_command_stage2(tmp_path: Path) -> None:
    cmd = train_command(
        _step("precision_residual_phaseforge", 42, "stage2"),
        outputs_base=tmp_path / "outputs",
        defaults=(),
    )
    assert "models=precision_residual_phaseforge" in cmd
    assert "train=stage2" in cmd


def test_train_command_stage2_injects_provider_ckpt(tmp_path: Path) -> None:
    ckpt = tmp_path / "outputs" / "bc" / "stage1" / "x" / "checkpoint_best.pt"
    cmd = train_command(
        _step("precision_residual_scratch_moe", 42, "stage2"),
        outputs_base=tmp_path / "outputs",
        defaults=(),
        ckpt_path=ckpt,
    )
    assert f"train.stage1_ckpt_path={ckpt}" in cmd


def test_eval_command_targets_final_checkpoint(tmp_path: Path) -> None:
    ckpt = tmp_path / "outputs" / "precision_residual_phaseforge" / "stage2" / "ckpt" / "checkpoint_best.pt"
    cmd = eval_command(
        _step("precision_residual_phaseforge", 42, "eval"),
        ckpt_path=ckpt,
        outputs_base=tmp_path / "outputs",
        defaults=(),
    )
    assert cmd[0] == "phaseforge-eval"
    assert f"train.stage1_ckpt_path={ckpt}" in cmd
    assert "project.seed=42" in cmd
    # Regression: the eval step must select the rollout config group, not
    # only override eval.mode. Without this, the subprocess composes the
    # default (metrics) eval group and crashes on the missing 'bank' key
    # inside load_or_generate_bank (Hydra ConfigAttributeError).
    assert "eval=rollout" in cmd
    assert "eval.mode=rollout" in cmd


def _offline_eval_step(seed: int) -> Step:
    """Build an eval step whose method evaluates with the offline metric."""
    from phaseforge.runner.protocol import Method

    method = Method(
        index=1,
        name="bc",
        role="baseline",
        model="final_aligned_bc",
        data="common",
        stages=(1,),
        stage2_source=None,
        evaluate=True,
        evaluate_mode="offline",
    )
    return Step(kind="eval", method=method, seed=seed)


def test_eval_command_offline_selects_metrics_group(tmp_path: Path) -> None:
    ckpt = tmp_path / "outputs" / "bc" / "stage1" / "ckpt" / "checkpoint_best.pt"
    cmd = eval_command(
        _offline_eval_step(seed=42),
        ckpt_path=ckpt,
        outputs_base=tmp_path / "outputs",
        defaults=(),
    )
    # Offline mode -> metrics group, which has the offline schema but no
    # bank/env/episodes (load_or_generate_bank / gates are never called).
    assert "eval=metrics" in cmd
    assert "eval.mode=offline" in cmd
    assert "eval=rollout" not in cmd


def test_eval_command_unknown_mode_raises_keyerror(tmp_path: Path) -> None:
    """Defense: the mode->group map stays total so future mode additions
    fail loudly at command build rather than silently at runtime."""
    from phaseforge.runner.protocol import Method

    method = Method(
        index=1,
        name="bc",
        role="baseline",
        model="final_aligned_bc",
        data="common",
        stages=(1,),
        stage2_source=None,
        evaluate=True,
        evaluate_mode="bogus_mode",
    )
    step = Step(kind="eval", method=method, seed=42)
    with pytest.raises(KeyError):
        eval_command(
            step,
            ckpt_path=tmp_path / "ckpt.pt",
            outputs_base=tmp_path / "outputs",
            defaults=(),
        )


# ---------------------------------------------------------------------------
# Hydra composition contract for the eval config group
# ---------------------------------------------------------------------------


def test_eval_mode_override_alone_leaves_metrics_schema() -> None:
    """Regression: the buggy runner argv shape (``eval.mode=rollout``
    without the ``eval=rollout`` group selector) composes the default
    ``metrics`` group, which has no ``bank``/``env``/``episodes``
    sections. This test pins the composition contract that motivates
    Layer 1: the runner MUST emit ``eval=<group>``."""
    from hydra import compose, initialize

    with initialize(version_base="1.3", config_path="../../phaseforge/config"):
        cfg = compose(config_name="main", overrides=["eval.mode=rollout"])
    # The default eval group is metrics (offline), so these rollout-only
    # sections are absent. This is exactly the state the buggy runner
    # used to land the subprocess in.
    assert "bank" not in cfg.eval
    assert "episodes" not in cfg.eval
    # cfg.eval.env is also absent on the metrics schema.
    assert "env" not in cfg.eval


def test_eval_group_selector_loads_rollout_schema() -> None:
    """Regression: the fixed runner argv (``eval=rollout``) composes the
    rollout config group, which carries the full schema
    ``load_or_generate_bank`` requires."""
    from hydra import compose, initialize

    with initialize(version_base="1.3", config_path="../../phaseforge/config"):
        cfg = compose(config_name="main", overrides=["eval=rollout"])
    assert "bank" in cfg.eval
    assert "env" in cfg.eval
    assert "episodes" in cfg.eval
    assert cfg.eval.mode == "rollout"


def test_toolhang_python_auto_detects_dedicated_environment(tmp_path: Path) -> None:
    interpreter = tmp_path / ".venv-toolhang" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("", encoding="utf-8")
    assert runner_executor.resolve_toolhang_python(tmp_path) == interpreter.absolute()


def test_toolhang_python_missing_environment_fails_before_sweep(tmp_path: Path) -> None:
    with pytest.raises(CommandError, match="dedicated robosuite 1.5.0"):
        runner_executor.resolve_toolhang_python(tmp_path)


def test_toolhang_preflight_rejects_wrong_robosuite(monkeypatch, tmp_path: Path) -> None:
    class _Result:
        returncode = 0
        stdout = "1.5.1\n3.2.7\n"
        stderr = ""

    monkeypatch.setattr(runner_executor.subprocess, "run", lambda *args, **kwargs: _Result())
    with pytest.raises(CommandError, match="robosuite 1.5.1"):
        runner_executor.preflight_toolhang_python(tmp_path / "python")


# ---------------------------------------------------------------------------
# Runner state registry
# ---------------------------------------------------------------------------


def test_registry_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "_runner" / "state.json"
    state = RunnerState(path)
    state.mark("precision_residual_phaseforge", 42, "stage1", run_dir="precision_residual_phaseforge/stage1/x", ckpt="p/s1/c.pt")
    state.mark("precision_residual_phaseforge", 42, "stage2", run_dir="precision_residual_phaseforge/stage2/y", ckpt="p/s2/c.pt")
    state.mark("precision_residual_phaseforge", 42, "eval", ckpt="p/s2/c.pt", run_dir="eval/precision_residual_phaseforge/z")

    reloaded = RunnerState(path)
    assert reloaded.is_complete("precision_residual_phaseforge", 42, "stage1")
    assert reloaded.is_complete("precision_residual_phaseforge", 42, "eval")
    assert not reloaded.is_complete("precision_residual_phaseforge", 43, "stage1")
    assert reloaded.get_ckpt("precision_residual_phaseforge", 42, 2) == "p/s2/c.pt"
    assert reloaded.get_ckpt("precision_residual_phaseforge", 43, 1) is None


def test_registry_mark_failed(tmp_path: Path) -> None:
    state = RunnerState(tmp_path / "state.json")
    state.mark_failed("bc", 42, "stage1", "boom")
    assert not state.is_complete("bc", 42, "stage1")
    assert state.get("bc", 42, "stage1")["status"] == "failed"


def test_registry_corrupt_state_is_loud(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(RegistryError, match="corrupt"):
        RunnerState(path)


def test_registry_commit_gate_skips_stale_entries(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    state_old = RunnerState(path, expected_commit="aaaa1111")
    state_old.mark("precision_residual_phaseforge", 42, "stage1", run_dir="precision_residual_phaseforge/stage1/x", ckpt="p/s1/c.pt")

    # A later revision must not reuse the pre-fix artifact.
    state_new = RunnerState(path, expected_commit="bbbb2222")
    assert not state_new.is_complete("precision_residual_phaseforge", 42, "stage1")
    assert state_new.get_ckpt("precision_residual_phaseforge", 42, 1) is None

    # The revision that produced it still sees it.
    state_same = RunnerState(path, expected_commit="aaaa1111")
    assert state_same.is_complete("precision_residual_phaseforge", 42, "stage1")
    assert state_same.get_ckpt("precision_residual_phaseforge", 42, 1) == "p/s1/c.pt"


def test_registry_commit_gate_stale_without_commit_field(tmp_path: Path) -> None:
    # Entries recorded before commit tracking existed (empty git_commit) must
    # be stale under gating.
    path = tmp_path / "state.json"
    state_ungated = RunnerState(path)  # no expected_commit
    state_ungated.mark("bc", 42, "stage1", run_dir="bc/stage1/x", ckpt="p/bc/c.pt")

    state_gated = RunnerState(path, expected_commit="cccc3333")
    assert not state_gated.is_complete("bc", 42, "stage1")
    assert state_gated.get_ckpt("bc", 42, 1) is None


def test_registry_commit_gate_disabled_when_git_unavailable(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    state_gated = RunnerState(path, expected_commit="dddd4444")
    state_gated.mark("bc", 42, "stage1", run_dir="bc/stage1/x", ckpt="p/bc/c.pt")

    # No gating (git unavailable locally) → entries are current regardless.
    state_ungated = RunnerState(path)
    assert state_ungated.is_complete("bc", 42, "stage1")
    assert state_ungated.get_ckpt("bc", 42, 1) == "p/bc/c.pt"


def test_runner_no_commit_gate_flag() -> None:
    from phaseforge.runner.cli import parse_args

    args_default = parse_args([])
    assert not args_default.no_commit_gate

    args_flag = parse_args(["--no-commit-gate"])
    assert args_flag.no_commit_gate is True



# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def _make_run(
    base: Path,
    model: str,
    stage: int,
    name: str,
    seed: int,
    tag=None,
    completed: bool = True,
    seed_dir: bool = True,
    commit: str | None = None,
    num_experts: int | None = None,
    config_hash: str | None = None,
) -> None:
    run_dir = base / model / f"stage{stage}"
    if seed_dir:
        run_dir = run_dir / f"seed{seed}"
    run_dir = run_dir / name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Real (loadable) checkpoint payloads: the runner's contract verifier
    # torch.loads every resolved artifact, so dummy text files would fail
    # closed. Dense key set by default; MoE expert keys when requested.
    state: dict[str, object] = {
        "encoder.hidden.0.weight": torch.zeros(4, 4),
        "action_head.trunk.0.weight": torch.zeros(4, 4),
    }
    if num_experts is not None:
        for i in range(num_experts):
            state[f"moe_layer.experts.{i}.hidden.0.weight"] = torch.zeros(4, 4)
    torch.save({"model_state_dict": state, "stage": stage}, ckpt_dir / "checkpoint_best.pt")
    meta: dict[str, object] = {
        "kind": "train",
        "model_name": model,
        "stage": stage,
        "seed": seed,
        "git_commit": commit or "deadbeef",
        "tag": tag,
        "method": None,
    }
    if config_hash is not None:
        meta["config_hash"] = config_hash
    (run_dir / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    if completed:
        # Real runs get the sibling lifecycle marker written by RunWriter
        # only when the run finishes successfully.
        (run_dir.with_name(name + ".completed")).write_text("{}", encoding="utf-8")


def test_resolve_run_dir_seed_and_tag(tmp_path: Path) -> None:
    _make_run(tmp_path, "bc", 1, "2026-08-01_10-00-00_aaaa0001", seed=42)
    _make_run(tmp_path, "bc", 1, "2026-08-02_10-00-00_aaaa0002", seed=42, tag="robot_only")
    _make_run(tmp_path, "bc", 1, "2026-08-03_10-00-00_aaaa0003", seed=43)

    assert resolve_run_dir(tmp_path, "bc", 1, seed=42).name == "2026-08-01_10-00-00_aaaa0001"
    assert (
        resolve_run_dir(tmp_path, "bc", 1, seed=42, tag="robot_only").name
        == "2026-08-02_10-00-00_aaaa0002"
    )
    with pytest.raises(CheckpointError, match="seed 44"):
        resolve_run_dir(tmp_path, "bc", 1, seed=44)


def test_resolve_run_dir_ignores_crashed_runs(tmp_path: Path) -> None:
    # A run killed after its last checkpoint save but before mark_completed
    # has run_meta + checkpoint_best.pt but no .completed marker. It must
    # never be selected as an eval target.
    _make_run(tmp_path, "bc", 1, "2026-08-01_10-00-00_aaaa0001", seed=42, completed=False)
    _make_run(tmp_path, "bc", 1, "2026-08-03_10-00-00_aaaa0003", seed=42)

    assert resolve_run_dir(tmp_path, "bc", 1, seed=42).name == "2026-08-03_10-00-00_aaaa0003"
    with pytest.raises(CheckpointError, match="completed"):
        resolve_run_dir(tmp_path, "bc", 1, seed=43)


def test_checkpoint_exists_false_for_crashed_run(tmp_path: Path) -> None:
    _make_run(tmp_path, "bc", 1, "2026-08-01_10-00-00_aaaa0001", seed=42, completed=False)
    assert not checkpoint_exists(tmp_path, "bc", 1, seed=42)


def test_resolve_run_dir_filters_by_commit(tmp_path: Path) -> None:
    # A stale pre-fix run (older git commit) must never be selected when the
    # sweep runs at a newer revision, even if it is the newest completed run.
    _make_run(
        tmp_path,
        "precision_residual_phaseforge",
        1,
        "2026-08-03_10-00-00_stale0001",
        seed=42,
        commit="aaaa1111",
    )
    _make_run(
        tmp_path,
        "precision_residual_phaseforge",
        1,
        "2026-08-02_10-00-00_fixed0001",
        seed=42,
        commit="bbbb2222",
    )

    # The stale run is newest but must be skipped at the fixed revision.
    assert (
        resolve_run_dir(tmp_path, "precision_residual_phaseforge", 1, seed=42, expected_commit="bbbb2222").name
        == "2026-08-02_10-00-00_fixed0001"
    )
    # Only stale runs at the expected revision → nothing matches.
    with pytest.raises(CheckpointError, match="No completed"):
        resolve_run_dir(tmp_path, "precision_residual_phaseforge", 1, seed=42, expected_commit="cccc3333")
    # No gating → newest completed wins (backwards compatible).
    assert (
        resolve_run_dir(tmp_path, "precision_residual_phaseforge", 1, seed=42).name
        == "2026-08-03_10-00-00_stale0001"
    )


def test_resolve_run_dir_missing_stage(tmp_path: Path) -> None:
    with pytest.raises(CheckpointError, match="stage1"):
        resolve_run_dir(tmp_path, "precision_residual_phaseforge", 1, seed=42)


def test_resolve_run_dir_legacy_layout(tmp_path: Path) -> None:
    # Runs written before seeds became a directory dimension sit directly
    # under stage{N}/; the resolver must still find and seed-filter them.
    _make_run(tmp_path, "bc", 1, "2026-08-01_10-00-00_aaaa0001", seed=42, seed_dir=False)
    _make_run(tmp_path, "bc", 1, "2026-08-03_10-00-00_aaaa0003", seed=43, seed_dir=False)

    assert resolve_run_dir(tmp_path, "bc", 1, seed=42).name == "2026-08-01_10-00-00_aaaa0001"
    assert resolve_run_dir(tmp_path, "bc", 1, seed=43).name == "2026-08-03_10-00-00_aaaa0003"
    with pytest.raises(CheckpointError, match="seed 44"):
        resolve_run_dir(tmp_path, "bc", 1, seed=44)


def test_stage_checkpoint_relative_requires_best_ckpt(tmp_path: Path) -> None:
    run_dir = tmp_path / "bc" / "stage1" / "2026-08-01_10-00-00_aaaa0001"
    run_dir.mkdir(parents=True)
    with pytest.raises(CheckpointError, match="no .*checkpoint_best"):
        stage_checkpoint_relative(tmp_path, run_dir, "bc", 1)


def test_resolve_checkpoint_path_scan_fallback(tmp_path: Path) -> None:
    _make_run(tmp_path, "precision_residual_phaseforge", 2, "2026-08-01_10-00-00_aaaa0001", seed=42)
    state = RunnerState(tmp_path / "state.json")
    method = _protocol().method_by_name("precision_residual_phaseforge", task="Lift")
    assert method is not None
    ckpt = resolve_checkpoint_path(tmp_path, method, 2, seed=42, state=state)
    assert ckpt.is_file()
    assert "aaaa0001" in str(ckpt)

    with pytest.raises(CheckpointError, match="stage2"):
        resolve_checkpoint_path(tmp_path, method, 2, seed=44, state=state)


def test_resolve_checkpoint_path_prefers_registry(tmp_path: Path) -> None:
    _make_run(tmp_path, "precision_residual_phaseforge", 2, "2026-08-01_10-00-00_aaaa0001", seed=42)
    state = RunnerState(tmp_path / "state.json")
    state.mark(
        "precision_residual_phaseforge",
        42,
        "stage2",
        run_dir="x",
        ckpt="precision_residual_phaseforge/stage2/old/checkpoint_best.pt",
    )
    (tmp_path / "precision_residual_phaseforge" / "stage2" / "old").mkdir(parents=True, exist_ok=True)
    (tmp_path / "precision_residual_phaseforge" / "stage2" / "old" / "checkpoint_best.pt").write_text("newer")
    method = _protocol().method_by_name("precision_residual_phaseforge", task="Lift")
    assert method is not None
    ckpt = resolve_checkpoint_path(tmp_path, method, 2, seed=42, state=state)
    assert "old" in str(ckpt)


def test_stage2_resolves_exact_untagged_provider_ckpt(tmp_path: Path) -> None:
    """A stage-2 step must load the *untagged* provider checkpoint.

    The default BC run and its ``robot_only`` sibling share the
    ``bc/stage1/`` output tree; ``find_latest_checkpoint`` auto-detect would
    pick the newer tagged run and crash the stage-2 load with a dimension
    mismatch. The runner must resolve the exact untagged artifact instead.
    """
    _make_run(tmp_path, "bc", 1, "2026-08-01_10-00-00_aaaa0001", seed=42)
    _make_run(tmp_path, "bc", 1, "2026-08-02_10-00-00_aaaa0002", seed=42, tag="robot_only")
    step = _step("precision_residual_scratch_moe", 42, "stage2")
    ckpt = runner_cli._require_stage2_prereq(step, tmp_path)
    assert ckpt is not None
    assert "aaaa0001" in str(ckpt)
    assert "robot_only" not in str(ckpt)


def test_stage2_prereq_none_for_plain_stage1(tmp_path: Path) -> None:
    step = _step("bc", 42, "stage1")
    assert runner_cli._require_stage2_prereq(step, tmp_path) is None


def test_auto_dependency_provider_eligibility() -> None:
    protocol = _protocol()
    args_plain = runner_cli.parse_args(["--methods", "bc", "--seeds", "42"])
    args_stage = runner_cli.parse_args(["--methods", "bc", "--seeds", "42", "--stage", "2"])
    args_eval = runner_cli.parse_args(["--methods", "bc", "--seeds", "42", "--eval-only"])

    # A stage-2 consumer with an external provider is eligible.
    warmstart = _step("precision_residual_scratch_moe", 42, "stage2")
    provider = runner_cli._auto_dependency_provider(warmstart, protocol, args_plain)
    assert provider is not None and provider.name == "bc"

    # The privileged diagnostic needs the proposed Stage 1 cell.
    teacher = _step("precision_residual_teacher_forced", 42, "stage2")
    assert (
        runner_cli._auto_dependency_provider(teacher, protocol, args_plain).name
        == "precision_residual_phaseforge"
    )

    # Non-consumers are never eligible.
    assert (
        runner_cli._auto_dependency_provider(_step("bc", 42, "stage1"), protocol, args_plain)
        is None
    )
    assert (
        runner_cli._auto_dependency_provider(
            _step("precision_residual_scratch_moe", 42, "eval"), protocol, args_plain
        )
        is None
    )

    # Explicit scoping disables injection (a narrowed sweep fails pre-flight).
    assert runner_cli._auto_dependency_provider(warmstart, protocol, args_stage) is None
    assert runner_cli._auto_dependency_provider(warmstart, protocol, args_eval) is None


def test_resolve_stage2_auto_injects_provider(tmp_path: Path, monkeypatch) -> None:
    """A missing Stage 1 provider is auto-trained before the consumer runs."""
    protocol = _protocol()
    step = _step("precision_residual_scratch_moe", 42, "stage2")
    outputs = tmp_path / "outputs"
    state = RunnerState(runner_cli.RunnerState.default_path(outputs))
    args = runner_cli.parse_args(
        ["--methods", "precision_residual_scratch_moe", "--seeds", "42", "--outputs", str(outputs)]
    )
    ran: list[str] = []

    def _fake_run_step(
        step_, *, ckpt_path, outputs_base, defaults, cwd, log_level="WARNING", toolhang_python=None
    ):
        ran.append(step_.label)
        assert step_.kind == "train" and step_.stage == 1 and step_.dependency
        assert step_.method.name == "bc"
        _make_run(outputs_base, "bc", 1, "2026-08-17_10-00-00_dep0001", seed=42)

    monkeypatch.setattr(runner_cli, "run_step", _fake_run_step)

    ckpt = runner_cli._resolve_stage2_with_auto_dependency(
        step, protocol, outputs, state, args, None
    )

    assert ran == ["bc seed=42 stage1"]
    assert ckpt.is_file() and "bc" in str(ckpt)
    entry = state.get("bc", 42, "stage1")
    assert entry is not None and entry["status"] == "completed"
    assert (outputs / "bc" / "stage1" / "seed42").is_dir()


def test_resolve_stage2_scoped_run_still_fails_preflight(tmp_path: Path, monkeypatch) -> None:
    """--stage/--eval-only keep the strict pre-flight check (no auto-training)."""
    protocol = _protocol()
    step = _step("precision_residual_scratch_moe", 42, "stage2")
    outputs = tmp_path / "outputs"
    state = RunnerState(runner_cli.RunnerState.default_path(outputs))
    args = runner_cli.parse_args(
        ["--methods", "precision_residual_scratch_moe", "--seeds", "42", "--stage", "2", "--outputs", str(outputs)]
    )
    called = False

    def _fake_run_step(*args, **kwargs) -> None:  # pragma: no cover - must not run
        nonlocal called
        called = True

    monkeypatch.setattr(runner_cli, "run_step", _fake_run_step)
    with pytest.raises(CheckpointError, match="needs a bc stage 1 checkpoint"):
        runner_cli._resolve_stage2_with_auto_dependency(step, protocol, outputs, state, args, None)
    assert not called


def test_cli_auto_injects_missing_dependency(tmp_path: Path, monkeypatch, capsys) -> None:
    """A partial sweep auto-trains the missing provider and completes."""
    protocol_path = _write_protocol(
        tmp_path,
        {
            "name": "mini",
            "task": "Mini",
            "seeds": [42],
            "defaults": ["train.early_stopping.enabled=false"],
            "methods": [
                {
                    "index": 1,
                    "name": "precision_residual_phaseforge",
                    "role": "proposed",
                    "model": "precision_residual_phaseforge",
                    "data": "common",
                    "stages": [1, 2],
                    "stage2_source": "self",
                    "evaluate": True,
                },
                {
                    "index": 2,
                    "name": "bc",
                    "role": "floor",
                    "model": "final_aligned_bc",
                    "data": "common",
                    "stages": [1],
                    "evaluate": True,
                },
                {
                    "index": 5,
                    "name": "precision_residual_scratch_moe",
                    "role": "moe",
                    "model": "precision_residual_scratch_moe",
                    "data": "common",
                    "stages": [2],
                    "stage2_source": "bc",
                    "evaluate": True,
                },
            ],
        },
    )
    outputs = tmp_path / "outputs"
    calls: list[tuple[str, int, bool]] = []

    def _fake(
        step, *, ckpt_path, outputs_base, defaults, cwd, log_level="WARNING", toolhang_python=None
    ):
        calls.append((step.method.name, step.stage, step.dependency))
        _make_run(
            outputs_base,
            step.method.model_name,
            step.stage,
            f"2026-08-17_10-00-00_{len(calls):04d}",
            seed=step.seed,
            tag=step.method.output_tag,
            commit=runner_cli._current_commit(),
        )

    monkeypatch.setattr(runner_cli, "run_step", _fake)
    args = runner_cli.parse_args(
        [
            "--manifest",
            str(protocol_path),
            "--outputs",
            str(outputs),
            "--methods",
            "precision_residual_scratch_moe",
            "--seeds",
            "42",
            "--skip-eval",
        ]
    )
    assert runner_cli.run(args) == 0
    assert calls == [("bc", 1, True), ("precision_residual_scratch_moe", 2, False)]


def test_cli_dry_run_previews_auto_injection(tmp_path: Path, capsys) -> None:
    """Dry-run shows the provider it would auto-train instead of failing."""
    protocol_path = _write_protocol(
        tmp_path,
        {
            "name": "mini",
            "task": "Mini",
            "seeds": [42],
            "defaults": ["train.early_stopping.enabled=false"],
            "methods": [
                {
                    "index": 1,
                    "name": "precision_residual_phaseforge",
                    "role": "proposed",
                    "model": "precision_residual_phaseforge",
                    "data": "common",
                    "stages": [1, 2],
                    "stage2_source": "self",
                    "evaluate": True,
                },
                {
                    "index": 2,
                    "name": "bc",
                    "role": "floor",
                    "model": "final_aligned_bc",
                    "data": "common",
                    "stages": [1],
                    "evaluate": True,
                },
                {
                    "index": 5,
                    "name": "precision_residual_scratch_moe",
                    "role": "moe",
                    "model": "precision_residual_scratch_moe",
                    "data": "common",
                    "stages": [2],
                    "stage2_source": "bc",
                    "evaluate": True,
                },
            ],
        },
    )
    args = runner_cli.parse_args(
        [
            "--manifest",
            str(protocol_path),
            "--outputs",
            str(tmp_path / "outputs"),
            "--methods",
            "precision_residual_scratch_moe",
            "--seeds",
            "42",
            "--dry-run",
        ]
    )
    assert runner_cli.run(args) == 0
    out = capsys.readouterr().out
    assert "AUTO-INJECT dependency: bc seed=42 stage1" in out


# ---------------------------------------------------------------------------
# CLI orchestration
# ---------------------------------------------------------------------------


def test_cli_list(tmp_path: Path, capsys) -> None:
    protocol_path = _write_protocol(tmp_path, _valid_doc())
    args = runner_cli.parse_args(["--manifest", str(protocol_path), "--list"])
    assert runner_cli.run(args) == 0
    out = capsys.readouterr().out
    assert "precision_residual_phaseforge" in out
    assert "final_aligned_bc" in out
    assert "seeds: [42]" in out


def test_cli_dry_run_prints_commands_without_executing(tmp_path: Path, capsys) -> None:
    protocol_path = _write_protocol(tmp_path, _valid_doc())
    outputs = tmp_path / "outputs"
    args = runner_cli.parse_args(
        [
            "--manifest",
            str(protocol_path),
            "--outputs",
            str(outputs),
            "--methods",
            "bc",
            "--seeds",
            "42",
            "--dry-run",
        ]
    )
    assert runner_cli.run(args) == 0
    out = capsys.readouterr().out
    assert "WOULD RUN  bc seed=42 stage1" in out
    assert "phaseforge-train" in out
    assert "models=final_aligned_bc" in out
    # Eval cannot run before bc stage1 exists — dry-run reports the blocker.
    assert "BLOCKED  bc seed=42 eval" in out
    assert "No bc stage1 runs found" in out


def test_cli_dry_run_with_state_complete_skips(tmp_path: Path, capsys) -> None:
    protocol_path = _write_protocol(tmp_path, _valid_doc())
    outputs = tmp_path / "outputs"
    state = RunnerState(
        runner_cli.RunnerState.default_path(outputs), expected_commit=runner_cli._current_commit()
    )
    state.mark("bc", 42, "stage1", run_dir="bc/stage1/x", ckpt="bc/stage1/x/c.pt")
    args = runner_cli.parse_args(
        [
            "--manifest",
            str(protocol_path),
            "--outputs",
            str(outputs),
            "--methods",
            "bc",
            "--seeds",
            "42",
            "--dry-run",
        ]
    )
    assert runner_cli.run(args) == 0
    out = capsys.readouterr().out
    assert "skip bc seed=42 stage1 (already completed)" in out


def test_cli_continue_on_error_records_failures(tmp_path: Path, monkeypatch, capsys) -> None:
    protocol_path = _write_protocol(tmp_path, _valid_doc())

    def _boom(*args, **kwargs) -> None:
        raise CommandError("synthetic failure")

    monkeypatch.setattr(runner_cli, "run_step", _boom)
    outputs = tmp_path / "outputs"
    args = runner_cli.parse_args(
        [
            "--manifest",
            str(protocol_path),
            "--outputs",
            str(outputs),
            "--methods",
            "precision_residual_phaseforge",
            "--seeds",
            "42",
            "--continue-on-error",
        ]
    )
    assert runner_cli.run(args) == 1
    state = RunnerState(runner_cli.RunnerState.default_path(outputs))
    for phase in ("stage1", "stage2", "eval"):
        entry = state.get("precision_residual_phaseforge", 42, phase)
        assert entry is not None and entry["status"] == "failed"
    captured = capsys.readouterr()
    assert "FAILED precision_residual_phaseforge seed=42 stage1" in captured.out + captured.err


def test_cli_fails_fast_without_continue_on_error(tmp_path: Path, monkeypatch) -> None:
    protocol_path = _write_protocol(tmp_path, _valid_doc())

    def _boom(*args, **kwargs) -> None:
        raise CommandError("synthetic failure")

    monkeypatch.setattr(runner_cli, "run_step", _boom)
    args = runner_cli.parse_args(
        [
            "--manifest",
            str(protocol_path),
            "--outputs",
            str(tmp_path / "outputs"),
            "--methods",
            "precision_residual_phaseforge",
            "--seeds",
            "42",
        ]
    )
    assert runner_cli.run(args) == 1

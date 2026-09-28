"""CLI-level tests for the selection subsystem.

Complements the pure resolver tests in ``test_selection.py`` with the wired
behavior of ``phaseforge-sweep``: resolution preview, ``--tasks`` filtering,
the ``--expect-steps`` refusal gate, and the ``_runner/plan.json`` provenance
artifact. Mirrors the in-process ``parse_args`` → ``run`` pattern of
``test_runner.py``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from phaseforge.runner import cli as runner_cli


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


def _multi_task_doc() -> dict:
    """Minimal multi-task manifest: final names replicated across tasks."""
    return {
        "name": "mini_multi",
        "task": "all",
        "seeds": [42],
        "defaults": [],
        "methods": [
            {
                "index": 1,
                "name": "precision_residual_phaseforge",
                "role": "proposed",
                "model": "precision_residual_phaseforge",
                "data": "lift",
                "task": "Lift",
                "stages": [1, 2],
                "stage2_source": "self",
                "evaluate": True,
            },
            {
                "index": 2,
                "name": "precision_residual_phaseforge",
                "role": "proposed",
                "model": "precision_residual_phaseforge",
                "data": "can",
                "task": "Can",
                "stages": [1, 2],
                "stage2_source": "self",
                "evaluate": True,
            },
            {
                "index": 3,
                "name": "bc",
                "role": "floor",
                "model": "final_aligned_bc",
                "data": "lift",
                "task": "Lift",
                "stages": [1],
                "evaluate": True,
            },
            {
                "index": 4,
                "name": "bc",
                "role": "floor",
                "model": "final_aligned_bc",
                "data": "can",
                "task": "Can",
                "stages": [1],
                "evaluate": True,
            },
        ],
    }


def test_cli_selection_preview_prints_cells_and_breakdown(tmp_path: Path, capsys) -> None:
    protocol_path = _write_protocol(tmp_path, _valid_doc())
    args = runner_cli.parse_args(
        [
            "--manifest",
            str(protocol_path),
            "--outputs",
            str(tmp_path / "outputs"),
            "--methods",
            "bc",
            "--seeds",
            "42",
            "--dry-run",
        ]
    )
    assert runner_cli.run(args) == 0
    out = capsys.readouterr().out
    assert "[runner] selection: 1 cells from mini" in out
    assert "bc@Mini" in out
    assert "[runner] steps: eval=1 stage1=1" in out


def test_cli_bare_name_selects_across_tasks(tmp_path: Path, capsys) -> None:
    protocol_path = _write_protocol(tmp_path, _multi_task_doc())
    args = runner_cli.parse_args(
        [
            "--manifest",
            str(protocol_path),
            "--outputs",
            str(tmp_path / "outputs"),
            "--methods",
            "precision_residual_phaseforge",
            "--dry-run",
        ]
    )
    assert runner_cli.run(args) == 0
    out = capsys.readouterr().out
    assert "[runner] selection: 2 cells" in out
    assert "precision_residual_phaseforge@Lift" in out
    assert "precision_residual_phaseforge@Can" in out
    # 2 tasks x 1 seed x (stage1, stage2, eval).
    assert "[runner] plan (6 steps" in out


def test_cli_task_filter_narrows_selection(tmp_path: Path, capsys) -> None:
    protocol_path = _write_protocol(tmp_path, _multi_task_doc())
    args = runner_cli.parse_args(
        [
            "--manifest",
            str(protocol_path),
            "--outputs",
            str(tmp_path / "outputs"),
            "--methods",
            "precision_residual_phaseforge",
            "--tasks",
            "can",
            "--dry-run",
        ]
    )
    assert runner_cli.run(args) == 0
    out = capsys.readouterr().out
    assert "precision_residual_phaseforge@Can" in out
    assert "precision_residual_phaseforge@Lift" not in out


def test_cli_unknown_task_exits_2(tmp_path: Path, capsys) -> None:
    protocol_path = _write_protocol(tmp_path, _multi_task_doc())
    args = runner_cli.parse_args(
        [
            "--manifest",
            str(protocol_path),
            "--outputs",
            str(tmp_path / "outputs"),
            "--tasks",
            "square",
            "--dry-run",
        ]
    )
    assert runner_cli.run(args) == 2
    err = capsys.readouterr().err
    assert "Unknown task 'square'" in err
    # load_protocol orders methods by (task, index), so tasks list sorted.
    assert "Valid tasks: ['Can', 'Lift']" in err


def test_cli_expect_steps_mismatch_refuses_to_run(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    protocol_path = _write_protocol(tmp_path, _valid_doc())

    def _must_not_run(*args, **kwargs):  # pragma: no cover - refusal guard
        raise AssertionError("--expect-steps mismatch must refuse before executing")

    monkeypatch.setattr(runner_cli, "run_step", _must_not_run)
    args = runner_cli.parse_args(
        [
            "--manifest",
            str(protocol_path),
            "--outputs",
            str(tmp_path / "outputs"),
            "--methods",
            "bc",
            "--seeds",
            "42",
            "--expect-steps",
            "3",  # the actual plan is stage1 + eval = 2 steps
        ]
    )
    assert runner_cli.run(args) == 2
    err = capsys.readouterr().err
    assert "--expect-steps 3 but the plan has 2 steps" in err


def test_cli_expect_steps_match_proceeds(tmp_path: Path, capsys) -> None:
    protocol_path = _write_protocol(tmp_path, _valid_doc())
    args = runner_cli.parse_args(
        [
            "--manifest",
            str(protocol_path),
            "--outputs",
            str(tmp_path / "outputs"),
            "--methods",
            "bc",
            "--seeds",
            "42",
            "--expect-steps",
            "2",
            "--dry-run",
        ]
    )
    assert runner_cli.run(args) == 0


def test_cli_dry_run_writes_no_plan_artifact(tmp_path: Path) -> None:
    protocol_path = _write_protocol(tmp_path, _valid_doc())
    args = runner_cli.parse_args(
        [
            "--manifest",
            str(protocol_path),
            "--outputs",
            str(tmp_path / "outputs"),
            "--dry-run",
        ]
    )
    assert runner_cli.run(args) == 0
    assert not (tmp_path / "outputs" / "_runner" / "plan.json").exists()


def test_cli_writes_plan_artifact(tmp_path: Path, monkeypatch) -> None:
    protocol_path = _write_protocol(tmp_path, _multi_task_doc())
    outputs = tmp_path / "outputs"

    def _fake_run_step(step, **kwargs):  # succeed without producing artifacts
        return None

    monkeypatch.setattr(runner_cli, "run_step", _fake_run_step)
    args = runner_cli.parse_args(
        [
            "--manifest",
            str(protocol_path),
            "--outputs",
            str(outputs),
            "--methods",
            "precision_residual_phaseforge",
            "--tasks",
            "Can",
            "--continue-on-error",
        ]
    )
    runner_cli.run(args)

    artifact = outputs / "_runner" / "plan.json"
    assert artifact.is_file()
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["manifest"]["sha256"] == hashlib.sha256(
        protocol_path.read_bytes()
    ).hexdigest()
    assert payload["selection"] == {
        "method_tokens": ["precision_residual_phaseforge"],
        "tasks": ["Can"],
        "seeds": [],
    }
    assert [c["phase_key"] for c in payload["resolved_cells"]] == [
        "Can/precision_residual_phaseforge"
    ]
    assert payload["resolved_cells"][0]["stages"] == [1, 2]
    assert payload["step_count"] == 3  # stage1 + stage2 + eval, one seed
    assert payload["expect_steps"] is None
    assert isinstance(payload["argv"], list) and payload["argv"]


def test_cli_list_accepts_task_filter(tmp_path: Path, capsys) -> None:
    protocol_path = _write_protocol(tmp_path, _multi_task_doc())
    args = runner_cli.parse_args(
        [
            "--manifest",
            str(protocol_path),
            "--outputs",
            str(tmp_path / "outputs"),
            "--tasks",
            "Can",
            "--list",
        ]
    )
    assert runner_cli.run(args) == 0
    out = capsys.readouterr().out
    assert "data=can" in out
    assert "data=lift" not in out


def test_cli_plan_artifact_failure_aborts_before_running(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    protocol_path = _write_protocol(tmp_path, _valid_doc())

    def _must_not_run(*args, **kwargs):  # pragma: no cover - refusal guard
        raise AssertionError("an unrecordable sweep must not start")

    def _fail_write(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(runner_cli, "run_step", _must_not_run)
    monkeypatch.setattr(runner_cli, "_write_plan_artifact", _fail_write)
    args = runner_cli.parse_args(
        [
            "--manifest",
            str(protocol_path),
            "--outputs",
            str(tmp_path / "outputs"),
            "--methods",
            "bc",
            "--seeds",
            "42",
        ]
    )
    assert runner_cli.run(args) == 2
    assert "cannot write plan artifact" in capsys.readouterr().err


def test_cli_list_rejects_unknown_task(tmp_path: Path, capsys) -> None:
    protocol_path = _write_protocol(tmp_path, _multi_task_doc())
    args = runner_cli.parse_args(
        [
            "--manifest",
            str(protocol_path),
            "--outputs",
            str(tmp_path / "outputs"),
            "--tasks",
            "Nowhere",
            "--list",
        ]
    )
    assert runner_cli.run(args) == 2
    assert "Unknown task 'Nowhere'" in capsys.readouterr().err

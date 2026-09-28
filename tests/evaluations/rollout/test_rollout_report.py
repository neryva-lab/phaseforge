"""Tests for the aggregated rollout report (CSVs + paired statistics)."""

from __future__ import annotations

import json

from phaseforge.evaluations.rollout.report import (
    COMPARISONS_CSV,
    SUCCESS_CSV,
    build_rollout_report,
)


def _write_run(
    base,
    model: str,
    seed: int,
    run_name: str,
    outcomes: list[tuple[bool, bool]],  # (valid, success) per case
    *,
    tag: str | None = None,
    checkpoint: str = "cafe",
) -> None:
    run_dir = base / "eval" / model / f"seed{seed}" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    run_dir.joinpath("rollout_summary.json").write_text(
        json.dumps({"run_id": run_name, "model": model}), encoding="utf-8"
    )
    run_dir.joinpath("run_meta.json").write_text(
        json.dumps({"model_name": model, "seed": seed}), encoding="utf-8"
    )
    lines = []
    for index, (valid, success) in enumerate(outcomes):
        row = {
            "run_id": run_name,
            "model": model,
            "checkpoint_sha256": checkpoint,
            "task": "Lift",
            "training_seed": seed,
            "reset_seed": 2026,
            "episode_index": index,
            "valid_episode": valid,
            "steps": 100,
        }
        if tag is not None:
            row["tag"] = tag
        if valid:
            row["success"] = success
            row["timed_out"] = not success
            if not success:
                row["failure_category"] = "task_timeout"
        lines.append(json.dumps(row))
    run_dir.joinpath("episodes.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    run_dir.with_name(run_dir.name + ".completed").write_text("{}", encoding="utf-8")


def test_report_aggregates_and_writes_csvs(tmp_path) -> None:
    base = tmp_path / "outputs"
    # The proposed method solves all 5; bc solves 3 of 5 on the same cases.
    _write_run(
        base,
        "precision_residual_phaseforge",
        42,
        "r1",
        [(True, True)] * 5,
    )
    _write_run(
        base,
        "bc",
        42,
        "r2",
        [(True, True)] * 3 + [(True, False)] * 2,
    )
    # offline-only run without rollout_summary must be skipped
    offline_dir = base / "eval" / "bc" / "seed43" / "ro"
    offline_dir.mkdir(parents=True)
    offline_dir.joinpath("episodes.jsonl").write_text("", encoding="utf-8")

    report = build_rollout_report(base)

    assert report["run_count"] == 2
    assert report["episode_count"] == 10

    success_csv = (base / "_results" / SUCCESS_CSV).read_text(encoding="utf-8")
    assert "precision_residual_phaseforge" in success_csv and "bc" in success_csv
    assert "policy_failures" in success_csv

    comparisons_csv = (base / "_results" / COMPARISONS_CSV).read_text(encoding="utf-8")
    assert "mcnemar_exact_p" in comparisons_csv
    assert "newcombe_ci95_low" in comparisons_csv
    assert "discordant_baseline_wins" in comparisons_csv
    # 5 paired cases: baseline won 2 (cases 3,4) → b=2, c=0 → p = 2/2^2 = 0.5
    assert "0.5" in comparisons_csv

    report_json = json.loads(
        (base / "_results" / "rollout_report.json").read_text(encoding="utf-8")
    )
    comparison = report_json["comparison_rows"][0]
    assert comparison["model"] == "bc"
    assert comparison["discordant_baseline_wins"] == 2
    assert comparison["discordant_model_wins"] == 0
    assert comparison["mcnemar_exact_p"] == 0.5
    assert comparison["mcnemar_holm_p"] == 0.5
    # Effect direction is PhaseForge minus comparator: 1.0 - 0.6.
    assert comparison["diff"] == 0.4


def test_report_handles_empty_outputs(tmp_path) -> None:
    base = tmp_path / "empty"
    report = build_rollout_report(base)
    assert report["run_count"] == 0
    assert report["episode_count"] == 0


def test_paired_rows_per_seed_holms(tmp_path) -> None:
    base = tmp_path / "outputs"
    for seed in (42, 43):
        _write_run(base, "precision_residual_phaseforge", seed, f"p{seed}", [(True, True)] * 4)
        _write_run(base, "bc", seed, f"b{seed}", [(True, True)] * 2 + [(True, False)] * 2)
        _write_run(base, "precision_residual_scratch_moe", seed, f"s{seed}", [(True, False)] * 4)
    report = build_rollout_report(base)
    comparisons = report["comparison_rows"]
    assert len(comparisons) == 4  # 2 seeds x 2 non-baseline identities
    seeds = {c["training_seed"] for c in comparisons}
    assert seeds == {42, 43}
    # Holm applied per task and seed: for each cell the two p-values (0.125 for
    # the scratch-init control, 0.5 for bc) adjust to [0.25, 0.5].
    for seed in (42, 43):
        rows = [c for c in comparisons if c["training_seed"] == seed]
        assert sorted(c["mcnemar_holm_p"] for c in rows) == [0.25, 0.5]

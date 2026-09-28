"""CPU-only tests for the training-side provenance pipeline.

Covers the curve/summary schema validators, the per-run curve writer, the
global training ledger + reconciliation, the cache-provenance copy and
artifact manifest, the episode records + rollout statistics, the training
summarize tooling, the phase accumulator, and the persistence callback
end-to-end. Everything runs against ``tmp_path`` — no network, no GPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from omegaconf import DictConfig

from phaseforge.outputs_writer.curves import (
    TrainingCurveWriter,
    validate_curve_row,
    validate_summary,
)
from phaseforge.outputs_writer.episodes import (
    append_episode_record,
    paired_rollout_comparisons,
    read_episode_records,
    summarize_episodes,
    validate_episode_record,
    wilson_interval,
)
from phaseforge.outputs_writer.provenance import (
    copy_cache_provenance,
    sha256_file,
    write_artifact_manifest,
)
from phaseforge.outputs_writer.schema import SchemaError
from phaseforge.outputs_writer.training_summaries import (
    read_training_curves,
    summarize_rollout,
    summarize_training,
    training_aggregate_rows,
    training_cost_rows,
    write_training_curves_csv,
)
from phaseforge.outputs_writer.training_summary import (
    append_training_summary_row,
    has_reconciliation_record,
    read_training_summary_rows,
    reconcile_training_ledger,
    write_reconciliation_record,
)
from phaseforge.trains.callbacks.persistence import MetricPersistenceCallback
from phaseforge.trains.loops.base import _PhaseAccumulator
from phaseforge.utils.config import checkpoint_source_info

# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def make_curve(
    *,
    run_id: str = "a1b2c3d4",
    epoch: int = 1,
    global_step: int = 100,
    **overrides: Any,
) -> dict:
    row = {
        "run_id": run_id,
        "epoch": epoch,
        "global_step": global_step,
        "train/lr": 0.001,
        "epoch_wall_seconds": 3.5,
        "train_steps_per_second": 28.0,
        "train/loss_total": 0.041,
        "train/loss_action": 0.040,
        "val/loss_total": 0.043,
        "val/loss_action": 0.042,
    }
    row.update(overrides)
    return row


def make_summary(
    *,
    run_id: str = "a1b2c3d4",
    model: str = "precision_residual_phaseforge",
    stage: int = 1,
    seed: int | None = 42,
    kind: str = "train",
    **overrides,
) -> dict:
    row = {
        "run_id": run_id,
        "kind": kind,
        "model": model,
        "stage": stage,
        "seed": seed,
        "config_hash": "f89790f7520ddfdb",
        "data_config_hash": "cachedatahash",
        "data_provenance_path": "metadata/data_provenance.json",
        "git_sha": "c0e72de",
        "device": "cpu",
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:05:00+00:00",
        "wall_seconds": 300.0,
        "epochs_run": 5,
        "trainable_params": 1234,
        "total_params": 4321,
        "best_epoch": 4,
        "final_val": {"loss_total": 0.043, "loss_action": 0.042},
        "extra": {},
    }
    row.update(overrides)
    return row


def make_episode(
    *,
    run_id: str = "a1b2c3d4",
    model: str = "precision_residual_phaseforge",
    task: str = "pick_and_place",
    seed: int = 42,
    episode_index: int = 0,
    valid: bool = True,
    success: bool = True,
    **overrides: Any,
) -> dict:
    row = {
        "run_id": run_id,
        "model": model,
        "checkpoint_sha256": "deadbeef" * 8,
        "task": task,
        "training_seed": seed,
        "reset_seed": 7,
        "episode_index": episode_index,
        "valid_episode": valid,
    }
    if valid:
        row["success"] = success
        if not success:
            row["failure_category"] = "grasp_dropped"
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# Curve schema + writer
# ---------------------------------------------------------------------------


class TestCurveSchema:
    def test_valid_core_row_passes(self) -> None:
        validate_curve_row(make_curve())

    def test_valid_full_row_passes(self) -> None:
        validate_curve_row(
            make_curve(
                **{  # type: ignore[arg-type]
                    "train/loss_phase": 0.1,
                    "val/loss_phase": 0.2,
                    "train/phase_acc": 0.9,
                    "val/phase_acc": 0.88,
                    "train/phase_balanced_acc": 0.85,
                    "val/phase_balanced_acc": 0.84,
                    "val/routing_accuracy": 0.8,
                    "val/routing_balanced_accuracy": 0.79,
                    "val/topk_balance_score": 0.5,
                    "checkpoint_monitor": "val/loss_action",
                    "checkpoint_monitor_value": 0.042,
                    "peak_gpu_memory_mb": None,
                }
            )
        )

    def test_missing_required_key_fails(self) -> None:
        row = make_curve()
        del row["val/loss_action"]
        with pytest.raises(SchemaError, match="missing required keys"):
            validate_curve_row(row)

    def test_unknown_top_level_key_fails(self) -> None:
        row = make_curve()
        row["surprise"] = 1.0
        with pytest.raises(SchemaError, match="unknown top-level keys"):
            validate_curve_row(row)

    def test_bool_epoch_rejected(self) -> None:
        with pytest.raises(SchemaError, match="epoch"):
            validate_curve_row(make_curve(epoch=True))

    def test_float_global_step_rejected(self) -> None:
        with pytest.raises(SchemaError, match="global_step"):
            validate_curve_row(make_curve(global_step=100.0))  # type: ignore[arg-type]

    def test_inf_core_metric_rejected(self) -> None:
        with pytest.raises(SchemaError, match="train/loss_total"):
            validate_curve_row(make_curve(**{"train/loss_total": float("inf")}))  # type: ignore[arg-type]

    def test_nan_core_metric_accepted(self) -> None:
        validate_curve_row(make_curve(**{"train/loss_total": float("nan")}))  # type: ignore[arg-type]

    def test_inf_optional_metric_rejected(self) -> None:
        with pytest.raises(SchemaError, match="routing_accuracy"):
            validate_curve_row(make_curve(routing_accuracy=float("inf")))

    def test_monitor_requires_value(self) -> None:
        with pytest.raises(SchemaError, match="checkpoint_monitor_value"):
            validate_curve_row(make_curve(checkpoint_monitor="val/loss_action"))

    def test_peak_memory_nullable_and_numeric(self) -> None:
        validate_curve_row(make_curve(peak_gpu_memory_mb=None))
        validate_curve_row(make_curve(peak_gpu_memory_mb=1234.5))
        with pytest.raises(SchemaError, match="peak_gpu_memory_mb"):
            validate_curve_row(make_curve(peak_gpu_memory_mb="lots"))


class TestSummarySchema:
    def test_valid_train_summary_passes(self) -> None:
        validate_summary(make_summary())

    def test_valid_eval_summary_passes(self) -> None:
        validate_summary(make_summary(kind="eval", stage=2))

    def test_missing_required_key_fails(self) -> None:
        row = make_summary()
        del row["wall_seconds"]
        with pytest.raises(SchemaError, match="missing required keys"):
            validate_summary(row)

    def test_unknown_top_level_key_fails(self) -> None:
        row = make_summary()
        row["mystery"] = 1
        with pytest.raises(SchemaError, match="unknown top-level keys"):
            validate_summary(row)

    def test_bad_kind_fails(self) -> None:
        with pytest.raises(SchemaError, match="kind"):
            validate_summary(make_summary(kind="rollout"))

    def test_float_epochs_rejected(self) -> None:
        with pytest.raises(SchemaError, match="epochs_run"):
            validate_summary(make_summary(epochs_run=5.0))

    def test_seed_nullable(self) -> None:
        validate_summary(make_summary(seed=None))

    def test_best_epoch_nullable(self) -> None:
        validate_summary(make_summary(best_epoch=None))

    def test_best_val_monitor_nullable(self) -> None:
        validate_summary(make_summary(best_val_monitor=None))
        validate_summary(make_summary(best_val_monitor=0.042))
        with pytest.raises(SchemaError, match="best_val_monitor"):
            validate_summary(make_summary(best_val_monitor="good"))

    def test_source_stage1_nullable_or_dict(self) -> None:
        validate_summary(make_summary(source_stage1=None))
        validate_summary(make_summary(source_stage1={"run_id": "x"}))
        with pytest.raises(SchemaError, match="source_stage1"):
            validate_summary(make_summary(source_stage1=[1, 2]))

    def test_global_steps_nullable_or_int(self) -> None:
        validate_summary(make_summary(global_steps=None))
        validate_summary(make_summary(global_steps=500))
        with pytest.raises(SchemaError, match="global_steps"):
            validate_summary(make_summary(global_steps=500.0))

    def test_balance_coeff_finite(self) -> None:
        validate_summary(make_summary(balance_coeff=0.05))
        with pytest.raises(SchemaError, match="balance_coeff"):
            validate_summary(make_summary(balance_coeff=float("inf")))

    def test_freeze_encoder_bool(self) -> None:
        validate_summary(make_summary(freeze_encoder=False))
        with pytest.raises(SchemaError, match="freeze_encoder"):
            validate_summary(make_summary(freeze_encoder="yes"))

    def test_tag_and_method_str_or_null(self) -> None:
        validate_summary(make_summary(tag="alternate", method="alternate_method"))
        validate_summary(make_summary(tag=None, method=None))
        with pytest.raises(SchemaError, match="tag"):
            validate_summary(make_summary(tag=7))
        with pytest.raises(SchemaError, match="method"):
            validate_summary(make_summary(method=["bc"]))


class TestCurveWriter:
    def test_append_and_read_roundtrip(self, tmp_path: Path) -> None:
        writer = TrainingCurveWriter(tmp_path / "run")
        writer.append_curve_row(make_curve(epoch=1))
        writer.append_curve_row(make_curve(epoch=2, global_step=200))
        rows = writer.read_curves()
        assert len(rows) == 2
        assert [r["epoch"] for r in rows] == [1, 2]
        assert (tmp_path / "run" / "metrics" / "training_curves.jsonl").exists()

    def test_append_is_idempotent_per_epoch(self, tmp_path: Path) -> None:
        writer = TrainingCurveWriter(tmp_path / "run")
        writer.append_curve_row(make_curve(epoch=1))
        writer.append_curve_row(make_curve(epoch=1, global_step=999))
        rows = writer.read_curves()
        assert len(rows) == 1
        assert rows[0]["global_step"] == 100

    def test_append_validates_before_write(self, tmp_path: Path) -> None:
        writer = TrainingCurveWriter(tmp_path / "run")
        bad = make_curve(epoch="1")  # type: ignore[arg-type]
        with pytest.raises(SchemaError):
            writer.append_curve_row(bad)
        assert not (tmp_path / "run" / "metrics" / "training_curves.jsonl").exists()

    def test_read_skips_truncated_trailing_line(self, tmp_path: Path) -> None:
        writer = TrainingCurveWriter(tmp_path / "run")
        writer.append_curve_row(make_curve(epoch=1))
        path = writer.curves_path
        with open(path, "a", encoding="utf-8") as f:
            f.write('{"run_id": "crash')
        rows = writer.read_curves()
        assert len(rows) == 1

    def test_write_summary(self, tmp_path: Path) -> None:
        writer = TrainingCurveWriter(tmp_path / "run")
        path = writer.write_summary(make_summary())
        assert path.name == "summary.json"
        assert json.loads(path.read_text())["run_id"] == "a1b2c3d4"

    def test_write_summary_validates(self, tmp_path: Path) -> None:
        writer = TrainingCurveWriter(tmp_path / "run")
        with pytest.raises(SchemaError):
            writer.write_summary(make_summary(epochs_run="5"))


# ---------------------------------------------------------------------------
# Training summary ledger
# ---------------------------------------------------------------------------


class TestTrainingSummaryLedger:
    def test_append_and_read_roundtrip(self, tmp_path: Path) -> None:
        results_dir = tmp_path / "_results"
        append_training_summary_row(results_dir, make_summary(run_id="aaaa0001"))
        append_training_summary_row(results_dir, make_summary(run_id="bbbb0002", seed=43))
        rows = read_training_summary_rows(results_dir)
        assert [r["run_id"] for r in rows] == ["aaaa0001", "bbbb0002"]
        assert (results_dir / "training_summary.jsonl").exists()

    def test_append_is_idempotent(self, tmp_path: Path) -> None:
        results_dir = tmp_path / "_results"
        append_training_summary_row(results_dir, make_summary(run_id="aaaa0001"))
        append_training_summary_row(
            results_dir, make_summary(run_id="aaaa0001", wall_seconds=999.0)
        )
        rows = read_training_summary_rows(results_dir)
        assert len(rows) == 1
        assert rows[0]["wall_seconds"] == 300.0

    def test_append_validates_before_write(self, tmp_path: Path) -> None:
        results_dir = tmp_path / "_results"
        with pytest.raises(SchemaError):
            append_training_summary_row(results_dir, make_summary(epochs_run="5"))
        assert not (results_dir / "training_summary.jsonl").exists()

    def test_reconciliation_record(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "2026-01-01_00-00-00_aaaa0001"
        path = write_reconciliation_record(run_dir, RuntimeError("boom"))
        assert path.name == "training_summary_pending.json"
        assert has_reconciliation_record(run_dir)
        record = json.loads(path.read_text())
        assert record["run_id"] == "aaaa0001"
        assert "boom" in record["error"]

    def test_reconcile_builds_ledger_from_run_local_summaries(self, tmp_path: Path) -> None:
        outputs = tmp_path / "outputs"
        run_dir = outputs / "precision_residual_phaseforge" / "stage1" / "2026-01-01_00-00-00_aaaa0001"
        metrics = run_dir / "metrics"
        metrics.mkdir(parents=True)
        (metrics / "summary.json").write_text(
            json.dumps(make_summary(run_id="aaaa0001", stage=1)), encoding="utf-8"
        )
        # A pre-existing ledger row must be preserved, not duplicated.
        append_training_summary_row(tmp_path / "_results", make_summary(run_id="cccc0003", seed=99))

        result = reconcile_training_ledger(tmp_path / "_results", outputs)
        assert result["scanned"] == 1
        assert result["appended"] == 1
        assert result["duplicates"] == 0
        rows = read_training_summary_rows(tmp_path / "_results")
        ids = {r["run_id"] for r in rows}
        assert ids == {"aaaa0001", "cccc0003"}
        added = next(r for r in rows if r["run_id"] == "aaaa0001")
        assert added["run_dir"] == "precision_residual_phaseforge/stage1/2026-01-01_00-00-00_aaaa0001"

    def test_reconcile_skips_duplicates(self, tmp_path: Path) -> None:
        outputs = tmp_path / "outputs"
        run_dir = outputs / "precision_residual_phaseforge" / "stage1" / "2026-01-01_00-00-00_aaaa0001"
        metrics = run_dir / "metrics"
        metrics.mkdir(parents=True)
        (metrics / "summary.json").write_text(
            json.dumps(make_summary(run_id="aaaa0001")), encoding="utf-8"
        )
        append_training_summary_row(tmp_path / "_results", make_summary(run_id="aaaa0001"))
        result = reconcile_training_ledger(tmp_path / "_results", outputs)
        assert result["duplicates"] == 1
        assert result["appended"] == 0
        assert len(read_training_summary_rows(tmp_path / "_results")) == 1

    @pytest.mark.parametrize(
        "content",
        ['{"bad": "row"}', "{ not json"],
    )
    def test_reconcile_raises_on_corrupt_run_local_summary(
        self, tmp_path: Path, content: str
    ) -> None:
        outputs = tmp_path / "outputs"
        run_dir = outputs / "precision_residual_phaseforge" / "stage1" / "2026-01-01_00-00-00_aaaa0001"
        metrics = run_dir / "metrics"
        metrics.mkdir(parents=True)
        (metrics / "summary.json").write_text(content, encoding="utf-8")
        with pytest.raises(SchemaError):
            reconcile_training_ledger(tmp_path / "_results", outputs)


# ---------------------------------------------------------------------------
# Provenance copy + artifact manifest
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_sha256_file(self, tmp_path: Path) -> None:
        path = tmp_path / "data.txt"
        path.write_text("hello world", encoding="utf-8")
        assert sha256_file(path) == (
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        )

    def test_copy_cache_provenance(self, tmp_path: Path) -> None:
        cache = tmp_path / "cache"
        manifest_dir = cache / "hash1234"
        manifest_dir.mkdir(parents=True)
        (manifest_dir / "manifest.json").write_text(
            json.dumps({"provenance": {"raw_files": {"a.pt": "aa"}}}),
            encoding="utf-8",
        )
        run_dir = tmp_path / "run"
        payload = copy_cache_provenance(run_dir, cache, "hash1234")
        assert payload["config_hash"] == "hash1234"
        written = json.loads((run_dir / "metadata" / "data_provenance.json").read_text())
        assert written["provenance"]["raw_files"] == {"a.pt": "aa"}

    def test_copy_cache_provenance_missing_manifest(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="manifest"):
            copy_cache_provenance(tmp_path / "run", tmp_path / "cache", "nope")

    def test_write_artifact_manifest(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True)
        (run_dir / "config.yaml").write_text("a: 1", encoding="utf-8")
        payload = write_artifact_manifest(
            run_dir,
            {
                "config": "config.yaml",
                "missing": "never_written.pt",
                "optional": None,
            },
        )
        assert payload["complete"] is False
        assert payload["missing"] == ["missing", "optional"]
        assert payload["artifacts"]["config"]["present"] is True
        assert payload["artifacts"]["config"]["sha256"]
        assert payload["artifacts"]["missing"]["present"] is False
        assert payload["artifacts"]["optional"]["present"] is False

    def test_write_artifact_manifest_complete(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True)
        (run_dir / "a.txt").write_text("x", encoding="utf-8")
        payload = write_artifact_manifest(run_dir, {"a": "a.txt"})
        assert payload["complete"] is True
        assert "missing" not in payload


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------


class TestEpisodes:
    def test_valid_success_record_passes(self) -> None:
        validate_episode_record(make_episode())

    def test_valid_failure_requires_category(self) -> None:
        row = make_episode(success=False)
        del row["failure_category"]
        with pytest.raises(SchemaError, match="failure_category"):
            validate_episode_record(row)
        validate_episode_record(make_episode(success=False))

    def test_invalid_record_must_not_carry_success(self) -> None:
        row = make_episode(valid=False)
        row["success"] = False
        with pytest.raises(SchemaError, match="invalid"):
            validate_episode_record(row)

    def test_valid_record_requires_success(self) -> None:
        row = make_episode()
        del row["success"]
        with pytest.raises(SchemaError, match="success"):
            validate_episode_record(row)

    def test_invalid_record_carries_exception(self) -> None:
        validate_episode_record(make_episode(valid=False, exception="Timeout"))

    def test_unknown_key_fails(self) -> None:
        row = make_episode()
        row["surprise"] = 1
        with pytest.raises(SchemaError, match="unknown"):
            validate_episode_record(row)

    def test_float_training_seed_rejected(self) -> None:
        with pytest.raises(SchemaError, match="training_seed"):
            validate_episode_record(make_episode(seed=42.0))  # type: ignore[arg-type]

    def test_append_read_roundtrip(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "eval_run"
        append_episode_record(output_dir, make_episode(episode_index=0))
        append_episode_record(output_dir, make_episode(episode_index=1, success=False))
        rows = read_episode_records(output_dir)
        assert len(rows) == 2
        assert rows[1]["success"] is False
        assert (output_dir / "episodes.jsonl").exists()

    def test_append_validates_before_write(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "eval_run"
        with pytest.raises(SchemaError):
            append_episode_record(output_dir, make_episode(seed=42.0))  # type: ignore[arg-type]
        assert not (output_dir / "episodes.jsonl").exists()

    def test_wilson_interval_zero_n(self) -> None:
        low, high = wilson_interval(0, 0)
        assert low != low
        assert high != high

    def test_wilson_interval_endpoints(self) -> None:
        assert wilson_interval(5, 5) == (1.0, 1.0)
        assert wilson_interval(0, 5) == (0.0, 0.0)
        low, high = wilson_interval(3, 5)
        assert 0.0 <= low <= 0.6 <= high <= 1.0

    def test_summarize_episodes_groups_and_counts(self) -> None:
        rows = [
            make_episode(task="push", model="precision_residual_phaseforge", seed=42, episode_index=0),
            make_episode(task="push", model="precision_residual_phaseforge", seed=42, episode_index=1),
            make_episode(task="push", model="precision_residual_phaseforge", seed=42, episode_index=2, success=False),
            make_episode(task="push", model="bc", seed=42, episode_index=0),
            make_episode(
                task="push",
                model="precision_residual_phaseforge",
                seed=42,
                episode_index=3,
                valid=False,
                exception="Timeout",
            ),
        ]
        summaries = summarize_episodes(rows)
        by_key = {(s["task"], s["model"], s["tag"], s["training_seed"]): s for s in summaries}
        pf = by_key[("push", "precision_residual_phaseforge", None, 42)]
        assert pf["valid_episodes"] == 3
        assert pf["successes"] == 2
        assert pf["success_rate"] == pytest.approx(2 / 3)
        assert pf["invalid_attempts"] == 1
        assert pf["wilson_ci95_low"] < pf["success_rate"] < pf["wilson_ci95_high"]
        assert by_key[("push", "bc", None, 42)]["success_rate"] == 1.0

    def test_summarize_episodes_splits_tagged_variants(self) -> None:
        # ``bc`` and ``bc``/``robot_only`` share a model name; their rollout
        # episodes must never be merged into one success row.
        rows = [
            make_episode(task="push", model="bc", seed=42, episode_index=i) for i in range(3)
        ] + [
            make_episode(task="push", model="bc", tag="robot_only", seed=42, episode_index=i)
            for i in range(3)
        ]
        summaries = summarize_episodes(rows)
        by_key = {(s["task"], s["model"], s["tag"], s["training_seed"]): s for s in summaries}
        assert by_key[("push", "bc", None, 42)]["valid_episodes"] == 3
        assert by_key[("push", "bc", "robot_only", 42)]["valid_episodes"] == 3
        assert len(summaries) == 2

    def test_paired_rollout_comparisons(self) -> None:
        rows = []
        for model in ("precision_residual_phaseforge", "bc"):
            for episode_index in range(4):
                rows.append(
                    make_episode(
                        task="push",
                        model=model,
                        seed=42,
                        episode_index=episode_index,
                        success=episode_index < (3 if model == "precision_residual_phaseforge" else 2),
                    )
                )
        comparisons = paired_rollout_comparisons(
            rows, baseline="precision_residual_phaseforge"
        )
        assert len(comparisons) == 1
        comp = comparisons[0]
        assert comp["baseline"] == "precision_residual_phaseforge"
        assert comp["model"] == "bc"
        assert comp["diff"] == pytest.approx(0.75 - 0.50)


# ---------------------------------------------------------------------------
# Phase accumulator
# ---------------------------------------------------------------------------


class TestPhaseAccumulator:
    def test_micro_and_balanced(self) -> None:
        acc = _PhaseAccumulator(device=torch.device("cpu"), num_phases=2)
        logits = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        targets = torch.tensor([0, 1, 1, 0])
        acc.update(logits, targets)
        micro, balanced = acc.compute()
        # predictions: class0 correct (x2), class1 wrong then right -> 3/4
        assert micro == pytest.approx(0.75)
        # per-class recall: class0 = 2/2, class1 = 1/2 -> mean 0.75
        assert balanced == pytest.approx(0.75)

    def test_imbalanced_balanced_recall(self) -> None:
        acc = _PhaseAccumulator(device=torch.device("cpu"), num_phases=2)
        logits = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        targets = torch.tensor([0, 0, 1, 1, 1])
        acc.update(logits, targets)
        micro, balanced = acc.compute()
        assert micro == pytest.approx(1.0)
        assert balanced == pytest.approx(1.0)
        # A class never seen does not distort the balanced score.
        logits2 = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        targets2 = torch.tensor([0, 1])  # class1 wrong
        acc.update(logits2, targets2)
        micro2, balanced2 = acc.compute()
        # Combined: class0 3/3, class1 3/4 -> micro 6/7, balanced 0.875.
        assert micro2 == pytest.approx(6 / 7)
        assert balanced2 == pytest.approx(0.875)

    def test_masked_multistep(self) -> None:
        acc = _PhaseAccumulator(device=torch.device("cpu"), num_phases=2)
        logits = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]])
        targets = torch.tensor([[0, 1], [1, 0]])
        mask = torch.tensor([[True, False], [True, True]])
        acc.update(logits, targets, mask=mask)
        micro, balanced = acc.compute()
        # valid: (b0,t0)=0 right, (b1,t0)=1 right, (b1,t1)=0 right -> 3/3
        assert micro == pytest.approx(1.0)
        assert balanced == pytest.approx(1.0)

    def test_empty_update_no_data(self) -> None:
        acc = _PhaseAccumulator(device=torch.device("cpu"), num_phases=2)
        assert not acc.has_data
        logits = torch.tensor([[1.0, 0.0]])
        targets = torch.tensor([0])
        mask = torch.tensor([False])
        acc.update(logits, targets, mask=mask)
        assert not acc.has_data
        with pytest.raises(ValueError, match="no data"):
            acc.compute()
        acc.reset()
        assert not acc.has_data


# ---------------------------------------------------------------------------
# Persistence callback (end-to-end on a fake trainer)
# ---------------------------------------------------------------------------


def _persistence_cfg(stage: int, model_name: str) -> DictConfig:
    return DictConfig(
        {
            "project": {"seed": 42, "device": "cpu"},
            "models": {
                "name": model_name,
                "_target_": f"phaseforge.models.{model_name}",
                "router": {"balance_coeff": 0.05},
            },
            "train": {
                "stage": stage,
                "epochs": 2,
                "checkpoint": {"monitor": "val/loss_total"},
                "lambda_phase": 0.1,
                "freeze_encoder": False,
            },
        }
    )


class _FakeHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)


class _FakeModel(nn.Module):
    def __init__(self, *, with_phase_head: bool) -> None:
        super().__init__()
        if with_phase_head:
            self.phase_head = _FakeHead()
        self.stage = 1


class _FakeBestCallback:
    def __init__(self, best_path: Path) -> None:
        self.best_ckpt_path = str(best_path)
        self._topk = [(0.5, 1)]
        self.best_score = 0.043


class _FakeTrainer:
    def __init__(
        self, cfg: DictConfig, model: nn.Module, best_path: Path, lr: float = 0.001
    ) -> None:
        self.cfg = cfg
        self.train_cfg = cfg.train
        self.model = model
        self.device = torch.device("cpu")
        self.current_epoch = 0
        self.global_step = 0
        self.optimizer = _FakeOptimizer(lr)
        self.callbacks = [_FakeBestCallback(best_path)]

    def epoch_timing(self) -> dict[str, float | None]:
        return {
            "epoch_wall_seconds": 2.0,
            "train_steps_per_second": 10.0,
            "peak_gpu_memory_mb": None,
        }

    def parameter_counts(self) -> dict[str, int]:
        return {"trainable_params": 5, "total_params": 9}

    def epoch_train_metrics(self) -> dict[str, float]:
        return {}


class _FakeOptimizer:
    def __init__(self, lr: float) -> None:
        self.param_groups = [{"lr": lr}]


def _run_callback(
    cfg: DictConfig,
    model: nn.Module,
    run_dir: Path,
    metrics: dict[str, Any],
    val_metrics: dict[str, float],
) -> MetricPersistenceCallback:
    best_path = run_dir / "checkpoints" / "checkpoint_best.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    best_path.write_bytes(b"dummy")
    trainer = _FakeTrainer(cfg, model, best_path)
    cb = MetricPersistenceCallback(
        run_dir=run_dir,
        run_id="a1b2c3d4",
        data_config_hash="cachedatahash",
    )
    cb.on_train_start(trainer)
    for epoch in (1, 2):
        trainer.current_epoch = epoch
        cb.on_epoch_start(trainer)
        for batch_idx in range(2):
            trainer.global_step += 1
            cb.on_train_batch(
                trainer,
                batch={},
                out=model,
                metrics=metrics,
                n=4,
                step=trainer.global_step,
            )
        cb.on_epoch_end(trainer, val_metrics)
    trainer.current_epoch = 2
    trainer.global_step = 200
    cb.on_train_end(trainer)
    return cb


class TestPersistenceCallback:
    def test_writes_curves_and_summary(self, tmp_path: Path) -> None:
        cfg = _persistence_cfg(1, "precision_residual_phaseforge")
        run_dir = tmp_path / "run"
        model = _FakeModel(with_phase_head=True)
        metrics = {
            "loss_total": torch.tensor(0.041),
            "loss_action": torch.tensor(0.040),
            "loss_phase": torch.tensor(0.01),
        }
        val_metrics = {
            "loss_total": 0.043,
            "loss_action": 0.042,
            "loss_phase": 0.02,
        }
        _run_callback(cfg, model, run_dir, metrics, val_metrics)

        writer = TrainingCurveWriter(run_dir)
        rows = writer.read_curves()
        assert len(rows) == 2
        assert rows[0]["epoch"] == 1
        assert rows[0]["global_step"] == 2
        assert rows[0]["train/lr"] == 0.001
        assert rows[0]["checkpoint_monitor"] == "val/loss_total"
        assert rows[0]["checkpoint_monitor_value"] == pytest.approx(0.043)
        assert rows[0]["train/loss_total"] == pytest.approx(0.041)
        assert rows[0]["val/loss_phase"] == pytest.approx(0.02)

        summary = json.loads((run_dir / "metrics" / "summary.json").read_text())
        validate_summary(summary)
        assert summary["kind"] == "train"
        assert summary["model"] == "precision_residual_phaseforge"
        assert summary["stage"] == 1
        assert summary["seed"] == 42
        assert summary["epochs_run"] == 2
        assert summary["global_steps"] == 200
        assert summary["best_epoch"] == 1
        assert summary["best_checkpoint"] == "checkpoints/checkpoint_best.pt"
        assert summary["best_checkpoint_sha256"]
        assert summary["best_val_monitor"] == pytest.approx(0.043)
        assert summary["trainable_params"] == 5
        assert summary["total_params"] == 9
        assert summary["lambda_phase"] == pytest.approx(0.1)
        assert summary["balance_coeff"] == pytest.approx(0.05)
        assert summary["freeze_encoder"] is False
        assert summary["data_config_hash"] == "cachedatahash"
        assert summary["source_stage1"] == {
            "run_id": None,
            "checkpoint": None,
            "sha256": None,
            "model": None,
            "seed": None,
            "config_hash": None,
            "git_commit": None,
        }

    def test_bc_omits_phase_fields(self, tmp_path: Path) -> None:
        cfg = _persistence_cfg(1, "bc")
        run_dir = tmp_path / "run"
        model = _FakeModel(with_phase_head=False)
        metrics = {
            "loss_total": torch.tensor(0.041),
            "loss_action": torch.tensor(0.040),
            "loss_phase": torch.tensor(0.0),
        }
        _run_callback(cfg, model, run_dir, metrics, {"loss_total": 0.043, "loss_action": 0.042})
        rows = TrainingCurveWriter(run_dir).read_curves()
        assert "train/loss_phase" not in rows[0]
        assert "val/loss_phase" not in rows[0]

    def test_unknown_val_metric_dropped_without_crashing(self, tmp_path: Path) -> None:
        cfg = _persistence_cfg(1, "precision_residual_phaseforge")
        run_dir = tmp_path / "run"
        model = _FakeModel(with_phase_head=True)
        metrics = {"loss_total": torch.tensor(0.041), "loss_action": torch.tensor(0.040)}
        val_metrics = {
            "loss_total": 0.043,
            "loss_action": 0.042,
            "val/mystery": 1.0,
        }
        _run_callback(cfg, model, run_dir, metrics, val_metrics)
        rows = TrainingCurveWriter(run_dir).read_curves()
        assert "val/mystery" not in rows[0]
        assert "val/loss_action" in rows[0]

    def test_precision_residual_metrics_persisted_to_curves(self, tmp_path: Path) -> None:
        cfg = _persistence_cfg(2, "precision_residual_phaseforge")
        run_dir = tmp_path / "run"
        model = _FakeModel(with_phase_head=True)
        metrics = {
            "loss_total": torch.tensor(0.041),
            "loss_action": torch.tensor(0.040),
            "loss_release": torch.tensor(0.005),
            "loss_action_pos": torch.tensor(0.010),
            "loss_action_rot": torch.tensor(0.020),
            "loss_action_grip": torch.tensor(0.001),
            "loss_action_place": torch.tensor(0.015),
        }
        val_metrics = {
            "loss_total": 0.043,
            "loss_action": 0.042,
            "loss_release": 0.006,
            "loss_action_pos": 0.012,
            "loss_action_rot": 0.022,
            "loss_action_grip": 0.002,
            "loss_action_place": 0.018,
        }
        _run_callback(cfg, model, run_dir, metrics, val_metrics)
        rows = TrainingCurveWriter(run_dir).read_curves()
        assert rows[0]["train/loss_release"] == pytest.approx(0.005)
        assert rows[0]["val/loss_release"] == pytest.approx(0.006)
        assert rows[0]["train/loss_action_pos"] == pytest.approx(0.010)
        assert rows[0]["val/loss_action_pos"] == pytest.approx(0.012)
        assert rows[0]["train/loss_action_rot"] == pytest.approx(0.020)
        assert rows[0]["val/loss_action_rot"] == pytest.approx(0.022)
        assert rows[0]["train/loss_action_grip"] == pytest.approx(0.001)
        assert rows[0]["val/loss_action_grip"] == pytest.approx(0.002)
        assert rows[0]["train/loss_action_place"] == pytest.approx(0.015)
        assert rows[0]["val/loss_action_place"] == pytest.approx(0.018)

    def test_prefixed_monitor_value_resolved(self, tmp_path: Path) -> None:
        # Stage 2 returns already-prefixed keys (e.g. ``val/routing_entropy``);
        # the curve row's checkpoint_monitor_value must still be populated.
        cfg = _persistence_cfg(1, "precision_residual_phaseforge")
        cfg.train.checkpoint.monitor = "val/routing_entropy"
        run_dir = tmp_path / "run"
        model = _FakeModel(with_phase_head=True)
        metrics = {"loss_total": torch.tensor(0.041), "loss_action": torch.tensor(0.040)}
        val_metrics = {
            "loss_total": 0.043,
            "loss_action": 0.042,
            "val/routing_entropy": 0.88,
        }
        _run_callback(cfg, model, run_dir, metrics, val_metrics)
        rows = TrainingCurveWriter(run_dir).read_curves()
        assert rows[0]["checkpoint_monitor"] == "val/routing_entropy"
        assert rows[0]["checkpoint_monitor_value"] == pytest.approx(0.88)

    def test_resumed_run_appends_no_duplicate_epoch(self, tmp_path: Path) -> None:
        cfg = _persistence_cfg(1, "precision_residual_phaseforge")
        run_dir = tmp_path / "run"
        model = _FakeModel(with_phase_head=True)
        metrics = {
            "loss_total": torch.tensor(0.041),
            "loss_action": torch.tensor(0.040),
        }
        val_metrics = {"loss_total": 0.043, "loss_action": 0.042}
        best_path = run_dir / "checkpoints" / "checkpoint_best.pt"
        best_path.parent.mkdir(parents=True, exist_ok=True)
        best_path.write_bytes(b"dummy")
        trainer = _FakeTrainer(cfg, model, best_path)
        cb = MetricPersistenceCallback(
            run_dir=run_dir, run_id="a1b2c3d4", data_config_hash="cachedatahash"
        )
        cb.on_train_start(trainer)
        trainer.current_epoch = 1
        cb.on_epoch_start(trainer)
        trainer.global_step = 50
        cb.on_train_batch(trainer, {}, model, metrics, 4, 50)
        cb.on_epoch_end(trainer, val_metrics)
        # Resume path: epoch 1 already persisted; a new process re-runs it.
        cb2 = MetricPersistenceCallback(
            run_dir=run_dir, run_id="a1b2c3d4", data_config_hash="cachedatahash"
        )
        cb2.on_train_start(trainer)
        cb2.on_epoch_start(trainer)
        trainer.global_step = 100
        cb2.on_train_batch(trainer, {}, model, metrics, 4, 100)
        cb2.on_epoch_end(trainer, val_metrics)
        rows = TrainingCurveWriter(run_dir).read_curves()
        assert len(rows) == 1


# ---------------------------------------------------------------------------
# Summarize tooling
# ---------------------------------------------------------------------------


class TestTrainingSummaries:
    def test_aggregate_rows_mean_std_over_seeds(self) -> None:
        rows = [
            make_summary(run_id="a1", seed=42, epochs_run=5),
            make_summary(run_id="b2", seed=43, epochs_run=7),
        ]
        aggs = training_aggregate_rows(rows)
        assert len(aggs) == 1
        agg = aggs[0]
        assert agg["model"] == "precision_residual_phaseforge"
        assert agg["stage"] == 1
        assert agg["n_seeds"] == 2
        assert agg["loss_total_mean"] == pytest.approx(0.043)
        assert agg["loss_total_std"] == pytest.approx(0.0)
        assert agg["epochs_run_mean"] == pytest.approx(6.0)
        assert agg["trainable_params_mean"] == pytest.approx(1234.0)

    def test_cost_rows_total_steps(self) -> None:
        rows = [
            make_summary(run_id="a1", seed=42, wall_seconds=100.0, global_steps=100),
            make_summary(run_id="b2", seed=43, wall_seconds=300.0, global_steps=200),
        ]
        costs = training_cost_rows(rows)
        assert len(costs) == 1
        cost = costs[0]
        assert cost["wall_seconds_mean"] == pytest.approx(200.0)
        assert cost["wall_seconds_std"] == pytest.approx(141.4213562)
        assert cost["total_global_steps"] == pytest.approx(300.0)

    def test_cost_rows_tolerates_nullable_fields(self) -> None:
        # wall_seconds / global_steps are schema-nullable: a null seed must
        # not crash the cost table, it is simply excluded from the means.
        rows = [
            make_summary(run_id="a1", seed=42, wall_seconds=100.0, global_steps=100),
            make_summary(run_id="b2", seed=43, wall_seconds=None, global_steps=None),
        ]
        costs = training_cost_rows(rows)
        cost = costs[0]
        assert cost["n_seeds"] == 2
        assert cost["wall_seconds_mean"] == pytest.approx(100.0)
        assert cost["total_global_steps"] == pytest.approx(100.0)

    def test_aggregate_and_cost_split_tagged_variants(self) -> None:
        # ``bc`` and ``bc``/``robot_only`` share a model name but must not be
        # merged into one training aggregate / cost row.
        rows = [
            make_summary(run_id=f"bc{s}", model="bc", stage=1, seed=s, wall_seconds=50.0)
            for s in (42, 43, 44)
        ]
        rows += [
            make_summary(
                run_id=f"ro{s}",
                model="bc",
                tag="robot_only",
                stage=1,
                seed=s,
                wall_seconds=60.0,
            )
            for s in (42, 43, 44)
        ]
        aggs = training_aggregate_rows(rows)
        assert [(a["model"], a["tag"], a["stage"]) for a in aggs] == [
            ("bc", "", 1),
            ("bc", "robot_only", 1),
        ]
        costs = training_cost_rows(rows)
        assert [(c["model"], c["tag"], c["stage"]) for c in costs] == [
            ("bc", "", 1),
            ("bc", "robot_only", 1),
        ]
        assert costs[0]["n_seeds"] == 3 and costs[1]["n_seeds"] == 3

    def test_read_training_curves_located_via_run_dir(self, tmp_path: Path) -> None:
        outputs = tmp_path / "outputs"
        run_dir = outputs / "precision_residual_phaseforge" / "stage1" / "2026-01-01_00-00-00_aaaa0001"
        metrics = run_dir / "metrics"
        metrics.mkdir(parents=True)
        writer = TrainingCurveWriter(run_dir)
        writer.append_curve_row(make_curve(run_id="aaaa0001", epoch=1))
        rows = read_training_curves(
            outputs,
            [
                make_summary(
                    run_id="aaaa0001",
                    run_dir="precision_residual_phaseforge/stage1/2026-01-01_00-00-00_aaaa0001",
                )
            ],
        )
        assert len(rows) == 1
        assert rows[0]["run_id"] == "aaaa0001"
        assert rows[0]["model"] == "precision_residual_phaseforge"
        assert rows[0]["stage"] == 1

    def test_write_training_curves_csv(self, tmp_path: Path) -> None:
        curve_rows = [
            {"model": "precision_residual_phaseforge", "stage": 1, "seed": 42, **make_curve(run_id="a", epoch=1)},
            {
                "model": "precision_residual_phaseforge",
                "stage": 1,
                "seed": 43,
                **make_curve(run_id="b", epoch=1, global_step=150),
            },
        ]
        path = write_training_curves_csv(curve_rows, tmp_path / "curves.csv")
        text = path.read_text()
        assert text.splitlines()[0].startswith("model,tag,stage,epoch")
        assert "train/loss_total_mean" in text
        assert "precision_residual_phaseforge,,1,1" in text

    def test_write_training_curves_csv_tolerates_missing_metrics(self, tmp_path: Path) -> None:
        # Seed 43 emits a routing metric seed 42 lacks; the aggregate for it
        # must be NaN (n=0), not a KeyError.
        curve_rows = [
            {
                "model": "precision_residual_phaseforge",
                "stage": 1,
                "seed": 42,
                **make_curve(run_id="a", epoch=1),
            },
            {
                "model": "precision_residual_phaseforge",
                "stage": 1,
                "seed": 43,
                **make_curve(run_id="b", epoch=1, global_step=150),
                "val/routing_entropy": 0.5,
            },
        ]
        path = write_training_curves_csv(curve_rows, tmp_path / "curves.csv")
        text = path.read_text()
        assert "val/routing_entropy_mean" in text
        rows = list(__import__("csv").DictReader(text.splitlines()))
        row = rows[0]
        assert row["val/routing_entropy_n"] == "1"
        assert row["val/routing_entropy_mean"] == "0.5"
        assert row["train/loss_total_n"] == "2"

    def test_summarize_training_end_to_end(self, tmp_path: Path) -> None:
        outputs = tmp_path / "outputs"
        for run_id, seed, epoch in (
            ("aaaa0001", 42, 1),
            ("bbbb0002", 43, 1),
        ):
            run_dir = outputs / "precision_residual_phaseforge" / "stage1" / f"2026-01-01_00-00-00_{run_id}"
            metrics = run_dir / "metrics"
            metrics.mkdir(parents=True)
            (metrics / "summary.json").write_text(
                json.dumps(make_summary(run_id=run_id, seed=seed)),
                encoding="utf-8",
            )
            TrainingCurveWriter(run_dir).append_curve_row(make_curve(run_id=run_id, epoch=epoch))
        reconcile_training_ledger(outputs / "_results", outputs)

        paths = summarize_training(outputs)
        assert set(paths) == {"training_aggregates", "training_cost", "training_curves"}
        for name, path in paths.items():
            assert path.exists(), name
        agg = (outputs / "_summaries" / "training_aggregates.csv").read_text()
        assert agg.splitlines()[0].startswith("model,tag,stage,n_seeds")
        assert "precision_residual_phaseforge,,1,2" in agg
        curves = (outputs / "_summaries" / "training_curves.csv").read_text()
        assert curves.splitlines()[0].startswith("model,tag,stage,epoch")
        assert "precision_residual_phaseforge,,1,1" in curves

    def test_summarize_training_is_idempotent(self, tmp_path: Path) -> None:
        outputs = tmp_path / "outputs"
        run_dir = outputs / "precision_residual_phaseforge" / "stage1" / "2026-01-01_00-00-00_aaaa0001"
        metrics = run_dir / "metrics"
        metrics.mkdir(parents=True)
        (metrics / "summary.json").write_text(
            json.dumps(make_summary(run_id="aaaa0001")), encoding="utf-8"
        )
        reconcile_training_ledger(outputs / "_results", outputs)
        first = summarize_training(outputs)
        second = summarize_training(outputs)
        assert first["training_aggregates"].read_text() == second["training_aggregates"].read_text()

    def test_summarize_rollout_end_to_end(self, tmp_path: Path) -> None:
        outputs = tmp_path / "outputs"
        eval_run = outputs / "eval" / "precision_residual_phaseforge" / "2026-01-01_00-00-00_cccc0003"
        for model in ("precision_residual_phaseforge", "bc"):
            for idx in range(4):
                append_episode_record(
                    eval_run,
                    make_episode(
                        run_id="cccc0003",
                        model=model,
                        task="push",
                        episode_index=idx,
                        success=idx < (3 if model == "precision_residual_phaseforge" else 2),
                    ),
                )
        paths = summarize_rollout(outputs)
        assert set(paths) == {"rollout_success", "rollout_comparisons"}
        success = (outputs / "_summaries" / "rollout_success.csv").read_text()
        assert success.splitlines()[0].startswith("task,model,tag,training_seed")
        assert "push,precision_residual_phaseforge,,42" in success
        comparisons = (outputs / "_summaries" / "rollout_comparisons.csv").read_text()
        assert comparisons.splitlines()[0].startswith("task,training_seed,baseline")
        assert "0.25" in comparisons

    def test_summarize_rollout_empty_writes_empty_files(self, tmp_path: Path) -> None:
        outputs = tmp_path / "outputs"
        paths = summarize_rollout(outputs)
        assert paths["rollout_success"].read_text() == ""
        assert paths["rollout_comparisons"].read_text() == ""


# ---------------------------------------------------------------------------
# Checkpoint source resolution
# ---------------------------------------------------------------------------


class TestCheckpointSourceInfo:
    def _source(self, tmp_path: Path) -> Path:
        run_dir = tmp_path / "bc" / "stage1" / "2026-01-01_00-00-00_aaaa0001"
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        ckpt = ckpt_dir / "checkpoint_best.pt"
        ckpt.write_text("dummy", encoding="utf-8")
        (run_dir / "run_meta.json").write_text(
            json.dumps(
                {
                    "model_name": "bc",
                    "seed": 42,
                    "config_hash": "abcd1234",
                    "git_commit": "c0e72de",
                }
            ),
            encoding="utf-8",
        )
        return ckpt

    def test_resolves_source_identity(self, tmp_path: Path) -> None:
        ckpt = self._source(tmp_path)
        info = checkpoint_source_info(ckpt, base=tmp_path)
        assert info is not None
        assert info["run_id"] == "aaaa0001"
        assert info["model"] == "bc"
        assert info["seed"] == 42
        assert info["config_hash"] == "abcd1234"
        assert info["git_commit"] == "c0e72de"
        assert isinstance(info["checkpoint"], str)
        assert info["checkpoint"].endswith(
            "bc/stage1/2026-01-01_00-00-00_aaaa0001/checkpoints/checkpoint_best.pt"
        )
        assert isinstance(info["sha256"], str)
        assert len(info["sha256"]) == 64

    def test_returns_none_for_missing_ckpt(self, tmp_path: Path) -> None:
        assert checkpoint_source_info(tmp_path / "nope.pt", base=tmp_path) is None

    def test_legacy_run_dir_resolves_minimal(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "bc" / "stage1" / "legacy-name"
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        ckpt = ckpt_dir / "checkpoint_best.pt"
        ckpt.write_text("dummy", encoding="utf-8")
        info = checkpoint_source_info(ckpt, base=tmp_path)
        assert info is not None
        assert info["run_id"] is None
        assert info["model"] is None
        assert info["sha256"]

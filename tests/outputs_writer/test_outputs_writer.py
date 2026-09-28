"""CPU-only tests for the outputs_writer package.

Covers the schema validator, the atomic results + run ledgers, the
RunWriter lifecycle markers, the environment fingerprint, and the
aggregation / bootstrap / paired-comparison tables end-to-end. Everything
runs against ``tmp_path`` — no network, no GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from phaseforge.outputs_writer.backfill import (
    backfill_results,
    backfill_training_summary,
    collect_run_meta,
)
from phaseforge.outputs_writer.curves import validate_curve_row
from phaseforge.outputs_writer.ledger import LedgerRow, RunLedger
from phaseforge.outputs_writer.metadata import collect_environment
from phaseforge.outputs_writer.results import append_result_row, read_result_rows
from phaseforge.outputs_writer.schema import (
    OPTIONAL_METRIC_FIELDS,
    ResultRow,
    SchemaError,
    validate_row,
)
from phaseforge.outputs_writer.summarize import summarize_all
from phaseforge.outputs_writer.tables import (
    METRIC_COLUMNS,
    aggregate_rows,
    bootstrap_ci,
    paired_wilcoxon,
    write_aggregates_csv,
    write_bootstrap_csv,
    write_paired_wilcoxon_csv,
)
from phaseforge.outputs_writer.writer import RunWriter, parse_run_dir

# ---------------------------------------------------------------------------
# Fixtures / factories
# ---------------------------------------------------------------------------


def make_row(
    *,
    model: str = "precision_residual_phaseforge",
    stage: int = 2,
    seed: int = 42,
    action_mse: float = 0.028,
    with_metrics: bool = True,
    tag: str | None = None,
    method: str | None = None,
    data_config_hash: str | None = None,
    reset_bank: str | None = None,
    reset_seed: int | None = None,
) -> dict:
    row = {
        "run_id": "a1b2c3d4",
        "timestamp": "2026-01-01_00-00-00",
        "model": model,
        "stage": stage,
        "seed": seed,
        "git_sha": "c0e72de",
        "config_hash": "f89790f7520ddfdb",
        "device": "cuda:0",
        "ckpt_path": "outputs/precision_residual_phaseforge/stage2/2026-01-01_00-00-00_a1b2c3d4/model.pt",
        "action_mse": action_mse,
        "tag": tag,
        "method": method,
        "data_config_hash": data_config_hash,
        "reset_bank": reset_bank,
        "reset_seed": reset_seed,
    }
    if with_metrics:
        row.update(
            {
                "routing_entropy": 0.9,
                "topk_balance_score": 0.99,
                "phase_expert_nmi": 0.46,
            }
        )
    return row


def make_result_row(**kwargs) -> ResultRow:
    return ResultRow(**make_row(**kwargs))


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestSchema:
    def test_valid_full_row_passes(self) -> None:
        validate_row(make_row())

    def test_valid_bc_row_without_optional_metrics_passes(self) -> None:
        validate_row(make_row(model="bc", with_metrics=False))

    def test_missing_required_key_fails(self) -> None:
        row = make_row()
        del row["action_mse"]
        with pytest.raises(SchemaError, match="missing required keys"):
            validate_row(row)

    def test_unknown_top_level_key_fails(self) -> None:
        row = make_row()
        row["surprise"] = 1.0
        with pytest.raises(SchemaError, match="unknown top-level keys"):
            validate_row(row)

    def test_bool_stage_rejected(self) -> None:
        row = make_row(stage=True)
        with pytest.raises(SchemaError, match="stage"):
            validate_row(row)

    def test_float_seed_rejected(self) -> None:
        row = make_row(seed=42.0)
        with pytest.raises(SchemaError, match="seed"):
            validate_row(row)

    def test_string_seed_rejected(self) -> None:
        row = make_row(seed="42")
        with pytest.raises(SchemaError, match="seed"):
            validate_row(row)

    def test_string_model_rejected(self) -> None:
        row = make_row(model=7)
        with pytest.raises(SchemaError, match="model"):
            validate_row(row)

    def test_inf_action_mse_rejected(self) -> None:
        row = make_row(action_mse=float("inf"))
        with pytest.raises(SchemaError, match="action_mse"):
            validate_row(row)

    def test_nan_action_mse_accepted(self) -> None:
        validate_row(make_row(action_mse=float("nan")))

    def test_inf_optional_metric_rejected(self) -> None:
        row = make_row()
        row["routing_entropy"] = float("inf")
        with pytest.raises(SchemaError, match="routing_entropy"):
            validate_row(row)

    def test_string_optional_metric_rejected(self) -> None:
        row = make_row()
        row["topk_balance_score"] = "high"
        with pytest.raises(SchemaError, match="topk_balance_score"):
            validate_row(row)

    def test_non_dict_row_rejected(self) -> None:
        with pytest.raises(SchemaError, match="must be a dict"):
            validate_row(["not", "a", "dict"])  # type: ignore[arg-type]

    def test_extra_must_be_dict(self) -> None:
        row = make_row()
        row["extra"] = "nope"
        with pytest.raises(SchemaError, match="extra"):
            validate_row(row)

    def test_extra_dict_accepted(self) -> None:
        row = make_row()
        row["extra"] = {"note": "forward-compat"}
        validate_row(row)

    def test_result_row_roundtrip(self) -> None:
        row = make_result_row()
        validate_row(row.to_dict())
        assert row.to_dict()["action_mse"] == 0.028

    def test_tag_and_method_str_or_null(self) -> None:
        validate_row(make_row(tag="alternate", method="alternate_method"))
        validate_row(make_row(tag=None, method=None))
        row = make_row(tag="alternate")
        row["method"] = 7
        with pytest.raises(SchemaError, match="method"):
            validate_row(row)
        row = make_row()
        row["tag"] = 3
        with pytest.raises(SchemaError, match="tag"):
            validate_row(row)

    def test_result_row_defaults_tag_method_to_none(self) -> None:
        row = make_result_row()
        assert row.tag is None
        assert row.method is None

    def test_provenance_fields_accept_str_or_null(self) -> None:
        row = make_row()
        row["data_config_hash"] = "a2da6ba3"
        row["reset_bank"] = "bank_abc"
        row["reset_seed"] = 2026
        validate_row(row)
        validate_row(make_row(data_config_hash="a2da6ba3"))
        validate_row(make_row(data_config_hash=None, reset_bank=None, reset_seed=None))

    def test_provenance_fields_reject_bad_types(self) -> None:
        row = make_row()
        row["data_config_hash"] = 7
        with pytest.raises(SchemaError, match="data_config_hash"):
            validate_row(row)
        row = make_row()
        row["reset_bank"] = 3
        with pytest.raises(SchemaError, match="reset_bank"):
            validate_row(row)
        row = make_row()
        row["reset_seed"] = "2026"
        with pytest.raises(SchemaError, match="reset_seed"):
            validate_row(row)
        row = make_row()
        row["reset_seed"] = 2026.0
        with pytest.raises(SchemaError, match="reset_seed"):
            validate_row(row)

    def test_provenance_fields_roundtrip_via_result_row(self) -> None:
        row = make_result_row(
            data_config_hash="a2da6ba3",
            reset_bank="bank_abc",
            reset_seed=2026,
        )
        validate_row(row.to_dict())
        assert row.to_dict()["data_config_hash"] == "a2da6ba3"
        assert row.to_dict()["reset_bank"] == "bank_abc"
        assert row.to_dict()["reset_seed"] == 2026


# ---------------------------------------------------------------------------
# Results ledger
# ---------------------------------------------------------------------------


class TestCurveSchema:
    """Curve-row schema (phaseforge.outputs_writer.curves.validate_curve_row).

    The persistence callback builds one row per epoch and appends it only
    after strict validation; a stage-2 run that emits a metric the schema
    does not know aborts the whole sweep (regression: V2-C/D train scalars
    and the V2-C validation switch rate were added to the trainer without
    extending this allowlist).
    """

    def _core(self) -> dict:
        return {
            "run_id": "r1",
            "epoch": 0,
            "global_step": 0,
            "train/lr": 1e-3,
            "epoch_wall_seconds": 1.5,
            "train_steps_per_second": 40.0,
            "train/loss_total": 0.05,
            "train/loss_action": 0.04,
            "val/loss_total": 0.06,
            "val/loss_action": 0.05,
        }

    def test_core_row_passes(self) -> None:
        validate_curve_row(self._core())

    def test_v2_train_scalars_pass(self) -> None:
        row = self._core()
        row.update(
            {
                "train/loss_sticky": 0.0,
                "train/loss_teacher_kl": 0.0,
                "train/teacher_lambda": 0.5,
                "val/routing_switch_rate": 0.35,
            }
        )
        validate_curve_row(row)

    def test_stage1_phase_fields_pass(self) -> None:
        row = self._core()
        row.update(
            {
                "train/loss_phase": 1.2,
                "train/grad_cos_action_phase": 0.3,
                "train/lambda_phase": 1.0,
                "val/loss_phase": 1.3,
                "train/phase_acc": 0.9,
                "val/phase_acc": 0.88,
            }
        )
        validate_curve_row(row)

    def test_precision_residual_stage2_fields_pass(self) -> None:
        row = self._core()
        row.update(
            {
                "train/loss_release": 0.005,
                "train/release_lambda": 0.1,
                "val/loss_release": 0.006,
                "train/loss_action_pos": 0.01,
                "val/loss_action_pos": 0.012,
                "train/loss_action_rot": 0.02,
                "val/loss_action_rot": 0.022,
                "train/loss_action_grip": 0.001,
                "val/loss_action_grip": 0.002,
                "train/loss_action_place": 0.015,
                "val/loss_action_place": 0.018,
            }
        )
        validate_curve_row(row)

    def test_missing_required_key_fails(self) -> None:
        row = self._core()
        del row["val/loss_action"]
        with pytest.raises(SchemaError, match="missing required keys"):
            validate_curve_row(row)

    def test_unknown_top_level_key_fails(self) -> None:
        row = self._core()
        row["train/surprise"] = 1.0
        with pytest.raises(SchemaError, match="unknown top-level keys"):
            validate_curve_row(row)

    def test_bool_epoch_rejected(self) -> None:
        row = self._core()
        row["epoch"] = True
        with pytest.raises(SchemaError, match="epoch"):
            validate_curve_row(row)

    def test_optional_numeric_rejects_bool(self) -> None:
        row = self._core()
        row["train/teacher_lambda"] = True
        with pytest.raises(SchemaError, match="train/teacher_lambda"):
            validate_curve_row(row)


class TestResults:
    def test_append_and_read_roundtrip(self, tmp_path: Path) -> None:
        results_dir = tmp_path / "_results"
        append_result_row(results_dir, make_row(seed=42))
        append_result_row(results_dir, make_row(seed=43))
        rows = read_result_rows(results_dir)
        assert len(rows) == 2
        assert [r["seed"] for r in rows] == [42, 43]
        assert (results_dir / "results.jsonl").exists()

    def test_append_validates_before_write(self, tmp_path: Path) -> None:
        results_dir = tmp_path / "_results"
        bad = make_row()
        bad["stage"] = "2"
        with pytest.raises(SchemaError):
            append_result_row(results_dir, bad)
        assert not (results_dir / "results.jsonl").exists()

    def test_read_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert read_result_rows(tmp_path / "_results") == []

    def test_read_skips_truncated_trailing_line(self, tmp_path: Path) -> None:
        results_dir = tmp_path / "_results"
        results_dir.mkdir(parents=True)
        target = results_dir / "results.jsonl"
        target.write_text(
            json.dumps(make_row(seed=42)) + "\n" + '{"run_id": "crash',
            encoding="utf-8",
        )
        rows = read_result_rows(results_dir)
        assert len(rows) == 1
        assert rows[0]["seed"] == 42

    def test_read_raises_on_mid_file_corruption(self, tmp_path: Path) -> None:
        results_dir = tmp_path / "_results"
        results_dir.mkdir(parents=True)
        target = results_dir / "results.jsonl"
        target.write_text(
            json.dumps(make_row(seed=42))
            + "\n"
            + '{"run_id": "broken"'
            + "\n"
            + json.dumps(make_row(seed=44))
            + "\n",
            encoding="utf-8",
        )
        with pytest.raises(json.JSONDecodeError):
            read_result_rows(results_dir)


# ---------------------------------------------------------------------------
# Run ledger
# ---------------------------------------------------------------------------


class TestLedger:
    def _row(self, run_id: str, **overrides) -> LedgerRow:
        fields = dict(
            run_id=run_id,
            kind="eval",
            timestamp="2026-01-01_00-00-00",
            model="precision_residual_phaseforge",
            config_hash="f89790f7520ddfdb",
            git_sha="c0e72de",
            status="pending",
            path=f"outputs/precision_residual_phaseforge/stage2/2026-01-01_00-00-00_{run_id}",
            stage=2,
            seed=42,
        )
        fields.update(overrides)
        return LedgerRow(**fields)

    def test_append_and_read(self, tmp_path: Path) -> None:
        ledger = RunLedger(tmp_path / "_ledger")
        ledger.append(self._row("aaaaaaaa"))
        ledger.append(self._row("bbbbbbbb", seed=43))
        rows = ledger.read_all()
        assert [r.run_id for r in rows] == ["aaaaaaaa", "bbbbbbbb"]
        assert rows[0].created_at  # auto-stamped
        assert rows[1].seed == 43

    def test_append_rejects_invalid_kind(self, tmp_path: Path) -> None:
        ledger = RunLedger(tmp_path / "_ledger")
        with pytest.raises(ValueError, match="kind"):
            ledger.append(self._row("aaaaaaaa", kind="rollout"))

    def test_append_rejects_invalid_status(self, tmp_path: Path) -> None:
        ledger = RunLedger(tmp_path / "_ledger")
        with pytest.raises(ValueError, match="status"):
            ledger.append(self._row("aaaaaaaa", status="running"))

    def test_update_status_rewrites(self, tmp_path: Path) -> None:
        ledger = RunLedger(tmp_path / "_ledger")
        ledger.append(self._row("aaaaaaaa"))
        ledger.update_status("aaaaaaaa", "completed")
        rows = ledger.read_all()
        assert len(rows) == 1
        assert rows[0].status == "completed"
        assert ledger.find_by_id("aaaaaaaa").status == "completed"

    def test_update_status_unknown_raises(self, tmp_path: Path) -> None:
        ledger = RunLedger(tmp_path / "_ledger")
        ledger.append(self._row("aaaaaaaa"))
        with pytest.raises(KeyError, match="deadbeef"):
            ledger.update_status("deadbeef", "completed")

    def test_update_stage_rewrites(self, tmp_path: Path) -> None:
        ledger = RunLedger(tmp_path / "_ledger")
        ledger.append(self._row("aaaaaaaa", kind="eval", stage=1))
        ledger.update_stage("aaaaaaaa", 2)
        rows = ledger.read_all()
        assert len(rows) == 1
        assert rows[0].stage == 2
        assert ledger.find_by_id("aaaaaaaa").stage == 2

    def test_update_stage_unknown_raises(self, tmp_path: Path) -> None:
        ledger = RunLedger(tmp_path / "_ledger")
        ledger.append(self._row("aaaaaaaa"))
        with pytest.raises(KeyError, match="deadbeef"):
            ledger.update_stage("deadbeef", 2)

    def test_update_stage_rejects_non_int(self, tmp_path: Path) -> None:
        ledger = RunLedger(tmp_path / "_ledger")
        ledger.append(self._row("aaaaaaaa"))
        with pytest.raises(ValueError, match="stage"):
            ledger.update_stage("aaaaaaaa", "2")  # type: ignore[arg-type]

    def test_flush_builds_index(self, tmp_path: Path) -> None:
        ledger = RunLedger(tmp_path / "_ledger")
        ledger.append(self._row("aaaaaaaa"))
        ledger.flush()
        index = json.loads((tmp_path / "_ledger" / "index.json").read_text())
        assert index["count"] == 1
        assert index["runs"][0]["run_id"] == "aaaaaaaa"

    def test_index_throttle_rebuilds_on_append(self, tmp_path: Path) -> None:
        ledger = RunLedger(tmp_path / "_ledger", index_throttle=2)
        ledger.append(self._row("aaaaaaaa"))
        assert not (tmp_path / "_ledger" / "index.json").exists()
        ledger.append(self._row("bbbbbbbb"))
        assert (tmp_path / "_ledger" / "index.json").exists()

    def test_missing_ledger_reads_empty(self, tmp_path: Path) -> None:
        assert RunLedger(tmp_path / "_ledger").read_all() == []

    def test_read_tolerates_truncated_trailing_line(self, tmp_path: Path) -> None:
        ledger = RunLedger(tmp_path / "_ledger")
        ledger.append(self._row("aaaaaaaa"))
        with open(ledger.jsonl_path, "a", encoding="utf-8") as f:
            f.write('{"run_id": "crash')
        rows = ledger.read_all()
        assert [r.run_id for r in rows] == ["aaaaaaaa"]

    def test_status_update_persists_after_reopen(self, tmp_path: Path) -> None:
        ledger = RunLedger(tmp_path / "_ledger")
        ledger.append(self._row("aaaaaaaa"))
        ledger.update_status("aaaaaaaa", "failed")
        reopened = RunLedger(tmp_path / "_ledger")
        assert reopened.find_by_id("aaaaaaaa").status == "failed"

    def test_update_status_tolerates_truncated_trailing_line(self, tmp_path: Path) -> None:
        """A crash mid-append must not block the status update or drop rows."""
        ledger = RunLedger(tmp_path / "_ledger")
        ledger.append(self._row("aaaaaaaa"))
        with open(ledger.jsonl_path, "a", encoding="utf-8") as f:
            f.write('{"run_id": "crash')
        ledger.update_status("aaaaaaaa", "completed")
        rows = ledger.read_all()
        assert [r.run_id for r in rows] == ["aaaaaaaa"]
        assert rows[0].status == "completed"

    def test_update_status_refuses_to_rewrite_corrupted_ledger(self, tmp_path: Path) -> None:
        """Mid-file corruption must surface (not be silently dropped) when the
        rewrite path runs — otherwise a corrupt row would be lost forever."""
        ledger = RunLedger(tmp_path / "_ledger")
        ledger.append(self._row("aaaaaaaa"))
        ledger.append(self._row("bbbbbbbb"))
        with open(ledger.jsonl_path, encoding="utf-8") as f:
            text = f.read()
        ledger.jsonl_path.write_text('{"run_id": "broken"\n' + text, encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            ledger.update_status("aaaaaaaa", "completed")
        # The source of truth is untouched — no silent data loss.
        assert '{"run_id": "broken"' in ledger.jsonl_path.read_text()


# ---------------------------------------------------------------------------
# RunWriter
# ---------------------------------------------------------------------------


class TestRunWriter:
    def _run_dir(self, tmp_path: Path, name: str = "2026-01-01_00-00-00_a1b2c3d4") -> Path:
        run_dir = tmp_path / name
        run_dir.mkdir(parents=True)
        return run_dir

    def test_init_requires_existing_dir(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            RunWriter(tmp_path / "nope")

    def test_init_writes_pending_marker(self, tmp_path: Path) -> None:
        run_dir = self._run_dir(tmp_path)
        writer = RunWriter(run_dir)
        assert writer.status == "pending"
        assert run_dir.with_name(run_dir.name + ".pending").exists()
        assert not run_dir.with_name(run_dir.name + ".completed").exists()

    def test_mark_completed(self, tmp_path: Path) -> None:
        run_dir = self._run_dir(tmp_path)
        writer = RunWriter(run_dir)
        marker = writer.mark_completed()
        assert marker.name.endswith(".completed")
        assert not run_dir.with_name(run_dir.name + ".pending").exists()
        timings = json.loads((run_dir / "timings.json").read_text())
        assert timings["status"] == "completed"
        assert timings["wall_seconds"] >= 0

    def test_mark_failed_writes_exception(self, tmp_path: Path) -> None:
        run_dir = self._run_dir(tmp_path)
        writer = RunWriter(run_dir)
        marker = writer.mark_failed(RuntimeError("boom"))
        assert marker.name.endswith(".failed")
        timings = json.loads((run_dir / "timings.json").read_text())
        assert timings["status"] == "failed"
        assert "boom" in (run_dir / "logs" / "exception.txt").read_text()

    def test_double_mark_raises(self, tmp_path: Path) -> None:
        run_dir = self._run_dir(tmp_path)
        writer = RunWriter(run_dir)
        writer.mark_completed()
        with pytest.raises(RuntimeError, match="already closed"):
            writer.mark_completed()

    def test_write_environment(self, tmp_path: Path) -> None:
        run_dir = self._run_dir(tmp_path)
        writer = RunWriter(run_dir)
        path = writer.write_environment({"git_sha": "abc123"})
        assert json.loads(path.read_text())["git_sha"] == "abc123"

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("2026-01-01_00-00-00_a1b2c3d4", ("2026-01-01_00-00-00", None, "a1b2c3d4")),
            ("2026-01-01_00-00-00_tag01_a1b2c3d4", ("2026-01-01_00-00-00", "tag01", "a1b2c3d4")),
            ("legacy-name", ("legacy-name", None, "")),
        ],
    )
    def test_parse_run_dir(self, name: str, expected) -> None:
        assert parse_run_dir(name) == expected


# ---------------------------------------------------------------------------
# Environment fingerprint
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_collect_environment_shape(self) -> None:
        env = collect_environment(
            data_config_hash="datahash",
            config_hash="confighash",
            extra={"device": "cpu"},
        )
        assert env["python"].startswith("3.")
        assert env["data_config_hash"] == "datahash"
        assert env["config_hash"] == "confighash"
        assert env["extra"] == {"device": "cpu"}
        assert env["git_sha"]
        for pkg in ("torch", "numpy", "hydra-core", "omegaconf", "filelock"):
            assert pkg in env["packages"]

    def test_collect_environment_without_extra(self) -> None:
        env = collect_environment()
        assert "extra" not in env


# ---------------------------------------------------------------------------
# Tables / aggregation
# ---------------------------------------------------------------------------


class TestTables:
    def test_aggregate_rows_groups_and_counts(self) -> None:
        rows = [
            make_result_row(model="precision_residual_phaseforge", stage=2, seed=42, action_mse=0.028),
            make_result_row(model="precision_residual_phaseforge", stage=2, seed=43, action_mse=0.030),
            make_result_row(model="bc", stage=1, seed=42, action_mse=0.027, with_metrics=False),
        ]
        aggs = aggregate_rows(rows)
        assert [(a.model, a.stage) for a in aggs] == [("bc", 1), ("precision_residual_phaseforge", 2)]
        pf = aggs[1]
        assert pf.n_seeds == 2
        assert pf.n_rows == 2
        assert pf.action_mse_mean == pytest.approx(0.029)
        # bc has no routing metrics: NaN mean, n=0.
        bc = aggs[0]
        assert bc.phase_expert_nmi_n == 0
        assert bc.phase_expert_nmi_mean != bc.phase_expert_nmi_mean  # NaN

    def test_aggregate_std_over_seeds(self) -> None:
        rows = [
            make_result_row(model="precision_residual_phaseforge", stage=2, seed=42, action_mse=0.02),
            make_result_row(model="precision_residual_phaseforge", stage=2, seed=43, action_mse=0.04),
            make_result_row(model="precision_residual_phaseforge", stage=2, seed=44, action_mse=0.06),
        ]
        agg = aggregate_rows(rows)[0]
        assert agg.action_mse_mean == pytest.approx(0.04)
        assert agg.action_mse_std == pytest.approx(0.02)

    def test_aggregate_splits_tagged_variants(self) -> None:
        # ``bc`` and ``bc``/``robot_only`` share a model name but must never
        # be merged into one aggregate row (different observation spaces).
        rows = [
            make_result_row(model="bc", stage=1, seed=s, action_mse=0.028, with_metrics=False)
            for s in (42, 43, 44)
        ]
        rows += [
            make_result_row(
                model="bc",
                tag="robot_only",
                stage=1,
                seed=s,
                action_mse=0.020,
                with_metrics=False,
            )
            for s in (42, 43, 44)
        ]
        aggs = aggregate_rows(rows)
        assert [(a.model, a.tag, a.stage) for a in aggs] == [
            ("bc", "", 1),
            ("bc", "robot_only", 1),
        ]
        untagged, tagged = aggs
        assert untagged.n_seeds == 3 and untagged.n_rows == 3
        assert untagged.action_mse_mean == pytest.approx(0.028)
        assert tagged.action_mse_mean == pytest.approx(0.020)
        assert tagged.tag == "robot_only"

    def test_paired_wilcoxon_never_cross_pairs_tags(self) -> None:
        # Tagged and untagged variants of the same model share (stage, seed)
        # keys; the pairing key must include the tag so they cannot pair.
        rows = [
            make_result_row(model="precision_residual_phaseforge", stage=1, seed=s, action_mse=0.02)
            for s in (42, 43, 44)
        ]
        rows += [
            make_result_row(
                model="bc",
                tag="robot_only",
                stage=1,
                seed=s,
                action_mse=0.02,
                with_metrics=False,
            )
            for s in (42, 43, 44)
        ]
        assert (
            paired_wilcoxon(
                rows,
                method_a=("precision_residual_phaseforge", None),
                method_b=("bc", "robot_only"),
                metric="action_mse",
            )
            is None
        )

    def test_wilcoxon_csv_sorts_mixed_tag_none(self, tmp_path: Path) -> None:
        # A model with both a tagged and an untagged variant yields (model,
        # None) and (model, str) identities; sorting them must not raise even
        # when the baseline pairs against one of them.
        rows = [
            make_result_row(model="bc", stage=1, seed=s, action_mse=0.03, with_metrics=False)
            for s in (42, 43, 44)
        ]
        rows += [
            make_result_row(
                model="bc",
                tag="robot_only",
                stage=1,
                seed=s,
                action_mse=0.02,
                with_metrics=False,
            )
            for s in (42, 43, 44)
        ]
        rows += [
            make_result_row(model="precision_residual_scratch_moe", stage=1, seed=s, action_mse=0.05)
            for s in (42, 43, 44)
        ]
        path = write_paired_wilcoxon_csv(rows, tmp_path / "paired_wilcoxon.csv", baseline="bc")
        text = path.read_text()
        assert text.splitlines()[0].startswith("method_a,tag_a,method_b,tag_b,metric")
        assert "precision_residual_scratch_moe" in text

    def test_wilcoxon_csv_pairs_task_tags(self, tmp_path: Path) -> None:
        # Regression: task rows carry the protocol
        # task as their tag. The old hardcoded baseline identity
        # (baseline, None) matched no tagged row, so the CSV was silently
        # empty; the baseline must resolve per tag, and the pairing key must
        # not include stage (PhaseForge final stage 2 vs BC final stage 1).
        import csv as csv_mod

        rows = []
        for task in ("Can", "Square"):
            for seed in (42, 43, 44):
                rows.append(
                    make_result_row(
                        model="precision_residual_phaseforge", tag=task, stage=2, seed=seed, action_mse=0.02
                    )
                )
                rows.append(
                    make_result_row(
                        model="bc",
                        tag=task,
                        stage=1,
                        seed=seed,
                        action_mse=0.03,
                        with_metrics=False,
                    )
                )
        path = write_paired_wilcoxon_csv(
            rows, tmp_path / "paired_wilcoxon.csv", baseline="precision_residual_phaseforge"
        )
        with open(path, newline="", encoding="utf-8") as f:
            records = list(csv_mod.DictReader(f))
        action_rows = [r for r in records if r["metric"] == "action_mse"]
        assert {r["tag_a"] for r in action_rows} == {"Can", "Square"}
        assert all(r["method_a"] == "precision_residual_phaseforge" for r in action_rows)
        assert all(r["method_b"] == "bc" for r in action_rows)
        assert all(int(r["n_pairs"]) == 3 for r in action_rows)
        # A tag the baseline never runs (robot-only style) has no baseline
        # rows, so no comparison is emitted for it.
        rows += [
            make_result_row(
                model="bc",
                tag="Can__robot_only",
                stage=1,
                seed=s,
                action_mse=0.01,
                with_metrics=False,
            )
            for s in (42, 43, 44)
        ]
        path = write_paired_wilcoxon_csv(
            rows, tmp_path / "paired_wilcoxon2.csv", baseline="precision_residual_phaseforge"
        )
        with open(path, newline="", encoding="utf-8") as f:
            records2 = list(csv_mod.DictReader(f))
        assert all(r["tag_b"] in {"Can", "Square"} for r in records2)

    def test_paired_wilcoxon_pairs_across_final_stages(self) -> None:
        # Each method is evaluated at its own final stage (MoE family 2, BC
        # family 1); a stage-locked pairing key would drop every
        # PhaseForge-vs-BC pair, so the key is (seed, tag).
        rows = [
            make_result_row(model="precision_residual_phaseforge", tag="Can", stage=2, seed=s, action_mse=0.02)
            for s in (42, 43, 44)
        ]
        rows += [
            make_result_row(
                model="bc", tag="Can", stage=1, seed=s, action_mse=0.05, with_metrics=False
            )
            for s in (42, 43, 44)
        ]
        record = paired_wilcoxon(
            rows,
            method_a=("precision_residual_phaseforge", "Can"),
            method_b=("bc", "Can"),
            metric="action_mse",
        )
        assert record is not None
        assert record["n_pairs"] == 3
        assert record["mean_diff"] == pytest.approx(-0.03)

    def test_bootstrap_ci_is_deterministic_and_finite(self) -> None:
        values = [0.01, 0.02, 0.03, 0.04, 0.05]
        first = bootstrap_ci(values)
        second = bootstrap_ci(values)
        assert first == second
        mean, lo, hi = first
        assert lo <= mean <= hi
        assert lo > 0

    def test_bootstrap_ci_all_nan(self) -> None:
        mean, lo, hi = bootstrap_ci([float("nan"), float("nan")])
        assert mean != mean  # NaN
        assert lo != lo
        assert hi != hi

    def test_paired_wilcoxon_with_three_seeds(self) -> None:
        rows = [
            make_result_row(model="precision_residual_phaseforge", stage=2, seed=s, action_mse=0.02)
            for s in (42, 43, 44)
        ]
        rows += [
            make_result_row(model="precision_residual_scratch_moe", stage=2, seed=s, action_mse=0.06)
            for s in (42, 43, 44)
        ]
        result = paired_wilcoxon(
            rows,
            method_a=("precision_residual_phaseforge", None),
            method_b=("precision_residual_scratch_moe", None),
            metric="action_mse",
        )
        assert result is not None
        assert result["n_pairs"] == 3
        assert result["p_value"] <= 1.0

    def test_paired_wilcoxon_below_min_pairs(self) -> None:
        rows = [
            make_result_row(model="precision_residual_phaseforge", stage=2, seed=42, action_mse=0.02),
            make_result_row(model="precision_residual_scratch_moe", stage=2, seed=42, action_mse=0.06),
        ]
        assert (
            paired_wilcoxon(
                rows,
                method_a=("precision_residual_phaseforge", None),
                method_b=("precision_residual_scratch_moe", None),
                metric="action_mse",
            )
            is None
        )

    def test_csv_writers_produce_headers(self, tmp_path: Path) -> None:
        rows = [make_result_row(model="precision_residual_phaseforge", stage=2, seed=s) for s in (42, 43, 44)]
        rows += [
            make_result_row(model="precision_residual_scratch_moe", stage=2, seed=s, action_mse=0.06)
            for s in (42, 43, 44)
        ]
        agg_path = write_aggregates_csv(rows, tmp_path / "aggregates.csv")
        boot_path = write_bootstrap_csv(rows, tmp_path / "bootstrap_ci.csv")
        wilcox_path = write_paired_wilcoxon_csv(
            rows, tmp_path / "paired_wilcoxon.csv", baseline="precision_residual_phaseforge"
        )
        agg_text = agg_path.read_text()
        assert agg_text.splitlines()[0].startswith("model,tag,stage,n_seeds")
        assert "precision_residual_phaseforge" in agg_text
        boot_text = boot_path.read_text()
        assert boot_text.splitlines()[0].startswith("model,tag,stage,metric,n,mean")
        assert "action_mse" in boot_text
        wilcox_text = wilcox_path.read_text()
        assert wilcox_text.splitlines()[0].startswith("method_a,tag_a,method_b,tag_b,metric")
        assert "precision_residual_phaseforge" in wilcox_text

    def test_metric_columns_include_action_mse_first(self) -> None:
        assert METRIC_COLUMNS[0] == "action_mse"
        assert set(METRIC_COLUMNS) == {"action_mse", *OPTIONAL_METRIC_FIELDS}


# ---------------------------------------------------------------------------
# Summarize
# ---------------------------------------------------------------------------


class TestSummarize:
    def test_summarize_all_raises_without_results(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="results.jsonl"):
            summarize_all(tmp_path)

    def test_summarize_all_end_to_end(self, tmp_path: Path) -> None:
        results_dir = tmp_path / "_results"
        for seed in (42, 43, 44):
            append_result_row(results_dir, make_row(seed=seed))
        paths = summarize_all(tmp_path)
        assert set(paths) == {"aggregates", "bootstrap", "wilcoxon", "metrics"}
        for name, path in paths.items():
            assert path.exists(), name
        metrics = json.loads(paths["metrics"].read_text())
        assert "precision_residual_phaseforge__stage2__tagdefault" in metrics["summary"]

    def test_summarize_all_is_idempotent(self, tmp_path: Path) -> None:
        results_dir = tmp_path / "_results"
        append_result_row(results_dir, make_row(seed=42))
        first = summarize_all(tmp_path)
        second = summarize_all(tmp_path)
        assert first["aggregates"].read_text() == second["aggregates"].read_text()

    def test_summarize_all_validates_every_row(self, tmp_path: Path) -> None:
        results_dir = tmp_path / "_results"
        results_dir.mkdir(parents=True)
        (results_dir / "results.jsonl").write_text(
            json.dumps(make_row(seed=42)) + "\n" + '{"bad": "row"}\n',
            encoding="utf-8",
        )
        with pytest.raises(SchemaError):
            summarize_all(tmp_path)


# ---------------------------------------------------------------------------
# Backfill migration
# ---------------------------------------------------------------------------


class TestBackfillTags:
    def _write_run_meta(self, outputs: Path, model: str, run_id: str, *, tag: str) -> None:
        run_dir = outputs / model / "stage1" / f"2026-01-01_00-00-00_{run_id}"
        run_dir.mkdir(parents=True)
        (run_dir / "run_meta.json").write_text(
            json.dumps({"tag": tag, "model_name": model, "seed": 42}),
            encoding="utf-8",
        )

    def test_backfill_results_from_eval_run_meta(self, tmp_path: Path) -> None:
        outputs = tmp_path / "outputs"
        # The row's own run_id is the eval run's; its run_meta carries the tag.
        self._write_run_meta(outputs, "bc", "ee000001", tag="robot_only")
        results_dir = tmp_path / "_results"
        results_dir.mkdir(parents=True)
        row = make_row(model="bc", stage=1, tag=None)
        row["run_id"] = "ee000001"
        row["ckpt_path"] = (
            "outputs/bc/stage1/seed42/2026-01-01_00-00-00_aa000001/checkpoints/checkpoint_best.pt"
        )
        (results_dir / "results.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        report = backfill_results(results_dir, collect_run_meta(outputs))
        assert report == {"changed": 1, "rows": 1}
        rows = read_result_rows(results_dir)
        assert rows[0]["tag"] == "robot_only"
        validate_row(rows[0])

    def test_backfill_results_falls_back_to_ckpt_run(self, tmp_path: Path) -> None:
        # Eval run_meta is missing (not copied); the tag must come from the
        # evaluated checkpoint's own run dir instead.
        outputs = tmp_path / "outputs"
        self._write_run_meta(outputs, "bc", "aa000001", tag="robot_only")
        results_dir = tmp_path / "_results"
        results_dir.mkdir(parents=True)
        row = make_row(model="bc", stage=1, tag=None)
        row["run_id"] = "ee000001"
        row["ckpt_path"] = (
            "outputs/bc/stage1/seed42/2026-01-01_00-00-00_aa000001/checkpoints/checkpoint_best.pt"
        )
        (results_dir / "results.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        backfill_results(results_dir, collect_run_meta(outputs))
        rows = read_result_rows(results_dir)
        assert rows[0]["tag"] == "robot_only"

    def test_backfill_leaves_tagged_rows_untouched(self, tmp_path: Path) -> None:
        results_dir = tmp_path / "_results"
        results_dir.mkdir(parents=True)
        (results_dir / "results.jsonl").write_text(
            json.dumps(make_row(tag="robot_only")) + "\n", encoding="utf-8"
        )
        report = backfill_results(results_dir, {})
        assert report == {"changed": 0, "rows": 1}
        assert read_result_rows(results_dir)[0]["tag"] == "robot_only"

    def test_backfill_is_idempotent(self, tmp_path: Path) -> None:
        outputs = tmp_path / "outputs"
        self._write_run_meta(outputs, "bc", "aa000001", tag="robot_only")
        results_dir = tmp_path / "_results"
        results_dir.mkdir(parents=True)
        row = make_row(model="bc", stage=1, tag=None)
        row["ckpt_path"] = (
            "outputs/bc/stage1/seed42/2026-01-01_00-00-00_aa000001/checkpoints/checkpoint_best.pt"
        )
        (results_dir / "results.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        first = backfill_results(results_dir, collect_run_meta(outputs))
        second = backfill_results(results_dir, collect_run_meta(outputs))
        assert first["changed"] == 1
        assert second["changed"] == 0
        assert len(read_result_rows(results_dir)) == 1

    def test_backfill_training_summary(self, tmp_path: Path) -> None:
        from phaseforge.outputs_writer.training_summary import (
            read_training_summary_rows,
        )

        outputs = tmp_path / "outputs"
        run_dir = outputs / "bc" / "stage1" / "2026-01-01_00-00-00_aa000001"
        run_dir.mkdir(parents=True)
        (run_dir / "run_meta.json").write_text(
            json.dumps({"tag": "robot_only", "model_name": "bc", "seed": 42}),
            encoding="utf-8",
        )
        results_dir = tmp_path / "_results"
        results_dir.mkdir(parents=True)
        summary = {
            "run_id": "aa000001",
            "kind": "train",
            "model": "bc",
            "stage": 1,
            "seed": 42,
            "config_hash": "f89790f7520ddfdb",
            "data_config_hash": "cachedatahash",
            "data_provenance_path": "metadata/data_provenance.json",
            "git_sha": "c0e72de",
            "device": "cpu",
            "started_at": "2026-01-01T00:00:00+00:00",
            "finished_at": "2026-01-01T00:01:00+00:00",
            "wall_seconds": 60.0,
            "epochs_run": 5,
            "trainable_params": 100,
            "total_params": 200,
            "best_epoch": 4,
            "final_val": {"loss_total": 0.05, "loss_action": 0.04},
            "extra": {},
        }
        (results_dir / "training_summary.jsonl").write_text(
            json.dumps(summary) + "\n", encoding="utf-8"
        )
        report = backfill_training_summary(results_dir, collect_run_meta(outputs))
        assert report == {"changed": 1, "rows": 1}
        rows = read_training_summary_rows(results_dir)
        assert rows[0]["tag"] == "robot_only"
        assert rows[0]["run_id"] == "aa000001"

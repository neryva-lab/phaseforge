"""PhaseForge outputs package.

Adapted from the ``csd_observer.outputs`` reference (originally from a
change-point-detection benchmark). The reference provides the *mechanisms*
— atomic append-only writes with a cross-process lock, schema-validated
result rows, environment fingerprinting, lifecycle markers, and a global
ledger — re-cast for PhaseForge's per-run layout and offline-evaluation
result schema.

Public API (re-exported below): :class:`RunWriter`, :class:`RunLedger`,
:class:`LedgerRow`, :class:`ResultRow`, :func:`validate_row`,
:func:`collect_environment`, :func:`append_result_row`,
:func:`read_result_rows`, the aggregation and statistical helpers
(``aggregate_rows``, ``bootstrap_ci``, ``paired_wilcoxon`` and their CSV
writers), and :func:`summarize_all`.

Training-side provenance (final specification): the per-run
:class:`TrainingCurveWriter` (curves + summary), the global training
ledger (:func:`append_training_summary_row`,
:func:`reconcile_training_ledger`), the cache-provenance copy and artifact
manifest (:func:`copy_cache_provenance`, :func:`write_artifact_manifest`,
:func:`sha256_file`), and the rollout episode records
(:func:`append_episode_record`, :func:`summarize_episodes`,
:func:`paired_rollout_comparisons`).
"""

from phaseforge.outputs_writer.curves import (
    CURVE_CORE_REQUIRED,
    CURVE_OPTIONAL_NUMERIC,
    SUMMARY_REQUIRED,
    TrainingCurveWriter,
    validate_curve_row,
    validate_summary,
)
from phaseforge.outputs_writer.episodes import (
    EPISODE_REQUIRED,
    append_episode_record,
    paired_rollout_comparisons,
    read_episode_records,
    summarize_episodes,
    validate_episode_record,
    wilson_interval,
)
from phaseforge.outputs_writer.ledger import LedgerRow, RunLedger
from phaseforge.outputs_writer.metadata import collect_environment
from phaseforge.outputs_writer.provenance import (
    copy_cache_provenance,
    sha256_file,
    write_artifact_manifest,
)
from phaseforge.outputs_writer.results import append_result_row, read_result_rows
from phaseforge.outputs_writer.schema import (
    OPTIONAL_METRIC_FIELDS,
    ResultRow,
    SchemaError,
    validate_row,
)
from phaseforge.outputs_writer.summarize import summarize_all
from phaseforge.outputs_writer.tables import (
    aggregate_rows,
    bootstrap_ci,
    paired_wilcoxon,
    write_aggregates_csv,
    write_bootstrap_csv,
    write_paired_wilcoxon_csv,
)
from phaseforge.outputs_writer.training_summaries import (
    read_training_curves,
    summarize_rollout,
    summarize_training,
    training_aggregate_rows,
    training_cost_rows,
)
from phaseforge.outputs_writer.training_summary import (
    append_training_summary_row,
    has_reconciliation_record,
    read_training_summary_rows,
    reconcile_training_ledger,
    reconciliation_record_path,
    write_reconciliation_record,
)
from phaseforge.outputs_writer.writer import RunWriter, parse_run_dir

__all__ = [
    "RunWriter",
    "parse_run_dir",
    "RunLedger",
    "LedgerRow",
    "ResultRow",
    "SchemaError",
    "validate_row",
    "OPTIONAL_METRIC_FIELDS",
    "collect_environment",
    "append_result_row",
    "read_result_rows",
    "aggregate_rows",
    "bootstrap_ci",
    "paired_wilcoxon",
    "write_aggregates_csv",
    "write_bootstrap_csv",
    "write_paired_wilcoxon_csv",
    "summarize_all",
    "CURVE_CORE_REQUIRED",
    "CURVE_OPTIONAL_NUMERIC",
    "SUMMARY_REQUIRED",
    "TrainingCurveWriter",
    "validate_curve_row",
    "validate_summary",
    "append_training_summary_row",
    "read_training_summary_rows",
    "reconcile_training_ledger",
    "write_reconciliation_record",
    "reconciliation_record_path",
    "has_reconciliation_record",
    "copy_cache_provenance",
    "write_artifact_manifest",
    "sha256_file",
    "EPISODE_REQUIRED",
    "validate_episode_record",
    "append_episode_record",
    "read_episode_records",
    "wilson_interval",
    "summarize_episodes",
    "paired_rollout_comparisons",
    "training_aggregate_rows",
    "training_cost_rows",
    "read_training_curves",
    "summarize_training",
    "summarize_rollout",
]

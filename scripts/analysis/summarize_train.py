"""Regenerate paper-ready training + rollout tables from the provenance ledgers.

Reads the schema-validated ``_results/training_summary.jsonl`` (training
side) and every ``eval/**/episodes.jsonl`` (rollout side) and writes:

    * ``training_aggregates.csv`` — per (model, stage) mean / std / n over
      seeds of the final validation scalars, monitor, epochs and params
    * ``training_cost.csv``       — per (model, stage) wall time, epochs,
      total optimizer steps and parameter counts (appendix cost table)
    * ``training_curves.csv``     — per (model, stage, epoch) mean / std / n
      over seeds of every curve metric (plot source)
    * ``rollout_success.csv``     — per (task, model, training seed) success
      rates with Wilson intervals over valid episodes
    * ``rollout_comparisons.csv`` — paired PhaseForge-minus-baseline success
      differences per (task, training seed)

Usage:
    uv run python scripts/analysis/summarize_train.py [--outputs outputs] [--baseline precision_residual_phaseforge]

Requires:
    - At least one completed training run (a non-empty
      ``_results/training_summary.jsonl``). The rollout tables are written
      only when episode records exist. Idempotent — safe to re-run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from phaseforge.outputs_writer.training_summaries import (
    summarize_rollout,
    summarize_training,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("outputs"),
        help="Outputs base directory (default: %(default)s).",
    )
    parser.add_argument(
        "--baseline",
        default="precision_residual_phaseforge",
        help="Reference method for the paired rollout comparisons CSV (default: %(default)s).",
    )
    args = parser.parse_args()

    training_paths = summarize_training(args.outputs)
    rollout_paths = summarize_rollout(args.outputs, baseline=args.baseline)
    print(f"Summaries written under {args.outputs / '_summaries'}:")
    for name, path in training_paths.items():
        print(f"  {name:<22} {path}")
    for name, path in rollout_paths.items():
        print(f"  {name:<22} {path}")


if __name__ == "__main__":
    main()

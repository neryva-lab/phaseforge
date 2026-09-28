"""Regenerate paper-ready aggregate tables from the eval results ledger.

Reads every schema-validated row in ``<outputs>/_results/results.jsonl``
and writes three artifacts under ``<outputs>/_summaries/``:

    * ``aggregates.csv``          — per (model, stage) mean / std / n per metric
    * ``bootstrap_ci.csv``        — percentile bootstrap 95% CIs per row
    * ``paired_wilcoxon.csv``     — one-sided Wilcoxon vs. the baseline method
      (default ``phaseforge``), paired on (stage, seed)

Usage:
    uv run python scripts/analysis/summarize_eval.py [--outputs outputs] [--baseline precision_residual_phaseforge]

Requires:
    - At least one completed ``phaseforge-eval`` run (a non-empty
      ``results.jsonl``). Idempotent — safe to re-run after every eval sweep.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from phaseforge.outputs_writer.summarize import summarize_all


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
        help="Reference method for the paired Wilcoxon CSV (default: %(default)s).",
    )
    args = parser.parse_args()

    paths = summarize_all(args.outputs, baseline=args.baseline)
    print(f"Summaries written under {args.outputs / '_summaries'}:")
    for name, path in paths.items():
        print(f"  {name:<10} {path}")


if __name__ == "__main__":
    main()

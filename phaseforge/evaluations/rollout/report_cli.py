"""``phaseforge-rollout-report`` — aggregate rollout runs into paper tables.

Usage::

    phaseforge-rollout-report [outputs_base] [out_dir]

``outputs_base`` defaults to ``./outputs`` (the project output base);
``out_dir`` defaults to ``{outputs_base}/_results``. Idempotent: re-running
simply re-aggregates every completed rollout eval run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "outputs_base",
        nargs="?",
        default="outputs",
        help="Project output base (default: ./outputs)",
    )
    parser.add_argument(
        "out_dir",
        nargs="?",
        default=None,
        help="Where to write the CSVs (default: {outputs_base}/_results)",
    )
    args = parser.parse_args(argv)

    from phaseforge.evaluations.rollout.report import build_rollout_report

    report = build_rollout_report(
        Path(args.outputs_base), Path(args.out_dir) if args.out_dir else None
    )
    base = Path(args.outputs_base) / "_results"
    print(
        f"Aggregated {report['run_count']} rollout run(s), "
        f"{report['episode_count']} episodes -> {base}"
    )
    for skipped in report["skipped_runs"]:
        print(f"  skipped: {skipped}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

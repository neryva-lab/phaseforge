"""Stage F confirmation gate: Can-first, Square-only-if-decisive.

Decides whether the Can evidence justifies expanding to Square. The rule
is strict and Wilson-based: proceed iff the
IS-PhaseForge success-rate lower 95% bound strictly exceeds every
control's upper 95% bound (no Wilson overlap = decisive).

Usage:
    python -m scripts.analysis.confirm_gate --is eval_results.json \\
        --control bc.json --control pf_direct.json --out gate.json

Exit 0 = proceed to Square; 1 = hold (Can not decisive); 2 = usage error
(missing keys/files). Pure functions below are unit-tested.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SUCCESS_KEY = "eval/rollout/success_rate"
LOW_KEY = "eval/rollout/wilson_ci95_low"
HIGH_KEY = "eval/rollout/wilson_ci95_high"


def decide(
    is_result: dict, controls: list[dict]
) -> tuple[bool, dict]:
    """Apply the decisive-Can rule.

    Returns ``(proceed, report)``. Raises ``KeyError`` when a required
    Wilson key is absent (fail-closed: a gate decision needs real bounds).
    """
    is_low = float(is_result[LOW_KEY])
    is_rate = float(is_result[SUCCESS_KEY])
    control_rows = [
        {
            "success_rate": float(control[SUCCESS_KEY]),
            "wilson_high": float(control[HIGH_KEY]),
        }
        for control in controls
    ]
    best_high = max(row["wilson_high"] for row in control_rows)
    proceed = bool(is_low > best_high)
    report = {
        "proceed_to_square": proceed,
        "is_success_rate": is_rate,
        "is_wilson_low": is_low,
        "best_control_wilson_high": best_high,
        "controls": control_rows,
        "rule": "proceed iff IS Wilson-95 lower bound > every control Wilson-95 upper",
    }
    return proceed, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--is", dest="is_path", required=True, help="IS eval_results.json.")
    parser.add_argument("--control", action="append", required=True, help="Control results.")
    parser.add_argument("--out", default=None, help="Gate JSON path (default: stdout).")
    args = parser.parse_args(argv)
    try:
        is_result = json.loads(Path(args.is_path).read_text(encoding="utf-8"))
        controls = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.control]
        proceed, report = decide(is_result, controls)
    except (OSError, ValueError, KeyError) as exc:
        print(f"confirm_gate ERROR: {exc}", file=sys.stderr)
        return 2
    if args.out is not None:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    else:
        print(json.dumps(report, indent=2))
    if proceed:
        return 0
    print("confirm_gate: HOLD — Can evidence is not decisive.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

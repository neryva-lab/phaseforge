"""Post-hoc tie-break: re-select the stage-1 checkpoint within the monitor
plateau using val/loss_phase as the secondary criterion.

The stage-1 monitor is ``best val/loss_action``; when many epochs form a
flat plateau (within ``plateau_tolerance`` of the best), the phase head at
the selected epoch inherits training-phase variance (seed variance). This
script re-selects the checkpoint *from the same training curve* by the
cheapest, most targeted criterion:

  1. compute the plateau: epochs whose val/loss_action <= best * (1 + tol)
  2. among those epochs pick the one with the lowest val/loss_phase
  3. re-point ``checkpoints/checkpoint_best.pt`` at that epoch's periodic
     snapshot (requires ``train.checkpoint.every_n_epochs: 1`` retention)
  4. record the decision in ``tie_break_selection.json`` next to the run

Pure post-processing: the training algorithm and the eval pipeline are
untouched. Idempotent and re-runnable.

Usage:
  uv run python scripts/analysis/tie_break_selector.py <stage1-run-dir> [--tolerance 0.02]
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path


def load_curve(run_dir: Path) -> list[dict]:
    curve_path = run_dir / "metrics" / "training_curves.jsonl"
    if not curve_path.is_file():
        sys.exit(f"training_curves.jsonl not found under {run_dir}")
    rows = [json.loads(line) for line in curve_path.read_text().splitlines()]
    if not rows:
        sys.exit(f"training_curves.jsonl is empty under {run_dir}")
    return rows


def select_tie_break_epoch(rows: list[dict], tolerance: float) -> dict:
    epochs: list[int] = []
    action_vals: list[float] = []
    phase_vals: list[float] = []
    for r in rows:
        action = r.get("val/loss_action")
        phase = r.get("val/loss_phase")
        if not isinstance(action, (int, float)) or not math.isfinite(float(action)):
            sys.exit("rows lack a finite val/loss_action — cannot apply the tie-break")
        if not isinstance(phase, (int, float)) or not math.isfinite(float(phase)):
            sys.exit("rows lack a finite val/loss_phase — cannot apply the tie-break")
        epochs.append(int(r["epoch"]))
        action_vals.append(float(action))
        phase_vals.append(float(phase))

    best_idx = min(range(len(rows)), key=lambda i: action_vals[i])
    threshold = action_vals[best_idx] * (1.0 + tolerance)
    plateau_idx = [i for i in range(len(rows)) if action_vals[i] <= threshold]
    if len(plateau_idx) < 2:
        tie_idx = best_idx
    else:
        tie_idx = min(plateau_idx, key=lambda i: phase_vals[i])

    return {
        "monitor_epoch": epochs[best_idx],
        "tie_break_epoch": epochs[tie_idx],
        "monitor_val_action": action_vals[best_idx],
        "tie_break_val_action": action_vals[tie_idx],
        "monitor_val_phase": phase_vals[best_idx],
        "tie_break_val_phase": phase_vals[tie_idx],
        "plateau_size": len(plateau_idx),
        "plateau_epochs": [epochs[i] for i in plateau_idx],
        "changed": epochs[tie_idx] != epochs[best_idx],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "run_dir", type=Path, help="stage-1 run directory (training_curves.jsonl + checkpoints/)"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help="plateau tolerance (default 0.02)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the decision without touching checkpoint_best.pt",
    )
    args = parser.parse_args()
    if not 0.0 <= args.tolerance < 1.0:
        sys.exit(f"--tolerance must be in [0, 1), got {args.tolerance}")

    rows = load_curve(args.run_dir)
    decision = select_tie_break_epoch(rows, args.tolerance)

    ckpt_dir = args.run_dir / "checkpoints"
    snapshot = ckpt_dir / f"checkpoint_epoch_{decision['tie_break_epoch']:04d}.pt"
    alias = ckpt_dir / "checkpoint_best.pt"

    if decision["changed"] and not snapshot.is_file():
        sys.exit(
            f"tie-break epoch {decision['tie_break_epoch']} has no periodic snapshot "
            f"({snapshot.name}) — re-run stage-1 with train.checkpoint.every_n_epochs=1"
        )

    print(
        f"monitor best:  epoch {decision['monitor_epoch']:>3}  "
        f"val/action={decision['monitor_val_action']:.4f}  "
        f"val/phase={decision['monitor_val_phase']:.4f}"
    )
    print(
        f"tie-break:     epoch {decision['tie_break_epoch']:>3}  "
        f"val/action={decision['tie_break_val_action']:.4f}  "
        f"val/phase={decision['tie_break_val_phase']:.4f}"
    )
    plateau = f"plateau (tol={args.tolerance}): {decision['plateau_size']} epochs"
    print(f"{plateau} -> changed={decision['changed']}")

    if decision["changed"] and not args.dry_run:
        shutil.copyfile(snapshot, alias)
        print(f"re-pointed {alias.name} <- {snapshot.name}")
    elif decision["changed"] and args.dry_run:
        print(f"[dry-run] would re-point {alias.name} <- {snapshot.name}")

    applied = decision["changed"] and not args.dry_run
    record = {**decision, "tolerance": args.tolerance, "applied": applied}
    out_path = args.run_dir / "tie_break_selection.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2, sort_keys=True)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

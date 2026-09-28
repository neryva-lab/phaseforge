"""CLI tool to find, list, and inspect checkpoints.

Centralised entry point for checkpoint discovery.  Uses the same
:mod:`phaseforge.utils.config` functions that ``phaseforge-train``
and ``phaseforge-eval`` rely on internally, so results are always
consistent.

Usage::

    # Print the path to the latest final PhaseForge Stage 1 checkpoint
    python scripts/utils/find_checkpoint.py --model precision_residual_phaseforge --stage 1

    # List every checkpoint for a model+stage with metadata
    python scripts/utils/find_checkpoint.py --model precision_residual_phaseforge --stage 1 --list

    # List ALL checkpoints across every model and stage
    python scripts/utils/find_checkpoint.py --list-all

    # Validate one or more checkpoint files
    python scripts/utils/find_checkpoint.py --checkpoint path/to/ckpt.pt --validate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── ensure the project is on sys.path ──────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from phaseforge.utils.config import (  # noqa: E402
    CheckpointInfo,
    _project_root,
    find_latest_checkpoint,
    resolve_checkpoint_source,
    scan_checkpoints,
    validate_checkpoint,
)

# ── helpers ────────────────────────────────────────────────────────────────


def _collect_all(base: str) -> list[CheckpointInfo]:
    """Scan ``outputs/`` and return every checkpoint found."""
    root = _project_root() / Path(base)
    if not root.is_dir():
        return []

    results: list[CheckpointInfo] = []
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("."):
            continue
        for stage_dir in sorted(model_dir.iterdir()):
            if not stage_dir.name.startswith("stage"):
                continue
            suffix = stage_dir.name[5:]
            if not suffix.isdigit():
                continue
            stage = int(suffix)
            results.extend(scan_checkpoints(model_dir.name, stage, base))
    return results


def _print_table(checkpoints: list[CheckpointInfo]) -> None:
    """Print a human-readable table of checkpoint metadata."""
    if not checkpoints:
        print("  (none)")
        return

    # Column widths
    w_model = max(len(c.model_name) for c in checkpoints)
    w_model = max(w_model, len("Model"))
    w_stage = 5
    w_ts = 19
    w_run = max(len(c.run_id) for c in checkpoints)
    w_run = max(w_run, len("Run ID"))
    w_tag = 12

    sep = "  "
    hdr = (
        f"{'Model':<{w_model}}{sep}"
        f"{'Stage':<{w_stage}}{sep}"
        f"{'Timestamp':<{w_ts}}{sep}"
        f"{'Run ID':<{w_run}}{sep}"
        f"{'Tag':<{w_tag}}{sep}"
        f"Checkpoint"
    )
    print(hdr)
    print("-" * len(hdr))

    for c in checkpoints:
        tag = c.tag or ""
        print(
            f"{c.model_name:<{w_model}}{sep}"
            f"{c.stage:<{w_stage}}{sep}"
            f"{c.timestamp:<{w_ts}}{sep}"
            f"{c.run_id:<{w_run}}{sep}"
            f"{tag:<{w_tag}}{sep}"
            f"{c.path}"
        )


# ── subcommands ────────────────────────────────────────────────────────────


def cmd_latest(args: argparse.Namespace) -> None:
    """Print the path to the latest checkpoint for *model+stage*."""
    source = resolve_checkpoint_source(args.model) if args.resolve else args.model
    ckpt = find_latest_checkpoint(
        args.model,
        stage=args.stage,
        base=args.base,
        resolve_alias=args.resolve,
    )
    if ckpt is None:
        print(
            f"No checkpoint found for '{args.model}' stage {args.stage} "
            f"(looked under '{source}/stage{args.stage}').",
            file=sys.stderr,
        )
        sys.exit(1)
    print(ckpt)


def cmd_list(args: argparse.Namespace) -> None:
    """List checkpoints for *model+stage* with metadata."""
    source = resolve_checkpoint_source(args.model) if args.resolve else args.model
    cps = scan_checkpoints(source, stage=args.stage, base=args.base)
    if not cps:
        print(
            f"No checkpoints for '{args.model}' stage {args.stage} "
            f"(scanned '{source}/stage{args.stage}').",
            file=sys.stderr,
        )
        sys.exit(1)
    _print_table(cps)


def cmd_list_all(args: argparse.Namespace) -> None:
    """List every checkpoint across all models."""
    cps = _collect_all(args.base)
    if not cps:
        print(f"No checkpoints found under '{args.base}/'.", file=sys.stderr)
        sys.exit(1)
    _print_table(cps)


def cmd_validate(args: argparse.Namespace) -> None:
    """Validate one or more checkpoint paths."""
    all_ok = True
    for ckpt_path in args.checkpoint:
        p = Path(ckpt_path)
        ok = validate_checkpoint(p)
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {p}")
        if not ok:
            all_ok = False
    sys.exit(0 if all_ok else 1)


# ── CLI ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find and inspect PhaseForge checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--base",
        default="outputs",
        help="Base output directory (default: outputs)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # latest
    p_latest = sub.add_parser("latest", help="Print latest checkpoint path")
    p_latest.add_argument("--model", required=True, help="Model name")
    p_latest.add_argument("--stage", type=int, default=1, help="Stage (1 or 2)")
    p_latest.add_argument(
        "--no-resolve",
        dest="resolve",
        action="store_false",
        default=True,
        help="Disable final provider resolution",
    )
    p_latest.set_defaults(func=cmd_latest)

    # list
    p_list = sub.add_parser("list", help="List checkpoints for a model")
    p_list.add_argument("--model", required=True, help="Model name")
    p_list.add_argument("--stage", type=int, default=1, help="Stage (1 or 2)")
    p_list.add_argument(
        "--no-resolve",
        dest="resolve",
        action="store_false",
        default=True,
        help="Disable model alias resolution",
    )
    p_list.set_defaults(func=cmd_list)

    # list-all
    p_all = sub.add_parser("list-all", help="List all checkpoints")
    p_all.set_defaults(func=cmd_list_all)

    # validate
    p_val = sub.add_parser("validate", help="Validate checkpoint file(s)")
    p_val.add_argument(
        "checkpoint",
        nargs="+",
        help="Path(s) to checkpoint_best.pt to validate",
    )
    p_val.set_defaults(func=cmd_validate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

"""Observability audit CLI for regime labels (CPU-only).

Audits regime-label observability from ``x_t`` alone over a processed
cache directory and writes ``observability_report.json``.

Usage:
    python -m scripts.analysis.observability_audit --cache-dir <hash_dir> \\
        --labels phase_topo --out observability_report.json

Exit codes: 0 = audit passed; 1 = audit failed (unobservable regimes);
2 = usage/loading error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def _load_cache_trajectories(cache_dir: Path) -> list[dict]:
    """Load trajectory dicts from a processed cache directory."""
    import torch

    traj_dir = cache_dir / "trajectories"
    if not traj_dir.is_dir():
        raise FileNotFoundError(f"No trajectories/ in cache dir {cache_dir}.")
    files = sorted(traj_dir.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No trajectory files in {traj_dir}.")
    return [torch.load(f, map_location="cpu", weights_only=False) for f in files]


def main(argv: list[str] | None = None) -> int:
    import numpy as np

    from phaseforge.data.topo.observability import audit_regimes

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, help="Processed cache directory.")
    parser.add_argument(
        "--labels",
        default="phase_topo",
        choices=("phase_topo", "phase_dynamic", "phase_rule", "phase"),
        help="Trajectory label field to audit.",
    )
    parser.add_argument("--out", default=None, help="Report JSON path (default: stdout).")
    parser.add_argument("--min-macro-f1", type=float, default=0.6)
    parser.add_argument("--min-occupancy", type=float, default=0.01)
    parser.add_argument(
        "--allow-aliased",
        action="store_true",
        help="Exit 0 despite merge candidates (records them, still reports).",
    )
    args = parser.parse_args(argv)

    cache_dir = Path(args.cache_dir)
    try:
        trajectories = _load_cache_trajectories(cache_dir)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"observability_audit ERROR: {exc}", file=sys.stderr)
        return 2

    def _to_numpy(value) -> np.ndarray:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    states, labels, traj_ids = [], [], []
    num_regimes = 0
    for idx, traj in enumerate(trajectories):
        if args.labels not in traj:
            print(
                f"observability_audit ERROR: trajectory {idx} lacks {args.labels!r}.",
                file=sys.stderr,
            )
            return 2
        states.append(_to_numpy(traj["state"]))
        labels.append(_to_numpy(traj[args.labels]).reshape(-1))
        traj_ids.append(np.full(_to_numpy(traj["state"]).shape[0], idx, dtype=np.int64))
        num_regimes = max(num_regimes, int(np.max(labels[-1])) + 1)
    try:
        report = audit_regimes(
            np.concatenate(states, axis=0),
            np.concatenate(labels, axis=0),
            np.concatenate(traj_ids, axis=0),
            num_regimes,
            min_macro_f1=args.min_macro_f1,
            min_occupancy=args.min_occupancy,
        )
    except ValueError as exc:
        print(f"observability_audit ERROR: {exc}", file=sys.stderr)
        return 2

    payload = report.to_dict()
    if args.out is not None:
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2))
    if report.passed or args.allow_aliased:
        return 0
    print(
        "observability_audit FAILED: " + "; ".join(report.failure_reasons),
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

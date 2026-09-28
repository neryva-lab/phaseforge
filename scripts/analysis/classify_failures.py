"""Failure-taxonomy classifier over full rollout traces.

Maps each traced episode to one class: the pass-through outcomes
(``success``, ``infrastructure``, ``policy_invalid_action``,
``policy_exception``) or, for unsuccessful valid episodes, the first
firing rule below (ordered by diagnostic specificity):

1. ``gain_collapse`` — impedance gains near zero persistently.
2. ``action_saturation`` — pre-clip commands saturated persistently.
3. ``routing_ambiguity`` — small router margin with high entropy (or,
   for hard routing without entropy, a high expert flip rate).
4. ``expert_conflict`` — large top-1 vs top-2 action disagreement.
5. ``ood_drift`` — large nearest-train-state distance.
6. ``target_chasing`` — ``‖T − y‖`` grows over the episode.
7. ``controller_limit`` — valid actions move the end-effector negligibly.
8. ``timeout_unclassified`` — none of the above fired (honest fallback).

``reset_geometry`` is a *cross-run* diagnosis (same case fails in every
run) applied by :func:`summarize_runs`, overriding single-run classes.

Thresholds are proposed defaults, not spec-fixed values. Pure functions
(CPU, deterministic) plus a CLI. Usage:
    python -m scripts.analysis.classify_failures --trace trace.jsonl \\
        --trace other.jsonl --out failure_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

#: Single-run failure classes in rule priority order (plus outcomes).
FAILURE_CLASSES: tuple[str, ...] = (
    "success",
    "infrastructure",
    "policy_invalid_action",
    "policy_exception",
    "gain_collapse",
    "action_saturation",
    "routing_ambiguity",
    "expert_conflict",
    "ood_drift",
    "target_chasing",
    "controller_limit",
    "timeout_unclassified",
    "reset_geometry",
)


@dataclass(frozen=True)
class Thresholds:
    """Proposed-default thresholds (not spec-fixed)."""

    gain_eps: float = 1e-3
    saturated_u: float = 3.0
    clip_frac: float = 0.8
    margin_tau: float = 0.1
    entropy_hi: float = 0.5
    flip_rate_hi: float = 0.3
    disagreement_hi: float = 0.5
    ood_dist: float = 6.0
    chase_growth: float = 0.5
    still_eps: float = 1e-3


def _present(values: list) -> list[float]:
    return [float(v) for v in values if v is not None]


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _slope(lengths: list[float]) -> float:
    """Least-squares slope of values over step order (deterministic)."""
    count = len(lengths)
    if count < 2:
        return 0.0
    mean_x = (count - 1) / 2.0
    mean_y = sum(lengths) / count
    denom = sum((i - mean_x) ** 2 for i in range(count))
    if denom == 0.0:
        return 0.0
    return sum((i - mean_x) * (y - mean_y) for i, y in enumerate(lengths)) / denom


def classify_episode(rows: list[dict], thresholds: Thresholds | None = None) -> str:
    """Classify one episode's trace rows (in timestep order)."""
    if not rows:
        return "timeout_unclassified"
    thresh = thresholds or Thresholds()
    termination = str(rows[-1].get("termination_reason") or "")
    if termination == "success":
        return "success"
    if termination == "infrastructure":
        return "infrastructure"
    if termination in ("policy_invalid_action", "policy_exception"):
        return termination

    gains = [
        row.get("expert_gains") for row in rows if isinstance(row.get("expert_gains"), list)
    ]
    if gains:
        per_step = [sum(abs(v) for v in g) / len(g) for g in gains if g]
        if per_step and (_median(per_step) or 0.0) < thresh.gain_eps:
            return "gain_collapse"

    commands = [row.get("pre_clip_command") for row in rows]
    listed = [cmd for cmd in commands if isinstance(cmd, list)]
    if listed:
        saturated = sum(1 for cmd in listed if any(abs(v) > thresh.saturated_u for v in cmd))
        if len(rows) > 0 and saturated / len(rows) > thresh.clip_frac:
            return "action_saturation"

    margins = _present([row.get("router_margin") for row in rows])
    entropies = _present([row.get("router_entropy") for row in rows])
    selected = [row.get("selected_expert") for row in rows]
    if margins:
        small_margin = (_median(margins) or 0.0) < thresh.margin_tau
        if small_margin:
            if entropies:
                if (_median(entropies) or 0.0) > thresh.entropy_hi:
                    return "routing_ambiguity"
            else:
                flips = sum(
                    1 for prev, cur in zip(selected, selected[1:])
                    if prev is not None and cur is not None and prev != cur
                )
                denom = max(1, len(selected) - 1)
                if flips / denom > thresh.flip_rate_hi:
                    return "routing_ambiguity"

    disagreements = _present([row.get("expert_disagreement") for row in rows])
    if disagreements and (_median(disagreements) or 0.0) > thresh.disagreement_hi:
        return "expert_conflict"

    ood = _present([row.get("nearest_train_dist") for row in rows])
    if ood and (_median(ood) or 0.0) > thresh.ood_dist:
        return "ood_drift"

    targets = [row.get("expert_target") for row in rows]
    task_vars = [row.get("task_vars") for row in rows]
    if all(t is not None for t in targets) and all(v is not None for v in task_vars):
        import math

        gaps = [
            math.dist(t, v) for t, v in zip(targets, task_vars)  # type: ignore[arg-type]
        ]
        if _slope(gaps) > 0.0 and (gaps[-1] - gaps[0]) > thresh.chase_growth:
            return "target_chasing"

    eef = [
        row.get("raw_obs_summary", {}).get("eef_pos")
        if isinstance(row.get("raw_obs_summary"), dict)
        else None
        for row in rows
    ]
    if all(isinstance(p, list) and len(p) == 3 for p in eef):
        import math

        steps = [
            math.dist(a, b) for a, b in zip(eef, eef[1:])  # type: ignore[arg-type]
        ]
        if steps and (_median(steps) or 0.0) < thresh.still_eps:
            return "controller_limit"

    return "timeout_unclassified"


def summarize_runs(runs: dict[str, list[dict]]) -> dict:
    """Classify every episode in every run; detect cross-run reset geometry.

    Args:
        runs: Mapping of run label to that run's trace rows.

    Returns:
        ``{"episodes": [...], "counts": {...}, "reset_geometry_cases": [...]}``.
        Cases failing (non-success) in *every* run with >= 2 runs are
        overridden to ``reset_geometry``.
    """
    episodes: list[dict] = []
    by_case: dict[int, list[str]] = {}
    for label, rows in runs.items():
        grouped: dict[int, list[dict]] = {}
        for row in rows:
            grouped.setdefault(int(row.get("episode_id", -1)), []).append(row)
        for episode_id, episode_rows in sorted(grouped.items()):
            episode_rows.sort(key=lambda r: int(r.get("timestep", 0)))
            first = episode_rows[0]
            classes = classify_episode(episode_rows)
            record = {
                "run": label,
                "episode_id": int(episode_id),
                "case_id": first.get("case_id"),
                "termination": episode_rows[-1].get("termination_reason"),
                "class": classes,
            }
            episodes.append(record)
            if isinstance(first.get("case_id"), int):
                by_case.setdefault(int(first["case_id"]), []).append(classes)
    reset_geometry_cases: list[int] = []
    if len(runs) >= 2:
        for case_id, classes in by_case.items():
            if len(classes) == len(runs) and all(c != "success" for c in classes):
                reset_geometry_cases.append(case_id)
                for record in episodes:
                    if record["case_id"] == case_id:
                        record["class"] = "reset_geometry"
    counts: dict[str, int] = {}
    for record in episodes:
        counts[record["class"]] = counts.get(record["class"], 0) + 1
    return {
        "episodes": episodes,
        "counts": counts,
        "reset_geometry_cases": sorted(reset_geometry_cases),
    }


def main(argv: list[str] | None = None) -> int:
    from phaseforge.evaluations.rollout.trace import read_trace_rows

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", action="append", required=True, help="trace.jsonl path.")
    parser.add_argument("--out", default=None, help="Report JSON path (default: stdout).")
    args = parser.parse_args(argv)
    runs: dict[str, list[dict]] = {}
    for path in args.trace:
        rows = read_trace_rows(path)
        if not rows:
            print(f"classify_failures ERROR: no trace rows in {path}.", file=sys.stderr)
            return 2
        runs[str(path)] = rows
    report = summarize_runs(runs)
    if args.out is not None:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    else:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

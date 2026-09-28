"""Seed-stratified bootstrap statistics for the rollout evaluation.

Replaces the pooled-episode Wilson interval, which is pseudoreplication:
episodes from the same training seed share one trained policy and are
correlated. Here we resample *seeds* with replacement (Agarwal et al. 2021,
applied at the seed level, N = 3 seeds) and
report, **per task** (Lift / Can / Square / ToolHang / Transport — never
pooled across tasks):

  - per-method seed-level means + percentile bootstrap 95% CIs
  - a pairwise probability-of-improvement matrix P(X > Y) over the same
    resamples (ties counted as 0.5)

Usage:
    python scripts/analysis/stratified_stats.py [--root outputs] [--json out.json]

Every episodes.jsonl under <root>/**/eval/<method>/seed*/<run>/ is loaded;
grouping uses the explicit ``task``, ``model`` and ``training_seed`` fields
so the directory layout is not trusted. (The grouping key includes ``task``:
grouping on (model, seed) alone would pool the five-task matrix into
cross-task averages.)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

N_RESAMPLES_DEFAULT = 100_000
RNG_SEED_DEFAULT = 12345


def load_episodes(roots: list[Path]) -> dict[tuple[str, str, int], list[bool]]:
    """Load per-episode success booleans grouped by (task, model, training_seed)."""
    groups: dict[tuple[str, str, int], list[bool]] = defaultdict(list)
    found = 0
    for root in roots:
        if not root.exists():
            sys.exit(f"error: root does not exist: {root}")
        for path in root.rglob("episodes.jsonl"):
            found += 1
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if not row.get("valid_episode", True):
                    continue
                task = row.get("task") or "unknown"
                groups[(task, row["model"], row["training_seed"])].append(
                    bool(row["success"])
                )
    if found == 0:
        sys.exit(f"error: no episodes.jsonl found under {roots}")
    return dict(groups)


def seed_means(
    groups: dict[tuple[str, str, int], list[bool]],
) -> dict[str, dict[str, dict[int, float]]]:
    """Nested per-task dict of method -> training_seed -> success rate."""
    tasks: dict[str, dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for (task, model, seed), outcomes in groups.items():
        if not outcomes:
            continue
        tasks[task][model][seed] = sum(outcomes) / len(outcomes)
    return {task: dict(methods) for task, methods in tasks.items()}


def bootstrap_seed_means(
    seed_rates: dict[int, float],
    n_resamples: int,
    rng: random.Random,
) -> list[float]:
    """Draw ``n_resamples`` seed-level resamples of the estimator mean(p_s)."""
    seeds = list(seed_rates)
    rates = [seed_rates[s] for s in seeds]
    means: list[float] = []
    for _ in range(n_resamples):
        drawn = [rates[rng.randrange(len(rates))] for _ in rates]
        means.append(sum(drawn) / len(drawn))
    return means


def percentile_ci(samples: list[float], alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided percentile interval over a sorted list of bootstrap draws."""
    ordered = sorted(samples)
    n = len(ordered)
    lo = ordered[int(round(n * alpha / 2)) - 1]
    hi = ordered[int(round(n * (1 - alpha / 2))) - 1]
    return lo, hi


def exact_bootstrap_seed_means(seed_rates: dict[int, float]) -> list[float]:
    """Exact distribution of the seed-mean estimator: all n^n draws, equally likely.

    Used in tests to check the Monte Carlo approximation (with n = 3 seeds
    there are only 27 distinct draw vectors).
    """
    rates = list(seed_rates.values())
    n = len(rates)
    means: list[float] = []
    tally = [0] * n

    def visit(pos: int) -> None:
        if pos == n:
            means.append(sum(rates[i] * c for i, c in enumerate(tally)) / n)
            return
        for i in range(n):
            tally[i] += 1
            visit(pos + 1)
            tally[i] -= 1

    visit(0)
    return means


def probability_of_improvement(
    x_rates: dict[int, float],
    y_rates: dict[int, float],
    n_resamples: int,
    rng: random.Random,
) -> float:
    """P(X > Y) under independent seed resampling; ties count 0.5."""
    x_seeds = list(x_rates)
    y_seeds = list(y_rates)
    x_vals = [x_rates[s] for s in x_seeds]
    y_vals = [y_rates[s] for s in y_seeds]
    better = ties = 0
    for _ in range(n_resamples):
        x = sum(x_vals[rng.randrange(len(x_vals))] for _ in x_seeds) / len(x_seeds)
        y = sum(y_vals[rng.randrange(len(y_vals))] for _ in y_seeds) / len(y_seeds)
        if x > y:
            better += 1
        elif x == y:
            ties += 1
    return (better + 0.5 * ties) / n_resamples


def format_table(
    methods: dict[str, dict[int, float]],
    cis: dict[str, tuple[float, float]],
    n_resamples: int,
) -> str:
    header = (
        f"{'method':<34}{'s42':>7}{'s43':>7}{'s44':>7}{'seed mean':>10}"
        f"{'strat. bootstrap 95% CI':>26}"
    )
    rows = [header, "-" * len(header)]
    for method in sorted(methods):
        rates = methods[method]
        seeds = " ".join(f"{rates.get(s, float('nan')):>6.3f}" for s in (42, 43, 44))
        mean = sum(rates.values()) / len(rates)
        lo, hi = cis[method]
        rows.append(
            f"{method:<34}{seeds}{mean:>9.3f}  [{lo:.3f}, {hi:.3f}]  "
            f"(bootstrap n={n_resamples})"
        )
    return "\n".join(rows)


def format_poi_matrix(
    methods: dict[str, dict[int, float]],
    poi: dict[tuple[str, str], float],
    n_resamples: int,
) -> str:
    names = sorted(methods)
    header = f"{'P(X > Y)':<16}" + "".join(f"{n[:16]:>17}" for n in names)
    rows = [header, "-" * len(header)]
    for x in names:
        row = f"{x[:16]:<16}"
        for y in names:
            if x == y:
                row += f"{'—':>17}"
            else:
                row += f"{poi[(x, y)] * 100:>16.1f}%"
        rows.append(row)
    rows.append(f"(bootstrap n={n_resamples}; ties count as 0.5)")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        # Use default=None + post-processing, not default=["outputs"] with
        # action="append": the latter would append to the list default and
        # silently mix pre-final rows into an explicit --root scan,
        # violating the namespace invariant.
        "--root",
        action="append",
        default=None,
        type=Path,
    )
    parser.add_argument("--n-resamples", type=int, default=N_RESAMPLES_DEFAULT)
    parser.add_argument("--rng-seed", type=int, default=RNG_SEED_DEFAULT)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    args.root = [Path(r) for r in (args.root or ["outputs"])]

    groups = load_episodes(args.root)
    tasks = seed_means(groups)
    if not tasks:
        sys.exit("error: no (task, model, training_seed) groups with outcomes found")

    rng = random.Random(args.rng_seed)
    payload_tasks: dict[str, dict[str, object]] = {}
    for task in sorted(tasks):
        methods = tasks[task]
        cis: dict[str, tuple[float, float]] = {}
        poi: dict[tuple[str, str], float] = {}
        for method, rates in methods.items():
            draws = bootstrap_seed_means(rates, args.n_resamples, rng)
            cis[method] = percentile_ci(draws)
        for x in methods:
            for y in methods:
                if x == y:
                    continue
                poi[(x, y)] = probability_of_improvement(
                    methods[x], methods[y], args.n_resamples, rng
                )

        print(f"=== task: {task} ({len(methods)} method(s)) ===")
        print(format_table(methods, cis, args.n_resamples))
        print()
        print(format_poi_matrix(methods, poi, args.n_resamples))
        print()

        payload_tasks[task] = {
            "methods": {
                m: {
                    "seed_rates": {str(s): r for s, r in rates.items()},
                    "seed_mean": sum(rates.values()) / len(rates),
                    "bootstrap_ci": list(cis[m]),
                }
                for m, rates in methods.items()
            },
            "probability_of_improvement": {
                f"{x} > {y}": poi[(x, y)] for (x, y) in poi
            },
        }

    if args.json is not None:
        payload = {
            "rng_seed": args.rng_seed,
            "n_resamples": args.n_resamples,
            "tasks": payload_tasks,
        }
        args.json.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
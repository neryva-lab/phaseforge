"""Tests for the seed-stratified bootstrap statistics, grouped by task."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from scripts.analysis.stratified_stats import (
    bootstrap_seed_means,
    exact_bootstrap_seed_means,
    format_poi_matrix,
    format_table,
    load_episodes,
    main,
    percentile_ci,
    probability_of_improvement,
    seed_means,
)


def _write_episodes(
    root: Path, model: str, seed: int, successes: list[bool], task: str = "Lift", run: str = "run_1"
) -> None:
    run_dir = root / "eval" / model / f"seed{seed}" / run
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, ok in enumerate(successes):
        lines.append(
            "{"
            f'"task": "{task}", "model": "{model}", "training_seed": {seed}, '
            f'"episode_index": {i}, "success": {str(ok).lower()}, '
            f'"valid_episode": true'
            "}"
        )
    (run_dir / "episodes.jsonl").write_text("\n".join(lines), encoding="utf-8")


def test_load_and_group_by_task_model_and_seed(tmp_path: Path) -> None:
    _write_episodes(tmp_path, "alpha", 42, [True, False, True])
    _write_episodes(tmp_path, "alpha", 43, [True, True])
    _write_episodes(tmp_path, "beta", 42, [False, False])
    groups = load_episodes([tmp_path])
    assert groups[("Lift", "alpha", 42)] == [True, False, True]
    assert groups[("Lift", "alpha", 43)] == [True, True]
    assert groups[("Lift", "beta", 42)] == [False, False]


def test_tasks_are_never_pooled(tmp_path: Path) -> None:
    # Regression: the same method run on two tasks must stay in separate
    # groups — the old (model, seed) key averaged across tasks into one
    # meaningless number.
    _write_episodes(tmp_path, "alpha", 42, [True] * 10, task="Lift", run="lift_run")
    _write_episodes(tmp_path, "alpha", 42, [False] * 10, task="Can", run="can_run")
    groups = load_episodes([tmp_path])
    assert groups[("Lift", "alpha", 42)] == [True] * 10
    assert groups[("Can", "alpha", 42)] == [False] * 10
    methods = seed_means(groups)
    assert methods["Lift"]["alpha"][42] == 1.0
    assert methods["Can"]["alpha"][42] == 0.0


def test_main_reports_per_task_json(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    for task, outcomes in (("Can", [True] * 6), ("Square", [False] * 6)):
        _write_episodes(
            tmp_path,
            "precision_residual_phaseforge",
            42,
            outcomes,
            task=task,
            run=f"{task.lower()}_run",
        )
        _write_episodes(tmp_path, "bc", 42, outcomes, task=task, run=f"{task.lower()}_run")
    out = tmp_path / "stats.json"
    code = main(["--root", str(tmp_path), "--json", str(out), "--n-resamples", "2000"])
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert set(payload["tasks"]) == {"Can", "Square"}
    assert (
        payload["tasks"]["Can"]["methods"]["precision_residual_phaseforge"]["seed_rates"]["42"]
        == 1.0
    )
    assert (
        payload["tasks"]["Square"]["methods"]["precision_residual_phaseforge"]["seed_rates"]["42"]
        == 0.0
    )
    printed = capsys.readouterr().out
    assert "task: Can" in printed and "task: Square" in printed


def test_invalid_episodes_excluded(tmp_path: Path) -> None:
    run_dir = tmp_path / "eval" / "alpha" / "seed42" / "run_1"
    run_dir.mkdir(parents=True)
    (run_dir / "episodes.jsonl").write_text(
        '{"task": "Lift", "model": "alpha", "training_seed": 42, "success": true, '
        '"valid_episode": false}\n'
        '{"task": "Lift", "model": "alpha", "training_seed": 42, "success": false, '
        '"valid_episode": true}\n',
        encoding="utf-8",
    )
    groups = load_episodes([tmp_path])
    assert groups[("Lift", "alpha", 42)] == [False]


def test_missing_root_exits(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        load_episodes([tmp_path / "nope"])


def test_seed_means_are_unweighted_across_unequal_counts(tmp_path: Path) -> None:
    _write_episodes(tmp_path, "alpha", 42, [True] * 5 + [False] * 5)
    _write_episodes(tmp_path, "alpha", 43, [True, False])
    methods = seed_means(load_episodes([tmp_path]))
    assert methods["Lift"]["alpha"][42] == 0.5
    assert methods["Lift"]["alpha"][43] == 0.5


def test_mc_bootstrap_matches_exact_distribution() -> None:
    from collections import Counter

    rates = {42: 0.60, 43: 0.48, 44: 0.54}
    # Bin values at 1e-9 before counting: the exact enumerator and the MC
    # sampler compute the same means in different summation orders, so e.g.
    # 0.56 materializes as BOTH 0.56 and 0.5599999999999999 — without
    # quantization the MC mass splits across float neighbors of one exact
    # value and the 4-sigma comparison fails on an artifact, not statistics.
    exact = Counter(round(v, 9) for v in exact_bootstrap_seed_means(rates))
    n_exact = sum(exact.values())
    n_mc = 200_000
    mc = Counter(round(v, 9) for v in bootstrap_seed_means(rates, n_mc, random.Random(7)))
    for value, count in exact.items():
        p_exact = count / n_exact
        p_mc = mc[value] / n_mc
        sigma = (p_exact * (1 - p_exact) / n_mc) ** 0.5
        assert abs(p_mc - p_exact) < 4 * sigma + 1e-9


def test_percentile_ci_contains_estimate_and_is_monotone() -> None:
    rates = {42: 0.60, 43: 0.48, 44: 0.54}
    draws = bootstrap_seed_means(rates, 100_000, random.Random(7))
    lo, hi = percentile_ci(draws)
    mean = sum(rates.values()) / 3
    assert lo <= mean <= hi
    assert lo < hi


def test_percentile_ci_degenerates_for_single_seed() -> None:
    draws = bootstrap_seed_means({42: 0.5}, 1_000, random.Random(7))
    lo, hi = percentile_ci(draws)
    assert lo == hi == 0.5


def test_poi_direction_and_scale() -> None:
    rng = random.Random(7)
    strong = {42: 0.9, 43: 0.9, 44: 0.9}
    weak = {42: 0.5, 43: 0.5, 44: 0.5}
    p = probability_of_improvement(strong, weak, 10_000, rng)
    assert p > 0.99
    q = probability_of_improvement(weak, strong, 10_000, rng)
    assert q < 0.01
    assert p + q == 1.0


def test_poi_ties_count_half() -> None:
    same = {42: 0.6, 43: 0.5, 44: 0.5}
    rng = random.Random(7)
    p = probability_of_improvement(same, same, 10_000, rng)
    q = probability_of_improvement(same, same, 10_000, random.Random(7))
    assert 0.45 < p < 0.5  # iid draws: P(X>Y) = (1 - P(X=Y))/2 < 0.5
    assert p == q


def test_determinism() -> None:
    rates = {42: 0.60, 43: 0.48, 44: 0.54}
    a = bootstrap_seed_means(rates, 5_000, random.Random(99))
    b = bootstrap_seed_means(rates, 5_000, random.Random(99))
    assert a == b


def test_table_and_matrix_smoke(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    _write_episodes(tmp_path, "alpha", 42, [True] * 5)
    _write_episodes(tmp_path, "alpha", 43, [False] * 5)
    _write_episodes(tmp_path, "alpha", 44, [True] * 5)
    _write_episodes(tmp_path, "beta", 42, [False] * 5)
    _write_episodes(tmp_path, "beta", 43, [False] * 5)
    _write_episodes(tmp_path, "beta", 44, [False] * 5)
    groups = load_episodes([tmp_path])
    methods = seed_means(groups)["Lift"]
    draws = {m: bootstrap_seed_means(r, 1_000, random.Random(7)) for m, r in methods.items()}
    cis = {m: percentile_ci(d) for m, d in draws.items()}
    poi = {
        ("alpha", "beta"): probability_of_improvement(
            methods["alpha"], methods["beta"], 1_000, random.Random(7)
        ),
        ("beta", "alpha"): probability_of_improvement(
            methods["beta"], methods["alpha"], 1_000, random.Random(7)
        ),
    }
    table = format_table(methods, cis, 1_000)
    matrix = format_poi_matrix(methods, poi, 1_000)
    assert "alpha" in table and "0.667" in table
    # alpha (seed means 1/0/1) vs beta (0/0/0): P(alpha > 0) = 1 - (1/3)^3 = 26/27
    assert abs(poi[("alpha", "beta")] - 26 / 27) < 0.02
    assert poi[("beta", "alpha")] < 0.05
    assert "—" in matrix

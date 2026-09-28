"""Reset-path determinism gate (standing regression harness).

Gates the ADOPTED reset path of the rollout protocol: the canonicalized
soft branch (``hard_reset=False`` + hidden-state canonicalization +
deterministic construction seeding, ``robosuite_adapter.py``). The protocol
property under test is **within-path bitwise determinism**: the same frozen
reset case must produce a bitwise-identical trajectory regardless of

  * episode history on the same environment (other cases rolled in
    between), and
* environment construction (a second adapter built at a different
    global-RNG position — standing in for a different process/method).

Both comparisons run against the **dataset-pinned** environment (composed
from the real data config; ``lite_physics`` exactly as the dataset recorded
it), never robosuite constructor defaults. A retired hard-reset arm is kept
for context only: it re-samples object placements before every restore, so
a small first-step deviation vs the protocol arm is expected and does NOT
gate.

GATE: exit code 0 iff every history and construction comparison across
every requested task is BITWISE EQUAL. Any future reset-path change must
re-run this gate and pass before adoption.

``--bank-smoke`` runs a different check instead of the equivalence arms:
reset-bank GENERATION diversity under the patched adapter. robosuite's
``_reset_internal`` re-samples object placements on every reset — soft
(``hard_reset=False``) included, because it runs in both branches of
``MujocoEnv.reset()`` — so bank generation must still produce distinct
cases; the smoke checks this property on the exact eval-path adapter.
Generated banks are NOT written to disk.

Run with the rollout venv from the repository root:

    .venv-rollout/Scripts/python.exe scripts/dev/ab_reset_equivalence.py
    .venv-rollout/Scripts/python.exe scripts/dev/ab_reset_equivalence.py --tasks transport

Tool Hang requires its dedicated interpreter:

    .venv-toolhang/Scripts/python.exe scripts/dev/ab_reset_equivalence.py --tasks tool_hang
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

ACTION_SEED_BASE = 20260822


def compose_task(task: str):
    """Compose the pinned data config; return (cfg, meta, spec, action_dim)."""
    from hydra import compose, initialize_config_dir

    from phaseforge.evaluations.envs.robosuite_adapter import StateSpec
    from phaseforge.evaluations.rollout.runner import resolve_pinned_metadata

    with initialize_config_dir(
        config_dir=str(REPO / "phaseforge" / "config"), version_base="1.3"
    ):
        cfg = compose(config_name="main", overrides=[f"data={task}", "eval=rollout"])
    meta = resolve_pinned_metadata(cfg)
    spec = StateSpec(
        keys=tuple(str(k["key"]) for k in cfg.data.state_keys),
        dims=tuple(int(k["dim"]) for k in cfg.data.state_keys),
    )
    return cfg, meta, spec, int(cfg.data.action_dim)


def make_adapter(cfg, meta, spec, action_dim, rng_position: int):
    """Build an adapter after moving the global RNG to a distinct position.

    Construction-time randomization is deterministic-seeded inside the
    adapter, so the position should not matter — that invariance is exactly
    what the construction arm proves.
    """
    np.random.seed(1000 + rng_position)
    np.random.uniform(size=3 + 11 * rng_position)
    from phaseforge.evaluations.envs.robosuite_adapter import RobosuiteStateAdapter

    return RobosuiteStateAdapter(meta, spec, action_dim=action_dim)


def actions_for(case_index: int, horizon: int, action_dim: int) -> np.ndarray:
    """Deterministic open-loop action sequence, independent of global RNG."""
    rng = np.random.default_rng(ACTION_SEED_BASE + 1000 * case_index)
    return rng.uniform(-1, 1, size=(horizon, action_dim)).astype(np.float64)


def roll(adapter, bank, case_index: int, horizon: int, action_dim: int):
    """Restore a case and roll a fixed action sequence; return trajectory."""
    actions = actions_for(case_index, horizon, action_dim)
    adapter.reset_to(bank.case(case_index).states)
    trace = []
    for t in range(horizon):
        state, _, _, _ = adapter.step(actions[t])
        trace.append(np.asarray(state, dtype=np.float64))
    return np.array(trace)


def worst_diff(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return float(np.max(np.abs(a[:n] - b[:n])))


def run_task(task: str, num_cases: int, horizon: int) -> list[str]:
    """Run the gate arms for one task; return failure descriptions.

    Exactly one environment is alive at a time (the protocol itself runs
    one env per process; holding several MuJoCo models simultaneously can
    exhaust allocation on small boxes). The reference trajectories from the
    first arm are plain numpy arrays and outlive their environment.
    """
    import gc

    from phaseforge.evaluations.rollout.runner import load_or_generate_bank

    cfg, meta, spec, action_dim = compose_task(task)
    bank = load_or_generate_bank(cfg, meta)
    case_ids = list(range(num_cases))

    print(f"\n=== task {task} (env {meta.env_name}, robosuite {meta.env_version}, "
          f"bank {bank.bank_id}, {num_cases} cases x {horizon} steps) ===")
    failures: list[str] = []
    t0 = time.perf_counter()

    # Arm 1 — reference + history, one environment.
    adapter_a = make_adapter(cfg, meta, spec, action_dim, rng_position=1)
    first = {ci: roll(adapter_a, bank, ci, horizon, action_dim) for ci in case_ids}
    second = {}
    for ci in case_ids:
        other = (ci + 1) % num_cases
        roll(adapter_a, bank, other, horizon, action_dim)
        second[ci] = roll(adapter_a, bank, ci, horizon, action_dim)
    adapter_a.close()
    del adapter_a
    gc.collect()

    # Arm 2 — independent construction at a different global-RNG position.
    adapter_b = make_adapter(cfg, meta, spec, action_dim, rng_position=2)
    from_b = {ci: roll(adapter_b, bank, ci, horizon, action_dim) for ci in case_ids}
    adapter_b.close()
    del adapter_b
    gc.collect()

    # Arm 3 — retired hard branch, informational only.
    adapter_h = make_adapter(cfg, meta, spec, action_dim, rng_position=3)
    adapter_h.env.hard_reset = True
    from_h = {ci: roll(adapter_h, bank, ci, horizon, action_dim) for ci in case_ids}
    adapter_h.close()
    del adapter_h
    gc.collect()

    elapsed = time.perf_counter() - t0
    for ci in case_ids:
        d_hist = worst_diff(first[ci], second[ci])
        d_con = worst_diff(first[ci], from_b[ci])
        d_hard = worst_diff(first[ci], from_h[ci])
        gate_hist = "BITWISE EQUAL" if d_hist == 0.0 else f"DIFFER ({d_hist:.3e})"
        gate_con = "BITWISE EQUAL" if d_con == 0.0 else f"DIFFER ({d_con:.3e})"
        print(f"case {ci:02d}: GATE history {gate_hist} | GATE construction {gate_con} "
              f"| INFO retired-hard max|diff| {d_hard:.3e}")
        if d_hist != 0.0:
            failures.append(f"{task} case {ci}: history arm differs ({d_hist:.3e})")
        if d_con != 0.0:
            failures.append(f"{task} case {ci}: construction arm differs ({d_con:.3e})")

    print(f"({elapsed:.1f} s)")
    return failures


def run_bank_smoke(task: str, num_cases: int) -> list[str]:
    """Generation-diversity smoke for one task; return failure descriptions.

    Runs ``generate_reset_bank`` through the exact eval-path adapter factory
    (patched adapter, soft branch) and requires all sampled cases to be
    pairwise distinct beyond ``reset_bank.MIN_CASE_DISTANCE``. The bank is
    generated in memory only — nothing touches the frozen banks on disk.
    """
    from phaseforge.evaluations.rollout.reset_bank import (
        MIN_CASE_DISTANCE,
        generate_reset_bank,
    )
    from phaseforge.evaluations.rollout.runner import _adapter_from_config

    cfg, meta, spec, action_dim = compose_task(task)
    print(f"\n=== task {task} (env {meta.env_name}) bank-generation smoke: "
          f"{num_cases} cases, patched soft-reset adapter ===")
    try:
        bank = generate_reset_bank(
            lambda: _adapter_from_config(cfg, meta, seed=2026),
            meta,
            task=str(cfg.data.source.task_name),
            seed=2026,
            num_cases=num_cases,
            max_attempts_per_case=40,
        )
    except Exception as exc:  # noqa: BLE001 — any generation failure gates
        return [f"{task}: bank generation raised {type(exc).__name__}: {exc}"]

    states = np.stack([c.states for c in bank.cases])
    distances = [
        float(np.linalg.norm(states[i][1:] - states[j][1:]))
        for i in range(num_cases)
        for j in range(i + 1, num_cases)
    ]
    min_dist = min(distances)
    ok = len(bank.cases) == num_cases and min_dist > MIN_CASE_DISTANCE
    print(f"cases={len(bank.cases)}  min pairwise L2={min_dist:.4f} "
          f"(threshold {MIN_CASE_DISTANCE}) -> "
          f"{'DISTINCT OK' if ok else 'DUPLICATES / SHORT BANK'}")
    if not ok:
        return [f"{task}: bank generation diversity failed (min L2 {min_dist:.4f})"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        default="lift,can,square,transport",
        help="comma-separated data config names (default: lift,can,square,transport; "
        "run tool_hang separately with .venv-toolhang)",
    )
    parser.add_argument("--cases", type=int, default=3, help="bank cases per task")
    parser.add_argument("--horizon", type=int, default=80, help="steps per probe")
    parser.add_argument(
        "--bank-smoke",
        action="store_true",
        help="run the bank-generation diversity smoke instead of the "
        "equivalence arms (nothing is written to disk)",
    )
    parser.add_argument(
        "--smoke-cases",
        type=int,
        default=5,
        help="cases for --bank-smoke (default: 5)",
    )
    args = parser.parse_args()

    all_failures: list[str] = []
    for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        if args.bank_smoke:
            all_failures.extend(run_bank_smoke(task, args.smoke_cases))
        else:
            all_failures.extend(run_task(task, args.cases, args.horizon))

    print("\n" + "=" * 72)
    if all_failures:
        print("GATE FAILED:")
        for failure in all_failures:
            print(f"  - {failure}")
        return 1
    if args.bank_smoke:
        print("SMOKE PASSED: bank generation under the patched (soft-reset) adapter "
              "produces distinct cases on every requested task.")
    else:
        print("GATE PASSED: adopted reset path is bitwise-deterministic "
              "(history and construction arms) on every requested task.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Rollout hot-path benchmark (dev/cloud measurement harness).

Measures the per-step cost components of the state-only rollout loop on the
installed pinned stack (robosuite/mujoco/torch):

  A. MuJoCo physics cost per policy step on the *protocol* environment —
     the dataset-pinned env_kwargs composed from the real data config
     (``lite_physics`` exactly as the dataset recorded it), never
     robosuite's constructor defaults.
  B. R50-shaped PhaseForge batch=1 inference microbenchmark, including the
     CPU thread-count effect (Lift dims; synthetic weights, not a checkpoint;
     skipped for --task values with other dimensions).
  C. Per-step Python overheads: action validation, state extraction, causal
     phase labeler, episode-record append durability.
  D. Reset cost against the real frozen reset bank: the protocol's soft
     branch vs the retired hard branch (context only).

Run with the rollout venv from the repository root:

    .venv-rollout/Scripts/python.exe scripts/dev/bench_rollout_hotpath.py
    .venv-rollout/Scripts/python.exe scripts/dev/bench_rollout_hotpath.py --task transport

Tool Hang requires its dedicated interpreter:

    .venv-toolhang/Scripts/python.exe scripts/dev/bench_rollout_hotpath.py --task tool_hang

Purpose: measure on the actual evaluation machine before the five-task
sweep, so the wall-clock budget reflects that machine's numbers rather
than the development machine's.
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def bench(fn, n=200, warmup=20):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return {"min": ts[0] * 1e3, "med": ts[len(ts) // 2] * 1e3, "p90": ts[int(len(ts) * 0.9)] * 1e3}


def fmt(r):
    return f"min {r['min']:8.3f}  med {r['med']:8.3f}  p90 {r['p90']:8.3f}  ms"


def _pinned_adapter(task: str):
    """Compose the real data config and build the protocol adapter.

    Uses the dataset-pinned env_kwargs (``lite_physics`` as recorded by the
    dataset — False for every task), so every measured number is the
    protocol's, not robosuite's constructor-default path.
    """
    from hydra import compose, initialize_config_dir

    from phaseforge.evaluations.envs.robosuite_adapter import (
        RobosuiteStateAdapter,
        StateSpec,
    )
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
    adapter = RobosuiteStateAdapter(meta, spec, action_dim=int(cfg.data.action_dim))
    return adapter, cfg, meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="lift", help="data config name (default: lift)")
    args = parser.parse_args()

    print("=" * 72)
    print(f"A. MuJoCo physics — protocol env (data={args.task})")
    print("=" * 72)
    ad, cfg, meta = _pinned_adapter(args.task)
    env = ad.env
    print(
        f"env={meta.env_name}  robosuite={meta.env_version}  "
        f"lite_physics={env.lite_physics}  control_freq={env.control_freq}  "
        f"timestep={env.model_timestep}  "
        f"substeps/step={int(round(env.control_timestep / env.model_timestep))}"
    )
    act = np.zeros(int(cfg.data.action_dim), dtype=np.float64)
    r_step = bench(lambda: env.step(act), n=200, warmup=30)
    print(f"env.step            {fmt(r_step)}")
    r_obs = bench(lambda: env._get_observations(force_update=True), n=500, warmup=50)
    print(f"_get_observations   {fmt(r_obs)}")
    r_succ = bench(lambda: env._check_success(), n=1000, warmup=100)
    print(f"_check_success      {fmt(r_succ)}")

    if args.task == "lift":
        print()
        print("=" * 72)
        print("B. R50-shaped inference microbenchmark, batch=1 (CPU; synthetic weights)")
        print("=" * 72)
        import torch

        from phaseforge.models.components.action_head import ActionHead
        from phaseforge.models.components.encoder import StateEncoder
        from phaseforge.models.components.expert import ExpertMLP
        from phaseforge.models.components.phase_head import PhaseClassificationHead
        from phaseforge.models.components.router import TopKRouter
        from phaseforge.models.phase_moe import PhaseBootstrappedMoE

        class DS:
            def __init__(self):
                g = torch.Generator().manual_seed(0)
                self.states = torch.randn(64, 19, generator=g)
                self.actions = torch.randn(64, 7, generator=g)
                self.phases = torch.randint(0, 6, (64,), generator=g)

            def __len__(self):
                return len(self.states)

            def __getitem__(self, i):
                return {
                    "state": self.states[i],
                    "action": self.actions[i],
                    "phase": self.phases[i],
                }

        from torch.utils.data import DataLoader

        model = PhaseBootstrappedMoE(
            encoder=StateEncoder(19, [256, 256, 256], 128, activation="gelu",
                                 dropout=0.1, use_residual=True),
            action_head=ActionHead(128, 7, head_type="deterministic", hidden_dim=256),
            phase_head=PhaseClassificationHead(128, 6),
            router=TopKRouter(128, num_experts=6, top_k=2, noise_std=0.1,
                              balance_coeff=0.01, normalize_input=True),
            expert=ExpertMLP(128, [256], 7, activation="gelu"),
            expert_init={"type": "random"},
        )
        model.bootstrap_moe(dataloader=DataLoader(DS(), batch_size=16), device="cpu")
        model.eval()

        x = torch.randn(1, 19)
        with torch.inference_mode():
            r_fwd = bench(lambda: model.get_action(x), n=500, warmup=100)
        print(f"forward thr={torch.get_num_threads()}          {fmt(r_fwd)}")
        previous_intraop_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with torch.inference_mode():
                r_fwd1 = bench(lambda: model.get_action(x), n=500, warmup=100)
        finally:
            torch.set_num_threads(previous_intraop_threads)
        print(f"forward thr=1                 {fmt(r_fwd1)}")
    else:
        print("\nB. skipped (canonical-model section is defined on Lift dims)")

    print()
    print("=" * 72)
    print(f"C/D. Adapter overheads + reset cost (frozen {args.task} bank)")
    print("=" * 72)
    from phaseforge.data.robomimic.phase_labeler import CausalPhaseStepLabeler
    from phaseforge.evaluations.rollout.runner import load_or_generate_bank
    from phaseforge.outputs_writer.episodes import append_episode_record

    print(f"validate_action     {fmt(bench(lambda: ad.validate_action(act), n=2000, warmup=200))}")
    obsp = env._get_observations(force_update=True)
    print(f"extract_state       {fmt(bench(lambda: ad.extract_state(obsp), n=2000, warmup=200))}")

    calib = {
        "closed_level": 0.02, "open_level": 0.04, "mirror": False,
        "velocity_threshold": 0.01, "min_duration": 5, "filter_size": 7,
        "eef_pos_slice": [0, 3], "gripper_qpos_slice": [7, 9], "num_phases": 6,
    }
    lab = CausalPhaseStepLabeler(calib)
    s_dim = np.random.randn(int(cfg.data.state_dim)).astype(np.float32)
    print(f"phase_labeler.step  {fmt(bench(lambda: lab.step(s_dim), n=5000, warmup=500))}")

    bank = load_or_generate_bank(cfg, meta)
    print(f"bank load+verify    {bank.num_cases} cases ({bank.bank_id})")
    case = bank.cases[0]

    def reset_protocol():
        ad.env.hard_reset = False
        ad.reset_to(case.states)

    def reset_retired_hard():
        ad.env.hard_reset = True
        ad.reset_to(case.states)

    rs = bench(reset_protocol, n=50, warmup=5)
    print(f"reset_to PROTOCOL (soft, canonicalized)  {fmt(rs)}")
    rh = bench(reset_retired_hard, n=8, warmup=2)
    print(f"reset_to RETIRED (hard reset)            {fmt(rh)}   (context only)")
    ad.close()

    row = {
        "run_id": "bench", "model": "bench", "checkpoint_sha256": "x",
        "task": meta.env_name, "training_seed": 42, "reset_seed": 2026,
        "episode_index": 0, "valid_episode": True, "success": False,
        "timed_out": True, "termination_reason": "task_timeout",
        "failure_category": "task_timeout",
    }
    with tempfile.TemporaryDirectory() as td:
        t0 = time.perf_counter()
        for i in range(50):
            row["episode_index"] = i
            append_episode_record(td, dict(row))
        dt = (time.perf_counter() - t0) / 50 * 1e3
    print(f"append+fsync mean   {dt:.3f} ms/episode ({dt * 50:.0f} ms per 50-ep run)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

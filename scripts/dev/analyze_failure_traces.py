"""Forensic trace analysis tool for failure episodes.

Analyzes `trace.jsonl` produced by rollout evaluation with `eval.episodes.trace_level=full`.
Specifically diagnoses:
1. Failure Mode A (Conservative Hover: Eps 3, 13, 20, 44):
   - Raw Z-axis expert output pre_clip_command[2] and final_action[2].
   - Router margin (d_k2 - d_k1) and expert selection (Transport vs Place hesitation).
2. Failure Mode B (Rim Deflection / Millimeter Miss: Eps 0, 8, 17, 29, 30, 32, 47, 48):
   - Lateral end-effector velocity (x_dot, y_dot) at release timestep.
   - Release height z_eef relative to bin rim height (z_rim ~ 0.82m).
   - Gripper timing and settling distance to bin center.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# Robosuite PickPlaceCan bin bounding box and reference geometry
BIN_CENTER = (-0.15, 0.10, 0.83)
BIN_RIM_Z = 0.820
BIN_BOUNDS_X = (-0.20, -0.10)
BIN_BOUNDS_Y = (0.05, 0.15)
BIN_BOUNDS_Z = (0.80, 0.86)

HOVER_EPISODES = {3, 13, 20, 44}
RIM_EPISODES = {0, 8, 17, 29, 30, 32, 47, 48}


def analyze_traces(trace_path: Path) -> dict[str, Any]:
    if not trace_path.is_file():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")

    episodes: dict[int, list[dict[str, Any]]] = {}
    with open(trace_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            ep_id = int(row["episode_id"])
            if ep_id not in episodes:
                episodes[ep_id] = []
            episodes[ep_id].append(row)

    print(f"Loaded {len(episodes)} episodes from {trace_path}\n")

    report: dict[str, Any] = {
        "hover_failures": {},
        "rim_failures": {},
        "summary": {},
    }

    print("================================================================================")
    print("ANALYSIS OF FAILURE MODE A: CONSERVATIVE HOVER (Eps 3, 13, 20, 44)")
    print("================================================================================")
    for ep_id in sorted(HOVER_EPISODES):
        if ep_id not in episodes:
            print(f"Episode {ep_id}: Not found in trace.")
            continue
        steps = episodes[ep_id]
        terminal_window = steps[-50:]

        z_cmds = [
        s.get("final_action", [0, 0, 0])[2] for s in terminal_window if s.get("final_action")
    ]
        z_pre = [
        s.get("pre_clip_command", [0, 0, 0])[2]
        for s in terminal_window
        if s.get("pre_clip_command")
    ]
        margins = [
        s.get("router_margin", 0.0) for s in terminal_window if s.get("router_margin") is not None
    ]
        experts = [
        s.get("selected_expert") for s in terminal_window if s.get("selected_expert") is not None
    ]

        mean_z_cmd = sum(z_cmds) / len(z_cmds) if z_cmds else 0.0
        mean_z_pre = sum(z_pre) / len(z_pre) if z_pre else 0.0
        mean_margin = sum(margins) / len(margins) if margins else 0.0

        expert_counts = {}
        for exp in experts:
            expert_counts[exp] = expert_counts.get(exp, 0) + 1

        print(
            f"Episode {ep_id:2d}: Total Steps={len(steps)} | "
            f"Terminal Z-cmd mean={mean_z_cmd:+.4f}, Pre-clip={mean_z_pre:+.4f} | "
            f"Margin={mean_margin:.4f} | Experts={expert_counts}"
        )

        report["hover_failures"][ep_id] = {
            "steps": len(steps),
            "mean_terminal_z_cmd": mean_z_cmd,
            "mean_terminal_z_preclip": mean_z_pre,
            "mean_terminal_margin": mean_margin,
            "expert_distribution": expert_counts,
        }

    print("\n================================================================================")
    print("ANALYSIS OF FAILURE MODE B: RIM DEFLECTION / MISS (Eps 0, 8, 17, 29, 30, 32, 47, 48)")
    print("================================================================================")
    for ep_id in sorted(RIM_EPISODES):
        if ep_id not in episodes:
            print(f"Episode {ep_id}: Not found in trace.")
            continue
        steps = episodes[ep_id]

        # Find release timestep (where gripper command turns positive/opens, or last step)
        release_step = None
        for step in steps:
            action = step.get("final_action")
            if action and len(action) >= 7 and action[6] > 0.0:  # gripper opening
                release_step = step
                break
        if release_step is None:
            release_step = steps[-1]

        t_rel = release_step.get("timestep", 0)
        action_rel = release_step.get("final_action", [0] * 7)
        raw_obs = release_step.get("raw_obs_summary", {})

        eef_pos = raw_obs.get("robot0_eef_pos", [0, 0, 0])
        z_eef = eef_pos[2] if len(eef_pos) > 2 else 0.0
        z_rel_rim = z_eef - BIN_RIM_Z

        # Estimate lateral velocity at release from final_action or task vars
        v_lateral = (action_rel[0] ** 2 + action_rel[1] ** 2) ** 0.5

        print(
            f"Episode {ep_id:2d}: Release at t={t_rel:3d} | GripCmd={action_rel[6]:+.3f} | "
            f"EEF Z={z_eef:.4f} (Rel to rim: {z_rel_rim:+.4f}m) | "
            f"Lat-Action-Norm={v_lateral:.4f} | "
            f"Lat-Cmd=({action_rel[0]:+.3f}, {action_rel[1]:+.3f})"
        )

        report["rim_failures"][ep_id] = {
            "release_timestep": t_rel,
            "gripper_cmd": action_rel[6],
            "eef_z": z_eef,
            "z_rel_to_rim": z_rel_rim,
            "lateral_action_norm": v_lateral,
            "lateral_cmd": (action_rel[0], action_rel[1]),
        }

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze trace.jsonl for failure episode forensics."
    )
    parser.add_argument("--trace", type=str, required=True, help="Path to trace.jsonl")
    parser.add_argument(
        "--output", type=str, default=None, help="Optional output path for json summary"
    )
    args = parser.parse_args()

    trace_file = Path(args.trace)
    report = analyze_traces(trace_file)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()

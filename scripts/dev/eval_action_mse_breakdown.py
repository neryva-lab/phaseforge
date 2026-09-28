"""Diagnostic tool: per-dimension and per-phase action MSE breakdown.

Computes:
1. Per-dimension MSE: pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, gripper.
2. Per-phase MSE: Approach (0), Pre-grasp (1), Grasp (2), Transport (3), Place (4), Retract (5).
3. Phase x Dimension MSE matrix.
"""

from __future__ import annotations

from typing import Any

import torch

DIM_NAMES = ["pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z", "gripper"]
PHASE_NAMES = ["0_Approach", "1_PreGrasp", "2_Grasp", "3_Transport", "4_Place", "5_Retract"]


def compute_mse_breakdown(
    action_preds: torch.Tensor,
    action_targets: torch.Tensor,
    phases: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Compute detailed MSE breakdown across dimensions and phases.

    Args:
        action_preds: (N, 7) or (N, T, 7)
        action_targets: (N, 7) or (N, T, 7)
        phases: (N,) or (N, T)
        mask: optional bool mask of valid entries
    """
    if action_preds.ndim > 2:
        action_preds = action_preds.reshape(-1, action_preds.size(-1))
        action_targets = action_targets.reshape(-1, action_targets.size(-1))
        phases = phases.reshape(-1)
        if mask is not None:
            mask = mask.reshape(-1)

    if mask is not None:
        valid = mask.bool()
        action_preds = action_preds[valid]
        action_targets = action_targets[valid]
        phases = phases[valid]

    diff_sq = (action_preds - action_targets) ** 2  # (N, 7)
    global_mse = float(diff_sq.mean().item())

    # 1. Per-dimension MSE
    per_dim = {}
    for i, name in enumerate(DIM_NAMES):
        per_dim[name] = float(diff_sq[:, i].mean().item())

    # Summary groups
    per_dim["trans_mean"] = float(diff_sq[:, 0:3].mean().item())
    per_dim["rot_mean"] = float(diff_sq[:, 3:6].mean().item())
    per_dim["gripper"] = float(diff_sq[:, 6].mean().item())

    # 2. Per-phase MSE
    per_phase = {}
    phase_x_dim = {}
    for p_idx in range(6):
        p_mask = phases == p_idx
        count = int(p_mask.sum().item())
        p_name = PHASE_NAMES[p_idx] if p_idx < len(PHASE_NAMES) else f"Phase_{p_idx}"
        if count > 0:
            p_diff = diff_sq[p_mask]
            per_phase[p_name] = {
                "count": count,
                "overall_mse": float(p_diff.mean().item()),
                "trans_mse": float(p_diff[:, 0:3].mean().item()),
                "rot_mse": float(p_diff[:, 3:6].mean().item()),
                "grip_mse": float(p_diff[:, 6].mean().item()),
            }
            phase_x_dim[p_name] = {
                DIM_NAMES[i]: float(p_diff[:, i].mean().item()) for i in range(7)
            }
        else:
            per_phase[p_name] = {"count": 0, "overall_mse": None}
            phase_x_dim[p_name] = {DIM_NAMES[i]: None for i in range(7)}

    return {
        "global_mse": global_mse,
        "per_dimension": per_dim,
        "per_phase": per_phase,
        "phase_x_dim": phase_x_dim,
    }


def format_breakdown_table(breakdown: dict[str, Any]) -> str:
    """Format the breakdown as a readable ASCII report."""
    lines = []
    lines.append("=" * 78)
    lines.append(f"ACTION MSE BREAKDOWN (Global MSE: {breakdown['global_mse']:.6f})")
    lines.append("=" * 78)

    lines.append("\n1. Per-Dimension MSE:")
    lines.append(f"  Translation (X, Y, Z): {breakdown['per_dimension']['pos_x']:.5f}, "
                 f"{breakdown['per_dimension']['pos_y']:.5f}, "
                 f"{breakdown['per_dimension']['pos_z']:.5f}  "
                 f"| Mean: {breakdown['per_dimension']['trans_mean']:.5f}")
    lines.append(f"  Rotation   (R, P, Y): {breakdown['per_dimension']['rot_x']:.5f}, "
                 f"{breakdown['per_dimension']['rot_y']:.5f}, "
                 f"{breakdown['per_dimension']['rot_z']:.5f}  "
                 f"| Mean: {breakdown['per_dimension']['rot_mean']:.5f}")
    lines.append(f"  Gripper             : {breakdown['per_dimension']['gripper']:.5f}")

    lines.append("\n2. Per-Phase MSE:")
    lines.append(
        f"  {'Phase':<14} | {'Count':<7} | {'Overall MSE':<12} | "
        f"{'Trans MSE':<10} | {'Rot MSE':<10} | {'Grip MSE':<10}"
    )
    lines.append("  " + "-" * 74)
    for p_name, data in breakdown["per_phase"].items():
        if data["count"] > 0:
            lines.append(
                f"  {p_name:<14} | {data['count']:<7} | {data['overall_mse']:<12.5f} | "
                f"{data['trans_mse']:<10.5f} | {data['rot_mse']:<10.5f} | {data['grip_mse']:<10.5f}"
            )
        else:
            lines.append(f"  {p_name:<14} | 0       | N/A")

    lines.append("=" * 78)
    return "\n".join(lines)


if __name__ == "__main__":
    # Self-test with synthetic data
    torch.manual_seed(42)
    N = 1000
    preds = torch.randn(N, 7) * 0.1
    targets = torch.zeros(N, 7)
    phases = torch.randint(0, 6, (N,))
    res = compute_mse_breakdown(preds, targets, phases)
    print(format_breakdown_table(res))

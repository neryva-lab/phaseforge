"""Trace Robosuite PickPlaceCan success predicate metrics during rollout.

Logs:
1. Object position at release vs target bin bounding box.
2. Distance from end-effector to object (r_reach predicate).
3. Contact / placement error (per-axis bin containment, distance to bin center).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def evaluate_predicate_geometry(
    obj_pos: np.ndarray,
    bin2_pos: np.ndarray,
    bin_size: np.ndarray,
    bin_id: int = 0,
    eef_pos: np.ndarray | None = None,
) -> dict[str, Any]:
    """Check Robosuite PickPlace native predicate boundaries.

    Mirrors robosuite.environments.manipulation.pick_place.PickPlace.not_in_bin
    and _check_success.
    """
    bin_x_low = float(bin2_pos[0])
    bin_y_low = float(bin2_pos[1])
    if bin_id == 0 or bin_id == 2:
        bin_x_low -= float(bin_size[0]) / 2.0
    if bin_id < 2:
        bin_y_low -= float(bin_size[1]) / 2.0

    bin_x_high = bin_x_low + float(bin_size[0]) / 2.0
    bin_y_high = bin_y_low + float(bin_size[1]) / 2.0
    bin_z_low = float(bin2_pos[2])
    bin_z_high = bin_z_low + 0.10

    in_x = bin_x_low < float(obj_pos[0]) < bin_x_high
    in_y = bin_y_low < float(obj_pos[1]) < bin_y_high
    in_z = bin_z_low < float(obj_pos[2]) < bin_z_high
    in_bin = in_x and in_y and in_z

    bin_center = np.array([
        0.5 * (bin_x_low + bin_x_high),
        0.5 * (bin_y_low + bin_y_high),
        0.5 * (bin_z_low + bin_z_high),
    ])
    center_dist = float(np.linalg.norm(obj_pos[:3] - bin_center))

    r_reach = None
    dist_to_eef = None
    if eef_pos is not None:
        dist_to_eef = float(np.linalg.norm(eef_pos[:3] - obj_pos[:3]))
        r_reach = float(1.0 - np.tanh(10.0 * dist_to_eef))

    return {
        "in_bin": in_bin,
        "in_x": in_x,
        "in_y": in_y,
        "in_z": in_z,
        "x_bounds": [bin_x_low, bin_x_high],
        "y_bounds": [bin_y_low, bin_y_high],
        "z_bounds": [bin_z_low, bin_z_high],
        "obj_pos": [float(v) for v in obj_pos[:3]],
        "bin_center": [float(v) for v in bin_center],
        "dist_to_center": center_dist,
        "dist_to_eef": dist_to_eef,
        "r_reach": r_reach,
        "success": bool(in_bin and (r_reach is not None and r_reach < 0.6)),
    }


if __name__ == "__main__":
    # Test with standard Robosuite PickPlaceCan dimensions
    bin2_pos = np.array([0.1, 0.25, 0.8])
    bin_size = np.array([0.2, 0.2, 0.05])
    # Place can dead center in bin 0
    test_obj_center = np.array([0.05, 0.20, 0.83])
    res = evaluate_predicate_geometry(
        test_obj_center, bin2_pos, bin_size, bin_id=0, eef_pos=np.array([0.05, 0.20, 0.95])
    )
    print("Predicate Test (Center):", json.dumps(res, indent=2))

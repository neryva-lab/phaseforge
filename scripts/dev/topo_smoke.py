"""Topo discovery smoke test (CPU-only).

Exercises the full observation-consistent discovery chain on synthetic
data (no raw dataset needed): task-space extraction -> PELT segmentation
-> segment clustering -> observability audit.

Usage:
    python scripts/dev/topo_smoke.py [--seed 0]

Exit 0 with a one-line summary per stage; exit 1 on any failure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main(argv: list[str] | None = None) -> int:
    import numpy as np

    from phaseforge.data.topo.artifacts import TOPO_ARTIFACT_VERSION
    from phaseforge.data.topo.cluster import cluster_segments, segment_features
    from phaseforge.data.topo.observability import audit_regimes
    from phaseforge.data.topo.pelt import run_pelt
    from phaseforge.data.topo.task_vars import concat_task_matrix, extract_task_vars

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    rng = np.random.default_rng(args.seed)

    # Two contact modes: far/hold (open gripper) vs near/grasp (closed).
    def _traj(offset: float, aperture: float, length: int = 120) -> np.ndarray:
        state = rng.normal(0, 0.05, (length, 23))
        state[:, 0:3] = rng.normal(offset, 0.05, (length, 3))
        state[:, 7:9] = aperture
        return state

    keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
    dims = [3, 4, 2, 14]
    trajs = [_traj(0.0, 0.04), _traj(0.5, 0.04), _traj(0.0, 0.005), _traj(0.5, 0.005)]
    matrices = [concat_task_matrix(extract_task_vars(t, keys, dims)) for t in trajs]
    print(f"task_vars: {len(matrices)} trajs -> matrix dim {matrices[0].shape[1]}")

    all_bounds = [run_pelt(m, penalty_beta=10.0, min_segment_len=5) for m in matrices]
    segments, owners = [], []
    for idx, (matrix, bounds) in enumerate(zip(matrices, all_bounds)):
        for j in range(len(bounds) - 1):
            segments.append(matrix[bounds[j] : bounds[j + 1]])
            owners.append(idx)
    print(f"pelt: {len(segments)} segments total")
    if len(segments) < 2:
        print("topo_smoke FAILED: fewer than 2 segments for K=2 clustering.")
        return 1

    feats = segment_features(segments)
    labels = cluster_segments(feats, num_clusters=2, method="kmeans", seed=args.seed)
    print(f"cluster: K=2 occupancy {[int((labels == k).sum()) for k in (0, 1)]}")

    flat_states = np.concatenate(trajs, axis=0)
    seg_labels = np.zeros(flat_states.shape[0], dtype=np.int64)
    cursor = 0
    for idx, traj in enumerate(trajs):
        seg_labels[cursor : cursor + len(traj)] = labels[
            [i for i, o in enumerate(owners) if o == idx][0]
        ]
        cursor += len(traj)
    traj_ids = np.concatenate(
        [np.full(len(t), i, dtype=np.int64) for i, t in enumerate(trajs)]
    )
    report = audit_regimes(flat_states, seg_labels, traj_ids, 2)
    print(
        f"audit: passed={report.passed} macro_f1={report.macro_f1:.3f} "
        f"artifact=v{TOPO_ARTIFACT_VERSION}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

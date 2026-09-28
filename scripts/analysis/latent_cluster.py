"""Latent-cluster acceptance metrics for Stage 1 (CPU-only).

Loads a training run's encoder from its checkpoint, embeds cached states,
and quantifies regime clustering:

* k-NN regime accuracy (trajectory-aware GroupKFold, never resubstitution),
* mean intra-regime / inter-regime latent distance,
* silhouette score,
* optional t-SNE scatter for inspection.

Usage:
    python scripts/analysis/latent_cluster.py --run-dir <run_dir> \\
        --labels phase --out supcon_metrics.json

Exit 0 always (this is a measurement tool, not a gate); `--strict`
exits 1 when `silhouette <= 0.2` or `knn_acc <= 0.7` (proposed defaults).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def _load_run(run_dir: Path):
    """Rebuild the model encoder from a training run directory."""
    import torch
    from omegaconf import OmegaConf

    from phaseforge.utils.registry import build_model

    resolved = run_dir / "resolved_config.yaml"
    if not resolved.is_file():
        raise FileNotFoundError(f"No resolved_config.yaml in run dir {run_dir}.")
    cfg = OmegaConf.load(str(resolved))
    checkpoint = run_dir / "checkpoints" / "checkpoint_best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"No checkpoints/checkpoint_best.pt in {run_dir}.")
    model = build_model(cfg)
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing or unexpected:
        print(f"latent_cluster: state-dict gaps missing={missing} unexpected={unexpected}")
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        raise RuntimeError("The run's model has no encoder attribute.")
    encoder.eval()
    return cfg, encoder


def _load_cache_states(data_config_hash: str):
    """Load (state, label-field candidates, traj ids) from a processed cache."""
    import torch

    from phaseforge.data.paths import processed_cache_root

    cache_dir = processed_cache_root() / data_config_hash
    traj_files = sorted((cache_dir / "trajectories").glob("*.pt"))
    if not traj_files:
        raise FileNotFoundError(f"No trajectories under cache {cache_dir}.")
    states, labels, traj_ids, extras = [], [], [], {}
    for idx, path in enumerate(traj_files):
        traj = torch.load(str(path), map_location="cpu", weights_only=False)
        states.append(traj["state"])
        labels.append(traj["phase"])
        traj_ids.append(torch.full((traj["state"].shape[0],), idx, dtype=torch.long))
        for key in ("phase_dynamic", "phase_rule", "phase_topo"):
            if key in traj:
                extras.setdefault(key, []).append(traj[key])
    import torch as _torch

    merged_extras = {k: _torch.cat(v, dim=0) for k, v in extras.items()}
    return (
        torch.cat(states, dim=0),
        torch.cat(labels, dim=0),
        torch.cat(traj_ids, dim=0),
        merged_extras,
    )


def main(argv: list[str] | None = None) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import numpy as np
    import torch
    from sklearn.metrics import silhouette_score
    from sklearn.model_selection import GroupKFold
    from sklearn.neighbors import KNeighborsClassifier

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Training run directory.")
    parser.add_argument("--labels", default="phase", help="Label field for regimes.")
    parser.add_argument("--max-points", type=int, default=5000)
    parser.add_argument("--out", default=None, help="Metrics JSON path (default: stdout).")
    parser.add_argument("--embed-out", default=None, help="Optional t-SNE embedding (.npy).")
    parser.add_argument("--strict", action="store_true", help="Gate on proposed defaults.")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    try:
        _cfg, encoder = _load_run(run_dir)
        run_meta_path = run_dir / "run_meta.json"
        data_hash = json.loads(run_meta_path.read_text(encoding="utf-8"))["data_config_hash"]
        states, labels, traj_ids, extras = _load_cache_states(str(data_hash))
    except (FileNotFoundError, RuntimeError, KeyError, ValueError) as exc:
        print(f"latent_cluster ERROR: {exc}", file=sys.stderr)
        return 2

    if args.labels == "phase":
        regime = labels
    elif args.labels in extras:
        regime = extras[args.labels]
    else:
        print(f"latent_cluster ERROR: label field {args.labels!r} not in cache.", file=sys.stderr)
        return 2

    cap = int(args.max_points)
    if states.shape[0] > cap:
        rng = np.random.default_rng(0)
        keep = np.sort(rng.choice(states.shape[0], size=cap, replace=False))
        states, regime, traj_ids = states[keep], regime[keep], traj_ids[keep]
    with torch.no_grad():
        latents = encoder(states.float()).detach().cpu().numpy()
    regime_np = np.asarray(regime.numpy()).reshape(-1)
    groups = np.asarray(traj_ids.numpy()).reshape(-1)

    knn = KNeighborsClassifier(n_neighbors=5)
    folds = min(3, len(np.unique(groups)))
    if folds < 2:
        knn.fit(latents, regime_np)
        knn_acc = float(knn.score(latents, regime_np))
    else:
        scores = []
        for train_idx, test_idx in GroupKFold(n_splits=folds).split(latents, regime_np, groups):
            knn.fit(latents[train_idx], regime_np[train_idx])
            scores.append(knn.score(latents[test_idx], regime_np[test_idx]))
        knn_acc = float(np.mean(scores))

    sub_n = min(2000, latents.shape[0])
    sub = np.random.default_rng(1).choice(latents.shape[0], size=sub_n, replace=False)
    sub = np.sort(sub)
    dists = np.linalg.norm(latents[sub, None, :] - latents[None, sub, :], axis=-1)
    same = regime_np[sub, None] == regime_np[None, sub]
    intra = float(dists[same].mean()) if same.any() else 0.0
    inter = float(dists[~same].mean()) if (~same).any() else 0.0
    try:
        silhouette = float(silhouette_score(latents[sub], regime_np[sub], random_state=0))
    except ValueError:
        silhouette = float("nan")

    if args.embed_out is not None:
        from sklearn.manifold import TSNE

        embedding = TSNE(n_components=2, perplexity=30, random_state=0).fit_transform(
            latents[sub]
        )
        np.save(args.embed_out, embedding)

    payload = {
        "knn_acc": knn_acc,
        "intra_dist": intra,
        "inter_dist": inter,
        "silhouette": silhouette,
        "n_points": int(latents.shape[0]),
        "num_regimes": int(len(np.unique(regime_np))),
        "labels": args.labels,
    }
    if args.out is not None:
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2))
    if args.strict and not (silhouette > 0.2 and knn_acc > 0.7):
        print("latent_cluster: strict gate not met.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

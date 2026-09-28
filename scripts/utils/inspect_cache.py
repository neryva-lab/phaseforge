"""Script to inspect the processed PhaseForge cache."""

import argparse
import json
from pathlib import Path

import torch


def inspect_cache(cache_dir: str):
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        print(f"Error: Cache directory '{cache_dir}' not found.")
        print("Make sure you are pointing to the correct data root and hash.")
        return

    # 1. Manifest
    manifest_file = cache_path / "manifest.json"
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text())
        print("=== MANIFEST ===")
        print(json.dumps(manifest, indent=2))
        print()
    else:
        print("No manifest.json found.")

    # 2. Norm stats
    norm_file = cache_path / "norm_stats.pt"
    if norm_file.exists():
        norm_stats = torch.load(norm_file, weights_only=False)
        print("=== NORM STATS ===")
        for k, v in norm_stats.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: shape={tuple(v.shape)}")
            else:
                print(f"  {k}: {v}")
        print()
    else:
        print("No norm_stats.pt found.")

    # 3. Trajectory
    traj_dir = cache_path / "trajectories"
    if traj_dir.exists():
        traj_files = sorted(traj_dir.glob("*.pt"))
        print(f"=== TRAJECTORIES ({len(traj_files)} total) ===")
        if traj_files:
            # Inspect the first trajectory
            first_traj = torch.load(traj_files[0], weights_only=False)
            print(f"Inspecting first trajectory ({traj_files[0].name}):")
            for k, v in first_traj.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")
                else:
                    print(f"  {k}: {v}")

            if "phase" in first_traj:
                unique, counts = torch.unique(first_traj["phase"], return_counts=True)
                dist = dict(zip(unique.tolist(), counts.tolist()))
                print(f"  Phase distribution: {dist}")
        print()
    else:
        print("No trajectories/ directory found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect PhaseForge cached data.")
    # Default to a known cache hash
    parser.add_argument(
        "--hash",
        type=str,
        default="a4c74be17f117a4b",
        help="The config hash directory name (e.g. a4c74be17f117a4b)",
    )
    parser.add_argument("--data-root", type=str, default="data", help="Path to the 'data' folder")
    args = parser.parse_args()

    cache_dir = Path(args.data_root) / "processed" / "cache" / args.hash
    print(f"Inspecting Cache Directory: {cache_dir}\n")
    inspect_cache(str(cache_dir))

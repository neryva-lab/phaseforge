"""Script to upload the processed PhaseForge data to Hugging Face."""

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def upload_dataset(data_dir: str, repo_id: str, token: str):
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: Directory {data_dir} does not exist.")
        return

    print(f"Preparing to upload {data_dir} to https://huggingface.co/datasets/{repo_id}")

    api = HfApi(token=token)

    # Ensure the dataset repository exists (created private by default)
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=True)
        print(f"Repository '{repo_id}' is ready.")
    except Exception as e:
        print(f"Failed to create or access repository: {e}")
        print("Please check that your token has 'WRITE' permissions.")
        return

    print("Starting upload (this may take a few minutes)...")
    try:
        api.upload_folder(
            folder_path=str(data_path),
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo="data",  # This will place it in a 'data' folder in the repo
            commit_message="Upload processed PhaseForge cache data",
        )
        print("Upload complete! \u2705")
        print(f"View your dataset at: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"Upload failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload dataset to HuggingFace")
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repo ID (e.g. 'username/phaseforge-cache')",
    )
    parser.add_argument(
        "--token", type=str, required=True, help="HuggingFace API token with WRITE access"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data", help="Local directory to upload (default: 'data')"
    )

    args = parser.parse_args()
    upload_dataset(args.data_dir, args.repo_id, args.token)

"""Download raw robomimic dataset artifacts from the Hugging Face mirror.

Usage:
    uv run python scripts/utils/download_datasets.py                 # Download all 5 tasks
    uv run python scripts/utils/download_datasets.py --tasks lift    # Download only Lift
    uv run python scripts/utils/download_datasets.py --tasks lift can square tool_hang transport
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from phaseforge.data.ingestion.hf_downloader import download_hf_file

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ID = "amandlek/robomimic"

TASK_PATHS: dict[str, str] = {
    "lift": "v1.5/lift/ph/low_dim_v15.hdf5",
    "can": "v1.5/can/ph/low_dim_v15.hdf5",
    "square": "v1.5/square/ph/low_dim_v15.hdf5",
    "tool_hang": "v1.5/tool_hang/ph/low_dim_v15.hdf5",
    "transport": "v1.5/transport/ph/low_dim_v15.hdf5",
}


def download_datasets(
    tasks: list[str],
    data_dir: Path,
) -> None:
    logger.info("Target directory: %s", data_dir.resolve())
    for task in tasks:
        task_lower = task.lower().strip()
        if task_lower not in TASK_PATHS:
            raise ValueError(
                f"Unknown task {task!r}. Known tasks: {list(TASK_PATHS.keys())}"
            )
        rel_path = TASK_PATHS[task_lower]
        dest_dir = data_dir / task_lower
        dest_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading/verifying task '%s' -> %s...", task_lower, dest_dir)
        download_hf_file(
            repo_id=REPO_ID,
            path=rel_path,
            dest_dir=dest_dir,
        )
        logger.info("Task '%s' is ready. \u2705", task_lower)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download raw robomimic datasets from HuggingFace."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(TASK_PATHS.keys()),
        help=f"Tasks to download (default: all). Choices: {list(TASK_PATHS.keys())}",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/robomimic"),
        help="Base destination directory for raw datasets (default: data/raw/robomimic)",
    )
    args = parser.parse_args()

    try:
        download_datasets(args.tasks, args.data_dir)
        logger.info("All requested datasets successfully provisioned.")
    except Exception as exc:
        logger.error("Download failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

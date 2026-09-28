"""Verify generated publication assets: ``python -m studies.analysis.scripts.verify``.

Checks (fail-closed, one exit code):
1. Registry shape — exactly the planned 23 assets (F1-F4, T1-T3, A1-A16),
   each with declared outputs and a loadable generator (except schematics).
2. Presence — every declared output exists under paper_root and is non-empty.
3. Manifest consistency — the generation manifest's assets match the
   registry ids and every recorded output exists with a matching sha256.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from studies.analysis.assets import ASSET_REGISTRY, load_generator
from studies.analysis.common import io as cio
from studies.analysis.common.config import generation_manifest_path, paper_root

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s | %(message)s")
logger = logging.getLogger("studies.analysis.verify")

PLANNED_IDS = ["F1", "F2", "F3", "F4"] + ["T1", "T2", "T3"] + [f"A{i}" for i in range(1, 16)]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-manifest", action="store_true", help="Skip generation-manifest consistency checks."
    )
    parser.add_argument(
        "--allow-missing-schematics",
        action="store_true",
        help="Do not fail verification for schematic assets (e.g. F1) waiting for manual design.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    problems: list[str] = []

    # 1. Registry shape.
    if sorted(ASSET_REGISTRY) != sorted(PLANNED_IDS):
        problems.append(
            f"registry ids diverge from the plan: {sorted(set(ASSET_REGISTRY) ^ set(PLANNED_IDS))}"
        )
    for spec in ASSET_REGISTRY.values():
        if not spec.outputs:
            problems.append(f"{spec.id}: no declared outputs")
        if spec.kind == "schematic":
            continue
        try:
            if load_generator(spec) is None:
                problems.append(f"{spec.id}: no generator")
        except (ImportError, AttributeError) as exc:
            problems.append(f"{spec.id}: generator unusable ({exc})")

    # 2. Presence + non-empty.
    for spec in ASSET_REGISTRY.values():
        if spec.kind == "schematic" and args.allow_missing_schematics:
            continue
        for rel in spec.outputs:
            path = paper_root() / rel
            if not path.is_file():
                problems.append(
                    f"{spec.id}: missing {rel}"
                    + (" (schematic: place manually)" if spec.kind == "schematic" else "")
                )
            elif path.stat().st_size == 0:
                problems.append(f"{spec.id}: empty {rel}")

    # 3. Manifest consistency.
    manifest_path = generation_manifest_path()
    if not manifest_path.is_file():
        problems.append(f"generation manifest missing: {manifest_path}")
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for asset_id, record in manifest.get("assets", {}).items():
                if asset_id not in ASSET_REGISTRY:
                    problems.append(f"manifest references unknown asset {asset_id}")
                for rel, digest in record.get("sha256", {}).items():
                    path = paper_root() / rel
                    if not path.is_file():
                        problems.append(f"manifest: {asset_id} output gone: {rel}")
                    elif cio.sha256_file(path) != digest:
                        problems.append(f"manifest: {asset_id} sha mismatch: {rel}")
        except (json.JSONDecodeError, OSError) as exc:
            problems.append(f"generation manifest unreadable: {exc}")

    if problems:
        for problem in problems:
            logger.error(problem)
        logger.error("verify FAILED (%d problems)", len(problems))
        return 1
    logger.info("verify OK: %d assets, all outputs present and consistent", len(ASSET_REGISTRY))
    return 0


if __name__ == "__main__":
    sys.exit(main())

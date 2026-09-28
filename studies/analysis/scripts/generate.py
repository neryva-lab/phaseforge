"""Generate publication assets: ``python -m studies.analysis.scripts.generate``.

Options mirror the runner's philosophy: fail-closed coverage checks, a
``--check`` dry mode, and a generation manifest so every export is traceable
to the exact run data.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from studies.analysis.assets import ASSET_REGISTRY, load_generator, specs_by_section
from studies.analysis.common import io as cio
from studies.analysis.common.config import generation_manifest_path, paper_root
from studies.analysis.dataset import build_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s | %(message)s")
logger = logging.getLogger("studies.analysis.generate")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset", default=None, help="Comma/space list of asset ids (e.g. 'T1,F2'). Default: all."
    )
    parser.add_argument("--section", choices=("main", "appendix"), default=None)
    parser.add_argument("--figures-only", action="store_true")
    parser.add_argument("--tables-only", action="store_true")
    parser.add_argument(
        "--check", action="store_true", help="Validate coverage + planned assets; render nothing."
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Do not fail on incomplete run coverage (dev iteration).",
    )
    return parser.parse_args(argv)


def _select(args: argparse.Namespace) -> list:
    specs = specs_by_section(args.section)
    if args.asset:
        wanted = [t.strip() for t in args.asset.replace(",", " ").split()]
        unknown = [t for t in wanted if t not in ASSET_REGISTRY]
        if unknown:
            logger.error("Unknown asset ids %s; valid: %s", unknown, sorted(ASSET_REGISTRY))
            sys.exit(2)
        specs = [ASSET_REGISTRY[t] for t in wanted]
    if args.figures_only:
        specs = [s for s in specs if s.kind == "figure"]
    if args.tables_only:
        specs = [s for s in specs if s.kind == "table"]
    return specs


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    specs = _select(args)

    dataset = build_dataset(
        strict=not args.allow_partial and not args.check, load_curves=not args.check
    )
    report = dataset.coverage()
    logger.info("coverage: %s", report.summary())
    if args.check:
        for spec in specs:
            kind = "schematic (manual)" if spec.kind == "schematic" else spec.kind
            logger.info("  %-4s %-9s %-6s %s", spec.id, kind, spec.priority, spec.title)
        return 0 if (report.ok or args.allow_partial) else 2

    outputs: dict[str, list[str]] = {}
    failures: list[str] = []
    for spec in specs:
        generator = load_generator(spec)
        if generator is None:
            logger.info("%-4s schematic — expecting manual placement: %s", spec.id, spec.outputs[0])
            continue
        try:
            paths = generator(dataset)
        except Exception as exc:  # noqa: BLE001 - report every asset failure
            logger.error("%-4s FAILED: %s", spec.id, exc)
            failures.append(spec.id)
            continue
        outputs[spec.id] = [str(p.relative_to(paper_root())) for p in paths]
        logger.info("%-4s -> %s", spec.id, ", ".join(outputs[spec.id]))

    manifest = {
        "version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "argv": sys.argv,
        "coverage": {
            "present_evals": report.present_evals,
            "missing_evals": [[t, m, s] for t, m, s in report.missing_evals],
            "missing_stage_runs": [[t, m, s, st] for t, m, s, st in report.missing_stage_runs],
        },
        "assets": {
            asset_id: {
                "outputs": rel_paths,
                "sha256": {rel: cio.sha256_file(paper_root() / rel) for rel in rel_paths},
            }
            for asset_id, rel_paths in outputs.items()
        },
    }
    manifest_path: Path = generation_manifest_path()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(__import__("json").dumps(manifest, indent=2), encoding="utf-8")
    logger.info("generation manifest: %s", manifest_path)
    if failures:
        logger.error("failed assets: %s", failures)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

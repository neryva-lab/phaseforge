"""Frozen-config check for Phase 1 confirmation runs (CPU-only).

Compares two ``resolved_config.yaml`` files (or the run directories that
contain them) and fails when anything but the intended axes differ.

Frozen (must match exactly):
  * ``data.source.task_name``, ``data.state_dim``, ``data.action_dim``
  * ``eval.mode`` (``rollout``), ``eval.bank.seed`` (2026),
    ``eval.bank.num_cases`` (50), ``eval.episodes.horizon``

Allowed to differ (reported, never failing):
  * ``project.seed`` (the 3-seed rerun uses 42/43/44 by design)
  * ``project.tag`` / ``project.method`` labels

Usage:
    python scripts/dev/freeze_check.py <runA> <runB>
where each arg is a run directory or a ``resolved_config.yaml`` path.
Exit 0 iff frozen keys match; exit 1 with a diff listing otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FROZEN_BANK_SEED = 2026
FROZEN_BANK_CASES = 50

#: (dotted key, expected value or None) pins beyond plain A==B equality.
_FROZEN_PINS: tuple[tuple[str, object], ...] = (
    ("eval.bank.seed", FROZEN_BANK_SEED),
    ("eval.bank.num_cases", FROZEN_BANK_CASES),
)

#: Dotted keys compared for A==B equality.
_FROZEN_EQUALITY_KEYS: tuple[str, ...] = (
    "data.source.task_name",
    "data.state_dim",
    "data.action_dim",
    "eval.mode",
    "eval.bank.seed",
    "eval.bank.num_cases",
    "eval.episodes.horizon",
)

#: Informational keys: reported when different, never failing.
_INFO_KEYS: tuple[str, ...] = (
    "project.seed",
    "project.tag",
    "project.method",
)


def _resolve_config_path(arg: str) -> Path:
    """Accept a run directory or a yaml path; return the yaml path."""
    path = Path(arg)
    if path.is_dir():
        candidate = path / "resolved_config.yaml"
        if not candidate.is_file():
            raise FileNotFoundError(f"No resolved_config.yaml in run dir {path}")
        return candidate
    if path.is_file():
        return path
    raise FileNotFoundError(f"Not a run dir or file: {arg}")


def _get_dotted(cfg: object, dotted: str, default: object = None) -> object:
    """Read ``a.b.c`` from an OmegaConf/dict config, default when absent."""
    current: object = cfg
    for part in dotted.split("."):
        get = getattr(current, "get", None)
        if callable(get):
            try:
                current = get(part, default)
            except Exception:
                return default
        elif isinstance(current, dict):
            current = current.get(part, default)
        else:
            return default
        if current is None:
            return default
    return current


def compare_resolved_configs(a_path: str | Path, b_path: str | Path) -> list[str]:
    """Compare two resolved configs; return mismatch descriptions (empty=pass)."""
    from omegaconf import OmegaConf

    a_cfg = OmegaConf.load(str(a_path))
    b_cfg = OmegaConf.load(str(b_path))
    mismatches: list[str] = []
    for key in _FROZEN_EQUALITY_KEYS:
        a_val = _get_dotted(a_cfg, key, default="__absent__")
        b_val = _get_dotted(b_cfg, key, default="__absent__")
        if a_val != b_val:
            mismatches.append(f"{key}: {a_val!r} != {b_val!r}")
    for key, pinned in _FROZEN_PINS:
        # Training configs (eval=metrics) carry no bank section at all;
        # absent-on-both means nothing to pin. Absent-on-one is a real
        # mismatch (different config kinds or a dropped section).
        a_val = _get_dotted(a_cfg, key, default="__absent__")
        b_val = _get_dotted(b_cfg, key, default="__absent__")
        if a_val == "__absent__" and b_val == "__absent__":
            continue
        for label, val in (("A", a_val), ("B", b_val)):
            if val != pinned:
                mismatches.append(f"{label} {key}={val!r} (frozen pin {pinned!r})")
    return mismatches


def _info_lines(a_path: Path, b_path: Path) -> list[str]:
    """Allowed-difference summary (informational only)."""
    from omegaconf import OmegaConf

    lines: list[str] = []
    try:
        a_cfg = OmegaConf.load(str(a_path))
        b_cfg = OmegaConf.load(str(b_path))
    except Exception:
        return lines
    for key in _INFO_KEYS:
        a_val = _get_dotted(a_cfg, key, default=None)
        b_val = _get_dotted(b_cfg, key, default=None)
        lines.append(f"info {key}: A={a_val!r} B={b_val!r}")
    for label, cfg_path in (("A", a_path), ("B", b_path)):
        meta = cfg_path.parent / "run_meta.json"
        if meta.is_file():
            try:
                info = json.loads(meta.read_text(encoding="utf-8"))
                lines.append(
                    f"info {label} run_meta: "
                    f"git_sha={info.get('git_sha')!r} "
                    f"config_hash={info.get('config_hash')!r}"
                )
            except (OSError, ValueError):
                lines.append(f"info {label} run_meta: unreadable")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a", help="Run dir or resolved_config.yaml (A)")
    parser.add_argument("run_b", help="Run dir or resolved_config.yaml (B)")
    args = parser.parse_args(argv)
    try:
        a_path = _resolve_config_path(args.run_a)
        b_path = _resolve_config_path(args.run_b)
    except FileNotFoundError as exc:
        print(f"freeze_check FAILED: {exc}", file=sys.stderr)
        return 2
    try:
        mismatches = compare_resolved_configs(a_path, b_path)
    except Exception as exc:
        print(f"freeze_check FAILED: cannot parse configs: {exc}", file=sys.stderr)
        return 2
    for line in _info_lines(a_path, b_path):
        print(line)
    if mismatches:
        print("freeze_check FAILED: frozen keys differ:")
        for item in mismatches:
            print(f"  - {item}")
        return 1
    print("freeze_check OK: frozen keys match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

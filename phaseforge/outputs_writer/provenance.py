"""Data provenance copy and per-run artifact manifest.

Two provenance artifacts defined by the final specification:

* ``metadata/data_provenance.json`` — a copy of the cache manifest's
  ``provenance`` block (raw-file SHA-256 values, split demo keys, schema,
  normalization, phase-labeler config, environment metadata) plus the cache
  identity it came from. A data config hash alone is not a sufficient
  provenance record.
* ``metadata/artifact_manifest.json`` — SHA-256 and byte size for every
  paper input produced by the run (resolved config, run metadata,
  environment, data provenance, timings, curves, summary, selected
  checkpoint, eval/episode records). It excludes itself and lock files, is
  written only after those inputs are closed, and is flagged ``complete``
  only when every required input exists and was hashed.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file (handles multi-GB artifacts)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _iso_now() -> str:
    return datetime.now(UTC).isoformat()


def copy_cache_provenance(
    run_dir: str | Path,
    cache_root: str | Path,
    config_hash: str,
) -> dict[str, Any]:
    """Copy a cache manifest's provenance block into a run directory.

    Writes ``metadata/data_provenance.json`` containing the cache identity
    (``config_hash``, source manifest path, copy timestamp) and the full
    ``provenance`` block from the manifest.

    Raises:
        FileNotFoundError: the manifest for ``config_hash`` does not exist.
    """
    run_dir = Path(run_dir)
    manifest_path = Path(cache_root) / config_hash / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Cache manifest not found: {manifest_path}. Cannot copy the "
            "data provenance into the run directory."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    provenance = manifest.get("provenance", {})
    payload: dict[str, Any] = {
        "config_hash": config_hash,
        "source_manifest": str(manifest_path),
        "copied_at": _iso_now(),
        "provenance": provenance,
    }
    metadata_dir = run_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    path = metadata_dir / "data_provenance.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return payload


def write_artifact_manifest(
    run_dir: str | Path,
    inputs: dict[str, str | Path | None],
) -> dict[str, Any]:
    """Hash every paper input and write ``metadata/artifact_manifest.json``.

    Args:
        run_dir: The run directory (manifest written under ``metadata/``).
        inputs: Mapping of artifact name to its path **relative to
            ``run_dir``**, or ``None`` when the artifact is expected but
            absent (that artifact is then recorded as missing and the
            manifest is ``complete: false``).

    The manifest itself and any lock files are never included. Returns the
    manifest dict.
    """
    run_dir = Path(run_dir)
    artifacts: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for name, rel in inputs.items():
        entry: dict[str, Any] = {"path": str(rel) if rel is not None else None}
        if rel is None:
            entry["present"] = False
            missing.append(name)
        else:
            path = run_dir / rel
            if not path.is_file():
                entry["present"] = False
                missing.append(name)
            else:
                stat = path.stat()
                entry.update(
                    {
                        "present": True,
                        "size": int(stat.st_size),
                        "sha256": sha256_file(path),
                    }
                )
        artifacts[name] = entry

    payload: dict[str, Any] = {
        "run_dir": str(run_dir),
        "written_at": _iso_now(),
        "complete": not missing,
        "artifacts": artifacts,
    }
    if missing:
        payload["missing"] = sorted(missing)

    metadata_dir = run_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    path = metadata_dir / "artifact_manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


__all__ = ["sha256_file", "copy_cache_provenance", "write_artifact_manifest"]

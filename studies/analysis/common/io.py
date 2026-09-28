"""Shared I/O helpers: atomic writes and JSONL iteration."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def to_long_path(path: Path | str) -> Path:
    s = str(path)
    if os.name == "nt" and not s.startswith("\\\\?\\"):
        abs_p = os.path.abspath(s)
        return Path("\\\\?\\" + abs_p)
    return Path(path)


def read_json(path: Path) -> Any:
    p = to_long_path(path)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON {path}: {exc}") from exc


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    p = to_long_path(path)
    try:
        with p.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{lineno}: bad JSON ({exc})") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{lineno}: expected a JSON object")
                yield record
    except OSError as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc


def sha256_file(path: Path) -> str:
    import hashlib

    p = to_long_path(path)
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()

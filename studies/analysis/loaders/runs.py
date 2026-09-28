"""Run discovery: map (namespace, task, method, seed, stage) -> run directory.

Layout produced by the runner (verified against real sweeps)::

    <root>/<model_name>/stage<N>/seed<S>/<ts>_<tag>_<runid>/    train runs
    <root>/eval/<model_name>/seed<S>/<ts>_<tag>_<runid>/        eval runs

Each train/eval run carries ``run_meta.json``; completion is marked by a
``.completed`` sibling file. Discovery is fail-closed: duplicate completed
runs for one cell and unreadable metadata abort with the offending paths.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from studies.analysis.common import io as cio

_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_(?P<tag>.+?)_(?P<runid>[0-9a-f]{8})$")


@dataclass(frozen=True)
class TrainRun:
    namespace: str
    model_name: str
    task: str | None
    method: str
    seed: int
    stage: int
    path: Path
    git_commit: str | None
    config_hash: str | None
    tag: str | None

    @property
    def key(self) -> tuple[str, str | None, str, int, int]:
        return (self.namespace, self.task, self.method, self.seed, self.stage)


@dataclass(frozen=True)
class EvalRun:
    namespace: str
    model_name: str
    task: str | None
    method: str
    seed: int
    path: Path
    git_commit: str | None
    tag: str | None

    @property
    def key(self) -> tuple[str, str | None, str, int]:
        return (self.namespace, self.task, self.method, self.seed)


def _parse_dir_name(name: str) -> tuple[str, str] | None:
    m = _DIR_RE.match(name)
    if m is None:
        return None
    return m.group("tag"), m.group("runid")


def _task_from_tag(tag: str | None) -> str | None:
    if not tag:
        return None
    return tag.split("__", 1)[0] or None


def _resolve_task(namespace: str, method: str, tag: str | None) -> str | None:
    """Task of a run, resolved through the manifest (name, tag) identity."""
    from studies.analysis.common import registry

    for m in registry.methods(namespace):
        if m.name == method and (m.tag or None) == (tag or None):
            return m.task
    if tag:
        clean_tag = tag.split("__", 1)[0]
        if clean_tag in ("Lift", "Can", "Square", "ToolHang", "Transport"):
            return clean_tag
    return _task_from_tag(tag)


def _find_runner_roots(root: Path) -> list[Path]:
    """Return all directories that act as output runner roots (contain 'eval' or model dirs)."""
    p_root = cio.to_long_path(root)
    if not p_root.is_dir():
        return []
    roots: list[Path] = []
    import os
    for dirpath, dirnames, _ in os.walk(str(p_root)):
        if "eval" in dirnames:
            dp = Path(dirpath)
            roots.append(dp)
    return roots if roots else [p_root]


def scan_namespace(namespace: str, root: Path) -> tuple[list[TrainRun], list[EvalRun]]:
    """Discover every completed run under one output namespace."""
    p_root = cio.to_long_path(root)
    if not p_root.is_dir():
        raise ValueError(f"Namespace root {root} does not exist (namespace {namespace!r})")

    runner_roots = _find_runner_roots(p_root)
    raw_train: list[TrainRun] = []
    raw_eval: list[EvalRun] = []
    for r_root in runner_roots:
        for run_dir in sorted(p for p in r_root.iterdir() if p.is_dir() and p.name not in ("eval", "_ledger", "_results", "_runner")):
            _scan_train_tree(namespace, run_dir, raw_train)
        eval_root = r_root / "eval"
        if eval_root.is_dir():
            for model_dir in sorted(p for p in eval_root.iterdir() if p.is_dir()):
                _scan_eval_tree(namespace, model_dir, raw_eval)

    # Deduplicate newest-wins for runs sharing the same cell key
    train_by_key: dict[tuple, TrainRun] = {}
    for r in raw_train:
        if r.key not in train_by_key or r.path.name > train_by_key[r.key].path.name:
            train_by_key[r.key] = r

    eval_by_key: dict[tuple, EvalRun] = {}
    for r in raw_eval:
        if r.key not in eval_by_key or r.path.name > eval_by_key[r.key].path.name:
            eval_by_key[r.key] = r

    train_runs = list(train_by_key.values())
    eval_runs = list(eval_by_key.values())
    return train_runs, eval_runs


def _meta_or_skip(run_dir: Path) -> dict | None:
    """Return run_meta for a completed run, else None (incomplete runs are skipped)."""
    p_run = cio.to_long_path(run_dir)
    comp = p_run.parent / f"{p_run.name}.completed"
    if not p_run.is_dir() or not comp.exists():
        return None
    try:
        meta = cio.read_json(p_run / "run_meta.json")
    except ValueError:
        return None
    return meta if isinstance(meta, dict) else None


def _scan_train_tree(namespace: str, model_dir: Path, out: list[TrainRun]) -> None:
    p_model = cio.to_long_path(model_dir)
    model_name = p_model.name
    for stage_dir in sorted(p_model.glob("stage*")):
        m = re.fullmatch(r"stage(\d+)", stage_dir.name)
        if m is None or not stage_dir.is_dir():
            continue
        stage = int(m.group(1))
        for seed_dir in sorted(p for p in stage_dir.iterdir() if p.is_dir()):
            for run_dir in sorted(p for p in seed_dir.iterdir() if p.is_dir()):
                meta = _meta_or_skip(run_dir)
                if meta is None:
                    continue
                tag = meta.get("tag")
                method_name = str(meta.get("method") or meta.get("method_name") or model_name)
                out.append(
                    TrainRun(
                        namespace=namespace,
                        model_name=str(meta.get("model_name", model_name)),
                        task=_resolve_task(namespace, method_name, tag),
                        method=method_name,
                        seed=int(meta["seed"]) if "seed" in meta else -1,
                        stage=stage if "stage" in meta else int(meta.get("stage", stage)),
                        path=run_dir,
                        git_commit=meta.get("git_commit"),
                        config_hash=meta.get("config_hash"),
                        tag=tag,
                    )
                )


def _scan_eval_tree(namespace: str, model_dir: Path, out: list[EvalRun]) -> None:
    p_model = cio.to_long_path(model_dir)
    model_name = p_model.name
    for seed_dir in sorted(p for p in p_model.iterdir() if p.is_dir()):
        for run_dir in sorted(p for p in seed_dir.iterdir() if p.is_dir()):
            meta = _meta_or_skip(run_dir)
            if meta is None:
                continue
            tag = meta.get("tag")
            method_name = str(meta.get("method") or meta.get("method_name") or model_name)
            out.append(
                EvalRun(
                    namespace=namespace,
                    model_name=str(meta.get("model_name", model_name)),
                    task=_resolve_task(namespace, method_name, tag),
                    method=method_name,
                    seed=int(meta["seed"]) if "seed" in meta else -1,
                    path=run_dir,
                    git_commit=meta.get("git_commit"),
                    tag=tag,
                )
            )


def assert_no_duplicates(train_runs: list[TrainRun], eval_runs: list[EvalRun]) -> None:
    """Fail closed on two completed runs claiming the same cell."""
    seen: dict[tuple, Path] = {}
    for run in train_runs:
        if run.key in seen:
            raise ValueError(
                f"Duplicate completed train run for {run.key}: {seen[run.key]} and {run.path}"
            )
        seen[run.key] = run.path
    seen_eval: dict[tuple, Path] = {}
    for run in eval_runs:
        if run.key in seen_eval:
            raise ValueError(
                f"Duplicate completed eval run for {run.key}: {seen_eval[run.key]} and {run.path}"
            )
        seen_eval[run.key] = run.path

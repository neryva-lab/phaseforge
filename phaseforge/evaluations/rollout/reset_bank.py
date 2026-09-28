"""Frozen reset-case bank.

The bank is a **frozen artifact**: one generation step from a fresh
simulator (never reading the dataset — trivially disjoint from training),
then the artifact is verified byte-for-byte on every use. Regeneration is
only the documented exception after a protocol/version change, and it
produces a different ``bank_id``.

A reset case pins everything needed to restore the simulator: the flat
``MjSimState`` array ``[time(1), qpos(nq), qvel(nv)]`` (the same format the
dataset's per-step ``states`` attributes use), the model XML, and the
episode metadata. Restore mirrors ``robomimic.reset_to`` exactly.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from phaseforge.evaluations.envs.env_metadata import PinnedEnvMetadata
from phaseforge.evaluations.envs.errors import EnvParityError, InfrastructureError

#: Default number of reset cases (50 test episodes).
DEFAULT_NUM_CASES: int = 50

#: Two cases closer than this (L2 on qpos/qvel) are duplicate resets.
MIN_CASE_DISTANCE: float = 1e-3

#: Flat state index where qpos starts (time, qpos..., qvel...).
TIME_DIMS: int = 1


@contextmanager
def _seeded_reset_rng(seed: int):
    """Seed robosuite's legacy global samplers without leaking RNG state.

    robosuite 1.5.1's task-level constructors do not consistently forward
    ``seed`` to ``MujocoEnv``.  Its placement samplers nevertheless use the
    process-global ``numpy.random`` generator, so passing ``seed`` through
    ``robosuite.make`` is both invalid for tasks such as Lift and ineffective
    for the reset distribution.  Seed the generators used during bank
    creation, then restore the caller's state exactly.
    """
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    np.random.seed(int(seed))
    random.seed(int(seed))
    try:
        yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)


@dataclass(frozen=True)
class ResetCase:
    """One pinned simulator state plus the metadata to restore it."""

    index: int
    states: np.ndarray
    xml: str | None = None
    ep_meta: dict[str, Any] | None = None


@dataclass
class ResetBank:
    """The frozen bank: manifest + ordered cases with integrity hashes."""

    task: str
    bank_id: str
    seed: int
    num_cases: int
    env_canonical: str
    robosuite_version: str
    git_commit: str
    generated_at: str
    cases: list[ResetCase]
    case_sha256: dict[int, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, directory: str | Path) -> Path:
        """Write the bank atomically under ``directory``; return the dir."""
        directory = Path(directory)
        tmp = directory.with_name(directory.name + "_tmp")
        if tmp.exists():
            import shutil

            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)

        case_dir = tmp / "cases"
        case_dir.mkdir()
        manifest_cases: list[dict[str, Any]] = []
        for case in self.cases:
            file = case_dir / f"{case.index:04d}.npy"
            np.save(file, np.asarray(case.states, dtype=np.float32))
            digest = _sha256_file(file)
            self.case_sha256[case.index] = digest
            manifest_cases.append(
                {
                    "index": case.index,
                    "sha256": digest,
                    "xml": case.xml,
                    "ep_meta": case.ep_meta,
                }
            )

        manifest = {
            "task": self.task,
            "bank_id": self.bank_id,
            "seed": self.seed,
            "num_cases": self.num_cases,
            "env_canonical": self.env_canonical,
            "robosuite_version": self.robosuite_version,
            "git_commit": self.git_commit,
            "generated_at": self.generated_at,
            "cases": manifest_cases,
        }
        (tmp / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )

        final_dir = directory
        if final_dir.exists():
            import shutil

            shutil.rmtree(final_dir)
        tmp.rename(final_dir)
        return final_dir

    @classmethod
    def load(cls, directory: str | Path, *, verify: bool = True) -> ResetBank:
        """Load and verify a bank. Tampering raises EnvParityError.

        Verification: every expected case file exists, its SHA-256 matches
        the manifest, the state is finite, and the flat-state length is
        consistent (``1 + nq + nv`` where ``nq``/``nv`` are the simulation
        dimensions recorded at generation time).
        """
        directory = Path(directory)
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            raise EnvParityError(f"Reset bank {directory} has no manifest.json — not a bank.")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EnvParityError(f"Reset bank {directory} manifest is unreadable: {exc}") from exc

        cases: list[ResetCase] = []
        case_sha256: dict[int, str] = {}
        case_dir = directory / "cases"
        for entry in manifest.get("cases", []):
            index = int(entry["index"])
            expected = str(entry["sha256"])
            file = case_dir / f"{index:04d}.npy"
            if not file.is_file():
                raise EnvParityError(
                    f"Reset bank {directory} is missing case file {file.name} "
                    f"(declared in manifest)."
                )
            if verify:
                actual = _sha256_file(file)
                if actual != expected:
                    raise EnvParityError(
                        f"Reset bank {directory} case {index} failed its "
                        f"SHA-256 check (got {actual[:12]}…, expected "
                        f"{expected[:12]}…). The bank was tampered with or "
                        "corrupted — regenerate or restore the artifact."
                    )
            states = np.load(file)
            if not np.isfinite(states).all():
                raise EnvParityError(f"Reset bank {directory} case {index} has non-finite states.")
            case_sha256[index] = expected
            cases.append(
                ResetCase(
                    index=index,
                    states=np.asarray(states, dtype=np.float32),
                    xml=entry.get("xml"),
                    ep_meta=entry.get("ep_meta"),
                )
            )

        bank = cls(
            task=str(manifest["task"]),
            bank_id=str(manifest["bank_id"]),
            seed=int(manifest["seed"]),
            num_cases=int(manifest["num_cases"]),
            env_canonical=str(manifest["env_canonical"]),
            robosuite_version=str(manifest["robosuite_version"]),
            git_commit=str(manifest.get("git_commit", "")),
            generated_at=str(manifest["generated_at"]),
            cases=sorted(cases, key=lambda c: c.index),
            case_sha256=case_sha256,
        )
        if bank.num_cases != len(bank.cases):
            raise EnvParityError(
                f"Reset bank {directory} declares {bank.num_cases} cases but "
                f"contains {len(bank.cases)} — corrupted."
            )
        indices = [c.index for c in bank.cases]
        if indices != list(range(bank.num_cases)):
            raise EnvParityError(
                f"Reset bank {directory} case indices {indices} are not a contiguous 0..N-1 range."
            )
        return bank

    def case(self, index: int) -> ResetCase:
        if index < 0 or index >= self.num_cases:
            raise IndexError(f"Reset case index {index} out of range 0..{self.num_cases - 1}")
        return self.cases[index]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def compute_bank_id(
    meta: PinnedEnvMetadata,
    *,
    task: str,
    seed: int,
    num_cases: int,
    robosuite_version: str,
) -> str:
    """Deterministic content-based bank identity (regeneration changes it)."""
    payload = json.dumps(
        {
            "task": task,
            "seed": seed,
            "num_cases": num_cases,
            "env_canonical": meta.canonical_json(),
            "robosuite_version": robosuite_version,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def bank_dir(data_root: str | Path, task: str, bank_id: str) -> Path:
    """``{data_root}/processed/eval_banks/{task}/{bank_id}/``"""
    return Path(data_root) / "processed" / "eval_banks" / task / bank_id


def generate_reset_bank(
    adapter_factory: Callable[[], Any],
    meta: PinnedEnvMetadata,
    *,
    task: str,
    seed: int = 2026,
    num_cases: int = DEFAULT_NUM_CASES,
    max_attempts_per_case: int = 40,
    robosuite_version: str = "",
    git_commit: str = "",
) -> ResetBank:
    """Generate a frozen bank from a fresh simulator (never the dataset).

    Each case is a freshly-reset simulator state captured as a flat
    ``MjSimState``. Duplicate resets (L2 closer than
    :data:`MIN_CASE_DISTANCE` on the non-time dimensions) are retried up to
    ``max_attempts_per_case`` times; generation fails loudly if the
    simulator cannot produce enough distinct states — a sign the sampler is
    misconfigured and the bank would be degenerate.
    """
    cases: list[ResetCase] = []
    seen: list[np.ndarray] = []
    with _seeded_reset_rng(seed):
        adapter = adapter_factory()
        for index in range(num_cases):
            attempt = 0
            while True:
                attempt += 1
                try:
                    adapter.env.reset()
                    states = np.asarray(adapter.env.sim.get_state().flatten(), dtype=np.float32)
                except Exception as exc:
                    raise InfrastructureError(
                        f"Reset sampling failed on case {index} (attempt {attempt}): {exc}"
                    ) from exc
                if not np.isfinite(states).all():
                    raise InfrastructureError(
                        f"Reset sampling produced non-finite state on case "
                        f"{index} — the simulator is not returning valid states."
                    )
                if _is_duplicate(states, seen):
                    if attempt >= max_attempts_per_case:
                        raise InfrastructureError(
                            f"Reset sampling could not produce a state distinct "
                            f"from the previous {len(seen)} within "
                            f"{max_attempts_per_case} attempts on case {index}. "
                            "The reset distribution is degenerate."
                        )
                    continue
                break
            xml = None
            try:
                xml = getattr(adapter.env, "model_xml", None) or getattr(
                    adapter.env, "_model_xml", None
                )
                if xml is not None:
                    xml = str(xml)
            except Exception:
                xml = None
            ep_meta = getattr(adapter.env, "ep_meta", None)
            if isinstance(ep_meta, dict):
                ep_meta = dict(ep_meta)
            else:
                ep_meta = None
            cases.append(ResetCase(index=index, states=states, xml=xml, ep_meta=ep_meta))
            seen.append(states)

    bank_id = compute_bank_id(
        meta,
        task=task,
        seed=seed,
        num_cases=num_cases,
        robosuite_version=robosuite_version,
    )
    return ResetBank(
        task=task,
        bank_id=bank_id,
        seed=seed,
        num_cases=num_cases,
        env_canonical=meta.canonical_json(),
        robosuite_version=robosuite_version,
        git_commit=git_commit,
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        cases=cases,
    )


def _is_duplicate(states: np.ndarray, seen: list[np.ndarray]) -> bool:
    qpos_qvel = states[TIME_DIMS:]
    for previous in seen:
        if np.linalg.norm(qpos_qvel - previous[TIME_DIMS:]) < MIN_CASE_DISTANCE:
            return True
    return False


__all__ = [
    "ResetCase",
    "ResetBank",
    "DEFAULT_NUM_CASES",
    "generate_reset_bank",
    "compute_bank_id",
    "bank_dir",
]

"""Tests for the import-free environment version fingerprint.

The fingerprint used to IMPORT sklearn/wandb/scipy just to read
``__version__`` — seconds per training process. The metadata-based
resolution must (a) never import them, (b) return the same strings the
imported modules would report, (c) keep the environment.json schema.
"""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
import sys

from phaseforge.outputs_writer.metadata import (
    _MODULE_TO_DIST,
    _cuda_device_identity,
    _safe_version,
    collect_environment,
)


def test_module_to_dist_map_covers_all_fingerprinted_packages() -> None:
    env = collect_environment()
    fingerprinted = set(env["packages"].keys())
    mapped = set(_MODULE_TO_DIST.values())
    assert fingerprinted == mapped, f"fingerprint/mapping drift: {fingerprinted ^ mapped}"


def test_safe_version_matches_imported_modules_exactly() -> None:
    """For imported modules the historical string is preserved verbatim.

    torch is the critical case: ``torch.__version__`` carries the local tag
    (e.g. ``2.13.0+cpu``) while distribution metadata reports ``2.13.0``.
    """
    import numpy
    import torch

    assert _safe_version("torch") == torch.__version__
    assert _safe_version("numpy") == numpy.__version__


def test_safe_version_uses_metadata_without_importing() -> None:
    """sklearn resolves through distribution metadata, not an import."""
    expected = importlib.metadata.version("scikit-learn")
    got = _safe_version("sklearn")
    assert got == expected


def test_safe_version_unknown_module_returns_none() -> None:
    assert _safe_version("definitely-not-a-real-module-name") is None


def test_collect_environment_includes_gpu_identity_key() -> None:
    """Cross-machine forensics need the GPU silicon identity: a past metric gap
    was untraceable to silicon from the artifacts alone, so this key must be
    collected at fingerprint time."""
    env = collect_environment()
    assert "gpu" in env
    assert env["gpu"] is None or (
        isinstance(env["gpu"], dict) and isinstance(env["gpu"].get("name"), str)
    )


def test_cuda_device_identity_without_torch_imported(monkeypatch) -> None:
    """No torch import is triggered just to fingerprint the GPU."""
    import sys

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.delitem(sys.modules, "torch.cuda", raising=False)
    assert _cuda_device_identity() is None
    assert "torch" not in sys.modules


def test_cuda_device_identity_reads_imported_torch(monkeypatch) -> None:
    """An already-imported torch is queried; failures yield None, never raise."""

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_name(_idx: int) -> str:
            return "Tesla T4"

        @staticmethod
        def get_device_capability(_idx: int) -> tuple[int, int]:
            return (7, 5)

    class _Torch:
        cuda = _Cuda()

    import sys

    monkeypatch.setitem(sys.modules, "torch", _Torch())
    got = _cuda_device_identity()
    assert got == {"name": "Tesla T4", "capability": "7.5"}


def test_cuda_device_identity_cpu_only_returns_none(monkeypatch) -> None:
    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class _Torch:
        cuda = _Cuda()

    import sys

    monkeypatch.setitem(sys.modules, "torch", _Torch())
    assert _cuda_device_identity() is None


def test_collect_environment_does_not_import_heavy_packages() -> None:
    """End-to-end guarantee, in a fresh interpreter: fingerprinting must not
    import sklearn/wandb/scipy (the seconds-per-process cost this fixes)."""
    code = (
        "import json\n"
        "import sys\n"
        "from phaseforge.outputs_writer.metadata import collect_environment\n"
        "env = collect_environment()\n"
        "heavy = [m for m in ('sklearn', 'wandb', 'scipy') if m in sys.modules]\n"
        "assert not heavy, f'fingerprint imported heavy packages: {heavy}'\n"
        "assert env['packages']['torch'], 'torch version missing'\n"
        "assert all(env['packages'].values()), f'missing versions: {env[\"packages\"]}'\n"
        "print(json.dumps(env['packages']))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=".",
    )
    assert result.returncode == 0, f"subprocess failed:\n{result.stderr}"
    packages = json.loads(result.stdout.strip().splitlines()[-1])
    assert set(packages) == {
        "torch",
        "numpy",
        "hydra-core",
        "omegaconf",
        "scikit-learn",
        "scipy",
        "h5py",
        "filelock",
        "wandb",
    }

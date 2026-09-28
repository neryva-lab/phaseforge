"""HuggingFace provisioning of raw robomimic dataset artifacts.

The ingestion FSM stays fail-closed by default (``VALIDATE_SOURCE`` raises
when the raw directory is missing). Setting ``data.source.auto_download=true``
routes the FSM through the ``PROVISION_SOURCE`` state, which downloads the
configured artifact from the HuggingFace mirror and verifies its SHA-256
(pinned in config, else the mirror's own LFS metadata) before the ingester
ever touches it.

Idempotency contract: an existing file whose checksum matches is left
untouched; a checksum mismatch is a hard error and is never silently
overwritten. When no checksum is available (no pinned value and no mirror
LFS metadata) the file is sanity-checked as a readable HDF5 archive instead.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _hdf5_sanity_check(path: Path) -> None:
    """Raise ``RuntimeError`` when ``path`` is not a readable HDF5 archive."""
    try:
        import h5py

        with h5py.File(path, "r"):
            pass
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"{path} is not a readable HDF5 file: {exc}") from exc


def fetch_mirror_sha256(repo_id: str, path: str) -> str | None:
    """Return the mirror's recorded SHA-256 for a dataset file (LFS), if any.

    Raises:
        RuntimeError: When the metadata cannot be read (network failure,
            missing artifact, ...). Callers decide whether that is fatal.
    """
    try:
        from huggingface_hub import get_paths_info
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "huggingface_hub is required for auto-download provisioning; "
            "install it or provision the dataset manually."
        ) from exc
    try:
        infos = get_paths_info(repo_id=repo_id, paths=path, repo_type="dataset")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Could not read HuggingFace metadata for {repo_id}/{path} — "
            f"is data.source.huggingface.path correct? ({exc})"
        ) from exc
    if not infos:
        return None
    lfs = getattr(infos[0], "lfs", None)
    sha = getattr(lfs, "sha256", None) if lfs is not None else None
    return str(sha) if sha else None


def download_hf_file(
    repo_id: str,
    path: str,
    dest_dir: str | Path,
    pinned_sha256: str | None = None,
) -> Path:
    """Download a dataset file into ``dest_dir`` (flat basename), SHA-verified.

    The expected checksum is the pinned value when given, else the mirror's
    own LFS metadata; when neither is available the file is sanity-checked as
    a readable HDF5 archive. An existing file with a matching checksum is
    returned untouched; a checksum mismatch is a hard error (never an
    overwrite). When the mirror metadata cannot be read but the file already
    exists, the existing file is sanity-checked instead of failing (re-runs
    stay robust to network hiccups); a fresh download still fails closed.

    Returns:
        The verified destination path.

    Raises:
        RuntimeError: Download failure, checksum mismatch, or unreadable file.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "huggingface_hub is required for auto-download provisioning; "
            "install it or provision the dataset manually."
        ) from exc

    dest_dir = Path(dest_dir)
    name = Path(path).name
    if not name:
        raise RuntimeError(
            f"HuggingFace artifact path {path!r} has no file name; "
            "data.source.huggingface.path must point at a file."
        )
    dest = dest_dir / name
    if dest.is_dir():
        raise RuntimeError(f"{dest} is a directory — {path!r} must name a file, not a folder.")
    dest_dir.mkdir(parents=True, exist_ok=True)

    expected = pinned_sha256
    if expected is None:
        try:
            expected = fetch_mirror_sha256(repo_id, path)
        except RuntimeError as exc:
            if not dest.exists():
                raise
            logger.warning(
                "  mirror metadata unavailable (%s) — falling back to an "
                "HDF5 sanity check for the existing file.",
                exc,
            )
            expected = None

    if dest.exists():
        if expected is not None:
            actual = _sha256_of(dest)
            if actual == expected:
                logger.info("  already provisioned: %s (SHA-256 verified)", dest)
                return dest
            raise RuntimeError(
                f"Existing file {dest} has SHA-256 {actual}, expected {expected}. "
                "Refusing to overwrite — remove or repair it manually."
            )
        _hdf5_sanity_check(dest)
        logger.info("  already provisioned: %s (HDF5 sanity check passed)", dest)
        return dest

    logger.info("  downloading %s/%s -> %s", repo_id, path, dest)
    try:
        cached = hf_hub_download(repo_id=repo_id, filename=path, repo_type="dataset")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to download {repo_id}/{path} from HuggingFace: {exc}") from exc
    shutil.copy2(cached, dest)

    if expected is not None:
        actual = _sha256_of(dest)
        if actual != expected:
            dest.unlink(missing_ok=True)
            raise RuntimeError(
                f"Downloaded file {dest} has SHA-256 {actual}, expected "
                f"{expected}. The mirror artifact changed — refusing to ingest."
            )
        logger.info("  downloaded %s (SHA-256 verified: %s)", dest.name, actual)
    else:
        try:
            _hdf5_sanity_check(dest)
        except RuntimeError:
            dest.unlink(missing_ok=True)
            raise
        logger.info("  downloaded %s (HDF5 sanity check passed)", dest.name)

    return dest

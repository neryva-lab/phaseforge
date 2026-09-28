"""Unit tests for the HuggingFace downloader (no network: monkeypatched).

Covers the idempotency contract and every failure path of
``download_hf_file`` plus the mirror-metadata helper.
"""

import h5py
import pytest

from phaseforge.data.ingestion import hf_downloader
from phaseforge.data.ingestion.hf_downloader import (
    _hdf5_sanity_check,
    _sha256_of,
    download_hf_file,
    fetch_mirror_sha256,
)

GARBAGE = b"not-an-hdf5-file"


def _make_real_hdf5(path) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset("demo_0/actions", data=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_verified_existing_file_is_noop(tmp_path, monkeypatch) -> None:
    dest = tmp_path / "low_dim_v15.hdf5"
    dest.write_bytes(b"payload")
    sha = _sha256_of(dest)
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda *a, **k: pytest.fail("must not download"),
    )

    result = download_hf_file(
        "amandlek/robomimic",
        "v1.5/lift/ph/low_dim_v15.hdf5",
        tmp_path,
        pinned_sha256=sha,
    )

    assert result == dest
    assert dest.read_bytes() == b"payload"


def test_existing_file_checksum_mismatch_preserves_file(tmp_path) -> None:
    dest = tmp_path / "low_dim_v15.hdf5"
    dest.write_bytes(b"payload")
    original = dest.read_bytes()

    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        download_hf_file(
            "amandlek/robomimic",
            "v1.5/lift/ph/low_dim_v15.hdf5",
            tmp_path,
            pinned_sha256="0" * 64,
        )

    assert dest.read_bytes() == original


def test_metadata_failure_falls_back_to_sanity_check(tmp_path, monkeypatch) -> None:
    dest = tmp_path / "low_dim_v15.hdf5"
    _make_real_hdf5(dest)

    def raise_meta(repo_id, path):
        raise RuntimeError("network down")

    monkeypatch.setattr(hf_downloader, "fetch_mirror_sha256", raise_meta)
    result = download_hf_file("amandlek/robomimic", "v1.5/lift/ph/low_dim_v15.hdf5", tmp_path)

    assert result == dest


def test_metadata_failure_is_fatal_when_no_file(tmp_path, monkeypatch) -> None:
    def raise_meta(repo_id, path):
        raise RuntimeError("network down")

    monkeypatch.setattr(hf_downloader, "fetch_mirror_sha256", raise_meta)
    with pytest.raises(RuntimeError, match="network down"):
        download_hf_file("amandlek/robomimic", "v1.5/lift/ph/low_dim_v15.hdf5", tmp_path)


def test_fresh_download_sha_verified(tmp_path, monkeypatch) -> None:
    cached = tmp_path / "cache" / "low_dim_v15.hdf5"
    cached.parent.mkdir()
    cached.write_bytes(b"payload")
    sha = _sha256_of(cached)

    monkeypatch.setattr(hf_downloader, "fetch_mirror_sha256", lambda *a: sha)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **k: str(cached))

    dest = download_hf_file("amandlek/robomimic", "v1.5/lift/ph/low_dim_v15.hdf5", tmp_path)

    assert dest.name == "low_dim_v15.hdf5"
    assert _sha256_of(dest) == sha


def test_fresh_download_checksum_mismatch_removes_file(tmp_path, monkeypatch) -> None:
    cached = tmp_path / "cache" / "low_dim_v15.hdf5"
    cached.parent.mkdir()
    cached.write_bytes(b"payload")

    monkeypatch.setattr(hf_downloader, "fetch_mirror_sha256", lambda *a: "f" * 64)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **k: str(cached))

    with pytest.raises(RuntimeError, match="refusing to ingest"):
        download_hf_file("amandlek/robomimic", "v1.5/lift/ph/low_dim_v15.hdf5", tmp_path)

    assert not (tmp_path / "low_dim_v15.hdf5").exists()


def test_fresh_download_sanity_checked_when_no_checksum(tmp_path, monkeypatch) -> None:
    cached = tmp_path / "cache" / "low_dim_v15.hdf5"
    cached.parent.mkdir()
    _make_real_hdf5(cached)

    monkeypatch.setattr(hf_downloader, "fetch_mirror_sha256", lambda *a: None)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **k: str(cached))

    dest = download_hf_file("amandlek/robomimic", "v1.5/lift/ph/low_dim_v15.hdf5", tmp_path)

    assert dest.exists()


def test_fresh_download_invalid_hdf5_removed(tmp_path, monkeypatch) -> None:
    cached = tmp_path / "cache" / "low_dim_v15.hdf5"
    cached.parent.mkdir()
    cached.write_bytes(GARBAGE)

    monkeypatch.setattr(hf_downloader, "fetch_mirror_sha256", lambda *a: None)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **k: str(cached))

    with pytest.raises(RuntimeError, match="not a readable HDF5"):
        download_hf_file("amandlek/robomimic", "v1.5/lift/ph/low_dim_v15.hdf5", tmp_path)

    assert not (tmp_path / "low_dim_v15.hdf5").exists()


def test_empty_basename_guard(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="no file name"):
        download_hf_file("amandlek/robomimic", "", tmp_path, pinned_sha256="0" * 64)


def test_directory_destination_guard(tmp_path) -> None:
    (tmp_path / "ph").mkdir()
    with pytest.raises(RuntimeError, match="must name a file"):
        download_hf_file("amandlek/robomimic", "v1.5/lift/ph", tmp_path)


def test_sanity_check_raises_on_garbage(tmp_path) -> None:
    bad = tmp_path / "bad.hdf5"
    bad.write_bytes(GARBAGE)
    with pytest.raises(RuntimeError, match="not a readable HDF5"):
        _hdf5_sanity_check(bad)


def test_fetch_mirror_sha256_import_error_message(tmp_path, monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "huggingface_hub":
            raise ImportError("no hub")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="huggingface_hub is required"):
        fetch_mirror_sha256("amandlek/robomimic", "v1.5/lift/ph/low_dim_v15.hdf5")

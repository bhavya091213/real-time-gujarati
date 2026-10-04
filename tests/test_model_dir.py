"""Tests for gujusub.model_dir — materialising a symlinked HF snapshot into real files."""

import os
from pathlib import Path

import pytest

from gujusub import model_dir


def _fake_hub_snapshot(root: Path) -> Path:
    """Mimic the huggingface_hub cache: snapshot files are symlinks into blobs/."""
    blobs = root / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "aaa").write_bytes(b"onnx-graph")
    (blobs / "bbb").write_bytes(b"external-weights")
    (blobs / "ccc").write_bytes(b"gitattributes")

    snap = root / "snapshots" / "rev1"
    (snap / "assets").mkdir(parents=True)
    os.symlink("../../../blobs/aaa", snap / "assets" / "encoder.onnx")
    os.symlink("../../../blobs/bbb", snap / "assets" / "encoder.weight")
    os.symlink("../../blobs/ccc", snap / ".gitattributes")
    return snap


def test_materialize_replaces_symlinks_with_real_files(tmp_path: Path) -> None:
    snap = _fake_hub_snapshot(tmp_path / "hub")
    dest = tmp_path / "out"

    result = model_dir.materialize_snapshot(snap, dest)

    assert result == dest
    onnx = dest / "assets" / "encoder.onnx"
    weight = dest / "assets" / "encoder.weight"
    assert onnx.is_file() and not onnx.is_symlink()
    assert weight.is_file() and not weight.is_symlink()
    assert onnx.read_bytes() == b"onnx-graph"
    assert weight.read_bytes() == b"external-weights"
    # Both files physically live in the same directory — what onnxruntime checks.
    assert onnx.resolve().parent == weight.resolve().parent == (dest / "assets").resolve()


def test_materialize_skips_hidden_entries(tmp_path: Path) -> None:
    snap = _fake_hub_snapshot(tmp_path / "hub")
    dest = model_dir.materialize_snapshot(snap, tmp_path / "out")
    assert not (dest / ".gitattributes").exists()


def test_materialize_is_idempotent(tmp_path: Path) -> None:
    snap = _fake_hub_snapshot(tmp_path / "hub")
    dest = tmp_path / "out"
    model_dir.materialize_snapshot(snap, dest)
    first_inode = (dest / "assets" / "encoder.onnx").stat().st_ino

    model_dir.materialize_snapshot(snap, dest)

    assert (dest / "assets" / "encoder.onnx").stat().st_ino == first_inode


def test_materialize_replaces_stale_partial_file(tmp_path: Path) -> None:
    snap = _fake_hub_snapshot(tmp_path / "hub")
    dest = tmp_path / "out"
    (dest / "assets").mkdir(parents=True)
    (dest / "assets" / "encoder.onnx").write_bytes(b"trunc")  # wrong size → stale

    model_dir.materialize_snapshot(snap, dest)

    assert (dest / "assets" / "encoder.onnx").read_bytes() == b"onnx-graph"


def test_materialize_falls_back_to_copy_when_link_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snap = _fake_hub_snapshot(tmp_path / "hub")

    def no_link(*_a, **_k):
        raise OSError("cross-device link")

    monkeypatch.setattr(model_dir.os, "link", no_link)
    dest = model_dir.materialize_snapshot(snap, tmp_path / "out")

    assert (dest / "assets" / "encoder.onnx").read_bytes() == b"onnx-graph"


def test_local_model_dir_uses_snapshot_download_and_repo_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snap = _fake_hub_snapshot(tmp_path / "hub")
    calls: list[str] = []

    def fake_snapshot_download(repo_id: str, **_k) -> str:
        calls.append(repo_id)
        return str(snap)

    monkeypatch.setattr(model_dir, "snapshot_download", fake_snapshot_download)

    dest = model_dir.local_model_dir("org/model-name", root=tmp_path / "root")

    assert calls == ["org/model-name"]
    assert dest == tmp_path / "root" / "org--model-name"
    assert (dest / "assets" / "encoder.onnx").is_file()


def test_default_root_honours_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GUJUSUB_MODEL_DIR", str(tmp_path / "custom"))
    assert model_dir.default_root() == tmp_path / "custom"

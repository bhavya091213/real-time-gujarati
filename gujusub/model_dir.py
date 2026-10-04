"""Materialise a Hugging Face snapshot into a directory of real files.

Why this exists: huggingface_hub's cache stores every file as a symlink into
``blobs/``. The IndicConformer encoder is an ONNX graph whose weights live in
~370 external-data sidecar files next to it. onnxruntime >= 1.24 validates
that every external-data path resolves *inside the model's own directory*;
with the symlinked cache they resolve into ``blobs/`` instead, and session
creation fails with::

    External data path validation failed for initializer ...

So before building a session we copy (hardlink when possible, so no extra
disk is used) the snapshot into a plain directory under
``~/.cache/gujusub``. Override the location with ``GUJUSUB_MODEL_DIR``.
"""

import logging
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

ENV_ROOT = "GUJUSUB_MODEL_DIR"
_DEFAULT_ROOT = "~/.cache/gujusub"


def default_root() -> Path:
    """Directory that holds materialised model copies."""
    return Path(os.environ.get(ENV_ROOT, _DEFAULT_ROOT)).expanduser()


def _is_hidden(rel: Path) -> bool:
    return any(part.startswith(".") for part in rel.parts)


def _is_current(target: Path, real: Path) -> bool:
    if not target.is_file() or target.is_symlink():
        return False
    return target.stat().st_size == real.stat().st_size


def _place(real: Path, target: Path) -> None:
    """Hardlink ``real`` to ``target`` (copy if the link fails), atomically."""
    tmp = target.with_name(target.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    try:
        os.link(real, tmp)
    except OSError:
        shutil.copy2(real, tmp)
    os.replace(tmp, target)


def materialize_snapshot(snapshot: Path, dest: Path) -> Path:
    """Mirror ``snapshot`` into ``dest`` with symlinks resolved to real files.

    Idempotent: files already present with the right size are left alone.
    Hidden entries (``.gitattributes``, ``.cache``) are skipped.
    """
    snapshot = Path(snapshot)
    dest = Path(dest)
    placed = 0
    for src in sorted(snapshot.rglob("*")):
        rel = src.relative_to(snapshot)
        if _is_hidden(rel):
            continue
        target = dest / rel
        if src.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        real = src.resolve()
        if _is_current(target, real):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        _place(real, target)
        placed += 1
    if placed:
        logger.info("materialised %d file(s) into %s", placed, dest)
    return dest


def local_model_dir(repo_id: str, root: Path | None = None) -> Path:
    """Download ``repo_id`` (cached by huggingface_hub) and return a symlink-free copy."""
    snapshot = Path(snapshot_download(repo_id))
    dest = (root or default_root()) / repo_id.replace("/", "--")
    return materialize_snapshot(snapshot, dest)

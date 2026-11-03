"""Shard IO.

Shards are NumPy archives plus a completion marker written last. A crash leaves
the marker absent, so resume will rebuild that shard instead of treating a
partial file as finished. Checksums cover the archive bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 3


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_savez(path: Path, **arrays: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.writing.npz"
    if tmp.exists():
        tmp.unlink()
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    return file_sha256(path)


def mark_complete(path: Path, checksum: str) -> None:
    marker = Path(str(path) + ".complete")
    tmp = marker.with_suffix(marker.suffix + ".tmp")
    tmp.write_text(checksum + "\n")
    os.replace(tmp, marker)


def is_complete(path: Path) -> bool:
    marker = Path(str(path) + ".complete")
    if not marker.is_file() or not path.is_file():
        return False
    checksum = marker.read_text().strip()
    return len(checksum) == 64 and checksum == file_sha256(path)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.writing"
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as blob:
        return {key: blob[key] for key in blob.files}

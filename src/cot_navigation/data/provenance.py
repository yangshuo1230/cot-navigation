"""Immutable generation identity and selected-world checkpoint provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cot_navigation.config import to_plain
from cot_navigation.data.shards import SCHEMA_VERSION, atomic_json, file_sha256


GENERATOR_VERSION = 3
PARAMETRIC_GENERATOR_VERSION = 4


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def prepare_generation(cfg) -> dict:
    """Fail before writing when an existing root belongs to a different dataset.

    Worker count and audit settings do not change content. Chunk size changes
    the storage layout and therefore requires a new root when resuming.
    """
    raw = to_plain(cfg)
    excluded = {"root", "workers", "chunk_size", "audit_fraction", "max_attempts", "graph_shape", "graph_seed"}
    # Resource guards do not alter a successfully prepared dataset.
    excluded |= {"max_prepared_paths", "max_prepared_path_nodes"}
    semantic = {k: v for k, v in raw.items() if k not in excluded}
    # False is the pre-v3 behavior. Omit defaults to retain old data identities;
    # enabling either change binds new data roots to the new semantics.
    for key in ("include_one_hop_queries", "context_holdout_first"):
        if not semantic.get(key):
            semantic.pop(key, None)
    if semantic.get("graph_family", "layered") == "layered":
        semantic.pop("graph_family", None)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": PARAMETRIC_GENERATOR_VERSION if cfg.knowledge == "parametric_graph" else GENERATOR_VERSION,
        "semantic": semantic,
    }
    expected = {**identity, "fingerprint": digest(identity), "chunk_size": cfg.chunk_size}
    root = Path(cfg.root)
    signature = root / "generation.json"
    if signature.exists():
        old = json.loads(signature.read_text())
        if old != expected:
            raise ValueError(f"generation configuration mismatch at {root}; use a new data.root (existing files were not changed)")
    elif root.exists() and any(root.iterdir()):
        raise ValueError(f"unversioned/nonempty data root {root}; use a new root rather than relabeling existing data")
    else:
        root.mkdir(parents=True, exist_ok=True)
        atomic_json(signature, expected)
    return expected


def dataset_fingerprint(root: Path, manifest: dict) -> str:
    """Exclude execution-only manifest fields such as workers and storage path."""
    signature = root / "generation.json"
    if not signature.exists():
        raise ValueError(f"missing generation provenance at {root}; regenerate this legacy dataset in a new root")
    generation = json.loads(signature.read_text())
    if generation["fingerprint"] != manifest.get("generation_fingerprint"):
        raise ValueError(f"manifest/generation fingerprint mismatch at {root}")
    return digest({"generation": generation["fingerprint"], "splits": manifest.get("splits")})


def world_fingerprint(root: Path) -> str:
    """Bind graph, node mapping, knowledge corpus, query splits and selection ids."""
    names = ["graph.npz", "walks.npz", "queries.npz", "paths.npz", "balanced_train.npy", "balanced_val.npy", "balanced_test.npy"]
    return digest({name: file_sha256(root / name) for name in names})

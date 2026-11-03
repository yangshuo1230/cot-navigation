"""Offline tokenization and verified reusable datasets.

Only flat token arrays and scalar row fields are persisted (no pickle). A
training process loads its protocol once, materializes its training rows in
RAM before the first update, then samples rows without graph search or IO.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np

from cot_navigation.config import filler_length
from cot_navigation.data.dataset import ContextStore, ParametricWorld
from cot_navigation.data.generate import generate_context, generate_parametric
from cot_navigation.data.provenance import digest, prepare_generation
from cot_navigation.data.shards import atomic_json, atomic_savez, file_sha256, is_complete, load_json, load_npz, mark_complete
from cot_navigation.data.validate import validate_context, validate_parametric
from cot_navigation.protocols.render import RenderedExample
from cot_navigation.protocols.tokenizer import Tokenizer

RENDERER_VERSION = 1
PROTOCOLS = ("direct", "path", "dfs_state", "bfs_state", "dfs_nodes", "bfs_nodes", "filler")
# Preserve legacy cache support, but prepare only active experiments by default.
DEFAULT_PROTOCOLS = ("direct", "path", "dfs_state")


def _identity(source, protocol, tok, filler_tokens):
    return {"renderer_version": RENDERER_VERSION, "source": source.fingerprint,
            "protocol": protocol, "vocab": tok.hash(),
            "filler_tokens": filler_tokens if protocol == "filler" else 0}


def _cache_path(source, identity):
    return source.root / "prepared" / f"{identity['protocol']}_{digest(identity)[:20]}.npz"


def _write_rows(source, protocol, tok, filler_tokens):
    identity = _identity(source, protocol, tok, filler_tokens)
    path = _cache_path(source, identity)
    if is_complete(path) and path.with_suffix(".json").exists() and load_json(path.with_suffix(".json")) == identity:
        return path
    tokens, offsets, fields, metadata, query_offsets = [], [0], [], [], [0]
    is_walk = protocol == "walk"
    count = (source.n_walks() if is_walk else len(source) if isinstance(source, ContextStore)
             else len(source.queries["label"]))
    for index in range(count):
        variants = (max(1, source.path_count(index))
                    if protocol == "path" and isinstance(source, ParametricWorld) else 1)
        for variant in range(variants):
            if is_walk:
                row = source.render_walk(index, tok)
            elif isinstance(source, ContextStore):
                row = source.render(index, protocol, tok, filler_tokens=filler_tokens)
            else:
                row = source.render_query(index, protocol, tok, filler_tokens=filler_tokens, path_index=variant)
            tokens.extend(row.tokens)
            offsets.append(len(tokens))
            fields.append((row.prompt_len, row.answer_index, row.label, row.filler_tokens))
            # DFS diagnostics reconstruct reference state only for evaluation;
            # training needs tokens/masks, not Python search-state objects.
            meta = {k: v for k, v in row.meta.items() if k not in ("reference_steps", "node_token", "full_body")}
            metadata.append(json.dumps(meta, separators=(",", ":")))
        query_offsets.append(len(fields))
    atomic_savez(path, tokens=np.asarray(tokens, dtype=np.int32), offsets=np.asarray(offsets, dtype=np.int64),
                 fields=np.asarray(fields, dtype=np.int32).reshape(-1, 4), metadata=np.asarray(metadata, dtype=str),
                 query_offsets=np.asarray(query_offsets, dtype=np.int64))
    atomic_json(path.with_suffix(".json"), identity)
    mark_complete(path, file_sha256(path))
    return path


class PreparedRows:
    """Read-only token arrays; no disk access in row lookup."""
    def __init__(self, source, protocol, tok, filler_tokens=0):
        identity = _identity(source, protocol, tok, filler_tokens)
        path = _cache_path(source, identity)
        if (not is_complete(path) or not path.with_suffix(".json").exists()
                or load_json(path.with_suffix(".json")) != identity):
            raise FileNotFoundError(f"missing/stale prepared tokens at {path}; run cot-nav data prepare first")
        self.protocol = protocol
        self.arrays = load_npz(path)

    def variants(self, index):
        a, b = self.arrays["query_offsets"][index:index + 2]
        return int(b - a)

    def row(self, index, variant=0):
        if not 0 <= variant < self.variants(index):
            raise IndexError("prepared row variant outside query")
        row_index = int(self.arrays["query_offsets"][index]) + variant
        a, b = self.arrays["offsets"][row_index:row_index + 2]
        tokens = self.arrays["tokens"][int(a):int(b)].tolist()
        prompt_len, answer_index, label, filler = map(int, self.arrays["fields"][row_index])
        trace = (list(range(prompt_len, answer_index)) + [answer_index + 1]
                 if self.protocol in ("path", "dfs_state", "bfs_state", "dfs_nodes", "bfs_nodes") else [])
        walk = list(range(2, len(tokens) - 1)) if self.protocol == "walk" else []
        return RenderedExample(tokens=tokens, prompt_len=prompt_len, answer_index=answer_index,
                               trace_indices=trace, walk_indices=walk, label=label, filler_tokens=filler,
                               meta=json.loads(str(self.arrays["metadata"][row_index])))


def _inventory(root):
    """Cheap reuse check: raw data, completion markers and token caches.

    Full content/structure checks happen when preparing. File replacement or
    modification invalidates this receipt and triggers those checks again.
    ctime_ns catches rewrites even if an external tool restores mtime.
    """
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "prepared_data.json":
            stat = path.stat()
            result[path.relative_to(root).as_posix()] = [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
    return result


@contextmanager
def _preparation_lock(root):
    import fcntl  # Server-side preparation; lock lives outside immutable root.
    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent / ("." + root.name + ".generation.lock")).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def prepare_dataset(cfg, protocols=DEFAULT_PROTOCOLS, *, force_validate=False):
    """Generate, validate and tokenize once; reuse the immutable result later."""
    protocols = tuple(dict.fromkeys(protocols))
    if not protocols or any(p not in PROTOCOLS for p in protocols):
        raise ValueError("prepare requires supported task protocols")
    root = Path(cfg.data.root)
    with _preparation_lock(root):
        generation = prepare_generation(cfg.data)
        tok = Tokenizer.from_config(cfg)
        signature = {"generation": generation["fingerprint"], "renderer_version": RENDERER_VERSION,
                     "vocab": tok.hash(), "filler_tokens": filler_length(cfg)}
        receipt = root / "prepared_data.json"
        if receipt.exists() and not force_validate:
            old = load_json(receipt)
            if (old.get("signature") == signature and set(protocols) <= set(old.get("protocols", []))
                    and old.get("audit_fraction", -1) >= cfg.data.audit_fraction
                    and old.get("inventory") == _inventory(root)):
                return {**old, "reused": True}
        if cfg.data.knowledge == "context_graph":
            generate_context(cfg.data, resume=True)
            report = validate_context(cfg.data)
            sources = [ContextStore(root)]
        else:
            manifest = generate_parametric(cfg.data, resume=True)
            report = validate_parametric(cfg.data)
            sources = [ParametricWorld(root / w["dir"]) for w in manifest["worlds"]]
        caches = []
        for source in sources:
            for protocol in protocols:
                caches.append(str(_write_rows(source, protocol, tok, filler_length(cfg)).relative_to(root)))
            if isinstance(source, ParametricWorld):
                caches.append(str(_write_rows(source, "walk", tok, 0).relative_to(root)))
        result = {"signature": signature, "protocols": list(protocols), "audit_fraction": cfg.data.audit_fraction,
                  "validation": report,
                  "caches": caches, "inventory": _inventory(root)}
        atomic_json(receipt, result)
        return {**result, "reused": False}

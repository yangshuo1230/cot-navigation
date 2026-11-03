"""Load generated records and render protocol examples.

Graph shards stay independent of the output protocol. Rendering preserves
complete trajectories; only Filler has a fixed number of extra positions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cot_navigation.data.shards import is_complete, load_json, load_npz
from cot_navigation.protocols.render import RenderedExample, context_prompt, parametric_prompt, render_task, render_walk
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.data.provenance import dataset_fingerprint, world_fingerprint
from cot_navigation.data.paths import PATH_TABLE_VERSION


def _adj_from_edges(n_nodes: int, src: np.ndarray, dst: np.ndarray) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(n_nodes)]
    for u, v in zip(src.tolist(), dst.tolist()):
        adj[int(u)].append(int(v))
    return adj


class ContextStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not is_complete(self.root / "manifest.json"):
            raise FileNotFoundError(f"incomplete context data at {self.root}")
        self.manifest = load_json(self.root / "manifest.json")
        self.fingerprint = dataset_fingerprint(self.root, self.manifest)
        self.shape_names = list(self.manifest["shape_names"])
        parts = []
        for split in ("train", "val", "test"):
            split_dir = self.root / split
            shards = [split_dir / s["file"] for s in self.manifest["splits"][split]["shards"]]
            if not shards:
                raise FileNotFoundError(f"no shards in {split_dir}")
            blocks = []
            for shard in shards:
                if not is_complete(shard):
                    raise FileNotFoundError(f"incomplete shard {shard}")
                blocks.append(load_npz(shard))
            merged = {key: np.concatenate([b[key] for b in blocks], axis=0) for key in blocks[0]}
            merged["split_id"] = np.full(merged["sample_id"].shape[0], {"train": 0, "val": 1, "test": 2}[split], dtype=np.int8)
            parts.append(merged)
        keys = parts[0].keys()
        self.arrays = {key: np.concatenate([p[key] for p in parts], axis=0) for key in keys}
        self.split_slices = {}
        cursor = 0
        for split, part in zip(("train", "val", "test"), parts):
            n = part["sample_id"].shape[0]
            self.split_slices[split] = (cursor, cursor + n)
            cursor += n

    def __len__(self) -> int:
        return int(self.arrays["sample_id"].shape[0])

    def indices(self, split: str) -> np.ndarray:
        start, end = self.split_slices[split]
        return np.arange(start, end, dtype=np.int64)

    def adj(self, index: int) -> list[list[int]]:
        n = int(self.arrays["node_token"].shape[1])
        return _adj_from_edges(n, self.arrays["edge_src"][index], self.arrays["edge_dst"][index])

    def node_vocab_ids(self, index: int, tok: Tokenizer) -> list[int]:
        return [tok.node_id(int(raw)) for raw in self.arrays["node_token"][index].tolist()]

    def render(self, index: int, protocol: str, tok: Tokenizer, *, filler_tokens: int = 0) -> RenderedExample:
        node_token = self.node_vocab_ids(index, tok)
        edges = list(
            zip(
                self.arrays["edge_src"][index].astype(int).tolist(),
                self.arrays["edge_dst"][index].astype(int).tolist(),
            )
        )
        src = int(self.arrays["s_topo"][index])
        dst = int(self.arrays["t_topo"][index])
        prompt = context_prompt(
            tok,
            node_token,
            edges,
            src,
            dst,
        )
        rendered = render_task(tok, protocol, int(self.arrays["label"][index]), prompt, self.adj(index), src, dst,
                               node_token, filler_tokens=filler_tokens)
        rendered.meta.update(
            {
                "index": index,
                "shape": self.shape_names[int(self.arrays["shape_id"][index])],
                "distance": int(self.arrays["distance"][index]),
                "sample_id": int(self.arrays["sample_id"][index]),
                "split": int(self.arrays["split_id"][index]),
            }
        )
        return rendered


class ParametricWorld:
    def __init__(self, world_dir: str | Path, *, include_queries: bool = True):
        self.root = Path(world_dir)
        if not is_complete(self.root / "world.json"):
            raise ValueError(f"incomplete parametric world at {self.root}")
        self.info = load_json(self.root / "world.json")
        if self.info.get("path_table_version") != PATH_TABLE_VERSION:
            raise ValueError(f"legacy world at {self.root} lacks the complete path table; regenerate in a new data.root")
        self.fingerprint = world_fingerprint(self.root)
        graph = load_npz(self.root / "graph.npz")
        walks = load_npz(self.root / "walks.npz")
        self.node_token_raw = graph["node_token"].astype(int)
        self.nbr_ptr = graph["nbr_ptr"].astype(int)
        self.nbr_flat = graph["nbr_flat"].astype(int)
        self.walk_offset = walks["offset"].astype(np.int64)
        self.walk_nodes = walks["nodes"].astype(int)
        self.walk_supplement = walks["is_supplement"].astype(int)
        # Stage A cannot sample/evaluate queries: no query arrays are loaded.
        self.queries = load_npz(self.root / "queries.npz") if include_queries else None
        self.paths = load_npz(self.root / "paths.npz") if include_queries else None
        self.balanced = {
            split: np.load(self.root / f"balanced_{split}.npy")
            for split in ("train", "val", "test")
        } if include_queries else {}

    @property
    def n_nodes(self) -> int:
        return int(self.node_token_raw.shape[0])

    def adj(self) -> list[list[int]]:
        adj: list[list[int]] = []
        for i in range(self.n_nodes):
            adj.append([int(v) for v in self.nbr_flat[self.nbr_ptr[i] : self.nbr_ptr[i + 1]].tolist()])
        return adj

    def node_vocab_ids(self, tok: Tokenizer) -> list[int]:
        return [tok.node_id(int(raw)) for raw in self.node_token_raw.tolist()]

    def n_walks(self) -> int:
        return int(self.walk_supplement.shape[0])

    def walk_nodes_at(self, index: int) -> list[int]:
        a = int(self.walk_offset[index])
        b = int(self.walk_offset[index + 1])
        return [int(v) for v in self.walk_nodes[a:b].tolist()]

    def path_count(self, query_index: int) -> int:
        if self.paths is None:
            raise ValueError("knowledge-only world does not expose query paths")
        a, b = self.paths["query_offsets"][query_index:query_index + 2]
        return int(b - a)

    def indices(self, split: str, view: str = "full") -> np.ndarray:
        """Unique query ids; one-hop knowledge tasks belong only to train."""
        if self.queries is None:
            raise ValueError("knowledge-only world does not expose queries")
        if view == "balanced":
            return self.balanced[split]
        mask = self.queries["split"] == {"train": 0, "val": 1, "test": 2}[split]
        if view == "one_hop":
            mask &= self.queries["distance"] == 1
        elif view == "multi_hop":
            mask &= self.queries["distance"] >= 2
        elif view != "full":
            raise ValueError(f"unknown query view: {view}")
        return np.flatnonzero(mask).astype(np.int64)

    def path_nodes_at(self, query_index: int, path_index: int) -> list[int]:
        if not 0 <= path_index < self.path_count(query_index):
            raise IndexError("path index outside this query's prepared paths")
        global_index = int(self.paths["query_offsets"][query_index]) + path_index
        a, b = self.paths["path_offsets"][global_index:global_index + 2]
        return self.paths["nodes"][int(a):int(b)].astype(int).tolist()

    def render_query(self, index: int, protocol: str, tok: Tokenizer, *, filler_tokens: int = 0,
                     path_index: int = 0) -> RenderedExample:
        if self.queries is None:
            raise ValueError("knowledge-only world does not expose queries")
        node_token = self.node_vocab_ids(tok)
        src = int(self.queries["s_topo"][index])
        dst = int(self.queries["t_topo"][index])
        label = int(self.queries["label"][index])
        path_nodes = self.path_nodes_at(index, path_index) if protocol == "path" and label == 1 else None
        prompt = parametric_prompt(tok, node_token, src, dst)
        rendered = render_task(tok, protocol, label, prompt, self.adj(), src, dst, node_token,
                               filler_tokens=filler_tokens, path_nodes=path_nodes)
        rendered.meta.update(
            {
                "index": int(index),
                "shape": self.info["shape"],
                "distance": int(self.queries["distance"][index]),
                "sample_id": int(index),
                "split": int(self.queries["split"][index]),
                "witness_seen": int(self.queries["witness_seen"][index]),
                "graph_seed": int(self.info["graph_seed"]),
                "path_index": path_index if protocol == "path" and label == 1 else None,
                "path_count": self.path_count(index),
            }
        )
        return rendered

    def render_walk(self, index: int, tok: Tokenizer) -> RenderedExample:
        rendered = render_walk(tok, self.node_vocab_ids(tok), self.walk_nodes_at(index))
        rendered.meta.update({"index": int(index), "kind": "walk", "supplement": int(self.walk_supplement[index])})
        return rendered


def collate_examples(rows: list[RenderedExample], pad_id: int, multiple: int = 1) -> dict[str, np.ndarray]:
    """Right-pad to a bucket multiple. Loss masks stay zero on pad positions."""
    width = max(len(row.tokens) for row in rows)
    if multiple > 1:
        width = ((width + multiple - 1) // multiple) * multiple
    batch = len(rows)
    tokens = np.full((batch, width), pad_id, dtype=np.int64)
    answer_index = np.full(batch, -1, dtype=np.int64)
    answer_mask = np.zeros((batch, width), dtype=np.float32)
    trace_mask = np.zeros((batch, width), dtype=np.float32)
    walk_mask = np.zeros((batch, width), dtype=np.float32)
    labels = np.full(batch, -1, dtype=np.int64)
    lengths = np.zeros(batch, dtype=np.int64)
    is_walk = np.zeros(batch, dtype=np.int64)
    for i, row in enumerate(rows):
        length = len(row.tokens)
        tokens[i, :length] = np.asarray(row.tokens, dtype=np.int64)
        lengths[i] = length
        labels[i] = row.label
        is_walk[i] = int(row.label < 0)
        if row.answer_index >= 0:
            answer_index[i] = row.answer_index
            answer_mask[i, row.answer_index] = 1.0
        if row.trace_indices:
            trace_mask[i, row.trace_indices] = 1.0
        if row.walk_indices:
            walk_mask[i, row.walk_indices] = 1.0
    return {
        "tokens": tokens,
        "answer_index": answer_index,
        "answer_mask": answer_mask,
        "trace_mask": trace_mask,
        "walk_mask": walk_mask,
        "labels": labels,
        "lengths": lengths,
        "is_walk": is_walk,
    }

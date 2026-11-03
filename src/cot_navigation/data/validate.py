"""Structural checks. The fast bitset is checked on every row; BFS is a sample.

BFS does not import the bitset recurrence. ``audit_fraction`` selects rows with
a stable stride so the audited subset does not depend on worker scheduling.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cot_navigation.config import DataConfig
from cot_navigation.data.dataset import ContextStore, ParametricWorld
from cot_navigation.data.graph import structure_hash
from cot_navigation.data.reachability import bfs_reaches, build_reach, reaches
from cot_navigation.data.shards import file_sha256, is_complete, load_json
from cot_navigation.protocols.dfs import dfs_search, dfs_summary


class ValidationError(RuntimeError):
    pass


def _fail(message: str) -> None:
    raise ValidationError(message)


def validate_context(cfg: DataConfig) -> dict:
    root = Path(cfg.root)
    if not is_complete(root / "manifest.json"):
        _fail("manifest incomplete")
    manifest = load_json(root / "manifest.json")
    if manifest["kind"] != "context_graph":
        _fail("not context_graph data")
    store = ContextStore(root)
    shape_names = store.shape_names
    if shape_names != list(cfg.shapes):
        _fail(f"shape order {shape_names} != config {list(cfg.shapes)}")
    report = {"rows": len(store), "bfs_checked": 0, "bfs_mismatch": 0, "splits": {}}
    hashes_by_split: dict[str, set[int]] = {}
    for split, (start, end) in store.split_slices.items():
        expected = {"train": cfg.splits.train, "val": cfg.splits.val, "test": cfg.splits.test}[split]
        if end - start != expected:
            _fail(f"{split} count {end - start} != {expected}")
        ids = store.arrays["sample_id"][start:end]
        if ids.tolist() != list(range(expected)):
            _fail(f"{split} sample_id is not 0..{expected - 1} in order")
        hashes: set[int] = set()
        for local, index in enumerate(range(start, end)):
            _check_context_row(cfg, store, index, shape_names)
            digest = int(store.arrays["struct_hash"][index])
            if digest in hashes:
                _fail(f"duplicate structure in {split}")
            hashes.add(digest)
            if _audit_this(local, cfg.audit_fraction):
                adj = store.adj(index)
                src = int(store.arrays["s_topo"][index])
                dst = int(store.arrays["t_topo"][index])
                label = int(store.arrays["label"][index])
                report["bfs_checked"] += 1
                if bfs_reaches(adj, src, dst) != (label == 1):
                    report["bfs_mismatch"] += 1
                    _fail(f"BFS mismatch at {split} sample {int(ids[local])}")
        hashes_by_split[split] = hashes
        report["splits"][split] = {"count": end - start, "unique_hashes": len(hashes)}
        for shard_meta in manifest["splits"][split]["shards"]:
            path = root / split / shard_meta["file"]
            digest = file_sha256(path)
            if digest != shard_meta["checksum"]:
                _fail(f"checksum mismatch {path}")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = hashes_by_split[left] & hashes_by_split[right]
        if overlap:
            _fail(f"structure hash crosses {left}/{right}")
    if report["bfs_mismatch"]:
        _fail("bfs mismatches")
    report["search"] = {split: _search_report({
        key: store.arrays[key][start:end] for key in
        ("dfs_trace_tokens", "dfs_pops", "dfs_max_stack", "dfs_visited_nodes", "dfs_branch_records")
    }) for split, (start, end) in store.split_slices.items()}
    report["acceptance"] = {split: manifest["splits"][split]["acceptance"] for split in manifest["splits"]}
    return report


def _audit_this(local: int, fraction: float) -> bool:
    if fraction >= 1:
        return True
    if fraction <= 0:
        return False
    stride = max(1, int(round(1 / fraction)))
    return local % stride == 0


def _check_context_row(cfg: DataConfig, store: ContextStore, index: int, shape_names: list[str]) -> None:
    shape = shape_names[int(store.arrays["shape_id"][index])]
    spec = cfg.shapes[shape]
    src_e = store.arrays["edge_src"][index].astype(int)
    dst_e = store.arrays["edge_dst"][index].astype(int)
    if src_e.shape[0] != cfg.n_edges:
        _fail("edge count")
    edges = list(zip(src_e.tolist(), dst_e.tolist()))
    if len(set(edges)) != len(edges):
        _fail("duplicate edge")
    for u, v in edges:
        if ((u // spec.width + 1 != v // spec.width) if cfg.graph_family == "layered" else not u < v):
            _fail(f"non-adjacent edge {(u, v)} in {shape}")
        if not (0 <= u < cfg.n_nodes and 0 <= v < cfg.n_nodes):
            _fail("edge endpoint out of range")
    raw = store.arrays["node_token"][index].astype(int)
    if len(set(raw.tolist())) != cfg.n_nodes or raw.min() < 0 or raw.max() >= cfg.node_vocab:
        _fail("node tokens are not an injection into the node vocab")
    order = store.arrays["node_order"][index].astype(int)
    if sorted(order.tolist()) != list(range(cfg.n_nodes)):
        _fail("NODES order is not a permutation")
    digest = structure_hash(spec.layers, spec.width, edges)
    if digest != int(store.arrays["struct_hash"][index]):
        _fail("stored structure hash does not match the edge set")
    adj = store.adj(index)
    reach = build_reach(adj)
    src = int(store.arrays["s_topo"][index])
    dst = int(store.arrays["t_topo"][index])
    label = int(store.arrays["label"][index])
    if src == dst:
        _fail("s=t query")
    if (dst // spec.width - src // spec.width) != int(store.arrays["distance"][index]):
        _fail("stored distance does not match layers")
    if reaches(reach, src, dst) != (label == 1):
        _fail("bitset label mismatch")
    steps = dfs_search(adj, src, dst)
    expect = "FOUND" if label == 1 else "EXHAUST"
    if steps[-1].action != expect:
        _fail("DFS terminal action mismatch")
    summary = dfs_summary(steps)
    for stored, key in (("dfs_trace_tokens", "trace_tokens"), ("dfs_pops", "pops"),
                        ("dfs_max_stack", "max_stack"), ("dfs_visited_nodes", "visited_nodes"),
                        ("dfs_branch_records", "branch_records")):
        if int(store.arrays[stored][index]) != summary[key]:
            _fail(f"stored search statistic mismatch: {stored}")
    # QUERY identifies valid endpoints even when isolated nodes have no EDGE.
    del reach


def validate_parametric(cfg: DataConfig) -> dict:
    root = Path(cfg.root)
    manifest = load_json(root / "manifest.json")
    if manifest["kind"] != "parametric_graph":
        _fail("not parametric data")
    report = {"worlds": []}
    for world_meta in manifest["worlds"]:
        world = ParametricWorld(root / world_meta["dir"])
        spec = cfg.shapes[world.info["shape"]]
        adj = world.adj()
        edges = [(u, v) for u, nbrs in enumerate(adj) for v in nbrs]
        if len(edges) != cfg.n_edges or len(set(edges)) != len(edges):
            _fail(f"edge count in {world_meta['dir']}")
        for u, v in edges:
            if ((u // spec.width + 1 != v // spec.width) if cfg.graph_family == "layered" else not u < v):
                _fail("non-adjacent fixed-graph edge")
        if world.info["min_edge_count"] < cfg.min_edge_coverage:
            _fail("edge coverage below minimum")
        reach = build_reach(adj)
        splits = {0: set(), 1: set(), 2: set()}
        n = len(world.queries["s_topo"])
        path_stats = _validate_paths(world, adj, cfg.graph_family == "layered")
        bfs_checked = 0
        summaries = []
        for i in range(n):
            src = int(world.queries["s_topo"][i])
            dst = int(world.queries["t_topo"][i])
            label = int(world.queries["label"][i])
            distance = int(world.queries["distance"][i])
            if (src == dst or distance < 1
                    or distance != dst // spec.width - src // spec.width):
                _fail("fixed query has invalid endpoints/distance")
            key = (src, dst)
            split = int(world.queries["split"][i])
            if split not in splits:
                _fail("invalid query split")
            if distance == 1 and (not cfg.include_one_hop_queries or split != 0):
                _fail("one-hop knowledge queries must be enabled and train-only")
            if key in splits[split]:
                _fail("duplicate query inside a split")
            splits[split].add(key)
            if reaches(reach, src, dst) != (label == 1):
                _fail("fixed query label mismatch")
            if _audit_this(i, cfg.audit_fraction):
                bfs_checked += 1
                if bfs_reaches(adj, src, dst) != (label == 1):
                    _fail("fixed query BFS mismatch")
            summary = dfs_summary(dfs_search(adj, src, dst))
            summaries.append({"dfs_trace_tokens": summary["trace_tokens"], "dfs_pops": summary["pops"],
                              "dfs_max_stack": summary["max_stack"], "dfs_visited_nodes": summary["visited_nodes"],
                              "dfs_branch_records": summary["branch_records"]})
            if cfg.graph_family == "layered" and distance >= 5 and int(world.queries["witness_seen"][i]) == 1:
                _fail("d>=5 witness was marked seen")
        for a, b in ((0, 1), (0, 2), (1, 2)):
            if splits[a] & splits[b]:
                _fail("query crosses splits")
        one_hop = world.indices("train", "one_hop")
        expected_one_hop = (spec.layers - 1) * spec.width**2 if cfg.include_one_hop_queries else 0
        if len(one_hop) != expected_one_hop:
            _fail("incomplete one-hop query coverage")
        # Also audit stored balanced ids: stale selectors must not expose test
        # queries or silently discard/relabel the new raw training pool.
        for split_id, name in enumerate(("train", "val", "test")):
            ids = world.balanced[name]
            if (len(set(ids.tolist())) != len(ids) or np.any(ids < 0) or np.any(ids >= n)
                    or np.any(world.queries["split"][ids] != split_id)
                    or np.any(world.queries["distance"][ids] < 2)):
                _fail("invalid balanced query selector")
            for distance in np.unique(world.queries["distance"][ids]):
                labels = world.queries["label"][ids[world.queries["distance"][ids] == distance]]
                if int((labels == 0).sum()) != int((labels == 1).sum()):
                    _fail("balanced selector has unmatched labels")
        # Walk edges are legal. Coverage counted directly.
        from collections import Counter

        counts: Counter[tuple[int, int]] = Counter()
        for w in range(world.n_walks()):
            nodes = world.walk_nodes_at(w)
            if len(nodes) < 2:
                _fail("empty walk")
            for u, v in zip(nodes, nodes[1:]):
                if v not in adj[u]:
                    _fail("walk uses a non-edge")
                counts[(u, v)] += 1
        for u, v in edges:
            if counts[(u, v)] < cfg.min_edge_coverage:
                _fail(f"edge {(u, v)} covered {counts[(u, v)]} times")
        report["worlds"].append(
            {
                "dir": world_meta["dir"],
                "queries": n,
                "one_hop_train": len(one_hop),
                "prepared_paths": path_stats,
                "split_queries": {name: int(np.sum(world.queries["split"] == split))
                                  for split, name in enumerate(("train", "val", "test"))},
                "balanced_queries": {name: len(ids) for name, ids in world.balanced.items()},
                "search": _search_report({key: np.asarray([s[key] for s in summaries]) for key in summaries[0]}) if summaries else {},
                "walks": world.n_walks(),
                "bfs_checked": bfs_checked,
                "witness_positive": world.info["witness_positive"],
                "positive_d_ge_5": world.info["positive_d_ge_5"],
            }
        )
    return report


def _validate_paths(world: ParametricWorld, adj: list[list[int]], layered=True) -> dict:
    """Prove completeness using valid unique paths and independent forward counts."""
    table = world.paths
    qoff, poff, nodes = (table[k] for k in ("query_offsets", "path_offsets", "nodes"))
    n_queries = len(world.queries["label"])
    for array in (qoff, poff, nodes):
        if array.ndim != 1 or array.dtype.kind not in "iu":
            _fail("path arrays must be one-dimensional integers")
    if (len(qoff) != n_queries + 1 or not len(poff) or qoff[0] != 0 or poff[0] != 0
            or qoff[-1] != len(poff) - 1 or poff[-1] != len(nodes)
            or np.any(qoff[1:] < qoff[:-1]) or np.any(poff[1:] <= poff[:-1])):
        _fail("invalid prepared path offsets")
    for i in range(n_queries):
        src, dst = (int(world.queries[k][i]) for k in ("s_topo", "t_topo"))
        # This recurrence runs forward from the source, independently of the
        # generator's reverse, target-based recurrence and DFS enumerator.
        counts = [0] * len(adj)
        counts[src] = 1
        for u in range(len(adj)):
            for v in adj[u]:
                counts[v] += counts[u]
        count = world.path_count(i)
        if count != counts[dst] or bool(count) != bool(world.queries["label"][i]):
            _fail(f"incomplete prepared paths for query {i}")
        seen = set()
        for j in range(count):
            path = tuple(world.path_nodes_at(i, j))
            if (not path or path[0] != src or path[-1] != dst
                    or (layered and len(path) != int(world.queries["distance"][i]) + 1)
                    or len(set(path)) != len(path)
                    or any(u < 0 or u >= len(adj) for u in path)
                    or any(v not in adj[u] for u, v in zip(path, path[1:]))):
                _fail(f"illegal prepared path for query {i}")
            if path in seen:
                _fail(f"duplicate prepared path for query {i}")
            seen.add(path)
    stats = {"total": len(poff) - 1, "node_entries": len(nodes),
             "max_per_query": int(np.diff(qoff).max(initial=0))}
    if stats != world.info.get("prepared_paths"):
        _fail("prepared path statistics disagree with world metadata")
    return stats


def _search_report(arrays: dict) -> dict:
    """Generation diagnostics independent of model outputs and answer scoring."""
    return {key: {"mean": float(np.mean(values)), "p95": float(np.quantile(values, .95)),
                  "max": int(np.max(values)), "nonzero_fraction": float(np.mean(values > 0))}
            for key, values in arrays.items() if len(values)}

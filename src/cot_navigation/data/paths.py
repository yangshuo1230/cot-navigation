"""Prepare every directed path for fixed-graph queries, without silent caps.

Internal node ids are topological (every edge u->v has u<v). Count paths and
storage first by DAG dynamic programming, then enumerate with reachability
pruning directly into flat arrays. DAG paths are automatically simple.
"""
from __future__ import annotations

import numpy as np

PATH_TABLE_VERSION = 1


def _suffix_counts(adj: list[list[int]], target: int):
    counts = [0] * len(adj)
    node_counts = [0] * len(adj)
    counts[target] = node_counts[target] = 1
    for u in range(target - 1, -1, -1):
        counts[u] = sum(counts[v] for v in adj[u])
        node_counts[u] = sum(node_counts[v] + counts[v] for v in adj[u])
    return counts, node_counts


def _paths(adj, source, target, counts):
    if not counts[source]:
        return
    if source == target:
        yield [source]
        return
    path = [source]
    stack = [iter(adj[source])]
    while stack:
        nxt = next(stack[-1], None)
        if nxt is None:
            stack.pop()
            path.pop()
            continue
        if not counts[nxt]:
            continue
        path.append(nxt)
        if nxt == target:
            yield path
            path.pop()
        else:
            stack.append(iter(adj[nxt]))


def prepare_paths(adj: list[list[int]], pairs: list[tuple[int, int]], *,
                  max_paths: int = 1_000_000, max_nodes: int = 20_000_000) -> dict[str, np.ndarray]:
    """Return query->path offsets, path->node offsets and flat node ids.

    Each query has its own segment, including held-out queries; only training
    query ids may be sampled for supervision. Negative queries have no paths.
    Limits fail before allocation/enumeration, rather than truncate the corpus.
    """
    n = len(adj)
    for u, nbrs in enumerate(adj):
        if len(nbrs) != len(set(nbrs)) or any(v <= u or v >= n for v in nbrs):
            raise ValueError("path preparation requires unique forward edges in topological node ids")
    if any(not (0 <= s < n and 0 <= t < n) for s, t in pairs):
        raise ValueError("path endpoint outside graph")
    cache = {t: _suffix_counts(adj, t) for t in {target for _, target in pairs}}
    query_offsets = [0]
    total_nodes = 0
    for s, t in pairs:
        count, node_count = cache[t]
        query_offsets.append(query_offsets[-1] + count[s])
        total_nodes += node_count[s]
    total_paths = query_offsets[-1]
    if total_paths > max_paths or total_nodes > max_nodes:
        raise ValueError(f"all-path preparation requires {total_paths} paths/{total_nodes} node entries; "
                         f"limits are {max_paths}/{max_nodes}. Increase limits or reduce graph complexity; "
                         "no paths were silently omitted")
    offsets = np.zeros(total_paths + 1, dtype=np.int64)
    nodes = np.empty(total_nodes, dtype=np.int32)
    path_index = cursor = 0
    for q, (s, t) in enumerate(pairs):
        for path in _paths(adj, s, t, cache[t][0]):
            length = len(path)
            nodes[cursor:cursor + length] = path
            cursor += length
            path_index += 1
            offsets[path_index] = cursor
        if path_index != query_offsets[q + 1]:
            raise RuntimeError("enumerated path count disagrees with DAG count")
    if cursor != total_nodes:
        raise RuntimeError("enumerated path storage disagrees with DAG count")
    return {"query_offsets": np.asarray(query_offsets, dtype=np.int64),
            "path_offsets": offsets, "nodes": nodes}


def query_paths(table, query_index):
    """Iterate prepared paths, in fixed DFS neighbor order, without copying."""
    a, b = table["query_offsets"][query_index:query_index + 2]
    for i in range(int(a), int(b)):
        start, end = table["path_offsets"][i:i + 2]
        yield table["nodes"][int(start):int(end)]

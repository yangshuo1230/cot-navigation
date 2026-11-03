"""Non-layered DAGs with local/long edges and planted reconvergence.

Internal IDs are a hidden topological ordering. ``layers`` and ``width``
define graph size and rank bands for query stratification, NOT graph layers.
Node tokens are independently renamed by the existing generator.
"""
from __future__ import annotations

import numpy as np

VERSION = 1


def sample_hard_edges(layers, width, n_edges, rng):
    """Exactly M distinct forward edges; random disjoint diamonds survive fill.

    Diamond vertices are sampled across the ordering rather than at fixed
    positions. Remaining edges use a 85% local / 15% global probability
    mixture, sampled without replacement. This is not uniform over DAGs.
    The local scale is width/2; longer topological paths can cross rank bands
    through several edges or skip bands in one edge.
    """
    n = layers * width
    if not 0 <= n_edges <= n * (n - 1) // 2:
        raise ValueError("invalid forward DAG edge count")
    pool = np.column_stack(np.triu_indices(n, 1))
    # About one diamond per eight nodes; never exceed the edge budget.
    n_diamonds = min(n // 8, n_edges // 4)
    groups = rng.permutation(n)[:4 * n_diamonds].reshape(-1, 4)
    chosen = set()
    for group in groups:
        a, b, c, d = sorted(map(int, group))
        chosen.update(((a, b), (a, c), (b, d), (c, d)))
    remain = n_edges - len(chosen)
    if remain:
        mask = np.asarray([tuple(map(int, edge)) not in chosen for edge in pool])
        available = pool[mask]
        span = available[:, 1] - available[:, 0]
        local = np.exp(-(span - 1) / max(1.0, width / 2))
        weights = .85 * local / local.sum() + .15 / len(local)
        picks = rng.choice(len(available), remain, replace=False, p=weights)
        chosen.update(tuple(map(int, available[i])) for i in picks)
    # Never depend on hash iteration order for seeded reproducibility.
    return sorted(chosen)


def shortest_distances(adj, source):
    """Topological DP, independent of path enumeration."""
    dist = [-1] * len(adj)
    dist[source] = 0
    for u in range(source, len(adj)):
        if dist[u] < 0:
            continue
        for v in adj[u]:
            value = dist[u] + 1
            if dist[v] < 0 or value < dist[v]:
                dist[v] = value
    return dist


def graph_statistics(adj, width):
    indeg = [0] * len(adj)
    depth = [0] * len(adj)
    categories = dict(within_band=0, adjacent_band=0, skip_band=0)
    merge_edges = 0
    for u, nbrs in enumerate(adj):
        for v in nbrs:
            indeg[v] += 1
            depth[v] = max(depth[v], depth[u] + 1)
            gap = v // width - u // width
            categories['within_band' if gap == 0 else 'adjacent_band' if gap == 1 else 'skip_band'] += 1
    merge_edges = sum(max(0, x - 1) for x in indeg)
    return dict(n_nodes=len(adj), n_edges=sum(map(len, adj)),
                longest_path=max(depth), branching_nodes=sum(len(x) >= 2 for x in adj),
                merging_nodes=sum(x >= 2 for x in indeg), merge_excess_edges=merge_edges,
                **categories)

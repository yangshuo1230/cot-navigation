"""Random-walk corpus for a fixed graph.

Walks are not conditioned on task queries. A sink ends the walk early; that
boundary is not supervised as "this node has no out-edge". After the random
walks, each directed edge is topped up to ``min_coverage`` one-hop records.
"""

from __future__ import annotations

from collections import Counter

from numpy.random import Generator


class WalkError(RuntimeError):
    pass


def generate_walks(
    adj: list[list[int]],
    n_walks: int,
    rng: Generator,
    min_coverage: int,
    lengths: tuple[int, ...] = (1, 2, 3, 4),
    max_start_tries: int = 10000,
) -> tuple[list[list[int]], dict[tuple[int, int], int], int]:
    """Return walks, edge counts, and how many one-hop supplements were added.

    Each walk is a node list with at least one edge. Supplements are appended
    after the random walks and are included in the counts.
    """
    n = len(adj)
    sources = [u for u, nbrs in enumerate(adj) if nbrs]
    if not sources:
        raise WalkError("graph has no edges to walk")
    walks: list[list[int]] = []
    counts: Counter[tuple[int, int]] = Counter()
    for _ in range(n_walks):
        start = None
        for _try in range(max_start_tries):
            cand = int(rng.integers(0, n))
            if adj[cand]:
                start = cand
                break
        if start is None:
            raise WalkError("failed to sample a non-sink start")
        target_len = int(lengths[int(rng.integers(0, len(lengths)))])
        nodes = [start]
        cursor = start
        for _step in range(target_len):
            nbrs = adj[cursor]
            if not nbrs:
                break
            nxt = nbrs[int(rng.integers(0, len(nbrs)))]
            nodes.append(nxt)
            counts[(cursor, nxt)] += 1
            cursor = nxt
        if len(nodes) < 2:
            raise WalkError("produced an empty walk after a non-sink start")
        walks.append(nodes)
    supplements = 0
    for src, nbrs in enumerate(adj):
        for dst in nbrs:
            need = min_coverage - counts[(src, dst)]
            if need <= 0:
                continue
            for _ in range(need):
                walks.append([src, dst])
                counts[(src, dst)] += 1
                supplements += 1
    return walks, dict(counts), supplements


def subpath_set(walks: list[list[int]], min_edges: int = 2) -> set[tuple[int, ...]]:
    """Contiguous subwalks with at least ``min_edges`` edges.

    Used only to audit whether a reference witness showed up. It does not
    filter the training distribution.
    """
    found: set[tuple[int, ...]] = set()
    for walk in walks:
        n = len(walk)
        for i in range(n):
            for j in range(i + min_edges, n):
                found.add(tuple(walk[i : j + 1]))
    return found

"""Reachability oracles and uniform pair sampling.

The fast oracle is a reverse-topological bitset DP: ``reach[u]`` has bit ``v``
set iff a directed ``u -> ... -> v`` path exists. Self is excluded, matching
the ``s != t`` task. The independent checker is BFS and does not use this
recurrence. Neither routine enumerates paths.
"""

from __future__ import annotations

import numpy as np


def build_reach(adj: list[list[int]]) -> list[int]:
    """Forward reachability bitsets, excluding the source itself.

    Nodes are topological: every edge goes to a higher index, so scanning
    from the last node backward is a valid DP order.
    """
    n = len(adj)
    reach = [0] * n
    for u in range(n - 1, -1, -1):
        bits = 0
        for v in adj[u]:
            bits |= (1 << v) | reach[v]
        reach[u] = bits
    return reach


def reaches(reach: list[int], src: int, dst: int) -> bool:
    return ((reach[src] >> dst) & 1) == 1


def bfs_reaches(adj: list[list[int]], src: int, dst: int) -> bool:
    """Independent path-existence check. Does not share the bitset recurrence."""
    if src == dst:
        return False
    seen = {src}
    stack = [src]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v == dst:
                return True
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return False


def _kth_bit(bits: int, start: int, width: int, k: int, want_one: bool) -> int:
    for offset in range(width):
        occupied = ((bits >> (start + offset)) & 1) == 1
        if occupied == want_one:
            if k == 0:
                return start + offset
            k -= 1
    raise RuntimeError("pair rank exceeds the pool")


def pool_index(reach: list[int], width: int, n_layers: int, distance: int, positive: bool):
    """Prefix sums over per-source pool sizes. Zero-size sources are omitted.

    A rank in ``0 .. total-1`` maps to exactly one pair, so sampling a uniform
    integer rank is uniform over the pool without building an ``N^2`` list.
    """
    nodes: list[int] = []
    counts: list[int] = []
    for layer in range(n_layers - distance):
        start = (layer + distance) * width
        mask = ((1 << width) - 1) << start
        for src in range(layer * width, (layer + 1) * width):
            ones = (reach[src] & mask).bit_count()
            count = ones if positive else width - ones
            if count:
                nodes.append(src)
                counts.append(count)
    if not counts:
        prefix = np.zeros(0, dtype=np.int64)
        return nodes, counts, prefix, 0
    prefix = np.cumsum(np.asarray(counts, dtype=np.int64), dtype=np.int64)
    return nodes, counts, prefix, int(prefix[-1])


def pair_at_rank(
    reach: list[int],
    width: int,
    distance: int,
    positive: bool,
    nodes: list[int],
    prefix: np.ndarray,
    rank: int,
) -> tuple[int, int]:
    # ``side='right'`` lands on the bucket whose exclusive prefix end exceeds rank.
    index = int(np.searchsorted(prefix, rank, side="right"))
    prev = 0 if index == 0 else int(prefix[index - 1])
    local = rank - prev
    src = nodes[index]
    layer = src // width
    start = (layer + distance) * width
    dst = _kth_bit(reach[src], start, width, local, want_one=positive)
    return src, dst


def sample_pair(
    reach: list[int],
    width: int,
    n_layers: int,
    distance: int,
    positive: bool,
    rng,
) -> tuple[int, int]:
    nodes, _counts, prefix, total = pool_index(reach, width, n_layers, distance, positive)
    if total <= 0:
        raise RuntimeError(f"empty {'positive' if positive else 'negative'} pool at d={distance}")
    rank = int(rng.integers(0, total))
    return pair_at_rank(reach, width, distance, positive, nodes, prefix, rank)

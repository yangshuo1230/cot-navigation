"""Layered DAG construction.

Internal ids are topological indices ``0 .. N-1``. Layer ``L`` occupies the
contiguous range ``[L*width, (L+1)*width)``. Edges exist only from layer ``L``
to layer ``L+1``. Token names are assigned later and are not these indices.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from numpy.random import Generator


@dataclass
class LayeredGraph:
    n_layers: int
    width: int
    # Out-neighbors in the order DFS must use. Isolated nodes map to [].
    adj: list[list[int]]
    edges: list[tuple[int, int]] = field(default_factory=list)

    @property
    def n_nodes(self) -> int:
        return self.n_layers * self.width

    def layer_of(self, node: int) -> int:
        return node // self.width


def candidate_edges(n_layers: int, width: int) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for layer in range(n_layers - 1):
        src = range(layer * width, (layer + 1) * width)
        dst = range((layer + 1) * width, (layer + 2) * width)
        for u in src:
            for v in dst:
                edges.append((u, v))
    return edges


def sample_edges(n_layers: int, width: int, n_edges: int, rng: Generator, graph_family="layered") -> list[tuple[int, int]]:
    """Draw exactly ``n_edges`` distinct adjacent-layer edges, uniformly.

    This is sampling without replacement, not an independent Bernoulli draw,
    so every shape with the same ``M`` has the same edge count.
    """
    if graph_family == "multiscale_dag_v1":
        from cot_navigation.data.hard_graph import sample_hard_edges
        return sample_hard_edges(n_layers, width, n_edges, rng)
    if graph_family != "layered":
        raise ValueError(f"unknown graph family {graph_family}")
    pool = candidate_edges(n_layers, width)
    if n_edges > len(pool):
        raise ValueError(f"need {n_edges} edges but only {len(pool)} candidates")
    # Uniform over subsets. Presentation order is a separate permutation.
    pick = rng.choice(len(pool), size=n_edges, replace=False)
    return [pool[int(i)] for i in pick]


def structure_hash(n_layers: int, width: int, edges: list[tuple[int, int]]) -> int:
    """Identity of the pre-rename edge set.

    A sorted-edge hash detects exact copies after relabeling tokens or
    reshuffling the edge list. It is not a graph-isomorphism test, and the
    project does not claim one.
    """
    hasher = hashlib.blake2s(digest_size=8)
    hasher.update(int(n_layers).to_bytes(2, "little"))
    hasher.update(int(width).to_bytes(2, "little"))
    for u, v in sorted(edges):
        hasher.update(int(u).to_bytes(2, "little"))
        hasher.update(int(v).to_bytes(2, "little"))
    return int.from_bytes(hasher.digest(), "little")


def adj_from_presentation(n_nodes: int, presented: list[tuple[int, int]]) -> list[list[int]]:
    """Neighbor order follows first appearance in the presented edge list."""
    adj: list[list[int]] = [[] for _ in range(n_nodes)]
    seen: set[tuple[int, int]] = set()
    for u, v in presented:
        if (u, v) in seen:
            raise ValueError(f"duplicate edge {(u, v)}")
        seen.add((u, v))
        adj[u].append(v)
    return adj


def shuffle_neighbors(adj: list[list[int]], rng: Generator) -> list[list[int]]:
    """Fixed-graph neighbor order: one random permutation per node, then frozen."""
    out: list[list[int]] = []
    for nbrs in adj:
        order = rng.permutation(len(nbrs))
        out.append([nbrs[int(i)] for i in order])
    return out


def edges_of(adj: list[list[int]]) -> list[tuple[int, int]]:
    return [(u, v) for u, nbrs in enumerate(adj) for v in nbrs]

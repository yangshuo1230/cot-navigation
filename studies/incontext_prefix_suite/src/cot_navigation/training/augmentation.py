"""Small online transformations over in-memory graphs and cached token rows.

Never mutate cached examples. RNGs are supplied by the trainer and checkpointed;
evaluation uses separate deterministic per-query seeds. No disk IO is performed.
"""
from dataclasses import replace
import numpy as np
from cot_navigation.protocols.render import render_task

SEARCH_PROTOCOLS = {"dfs_state", "bfs_state", "dfs_nodes", "bfs_nodes"}


def with_body(row, body):
    answer_index = row.prompt_len + len(body)
    return replace(row, tokens=row.tokens[:row.prompt_len] + body + row.tokens[row.answer_index:],
                   answer_index=answer_index,
                   trace_indices=list(range(row.prompt_len, answer_index)) + [answer_index + 1],
                   meta={**row.meta, "full_body_len": len(body), "full_body": body})


def random_path(adj, src, rng):
    """Uniform outgoing-edge choices until a sink, including a length-zero path.

    This is not uniform over complete paths. DAG structure guarantees termination;
    the guard rejects invalid cyclic graphs rather than silently truncating.
    """
    nodes = [src]
    while adj[nodes[-1]]:
        nxt = int(rng.choice(adj[nodes[-1]]))
        if nxt in nodes:
            raise ValueError("random negative path requires a DAG")
        nodes.append(nxt)
    return nodes


def transform(row, protocol, tok, adj, nodes, src, dst, *,
              edge_rng=None, search_rng=None, negative_rng=None):
    prompt = row.tokens[:row.prompt_len]
    if edge_rng is not None:
        edges = np.asarray(prompt[:-4]).reshape(-1, 3)
        prompt = edges[edge_rng.permutation(len(edges))].reshape(-1).tolist() + prompt[-4:]
        if protocol in SEARCH_PROTOCOLS:
            # The search candidate order must agree with the newly shown edge list.
            inverse = {token: i for i, token in enumerate(nodes)}
            adj = [[] for _ in nodes]
            for _, u, v in np.asarray(prompt[:-4]).reshape(-1, 3):
                adj[inverse[int(u)]].append(inverse[int(v)])
    if search_rng is not None:
        adj = [search_rng.permutation(vs).astype(int).tolist() for vs in adj]
    if protocol in SEARCH_PROTOCOLS and (edge_rng is not None or search_rng is not None):
        rendered = render_task(tok, protocol, row.label, prompt, adj, src, dst, nodes)
        row = replace(rendered, meta={**row.meta, **rendered.meta})
    elif prompt != row.tokens[:row.prompt_len]:
        row = replace(row, tokens=prompt + row.tokens[row.prompt_len:], meta=dict(row.meta))
    if protocol == "path" and row.label == 0 and negative_rng is not None:
        row = with_body(row, [nodes[v] for v in random_path(adj, src, negative_rng)])
    return row

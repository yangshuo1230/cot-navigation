"""Complete trajectories: EDGE u v ... QUERY s t |, with no ANS or budget.

DFS records use: stack | visited | candidates | action [node] ;
BFS records use the same tokens with a FIFO queue as the first field.
Node-only search emits unique first-expansion nodes, then the final answer.
All states precede their action. A search body that would end inside the last
few context positions is cut on a record boundary, then <STOP> and the answer.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from cot_navigation.protocols.dfs import Step, dfs_search, witness_path
from cot_navigation.protocols.bfs import bfs_search, first_expansion_order
from cot_navigation.protocols.tokenizer import Tokenizer

@dataclass
class RenderedExample:
    tokens: list[int]
    prompt_len: int
    answer_index: int
    trace_indices: list[int]
    walk_indices: list[int]
    label: int
    filler_tokens: int = 0
    meta: dict = field(default_factory=dict)

def render_steps(steps: list[Step], node_token: list[int], tok: Tokenizer) -> list[int]:
    ids = []
    for step in steps:
        # CUR is the last stack entry; VIS persists after POP.
        for nodes in (step.stack, step.visited, step.cands):
            ids.extend(node_token[v] for v in nodes)
            ids.append(tok.id("|"))
        ids.append(tok.id(f"<{step.action}>"))
        if step.action == "PUSH":
            assert step.arg is not None
            ids.append(node_token[step.arg])
        ids.append(tok.id(";"))
    return ids

def render_path_body(steps: list[Step], node_token: list[int]) -> list[int]:
    path = witness_path(steps)
    return [] if path is None else [node_token[v] for v in path]

def context_prompt(tok: Tokenizer, node_token: list[int], edges: list[tuple[int, int]],
                   src: int, dst: int) -> list[int]:
    ids = []
    for u, v in edges:
        ids.extend((tok.id("<EDGE>"), node_token[u], node_token[v]))
    return ids + parametric_prompt(tok, node_token, src, dst)

def parametric_prompt(tok: Tokenizer, node_token: list[int], src: int, dst: int) -> list[int]:
    return [tok.id("<QUERY>"), node_token[src], node_token[dst], tok.id("|")]

def render_task(tok: Tokenizer, protocol: str, label: int, prompt: list[int],
                adj: list[list[int]], src: int, dst: int, node_token: list[int],
                *, filler_tokens: int = 0, path_nodes: list[int] | None = None) -> RenderedExample:
    steps = (bfs_search if protocol.startswith("bfs_") else dfs_search)(adj, src, dst)
    if steps[-1].action != ("FOUND" if label == 1 else "EXHAUST"):
        raise ValueError("reference search and reachability label disagree")
    if protocol == "direct":
        body = []
    elif protocol == "path":
        if path_nodes is None:
            body = render_path_body(steps, node_token)
        else:
            if (label != 1 or not path_nodes or path_nodes[0] != src or path_nodes[-1] != dst
                    or len(path_nodes) != len(set(path_nodes))
                    or any(v < 0 or v >= len(adj) for v in path_nodes)
                    or any(v not in adj[u] for u, v in zip(path_nodes, path_nodes[1:]))):
                raise ValueError("selected path is not a valid witness for this query")
            body = [node_token[v] for v in path_nodes]
    elif protocol in {"dfs_state", "bfs_state"}:
        body = render_steps(steps, node_token, tok)
    elif protocol in {"dfs_nodes", "bfs_nodes"}:
        body = [node_token[v] for v in first_expansion_order(steps)]
    elif protocol == "filler":
        if filler_tokens < 1:
            raise ValueError("filler_tokens must be positive")
        body = [tok.id("<DOT>")] * filler_tokens
    else:
        raise KeyError(protocol)
    tokens = prompt + body + [tok.label_id(label), tok.id("<EOS>")]
    answer_index = len(prompt) + len(body)
    trace_indices = []
    if protocol in {"path", "dfs_state", "bfs_state", "dfs_nodes", "bfs_nodes"}:
        trace_indices = list(range(len(prompt), answer_index)) + [answer_index + 1]
    return RenderedExample(
        tokens=tokens, prompt_len=len(prompt), answer_index=answer_index,
        trace_indices=trace_indices, walk_indices=[], label=label,
        filler_tokens=filler_tokens if protocol == "filler" else 0,
        meta={"full_body_len": len(body), "full_body": body, "steps": len(steps),
              "reference_steps": steps, "node_token": node_token},
    )

def truncate_example(row: RenderedExample, tok: Tokenizer, max_seq_len: int, margin: int) -> RenderedExample:
    """Keep the answer outside the last ``margin`` context positions.

    State traces are cut after a completed ``;`` record. Path and node traces,
    which have no record marker, are cut on a token boundary. <STOP> is
    supervised, then the original YES/NO and EOS.
    """
    if margin <= 0 or len(row.tokens) <= max_seq_len - margin:
        return row
    limit = max_seq_len - margin
    answer = row.tokens[row.answer_index:]
    if len(answer) < 2:
        raise ValueError("truncated example is missing its YES/NO and EOS tail")
    stop = tok.id("<STOP>")
    tail = [stop, *answer]
    budget = limit - row.prompt_len - len(tail)
    if budget < 0:
        raise ValueError(f"prompt length {row.prompt_len} leaves no room for STOP and the answer")
    body = row.tokens[row.prompt_len:row.answer_index]
    if body and body[-1] == stop:
        body = body[:-1]
    semi = tok.id(";")
    if any(token == semi for token in body):
        cut_at = 0
        for index, token in enumerate(body[:budget]):
            if token == semi:
                cut_at = index + 1
    else:
        cut_at = budget
    new_body = body[:cut_at] + [stop]
    answer_index = row.prompt_len + len(new_body)
    updated = RenderedExample(
        tokens=row.tokens[:row.prompt_len] + new_body + answer,
        prompt_len=row.prompt_len, answer_index=answer_index,
        trace_indices=list(range(row.prompt_len, answer_index)) + [answer_index + 1],
        walk_indices=row.walk_indices, label=row.label, filler_tokens=row.filler_tokens,
        meta={**row.meta, "full_body_len": len(body), "truncated": True},
    )
    return updated


def render_walk(tok: Tokenizer, node_token: list[int], nodes: list[int]) -> RenderedExample:
    """Only transitions carry loss; EOS is a corpus boundary, not a sink label."""
    if len(nodes) < 2:
        raise ValueError("walk must contain at least one edge")
    ids = [tok.id("<WALK>"), *[node_token[u] for u in nodes], tok.id("<EOS>")]
    return RenderedExample(tokens=ids, prompt_len=2, answer_index=-1,
                           trace_indices=[], walk_indices=list(range(2, len(ids) - 1)), label=-1)

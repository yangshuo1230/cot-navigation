"""Complete trajectories: EDGE u v ... QUERY s t |, with no ANS or budget.

DFS records use: stack | visited | candidates | action [node] ;
All states precede their action. Training examples are never truncated.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from cot_navigation.protocols.dfs import Step, dfs_search, witness_path
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
                *, filler_tokens: int = 0) -> RenderedExample:
    steps = dfs_search(adj, src, dst)
    if steps[-1].action != ("FOUND" if label == 1 else "EXHAUST"):
        raise ValueError("reference DFS and reachability label disagree")
    if protocol == "direct":
        body = []
    elif protocol == "path":
        body = render_path_body(steps, node_token)
    elif protocol == "dfs_state":
        body = render_steps(steps, node_token, tok)
    elif protocol == "filler":
        if filler_tokens < 1:
            raise ValueError("filler_tokens must be positive")
        body = [tok.id("<DOT>")] * filler_tokens
    else:
        raise KeyError(protocol)
    tokens = prompt + body + [tok.label_id(label), tok.id("<EOS>")]
    answer_index = len(prompt) + len(body)
    trace_indices = []
    if protocol in {"path", "dfs_state"}:
        trace_indices = list(range(len(prompt), answer_index)) + [answer_index + 1]
    return RenderedExample(
        tokens=tokens, prompt_len=len(prompt), answer_index=answer_index,
        trace_indices=trace_indices, walk_indices=[], label=label,
        filler_tokens=filler_tokens if protocol == "filler" else 0,
        meta={"full_body_len": len(body), "full_body": body, "steps": len(steps),
              "reference_steps": steps, "node_token": node_token},
    )

def render_walk(tok: Tokenizer, node_token: list[int], nodes: list[int]) -> RenderedExample:
    """Only transitions carry loss; EOS is a corpus boundary, not a sink label."""
    if len(nodes) < 2:
        raise ValueError("walk must contain at least one edge")
    ids = [tok.id("<WALK>"), *[node_token[u] for u in nodes], tok.id("<EOS>")]
    return RenderedExample(tokens=ids, prompt_len=2, answer_index=-1,
                           trace_indices=[], walk_indices=list(range(2, len(ids) - 1)), label=-1)

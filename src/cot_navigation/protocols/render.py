"""Render protocol sequences and the loss-target index sets.

Trace budget
------------
Tokens that count toward ``T`` are the trace body only:

* ``dfs_state``: every token of the step records, including markers and node ids
* ``path``: ``<PATH>`` / ``<NONE>`` and the witness node ids
* ``filler``: exactly ``T`` ``<DOT>`` tokens
* ``direct``: empty body; ``T`` is written on the prompt and never inserts ``<CUT>``

The answer tail is not part of ``T`` and is not dropped to satisfy ``T``:

* unfinished path/DFS body: ``<CUT> <ANS> <YES|NO> <EOS>``
* finished path/DFS body: ``<ANS> <YES|NO> <EOS>``
* direct and filler: ``<ANS> <YES|NO> <EOS>`` (filler dots are the body)

``len(prompt) + T + 4`` must fit ``max_seq_len``. The 4 covers the longest tail.
Training, forced slicing, and the test controller all use this definition.
A slice may end inside a DFS record. Diagnostics grade only finished records.
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
    budget: int
    cut: bool
    meta: dict = field(default_factory=dict)


def mode_name(protocol: str) -> str:
    return {"direct": "<DIRECT>", "path": "<PATH>", "dfs_state": "<DFS>", "filler": "<FILLER>"}[protocol]


def render_steps(steps: list[Step], node_token: list[int], tok: Tokenizer) -> list[int]:
    """Map a DFS trace to vocab ids. ``node_token[topo]`` is a vocab id."""
    ids: list[int] = []
    for step in steps:
        ids.extend((tok.id("<STEP>"), tok.id("<CUR>"), node_token[step.cur], tok.id("<CAND>")))
        ids.extend(node_token[v] for v in step.cands)
        ids.append(tok.id("<STACK>"))
        ids.extend(node_token[v] for v in step.stack)
        ids.append(tok.id("<VIS>"))
        ids.extend(node_token[v] for v in step.visited)
        ids.extend((tok.id("<ACT>"), tok.id(f"<{step.action}>")))
        if step.action == "PUSH":
            assert step.arg is not None
            ids.append(node_token[step.arg])
        ids.append(tok.id("<ENDSTEP>"))
    return ids


def render_path_body(steps: list[Step], node_token: list[int], tok: Tokenizer) -> list[int]:
    path = witness_path(steps)
    if path is None:
        return [tok.id("<NONE>")]
    return [tok.id("<PATH>"), *[node_token[v] for v in path]]


def context_prompt(
    tok: Tokenizer,
    node_token: list[int],
    node_order: list[int],
    edges: list[tuple[int, int]],
    src: int,
    dst: int,
    protocol: str,
    budget: int,
) -> list[int]:
    ids = [tok.id("<GRAPH>"), tok.id("<NODES>")]
    ids.extend(node_token[topo] for topo in node_order)
    ids.append(tok.id("<EDGES>"))
    for u, v in edges:
        ids.extend((tok.id("<E>"), node_token[u], node_token[v]))
    ids.extend(
        (
            tok.id("<QUERY>"),
            node_token[src],
            node_token[dst],
            tok.mode_id(protocol),
            tok.id("<BUDGET>"),
            tok.budget_id(budget),
            tok.id("<OUT>"),
        )
    )
    return ids


def parametric_prompt(tok: Tokenizer, node_token: list[int], src: int, dst: int, protocol: str, budget: int) -> list[int]:
    return [
        tok.id("<QUERY>"),
        node_token[src],
        node_token[dst],
        tok.mode_id(protocol),
        tok.id("<BUDGET>"),
        tok.budget_id(budget),
        tok.id("<OUT>"),
    ]


def _tail(tok: Tokenizer, label: int, cut: bool) -> list[int]:
    label_id = tok.label_id(label)
    head = [tok.id("<CUT>")] if cut else []
    return head + [tok.id("<ANS>"), label_id, tok.id("<EOS>")]


def attach_answer(body: list[int], budget: int, label: int, tok: Tokenizer, use_budget: bool) -> tuple[list[int], bool]:
    """Slice the body to at most ``T`` tokens, then append the answer tail."""
    if use_budget and len(body) > budget:
        return body[:budget] + _tail(tok, label, cut=True), True
    return body + _tail(tok, label, cut=False), False


def _index_sets(tokens: list[int], prompt_len: int, tok: Tokenizer, supervise_trace: bool) -> tuple[int, list[int]]:
    # The scored label is the token after the tail's ANS, not a YES found earlier.
    ans_pos = _rfind(tokens, tok.id("<ANS>"), prompt_len)
    if ans_pos < 0 or ans_pos + 1 >= len(tokens):
        raise RuntimeError("rendered sequence has no answer slot")
    answer_index = ans_pos + 1
    if tokens[answer_index] not in (tok.yes, tok.no):
        raise RuntimeError("answer slot is not YES/NO")
    trace: list[int] = []
    if supervise_trace:
        trace = [i for i in range(prompt_len, len(tokens)) if i != answer_index]
    return answer_index, trace


def _rfind(tokens: list[int], target: int, start: int) -> int:
    for i in range(len(tokens) - 1, start - 1, -1):
        if tokens[i] == target:
            return i
    return -1


def render_task(
    tok: Tokenizer,
    protocol: str,
    budget: int,
    label: int,
    prompt: list[int],
    adj: list[list[int]],
    src: int,
    dst: int,
    node_token: list[int],
) -> RenderedExample:
    steps = dfs_search(adj, src, dst)
    if label == 1 and steps[-1].action != "FOUND":
        raise RuntimeError("positive query did not find a witness")
    if label == 0 and steps[-1].action != "EXHAUST":
        raise RuntimeError("negative query did not exhaust")
    if protocol == "direct":
        body: list[int] = []
        output, cut = attach_answer(body, budget, label, tok, use_budget=False)
        supervise_trace = False
    elif protocol == "path":
        body = render_path_body(steps, node_token, tok)
        output, cut = attach_answer(body, budget, label, tok, use_budget=True)
        supervise_trace = True
    elif protocol == "dfs_state":
        body = render_steps(steps, node_token, tok)
        output, cut = attach_answer(body, budget, label, tok, use_budget=True)
        supervise_trace = True
    elif protocol == "filler":
        body = [tok.id("<DOT>")] * budget
        output, cut = attach_answer(body, budget, label, tok, use_budget=False)
        cut = False
        supervise_trace = False
    else:
        raise KeyError(protocol)
    tokens = prompt + output
    answer_index, trace_indices = _index_sets(tokens, len(prompt), tok, supervise_trace)
    full_body = list(body)
    return RenderedExample(
        tokens=tokens,
        prompt_len=len(prompt),
        answer_index=answer_index,
        trace_indices=trace_indices,
        walk_indices=[],
        label=label,
        budget=budget,
        cut=cut,
        meta={"full_body_len": len(full_body), "full_body": full_body, "steps": len(steps),
              "reference_steps": steps, "node_token": node_token},
    )


def render_walk(tok: Tokenizer, node_token: list[int], nodes: list[int]) -> RenderedExample:
    """Supervise transition nodes only, not the start and not ``<EOS>``.

    The walk may stop early at a sink or at the sampled length. ``<EOS>`` marks
    that data boundary and is not a label for "no out-edge".
    """
    if len(nodes) < 2:
        raise ValueError("walk must contain at least one edge")
    ids = [tok.id("<WALK>")]
    ids.extend(node_token[u] for u in nodes)
    ids.append(tok.id("<EOS>"))
    # tokens: WALK start n1 n2 ... EOS
    # indices: 0    1     2  3      -1
    walk_indices = list(range(2, len(ids) - 1))
    return RenderedExample(
        tokens=ids,
        prompt_len=2,
        answer_index=-1,
        trace_indices=[],
        walk_indices=walk_indices,
        label=-1,
        budget=0,
        cut=False,
    )

"""Parse DFS records and score final answers.

The main score reads only the protocol answer slot: the token after the ANS
that ends the sequence (model-emitted or controller-inserted). A YES/NO that
appears earlier in a broken trace does not count. Trace checks are separate
and never veto a correct final answer.
"""

from __future__ import annotations

from cot_navigation.protocols.dfs import Step
from cot_navigation.protocols.tokenizer import Tokenizer


def parse_completed_steps(tokens: list[int], tok: Tokenizer) -> list[dict]:
    """Return finished STEP records. A trailing partial record is ignored."""
    step_id = tok.id("<STEP>")
    end_id = tok.id("<ENDSTEP>")
    records: list[dict] = []
    i = 0
    while i < len(tokens):
        if tokens[i] != step_id:
            i += 1
            continue
        j = i + 1
        while j < len(tokens) and tokens[j] != end_id:
            j += 1
        if j >= len(tokens):
            break
        parsed = _parse_record(tokens[i : j + 1], tok)
        # A STEP/ENDSTEP pair that is missing a required field is not a completed record.
        if parsed is not None:
            records.append(parsed)
        i = j + 1
    return records


def _parse_record(tokens: list[int], tok: Tokenizer) -> dict | None:
    def marker(name: str) -> int | None:
        token_id = tok.id(name)
        if token_id not in tokens:
            return None
        return tokens.index(token_id)

    def nodes_between(left: str, right: str) -> list[int]:
        a = marker(left) + 1
        b = marker(right)
        return tokens[a:b]

    order = [marker(name) for name in ("<CUR>", "<CAND>", "<STACK>", "<VIS>", "<ACT>")]
    if any(pos is None for pos in order) or order != sorted(order):
        return None
    if len(set(order)) != len(order) or order[1] != order[0] + 2:
        return None
    act_at = order[-1]
    assert act_at is not None
    if act_at + 1 >= len(tokens):
        return None
    action_id = tokens[act_at + 1]
    if not 0 <= action_id < len(tok.itos):
        return None
    action = tok.itos[action_id]
    if action not in {"<PUSH>", "<POP>", "<FOUND>", "<EXHAUST>"}:
        return None
    arg = None
    if action == "<PUSH>":
        if act_at + 3 != len(tokens) - 1:
            return None
        arg = tokens[act_at + 2]
    elif act_at + 2 != len(tokens) - 1:
        return None
    node_fields = [tokens[order[0] + 1], *nodes_between("<CAND>", "<STACK>"),
                   *nodes_between("<STACK>", "<VIS>"), *nodes_between("<VIS>", "<ACT>")]
    if arg is not None:
        node_fields.append(arg)
    if any(not tok.node_lo <= v < len(tok) for v in node_fields):
        return None
    return {
        "cur": tokens[marker("<CUR>") + 1],
        "cands": nodes_between("<CAND>", "<STACK>"),
        "stack": nodes_between("<STACK>", "<VIS>"),
        "visited": nodes_between("<VIS>", "<ACT>"),
        "action": action.strip("<>"),
        "arg": arg,
    }


def step_to_expected(step: Step, node_token: list[int]) -> dict:
    return {
        "cur": node_token[step.cur],
        "cands": [node_token[v] for v in step.cands],
        "stack": [node_token[v] for v in step.stack],
        "visited": [node_token[v] for v in step.visited],
        "action": step.action,
        "arg": None if step.arg is None else node_token[step.arg],
    }


def compare_completed(reference: list[Step], parsed: list[dict], node_token: list[int]) -> dict:
    """Field accuracy over completed parsed records aligned to the reference prefix."""
    n = len(parsed)
    field_ok = {"cur": 0, "cands": 0, "stack": 0, "visited": 0, "action": 0}
    first_error = None
    matched = 0
    for i in range(n):
        if i >= len(reference):
            if first_error is None:
                first_error = i
            continue
        exp = step_to_expected(reference[i], node_token)
        got = parsed[i]
        good = True
        for key in field_ok:
            if got[key] == exp[key] and (key != "action" or got["arg"] == exp["arg"]):
                field_ok[key] += 1
            else:
                good = False
        if got["arg"] != exp["arg"]:
            good = False
        if not good and first_error is None:
            first_error = i
        if good:
            matched += 1
    return {
        "n_completed": n,
        "n_reference": len(reference),
        "field_correct": field_ok,
        "first_error_record": first_error,
        "matched_records": matched,
        "complete_match": first_error is None and n == len(parsed),
    }


def answer_slot(tokens: list[int], prompt_len: int, tok: Tokenizer) -> int | None:
    """Index of the scored label, or None when the protocol slot is missing.

    Search is restricted to the token immediately following the last ANS at or
    after ``prompt_len``. Earlier YES/NO strings are ignored.
    """
    ans = tok.id("<ANS>")
    pos = None
    for i in range(prompt_len, len(tokens)):
        if tokens[i] == ans:
            pos = i
    if pos is None or pos + 1 >= len(tokens):
        return None
    return pos + 1


def score_final_answer(tokens: list[int], prompt_len: int, label: int, tok: Tokenizer) -> dict:
    slot = answer_slot(tokens, prompt_len, tok)
    if slot is None or tokens[slot] not in (tok.yes, tok.no):
        return {"correct": False, "format_fail": True, "pred": None}
    pred = 1 if tokens[slot] == tok.yes else 0
    return {"correct": pred == label, "format_fail": False, "pred": pred}

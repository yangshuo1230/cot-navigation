"""Compact DFS diagnostics never veto a correct final YES/NO answer."""
from __future__ import annotations
from cot_navigation.protocols.dfs import Step
from cot_navigation.protocols.tokenizer import Tokenizer

def parse_completed_steps(tokens: list[int], tok: Tokenizer) -> list[dict]:
    records, start = [], 0
    for end, token in enumerate(tokens):
        if token != tok.id(";"):
            continue
        parsed = _parse_record(tokens[start:end], tok)
        if parsed is not None:
            records.append(parsed)
        start = end + 1
    return records

def _parse_record(tokens: list[int], tok: Tokenizer) -> dict | None:
    fields = [[]]
    for token in tokens:
        if token == tok.id("|"):
            fields.append([])
        else:
            fields[-1].append(token)
    if len(fields) != 4:
        return None
    stack, visited, cands, action = fields
    if not stack or not visited or not action:
        return None
    actions = {tok.id(f"<{name}>"): name for name in ("PUSH", "POP", "FOUND", "EXHAUST")}
    name = actions.get(action[0])
    if name is None or len(action) != (2 if name == "PUSH" else 1):
        return None
    arg = action[1] if name == "PUSH" else None
    nodes = stack + visited + cands + ([] if arg is None else [arg])
    if any(not tok.node_lo <= v < len(tok) for v in nodes):
        return None
    return {"cur": stack[-1], "stack": stack, "visited": visited,
            "cands": cands, "action": name, "arg": arg}

def step_to_expected(step: Step, node_token: list[int]) -> dict:
    return {"cur": node_token[step.cur], "cands": [node_token[v] for v in step.cands],
            "stack": [node_token[v] for v in step.stack],
            "visited": [node_token[v] for v in step.visited], "action": step.action,
            "arg": None if step.arg is None else node_token[step.arg]}

def compare_completed(reference: list[Step], parsed: list[dict], node_token: list[int]) -> dict:
    field_ok = {key: 0 for key in ("cur", "cands", "stack", "visited", "action")}
    first_error, matched = None, 0
    for i, got in enumerate(parsed):
        if i >= len(reference):
            if first_error is None:
                first_error = i
            continue
        exp = step_to_expected(reference[i], node_token)
        good = True
        for key in field_ok:
            correct = got[key] == exp[key] and (key != "action" or got["arg"] == exp["arg"])
            field_ok[key] += int(correct)
            good &= correct
        if not good and first_error is None:
            first_error = i
        matched += int(good)
    return {"n_completed": len(parsed), "n_reference": len(reference),
            "field_correct": field_ok, "first_error_record": first_error,
            "matched_records": matched,
            "complete_match": first_error is None and len(parsed) == len(reference)}

def answer_slot(tokens: list[int], prompt_len: int, tok: Tokenizer) -> int | None:
    """The first generated YES/NO terminates the answer, regardless of trace quality."""
    for i in range(prompt_len, len(tokens)):
        if tokens[i] in (tok.yes, tok.no):
            return i
        if tokens[i] == tok.id("<EOS>"):
            return None
    return None

def score_final_answer(tokens: list[int], prompt_len: int, label: int, tok: Tokenizer) -> dict:
    slot = answer_slot(tokens, prompt_len, tok)
    if slot is None:
        return {"correct": False, "format_fail": True, "pred": None}
    pred = int(tokens[slot] == tok.yes)
    return {"correct": pred == label, "format_fail": False, "pred": pred}

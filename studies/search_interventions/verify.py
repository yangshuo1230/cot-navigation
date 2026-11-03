"""CPU-only reconstruction and independent token-level legality verification."""
import json
from pathlib import Path
import time

from run import build_cases, check_continuation
from cot_navigation.analysis.scale_followup import make_adapter
from cot_navigation.analysis.error_detection import semantic_slots
from cot_navigation.analysis.common import sha256, write_json
from cot_navigation.protocols.dfs import dfs_search
from cot_navigation.protocols.bfs import bfs_search
from cot_navigation.protocols.render import render_steps

root = Path.cwd()
folder = root / "studies/search_interventions"
verified = []
entries = json.loads((root / "studies/scale_mechanistic/results/selection.json").read_text())["models"]
paths = sorted(folder / "results/models" / (entry["id"] + ".json") for entry in entries
               if entry["protocol"] in {"dfs_state", "bfs_state"})
for path in paths:
    deadline = time.time() + 600
    while not path.exists():
        if time.time() > deadline:
            raise TimeoutError(f"Inference result not ready: {path}")
        time.sleep(2)
    result = json.loads(path.read_text())
    assert result["complete"] and result["checkpoint_unchanged"]
    am = make_adapter(result["entry"], root, "cpu")
    examples = [am.example(q, "test") for q in result["query_ids"]]
    cases = build_cases(am, examples, result["cases_per_family_limit"])
    reference = {ex.index: ex.prompt + render_steps(
        (bfs_search if am.protocol == "bfs_state" else dfs_search)(ex.adj, ex.source, ex.target), ex.nodes, am.tok)
        + [am.tok.label_id(ex.label)] for ex in examples}
    lookup = {(r["family"], r["query"], r["method"]): r for r in result["rows"]}
    assert len(lookup) == len(result["rows"])
    clean_checked = legal_branch_checked = 0
    expected_rows = 0
    for case in cases:
        ex = case["example"]
        for method, arm in case["arms"].items():
            expected_rows += 1
            row = lookup[(case["family"], case["query"], method)]
            assert row["prefix_sha256"] == arm["prefix_sha256"]
            assert row["oracle_answer"] == arm["oracle_answer"]
            assert row["binary_flip"] == (row["pred"] is not None and row["clean_pred"] is not None and row["pred"] != row["clean_pred"])
            assert am.tok.yes not in arm["prefix"] and am.tok.no not in arm["prefix"]
            body = arm["prefix"][len(ex.prompt):]
            slots = list(semantic_slots(ex, am.protocol, am.tok, body))
            prefix_legal = len(slots) == len(body) and all(body[i] in allowed for i, allowed, _ in slots)
            # Refine the generator's coarse state-match annotation using the
            # independent checker, including conflicts in CAND/action fields.
            row["history_consistent"] = prefix_legal
            row["input_prefix_legal"] = prefix_legal
            row["first_record_found"] = row["first_record_action"] == [am.tok.id("<FOUND>")]
            # Use all applicable arms, not just parseable transitions, as the
            # denominator. Terminal states have no required next state.
            current = arm["frontier"][0 if am.protocol == "bfs_state" else -1]
            inverse = {node: i for i, node in enumerate(ex.nodes)}
            eligible = {ex.nodes[v] for v in ex.adj[inverse[current]]} - set(arm["visited"])
            expects_nonterminal = current != ex.nodes[ex.target] and (bool(eligible) or len(arm["frontier"]) > 1)
            requires_next = arm["forced_action"] == "PUSH" or not arm["forced_action"] and expects_nonterminal
            if requires_next and row["next_state_correct"] is None:
                row["next_state_correct"] = False
            if method == "clean" or case["family"] == "legal_push" or method == "candidate_reorder":
                assert prefix_legal, (path.name, case["family"], method)
                if method == "clean":
                    clean_checked += 1
                    full = reference[ex.index]
                    assert arm["prefix"] == full[:len(arm["prefix"])]
                    replay = check_continuation(am, ex, arm, {"tokens": full, "pred": ex.label,
                        "generated": full[len(arm["prefix"]):]})
                    assert replay["suffix_complete_valid"], replay
                else:
                    legal_branch_checked += 1
    assert expected_rows == len(lookup)
    assert sha256(am.run / "checkpoints/best.pt") == result["entry"]["checkpoint_sha256"]
    result["independent_prefix_annotations"] = True
    result["next_state_denominator"] = "all arms requiring a nonterminal transition; missing or invalid transitions count as failure"
    result["verification_source_sha256"] = sha256(Path(__file__))
    write_json(path, result)
    verified.append({"model": result["entry"]["id"], "rows": expected_rows,
                     "clean_prefixes_legal": clean_checked, "legal_alternatives_checked": legal_branch_checked,
                     "checkpoint_unchanged": True})
    print("VERIFIED", path.stem, expected_rows, flush=True)
    del am
assert len(verified) == 20
write_json(folder / "results/verification.json", {"status": "PASS", "models": verified,
    "checks": ["deterministic full case reconstruction", "every prefix hash matches",
               "independent semantic_slots accepts every clean prefix, legal PUSH alternative and CAND reorder",
               "no answer token in prefixes", "every oracle label and binary flip reconstructed",
               "all 20 validation-selected checkpoint hashes unchanged"]})

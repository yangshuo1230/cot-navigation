"""Frozen DFS/BFS search-state interventions; no training or checkpoint writes.

Reference histories are legal, selected without looking at model outputs.
Every intervention has a clean continuation at the same logical cut.
Conflicting fields are explicitly distinguished from legal PUSH branches.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import time


def machine(adj, frontier, visited, target, fifo):
    """Oracle from an arbitrary active search state; discovery occurs on PUSH."""
    ff, seen = list(frontier), set(visited)
    for _ in range(4 * len(adj) + 4):
        if not ff:
            return 0
        u = ff[0 if fifo else -1]
        if u == target:
            return 1
        candidates = [v for v in adj[u] if v not in seen]
        if candidates:
            v = candidates[0]
            ff.append(v)
            seen.add(v)
        elif len(ff) == 1:
            return 0
        else:
            ff.pop(0 if fifo else -1)
    raise AssertionError("DAG search exceeded linear action bound")


def graph_oracle(adj, frontier, visited, target):
    """Independent reachability check: active frontier roots remain expandable."""
    blocked = set(visited) - set(frontier)
    work, seen = list(frontier), set(frontier)
    while work:
        u = work.pop()
        if u == target:
            return 1
        for v in adj[u]:
            if v not in blocked and v not in seen:
                seen.add(v)
                work.append(v)
    return 0


def parse_records(body, sep, semi):
    rows, start = [], 0
    for end, token in enumerate(body):
        if token != semi:
            continue
        parts = [[]]
        for value in body[start:end]:
            if value == sep:
                parts.append([])
            else:
                parts[-1].append(value)
        rows.append(parts if len(parts) == 4 else None)
        start = end + 1
    return rows, start == len(body)


def check_continuation(am, ex, arm, out):
    """Replay from the injected state, rather than the original query source.

    Forced candidate fields are omitted from the model's candidate score.
    A forced invalid terminal action is recorded as an input conflict, not
    attributed to a generated model error.
    """
    tok = am.tok
    adj = {ex.nodes[u]: set(ex.nodes[v] for v in vs) for u, vs in enumerate(ex.adj)}
    target = ex.nodes[ex.target]
    fifo = am.protocol == "bfs_state"
    ff, seen = list(arm["frontier"]), set(arm["visited"])
    suffix = out["tokens"][arm["record_start"]:]
    # The first sampled answer ends generation. Remove it from record replay.
    answer_at = next((i for i, v in enumerate(suffix) if v in (tok.yes, tok.no, tok.id("<STOP>"), tok.id("<EOS>"))), len(suffix))
    body = suffix[:answer_at]
    records, finished = parse_records(body, tok.id("|"), tok.id(";"))
    first_error = None
    terminal = None
    first_candidate = first_action = next_state = None
    premature = False
    for index, record in enumerate(records):
        if record is None:
            first_error = first_error or "malformed_record"
            break
        shown_ff, shown_vis, candidates, action = record
        if index == 1:
            next_state = shown_ff == ff and len(set(shown_vis)) == len(shown_vis) and set(shown_vis) == seen
        if terminal:
            first_error = first_error or "record_after_terminal"
            break
        if shown_ff != ff or len(shown_vis) != len(set(shown_vis)) or set(shown_vis) != seen:
            first_error = first_error or "state_mismatch"
            break
        current = ff[0 if fifo else -1]
        eligible = adj[current] - seen
        candidate_ok = len(candidates) == len(set(candidates)) and set(candidates) == eligible
        forced_candidates = index == 0 and arm["forced_candidates"]
        if index == 0 and not forced_candidates:
            first_candidate = candidate_ok
        if not candidate_ok and not forced_candidates:
            first_error = first_error or "candidate_mismatch"
        expected = "FOUND" if current == target else "PUSH" if eligible else "POP" if len(ff) > 1 else "EXHAUST"
        action_ok = bool(action and action[0] == tok.id(f"<{expected}>") and
                         (len(action) == 2 and action[1] in eligible if expected == "PUSH" else len(action) == 1))
        if index == 0 and not arm["forced_action"]:
            first_action = action_ok
        if action == [tok.id("<FOUND>")] and current != target and not (index == 0 and arm["forced_action"]):
            premature = True
        if not action_ok:
            first_error = first_error or ("forced_action_conflict" if index == 0 and arm["forced_action"] else "wrong_action")
            break
        if expected == "PUSH":
            ff.append(action[1])
            seen.add(action[1])
        elif expected == "POP":
            ff.pop(0 if fifo else -1)
        else:
            terminal = expected
    if not finished:
        first_error = first_error or "incomplete_record"
    complete = first_error is None and ((terminal == "FOUND" and out["pred"] == 1) or
                                         (terminal == "EXHAUST" and out["pred"] == 0))
    # Missing/unparseable generated records are failures for the relevant local check.
    if not arm["forced_candidates"] and first_candidate is None:
        first_candidate = False
    if not arm["forced_action"] and first_action is None:
        first_action = False
    if arm["forced_action"] == "PUSH" and next_state is None:
        next_state = False
    return {"first_candidates_correct": first_candidate, "first_action_legal": first_action,
            "next_state_correct": next_state, "suffix_complete_valid": complete,
            "first_error": first_error, "premature_found": premature,
            "first_record_action": records[0][3] if records and records[0] else None,
            "generated_records": len(records) - int(bool(arm["forced_action"])),
            "immediate_answer": bool(out["generated"] and out["generated"][0] in (tok.yes, tok.no))}


def build_cases(am, examples, count):
    from cot_navigation.protocols.dfs import dfs_search
    from cot_navigation.protocols.bfs import bfs_search
    from cot_navigation.protocols.render import render_steps
    fifo = am.protocol == "bfs_state"
    search = bfs_search if fifo else dfs_search
    groups = defaultdict(list)
    wanted = ("target_position", "frontier_order", "frontier_branch", "legal_push", "candidates", "false_found", "false_exhaust")
    for ex in examples:
        steps = search(ex.adj, ex.source, ex.target)
        assert (steps[-1].action == "FOUND") == bool(ex.label)
        history = list(ex.prompt)
        slots = []
        for i, step in enumerate(steps):
            # Keep space for an intervention and a useful continuation; use
            # early states only when the reference midpoint exceeds budget.
            if len(history) + 3 * len(ex.nodes) + 16 < am.model.max_seq_len - 384:
                slots.append((i, step, list(history)))
            history += render_steps([step], ex.nodes, am.tok)
        slots.sort(key=lambda item: (abs(item[0] - len(steps) / 2), item[0]))
        for family in wanted:
            if len(groups[family]) >= count:
                continue
            selected = None
            for index, step, hist in slots:
                ff, vv = list(step.stack), list(step.visited)
                cc = list(step.cands)
                curidx = 0 if fifo else len(ff) - 1
                current = ff[curidx]
                nonterminal = step.action not in {"FOUND", "EXHAUST"}
                spare = [v for v in range(len(ex.nodes)) if v not in vv and v != ex.target]
                # Matched controls prefer the degree of the replaced/current node.
                spare.sort(key=lambda v: (abs(len(ex.adj[v]) - len(ex.adj[current])), v))
                dead = [v for v in spare if not graph_oracle(ex.adj, [v], [v], ex.target)]
                reachable = [v for v in spare if graph_oracle(ex.adj, [v], [v], ex.target)]
                if family == "target_position" and ex.label == 0 and nonterminal and ex.target not in vv and dead:
                    selected = (index, step, hist, dead[0])
                elif family == "frontier_order" and nonterminal and len(ff) >= 2:
                    selected = (index, step, hist, None)
                elif family == "frontier_branch" and ex.label == 0 and nonterminal and reachable and dead:
                    selected = (index, step, hist, (reachable[0], dead[0]))
                elif family in {"legal_push", "candidates"} and step.action == "PUSH" and len(cc) >= 2:
                    nonedge = [v for v in spare if v not in ex.adj[current] and v != current]
                    if family == "legal_push" or nonedge:
                        selected = (index, step, hist, nonedge[0] if nonedge else None)
                elif family == "false_found" and ex.label == 0 and nonterminal:
                    selected = (index, step, hist, None)
                elif family == "false_exhaust" and step.action == "FOUND":
                    selected = (index, step, hist, None)
                if selected:
                    break
            if selected is None:
                continue
            index, s, hist, extra = selected
            orig_ff, orig_vv, orig_cc = list(s.stack), list(s.visited), list(s.cands)
            curidx = 0 if fifo else len(orig_ff) - 1
            case = {"family": family, "query": ex.index, "original_label": ex.label,
                    "record": index, "history_legal": True, "arms": {}, "example": ex}

            def arm(name, ff=None, vv=None, cc=None, action=None, arg=None, cut="candidates"):
                ff = list(orig_ff if ff is None else ff)
                vv = list(orig_vv if vv is None else vv)
                cc = list(orig_cc if cc is None else cc)
                assert ff and len(ff) == len(set(ff)) and len(vv) == len(set(vv)) and set(ff) <= set(vv)
                oracle = machine(ex.adj, ff, vv, ex.target, fifo)
                assert oracle == graph_oracle(ex.adj, ff, vv, ex.target)
                prefix = list(hist)
                for values in (ff, vv):
                    prefix.extend(ex.nodes[v] for v in values)
                    prefix.append(am.tok.id("|"))
                if cut in {"action", "after_action"}:
                    prefix.extend(ex.nodes[v] for v in cc)
                    prefix.append(am.tok.id("|"))
                if cut == "after_action":
                    prefix.append(am.tok.id(f"<{action}>"))
                    if action == "PUSH":
                        prefix.append(ex.nodes[arg])
                    prefix.append(am.tok.id(";"))
                assert len(prefix) < am.model.max_seq_len - am.cfg.protocol.cot_margin - 4
                case["arms"][name] = {"prefix": prefix, "record_start": len(hist),
                    "frontier": [ex.nodes[v] for v in ff], "visited": [ex.nodes[v] for v in vv],
                    "oracle_answer": oracle, "forced_candidates": cut != "candidates",
                    "given_candidates": [ex.nodes[v] for v in cc] if cut != "candidates" else None,
                    "forced_action": action if cut == "after_action" else None,
                    "history_consistent": ff == orig_ff and vv == orig_vv and
                        (cut != "after_action" or action == s.action and (action != "PUSH" or arg in orig_cc))}

            if family == "target_position":
                arm("clean")
                for node, tag in ((ex.target, "target"), (extra, "ordinary")):
                    seen = orig_vv + [node]
                    at_current = ([node] + orig_ff) if fifo else (orig_ff + [node])
                    away = (orig_ff + [node]) if fifo else ([node] + orig_ff)
                    arm(tag + "_current", ff=at_current, vv=seen)
                    arm(tag + "_noncurrent", ff=away, vv=seen)
                # Same target-token insertion, but outside frontier: priming control.
                arm("target_vis_only", vv=orig_vv + [ex.target])
                assert case["arms"]["target_current"]["oracle_answer"] == 1
                assert case["arms"]["target_noncurrent"]["oracle_answer"] == 1
                assert case["arms"]["ordinary_current"]["oracle_answer"] == 0
            elif family == "frontier_order":
                arm("clean")
                ff = list(orig_ff)
                ff[0], ff[-1] = ff[-1], ff[0]
                arm("swap_current", ff=ff)
                if len(ff) >= 3:
                    ff = list(orig_ff)
                    other = [i for i in range(len(ff)) if i != curidx]
                    ff[other[0]], ff[other[1]] = ff[other[1]], ff[other[0]]
                    arm("swap_noncurrent", ff=ff)
            elif family == "frontier_branch":
                arm("clean")
                for node, tag in zip(extra, ("branch_reachable", "branch_dead")):
                    ff = list(orig_ff)
                    ff[curidx] = node
                    arm(tag, ff=ff, vv=orig_vv + [node])
                assert case["arms"]["branch_reachable"]["oracle_answer"] == 1
                assert case["arms"]["branch_dead"]["oracle_answer"] == 0
            elif family == "legal_push":
                arm("clean", action="PUSH", arg=orig_cc[0], cut="after_action")
                arm("push_alternative", action="PUSH", arg=orig_cc[1], cut="after_action")
                # Match the rendered priority order as an additional legal control.
                reordered = [orig_cc[1], orig_cc[0], *orig_cc[2:]]
                arm("push_alternative_reordered", cc=reordered, action="PUSH", arg=orig_cc[1], cut="after_action")
            elif family == "candidates":
                arm("clean", cut="action")
                arm("candidate_remove", cc=orig_cc[1:], cut="action")
                arm("candidate_nonedge", cc=[extra, *orig_cc[1:]], cut="action")
                arm("candidate_reorder", cc=[orig_cc[1], orig_cc[0], *orig_cc[2:]], cut="action")
            elif family == "false_found":
                arm("clean", action=s.action, arg=s.arg, cut="after_action")
                arm("force_found", action="FOUND", cut="after_action")
            elif family == "false_exhaust":
                arm("clean", action="FOUND", cut="after_action")
                arm("force_exhaust", action="EXHAUST", cut="after_action")
            for name, item in case["arms"].items():
                item["prefix_sha256"] = hashlib.sha256(json.dumps(item["prefix"]).encode()).hexdigest()
            groups[family].append(case)
    return [case for family in wanted for case in groups[family]]


def evaluate(am, cases, batch):
    from cot_navigation.analysis.scale_followup import generate
    # Deduplicate identical prefixes across experimental families; each case
    # still retains its own clean comparator. Batch by length for efficient prefill.
    prefixes = {}
    for case in cases:
        for arm in case["arms"].values():
            prefixes[tuple(arm["prefix"])] = None
    ordered = sorted(prefixes, key=lambda p: (len(p), p))
    started = time.time()
    for at in range(0, len(ordered), batch):
        chunk = ordered[at:at + batch]
        outputs = generate(am, [list(p) for p in chunk], batch=batch)
        for prefix, out in zip(chunk, outputs):
            prefixes[prefix] = out
        print(" BATCH", at + len(chunk), "/", len(ordered), round(time.time() - started, 1), flush=True)
    rows, saved = [], defaultdict(list)
    for case in cases:
        ex = case["example"]
        clean = prefixes[tuple(case["arms"]["clean"]["prefix"])]["pred"]
        for method, arm in case["arms"].items():
            out = prefixes[tuple(arm["prefix"])]; pred = out["pred"]
            row = {"family": case["family"], "method": method, "query": ex.index,
                   "record": case["record"], "original_label": ex.label,
                   "clean_pred": clean, "pred": pred, "oracle_answer": arm["oracle_answer"],
                   "binary_flip": clean is not None and pred is not None and clean != pred,
                   "valid_answer_pair": clean is not None and pred is not None,
                   "oracle_correct": pred == arm["oracle_answer"],
                   "history_consistent": arm["history_consistent"],
                   "interrupted": out["cot_interrupted"], "forced_stop": out["forced_stop"],
                   "generation_failure": pred is None, "generated_tokens": len(out["generated"]),
                   "prefix_length": len(arm["prefix"]), "prefix_sha256": arm["prefix_sha256"],
                   **check_continuation(am, ex, arm, out)}
            if row["first_record_action"]:
                action = row["first_record_action"]
                given = arm["given_candidates"]
                row["pushed_first_displayed"] = (action == [am.tok.id("<PUSH>"), given[0]]
                                                   if given else None)
                row["pushed_injected_nonedge"] = (action == [am.tok.id("<PUSH>"), given[0]]
                                                   if method == "candidate_nonedge" else None)
            rows.append(row)
        if len(saved[case["family"]]) < 2:
            saved[case["family"]].append({
                "query": ex.index, "record": case["record"], "original_label": ex.label,
                "arms": {name: {k: v for k, v in arm.items() if k != "prefix"} |
                         {"prefix": arm["prefix"], "prefix_text": am.tok.decode(arm["prefix"]),
                          "continuation": prefixes[tuple(arm["prefix"])]["generated"],
                          "continuation_text": am.tok.decode(prefixes[tuple(arm["prefix"])]["generated"])}
                         for name, arm in case["arms"].items()}})
    return rows, dict(saved), len(ordered)


def self_test():
    # Exhaustively check arbitrary coherent active states in a small diamond,
    # independently comparing state-machine termination and graph reachability.
    import itertools
    adj = [[1, 2], [3], [3], []]
    checked = 0
    for n in range(1, 5):
        for ff in itertools.permutations(range(4), n):
            spare = [v for v in range(4) if v not in ff]
            for mask in range(1 << len(spare)):
                vv = list(ff) + [v for i, v in enumerate(spare) if mask >> i & 1]
                for target in range(4):
                    for fifo in (False, True):
                        assert machine(adj, ff, vv, target, fifo) == graph_oracle(adj, ff, vv, target)
                        checked += 1
    assert parse_records([1, 9, 1, 9, 2, 9, 8, 2, 10], 9, 10)[0] == [[[1], [1], [2], [8, 2]]]
    print("SELF_TEST_PASS", checked, "oracle comparisons", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--cases", type=int, default=24)
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("studies/search_interventions/results"))
    args = parser.parse_args()
    self_test()
    if args.self_test:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["WANDB_MODE"] = "disabled"
    import torch
    from cot_navigation.analysis.scale_followup import make_adapter
    from cot_navigation.analysis.common import write_json, sha256
    torch.set_num_threads(3)
    root = Path.cwd()
    previous = root / "studies/scale_mechanistic/results"
    entries = json.loads((previous / "selection.json").read_text())["models"]
    for entry in entries:
        if entry["protocol"] not in {"dfs_state", "bfs_state"} or (args.models and entry["id"] not in args.models):
            continue
        dest = args.output / "models" / (entry["id"] + ".json")
        if dest.exists() and json.loads(dest.read_text()).get("complete"):
            continue
        started = time.time()
        print("MODEL", entry["id"], flush=True)
        am = make_adapter(entry, root, "cuda:0")
        assert am.checkpoint["sha256"] == entry["checkpoint_sha256"]
        old = json.loads((previous / "models" / (entry["id"] + ".json")).read_text())
        examples = [am.example(q, "test") for q in old["query_ids"]["test"]]
        cases = build_cases(am, examples, args.cases)
        print(" CASES", dict(Counter(c["family"] for c in cases)), flush=True)
        rows, samples, unique = evaluate(am, cases, args.batch)
        unchanged = sha256(am.run / "checkpoints/best.pt") == entry["checkpoint_sha256"]
        assert unchanged
        write_json(dest, {"entry": entry, "checkpoint": am.checkpoint,
            "checkpoint_unchanged": unchanged, "source_sha256": sha256(Path(__file__)),
            "query_ids": [ex.index for ex in examples], "cases_per_family_limit": args.cases,
            "case_counts": dict(Counter(c["family"] for c in cases)), "unique_prefixes": unique,
            "rows": rows, "examples": samples, "complete": True,
            "elapsed_seconds": time.time() - started})
        print("DONE", entry["id"], round(time.time() - started, 1), flush=True)
        del am
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

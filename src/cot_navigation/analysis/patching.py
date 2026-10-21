"""Paired token interventions with graph-preserving controls.

These are interventions on serialized CoT/input tokens, not activation
interchange. The graph/query label stays unchanged for state/path edits.
Impossible histories can be OOD, so a changed answer establishes sensitivity
to that prefix, not by itself a faithful internal implementation of DFS/BFS.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from types import MethodType
import numpy as np
import torch

from cot_navigation.evaluation.rollout import generate
from .common import SEED, records


def reachable(adj, source, target, blocked=()):
    blocked, seen, todo = set(blocked), set(), [source]
    while todo:
        u = todo.pop()
        if u in blocked or u in seen:
            continue
        if u == target:
            return True
        seen.add(u)
        todo.extend(adj[u])
    return False


def generate_prefixes(am, prefixes, *, direct=False, batch_size=16, max_new=1024):
    result = []
    for begin in range(0, len(prefixes), batch_size):
        result.extend(generate(am.model, prefixes[begin:begin+batch_size],
                               "direct" if direct else "path", am.tok, "bf16",
                               max_new_tokens=max_new))
    return result


def free_rollouts(am, examples, batch_size=16, max_new=1024):
    results = []
    for begin in range(0, len(examples), batch_size):
        chunk = examples[begin:begin+batch_size]
        results.extend(generate(am.model, [ex.prompt for ex in chunk], am.protocol,
                                am.tok, "bf16", filler_tokens=am.filler,
                                max_new_tokens=max_new))
    prefixes = {}
    rows = []
    for ex, out in zip(examples, results):
        tokens = out["tokens"]
        end = next((i for i in range(len(ex.prompt), len(tokens))
                    if tokens[i] in {am.tok.yes, am.tok.no, am.tok.id("<EOS>")}), len(tokens))
        prefixes[ex.index] = tokens[:end]
        rows.append({"query": ex.index, "label": ex.label, "pred": out["pred"],
                     "correct": out["pred"] == ex.label, "tokens": len(tokens)-len(ex.prompt),
                     "limit": out["generation_limit_reached"]})
    return prefixes, rows


def state_case(am, ex, tokens):
    """Add one future goal-route node to VIS; matched dead-route node control.

    Cut BEFORE candidate/action fields. Regenerate the rest instead of keeping
    stale oracle candidates. Both insertions add exactly one node token.
    """
    if ex.label != 1:
        return None
    inv = {v: i for i, v in enumerate(ex.nodes)}
    all_records = records(tokens, ex.row.prompt_len, am.tok)
    # Prefer a genuine middle state but fall back deterministically.
    ordering = sorted(range(len(all_records)), key=lambda i: abs(i-len(all_records)/2))
    for index in ordering:
        r = all_records[index]
        if not all(v in inv for v in r["frontier"]+r["visited"]):
            continue
        current = r["frontier"][0 if am.protocol == "bfs_state" else -1]
        if current == ex.nodes[ex.target] or r["action"][0] in {am.tok.id("<FOUND>"), am.tok.id("<EXHAUST>")}:
            continue
        excluded = set(r["visited"]+r["frontier"]+[ex.nodes[ex.target]])
        route = [v for v in ex.nodes if v not in excluded and reachable(ex.adj, inv[current], inv[v])
                 and reachable(ex.adj, inv[v], ex.target)]
        dead = [v for v in ex.nodes if v not in excluded and reachable(ex.adj, inv[current], inv[v])
                and not reachable(ex.adj, inv[v], ex.target)]
        if not route or not dead:
            continue
        # Same out-degree where possible to avoid the trivial sink cue.
        pairs = [(u, v) for u in route for v in dead]
        critical, control = min(pairs, key=lambda p: (abs(len(ex.adj[inv[p[0]]])-len(ex.adj[inv[p[1]]])), p))
        cut = r["separators"][1]
        clean = tokens[:cut+1]
        attacked = tokens[:cut]+[critical]+tokens[cut:cut+1]
        matched = tokens[:cut]+[control]+tokens[cut:cut+1]
        # A same-length, same-membership reorder controls serialization changes.
        a = r["separators"][0]
        reorder = tokens[:a+1]+list(reversed(tokens[a+1:cut]))+tokens[cut:cut+1]
        assert len(attacked) == len(matched) == len(clean)+1
        assert Counter(reorder) == Counter(clean)
        return {"query": ex.index, "label": ex.label, "prefix_source": "generated",
                "prompt_length": ex.row.prompt_len,
                "prefixes": {"clean": clean, "visited_route": attacked,
                             "visited_irrelevant": matched, "visited_reorder": reorder},
                "detail": {"record": index, "current": current, "critical": critical,
                           "control": control, "degree_matched": len(ex.adj[inv[critical]]) == len(ex.adj[inv[control]]),
                           "visited": r["visited"], "critical_is_current_candidate": critical in r["candidates"]}}
    return None


def node_case(am, ex, tokens):
    p = ex.row.prompt_len
    positions = [i for i in range(p, len(tokens)) if tokens[i] in ex.nodes
                 and tokens[i] not in {ex.nodes[ex.source], ex.nodes[ex.target]}]
    if not positions:
        return None
    at = positions[len(positions)//2]
    inv = {v: i for i, v in enumerate(ex.nodes)}
    original = tokens[at]
    same = [v for v in ex.nodes if v != original and v not in {ex.nodes[ex.source], ex.nodes[ex.target]}
            and reachable(ex.adj, inv[v], ex.target) == reachable(ex.adj, inv[original], ex.target)]
    opposite = [v for v in ex.nodes if v not in {ex.nodes[ex.source], ex.nodes[ex.target]}
                and reachable(ex.adj, inv[v], ex.target) != reachable(ex.adj, inv[original], ex.target)]
    if not same or not opposite:
        return None
    prefix = tokens[:at+1]
    return {"query": ex.index, "label": ex.label, "prefix_source": "generated",
            "prefixes": {"clean": prefix, "node_opposite_route": prefix[:-1]+[opposite[0]],
                         "node_same_route": prefix[:-1]+[same[0]]},
            "detail": {"position": at, "original": original, "replacement": opposite[0]}}


def input_case(am, ex):
    """Direct has no CoT: intervene on QUERY target with same-label controls."""
    p, original = ex.prompt, ex.target
    opposite, same = [], []
    shape = am.cfg.data.graph_shape or next(iter(am.cfg.data.shapes))
    width = am.cfg.data.shapes[shape].width
    for target in range(len(ex.nodes)):
        if target in {original, ex.source} or target//width != original//width:
            continue
        y = int(reachable(ex.adj, ex.source, target))
        (same if y == ex.label else opposite).append((target, y))
    if not opposite or not same:
        return None
    variants = {"clean": p}
    expected = {"clean": ex.label}
    for name, (target, label) in [("target_flip", opposite[0]), ("target_same_label", same[0])]:
        ids = p.copy()
        ids[-2] = ex.nodes[target]
        variants[name], expected[name] = ids, label
    return {"query": ex.index, "label": ex.label, "prefix_source": "input",
            "prefixes": variants, "expected_labels": expected, "detail": {"target": ex.nodes[original]}}


@contextmanager
def dot_attention_ablation(am, prompt_length, mode):
    """Preserve every input token/position; block only selected DOT read edges."""
    import torch.nn.functional as F
    originals = []
    for block in am.model.blocks:
        attn = block.attn
        original = attn._attend
        originals.append((attn, original))
        def attend(self, q, k, v, keep, force_math, causal=False):
            if q.shape[-2] != k.shape[-2]:
                raise ValueError("DOT ablation is a full-prefill intervention")
            positions = torch.arange(q.shape[-2], device=q.device)
            qi, kj = positions[:, None], positions[None, :]
            mask = kj <= qi
            forbidden = ((qi >= prompt_length) & (kj >= prompt_length) & (kj < qi)
                         if mode == "chain" else (qi >= prompt_length) & (kj < prompt_length))
            mask = (mask & ~forbidden)[None, None]
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask), None
        attn._attend = MethodType(attend, attn)
    try:
        yield
    finally:
        for attn, original in originals:
            attn._attend = original


def summarize_pairs(cases, results, tok=None):
    groups = {}
    examples = []
    for case, scores in zip(cases, results):
        base = scores["clean"]
        expected = case.get("expected_labels", {})
        for name, out in scores.items():
            row = groups.setdefault(name, [])
            label = expected.get(name, case["label"])
            item = {"correct": out["pred"] == label, "flip": out["pred"] != base["pred"],
                    "base_correct": base["pred"] == case["label"],
                    "failure": out["pred"] is None, "limit": out["generation_limit_reached"]}
            if tok is not None and name in {"visited_route", "visited_irrelevant"}:
                inserted = case["detail"]["critical" if name == "visited_route" else "control"]
                future = [r for r in records(out["tokens"], case["prompt_length"], tok)
                          if r["start"] >= len(case["prefixes"][name])]
                item["retained"] = inserted in future[0]["visited"] if future else None
                item["pushed_inserted"] = any(a == tok.id("<PUSH>") and b == inserted
                                              for a, b in zip(out["generated"], out["generated"][1:]))
            row.append(item)
        if len(examples) < 2 or (len(examples) < 8 and any(s["pred"] != base["pred"] for s in scores.values())):
            saved_case = {k: v for k, v in case.items() if k != "prefixes"} | {
                "prefix_tokens": case["prefixes"],
                "outcomes": {k: {"pred": v["pred"], "generated": v["generated"][:120],
                                 "limit": v["generation_limit_reached"]} for k, v in scores.items()}}
            if tok is not None:
                saved_case["prefix_text"] = {k: tok.decode(ids) for k, ids in case["prefixes"].items()}
                for outcome in saved_case["outcomes"].values():
                    outcome["generated_text"] = tok.decode(outcome["generated"])
                for key in ["current", "critical", "control", "original", "replacement"]:
                    if key in saved_case["detail"]:
                        saved_case["detail"][key+"_name"] = tok.name(saved_case["detail"][key])
            examples.append(saved_case)
    summary = {}
    rng = np.random.default_rng(SEED)
    for name, rows in groups.items():
        n = len(rows)
        values = np.asarray([r["correct"] for r in rows], float)
        base_good = [r for r in rows if r["base_correct"]]
        delta = values-np.asarray([r["base_correct"] for r in rows], float)
        boots = delta[rng.integers(n, size=(2000, n))].mean(-1)
        summary[name] = {"n": n, "accuracy": float(values.mean()),
                         "answer_flip_rate": sum(r["flip"] for r in rows)/n,
                         "failure_rate": sum(r["failure"] for r in rows)/n,
                         "limit_rate": sum(r["limit"] for r in rows)/n,
                         "accuracy_delta_from_clean": float(delta.mean()),
                         "paired_bootstrap_delta_ci95": np.quantile(boots, [.025, .975]).tolist(),
                         "clean_correct_n": len(base_good),
                         "damage_on_clean_correct": (sum(not r["correct"] for r in base_good)/len(base_good)
                                                     if base_good else None)}
        if any("retained" in r for r in rows):
            known = [r["retained"] for r in rows if r.get("retained") is not None]
            summary[name]["next_record_available_n"] = len(known)
            summary[name]["inserted_node_retained_next_record"] = float(np.mean(known)) if known else None
            summary[name]["inserted_node_pushed_after_edit_rate"] = float(np.mean([r["pushed_inserted"] for r in rows]))
    contrasts = {}
    for attack, control in [("visited_route", "visited_irrelevant"), ("node_opposite_route", "node_same_route")]:
        if attack in groups and control in groups:
            delta = np.asarray([a["correct"]-b["correct"] for a, b in zip(groups[attack], groups[control])], float)
            boot = delta[rng.integers(len(delta), size=(2000, len(delta)))].mean(-1)
            contrasts[f"{attack}_minus_{control}"] = {"n": len(delta), "accuracy_difference": float(delta.mean()),
                                                      "paired_ci95": np.quantile(boot, [.025, .975]).tolist()}
    return {"conditions": summary, "paired_contrasts": contrasts, "interesting_examples": examples}


def run_patching(am, examples, free_sequences, *, max_cases=40, max_new=1024, original_predictions=None):
    cases = []
    for ex in examples:
        if am.protocol == "direct":
            case = input_case(am, ex)
        elif am.protocol == "filler":
            p, dots = ex.prompt, [am.tok.id("<DOT>")]*am.filler
            case = {"query": ex.index, "label": ex.label, "prefix_source": "input",
                    "prefixes": {"clean": p+dots, "half_dots": p+dots[:am.filler//2], "no_dots": p},
                    "detail": {"note": "changes positions and compute; no semantic visited field"}}
        elif am.protocol in {"dfs_state", "bfs_state"}:
            case = state_case(am, ex, free_sequences[ex.index])
        else:
            case = node_case(am, ex, free_sequences[ex.index])
        if case is not None:
            cases.append(case)
        if len(cases) >= max_cases:
            break
    if not cases:
        return {"status": "no eligible paired interventions", "n_candidates": len(examples)}
    flat, keys = [], []
    for i, case in enumerate(cases):
        for name, prefix in case["prefixes"].items():
            flat.append(prefix)
            keys.append((i, name))
    outs = generate_prefixes(am, flat, direct=am.protocol in {"direct", "filler"}, max_new=max_new)
    grouped = [{} for _ in cases]
    for (index, name), out in zip(keys, outs):
        grouped[index][name] = out
    if am.protocol == "filler":
        # Dot lengths are fixed within an experiment, so batch by prompt length.
        # Query shape is constant in our two tasks; assert rather than silently
        # applying the wrong mask to heterogeneous prompts.
        lengths = {len(c["prefixes"]["clean"])-am.filler for c in cases}
        if len(lengths) != 1:
            raise ValueError("Group variable-size Filler prompts by length before ablating")
        for mode in ["chain", "prompt"]:
            with dot_attention_ablation(am, next(iter(lengths)), mode):
                outputs = generate_prefixes(am, [c["prefixes"]["clean"] for c in cases], direct=True)
            for i, output in enumerate(outputs):
                grouped[i][f"block_dot_{mode}"] = output
    replay = {}
    if original_predictions is not None:
        replay = {"clean_replay_agreement": float(np.mean([scores['clean']['pred'] == original_predictions[c['query']]
                                                           for c, scores in zip(cases, grouped)])),
                  "original_accuracy_on_cases": float(np.mean([original_predictions[c['query']] == c['label'] for c in cases])),
                  "replay_unstable_query_ids": [c['query'] for c, scores in zip(cases, grouped)
                                                 if scores['clean']['pred'] != original_predictions[c['query']]]}
    return {"status": "complete", "n_candidates": len(examples),
            "n_cases": len(cases), "query_ids": [c["query"] for c in cases],
            "degree_matched_fraction": (float(np.mean([c["detail"]["degree_matched"] for c in cases]))
                                        if "degree_matched" in cases[0]["detail"] else None),
            **replay, **summarize_pairs(cases, grouped, am.tok)}

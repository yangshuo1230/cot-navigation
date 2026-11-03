"""Delayed readout of a past CAND error, with shared legal continuation.

Frozen navigation models. No activation/probe checkpoint files or W&B runs.
One event per original query; query-disjoint train/validation/test splits.
Context-only graph controls hold the full CoT text identical across labels.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import time

ANCHORS = ["pre_error", "error", "token_1", "token_4", "token_16", "token_64",
           "record_1_vis", "record_2_vis", "record_4_vis"]
HEADS = ["L1", "L2", "L3", "L4", "L5", "L6", "current_token", "edited_and_current_node"]


def reachable(adj, source, target):
    seen, todo = {source}, [source]
    for u in todo:
        if u == target:
            return True
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                todo.append(v)
    return False


def token_legality(ex, protocol, tok, sequence):
    from cot_navigation.analysis.error_detection import semantic_slots
    body = sequence[len(ex.prompt):]
    slots = list(semantic_slots(ex, protocol, tok, body))
    for i, allowed, _ in slots:
        if body[i] not in allowed:
            return len(ex.prompt) + i
    return None if len(slots) == len(body) else len(ex.prompt) + len(slots)


def edit_prompt(ex, source, old, new):
    """One node-token edit in an EDGE triple; keep edge count/order fixed."""
    prompt = list(ex.prompt)
    matches = [i + 2 for i in range(0, len(prompt) - 4, 3)
               if prompt[i + 1] == ex.nodes[source] and prompt[i + 2] == ex.nodes[old]]
    assert len(matches) == 1
    prompt[matches[0]] = ex.nodes[new]
    return prompt


def changed_adj(adj, source, old, new):
    result = [list(vs) for vs in adj]
    assert old in result[source] and new not in result[source]
    result[source][result[source].index(old)] = new
    return result


def graph_pair(am, ex, event, width):
    """Both labels contain one matched-size graph edit and identical CoT.

    Legal graph replaces the relevant edge. Wrong graph edits an unused
    source, preserving the original bad CAND. Stop at next VIS separator,
    before the changed edge could require another CAND disagreement.
    """
    import numpy as np
    u, old, new = event["current_index"], event["old_index"], event["new_index"]
    legal_adj = changed_adj(ex.adj, u, old, new)
    if reachable(legal_adj, ex.source, ex.target) != bool(ex.label):
        return None
    stop = event["positions"]["record_1_vis"]
    used = set(event["expanded_through_next_vis"])
    rng = np.random.default_rng(2026102209 + ex.index)
    alternatives = []
    for source, vs in enumerate(ex.adj):
        if source in used or source == u:
            continue
        for dest in vs:
            for replacement in range((dest // width) * width, min(len(ex.nodes), (dest // width + 1) * width)):
                if replacement != dest and replacement not in vs:
                    alternatives.append((source, dest, replacement))
    rng.shuffle(alternatives)
    for source, dest, replacement in alternatives:
        bad_adj = changed_adj(ex.adj, source, dest, replacement)
        if reachable(bad_adj, ex.source, ex.target) != bool(ex.label):
            continue
        shared = event["wrong"][len(ex.prompt):stop + 1]
        legal = edit_prompt(ex, u, old, new) + shared
        wrong = edit_prompt(ex, source, dest, replacement) + shared
        legal_ex, wrong_ex = replace(ex, adj=legal_adj), replace(ex, adj=bad_adj)
        assert token_legality(legal_ex, am.protocol, am.tok, legal) is None
        assert token_legality(wrong_ex, am.protocol, am.tok, wrong) == event["error_position"]
        assert legal[len(ex.prompt):] == wrong[len(ex.prompt):]
        positions = {k: min(v, stop) for k, v in event["positions"].items()}
        valid = {k: v <= stop for k, v in event["positions"].items()}
        return {**event, "correct": legal, "wrong": wrong, "positions": positions,
                "valid_anchors": valid, "neutral_edit": [source, dest, replacement],
                "related_edit": [u, old, new], "graph_control": True}
    return None


def build_events(am, split, target, scan):
    import numpy as np
    from cot_navigation.protocols.dfs import dfs_search
    from cot_navigation.protocols.bfs import bfs_search
    from cot_navigation.protocols.render import render_steps
    shape = am.cfg.data.shapes[am.cfg.data.graph_shape or "main"]
    width = int(shape.width if hasattr(shape, "width") else shape["width"])
    search = bfs_search if am.protocol == "bfs_state" else dfs_search
    events, graph_events, inventory = [], [], Counter()
    ids = am.sample_ids(split, scan)
    for query in ids:
        ex = am.example(query, split)
        inventory["queries_scanned"] += 1
        steps = search(ex.adj, ex.source, ex.target)
        starts, rendered, used = [], [], len(ex.prompt)
        for step in steps:
            body = render_steps([step], ex.nodes, am.tok)
            starts.append(used)
            rendered.append(body)
            used += len(body)
        sequence = ex.prompt + [v for body in rendered for v in body]
        terminal_at = starts[-1] + len(steps[-1].stack) + len(steps[-1].visited) + len(steps[-1].cands) + 3
        assert sequence[terminal_at] == am.tok.id(f"<{steps[-1].action}>")
        candidates, expanded = [], set()
        rng = np.random.default_rng(2026102201 + query)
        for i, step in enumerate(steps):
            first_expansion = step.cur not in expanded
            expanded.add(step.cur)
            if not first_expansion or step.action != "PUSH" or len(step.cands) < 2 or i + 4 >= len(steps) - 1:
                continue
            old = step.cands[-1]  # never alter the candidate actually PUSHed
            choices = [v for v in range((old // width) * width, min(len(ex.nodes), (old // width + 1) * width))
                       if v not in ex.adj[step.cur] and v not in step.visited and v != ex.target]
            if not choices:
                continue
            position = starts[i] + len(step.stack) + len(step.visited) + 2 + len(step.cands) - 1
            assert sequence[position] == ex.nodes[old]
            positions = {"pre_error": position - 1, "error": position,
                         **{f"token_{k}": position + k for k in (1, 4, 16, 64)},
                         **{f"record_{k}_vis": starts[i + k] + len(steps[i + k].stack) + len(steps[i + k].visited) + 1 for k in (1, 2, 4)}}
            last = max(positions.values())
            if last >= min(terminal_at, am.model.max_seq_len - 1):
                continue
            candidates.append((i, step, int(rng.choice(choices)), old, position, positions, last))
        if not candidates:
            inventory["no_eligible_event"] += 1
            continue
        i, step, new, old, position, positions, last = candidates[int(rng.integers(len(candidates)))]
        correct = sequence[:last + 1]
        wrong = list(correct)
        wrong[position] = ex.nodes[new]
        assert len(correct) == len(wrong) and [j for j, (a, b) in enumerate(zip(correct, wrong)) if a != b] == [position]
        assert step.arg == step.cands[0] and step.arg != old
        assert all(correct[p] == wrong[p] for key, p in positions.items() if key != "error")
        assert am.tok.yes not in correct and am.tok.no not in correct
        assert am.tok.id("<FOUND>") not in correct and am.tok.id("<EXHAUST>") not in correct
        assert token_legality(ex, am.protocol, am.tok, correct) is None
        assert token_legality(ex, am.protocol, am.tok, wrong) == position
        event = {"query": query, "original_label": ex.label, "correct": correct, "wrong": wrong,
                 "error_position": position, "positions": positions, "valid_anchors": dict.fromkeys(ANCHORS, True),
                 "current_token": ex.nodes[step.cur], "current_index": step.cur,
                 "old_index": old, "new_index": new, "prompt_length": len(ex.prompt),
                 "record": i, "graph_control": False,
                 "expanded_through_next_vis": sorted({s.cur for s in steps[:i + 2]})}
        events.append(event)
        if am.context:
            control = graph_pair(am, ex, event, width)
            if control:
                graph_events.append(control)
        if len(events) >= target:
            break
    inventory["pairs"] = len(events)
    inventory["graph_pairs"] = len(graph_events)
    inventory["yes_queries"] = sum(e["original_label"] for e in events)
    return events, graph_events, dict(inventory)


def extract(am, events, token_budget):
    import numpy as np
    import torch
    if not events:
        return None
    rows = [(eid, label, e, e[key]) for eid, e in enumerate(events) for label, key in ((0, "correct"), (1, "wrong"))]
    rows.sort(key=lambda r: (len(r[3]), r[0], r[1]))
    n, dim = len(rows), am.model.d_model if hasattr(am.model, "d_model") else am.cfg.model.d_model
    # All heads share dimensions; residual-only heads are padded with zeros.
    features = np.zeros((n, len(ANCHORS) * len(HEADS), 2 * dim + 4), np.float32)
    y, query, event_id = np.zeros(n, np.float32), np.zeros(n, np.int64), np.zeros(n, np.int64)
    valid = np.zeros((n, len(ANCHORS)), bool)
    embedding_table = am.model.embed.weight.detach().float().cpu().numpy()
    start = 0
    while start < n:
        # Never split a pair across batches: BF16 kernel/batch differences can
        # otherwise create spurious label information even before the edit.
        end = start + 2
        while end + 2 <= n and (end - start + 2) * len(rows[end + 1][3]) <= token_budget and end - start + 2 <= 16:
            end += 2
        chunk = rows[start:end]
        assert all(chunk[i][0] == chunk[i + 1][0] and chunk[i][1] == 0 and chunk[i + 1][1] == 1
                   for i in range(0, len(chunk), 2))
        with torch.inference_mode():
            if getattr(am, "analysis_precision", "bf16") == "fp32":
                lengths = [len(r[3]) for r in chunk]
                tokens = torch.full((len(chunk), max(lengths)), am.tok.pad, dtype=torch.long, device=am.inference_device)
                for b, r in enumerate(chunk):
                    tokens[b, :len(r[3])] = torch.tensor(r[3], dtype=torch.long, device=am.inference_device)
                capture = {}
                am.model(tokens, capture=capture)
                residual = capture["resid"]
            else:
                _, residual, _ = am.forward([r[3] for r in chunk])
            assert len(residual) == 6
            positions_tensor = torch.tensor([[r[2]["positions"][a] for a in ANCHORS] for r in chunk], device=am.inference_device)
            row_tensor = torch.arange(len(chunk), device=am.inference_device)[:, None]
            gathered = torch.stack([r[row_tensor, positions_tensor].float() for r in residual], dim=2).cpu().numpy()
            for b, (eid, label, e, seq) in enumerate(chunk):
                y[start + b], query[start + b], event_id[start + b] = label, e["query"], eid
                for a, anchor in enumerate(ANCHORS):
                    position = e["positions"][anchor]
                    valid[start + b, a] = e["valid_anchors"][anchor]
                    base = a * len(HEADS)
                    values = gathered[b, a]
                    features[start + b, base:base + 6, :dim] = values
                    current = embedding_table[seq[position]]
                    edited = embedding_table[seq[e["error_position"]]] if position >= e["error_position"] else np.zeros(dim, np.float32)
                    node = embedding_table[e["current_token"]]
                    meta = [position / 4096, e["error_position"] / 4096,
                            (position - e["error_position"]) / 4096, e["prompt_length"] / 4096]
                    features[start + b, base + 6, :dim] = current
                    features[start + b, base + 6, -4:] = meta
                    features[start + b, base + 7, :dim] = edited
                    features[start + b, base + 7, dim:2 * dim] = node
                    features[start + b, base + 7, -4:] = meta
            del residual
        start = end
        if start % 256 < 16 or start == n:
            print(" EXTRACT", start, "/", n, flush=True)
    data = {"x": features, "y": y, "query": query, "event": event_id,
            "kind": np.array(["past_candidate_error"] * n), "field": np.array(["candidates"] * n), "valid": valid}
    if not events[0]["graph_control"]:
        # The causal model must not distinguish identical prefixes before the edit.
        pairs = {eid: {} for eid in range(len(events))}
        for i, eid in enumerate(event_id):
            pairs[int(eid)][int(y[i])] = i
        maximum = max(float(np.max(np.abs(features[p[0], :6] - features[p[1], :6]))) for p in pairs.values())
        assert maximum < 1e-4, maximum
        data["pre_error_max_difference"] = maximum
    return data


def metrics(data, logits, metadata):
    import numpy as np
    from cot_navigation.analysis.set_metrics import auc_ap
    results = []
    for head, (anchor, layer) in enumerate(metadata):
        a = ANCHORS.index(anchor)
        mask = data["valid"][:, a]
        labels, scores, events = data["y"][mask].astype(bool), logits[mask, head], data["event"][mask]
        if not len(labels):
            continue
        pairs = {}
        for label, score, eid in zip(labels, scores, events):
            pairs.setdefault(int(eid), {})[int(label)] = float(score)
        ranking = [(v[1] > v[0]) + .5 * (v[1] == v[0]) for v in pairs.values()]
        results.append({"anchor": anchor, "layer": layer, "n": len(labels), "pairs": len(pairs),
                        "queries": len(set(data["query"][mask].tolist())), **auc_ap(labels, scores),
                        "paired_rank_accuracy": float(np.mean(ranking)),
                        "balanced_accuracy": float(((scores[labels] >= 0).mean() + (scores[~labels] < 0).mean()) / 2)})
    return results


def fit_all(datasets, graph_sets, steps, seeds):
    import numpy as np
    from cot_navigation.analysis.error_awareness import train_probe, predict
    output = []
    all_meta = [(a, layer) for a in ANCHORS for layer in HEADS]
    for hidden, name in ((0, "linear"), (64, "mlp64")):
        for seed in seeds:
            print(" PROBE", name, seed, flush=True)
            fitted = train_probe(datasets["train"], datasets["val"], hidden, seed, steps, 512, "cuda:0")
            item = {"architecture": name, "seed": seed,
                    "validation_selected_steps": fitted[3]["selected_steps"],
                    "test": metrics(datasets["test"], predict(fitted, datasets["test"]), all_meta)}
            if graph_sets and graph_sets["test"] is not None:
                item["graph_zero_shot"] = metrics(graph_sets["test"], predict(fitted, graph_sets["test"]), all_meta)
            output.append(item)
            del fitted
    # Independent graph-only training at the first subsequent VIS boundary.
    # Both classes have identical CoT text, including the edited token.
    if graph_sets and all(graph_sets.get(split) is not None and len(set(graph_sets[split]["query"].tolist())) >= 20
                          for split in ("train", "val", "test")):
        begin = ANCHORS.index("record_1_vis") * len(HEADS)
        compact = {split: {**data, "x": np.ascontiguousarray(data["x"][:, begin:begin + len(HEADS)])}
                   for split, data in graph_sets.items()}
        graph_meta = [("record_1_vis", h) for h in HEADS]
        for hidden, name in ((0, "linear"), (64, "mlp64")):
            for seed in seeds:
                print(" GRAPH PROBE", name, seed, flush=True)
                fitted = train_probe(compact["train"], compact["val"], hidden, seed, steps, 256, "cuda:0")
                output.append({"architecture": name, "seed": seed, "graph_trained": True,
                               "validation_selected_steps": fitted[3]["selected_steps"],
                               "test": metrics(compact["test"], predict(fitted, compact["test"]), graph_meta)})
                del fitted
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--models", nargs="+", default=["context_l10_dfs_state", "context_l10_bfs_state", "fixed_l10_dfs_state", "fixed_l10_bfs_state"])
    parser.add_argument("--train-pairs", type=int, default=768)
    parser.add_argument("--val-pairs", type=int, default=192)
    parser.add_argument("--test-pairs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--token-budget", type=int, default=8192)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--output", default="studies/history_error/results")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["WANDB_MODE"] = "disabled"
    import torch
    from cot_navigation.analysis.scale_followup import make_adapter
    from cot_navigation.analysis.common import sha256, write_json
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path.cwd()
    output = root / args.output
    entries = json.loads((root / "studies/scale_mechanistic/results/selection.json").read_text())["models"]
    for entry in entries:
        if entry["id"] not in args.models:
            continue
        path = output / "models" / (entry["id"] + ".json")
        if path.exists() and json.loads(path.read_text()).get("complete"):
            continue
        print("MODEL", entry["id"], flush=True)
        started = time.time()
        am = make_adapter(entry, root, "cuda:0")
        am.analysis_precision = args.precision
        assert am.checkpoint["sha256"] == entry["checkpoint_sha256"]
        datasets, graph_sets, summaries, samples = {}, {}, {}, {}
        split_queries = []
        for split, target in (("train", args.train_pairs), ("val", args.val_pairs), ("test", args.test_pairs)):
            print(" BUILD", split, target, flush=True)
            events, graph_events, inventory = build_events(am, split, target, max(target * 4, 512))
            assert len(events) >= 30, (entry["id"], split, inventory)
            split_queries.append({e["query"] for e in events})
            print(" DATA", split, inventory, flush=True)
            datasets[split] = extract(am, events, args.token_budget)
            graph_sets[split] = extract(am, graph_events, args.token_budget) if graph_events else None
            summaries[split] = {**inventory,
                "query_ids": [e["query"] for e in events], "graph_query_ids": [e["query"] for e in graph_events],
                "pre_error_max_difference": datasets[split]["pre_error_max_difference"],
                "mean_error_position": sum(e["error_position"] for e in events) / len(events),
                "mean_readout_delays": {a: sum(e["positions"][a] - e["error_position"] for e in events) / len(events) for a in ANCHORS}}
            if split == "test":
                samples["main"] = events[:2]
                samples["graph"] = graph_events[:2]
        assert not (split_queries[0] & split_queries[1] or split_queries[0] & split_queries[2] or split_queries[1] & split_queries[2])
        fitted = fit_all(datasets, graph_sets, args.steps, args.seeds)
        unchanged = sha256(am.run / "checkpoints/best.pt") == entry["checkpoint_sha256"]
        assert unchanged
        write_json(path, {"entry": entry, "settings": vars(args), "source_sha256": sha256(Path(__file__)),
            "checkpoint_unchanged": unchanged, "dataset": summaries, "probes": fitted,
            "examples": samples, "complete": True, "elapsed_seconds": time.time() - started,
            "anchors": ANCHORS, "heads": HEADS})
        print("DONE", entry["id"], round(time.time() - started, 1), flush=True)
        del am, datasets, graph_sets
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

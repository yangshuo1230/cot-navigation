#!/usr/bin/env python3
"""Simple-graph node models on hard DAGs, against hard-DAG-trained models.

Free generation and teacher forcing use the same 256 stratified test queries
(seed 1729). The gold node order is the search on the edge list in the prompt.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import plot_dfs_cot_attention as base  # noqa: E402
from cot_navigation.data.dataset import ContextStore  # noqa: E402
from cot_navigation.evaluation.rollout import generate  # noqa: E402
from cot_navigation.evaluation.sampling import stratified_ids  # noqa: E402
from cot_navigation.protocols.bfs import first_expansion_order  # noqa: E402

OUT = ROOT / "studies/nodes_transfer_20261026.json"
DEVICE = torch.device("cuda:0")
N = 256
SEED = 1729
BATCH = 16
MAX_NEW = 256

HARD = [
    ("4×4", ROOT / "data/hard_dag_v1/context_r4b4_n16_m32_train50000_s0"),
    ("6×6", ROOT / "data/hard_dag_v1/context_r6b6_n36_m72_train50000_s0"),
    ("10×10", ROOT / "data/hard_dag_v1/context_r10b10_n100_m200_train50000_s0"),
]
SIMPLE = [
    ("4×4", ROOT / "data/scale_v1/context_l4w4_n16_m24_train50000_s0"),
    ("6×6", ROOT / "data/scale_v1/context_l6w6_n36_m54_train50000_s0"),
    ("10×10", ROOT / "data/scale_v1/context_l10w10_n100_m150_train50000_s0"),
]
RUNS = {
    "dfs_nodes": {
        "simple": ROOT / "outputs/scale_v1/launches/scale_v1_s0/context/curriculum/dfs_nodes",
        "hard": ROOT / "outputs/hard_dag_v1/launches/hard_dag_v1_s0/context/curriculum/dfs_nodes",
    },
    "bfs_nodes": {
        "simple": ROOT / "outputs/scale_v1/launches/scale_v1_s0/context/curriculum/bfs_nodes",
        "hard": ROOT / "outputs/hard_dag_v1/launches/hard_dag_v1_s0/context/curriculum/bfs_nodes",
    },
}


def rate(hits, total):
    return None if total == 0 else hits / total


def pack_answer(rows):
    yes = [row for row in rows if row["label"] == 1]
    no = [row for row in rows if row["label"] == 0]
    yes_hit = sum(row["answer"] for row in yes)
    no_hit = sum(row["answer"] for row in no)
    return {
        "n": len(rows),
        "answer": rate(yes_hit + no_hit, len(rows)),
        "yes_recall": rate(yes_hit, len(yes)),
        "no_recall": rate(no_hit, len(no)),
        "balanced": None if not yes or not no else 0.5 * (yes_hit / len(yes) + no_hit / len(no)),
        "trace_exact": rate(sum(row["trace"] for row in rows), len(rows)),
        "format_fail": rate(sum(row["format_fail"] for row in rows), len(rows)),
        "mean_prefix_fraction": float(np.mean([row["prefix_fraction"] for row in rows])),
    }


def examples(store, tok, protocol, limit):
    ids = stratified_ids(
        store.indices("test"), store.arrays["shape_id"], store.arrays["distance"],
        store.arrays["label"], limit, SEED,
    )
    rows = []
    for index in ids:
        rendered = store.render(int(index), protocol, tok)
        steps = rendered.meta["reference_steps"]
        parent = {}
        for step in steps:
            if step.action == "PUSH" and step.arg not in parent:
                parent[step.arg] = step.cur
        order = first_expansion_order(steps)
        body = rendered.tokens[rendered.prompt_len:rendered.answer_index]
        kinds = []
        for k, node in enumerate(order):
            if k == 0:
                kinds.append("start")
            elif parent.get(node) == order[k - 1]:
                kinds.append("continue")
            else:
                kinds.append("return")
        rows.append({
            "tokens": rendered.tokens,
            "prompt_len": rendered.prompt_len,
            "answer_index": rendered.answer_index,
            "label": rendered.label,
            "body": body,
            "kinds": kinds,
            "distance": int(store.arrays["distance"][index]),
        })
    return rows


@torch.no_grad()
def teacher(model, rows):
    counts = {name: [0, 0] for name in ("node", "start", "continue", "return", "answer")}
    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start + BATCH]
        width = max(len(row["tokens"]) for row in chunk)
        ids = torch.zeros(len(chunk), width, dtype=torch.long, device=DEVICE)
        for i, row in enumerate(chunk):
            ids[i, :len(row["tokens"])] = torch.tensor(row["tokens"], dtype=torch.long, device=DEVICE)
        with torch.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=True):
            pred = model(ids).argmax(-1)
        pred = pred.tolist()
        for i, row in enumerate(chunk):
            tokens = row["tokens"]
            plen = row["prompt_len"]
            for k, kind in enumerate(row["kinds"]):
                hit = int(pred[i][plen + k - 1] == tokens[plen + k])
                counts[kind][0] += 1
                counts[kind][1] += hit
                counts["node"][0] += 1
                counts["node"][1] += hit
            ans = row["answer_index"]
            counts["answer"][0] += 1
            counts["answer"][1] += int(pred[i][ans - 1] == tokens[ans])
    return {name: {"n": n, "exact": rate(hit, n)} for name, (n, hit) in counts.items()}


@torch.no_grad()
def free(model, rows, protocol):
    scored = []
    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start + BATCH]
        prompts = [row["tokens"][:row["prompt_len"]] for row in chunk]
        generated = generate(
            model, prompts, protocol, model.tok, "bf16",
            max_new_tokens=MAX_NEW, cot_margin=8,
        )
        for row, gen in zip(chunk, generated):
            body = row["body"]
            got = gen["trace_tokens"]
            prefix = 0
            for a, b in zip(got, body):
                if a != b:
                    break
                prefix += 1
            answer = gen["pred"] == row["label"] and not gen["format_fail"]
            scored.append({
                "label": row["label"],
                "answer": int(answer),
                "trace": int(got == body and not gen["generation_limit_reached"]),
                "format_fail": int(gen["format_fail"]),
                "prefix_fraction": prefix / max(len(body), 1),
                "distance": row["distance"],
            })
        print(f"    free {start + len(chunk)}/{len(rows)}", flush=True)
    report = pack_answer(scored)
    by_distance = {}
    for distance in sorted({row["distance"] for row in scored}):
        group = [row for row in scored if row["distance"] == distance]
        by_distance[str(distance)] = {
            "n": len(group),
            "answer": rate(sum(row["answer"] for row in group), len(group)),
            "trace_exact": rate(sum(row["trace"] for row in group), len(group)),
        }
    report["by_distance"] = by_distance
    return report


def evaluate(model, tok, protocol, data_name, data_path):
    t0 = time.perf_counter()
    store = ContextStore(data_path)
    rows = examples(store, tok, protocol, N)
    forced = teacher(model, rows)
    print(f"  teacher {data_name} node {forced['node']['exact']:.3f} "
          f"continue {forced['continue']['exact']:.3f} return {forced['return']['exact']:.3f} "
          f"answer {forced['answer']['exact']:.3f}", flush=True)
    rolled = free(model, rows, protocol)
    print(f"  free {data_name} answer {rolled['answer']:.3f} yes {rolled['yes_recall']:.3f} "
          f"no {rolled['no_recall']:.3f} trace {rolled['trace_exact']:.3f} "
          f"prefix {rolled['mean_prefix_fraction']:.3f} ({time.perf_counter() - t0:.0f}s)", flush=True)
    return {
        "n": len(rows),
        "mean_body": float(np.mean([len(row["body"]) for row in rows])),
        "teacher": forced,
        "free": rolled,
    }


def main():
    results = {}
    for protocol, runs in RUNS.items():
        loaded = {}
        for name, path in runs.items():
            _cfg, tok, model, step = base.load_model(path, DEVICE)
            model.tok = tok
            loaded[name] = (model, tok, step)
            print(protocol, name, "step", step, flush=True)
        simple_tok = loaded["simple"][1]
        hard_tok = loaded["hard"][1]
        if simple_tok.yes != hard_tok.yes or simple_tok.node_id(0) != hard_tok.node_id(0):
            raise RuntimeError("tokenizers disagree")
        results[protocol] = {"steps": {name: loaded[name][2] for name in loaded}}
        for size, path in HARD:
            for name in ("simple", "hard"):
                key = f"{name}_on_hard_{size}"
                print(protocol, key, flush=True)
                results[protocol][key] = evaluate(loaded[name][0], loaded[name][1], protocol, size, path)
                OUT.write_text(json.dumps(results, indent=2) + "\n")
        for size, path in SIMPLE:
            key = f"simple_on_simple_{size}"
            print(protocol, key, flush=True)
            results[protocol][key] = evaluate(loaded["simple"][0], loaded["simple"][1], protocol, size, path)
            OUT.write_text(json.dumps(results, indent=2) + "\n")
        del loaded
        torch.cuda.empty_cache()
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()

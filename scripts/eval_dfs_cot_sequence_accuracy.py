#!/usr/bin/env python3
"""Candidate-sequence accuracy when older DFS records are masked.

Teacher forcing scores each candidate token, pushed node and action given the
gold prefix. Free generation scores the aligned trace: a reference record counts
as wrong once the generation stops or that step no longer matches.
"""
from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import plot_dfs_context_scale_attention as scale  # noqa: E402
import plot_dfs_cot_attention as base  # noqa: E402
import eval_dfs_cot_local_attention as ev  # noqa: E402
from cot_navigation.analysis.common import records  # noqa: E402
from cot_navigation.evaluation.rollout import generate  # noqa: E402
from cot_navigation.protocols.validate_trace import parse_completed_steps  # noqa: E402

OUT = ROOT / "figures/scale_v1/dfs_context_local_sequence_accuracy.png"
JSON = ROOT / "research/block_membership_probe_20261021/dfs_context_local_sequence.json"
DEVICE = "cuda:0"
FIELDS = ("stack", "visited", "cands", "action")


def mean(xs) -> float:
    return float(np.mean(xs)) if xs else float("nan")


@torch.inference_mode()
def teacher_tokens(model, items, tok, device) -> dict:
    actions = {tok.id(name) for name in ("<PUSH>", "<POP>", "<FOUND>", "<EXHAUST>")}
    counts = {name: {"n": 0, "full": 0, "local": 0} for name in ("candidates", "candidates_in_set", "push_node", "action", "answer")}
    for item in items:
        tokens = item["tokens"]
        if len(tokens) > ev.MAX_LEN or tokens[-2] not in (tok.yes, tok.no):
            continue
        parsed = records(tokens, item["prompt_len"], tok)
        if not parsed or tokens[parsed[-1]["end"] + 1] not in (tok.yes, tok.no):
            continue
        ids = torch.tensor(tokens, device=device, dtype=torch.long)[None]
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            ev.enable(model, False)
            full = model(ids)
            ev.enable(model, True)
            local = model(ids)
        cand_pos, cand_sets, push_pos, action_pos = [], [], [], []
        for record in parsed:
            a, b, c = record["separators"]
            cand_pos.extend(range(b + 1, c))
            cand_sets.extend([set(tokens[b + 1:c])] * (c - b - 1))
            action_at = c + 1
            if tokens[action_at] in actions:
                action_pos.append(action_at)
            if tokens[action_at] == tok.id("<PUSH>") and action_at + 1 < record["end"]:
                push_pos.append(action_at + 1)
        answer_at = parsed[-1]["end"]

        def add(name, positions):
            if not positions:
                return
            gold = [tokens[p] for p in positions]
            index = [p - 1 for p in positions]
            full_pred = full[0, index].argmax(-1).tolist()
            local_pred = local[0, index].argmax(-1).tolist()
            counts[name]["n"] += len(positions)
            counts[name]["full"] += sum(pred == gold_id for pred, gold_id in zip(full_pred, gold))
            counts[name]["local"] += sum(pred == gold_id for pred, gold_id in zip(local_pred, gold))

        add("candidates", cand_pos)
        if cand_pos:
            index = [p - 1 for p in cand_pos]
            full_pred = full[0, index].argmax(-1).tolist()
            local_pred = local[0, index].argmax(-1).tolist()
            counts["candidates_in_set"]["n"] += len(cand_pos)
            counts["candidates_in_set"]["full"] += sum(pred in pool for pred, pool in zip(full_pred, cand_sets))
            counts["candidates_in_set"]["local"] += sum(pred in pool for pred, pool in zip(local_pred, cand_sets))
        add("push_node", push_pos)
        add("action", action_pos)
        counts["answer"]["n"] += 1
        counts["answer"]["full"] += int(int(full[0, answer_at].argmax()) == tokens[answer_at + 1])
        counts["answer"]["local"] += int(int(local[0, answer_at].argmax()) == tokens[answer_at + 1])
        del full, local
    return {
        name: {"full": row["full"] / max(row["n"], 1), "local": row["local"] / max(row["n"], 1), "n": row["n"]}
        for name, row in counts.items()
    }


def align(gold, got) -> dict:
    fields = {key: 0 for key in FIELDS}
    full = prefix = 0
    broken = False
    first = None
    for index, exp in enumerate(gold):
        if index >= len(got):
            broken = True
            if first is None:
                first = {"truncated": True, **{key: False for key in FIELDS}}
            continue
        row = got[index]
        good = True
        detail = {}
        for key in FIELDS:
            ok = row[key] == exp[key] and (key != "action" or row["arg"] == exp["arg"])
            fields[key] += int(ok)
            detail[key] = ok
            good &= ok
        full += int(good)
        if good and not broken:
            prefix += 1
        elif first is None:
            first = {"truncated": False, **detail}
            broken = True
    if first is None:
        first = {"truncated": False, **{key: True for key in FIELDS}}
    return {"n": len(gold), "fields": fields, "full": full, "prefix": prefix, "first": first,
            "exact": full == len(gold) and len(got) == len(gold)}


def reduce_traces(scored: list[dict]) -> dict:
    n_ref = sum(row["n"] for row in scored)
    out = {"n_traces": len(scored), "n_records": n_ref}
    for key in FIELDS:
        out[key] = sum(row["fields"][key] for row in scored) / max(n_ref, 1)
    out["full_record"] = sum(row["full"] for row in scored) / max(n_ref, 1)
    out["prefix_fraction"] = mean([row["prefix"] / row["n"] for row in scored if row["n"]])
    out["exact_trace"] = mean([row["exact"] for row in scored])
    firsts = [row["first"] for row in scored if row["prefix"] < row["n"]]
    out["first_error_n"] = len(firsts)
    out["first_error"] = {
        "truncated": mean([row["truncated"] for row in firsts]) if firsts else 0.0,
        **{key: mean([(not row[key]) and (not row["truncated"]) for row in firsts]) if firsts else 0.0 for key in FIELDS},
    }
    return out


@torch.inference_mode()
def replay(records_got, item) -> dict:
    """Score each generated step against the DFS state implied by earlier actions."""
    inverse = {token: index for index, token in enumerate(item["node_tokens"])}
    visited = {item["source"]}
    stack = [item["source"]]
    cands_ok = action_ok = step_ok = prefix = 0
    broken = False
    for record in records_got:
        if not stack:
            broken = True
            break
        cur = stack[-1]
        try:
            wrote_cur = inverse[record["cur"]]
            wrote_cands = [inverse[token] for token in record["cands"]]
            wrote_arg = None if record["arg"] is None else inverse[record["arg"]]
        except KeyError:
            broken = True
            break
        legal = [node for node in item["adj"][cur] if node not in visited]
        legal_set = set(legal)
        set_ok = wrote_cur == cur and set(wrote_cands) == legal_set and len(wrote_cands) == len(legal_set)
        if cur == item["target"]:
            act_ok = record["action"] == "FOUND" and record["arg"] is None
        elif legal_set:
            act_ok = record["action"] == "PUSH" and wrote_arg in legal_set
        elif len(stack) > 1:
            act_ok = record["action"] == "POP" and record["arg"] is None
        else:
            act_ok = record["action"] == "EXHAUST" and record["arg"] is None
        cands_ok += int(set_ok)
        action_ok += int(act_ok and wrote_cur == cur)
        this = set_ok and act_ok
        step_ok += int(this)
        if this and not broken:
            prefix += 1
        else:
            broken = True
        if record["action"] == "PUSH" and wrote_arg is not None and wrote_arg not in visited:
            visited.add(wrote_arg)
            stack.append(wrote_arg)
        elif record["action"] == "POP" and len(stack) > 1:
            stack.pop()
        else:
            break
    n = max(len(records_got), 1)
    return {"n": len(records_got), "cands_set": cands_ok, "action_legal": action_ok,
            "step": step_ok, "prefix": prefix, "n_div": n}


def free_sequences(model, items, tok, device, local: bool) -> dict:
    ev.enable(model, local)
    scored, answers = [], []
    batch = 8
    for start in range(0, len(items), batch):
        chunk = items[start:start + batch]
        prompts = [item["tokens"][:item["prompt_len"]] for item in chunk]
        generated = generate(model, prompts, "dfs_state", tok, "bf16", cot_margin=8)
        for item, gen in zip(chunk, generated):
            gold = parse_completed_steps(item["tokens"][item["prompt_len"]:-2], tok)
            got = parse_completed_steps(gen["trace_tokens"], tok)
            row = align(gold, got)
            row["legal"] = replay(got, item)
            row["label"] = item["label"]
            row["answer"] = gen["pred"] == item["label"] and not gen["format_fail"]
            scored.append(row)
            answers.append(row["answer"])
        print(f"    free {'local' if local else 'full'} {min(start + batch, len(items))}/{len(items)}", flush=True)
    summary = reduce_traces(scored)
    labels = [row["label"] for row in scored]
    yes = np.asarray(labels) == 1
    correct = np.asarray(answers, dtype=bool)
    summary["answer_balanced"] = float(0.5 * (correct[yes].mean() + correct[~yes].mean()))
    n_steps = sum(row["legal"]["n"] for row in scored)
    summary["cands_set"] = sum(row["legal"]["cands_set"] for row in scored) / max(n_steps, 1)
    summary["action_legal"] = sum(row["legal"]["action_legal"] for row in scored) / max(n_steps, 1)
    summary["legal_prefix"] = mean([row["legal"]["prefix"] / row["legal"]["n_div"] for row in scored])
    return summary


def draw(rows: list[dict]) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.linewidth": 0.6,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.4), sharey=True)
    panels = (
        ("teacher", [("candidates", "Exact order"), ("candidates_in_set", "In the set"), ("action", "Action"), ("answer", "Answer")],
         "Teacher-forced tokens"),
        ("free", [("cands_set", "Candidate set"), ("action_legal", "Legal action"), ("legal_prefix", "Legal prefix"), ("answer_balanced", "Answer")],
         "Free generation"),
    )
    width = 0.36
    for row_index, (source, metrics, prefix) in enumerate(panels):
        for column, block in enumerate(rows):
            ax = axes[row_index, column]
            x = np.arange(len(metrics))
            full = [block[source]["full"][key] for key, _label in metrics]
            local = [block[source]["local"][key] for key, _label in metrics]
            ax.bar(x - width / 2, full, width, color="#0072B2", label="Full attention")
            ax.bar(x + width / 2, local, width, color="#D55E00", label="Current + previous")
            ax.set_xticks(x)
            ax.set_xticklabels([label for _key, label in metrics], fontsize=8)
            ax.set_ylim(0, 1.08)
            ax.axhline(0.5, color="#666666", linewidth=0.7, linestyle="--")
            ax.yaxis.grid(True, color="#dddddd", linewidth=0.6)
            ax.set_axisbelow(True)
            ax.set_title(f"{prefix}, {block['size']}", fontsize=10)
            if column == 0:
                ax.set_ylabel("Accuracy")
    axes[0, 2].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("In-context DFS sequence accuracy with older records masked", fontsize=12)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.08, wspace=0.16, hspace=0.32)
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=200, facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".png.tmp")
    Image.open(buffer).convert("RGB").save(tmp, format="PNG")
    tmp.replace(OUT)
    print(OUT)


def main() -> None:
    device = torch.device(DEVICE)
    _cfg, tok, model, step = base.load_model(base.CONTEXT_RUN, device)
    saved = []
    for name, root in scale.SIZES:
        arrays = scale.load_test(root)
        items = [scale.render_index(arrays, tok, index) for index in range(len(arrays["label"]))]
        width = arrays["node_token"].shape[1]
        for item in items:
            adj = [[] for _ in range(width)]
            index = item["index"]
            for src, dst in zip(arrays["edge_src"][index].tolist(), arrays["edge_dst"][index].tolist()):
                adj[int(src)].append(int(dst))
            item["adj"] = adj
        print(f"\n{name}", flush=True)
        forced = teacher_tokens(model, items, tok, device)
        print("  teacher", json.dumps(forced), flush=True)
        subset = ev.free_subset(items)
        free_full = free_sequences(model, subset, tok, device, False)
        free_local = free_sequences(model, subset, tok, device, True)
        print("  free full", json.dumps(free_full), flush=True)
        print("  free local", json.dumps(free_local), flush=True)
        saved.append({"size": name, "step": step, "teacher": {"full": {k: v["full"] for k, v in forced.items()},
                                                              "local": {k: v["local"] for k, v in forced.items()},
                                                              "detail": forced},
                      "free": {"full": free_full, "local": free_local}})
    draw(saved)
    JSON.parent.mkdir(parents=True, exist_ok=True)
    JSON.write_text(json.dumps(saved, indent=2))
    print(JSON)


if __name__ == "__main__":
    main()

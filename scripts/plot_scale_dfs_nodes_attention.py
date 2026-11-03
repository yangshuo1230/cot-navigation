#!/usr/bin/env python3
"""DFS-node versus DFS-state attention on in-context layered graphs.

DFS state reads five tokens after the chosen edge at layer 5. DFS nodes reads
the parent's edge relays at layer 4 only while the previous node is that parent;
a backtrack step reads the emitted prefix instead.
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

import plot_dfs_cot_attention as base  # noqa: E402
import plot_dfs_nodes_attention as probe  # noqa: E402
import plot_path_dag_attention as pathfig  # noqa: E402
from cot_navigation.protocols.render import render_task  # noqa: E402

FIG = ROOT / "figures/scale_v1/dfs_nodes_attention.png"
JSON_PATH = ROOT / "studies/block_membership_probe/scale_dfs_nodes_attention.json"
DEVICE = "cuda:0"
STATE_RUN = ROOT / "outputs/scale_v1/launches/scale_v1_s0/context/curriculum/dfs_state"
NODE_RUN = ROOT / "outputs/scale_v1/launches/scale_v1_s0/context/curriculum/dfs_nodes"
SIZES = [
    ("4×4", ROOT / "data/scale_v1/context_l4w4_n16_m24_train50000_s0"),
    ("6×6", ROOT / "data/scale_v1/context_l6w6_n36_m54_train50000_s0"),
    ("10×10", ROOT / "data/scale_v1/context_l10w10_n100_m150_train50000_s0"),
]


def mean(values):
    return float(np.mean(values)) if len(values) else None


@torch.no_grad()
def logits_of(model, tokens, device, hide=None):
    attn_mod = model.blocks[4].attn
    original = attn_mod._attend

    def wrapped(q, k, v, keep, force_math, causal=False):
        length = q.shape[-2]
        index = torch.arange(length, device=q.device)
        visible = index.view(1, 1, 1, length) <= index.view(1, 1, length, 1)
        for query, key in hide:
            if key < query:
                visible[0, 0, query, key] = False
        visible[0, 0].fill_diagonal_(True)
        return original(q, k, v, visible, True, causal=False)

    if hide is not None:
        attn_mod._attend = wrapped
    try:
        ids = torch.tensor(tokens, device=device, dtype=torch.long)[None]
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(ids, return_attn=hide is None)
        if hide is None:
            logits, weights = out
            attn = torch.stack([weight[0, 0].float().cpu() for weight in weights])
        else:
            logits, attn = out, None
        return logits[0].float().cpu(), attn
    finally:
        attn_mod._attend = original


def measure(state_model, node_model, tok, items, device, max_seq_len, margin):
    semi = tok.id(";")
    layers = 6
    edge_mass = [[] for _ in range(layers)]
    prefix_mass = [[] for _ in range(layers)]
    bar_mass = [[] for _ in range(layers)]
    state = {"k1": [], "k2": [], "chance": [], "slot": [], "slot_avg": [], "copy_dst": [], "same_token": []}
    hidden = {"k1": [], "k2": []}
    control = {"k1": [], "k2": []}
    kinds = {
        kind: {"k1": [], "k2": [], "chance": [], "relay": [], "chosen": [], "prefix": [], "parent": []}
        for kind in ("go", "back")
    }
    for item in items:
        state_row, _cut = probe._fit(render_task(
            tok, "dfs_state", item["label"], item["prompt"], item["adj"],
            item["source"], item["target"], item["node_tokens"],
        ), tok, max_seq_len, margin)
        logits, attn = logits_of(state_model, state_row.tokens, device)
        pos = plen = state_row.prompt_len
        hides, controls = [], []
        scored = []
        for step in item["steps"]:
            fields = []
            for nodes in (step.stack, step.visited, step.cands):
                start = pos
                pos += len(nodes) + 1
                fields.append((start, pos - 1))
            action = pos
            pos += 1
            extra = 1 if step.action == "PUSH" else 0
            semi_pos = pos + extra
            if semi_pos >= len(state_row.tokens) or state_row.tokens[semi_pos] != semi:
                break
            if step.action == "PUSH":
                bar = fields[1][1]
                gold = item["node_tokens"][step.arg]
                edge = probe.edge_index(item["edges"], step.cur, step.arg)
                slot = 3 * edge + 5
                hit = float(int(logits[bar].argmax()) == gold)
                bucket = "k1" if len(step.cands) == 1 else "k2"
                state[bucket].append(hit)
                if bucket == "k2":
                    state["chance"].append(1.0 / len(step.cands))
                row = attn[4, bar]
                state["slot"].append(float(row[slot]) if slot < row.shape[0] else 0.0)
                offs = [3 * index + 5 for index in range(len(item["edges"])) if 3 * index + 5 < row.shape[0]]
                state["slot_avg"].append(float(row[offs].mean()) if offs else 0.0)
                if slot < attn.shape[-1]:
                    state["copy_dst"].append(float(attn[0, slot, 3 * edge + 2]))
                    state["same_token"].append(float(state_row.tokens[slot] == gold))
                    hides.append((bar, slot))
                    controls.append((bar, 0 if slot != 0 else 1))
                    scored.append((bucket, gold, bar))
                pos += 1
            pos += 1
        del logits, attn
        for store, pairs in ((hidden, hides), (control, controls)):
            masked, _attn = logits_of(state_model, state_row.tokens, device, hide=pairs)
            for bucket, gold, bar in scored:
                store[bucket].append(float(int(masked[bar].argmax()) == gold))
            del masked
        node_row = render_task(
            tok, "dfs_nodes", item["label"], item["prompt"], item["adj"],
            item["source"], item["target"], item["node_tokens"],
        )
        logits, attn = logits_of(node_model, node_row.tokens, device)
        plen = node_row.prompt_len
        seen = {}
        for dec_pos, last, parent, gold in probe.node_positions(item["steps"], plen):
            if dec_pos + 1 >= len(node_row.tokens) or node_row.tokens[dec_pos + 1] != item["node_tokens"][gold]:
                break
            seen[last] = dec_pos
            kind = "back" if parent != last else "go"
            kids = [v for v in item["adj"][parent] if v not in seen]
            k = max(len(kids), 1)
            hit = float(int(logits[dec_pos].argmax()) == item["node_tokens"][gold])
            slot = kinds[kind]
            slot["k1" if k == 1 else "k2"].append(hit)
            if k >= 2:
                slot["chance"].append(1.0 / k)
            edge = probe.edge_index(item["edges"], parent, gold)
            relays = probe.relay_positions(item["edges"], parent)
            parent_at = dec_pos if parent == last else seen.get(parent)
            for layer in range(layers):
                row = attn[layer, dec_pos]
                edge_mass[layer].append(float(row[: plen - 4].sum()))
                prefix_mass[layer].append(float(row[plen:dec_pos].sum()) if dec_pos > plen else 0.0)
                bar_mass[layer].append(float(row[plen - 1]))
            row4 = attn[3, dec_pos]
            row5 = attn[4, dec_pos]
            slot["relay"].append(float(row4[relays].sum()) if relays else 0.0)
            slot["chosen"].append(float(row4[3 * edge + 3]) if edge is not None else 0.0)
            slot["prefix"].append(float(row5[plen:dec_pos].sum()) if dec_pos > plen else 0.0)
            slot["parent"].append(float(row5[parent_at]) if parent_at is not None else 0.0)
        del logits, attn
    def pack(kind):
        slot = kinds[kind]
        return {
            "one": mean(slot["k1"]), "one_n": len(slot["k1"]),
            "several": mean(slot["k2"]), "several_n": len(slot["k2"]),
            "chance": mean(slot["chance"]),
            "l4_relays": mean(slot["relay"]), "l4_chosen": mean(slot["chosen"]),
            "l5_prefix": mean(slot["prefix"]), "l5_parent": mean(slot["parent"]),
        }
    return {
        "state": {
            "one": mean(state["k1"]), "one_n": len(state["k1"]),
            "several": mean(state["k2"]), "several_n": len(state["k2"]),
            "chance": mean(state["chance"]),
            "slot": mean(state["slot"]), "slot_avg": mean(state["slot_avg"]),
            "copy_dst": mean(state["copy_dst"]), "same_token": mean(state["same_token"]),
            "hide_one": mean(hidden["k1"]), "hide_several": mean(hidden["k2"]),
            "ctrl_one": mean(control["k1"]), "ctrl_several": mean(control["k2"]),
        },
        "nodes": {
            "edge": [mean(v) for v in edge_mass],
            "prefix": [mean(v) for v in prefix_mass],
            "bar": [mean(v) for v in bar_mass],
            "go": pack("go"),
            "back": pack("back"),
        },
    }


def draw(payload):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.6,
        "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "white",
    })
    blue, vermillion, green, orange = "#0072B2", "#D55E00", "#009E73", "#E69F00"
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.0), constrained_layout=True)
    sizes = [block["size"] for block in payload]
    x = np.arange(len(sizes))
    layers = np.arange(1, 7)
    ax = axes[0]
    for block, color, marker in zip(payload, (blue, green, orange), ("o", "s", "^")):
        ax.plot(layers, block["nodes"]["edge"], color=color, marker=marker, label=f"{block['size']} edge list")
        ax.plot(layers, block["nodes"]["prefix"], color=color, marker=marker, ls="--", label=f"{block['size']} emitted nodes")
    ax.set_xticks(layers)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Attention mass")
    ax.set_ylim(0, 1.05)
    ax.set_title("DFS nodes, next expansion")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    width = 0.18
    series = (
        ("DFS L5, chosen +5", [block["state"]["slot"] for block in payload], blue, -1.5),
        ("DFS L5, average +5", [block["state"]["slot_avg"] for block in payload], "#56B4E9", -0.5),
        ("Nodes L4, continue", [block["nodes"]["go"]["l4_chosen"] for block in payload], green, 0.5),
        ("Nodes L4, backtrack", [block["nodes"]["back"]["l4_chosen"] for block in payload], vermillion, 1.5),
    )
    for label, values, color, offset in series:
        ax.bar(x + offset * width, values, width=width, color=color, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Attention mass")
    ax.set_title("Edge slot of the chosen child")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)

    ax = axes[2]
    width = 0.12
    series = (
        ("DFS, one child", [block["state"]["one"] for block in payload], blue, -2.5),
        ("DFS, two or more", [block["state"]["several"] for block in payload], "#56B4E9", -1.5),
        ("Nodes continue, one", [block["nodes"]["go"]["one"] for block in payload], green, -0.5),
        ("Nodes continue, 2+", [block["nodes"]["go"]["several"] for block in payload], "#F0E442", 0.5),
        ("Nodes backtrack, one", [block["nodes"]["back"]["one"] for block in payload], vermillion, 1.5),
        ("Nodes backtrack, 2+", [block["nodes"]["back"]["several"] for block in payload], orange, 2.5),
    )
    for label, values, color, offset in series:
        ax.bar(x + offset * width, values, width=width, color=color, label=label)
    chance = float(np.mean([block["state"]["chance"] for block in payload]))
    ax.axhline(chance, color="0.45", ls="--", lw=0.8, label=f"uniform, 2+ ({chance:.2f})")
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Exact next node")
    ax.set_title("Teacher-forced choice")
    ax.legend(frameon=False, fontsize=6.5)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    buf.seek(0)
    Image.open(buf).convert("RGB").save(FIG)


def main():
    device = torch.device(DEVICE)
    cfg, tok, state_model, state_step = base.load_model(STATE_RUN, device)
    _cfg, _tok, node_model, node_step = base.load_model(NODE_RUN, device)
    print("state", state_step, "nodes", node_step, flush=True)
    saved = []
    for name, root in SIZES:
        block = measure(
            state_model, node_model, tok, probe.choose(pathfig.load_test(root), tok),
            device, cfg.model.max_seq_len, cfg.protocol.cot_margin,
        )
        block["size"] = name
        block["state_step"] = state_step
        block["node_step"] = node_step
        saved.append(block)
        state, nodes = block["state"], block["nodes"]
        print(
            f"{name} DFS one {state['one']:.3f} n={state['one_n']} several {state['several']:.3f} "
            f"n={state['several_n']} chance {state['chance']:.3f} slot {state['slot']:.3f} "
            f"avg {state['slot_avg']:.3f} copy {state['copy_dst']:.3f} token {state['same_token']:.3f}",
            flush=True,
        )
        print(
            f"  hide several {state['hide_several']:.3f} one {state['hide_one']:.3f} "
            f"ctrl several {state['ctrl_several']:.3f} one {state['ctrl_one']:.3f}",
            flush=True,
        )
        for kind in ("go", "back"):
            slot = nodes[kind]
            print(
                f"  {kind} one {slot['one']:.3f} n={slot['one_n']} several {slot['several']:.3f} "
                f"n={slot['several_n']} chance {slot['chance']:.3f} L4 relays {slot['l4_relays']:.3f} "
                f"chosen {slot['l4_chosen']:.3f} L5 prefix {slot['l5_prefix']:.3f} parent {slot['l5_parent']:.3f}",
                flush=True,
            )
        print("  edge", " ".join(f"{v:.3f}" for v in nodes["edge"]), flush=True)
        print("  prefix", " ".join(f"{v:.3f}" for v in nodes["prefix"]), flush=True)
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(saved, indent=2))
    draw(saved)
    print(FIG)


if __name__ == "__main__":
    main()

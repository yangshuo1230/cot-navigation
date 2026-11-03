#!/usr/bin/env python3
"""Attention while in-context DFS nodes chooses the next expansion.

Full DFS writes a candidate list. The bar after the visited field reads the
current node's outgoing-edge relays, and <PUSH> copies the first candidate.
DFS nodes has neither stack nor candidates. Layers 2-5 at the next-node
position attend across the edge list; layer 6 attends to the prompt bar.
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
import plot_path_dag_attention as pathfig  # noqa: E402
from cot_navigation.protocols.bfs import first_expansion_order  # noqa: E402
from cot_navigation.protocols.dfs import dfs_search  # noqa: E402
from cot_navigation.protocols.render import context_prompt, render_task, truncate_example  # noqa: E402

FIG = ROOT / "figures/hard_dag_v1/dfs_nodes_attention.png"
JSON_PATH = ROOT / "studies/block_membership_probe/dfs_nodes_attention.json"
DEVICE = "cuda:0"
PER_LABEL = 64
STATE_RUN = ROOT / "outputs/hard_dag_v1/launches/hard_dag_v1_s0/context/curriculum/dfs_state"
NODE_RUN = ROOT / "outputs/hard_dag_v1/launches/hard_dag_v1_s0/context/curriculum/dfs_nodes"


def prepare(arrays, tok, index: int):
    width = arrays["node_token"].shape[1]
    node_tokens = [tok.node_id(int(v)) for v in arrays["node_token"][index].tolist()]
    edges = list(zip(arrays["edge_src"][index].astype(int).tolist(), arrays["edge_dst"][index].astype(int).tolist()))
    src, dst = int(arrays["s_topo"][index]), int(arrays["t_topo"][index])
    adj = [[] for _ in range(width)]
    for u, v in edges:
        adj[u].append(v)
    prompt = context_prompt(tok, node_tokens, edges, src, dst)
    return {
        "prompt": prompt,
        "node_tokens": node_tokens,
        "edges": edges,
        "adj": adj,
        "source": src,
        "target": dst,
        "label": int(arrays["label"][index]),
        "steps": dfs_search(adj, src, dst),
        "index": int(index),
    }


def choose(arrays, tok):
    picked = []
    for label in (1, 0):
        ids = [i for i in range(len(arrays["label"])) if int(arrays["label"][i]) == label]
        picked.extend(ids[:PER_LABEL])
    return [prepare(arrays, tok, i) for i in picked]


def edge_index(edges, src, dst):
    for index, (u, v) in enumerate(edges):
        if u == src and v == dst:
            return index
    return None


def relay_positions(edges, src):
    return [3 * index + 3 for index, (u, _v) in enumerate(edges) if u == src]


def node_positions(steps, prompt_len):
    order = first_expansion_order(steps)
    parent = {}
    for step in steps:
        if step.action == "PUSH" and step.arg not in parent:
            parent[step.arg] = step.cur
    return [
        (prompt_len + index, order[index], parent[order[index + 1]], order[index + 1])
        for index in range(len(order) - 1)
    ]


@torch.inference_mode()
def forward_all(model, tokens, device):
    ids = torch.tensor(tokens, device=device, dtype=torch.long)[None]
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits, weights = model(ids, return_attn=True)
    attn = torch.stack([weight[0, 0].float().cpu() for weight in weights])
    return logits[0].float().cpu(), attn


def mean(values):
    return float(np.mean(values)) if len(values) else None


def _fit(row, tok, max_seq_len, margin):
    if max_seq_len is None:
        return row, False
    limited = truncate_example(row, tok, max_seq_len, margin)
    return limited, len(limited.tokens) != len(row.tokens)


def measure(state_model, node_model, tok, items, device, max_seq_len=None, margin=0):
    vis_exact, vis_edge, vis_gold, vis_parent = [], [], [], []
    push_exact, push_cand = [], []
    layers = 6
    node = {key: [[] for _ in range(layers)] for key in ("bar", "edge", "prefix", "parent", "gold")}
    exact, direct, back = [], [], []
    state_cut = node_cut = 0
    semi = tok.id(";")
    for item in items:
        state_row, cut = _fit(render_task(
            tok, "dfs_state", item["label"], item["prompt"], item["adj"],
            item["source"], item["target"], item["node_tokens"],
        ), tok, max_seq_len, margin)
        state_cut += int(cut)
        logits, attn = forward_all(state_model, state_row.tokens, device)
        pos = state_row.prompt_len
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
                edge = edge_index(item["edges"], step.cur, step.arg)
                relay = 3 * edge + 3
                parent_relays = relay_positions(item["edges"], step.cur)
                row = attn[-1, fields[1][1]]
                gold = item["node_tokens"][step.arg]
                vis_exact.append(float(int(logits[fields[1][1]].argmax()) == gold))
                vis_edge.append(float(row[: state_row.prompt_len - 4].sum()))
                vis_gold.append(float(row[relay]) if relay < row.shape[0] else 0.0)
                vis_parent.append(float(row[parent_relays].sum()) if parent_relays else 0.0)
                push_exact.append(float(int(logits[action].argmax()) == gold))
                push_cand.append(float(attn[-1, action, fields[2][0]]))
                pos += 1
            pos += 1
        del logits, attn
        node_row, cut = _fit(render_task(
            tok, "dfs_nodes", item["label"], item["prompt"], item["adj"],
            item["source"], item["target"], item["node_tokens"],
        ), tok, max_seq_len, margin)
        node_cut += int(cut)
        logits, attn = forward_all(node_model, node_row.tokens, device)
        prompt_len = node_row.prompt_len
        tokens = node_row.tokens
        for dec_pos, last, parent, gold in node_positions(item["steps"], prompt_len):
            if dec_pos + 1 >= len(tokens) or tokens[dec_pos + 1] != item["node_tokens"][gold]:
                break
            edge = edge_index(item["edges"], parent, gold)
            relay = 3 * edge + 3
            parent_relays = relay_positions(item["edges"], parent)
            hit = float(int(logits[dec_pos].argmax()) == item["node_tokens"][gold])
            exact.append(hit)
            (back if parent != last else direct).append(hit)
            for layer in range(layers):
                row = attn[layer, dec_pos]
                node["bar"][layer].append(float(row[prompt_len - 1]))
                node["edge"][layer].append(float(row[: prompt_len - 4].sum()))
                node["prefix"][layer].append(float(row[prompt_len:dec_pos].sum()) if dec_pos > prompt_len else 0.0)
                node["parent"][layer].append(float(row[parent_relays].sum()) if parent_relays else 0.0)
                node["gold"][layer].append(float(row[relay]) if relay < row.shape[0] else 0.0)
        del logits, attn
    return {
        "state": {
            "n": len(vis_exact),
            "truncated": state_cut,
            "candidate_exact": mean(vis_exact),
            "edge": mean(vis_edge),
            "gold_relay": mean(vis_gold),
            "parent_relay": mean(vis_parent),
            "push_exact": mean(push_exact),
            "push_on_candidate": mean(push_cand),
        },
        "nodes_truncated": node_cut,
        "nodes": {
            "n": len(exact),
            "exact": mean(exact),
            "direct_exact": mean(direct),
            "direct_n": len(direct),
            "back_exact": mean(back),
            "back_n": len(back),
            "bar": [mean(v) for v in node["bar"]],
            "edge": [mean(v) for v in node["edge"]],
            "prefix": [mean(v) for v in node["prefix"]],
            "parent": [mean(v) for v in node["parent"]],
            "gold": [mean(v) for v in node["gold"]],
        },
    }


def draw(payload, path=None, title=None):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "0.9",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    blue, vermillion, green, orange = "#0072B2", "#D55E00", "#009E73", "#E69F00"
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9), constrained_layout=True)
    sizes = [block["size"] for block in payload]
    x = np.arange(len(sizes))
    layers = np.arange(1, 7)

    ax = axes[0]
    for block, color, marker in zip(payload, (blue, green, orange), ("o", "s", "^")):
        ax.plot(layers, block["nodes"]["edge"], color=color, marker=marker, label=f"{block['size']} edge list")
        ax.plot(layers, block["nodes"]["bar"], color=color, marker=marker, ls="--", label=f"{block['size']} prompt bar")
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
        ("DFS, parent relays", [block["state"]["parent_relay"] for block in payload], blue, -1.5),
        ("DFS, chosen relay", [block["state"]["gold_relay"] for block in payload], green, -0.5),
        ("Nodes L3, edge list", [block["nodes"]["edge"][2] for block in payload], orange, 0.5),
        ("Nodes L3, parent relays", [block["nodes"]["parent"][2] for block in payload], vermillion, 1.5),
    )
    for label, values, color, offset in series:
        ax.bar(x + offset * width, values, width=width, color=color, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Attention mass")
    ax.set_title("Where the edge read lands")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)

    ax = axes[2]
    width = 0.24
    series = (
        ("DFS candidate", [block["state"]["candidate_exact"] for block in payload], blue, -1),
        ("Nodes, continue", [block["nodes"]["direct_exact"] for block in payload], green, 0),
        ("Nodes, backtrack", [block["nodes"]["back_exact"] for block in payload], vermillion, 1),
    )
    for label, values, color, offset in series:
        ax.bar(x + offset * width, values, width=width, color=color, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Exact next node")
    ax.set_title("Teacher-forced choice")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)

    if title:
        fig.suptitle(title)
    target = Path(path) if path is not None else FIG
    target.parent.mkdir(parents=True, exist_ok=True)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    buf.seek(0)
    Image.open(buf).convert("RGB").save(target)


def main():
    device = torch.device(DEVICE)
    _cfg, tok, state_model, state_step = base.load_model(STATE_RUN, device)
    _cfg, _tok, node_model, node_step = base.load_model(NODE_RUN, device)
    print("state", state_step, "nodes", node_step, flush=True)
    saved = []
    for name, root in pathfig.SIZES:
        block = measure(state_model, node_model, tok, choose(pathfig.load_test(root), tok), device)
        block["size"] = name.replace("DAG ", "")
        block["state_step"] = state_step
        block["node_step"] = node_step
        saved.append(block)
        state, nodes = block["state"], block["nodes"]
        print(
            f"{block['size']} state n={state['n']} candidate {state['candidate_exact']:.3f} "
            f"edge {state['edge']:.3f} parent {state['parent_relay']:.3f} chosen {state['gold_relay']:.3f} "
            f"push {state['push_exact']:.3f} on_cand {state['push_on_candidate']:.3f}",
            flush=True,
        )
        print(
            f"  nodes n={nodes['n']} exact {nodes['exact']:.3f} continue {nodes['direct_exact']:.3f} "
            f"back {nodes['back_exact']:.3f}",
            flush=True,
        )
        print("  edge ", " ".join(f"{v:.3f}" for v in nodes["edge"]), flush=True)
        print("  bar  ", " ".join(f"{v:.3f}" for v in nodes["bar"]), flush=True)
        print("  parent", " ".join(f"{v:.3f}" for v in nodes["parent"]), flush=True)
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(saved, indent=2))
    draw(saved)
    print(FIG)


if __name__ == "__main__":
    main()

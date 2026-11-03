#!/usr/bin/env python3
"""DFS-node attention in the same layout as the DFS-state action heatmaps.

Rows are expansion steps. Columns partition the tokens visible at that step.
Layer 1 reads the current node, layer 4 reads its outgoing-edge slots, and
layer 6 reads nodes already written when the step has to leave that node.
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
from cot_navigation.protocols.render import context_prompt, render_task  # noqa: E402

OUT = ROOT / "figures/scale_v1/dfs_nodes_attention_examples.png"
JSON_PATH = ROOT / "studies/block_membership_probe/dfs_nodes_attention_examples.json"
DEVICE = "cuda:0"
COLUMNS = [
    "Query source",
    "Query target",
    "Prompt bar",
    "Parent edges",
    "Other graph",
    "Earlier nodes",
    "Self",
    "Other",
]
LAYERS = (1, 4, 6)


def prepare(arrays, tok, index):
    width = arrays["node_token"].shape[1]
    node_tokens = [tok.node_id(int(v)) for v in arrays["node_token"][index].tolist()]
    edges = list(zip(arrays["edge_src"][index].astype(int).tolist(), arrays["edge_dst"][index].astype(int).tolist()))
    src, dst = int(arrays["s_topo"][index]), int(arrays["t_topo"][index])
    adj = [[] for _ in range(width)]
    for u, v in edges:
        adj[u].append(v)
    prompt = context_prompt(tok, node_tokens, edges, src, dst)
    steps = dfs_search(adj, src, dst)
    parent = {}
    for step in steps:
        if step.action == "PUSH" and step.arg not in parent:
            parent[step.arg] = step.cur
    return {
        "tokens": render_task(tok, "dfs_nodes", int(arrays["label"][index]), prompt, adj, src, dst, node_tokens).tokens,
        "prompt_len": len(prompt),
        "node_tokens": node_tokens,
        "edges": edges,
        "adj": adj,
        "parent": parent,
        "order": first_expansion_order(steps),
        "source": src,
        "target": dst,
        "label": int(arrays["label"][index]),
        "index": int(index),
    }


def choose_hard(arrays, tok):
    best = None
    for index, label in enumerate(arrays["label"].tolist()):
        if int(label) != 1:
            continue
        item = prepare(arrays, tok, index)
        order, parent = item["order"], item["parent"]
        backs = sum(parent.get(order[i + 1]) != order[i] for i in range(len(order) - 1))
        if not 5 <= len(order) <= 8 or backs < 2:
            continue
        rank = (len(order), -backs, index)
        if best is None or rank < best[0]:
            best = (rank, item)
    if best is None:
        raise RuntimeError("no short hard-DAG trace with a backtrack")
    return best[1]


def parent_edge_positions(item, parent):
    if parent is None:
        return set()
    limit = item["prompt_len"] - 4
    return {3 * index + 3 for index, (src, _dst) in enumerate(item["edges"]) if src == parent and 3 * index + 3 < limit}


def row_masses(attn, item, receiver, parent):
    prompt_len = item["prompt_len"]
    row = attn[receiver, : receiver + 1].numpy()
    row = row / max(float(row.sum()), 1e-8)
    relays = parent_edge_positions(item, parent)
    graph_end = prompt_len - 4
    out = np.zeros(len(COLUMNS), dtype=np.float64)
    for pos, mass in enumerate(row):
        if pos == prompt_len - 3:
            bucket = 0
        elif pos == prompt_len - 2:
            bucket = 1
        elif pos == prompt_len - 1:
            bucket = 2
        elif pos in relays:
            bucket = 3
        elif pos < graph_end:
            bucket = 4
        elif prompt_len <= pos < receiver:
            bucket = 5
        elif pos == receiver:
            bucket = 6
        else:
            bucket = 7
        out[bucket] += mass
    return out


def example_matrix(attn, item):
    prompt_len = item["prompt_len"]
    order, parent = item["order"], item["parent"]
    inverse = {token: index for index, token in enumerate(item["node_tokens"])}
    rows, labels = [], []
    for index, node in enumerate(order):
        if index == len(order) - 1:
            rows.append(row_masses(attn, item, prompt_len + index, None))
            labels.append("Answer")
            break
        gold = order[index + 1]
        kind = "back*" if parent.get(gold) != node and not item["adj"][node] else (
            "back" if parent.get(gold) != node else "cont"
        )
        rows.append(row_masses(attn, item, prompt_len + index, parent.get(gold)))
        labels.append(f"{index + 1} {kind} {inverse[item['node_tokens'][gold]]}")
    return {"matrix": np.vstack(rows), "labels": labels}


@torch.no_grad()
def all_layers(model, tokens, device):
    ids = torch.tensor(tokens, device=device, dtype=torch.long)[None]
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        _logits, weights = model(ids, return_attn=True)
    return [weight[0, 0].float().cpu() for weight in weights]


def draw(panels):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.linewidth": 0.6,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.4))
    image = None
    for ax, panel in zip(axes.ravel(), panels):
        grid = panel["matrix"]
        image = ax.imshow(grid, vmin=0.0, vmax=1.0, cmap="cividis", aspect="auto")
        ax.set_xticks(range(len(COLUMNS)))
        ax.set_xticklabels(COLUMNS, rotation=55, ha="right", fontsize=7)
        ax.set_yticks(range(len(panel["labels"])))
        ax.set_yticklabels(panel["labels"], fontsize=7)
        ax.set_title(panel["title"], fontsize=9, pad=6)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                value = grid[i, j]
                if value < 0.08:
                    continue
                color = "white" if value < 0.62 else "black"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6, color=color)
    fig.subplots_adjust(left=0.08, right=0.90, top=0.88, bottom=0.16, wspace=0.55, hspace=0.62)
    colorbar = fig.add_axes((0.92, 0.28, 0.014, 0.48))
    fig.colorbar(image, cax=colorbar, label="Attention mass")
    fig.suptitle("DFS-node attention at each expansion", fontsize=12)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    Image.open(buf).convert("RGB").save(OUT)
    print(OUT)


def main():
    device = torch.device(DEVICE)
    layered_run = ROOT / "outputs/scale_v1/launches/scale_v1_s0/context/curriculum/dfs_nodes"
    hard_run = ROOT / "outputs/hard_dag_v1/launches/hard_dag_v1_s0/context/curriculum/dfs_nodes"
    _cfg, tok, model, step = base.load_model(layered_run, device)
    layered_data = pathfig.load_test(ROOT / "data/scale_v1/context_l4w4_n16_m24_train50000_s0")
    layered = prepare(layered_data, tok, 2)
    hard_cfg_model = base.load_model(hard_run, device)
    hard_tok, hard_model, hard_step = hard_cfg_model[1], hard_cfg_model[2], hard_cfg_model[3]
    hard = choose_hard(pathfig.load_test(pathfig.SIZES[0][1]), hard_tok)
    panels, saved = [], []
    for setting, item, net, weights, ckpt in (
        ("Layered 4×4", layered, model, None, step),
        ("Hard DAG 4×4", hard, hard_model, None, hard_step),
    ):
        layers = all_layers(net, item["tokens"], device)
        inverse = {token: index for index, token in enumerate(item["node_tokens"])}
        order = [inverse[item["node_tokens"][node]] for node in item["order"]]
        print(setting, "index", item["index"], "order", order, "step", ckpt, flush=True)
        for layer in LAYERS:
            parsed = example_matrix(layers[layer - 1], item)
            title = f"{setting}, layer {layer}\nYES {item['source']}→{item['target']}"
            panels.append({**parsed, "title": title})
            saved.append({
                "setting": setting, "layer": layer, "index": item["index"],
                "source": item["source"], "target": item["target"], "order": order,
                "labels": parsed["labels"], "matrix": parsed["matrix"].round(4).tolist(),
            })
            print(title.replace("\n", " "), parsed["labels"], flush=True)
    draw(panels)
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps({
        "columns": COLUMNS,
        "rows": "each row predicts the named node; back* leaves a node that has no outgoing edge",
        "examples": saved,
    }, indent=2))
    print(JSON_PATH)


if __name__ == "__main__":
    main()

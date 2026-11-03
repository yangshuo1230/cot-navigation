#!/usr/bin/env python3
"""DFS-state attention, one figure per record field.

In-context curriculum DFS, best checkpoint, held-out 4×4 graphs.
Candidates: layer 1 at the action token.
Stack: layer 4 at the bar that predicts the action.
Visited: layer 4 at the action token.
The schematic is one real step. The bars are the mean mass on that field.
"""
from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import plot_dfs_cot_attention as base  # noqa: E402
from cot_navigation.analysis.common import records  # noqa: E402
from cot_navigation.data.dataset import ContextStore  # noqa: E402

OUT = ROOT / "figures/scale_v1/dfs_fields"
JSON = ROOT / "studies/block_membership_probe/dfs_field_attention.json"
DEVICE = "cuda:1"
ACTIONS = ("PUSH", "POP", "FOUND", "EXHAUST")
BLUE, VERM, GREEN, GOLD = "#0072B2", "#D55E00", "#009E73", "#E69F00"
BLUE_F, VERM_F, GREEN_F, GOLD_F, GRAY_F = "#E6F1F8", "#FDE8DC", "#E5F5EF", "#FFF6DE", "#F3F3F3"
PER_ACTION = 40


def box(ax, x, y, w, h, text, face, edge, fs=8.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.08",
        facecolor=face, edgecolor=edge, linewidth=1.1, mutation_aspect=0.6, zorder=3,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="#1A1A1A", zorder=4)


def arrow(ax, start, end, color, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=9, linewidth=1.15, color=color,
        connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2, zorder=2,
    ))


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
    })


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    buf.seek(0)
    Image.open(buf).convert("RGB").save(path)
    print(path)


def axes_for(fig_w, fig_h, left, bottom, width, height):
    return [left / fig_w, bottom / fig_h, width / fig_w, height / fig_h]


def node_name(token, node_tokens):
    inverse = {value: index for index, value in enumerate(node_tokens)}
    if token in inverse:
        return f"V{inverse[token]}"
    return "PUSH"


@torch.inference_mode()
def all_layers(model, tokens, device):
    ids = torch.tensor(tokens, device=device, dtype=torch.long)[None]
    _logits, weights = model(ids, return_attn=True)
    layers = [weight[0, 0].float().cpu().numpy() for weight in weights]
    del weights
    return layers


def mass_on(layer, receiver, positions):
    row = layer[receiver, : receiver + 1]
    row = row / max(float(row.sum()), 1e-8)
    kept = [pos for pos in positions if pos <= receiver]
    return float(sum(row[pos] for pos in kept)), [(pos, float(row[pos])) for pos in kept]


def measure(model, store, tok, device):
    labels = store.arrays["label"]
    yes, no = [], []
    for index in store.indices("test").tolist():
        (yes if int(labels[index]) == 1 else no).append(int(index))
    chosen = [50520, 50599]
    chosen += [index for index in yes[:PER_ACTION] + no[:PER_ACTION] if index not in chosen]
    totals = {action: {key: [] for key in ("cands", "stack", "visited")} for action in ACTIONS}
    examples = {}
    wanted = {("PUSH", 50520), ("POP", 50520), ("FOUND", 50520), ("EXHAUST", 50599)}
    for index in chosen:
        item = base.prepare_context(store, tok, index)
        layers = all_layers(model, item["tokens"], device)
        tokens = item["tokens"]
        names = item["node_tokens"]
        seen = set()
        for rec in records(tokens, item["prompt_len"], tok):
            action = tok.name(rec["action"][0]).strip("<>")
            if action in seen or action not in totals:
                continue
            seen.add(action)
            a, b, c = rec["separators"]
            cands, stack, visited = range(b + 1, c), range(rec["start"], a), range(a + 1, b)
            totals[action]["cands"].append(mass_on(layers[0], c + 1, cands)[0])
            totals[action]["stack"].append(mass_on(layers[3], c, stack)[0])
            totals[action]["visited"].append(mass_on(layers[3], c + 1, visited)[0])
            if (action, index) in wanted and action not in examples:
                spec = {
                    "PUSH": (0, c + 1, cands, "L1 at PUSH"),
                    "POP": (3, c, stack, "L4 at the bar before POP"),
                    "FOUND": (3, c, stack, "L4 at the bar before FOUND"),
                    "EXHAUST": (3, c + 1, visited, "L4 at EXHAUST"),
                }[action]
                layer_i, receiver, positions, where = spec
                _total, pairs = mass_on(layers[layer_i], receiver, positions)
                examples[action] = {
                    "index": index,
                    "where": where,
                    "receiver": "PUSH" if action == "PUSH" else ("|" if action in {"POP", "FOUND"} else "EXHAUST"),
                    "nodes": [
                        {"name": node_name(tokens[pos], names), "mass": round(value, 3)}
                        for pos, value in pairs
                    ],
                }
                if action == "PUSH":
                    arg = node_name(rec["action"][1], names)
                    for node in examples[action]["nodes"]:
                        node["note"] = "pushed" if node["name"] == arg else ""
                if action in {"POP", "FOUND"} and examples[action]["nodes"]:
                    examples[action]["nodes"][-1]["note"] = "top"
        if all(len(totals[action]["cands"]) >= PER_ACTION for action in ACTIONS) and len(examples) == 4:
            break
    bars = {
        field: [
            {"action": action, "mass": float(np.mean(totals[action][field])), "n": len(totals[action][field])}
            for action in ACTIONS
        ]
        for field in ("cands", "stack", "visited")
    }
    return {"examples": examples, "bars": bars}


def schematic_span(example):
    w, gap = 1.08, 0.26
    receiver_w = 1.42 if len(example["receiver"]) > 3 else w
    width = 0.10 + len(example["nodes"]) * (w + gap) + receiver_w + 0.12
    return width, 2.03, w, receiver_w, gap


def paint_nodes(ax, example, receiver_face, receiver_edge):
    nodes = example["nodes"]
    width, height, w, receiver_w, gap = schematic_span(example)
    h = 0.46
    y = 0.78
    xs = []
    cursor = 0.10
    for _node in nodes:
        xs.append(cursor)
        cursor += w + gap
    rx = cursor
    peak = max(node["mass"] for node in nodes)
    for x, node in zip(xs, nodes):
        hot = node["mass"] >= 0.2 and node["mass"] >= peak - 1e-9
        face, edge = (VERM_F, VERM) if hot else (GRAY_F, "#B0B0B0")
        box(ax, x, y, w, h, node["name"], face, edge, fs=8)
        ax.text(x + w / 2, y - 0.12, f"{node['mass']:.2f}", ha="center", va="top", fontsize=7.4, color=edge)
        if node.get("note"):
            ax.text(x + w / 2, y - 0.34, node["note"], ha="center", va="top", fontsize=6.6, color="#666666")
    box(ax, rx, y, receiver_w, h, example["receiver"], receiver_face, receiver_edge, fs=7.2)
    src = (rx + receiver_w / 2, y + h)
    drawn = [(x, node) for x, node in zip(xs, nodes) if node["mass"] >= 0.15]
    for index, (x, node) in enumerate(drawn):
        rad = 0.18 + 0.12 * (len(drawn) - 1 - index)
        arrow(ax, src, (x + w * 0.7, y + h), VERM if node["mass"] >= peak - 1e-9 else "#8A8A8A", rad=rad)
    ax.set_xlim(0, width)
    ax.set_ylim(0.12, 2.15)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_title("(a)  " + example["where"], loc="left", fontsize=11, color="#1A1A1A", pad=2)
    return width, height - 0.18


def bars_panel(ax, rows, ylabel):
    x = np.arange(len(rows))
    values = [row["mass"] for row in rows]
    ax.bar(x, values, width=0.62, color=BLUE, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{row['action']}\n{row['n']}" for row in rows], fontsize=8)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.grid(True, color="#EEEEEE", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, fontsize=8.5)
    ax.set_xlabel("Action", fontsize=8.5)
    ax.set_title("(b)  Mean mass on this field", loc="left", fontsize=11, pad=2)
    for xpos, value in zip(x, values):
        if value >= 0.90:
            ax.text(xpos, value - 0.05, f"{value:.2f}", ha="center", va="top", fontsize=7.2, color="white")
        else:
            ax.text(xpos, value + 0.03, f"{value:.2f}", ha="center", va="bottom", fontsize=7.2, color=BLUE)


def draw_one(title, example, rows, ylabel, filename, receiver_face, receiver_edge):
    style()
    data_w, data_h, *_rest = schematic_span(example)
    left_h = 2.25
    left_w = left_h * data_w / data_h
    fig_w, fig_h = left_w + 6.35, 3.42
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(title, x=0.01, y=0.968, ha="left", fontsize=13, color="#1A1A1A")
    paint_nodes(
        fig.add_axes(axes_for(fig_w, fig_h, 0.06, 0.28, left_w, left_h)),
        example, receiver_face, receiver_edge,
    )
    bars_panel(
        fig.add_axes(axes_for(fig_w, fig_h, left_w + 0.85, 0.58, 5.15, 2.05)),
        rows, ylabel,
    )
    save(fig, OUT / filename)


def draw(payload):
    draw_one(
        "DFS candidates",
        payload["examples"]["PUSH"],
        payload["bars"]["cands"],
        "Layer 1 mass at the action",
        "candidates.png",
        GOLD_F, GOLD,
    )
    draw_one(
        "DFS stack",
        payload["examples"]["POP"],
        payload["bars"]["stack"],
        "Layer 4 mass at the preceding bar",
        "stack.png",
        GRAY_F, "#8A8A8A",
    )
    draw_one(
        "DFS visited",
        payload["examples"]["EXHAUST"],
        payload["bars"]["visited"],
        "Layer 4 mass at the action",
        "visited.png",
        GOLD_F, GOLD,
    )


def main():
    if JSON.exists() and "--recompute" not in sys.argv:
        payload = json.loads(JSON.read_text())
    else:
        device = torch.device(DEVICE)
        _cfg, tok, model, step = base.load_model(base.CONTEXT_RUN, device)
        store = ContextStore(base.CONTEXT_DATA)
        payload = measure(model, store, tok, device)
        payload["step"] = step
        JSON.parent.mkdir(parents=True, exist_ok=True)
        JSON.write_text(json.dumps(payload, indent=2))
        print(JSON)
    draw(payload)


if __name__ == "__main__":
    main()

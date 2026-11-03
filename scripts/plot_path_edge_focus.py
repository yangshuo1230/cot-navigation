#!/usr/bin/env python3
"""Is PATH edge-list attention on the edge being traversed, or spread out?

Each edge is one <EDGE> u v triple. At a path node, the current operation is
to write the next node, so the relevant edges leave that node and the gold
edge is the one actually taken.
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

OUT = ROOT / "figures/hard_dag_v1/path_edge_focus.png"
JSON = ROOT / "studies/block_membership_probe/path_edge_focus.json"
DEVICE = "cuda:3"
SERIES = (
    ("Edge being traversed", "gold", "#D55E00"),
    ("Heaviest edge", "top1", "#0072B2"),
    ("Five heaviest edges", "top5", "#009E73"),
)


def edge_table(item, tok):
    tokens, prompt_len = item["tokens"], item["prompt_len"]
    inverse = {token: index for index, token in enumerate(item["node_tokens"])}
    n_edges = (prompt_len - 4) // 3
    src = np.empty(n_edges, dtype=np.int64)
    dst = np.empty(n_edges, dtype=np.int64)
    for edge in range(n_edges):
        src[edge] = inverse[tokens[edge * 3 + 1]]
        dst[edge] = inverse[tokens[edge * 3 + 2]]
    return src, dst


def step_focus(attn, item) -> list[dict]:
    tokens, prompt_len = item["tokens"], item["prompt_len"]
    inverse = {token: index for index, token in enumerate(item["node_tokens"])}
    src, dst = edge_table(item, None)
    n_edges = len(src)
    out = []
    for index in range(len(item["path"]) - 1):
        receiver = prompt_len + index
        cur = inverse[tokens[receiver]]
        nxt = inverse[tokens[receiver + 1]]
        row = attn[receiver, : receiver + 1].astype(np.float64)
        row = row / max(float(row.sum()), 1e-8)
        triple = np.add.reduceat(row[: n_edges * 3], np.arange(0, n_edges * 3, 3))
        gold = (src == cur) & (dst == nxt)
        outgoing = (src == cur) & ~gold
        graph = float(triple.sum())
        weights = triple / max(graph, 1e-8)
        entropy = float(-(weights * np.log(weights + 1e-12)).sum())
        order = np.argsort(-triple)
        out.append({
            "gold": float(triple[gold].sum()),
            "outgoing": float(triple[outgoing].sum()),
            "rest": float(triple[~(src == cur)].sum()),
            "outside": float(1.0 - graph),
            "graph": graph,
            "n_edges": n_edges,
            "outdegree": int((src == cur).sum()),
            "n_eff": float(np.exp(entropy)),
            "top1": float(triple[order[0]]),
            "top5": float(triple[order[:5]].sum()),
            "first_edge": float(triple[0]),
            "last_edge": float(triple[-1]),
            "uniform_edge": 1.0 / n_edges,
        })
    return out


def summarize(rows: list[dict]) -> dict:
    def avg(key):
        return float(np.mean([row[key] for row in rows]))

    gold, outgoing, rest = avg("gold"), avg("outgoing"), avg("rest")
    graph = max(avg("graph"), 1e-8)
    outdegree = avg("outdegree")
    n_edges = avg("n_edges")
    return {
        "n": len(rows),
        "mass": {"gold": gold, "outgoing": outgoing, "rest": rest, "outside": avg("outside")},
        "graph": graph,
        "n_edges": n_edges,
        "outdegree": outdegree,
        "n_eff": avg("n_eff"),
        "gold_over_uniform": gold / (graph / n_edges),
        "outgoing_over_uniform": (gold + outgoing) / (graph * outdegree / n_edges),
        "gold_share_of_outgoing": gold / max(gold + outgoing, 1e-8),
        "top1": avg("top1"),
        "top5": avg("top5"),
        "first_edge": avg("first_edge"),
        "last_edge": avg("last_edge"),
    }


def draw(blocks: list[dict]) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "axes.linewidth": 0.6,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.2), sharey=True)
    x = np.arange(6)
    width = 0.24
    for ax, block in zip(axes, blocks):
        for offset, (label, key, color) in enumerate(SERIES):
            values = [layer[key] if key != "gold" else layer["mass"]["gold"] for layer in block["layers"]]
            ax.bar(x + (offset - 1) * width, values, width=width, color=color, label=label, linewidth=0)
        n_edges = block["layers"][-1]["n_edges"]
        ax.axhline(5.0 / n_edges, color="#888888", linestyle="--", linewidth=0.8, label="Uniform over the list")
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{i + 1}" for i in range(6)])
        ax.set_ylim(0, 1.05)
        ax.yaxis.grid(True, color="#dddddd", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_title(f"{block['size']}   ({n_edges:.0f} edges)", fontsize=11)
    axes[0].set_ylabel("Attention mass")
    axes[2].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.02))
    fig.suptitle("Edge-list attention sits on a few edges, not the one being traversed", fontsize=12)
    fig.subplots_adjust(left=0.07, right=0.78, top=0.84, bottom=0.12, wspace=0.16)
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
    _cfg, tok, model, step = base.load_model(pathfig.RUN, device)
    saved = []
    for name, root in pathfig.SIZES:
        arrays = pathfig.load_test(root)
        items = [pathfig.render_index(arrays, tok, index) for index in range(len(arrays["label"]))]
        pool = [item for item in items if 4 <= len(item["path"]) <= 8]
        chosen = []
        for label in (1, 0):
            group = sorted((item for item in pool if item["label"] == label), key=lambda item: item["index"])
            chosen.extend(group[:16])
        by_layer = [[] for _ in range(6)]
        for item in chosen:
            layers = pathfig.all_layers(model, item["tokens"], device)
            for layer, attn in enumerate(layers):
                by_layer[layer].extend(step_focus(attn, item))
            del layers
        block = {"size": name, "step": step, "layers": [summarize(rows) for rows in by_layer]}
        saved.append(block)
        print(f"\n{name} steps={block['layers'][0]['n']} edges={block['layers'][0]['n_edges']:.0f} "
              f"outdegree={block['layers'][0]['outdegree']:.2f}", flush=True)
        for index, layer in enumerate(block["layers"]):
            mass = layer["mass"]
            print(
                f"  L{index + 1}: gold {mass['gold']:.3f} other-out {mass['outgoing']:.3f} "
                f"rest {mass['rest']:.3f} outside {mass['outside']:.3f} "
                f"| neff {layer['n_eff']:.1f}/{layer['n_edges']:.0f} "
                f"top1 {layer['top1']:.3f} top5 {layer['top5']:.3f} "
                f"first {layer['first_edge']:.3f} last {layer['last_edge']:.3f} "
                f"gold× {layer['gold_over_uniform']:.1f}",
                flush=True,
            )
    draw(saved)
    JSON.parent.mkdir(parents=True, exist_ok=True)
    JSON.write_text(json.dumps(saved, indent=2))
    print(JSON)


if __name__ == "__main__":
    main()

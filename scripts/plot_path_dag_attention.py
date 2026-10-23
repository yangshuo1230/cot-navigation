#!/usr/bin/env python3
"""Attention while the in-context DAG PATH model copies a node sequence.

YES traces are the canonical DFS witness. NO traces are the variant-0 random
walk used in training, which starts at the source and stops at a sink. Rows
are the positions that predict the next path node, then the answer. One head.
"""
from __future__ import annotations

import hashlib
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
from cot_navigation.data.shards import load_npz  # noqa: E402
from cot_navigation.protocols.render import context_prompt, render_task  # noqa: E402
from cot_navigation.training.augmentation import random_path, with_body  # noqa: E402

RUN = ROOT / "outputs/hard_dag_v1/launches/hard_dag_v1_s0/context/curriculum/path"
FIG = ROOT / "figures/hard_dag_v1"
JSON = ROOT / "research/block_membership_probe_20261021/path_dag_attention.json"
DEVICE = "cuda:0"
COLUMNS = ["Query source", "Query target", "Graph", "Previous node", "Earlier path", "Self", "Other"]
SIZES = [
    ("DAG 4×4", ROOT / "data/hard_dag_v1/context_r4b4_n16_m32_train50000_s0"),
    ("DAG 6×6", ROOT / "data/hard_dag_v1/context_r6b6_n36_m72_train50000_s0"),
    ("DAG 10×10", ROOT / "data/hard_dag_v1/context_r10b10_n100_m200_train50000_s0"),
]
RECEIVERS = ("Prompt |", "Path step", "Answer")
PATH_SEED = int(hashlib.sha256(b"path").hexdigest()[:8], 16)


def load_test(root: Path):
    manifest = json.loads((root / "manifest.json").read_text())
    shard = root / "test" / manifest["splits"]["test"]["shards"][0]["file"]
    return load_npz(shard)


def render_index(arrays, tok, index: int):
    width = arrays["node_token"].shape[1]
    node_tokens = [tok.node_id(int(v)) for v in arrays["node_token"][index].tolist()]
    edges = list(zip(arrays["edge_src"][index].astype(int).tolist(), arrays["edge_dst"][index].astype(int).tolist()))
    src, dst = int(arrays["s_topo"][index]), int(arrays["t_topo"][index])
    adj = [[] for _ in range(width)]
    for u, v in edges:
        adj[u].append(v)
    prompt = context_prompt(tok, node_tokens, edges, src, dst)
    label = int(arrays["label"][index])
    row = render_task(tok, "path", label, prompt, adj, src, dst, node_tokens)
    if label == 0:
        rng = np.random.default_rng(PATH_SEED + index * 104729)
        row = with_body(row, [node_tokens[v] for v in random_path(adj, src, rng)])
    path = row.tokens[row.prompt_len:row.answer_index]
    return {
        "tokens": row.tokens,
        "prompt_len": row.prompt_len,
        "node_tokens": node_tokens,
        "source": src,
        "target": dst,
        "distance": int(arrays["distance"][index]),
        "label": label,
        "index": int(index),
        "path": path,
    }


@torch.inference_mode()
def all_layers(model, tokens, device):
    ids = torch.tensor(tokens, device=device, dtype=torch.long)[None]
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        _logits, weights = model(ids, return_attn=True)
    layers = [weight[0, 0].float().cpu().numpy() for weight in weights]
    del weights
    return layers


def topo_name(token: int, node_tokens: list[int]) -> str:
    inverse = {value: index for index, value in enumerate(node_tokens)}
    return str(inverse.get(token, "?"))


def trace_rows(attn, item, tok):
    tokens, prompt_len = item["tokens"], item["prompt_len"]
    node_set = set(item["node_tokens"])
    graph = {
        i for i in range(max(prompt_len - 4, 0))
        if tokens[i] in node_set or tok.name(tokens[i]) == "<EDGE>"
    }
    path = item["path"]
    q_src, q_dst = prompt_len - 3, prompt_len - 2
    rows, labels, kinds = [], [], []

    def masses(receiver: int) -> np.ndarray:
        row = attn[receiver, : receiver + 1]
        row = row / max(float(row.sum()), 1e-8)
        out = np.zeros(len(COLUMNS), dtype=np.float64)
        for pos in range(receiver + 1):
            if pos == receiver:
                bucket = 5
            elif pos == q_src:
                bucket = 0
            elif pos == q_dst:
                bucket = 1
            elif pos in graph:
                bucket = 2
            elif pos == receiver - 1 and pos >= prompt_len:
                bucket = 3
            elif prompt_len <= pos <= receiver - 2:
                bucket = 4
            else:
                bucket = 6
            out[bucket] += row[pos]
        return out

    rows.append(masses(prompt_len - 1))
    labels.append("Prompt |")
    kinds.append("prompt")
    for index, _token in enumerate(path):
        receiver = prompt_len + index
        nxt = receiver + 1
        if nxt < len(tokens) and tokens[nxt] in (tok.yes, tok.no):
            label, kind = "Answer", "answer"
        else:
            label, kind = f"→ {topo_name(tokens[nxt], item['node_tokens'])}", "step"
        rows.append(masses(receiver))
        labels.append(label)
        kinds.append(kind)
    return labels, kinds, np.vstack(rows)


def draw_examples(panels: list[dict]) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "axes.linewidth": 0.6,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.6))
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
    fig.subplots_adjust(left=0.07, right=0.90, top=0.88, bottom=0.16, wspace=0.42, hspace=0.55)
    colorbar = fig.add_axes((0.92, 0.28, 0.014, 0.48))
    fig.colorbar(image, cax=colorbar, label="Attention mass")
    fig.suptitle("In-context DAG PATH, last layer", fontsize=12)
    path = FIG / "path_attention_examples.png"
    _save(fig, path)
    print(path)


def draw_means(blocks: list[dict]) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "axes.linewidth": 0.6,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(3, len(RECEIVERS), figsize=(11.4, 7.4))
    image = None
    for row, block in enumerate(blocks):
        for column, name in enumerate(RECEIVERS):
            ax = axes[row, column]
            grid = block["means"][name]
            image = ax.imshow(grid, vmin=0.0, vmax=1.0, cmap="cividis", aspect="auto")
            ax.set_yticks(range(6))
            ax.set_yticklabels([f"L{i + 1}" for i in range(6)], fontsize=7)
            ax.set_xticks(range(len(COLUMNS)))
            if row == 2:
                ax.set_xticklabels(COLUMNS, rotation=55, ha="right", fontsize=7)
            else:
                ax.set_xticklabels([])
            if row == 0:
                ax.set_title(name, fontsize=10, pad=4)
            if column == 0:
                ax.set_ylabel(block["size"], fontsize=9)
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    value = grid[i, j]
                    if value < 0.12:
                        continue
                    color = "white" if value < 0.62 else "black"
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6, color=color)
    fig.subplots_adjust(left=0.07, right=0.90, top=0.88, bottom=0.16, wspace=0.28, hspace=0.22)
    colorbar = fig.add_axes((0.92, 0.24, 0.014, 0.56))
    fig.colorbar(image, cax=colorbar, label="Mean attention mass")
    fig.suptitle("In-context DAG PATH, mean attention by layer  ·  16 YES + 16 NO", fontsize=12)
    path = FIG / "path_attention_by_layer.png"
    _save(fig, path)
    print(path)


def _save(fig, path: Path) -> None:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180, facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".png.tmp")
    Image.open(buffer).convert("RGB").save(tmp, format="PNG")
    tmp.replace(path)


def main() -> None:
    device = torch.device(DEVICE)
    _cfg, tok, model, step = base.load_model(RUN, device)
    panels, saved_means, saved_examples = [], [], []
    for name, root in SIZES:
        arrays = load_test(root)
        items = [render_index(arrays, tok, index) for index in range(len(arrays["label"]))]
        pool = [item for item in items if 4 <= len(item["path"]) <= 8]
        print(f"{name} paths 4–8: {len(pool)}", flush=True)
        chosen = []
        for label in (1, 0):
            group = [item for item in pool if item["label"] == label]
            group.sort(key=lambda item: item["index"])
            chosen.extend(group[:16])
        sums = {key: np.zeros((6, len(COLUMNS))) for key in ("prompt", "step", "answer")}
        counts = {key: np.zeros(6) for key in sums}
        for item in chosen:
            layers = all_layers(model, item["tokens"], device)
            for layer_index, attn in enumerate(layers):
                _labels, kinds, matrix = trace_rows(attn, item, tok)
                grouped = {key: [] for key in sums}
                for kind, row in zip(kinds, matrix):
                    grouped[kind].append(row)
                for key, rows in grouped.items():
                    if rows:
                        sums[key][layer_index] += np.mean(rows, axis=0)
                        counts[key][layer_index] += 1
            del layers
        means = {}
        for key, title in zip(("prompt", "step", "answer"), RECEIVERS):
            grid = sums[key] / np.maximum(counts[key][:, None], 1)
            means[title] = grid
            print(f"\n{name} {title} n={int(counts[key][0])}", flush=True)
            for layer in range(6):
                order = np.argsort(-grid[layer])[:3]
                print("  L" + str(layer + 1) + ": " + ", ".join(
                    f"{COLUMNS[j]} {grid[layer, j]:.2f}" for j in order), flush=True)
        saved_means.append({"size": name, "n": len(chosen), "means": {
            title: means[title].round(4).tolist() for title in RECEIVERS
        }})
        panels_for_draw = []
        for label, row_name in ((1, "YES"), (0, "NO")):
            group = [item for item in pool if item["label"] == label]
            group.sort(key=lambda item: (abs(len(item["path"]) - 5), item["index"]))
            item = group[0]
            matrix_layers = all_layers(model, item["tokens"], device)
            labels, _kinds, matrix = trace_rows(matrix_layers[-1], item, tok)
            title = f"{name} {row_name}\n{item['source']}→{item['target']}, {len(item['path'])} nodes"
            panels.append({"title": title, "labels": labels, "matrix": matrix})
            panels_for_draw.append(title)
            saved_examples.append({
                "size": name, "step": step, "title": title, "labels": labels,
                "matrix": matrix.round(4).tolist(),
                **{key: item[key] for key in ("index", "source", "target", "distance", "label")},
            })
            del matrix_layers
        draw_block = {"size": name, "means": means}
        saved_means[-1]["_draw"] = draw_block
    draw_examples(panels)
    draw_means([row["_draw"] for row in saved_means])
    for row in saved_means:
        row.pop("_draw")
    JSON.parent.mkdir(parents=True, exist_ok=True)
    JSON.write_text(json.dumps({
        "columns": COLUMNS, "step": step, "means": saved_means, "examples": saved_examples,
    }, indent=2))
    print(JSON)


if __name__ == "__main__":
    main()

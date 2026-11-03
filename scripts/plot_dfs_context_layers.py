#!/usr/bin/env python3
"""All-layer attention for in-context curriculum DFS on 6x6, 8x8, and 10x10.

The example panels are the same short traces as the last-layer figure. Rows
add the prompt bar and the first stack token, which is where the query source
is read. A second figure averages the same buckets over the short-trace sample.
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
from cot_navigation.analysis.common import records  # noqa: E402

FIG = ROOT / "figures/scale_v1"
JSON = ROOT / "studies/block_membership_probe/dfs_context_attention_by_layer.json"
DEVICE = "cuda:0"
RECEIVERS = ("Prompt bar", "Stack start", "First PUSH", "Later PUSH", "Terminal", "Answer")


def trace_rows(attn, tokens, tok, node_tokens, prompt_len: int):
    """Prompt bar, first stack token, each action, and the answer semicolon."""
    node_set = set(node_tokens)
    graph = base.graph_positions(tokens, tok, node_tokens, prompt_len)
    parsed = records(tokens, prompt_len, tok)
    q_src, q_dst = prompt_len - 3, prompt_len - 2
    rows, labels, kinds = [], [], []

    def add(receiver: int, label: str, kind: str, record=None):
        if record is None:
            info = {
                "q_src": q_src,
                "q_dst": q_dst,
                "graph": graph,
                "cur_stack": set(),
                "cur_visited": set(),
                "cur_cands": set(),
                "earlier_nodes": {i for i in range(prompt_len, receiver) if tokens[i] in node_set},
            }
        else:
            a, b, c = record["separators"]
            earlier = range(prompt_len, record["start"])
            info = {
                "q_src": q_src,
                "q_dst": q_dst,
                "graph": graph,
                "cur_stack": set(range(record["start"], a)),
                "cur_visited": set(range(a + 1, b)),
                "cur_cands": set(range(b + 1, c)),
                "earlier_nodes": {i for i in earlier if tokens[i] in node_set},
            }
        rows.append(base.masses(attn, receiver, info, tokens))
        labels.append(label)
        kinds.append(kind)

    add(prompt_len - 1, "Prompt |", "prompt_bar")
    add(parsed[0]["start"], "Stack start", "stack_start", parsed[0])
    push_seen = 0
    for step_index, record in enumerate(parsed, start=1):
        action_pos = record["separators"][2] + 1
        action = tok.name(tokens[action_pos])
        arg = ""
        if action == "<PUSH>" and action_pos + 1 < record["end"]:
            arg = " " + base.node_index([tokens[action_pos + 1]], node_tokens)
        kind = "terminal"
        if action == "<PUSH>":
            push_seen += 1
            kind = "first_push" if push_seen == 1 else "later_push"
        add(action_pos, f"{step_index} {action[1:-1]}{arg}", kind, record)
    answer = next(i for i in range(len(tokens) - 1, prompt_len - 1, -1) if tokens[i] == tok.id(";"))
    add(answer, "Answer", "answer")
    return labels, kinds, np.vstack(rows)


def receiver_masses(attn, tokens, tok, node_tokens, prompt_len: int) -> dict[str, list[np.ndarray]]:
    _labels, kinds, matrix = trace_rows(attn, tokens, tok, node_tokens, prompt_len)
    grouped = {name: [] for name in ("prompt_bar", "stack_start", "first_push", "later_push", "terminal", "answer")}
    for kind, row in zip(kinds, matrix):
        grouped[kind].append(row)
    return grouped


def draw_examples(size: str, panels: list[dict]) -> Path:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.linewidth": 0.6,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(6, 2, figsize=(14.8, 20.4))
    image = None
    for column, example in enumerate(panels):
        for layer, (ax, grid) in enumerate(zip(axes[:, column], example["layers"])):
            image = ax.imshow(grid, vmin=0.0, vmax=1.0, cmap="cividis", aspect="auto")
            ax.set_yticks(range(len(example["labels"])))
            ax.set_yticklabels(example["labels"], fontsize=6)
            ax.set_xticks(range(len(base.COLUMNS)))
            if layer == 5:
                ax.set_xticklabels(base.COLUMNS, rotation=55, ha="right", fontsize=6.5)
            else:
                ax.set_xticklabels([])
            ax.set_title(f"L{layer + 1}", fontsize=9, pad=3, loc="left")
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    value = grid[i, j]
                    if value < 0.20:
                        continue
                    color = "white" if value < 0.62 else "black"
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=5.5, color=color)
        axes[0, column].set_title(
            f"{example['title']}\nL1", fontsize=9, pad=3, loc="center",
        )
    fig.subplots_adjust(left=0.08, right=0.90, top=0.94, bottom=0.05, wspace=0.38, hspace=0.18)
    colorbar = fig.add_axes((0.92, 0.25, 0.012, 0.50))
    fig.colorbar(image, cax=colorbar, label="Attention mass")
    fig.suptitle(f"In-context curriculum DFS on {size}, every layer", fontsize=13)
    slug = size.replace("×", "x")
    path = FIG / f"dfs_context_layers_{slug}.png"
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=160, facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".png.tmp")
    Image.open(buffer).convert("RGB").save(tmp, format="PNG")
    tmp.replace(path)
    print(path)
    return path


def draw_means(blocks: list[dict]) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.linewidth": 0.6,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(3, len(RECEIVERS), figsize=(16.8, 8.2))
    image = None
    for row, block in enumerate(blocks):
        for column, name in enumerate(RECEIVERS):
            ax = axes[row, column]
            grid = block["means"][name]
            image = ax.imshow(grid, vmin=0.0, vmax=1.0, cmap="cividis", aspect="auto")
            ax.set_yticks(range(6))
            ax.set_yticklabels([f"L{i + 1}" for i in range(6)], fontsize=7)
            ax.set_xticks(range(len(base.COLUMNS)))
            if row == 2:
                ax.set_xticklabels(base.COLUMNS, rotation=55, ha="right", fontsize=6.5)
            else:
                ax.set_xticklabels([])
            if row == 0:
                ax.set_title(name, fontsize=9, pad=4)
            if column == 0:
                ax.set_ylabel(block["size"], fontsize=9)
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    value = grid[i, j]
                    if value < 0.15:
                        continue
                    color = "white" if value < 0.62 else "black"
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=5.5, color=color)
    fig.subplots_adjust(left=0.05, right=0.91, top=0.88, bottom=0.16, wspace=0.28, hspace=0.18)
    colorbar = fig.add_axes((0.925, 0.22, 0.012, 0.58))
    fig.colorbar(image, cax=colorbar, label="Mean attention mass")
    fig.suptitle(
        "In-context DFS, mean attention by layer  ·  16 YES + 16 NO short traces",
        fontsize=12,
    )
    path = FIG / "dfs_context_layer_means.png"
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180, facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    tmp = path.with_suffix(".png.tmp")
    Image.open(buffer).convert("RGB").save(tmp, format="PNG")
    tmp.replace(path)
    print(path)


def main() -> None:
    device = torch.device(DEVICE)
    _cfg, tok, model, step = base.load_model(base.CONTEXT_RUN, device)
    saved_examples, saved_means = [], []
    mean_blocks = []
    for name, root in scale.SIZES:
        arrays = scale.load_test(root)
        pool = scale.candidates(arrays, tok)
        examples = []
        for label, row_name in ((1, "YES"), (0, "NO")):
            item = scale.pick(pool, label)
            layers = scale.all_layers(model, item["tokens"], device)
            labels = kinds = None
            matrices = []
            for attn in layers:
                labels, kinds, matrix = trace_rows(
                    attn, item["tokens"], tok, item["node_tokens"], item["prompt_len"],
                )
                matrices.append(matrix)
            title = f"{row_name} {item['source']}→{item['target']}, {item['distance']} hops"
            examples.append({"title": title, "labels": labels, "layers": matrices})
            saved_examples.append({
                "size": name,
                "step": step,
                "title": title,
                "labels": labels,
                "kinds": kinds,
                "layers": [matrix.round(4).tolist() for matrix in matrices],
                **{key: item[key] for key in ("index", "source", "target", "distance", "label")},
            })
            del layers
        draw_examples(name, examples)

        chosen = []
        for label in (1, 0):
            group = sorted((row for row in pool if row[1] == label), key=lambda row: row[3]["index"])
            chosen.extend(group[:16])
        keys = ("prompt_bar", "stack_start", "first_push", "later_push", "terminal", "answer")
        sums = {key: np.zeros((6, len(base.COLUMNS))) for key in keys}
        counts = {key: np.zeros(6) for key in keys}
        for _n, _label, _dist, item in chosen:
            layers = scale.all_layers(model, item["tokens"], device)
            for layer_index, attn in enumerate(layers):
                grouped = receiver_masses(attn, item["tokens"], tok, item["node_tokens"], item["prompt_len"])
                for key, rows in grouped.items():
                    if not rows:
                        continue
                    sums[key][layer_index] += np.mean(rows, axis=0)
                    counts[key][layer_index] += 1
            del layers
        means = {}
        block_json = {"size": name, "n": len(chosen), "receivers": {}}
        for key, title in zip(keys, RECEIVERS):
            grid = sums[key] / np.maximum(counts[key][:, None], 1)
            means[title] = grid
            block_json["receivers"][title] = {
                "n": counts[key].astype(int).tolist(),
                "layers": grid.round(4).tolist(),
            }
            print(f"\n{name} {title} n={int(counts[key][0])}", flush=True)
            for layer in range(6):
                order = np.argsort(-grid[layer])[:3]
                top = ", ".join(f"{base.COLUMNS[j]} {grid[layer, j]:.2f}" for j in order)
                print(f"  L{layer + 1}: {top}", flush=True)
        mean_blocks.append({"size": name, "means": means})
        saved_means.append(block_json)
    draw_means(mean_blocks)
    JSON.parent.mkdir(parents=True, exist_ok=True)
    JSON.write_text(json.dumps({
        "columns": base.COLUMNS,
        "means": saved_means,
        "examples": saved_examples,
    }, indent=2))
    print(JSON)


if __name__ == "__main__":
    main()

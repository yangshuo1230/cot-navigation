#!/usr/bin/env python3
"""In-context curriculum DFS attention on larger graphs.

Same last-layer action-token readout as the 4x4 examples. Also measures how
much attention lands on the query source and query target positions, including
the prompt bar and every layer.
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
from cot_navigation.analysis.common import records  # noqa: E402
from cot_navigation.data.shards import load_npz  # noqa: E402
from cot_navigation.protocols.render import context_prompt, render_task  # noqa: E402

OUT = ROOT / "figures/scale_v1/dfs_context_larger_attention.png"
JSON = ROOT / "research/block_membership_probe_20261021/dfs_context_larger_attention.json"
DEVICE = "cuda:0"
SIZES = [
    ("6×6", ROOT / "data/scale_v1/context_l6w6_n36_m54_train50000_s0"),
    ("8×8", ROOT / "data/scale_v1/context_l8w8_n64_m96_train50000_s0"),
    ("10×10", ROOT / "data/scale_v1/context_l10w10_n100_m150_train50000_s0"),
]


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
    row = render_task(tok, "dfs_state", int(arrays["label"][index]), prompt, adj, src, dst, node_tokens)
    return {
        "tokens": row.tokens,
        "prompt_len": row.prompt_len,
        "node_tokens": node_tokens,
        "source": src,
        "target": dst,
        "distance": int(arrays["distance"][index]),
        "label": int(arrays["label"][index]),
        "index": int(index),
    }


def candidates(arrays, tok):
    rows = []
    for index in range(len(arrays["label"])):
        item = render_index(arrays, tok, index)
        n = len(records(item["tokens"], item["prompt_len"], tok))
        if 5 <= n <= 9 and len(item["tokens"]) <= 1400:
            rows.append((n, item["label"], item["distance"], item))
    return rows


def pick(rows, label: int):
    pool = [row for row in rows if row[1] == label]
    pool.sort(key=lambda row: (abs(row[0] - 7), row[2], row[3]["index"]))
    return pool[0][3]


@torch.inference_mode()
def all_layers(model, tokens, device):
    ids = torch.tensor(tokens, device=device, dtype=torch.long)[None]
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        _logits, weights = model(ids, return_attn=True)
    layers = [weight[0, 0].float().cpu().numpy() for weight in weights]
    del weights
    return layers


def endpoint_mass(attn, receiver: int, q_src: int, q_dst: int) -> tuple[float, float]:
    row = attn[receiver, : receiver + 1]
    total = max(float(row.sum()), 1e-8)
    return float(row[q_src] / total), float(row[q_dst] / total) if q_dst <= receiver else 0.0


def summarize(layers, tokens, tok, prompt_len: int) -> dict:
    q_src, q_dst = prompt_len - 3, prompt_len - 2
    bar = prompt_len - 1
    action_pos = [record["separators"][2] + 1 for record in records(tokens, prompt_len, tok)]
    answer = next(i for i in range(len(tokens) - 1, prompt_len - 1, -1) if tokens[i] == tok.id(";"))
    receivers = action_pos + [answer]
    last = layers[-1]

    def pack(positions, attn):
        src, dst = [], []
        for pos in positions:
            s, d = endpoint_mass(attn, pos, q_src, q_dst)
            src.append(s)
            dst.append(d)
        return src, dst

    action_src, action_dst = pack(action_pos, last)
    answer_src, answer_dst = endpoint_mass(last, answer, q_src, q_dst)
    bar_src, bar_dst = endpoint_mass(last, bar, q_src, q_dst)
    # Strongest look at either endpoint from any CoT position, any layer.
    peak = 0.0
    for attn in layers:
        for pos in range(prompt_len, len(tokens)):
            s, d = endpoint_mass(attn, pos, q_src, q_dst)
            peak = max(peak, s, d)
    return {
        "length": len(tokens),
        "uniform": 1.0 / len(tokens),
        "action_source_mean": float(np.mean(action_src)),
        "action_target_mean": float(np.mean(action_dst)),
        "action_source_max": float(np.max(action_src)),
        "action_target_max": float(np.max(action_dst)),
        "answer_source": answer_src,
        "answer_target": answer_dst,
        "bar_source": bar_src,
        "bar_target": bar_dst,
        "peak_anywhere": peak,
        "n_actions": len(action_pos),
    }


def draw(panels: list[dict]) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.linewidth": 0.6,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.6))
    image = None
    for ax, panel in zip(axes.ravel(), panels):
        grid = panel["matrix"]
        image = ax.imshow(grid, vmin=0.0, vmax=1.0, cmap="cividis", aspect="auto")
        ax.set_xticks(range(len(base.COLUMNS)))
        ax.set_xticklabels(base.COLUMNS, rotation=55, ha="right", fontsize=7)
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
    fig.subplots_adjust(left=0.07, right=0.90, top=0.90, bottom=0.14, wspace=0.42, hspace=0.55)
    colorbar = fig.add_axes((0.92, 0.28, 0.014, 0.48))
    fig.colorbar(image, cax=colorbar, label="Attention mass")
    fig.suptitle("In-context curriculum DFS on larger graphs, last layer", fontsize=12)
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
    panels, saved, summary = [], [], []
    for name, root in SIZES:
        arrays = load_test(root)
        pool = candidates(arrays, tok)
        print(f"{name} eligible {len(pool)}", flush=True)
        # Aggregate over up to 16 YES and 16 NO, shortest first so the forward stays small.
        chosen = []
        for label in (1, 0):
            group = sorted((row for row in pool if row[1] == label), key=lambda row: row[3]["index"])
            chosen.extend(group[:16])
        stats = []
        for _n, _label, _dist, item in chosen:
            layers = all_layers(model, item["tokens"], device)
            stats.append(summarize(layers, item["tokens"], tok, item["prompt_len"]))
            del layers
        def col(key):
            values = np.array([row[key] for row in stats], dtype=np.float64)
            return {"mean": float(values.mean()), "p90": float(np.percentile(values, 90)), "max": float(values.max())}
        block = {
            "size": name,
            "n": len(stats),
            "uniform_mean": float(np.mean([row["uniform"] for row in stats])),
            "action_source": col("action_source_mean"),
            "action_target": col("action_target_mean"),
            "action_source_max": col("action_source_max"),
            "action_target_max": col("action_target_max"),
            "bar_source": col("bar_source"),
            "bar_target": col("bar_target"),
            "answer_source": col("answer_source"),
            "answer_target": col("answer_target"),
            "peak_anywhere": col("peak_anywhere"),
        }
        summary.append(block)
        print(name, json.dumps({k: block[k] for k in ("n", "uniform_mean", "action_source", "action_target", "bar_source", "bar_target", "peak_anywhere")}), flush=True)
        for label, row_name in ((1, "YES"), (0, "NO")):
            item = pick(pool, label)
            attn = all_layers(model, item["tokens"], device)[-1]
            parsed = base.example_matrix(attn, item["tokens"], tok, item["node_tokens"], item["prompt_len"])
            title = f"In-context {name}\n{row_name} {item['source']}→{item['target']}, {item['distance']} hops"
            panels.append({**parsed, "title": title})
            saved.append({"size": name, "step": step, "title": title, "labels": parsed["labels"],
                          "matrix": parsed["matrix"].round(4).tolist(),
                          **{k: item[k] for k in ("index", "source", "target", "distance", "label")}})
            del attn
    draw(panels)
    JSON.parent.mkdir(parents=True, exist_ok=True)
    JSON.write_text(json.dumps({"columns": base.COLUMNS, "summary": summary, "examples": saved}, indent=2))
    print(JSON)


if __name__ == "__main__":
    main()

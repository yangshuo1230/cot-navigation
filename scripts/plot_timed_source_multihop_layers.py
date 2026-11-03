#!/usr/bin/env python3
"""Layer readout of the multi-hop continuation of curr_source_timed_s0.

Parent: direct_learning curr_source_timed_s0 latest.pt.
Child: direct_multihop timed_source_last_joint_lr3e4 best.pt, step 110000.
Logged hop-3 test balanced accuracy is 0.804. Frozen inference only.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cot_navigation.config import load_config  # noqa: E402
from cot_navigation.models.transformer import build_model  # noqa: E402
from cot_navigation.protocols.tokenizer import Tokenizer  # noqa: E402

RUN = ROOT / "studies/direct_multihop/runs/direct_multihop_v1/timed_source_last_joint_lr3e4"
DATA = ROOT / "studies/direct_multihop/data_v1/arrays.npz"
FIG = ROOT / "figures/direct_multihop/timed_source_hop3_layers.png"
JSON = ROOT / "studies/direct_multihop/timed_source_hop3_layers.json"
DEVICE = "cuda:0"
YES, NO = 6, 7
N_LAYERS = 6


def graph(row: np.ndarray) -> dict:
    triples = row[:-4].reshape(-1, 3)
    source, target = int(row[-3]), int(row[-2])
    adj, radj = defaultdict(list), defaultdict(list)
    edges = []
    for index, (_edge, u, v) in enumerate(triples):
        rec = {
            "u": int(u),
            "v": int(v),
            "body": [3 * index, 3 * index + 1, 3 * index + 2],
            "follower": 3 * index + 3,
        }
        edges.append(rec)
        adj[rec["u"]].append(rec)
        radj[rec["v"]].append(rec)

    def walk(start, neighbors, field):
        seen, stack = {start}, [start]
        while stack:
            node = stack.pop()
            for rec in neighbors[node]:
                nxt = rec[field]
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen

    forward, backward = walk(source, adj, "v"), walk(target, radj, "u")
    dist = {source: 0}
    queue = [source]
    for node in queue:
        for rec in adj[node]:
            if rec["v"] not in dist:
                dist[rec["v"]] = dist[node] + 1
                queue.append(rec["v"])
    hops = {1: [], 2: [], 3: []}
    off = []
    for rec in edges:
        if rec["u"] in dist and rec["v"] in backward and dist[rec["u"]] + 1 == dist.get(rec["v"], -1):
            hop = dist[rec["u"]] + 1
            if hop in hops:
                hops[hop].append(rec)
            else:
                off.append(rec)
        else:
            off.append(rec)
    return {"source": source, "target": target, "hops": hops, "off": off, "edges": edges, "dist": dist}


def mass_at(row_attn: np.ndarray, positions: list[int]) -> float:
    if not positions:
        return 0.0
    return float(row_attn[positions].sum())


def balanced(pred: np.ndarray, labels: np.ndarray) -> dict:
    yes, no = labels == 1, labels == 0
    yes_acc = float((pred[yes] == YES).mean())
    no_acc = float((pred[no] == NO).mean())
    return {"yes": yes_acc, "no": no_acc, "bacc": 0.5 * (yes_acc + no_acc)}


@torch.inference_mode()
def forward(model, tokens: torch.Tensor):
    capture: dict = {}
    logits, weights = model(tokens, return_attn=True, capture=capture)
    lens = [model.lm_head(model.final_norm(hidden))[:, -1].float() for hidden in capture["resid"]]
    bar = torch.stack([w[:, 0, -1].float() for w in weights], dim=1)
    full = torch.stack([w[:, 0].float() for w in weights], dim=1)
    return logits[:, -1].float(), lens, bar, full


def hide_predict(model, tokens: torch.Tensor, hides, layers: set[int]) -> np.ndarray:
    originals = []

    def wrap(module):
        original = module.forward

        def fwd(x, positions, cache=None, layer_idx=0, cache_mode=None, return_attn=False, record_keep=None):
            batch, length, _ = x.shape
            index = torch.arange(length, device=x.device)
            keep = (index.view(1, -1) <= index.view(-1, 1)).view(1, 1, length, length).expand(batch, 1, length, length).clone()
            if layer_idx in layers:
                for b, keys in enumerate(hides):
                    if len(keys) == 0:
                        continue
                    keep[b, 0, length - 1, torch.as_tensor(keys, device=x.device)] = False
            eye = torch.arange(length, device=x.device)
            keep[:, 0, eye, eye] = True
            return original(x, positions, cache, layer_idx, cache_mode, True, keep)

        module.forward = fwd
        originals.append((module, original))

    for block in model.blocks:
        wrap(block.attn)
    try:
        with torch.inference_mode():
            logits, _weights = model(tokens, return_attn=True)
        return logits[:, -1].float().argmax(-1).cpu().numpy()
    finally:
        for module, original in originals:
            module.forward = original


def followers(records: list[dict], length: int) -> list[int]:
    return [rec["follower"] for rec in records if rec["follower"] < length]


def draw(payload: dict) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "0.9",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
    })
    blue, orange, green, chance, vermillion = "#0072B2", "#E69F00", "#009E73", "0.45", "#D55E00"
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.7), constrained_layout=True)
    layers = np.arange(1, N_LAYERS + 1)

    ax = axes[0]
    for hop, color, marker in ((1, blue, "o"), (2, orange, "s"), (3, green, "^")):
        ax.plot(layers, payload["lens"][str(hop)], color=color, marker=marker, label=f"{hop}-hop")
    ax.axhline(0.5, color=chance, ls="--", lw=1, label="Chance")
    ax.set_ylim(0, 1.02)
    ax.set_xticks(layers)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Logit lens at |")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    ax.plot(layers, payload["endpoint"]["source_out"], color=blue, marker="o", label="Source on its outgoing edges")
    ax.plot(layers, payload["endpoint"]["target_in"], color=green, marker="^", label="Target on its incoming edges")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(layers)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Attention mass")
    ax.set_title("3-hop YES, at the query endpoints")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)

    ax = axes[2]
    names = ["Baseline", "Source misses\nits out-edges", "Target misses\nits in-edges", "Bar misses\nthe edge list"]
    keys = ("baseline", "source_out", "target_in", "bar_edges")
    yes_rec = [payload["ablation"][k]["yes"] for k in keys]
    no_rec = [payload["ablation"][k]["no"] for k in keys]
    xpos = np.arange(len(names))
    ax.bar(xpos - 0.16, yes_rec, width=0.32, color=blue, label="YES recall")
    ax.bar(xpos + 0.16, no_rec, width=0.32, color=vermillion, label="NO recall")
    ax.axhline(0.5, color=chance, ls="--", lw=1)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(xpos)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Recall")
    ax.set_title("Block layers 1–4")
    ax.legend(frameon=False, fontsize=8)
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
    cfg = load_config([RUN / "resolved_config.yaml"])
    tok = Tokenizer.from_config(cfg)
    assert tok.yes == YES and tok.no == NO
    model = build_model(cfg.model, len(tok), tok.pad).to(device).eval()
    state = torch.load(RUN / "checkpoints/best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    step = int(state["step"])
    del state
    print("step", step, flush=True)

    blob = np.load(DATA)
    tokens_np = blob["test_x"].astype(np.int64)
    labels = blob["test_y"].astype(np.int64)
    hops = blob["test_hop"].astype(np.int64)
    lens, finals, bars, fulls = {}, {}, {}, {}
    for hop in (1, 2, 3):
        chosen = np.flatnonzero(hops == hop)
        tokens = torch.tensor(tokens_np[chosen], device=device)
        final, layer_logits, bar, full = forward(model, tokens)
        pred = final.argmax(-1).cpu().numpy()
        finals[hop] = balanced(pred, labels[chosen])
        lens[hop] = []
        for scores in layer_logits:
            lens[hop].append(balanced(scores.argmax(-1).cpu().numpy(), labels[chosen])["bacc"])
        bars[hop] = (chosen, bar.cpu().numpy())
        fulls[hop] = (chosen, full.cpu().numpy())
        print(f"hop {hop} final", finals[hop], "lens", ["%.3f" % v for v in lens[hop]], flush=True)
        del tokens, final, layer_logits, bar, full

    chosen, bar = bars[3]
    _idx, full = fulls[3]
    y = labels[chosen]
    info = [graph(tokens_np[i]) for i in chosen]
    length = tokens_np.shape[1]
    source_pos, target_pos = length - 3, length - 2
    endpoint = {"source_out": [], "target_in": [], "bar_source": [], "bar_target": []}
    for layer in range(N_LAYERS):
        src_m, tgt_m, bar_s, bar_t = [], [], [], []
        for row_i, rec in enumerate(info):
            if y[row_i] != 1:
                continue
            plane = full[row_i, layer]
            src_keys, tgt_keys = [], []
            for edge in rec["edges"]:
                if edge["u"] == rec["source"]:
                    src_keys.extend(edge["body"])
                if edge["v"] == rec["target"]:
                    tgt_keys.extend(edge["body"])
            src_m.append(mass_at(plane[source_pos], src_keys))
            tgt_m.append(mass_at(plane[target_pos], tgt_keys))
            bar_s.append(float(bar[row_i, layer, source_pos]))
            bar_t.append(float(bar[row_i, layer, target_pos]))
        endpoint["source_out"].append(float(np.mean(src_m)))
        endpoint["target_in"].append(float(np.mean(tgt_m)))
        endpoint["bar_source"].append(float(np.mean(bar_s)))
        endpoint["bar_target"].append(float(np.mean(bar_t)))
    print("endpoint", {k: ["%.3f" % v for v in endpoint[k]] for k in endpoint}, flush=True)

    example = None
    for row_i, index in enumerate(chosen):
        rec = info[row_i]
        if y[row_i] != 1 or not all(len(rec["hops"][hop]) == 1 for hop in (1, 2, 3)):
            continue
        path = [rec["source"]]
        for hop in (1, 2, 3):
            path.append(rec["hops"][hop][0]["v"])
        source_on_u = float(full[row_i, 2, source_pos, rec["hops"][1][0]["body"][1]])
        target_on_v = float(full[row_i, 2, target_pos, rec["hops"][3][0]["body"][2]])
        example = {
            "index": int(index),
            "path": [int(node) - 14 for node in path],
            "l3_source_on_out_source_slot": source_on_u,
            "l3_target_on_in_dest_slot": target_on_v,
        }
        break
    print("example", example, flush=True)
    del full

    def block(query_positions, key_fn, layers):
        originals = []
        hides = [key_fn(rec) for rec in info]

        def wrap(module):
            original = module.forward

            def fwd(x, positions, cache=None, layer_idx=0, cache_mode=None, return_attn=False, record_keep=None):
                batch, seq, _ = x.shape
                index = torch.arange(seq, device=x.device)
                keep = (index.view(1, -1) <= index.view(-1, 1)).view(1, 1, seq, seq).expand(batch, 1, seq, seq).clone()
                if layer_idx in layers:
                    for b, keys in enumerate(hides):
                        if not keys:
                            continue
                        key_index = torch.as_tensor(keys, device=x.device)
                        for query in query_positions:
                            keep[b, 0, query, key_index] = False
                eye = torch.arange(seq, device=x.device)
                keep[:, 0, eye, eye] = True
                return original(x, positions, cache, layer_idx, cache_mode, True, keep)

            module.forward = fwd
            originals.append((module, original))

        for block_module in model.blocks:
            wrap(block_module.attn)
        try:
            with torch.inference_mode():
                logits, _weights = model(torch.tensor(tokens_np[chosen], device=device), return_attn=True)
            return balanced(logits[:, -1].float().argmax(-1).cpu().numpy(), y)
        finally:
            for module, original in originals:
                module.forward = original

    def outgoing(rec):
        keys = []
        for edge in rec["edges"]:
            if edge["u"] == rec["source"]:
                keys.extend(edge["body"])
        return keys

    def incoming(rec):
        keys = []
        for edge in rec["edges"]:
            if edge["v"] == rec["target"]:
                keys.extend(edge["body"])
        return keys

    early = set(range(4))
    ablation = {
        "baseline": block([], lambda rec: [], early),
        "source_out": block([source_pos], outgoing, early),
        "target_in": block([target_pos], incoming, early),
        "bar_edges": block([length - 1], lambda rec: list(range(length - 4)), early),
    }
    for key, stats in ablation.items():
        print("ablation", key, stats, flush=True)

    payload = {
        "checkpoint": str(RUN / "checkpoints/best.pt"),
        "step": step,
        "parent": "curr_source_timed_s0/checkpoints/latest.pt",
        "final": {str(hop): finals[hop] for hop in finals},
        "lens": {str(hop): lens[hop] for hop in lens},
        "endpoint": endpoint,
        "ablation": ablation,
        "example": example,
    }
    draw(payload)
    JSON.parent.mkdir(parents=True, exist_ok=True)
    JSON.write_text(json.dumps(payload, indent=2))
    print(FIG)


if __name__ == "__main__":
    main()

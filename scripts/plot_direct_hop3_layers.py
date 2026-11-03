#!/usr/bin/env python3
"""Layer-wise Direct computation on 3-hop queries.

In-context hop curriculum never leaves chance on held-out graphs. The
curriculum Direct that reaches ~80% at hop 3 is the matched fixed DAG.
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
from cot_navigation.data.shards import load_npz  # noqa: E402
from cot_navigation.protocols.dfs import dfs_search, witness_path  # noqa: E402
from cot_navigation.protocols.render import context_prompt, parametric_prompt  # noqa: E402

CTX_RUN = ROOT / "outputs/explore_v1/launches/explore_v1_s012/context/standard/train50000/direct_curriculum/direct"
CTX_DATA = ROOT / "data/explore_v1/context_standard_n16_m24_train50000_val10000_test10000_s0"
FIX_RUN = ROOT / "outputs/explore_v1/launches/explore_v1_s012/fixed/standard/seed0/direct_curriculum/direct"
FIG = ROOT / "figures/explore_v1/direct_curriculum_hop3_layers.png"
JSON = ROOT / "studies/block_membership_probe/direct_curriculum_hop3_layers.json"
DEVICE = "cuda:0"
N_PER_CLASS = 64


def load_test(root: Path, hop: int = 3):
    manifest = json.loads((root / "manifest.json").read_text())
    blobs = []
    for spec in manifest["splits"]["test"]["shards"]:
        arrays = load_npz(root / "test" / spec["file"])
        if int(arrays["distance"][0]) != hop:
            continue
        blobs.append(arrays)
    if not blobs:
        raise RuntimeError(f"no test shard with hop {hop}")
    keys = blobs[0].keys()
    return {key: np.concatenate([blob[key] for blob in blobs], axis=0) for key in keys}


def adj_of(arrays, index, width):
    adj = [[] for _ in range(width)]
    for u, v in zip(arrays["edge_src"][index].astype(int).tolist(), arrays["edge_dst"][index].astype(int).tolist()):
        adj[u].append(v)
    return adj


def pick_hop3(arrays, per_class=N_PER_CLASS):
    dist = arrays["distance"].astype(int)
    label = arrays["label"].astype(int)
    chosen = []
    for y in (1, 0):
        idx = np.flatnonzero((dist == 3) & (label == y))
        chosen.extend(idx[:per_class].tolist())
    return chosen


@torch.inference_mode()
def forward(model, tokens, device):
    ids = torch.tensor(tokens, device=device, dtype=torch.long)[None]
    capture = {}
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits, weights = model(ids, capture=capture, return_attn=True)
    bar = len(tokens) - 1
    layers_attn = [w[0, 0, bar, : bar + 1].float().cpu().numpy() for w in weights]
    resid = [h[0, bar].float() for h in capture["resid"]]
    lens = []
    for h in resid:
        scores = model.lm_head(model.final_norm(h[None, None])).float().cpu()[0, 0]
        lens.append(scores)
    pred = int(logits[0, bar].argmax())
    del weights, capture
    return layers_attn, lens, pred


def ctx_item(arrays, tok, index):
    width = arrays["node_token"].shape[1]
    nodes = [tok.node_id(int(v)) for v in arrays["node_token"][index].tolist()]
    edges = list(zip(arrays["edge_src"][index].astype(int).tolist(), arrays["edge_dst"][index].astype(int).tolist()))
    src, dst = int(arrays["s_topo"][index]), int(arrays["t_topo"][index])
    prompt = context_prompt(tok, nodes, edges, src, dst)
    adj = adj_of(arrays, index, width)
    path = witness_path(dfs_search(adj, src, dst)) or []
    return {
        "tokens": prompt,
        "prompt_len": len(prompt),
        "nodes": nodes,
        "src": src,
        "dst": dst,
        "label": int(arrays["label"][index]),
        "path": path,
        "edges": edges,
    }


def ctx_buckets(attn, item, tok):
    tokens, pl = item["tokens"], item["prompt_len"]
    row = attn.astype(np.float64)
    row = row / max(float(row.sum()), 1e-8)
    n = (pl - 4) // 3
    src, dst, path = item["src"], item["dst"], item["path"]
    inverse = {token: i for i, token in enumerate(item["nodes"])}
    gold = set(zip(path, path[1:])) if len(path) >= 2 else set()
    mass = {
        "query_src": float(row[pl - 3]),
        "query_tgt": float(row[pl - 2]),
        "self": float(row[-1]),
        "gold_edges": 0.0,
        "src_out": 0.0,
        "tgt_in": 0.0,
        "other_edges": 0.0,
    }
    for e in range(n):
        u = inverse[tokens[e * 3 + 1]]
        v = inverse[tokens[e * 3 + 2]]
        triple = float(row[e * 3: e * 3 + 3].sum())
        if (u, v) in gold:
            mass["gold_edges"] += triple
        elif u == src:
            mass["src_out"] += triple
        elif v == dst:
            mass["tgt_in"] += triple
        else:
            mass["other_edges"] += triple
    return mass


def summarize_mass(rows):
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def yes_no(tok, scores):
    return float(scores[tok.yes] - scores[tok.no])


def draw(payload):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "axes.linewidth": 0.6,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.1))
    x = np.arange(6)
    ctx = payload["context"]
    acc = np.array(ctx["layer_acc"])
    chance = np.full(6, 0.5)
    axes[0].plot(x + 1, acc, color="#D55E00", marker="o", linewidth=1.6, label="In-context curriculum")
    axes[0].plot(x + 1, chance, color="#888888", linestyle="--", linewidth=0.8, label="Chance")
    axes[0].axhline(0.8, color="#0072B2", linestyle=":", linewidth=0.8, label="80%")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xticks(x + 1)
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("3-hop balanced accuracy")
    axes[0].set_title("Logit lens at |")
    axes[0].yaxis.grid(True, color="#dddddd", linewidth=0.6)
    axes[0].set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=8)

    parts = [
        ("Gold 3-hop edges", "gold_edges", "#D55E00"),
        ("Other edges from the source", "src_out", "#E69F00"),
        ("Other edges into the target", "tgt_in", "#0072B2"),
        ("Rest of the edge list", "other_edges", "#56B4E9"),
        ("Query source / target", "query", "#009E73"),
        ("The bar itself", "self", "#999999"),
    ]
    bottom = np.zeros(6)
    for label, key, color in parts:
        if key == "query":
            vals = np.array([m["query_src"] + m["query_tgt"] for m in ctx["attn"]])
        else:
            vals = np.array([m[key] for m in ctx["attn"]])
        axes[1].bar(x + 1, vals, bottom=bottom, width=0.72, color=color, label=label, linewidth=0)
        bottom += vals
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xticks(x + 1)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Attention mass")
    axes[1].set_title("In-context bar attention")
    axes[1].yaxis.grid(True, color="#dddddd", linewidth=0.6)
    axes[1].set_axisbelow(True)
    axes[1].legend(frameon=False, fontsize=7)

    fx = payload["fixed"]
    axes[2].plot(x + 1, fx["layer_acc"], color="#0072B2", marker="s", linewidth=1.6, label="Fixed curriculum")
    axes[2].plot(x + 1, chance, color="#888888", linestyle="--", linewidth=0.8)
    axes[2].axhline(0.8, color="#D55E00", linestyle=":", linewidth=0.8)
    axes[2].set_ylim(0, 1.05)
    axes[2].set_xticks(x + 1)
    axes[2].set_xlabel("Layer")
    axes[2].set_title("Fixed DAG, hop 3")
    axes[2].yaxis.grid(True, color="#dddddd", linewidth=0.6)
    axes[2].set_axisbelow(True)
    axes[2].legend(frameon=False, fontsize=8)
    fig.suptitle("Hop-curriculum Direct on 3-hop queries", fontsize=12)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.16, wspace=0.32)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    tmp = FIG.with_suffix(".png.tmp")
    Image.open(buf).convert("RGB").save(tmp, format="PNG")
    tmp.replace(FIG)
    print(FIG)


def run_context(tok, model, device):
    arrays = load_test(CTX_DATA)
    idx = pick_hop3(arrays)
    yes_id, no_id = tok.yes, tok.no
    attn_rows = [[] for _ in range(6)]
    pred_y = [[] for _ in range(6)]
    pred_n = [[] for _ in range(6)]
    last = []
    for index in idx:
        item = ctx_item(arrays, tok, index)
        attns, lens, pred = forward(model, item["tokens"], device)
        last.append((item["label"], pred == (yes_id if item["label"] == 1 else no_id)))
        for layer, (attn, scores) in enumerate(zip(attns, lens)):
            attn_rows[layer].append(ctx_buckets(attn, item, tok))
            pick = int(scores.argmax())
            correct = pick == (yes_id if item["label"] == 1 else no_id)
            (pred_y if item["label"] == 1 else pred_n)[layer].append(float(correct))
    def bacc(layer):
        return 0.5 * (float(np.mean(pred_y[layer])) + float(np.mean(pred_n[layer])))
    yes_last = [c for y, c in last if y == 1]
    no_last = [c for y, c in last if y == 0]
    return {
        "n": len(idx),
        "layer_acc": [bacc(layer) for layer in range(6)],
        "attn": [summarize_mass(rows) for rows in attn_rows],
        "final_bacc": 0.5 * (float(np.mean(yes_last)) + float(np.mean(no_last))),
        "yes_acc": float(np.mean(yes_last)),
        "no_acc": float(np.mean(no_last)),
    }


def run_fixed(cfg, tok, model, device):
    from cot_navigation.data.dataset import ParametricWorld
    world = Path(cfg.data.root)
    if not world.is_absolute():
        world = ROOT / world
    seed = cfg.data.graph_seed
    shape = cfg.data.graph_shape or "balanced"
    candidate = world / f"{shape}-seed{seed}"
    if (candidate / "world.json").exists() or (candidate / "graph.npz").exists():
        world = candidate
    store = ParametricWorld(world, include_queries=True)
    nodes = store.node_vocab_ids(tok)
    q = store.queries
    mask = (q["distance"] == 3) & (q["split"] == 2)
    ids = np.flatnonzero(mask)
    src, tgt, lab = q["s_topo"][ids].astype(int), q["t_topo"][ids].astype(int), q["label"][ids].astype(int)
    chosen = []
    for y in (1, 0):
        idx = np.flatnonzero(lab == y)[:N_PER_CLASS]
        chosen.extend(idx.tolist())
    pred_y = [[] for _ in range(6)]
    pred_n = [[] for _ in range(6)]
    attn_mean = [[] for _ in range(6)]
    last = []
    for i in chosen:
        tokens = parametric_prompt(tok, nodes, int(src[i]), int(tgt[i]))
        attns, lens, pred = forward(model, tokens, device)
        y = int(lab[i])
        last.append((y, pred == (tok.yes if y == 1 else tok.no)))
        for layer, (attn, scores) in enumerate(zip(attns, lens)):
            row = attn.astype(np.float64)
            row = row / max(float(row.sum()), 1e-8)
            attn_mean[layer].append({
                "marker": float(row[0]),
                "query_src": float(row[1]),
                "query_tgt": float(row[2]),
                "self": float(row[3]),
            })
            pick = int(scores.argmax())
            correct = pick == (tok.yes if y == 1 else tok.no)
            (pred_y if y == 1 else pred_n)[layer].append(float(correct))
    def bacc(layer):
        return 0.5 * (float(np.mean(pred_y[layer])) + float(np.mean(pred_n[layer])))
    yes_last = [c for y, c in last if y == 1]
    no_last = [c for y, c in last if y == 0]
    return {
        "n": len(chosen),
        "layer_acc": [bacc(layer) for layer in range(6)],
        "attn": [summarize_mass(rows) for rows in attn_mean],
        "final_bacc": 0.5 * (float(np.mean(yes_last)) + float(np.mean(no_last))),
        "yes_acc": float(np.mean(yes_last)),
        "no_acc": float(np.mean(no_last)),
    }


def main():
    device = torch.device(DEVICE)
    cfg, tok, model, step = base.load_model(CTX_RUN, device)
    print("context step", step, flush=True)
    ctx = run_context(tok, model, device)
    print("context hop3 bacc", ctx["final_bacc"], "yes", ctx["yes_acc"], "no", ctx["no_acc"], flush=True)
    print("context layers", ["%.3f" % v for v in ctx["layer_acc"]], flush=True)
    for i, m in enumerate(ctx["attn"]):
        print(f"  L{i+1} gold {m['gold_edges']:.3f} src_out {m['src_out']:.3f} tgt_in {m['tgt_in']:.3f} "
              f"rest {m['other_edges']:.3f} qsrc {m['query_src']:.3f} qtgt {m['query_tgt']:.3f} self {m['self']:.3f}", flush=True)
    del model
    torch.cuda.empty_cache()
    cfg, tok, model, step = base.load_model(FIX_RUN, device)
    print("fixed step", step, flush=True)
    fx = run_fixed(cfg, tok, model, device)
    print("fixed hop3 bacc", fx["final_bacc"], "yes", fx["yes_acc"], "no", fx["no_acc"], flush=True)
    print("fixed layers", ["%.3f" % v for v in fx["layer_acc"]], flush=True)
    for i, m in enumerate(fx["attn"]):
        print(f"  L{i+1} QUERY {m['marker']:.3f} qsrc {m['query_src']:.3f} qtgt {m['query_tgt']:.3f} self {m['self']:.3f}", flush=True)
    payload = {"context": ctx, "fixed": fx, "context_step": int(step)}
    draw(payload)
    JSON.parent.mkdir(parents=True, exist_ok=True)
    JSON.write_text(json.dumps(payload, indent=2))
    print(JSON)


if __name__ == "__main__":
    main()

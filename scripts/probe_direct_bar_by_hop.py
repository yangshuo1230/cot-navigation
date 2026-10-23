#!/usr/bin/env python3
"""Balanced accuracy of a linear probe at the Direct bar token, by layer and hop.

The bar token is where Direct predicts YES or NO. Each transformer layer is
probed on its own, and each hop is a separate probe on every stored query of
that layer gap. Chance for balanced accuracy is 0.5.
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

import probe_block_membership as base  # noqa: E402
import probe_direct_reachability_layers as layers  # noqa: E402

OUT = ROOT / "research/block_membership_probe_20261021/direct_bar_by_layer_hop.json"
FIG = ROOT / "figures/scale_v1/direct_bar_probe_by_layer_hop.png"
LINE = ROOT / "figures/scale_v1/direct_bar_probe_by_hop.png"
DEVICE = "cuda:0"
SERIES = [
    ("untrained", "Untrained", "#666666", "o", "--"),
    ("static", "Static 10×10", "#0072B2", "o", "-"),
    ("curriculum", "Curriculum", "#009E73", "s", "-"),
    ("within_only", "Within-block train", "#E69F00", "^", "-"),
    ("cross_allowed", "Cross-block train", "#D55E00", "D", "-"),
]


@torch.inference_mode()
def collect_bar(run: Path, device: torch.device, *, trained: bool) -> dict:
    cfg = base.load_config([run / "resolved_config.yaml"])
    if not trained:
        torch.manual_seed(cfg.train.model_seed)
        np.random.seed(cfg.train.model_seed)
    tok = base.Tokenizer.from_config(cfg)
    world = base.world_dir(cfg)
    store = base.ParametricWorld(world)
    token = base.world_token(cfg, tok, world)
    nodes = store.node_vocab_ids(tok)
    queries = store.queries
    mask = queries["distance"] >= 1
    src = queries["s_topo"][mask].astype(np.int64)
    tgt = queries["t_topo"][mask].astype(np.int64)
    label = queries["label"][mask].astype(np.int64)
    dist = queries["distance"][mask].astype(np.int64)
    prompts = [[token, tok.id("<QUERY>"), nodes[int(s)], nodes[int(t)], tok.id("|")]
               for s, t in zip(src, tgt)]
    model = base.build_model(cfg.model, len(tok), tok.pad).to(device).eval()
    step = 0
    if trained:
        state = torch.load(run / "checkpoints/best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        step = int(state["step"])
        del state
    bars = []
    for start in range(0, len(prompts), 256):
        ids = torch.tensor(prompts[start:start + 256], device=device, dtype=torch.long)
        capture: dict = {}
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            model(ids, capture=capture)
        bars.append(torch.stack([h[:, 4].float() for h in capture["resid"]], dim=1).cpu())
        del capture
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"step": step, "label": label, "distance": dist, "bars": torch.cat(bars, 0).numpy()}


def probe_hop(features: np.ndarray, labels: np.ndarray) -> dict:
    aucs, bals, accs = [], [], []
    for seed in base.SEEDS:
        train, val, test = base.split_by_class(labels, seed)
        if min(len(train), len(val), len(test)) == 0:
            continue
        if min(len(np.unique(labels[train])), len(np.unique(labels[val])), len(np.unique(labels[test]))) < 2:
            continue
        score = base.fit_scores(features[train], labels[train], features[val], labels[val], features[test], seed)
        held = labels[test]
        aucs.append(base.auc(held, score))
        bals.append(base.balanced_accuracy(held, score))
        accs.append(float(np.mean((score >= 0) == (held == 1))))
    return {
        "balanced_accuracy": float(np.mean(bals)),
        "balanced_accuracy_std": float(np.std(bals)),
        "accuracy": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "auc": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "n": int(len(labels)),
        "n_pos": int(np.sum(labels == 1)),
    }


def evaluate(pack: dict) -> dict:
    names = [f"L{index + 1}" for index in range(pack["bars"].shape[1])]
    by_hop = {}
    for hop in layers.HOPS:
        chosen = pack["distance"] == hop
        y = pack["label"][chosen]
        by_hop[str(hop)] = {
            name: probe_hop(pack["bars"][chosen, index], y)
            for index, name in enumerate(names)
        }
    return {"step": pack["step"], "by_hop": by_hop}


def draw(results: dict) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.2,
        "ytick.major.size": 3.2,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    hops = np.asarray(layers.HOPS, dtype=float)
    for key, label, color, marker, linestyle in SERIES:
        ys = np.asarray([results[key]["by_hop"][str(hop)]["L6"]["balanced_accuracy"] for hop in layers.HOPS])
        ax.plot(
            hops, ys, color=color, marker=marker, linestyle=linestyle,
            linewidth=1.6, markersize=5.5,
            markerfacecolor="white" if linestyle == "--" else color,
            markeredgecolor=color, markeredgewidth=1.0, label=label, zorder=3,
        )
    ax.axhline(0.5, color="#B3B3B3", lw=0.8, ls=(0, (3, 2)), zorder=0)
    ax.set_ylim(0.38, 1.03)
    ax.set_xticks(layers.HOPS)
    ax.set_xlim(0.7, 9.3)
    ax.yaxis.grid(True, color="#EEEEEE", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.set_title("Direct linear probe at |, fixed 10×10", fontsize=11, pad=8)
    ax.set_xlabel("Query hops", fontsize=10)
    ax.set_ylabel("Balanced accuracy", fontsize=10)
    ax.legend(
        frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5),
        handlelength=2.4, borderaxespad=0.2,
    )
    fig.subplots_adjust(left=0.10, right=0.70, top=0.90, bottom=0.14)
    buffer = BytesIO()
    save_png(fig, buffer, LINE)


def save_png(fig, buffer: BytesIO, path: Path) -> None:
    fig.savefig(buffer, format="png", dpi=200, facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".png.tmp")
    Image.open(buffer).convert("RGB").save(tmp, format="PNG")
    tmp.replace(path)
    print(path)


def draw_layers(results: dict) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.linewidth": 0.6,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })
    columns = [f"L{index}" for index in range(1, 7)]
    fig = plt.figure(figsize=(12.8, 8.2))
    grid = fig.add_gridspec(2, 6, left=0.05, right=0.90, top=0.90, bottom=0.06, wspace=0.85, hspace=0.38)
    slots = [(0, 0), (0, 2), (0, 4), (1, 1), (1, 3)]
    image = None
    for (key, title, *_rest), (row, col) in zip(SERIES, slots):
        ax = fig.add_subplot(grid[row, col:col + 2])
        values = np.array([
            [results[key]["by_hop"][str(hop)][name]["balanced_accuracy"] for name in columns]
            for hop in layers.HOPS
        ])
        image = ax.imshow(values, vmin=0.40, vmax=1.0, cmap="cividis", aspect="auto")
        ax.set_xticks(range(len(columns)))
        ax.set_xticklabels(columns, fontsize=8)
        ax.set_yticks(range(len(layers.HOPS)))
        ax.set_yticklabels([str(hop) for hop in layers.HOPS], fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Query hops")
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = values[i, j]
                color = "white" if value < 0.72 else "black"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.5, color=color)
    colorbar = fig.add_axes((0.92, 0.18, 0.014, 0.64))
    fig.colorbar(image, cax=colorbar, label="Balanced accuracy")
    fig.suptitle("Direct probe at | by layer and hop, fixed 10×10", fontsize=12)
    save_png(fig, BytesIO(), FIG)


def main() -> None:
    base.DEVICE = DEVICE
    device = torch.device(DEVICE)
    runs = {setting: run for setting, run in layers.direct_runs()}
    results = {}
    print("probe untrained", flush=True)
    results["untrained"] = evaluate(collect_bar(runs["static"], device, trained=False))
    for key, *_rest in SERIES:
        if key == "untrained":
            continue
        print(f"probe {key}", flush=True)
        pack = collect_bar(runs[key], device, trained=True)
        results[key] = evaluate(pack)
        print(f"  step {results[key]['step']}", flush=True)
        del pack
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "position": "residual at | after each transformer layer",
        "metric": "balanced accuracy of a linear probe on the probe's held-out queries",
        "chance": 0.5,
        "hops": "each hop is probed separately on all stored queries of that distance",
        "results": results,
    }, indent=2))
    draw(results)
    draw_layers(results)
    for key, label, *_rest in SERIES:
        print(label)
        for hop in layers.HOPS:
            row = results[key]["by_hop"][str(hop)]
            text = " ".join(f"{name} {row[name]['balanced_accuracy']:.3f}" for name in row)
            print(f"  hop {hop}: {text}")
    print(OUT)


if __name__ == "__main__":
    main()

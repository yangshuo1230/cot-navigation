#!/usr/bin/env python3
"""Reachability readout by transformer layer for fixed 10x10 Direct models.

The decision state is the residual at the bar token, which is where Direct
predicts YES or NO. Input is the concatenation of the source and target
embeddings. Each hop is probed on its own, so a short hop can be readable
before a long one.
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

OUT = ROOT / "studies/block_membership_probe/direct_reachability_layers.json"
FIG = ROOT / "figures/scale_v1/direct_reachability_by_layer.png"
DEVICE = "cuda:0"
HOPS = list(range(1, 10))
SETTINGS = [
    ("static", "Static 10×10"),
    ("curriculum", "Curriculum"),
    ("within_only", "Within-block train"),
    ("cross_allowed", "Cross-block train"),
]


def direct_runs() -> list[tuple[str, Path]]:
    return [(setting, run) for setting, protocol, run in base.jobs() if protocol == "direct"]


@torch.inference_mode()
def collect(run: Path, device: torch.device) -> dict:
    cfg = base.load_config([run / "resolved_config.yaml"])
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
    state = torch.load(run / "checkpoints/best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    step = int(state["step"])
    del state
    bars, embeds = [], []
    for start in range(0, len(prompts), 256):
        ids = torch.tensor(prompts[start:start + 256], device=device, dtype=torch.long)
        capture: dict = {}
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            model(ids, capture=capture)
        bars.append(torch.stack([h[:, 4].float() for h in capture["resid"]], dim=1).cpu())
        emb = model.embed(ids).float().cpu()
        embeds.append(torch.cat([emb[:, 2], emb[:, 3]], dim=1))
        del capture
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "step": step,
        "label": label,
        "distance": dist,
        "bars": torch.cat(bars, 0).numpy(),
        "pair": torch.cat(embeds, 0).numpy(),
    }


def layer_auc(features: np.ndarray, labels: np.ndarray, seed: int) -> float:
    train, val, test = base.split_by_class(labels, seed)
    if min(len(train), len(val), len(test)) == 0:
        return float("nan")
    if len(np.unique(labels[train])) < 2 or len(np.unique(labels[test])) < 2 or len(np.unique(labels[val])) < 2:
        return float("nan")
    scores = base.fit_scores(features[train], labels[train], features[val], labels[val], features[test], seed)
    return base.auc(labels[test], scores)


def evaluate(pack: dict) -> dict:
    names = ["input"] + [f"L{i + 1}" for i in range(pack["bars"].shape[1])]
    columns = {"input": pack["pair"]}
    for index in range(pack["bars"].shape[1]):
        columns[f"L{index + 1}"] = pack["bars"][:, index]
    by_hop = {}
    for hop in HOPS:
        chosen = pack["distance"] == hop
        y = pack["label"][chosen]
        hop_row = {}
        for name in names:
            x = columns[name][chosen]
            values = [layer_auc(x, y, seed) for seed in base.SEEDS]
            hop_row[name] = {"auc": float(np.nanmean(values)), "auc_std": float(np.nanstd(values)),
                             "n": int(chosen.sum()), "n_pos": int((y == 1).sum())}
        by_hop[str(hop)] = hop_row
    test = (pack["distance"] >= 2)
    # Official fixed evaluation starts at hop 2. Use that pool for the summary curve.
    y = pack["label"][test]
    overall = {}
    for name in names:
        x = columns[name][test]
        # Stratify the summary probe by hop as well as label.
        key = y + 10 * pack["distance"][test]
        values = []
        for seed in base.SEEDS:
            train, val, hold = base.split_by_class(key, seed)
            if len(np.unique(y[hold])) < 2:
                values.append(float("nan"))
                continue
            scores = base.fit_scores(x[train], y[train], x[val], y[val], x[hold], seed)
            values.append(base.auc(y[hold], scores))
        overall[name] = {"auc": float(np.nanmean(values)), "auc_std": float(np.nanstd(values)),
                         "n": int(test.sum())}
    return {"step": pack["step"], "by_hop": by_hop, "test_hops_2_9": overall}


def draw(results: dict) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.linewidth": 0.6,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })
    columns = ["input"] + [f"L{i}" for i in range(1, 7)]
    labels = ["Input", "L1", "L2", "L3", "L4", "L5", "L6"]
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.8))
    image = None
    for ax, (setting, title) in zip(axes.ravel(), SETTINGS):
        grid = np.array([[results[setting]["by_hop"][str(hop)][col]["auc"] for col in columns] for hop in HOPS])
        image = ax.imshow(grid, vmin=0.45, vmax=1.0, cmap="cividis", aspect="auto")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_yticks(range(len(HOPS)))
        ax.set_yticklabels([str(hop) for hop in HOPS], fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Query hops")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                value = grid[i, j]
                color = "white" if value < 0.72 else "black"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.5, color=color)
    fig.subplots_adjust(left=0.07, right=0.88, top=0.90, bottom=0.06, wspace=0.28, hspace=0.32)
    colorbar = fig.add_axes((0.90, 0.18, 0.018, 0.64))
    fig.colorbar(image, cax=colorbar, label="Test AUC")
    fig.suptitle("Direct reachability by layer, fixed 10×10", fontsize=12)
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=200, facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    tmp = FIG.with_suffix(".png.tmp")
    Image.open(buffer).convert("RGB").save(tmp, format="PNG")
    tmp.replace(FIG)
    print(FIG)


def main() -> None:
    base.DEVICE = DEVICE
    device = torch.device(DEVICE)
    results = {}
    for setting, run in direct_runs():
        print(f"probe {setting}", flush=True)
        pack = collect(run, device)
        results[setting] = evaluate(pack)
        overall = results[setting]["test_hops_2_9"]
        brief = " ".join(f"{name} {overall[name]['auc']:.3f}" for name in ["input", "L1", "L3", "L6"])
        print(f"  step {results[setting]['step']} {brief}", flush=True)
        del pack
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "position": "bar token, the position from which Direct predicts YES/NO",
        "input": "concatenation of source and target token embeddings",
        "hops": "each hop is probed separately on all stored queries of that distance",
        "summary": "one probe on queries of hop >= 2, stratified by hop and label",
        "results": results,
    }, indent=2))
    draw(results)
    print(OUT)


if __name__ == "__main__":
    main()

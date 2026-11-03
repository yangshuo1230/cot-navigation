#!/usr/bin/env python3
"""One-hop edge match versus the multi-hop degree shortcut.

One-hop weights: curr_source_timed_s0 latest.pt, step 1194957.
Multi-hop weights: timed_source_last_joint_lr3e4 best.pt, step 110000.
Frozen inference on each run's own test split.
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cot_navigation.config import load_config  # noqa: E402
from cot_navigation.models.transformer import build_model  # noqa: E402
from cot_navigation.protocols.tokenizer import Tokenizer  # noqa: E402

ONE_RUN = ROOT / "studies/direct_learning/runs/direct_learning_v1/curr_source_timed_s0"
ONE_DATA = ROOT / "studies/direct_learning/data_v1/full_v256_m24.npz"
MULTI_RUN = ROOT / "studies/direct_multihop/runs/direct_multihop_v1/timed_source_last_joint_lr3e4"
MULTI_DATA = ROOT / "studies/direct_multihop/data_v1/arrays.npz"
OUT = ROOT / "figures/direct_shortcut_20261024"
JSON = ROOT / "studies/direct_multihop/degree_shortcut.json"
DEVICE = "cuda:0"
YES, NO = 6, 7
NODE_LO = 14
BLUE, VERM, GREEN, GOLD = "#0072B2", "#D55E00", "#009E73", "#E69F00"
BLUE_F, VERM_F, GREEN_F, GOLD_F, GRAY_F = "#E6F1F8", "#FDE8DC", "#E5F5EF", "#FFF6DE", "#F3F3F3"


def name(token: int) -> str:
    if token == 2:
        return "<EDGE>"
    if token == 3:
        return "<QUERY>"
    if token == 4:
        return "|"
    if token >= NODE_LO:
        return f"V{token - NODE_LO}"
    return str(token)


def edges_of(row: np.ndarray) -> list[dict]:
    triples = row[:-4].reshape(-1, 3)
    out = []
    for index, (edge, u, v) in enumerate(triples):
        out.append({
            "u": int(u),
            "v": int(v),
            "slots": [3 * index, 3 * index + 1, 3 * index + 2],
            "follower": 3 * index + 3,
        })
    return out


def degrees(row: np.ndarray) -> tuple[int, int]:
    source, target = int(row[-3]), int(row[-2])
    triples = row[:-4].reshape(-1, 3)
    out = int((triples[:, 1] == source).sum())
    inn = int((triples[:, 2] == target).sum())
    return out, inn


def load_model(run: Path, device: torch.device):
    cfg = load_config([run / "resolved_config.yaml"])
    tok = Tokenizer.from_config(cfg)
    model = build_model(cfg.model, len(tok), tok.pad).to(device).eval()
    state = torch.load(run / "checkpoints" / ("latest.pt" if "direct_learning" in str(run) else "best.pt"), map_location="cpu", weights_only=False)
    # Caller passes the checkpoint path separately when the file is not latest/best by folder.
    model.load_state_dict(state["model"], strict=True)
    step = int(state["step"])
    del state
    return model, step


def load_ckpt(run: Path, filename: str, device: torch.device):
    cfg = load_config([run / "resolved_config.yaml"])
    tok = Tokenizer.from_config(cfg)
    model = build_model(cfg.model, len(tok), tok.pad).to(device).eval()
    state = torch.load(run / "checkpoints" / filename, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    step = int(state["step"])
    del state
    return model, step


@torch.inference_mode()
def predict(model, tokens: np.ndarray, device: torch.device) -> np.ndarray:
    batch = 512
    preds = []
    for start in range(0, len(tokens), batch):
        chunk = torch.tensor(tokens[start:start + batch], device=device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(chunk)
        preds.append(logits[:, -1].float().argmax(-1).cpu().numpy())
    return np.concatenate(preds)


@torch.inference_mode()
def attend(model, row: np.ndarray, device: torch.device) -> np.ndarray:
    tokens = torch.tensor(row, device=device, dtype=torch.long)[None]
    _logits, weights = model(tokens, return_attn=True)
    # [layers, query, key], head 0. Full precision: the example masses are small differences.
    return torch.stack([w[0, 0].float() for w in weights], dim=0).cpu().numpy()


def rate_table(outdeg, indeg, labels, pred) -> list[dict]:
    cells = defaultdict(lambda: {"pos_n": 0, "pos_yes": 0, "neg_n": 0, "neg_yes": 0})
    said_yes = pred == YES
    for out, inn, label, yes in zip(outdeg, indeg, labels, said_yes):
        cell = cells[(int(out), int(inn))]
        if int(label) == 1:
            cell["pos_n"] += 1
            cell["pos_yes"] += int(yes)
        else:
            cell["neg_n"] += 1
            cell["neg_yes"] += int(yes)
    rows = []
    for (out, inn), cell in sorted(cells.items()):
        if cell["pos_n"] < 12 or cell["neg_n"] < 12:
            continue
        rows.append({
            "out": out,
            "in": inn,
            "pos_n": cell["pos_n"],
            "neg_n": cell["neg_n"],
            "pos_yes": cell["pos_yes"] / cell["pos_n"],
            "neg_yes": cell["neg_yes"] / cell["neg_n"],
        })
    return rows


def balanced(pred, labels) -> dict:
    yes, no = labels == 1, labels == 0
    yes_acc = float((pred[yes] == YES).mean())
    no_acc = float((pred[no] == NO).mean())
    return {"yes": yes_acc, "no": no_acc, "bacc": 0.5 * (yes_acc + no_acc), "n": int(len(labels))}


def box(ax, x, y, w, h, text, face, edge, fs=8.0):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.08",
        facecolor=face, edgecolor=edge, linewidth=1.1,
        mutation_aspect=0.6, zorder=3,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="#1A1A1A", zorder=4)


def arrow(ax, start, end, color, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=10,
        linewidth=1.2, color=color, connectionstyle=f"arc3,rad={rad}",
        shrinkA=2, shrinkB=2, zorder=2,
    ))


def tag(ax, x, y, text, color, ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=8, color=color, zorder=5,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4})


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
    })


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    buf.seek(0)
    Image.open(buf).convert("RGB").save(path)
    print(path)


def _degree_panel(ax, rows: list[dict], pos_label: str, neg_label: str, ylabel_side: str = "left") -> None:
    x = np.arange(len(rows))
    pos = [row["pos_yes"] for row in rows]
    neg = [row["neg_yes"] for row in rows]
    ax.bar(x - 0.17, pos, width=0.34, color=BLUE, label=pos_label, zorder=3)
    ax.bar(x + 0.17, neg, width=0.34, color=VERM, label=neg_label, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{row['out']}, {row['in']}\n{row['pos_n']}/{row['neg_n']}" for row in rows], fontsize=7.2)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.axhline(0.5, color="#C8C8C8", lw=0.8, ls=(0, (3, 2)), zorder=0)
    ax.yaxis.grid(True, color="#EEEEEE", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    if ylabel_side == "right":
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
    ax.set_xlabel("Source out-edges, target in-edges", fontsize=8.5)
    ax.set_ylabel("Fraction answered YES", fontsize=8.5)
    ax.set_title("(b)  Answer versus degree", loc="left", fontsize=11, pad=2)
    ax.legend(frameon=False, fontsize=7.4, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, borderaxespad=0)
    for xpos, value, color in list(zip(x - 0.17, pos, [BLUE] * len(pos))) + list(zip(x + 0.17, neg, [VERM] * len(neg))):
        if value >= 0.90:
            ax.text(xpos, value - 0.045, f"{value:.2f}", ha="center", va="top", fontsize=6.2, color="white")
        else:
            ax.text(xpos, value + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=6.2, color=color)


def _one_attention(ax, one: dict) -> None:
    ax.set_xlim(0, 6.35)
    ax.set_ylim(0.18, 3.62)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_title("(a)  Attention on test query 0", loc="left", fontsize=11, color="#1A1A1A", pad=2)
    w, h = 1.08, 0.46
    y1, y0 = 1.92, 0.70
    tokens = [
        (0.08, y1, "<EDGE>", GRAY_F, "#8A8A8A"),
        (1.22, y1, one["u"], BLUE_F, BLUE),
        (2.36, y1, one["v"], VERM_F, VERM),
        (3.72, y1, "<EDGE>", GREEN_F, GREEN),
        (0.08, y0, "<QUERY>", GRAY_F, "#8A8A8A"),
        (1.22, y0, one["u"], BLUE_F, BLUE),
        (2.36, y0, one["v"], VERM_F, VERM),
        (3.50, y0, "|", GOLD_F, GOLD),
    ]
    for x, y, text, face, edge in tokens:
        box(ax, x, y, w, h, text, face, edge, fs=7.6)
    arrow(ax, (1.22 + w / 2, y1 + h), (3.72 + 0.16, y1 + h), BLUE, rad=-0.32)
    arrow(ax, (2.36 + w / 2, y1 + h), (3.72 + w * 0.62, y1 + h), VERM, rad=-0.12)
    tag(ax, 2.15, 3.12, f"L1  {one['l1_u']:.2f}", BLUE)
    tag(ax, 4.55, 3.18, f"L1  {one['l1_v']:.2f}", VERM, ha="left")
    arrow(ax, (3.50 + w / 2, y0 + h), (3.72 + w / 2, y1), GREEN, rad=0.0)
    tag(ax, 5.05, 1.42, f"L2  {one['l2_bar']:.3f}", GREEN, ha="left")
    ax.text(5.02, y1 + 0.16, "·  ·  ·", fontsize=12, color="#B0B0B0", va="center")
    ax.text(0.08, 0.32, f"{one['u']} → {one['v']} is in the list. The bar reads the copied edge.", fontsize=8, color="#444444")


def _multi_attention(ax, multi: dict) -> None:
    ax.set_xlim(0, 12.05)
    ax.set_ylim(0.02, 2.58)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_title("(a)  Attention on test query 6669", loc="left", fontsize=11, color="#1A1A1A", pad=2)
    groups = [
        ("first", 0.04, multi["hop1"], f"first hop   L3 {multi['src_on_first']:.3f}", BLUE, 1),
        ("middle", 3.02, multi["mid"], "middle   L3 0.00", "#A0A0A0", None),
        ("last", 6.00, multi["hop3"], f"last hop   L3 {multi['tgt_on_last']:.3f}", VERM, 2),
        ("extra", 8.98, multi["extra"], f"other in-edge   L3 {multi['tgt_on_extra']:.3f}", "#C47A4A", 2),
    ]
    tops = {}
    bw, gap = 0.90, 0.03
    ey = 1.78
    for key, x0, toks, caption, edge, highlight in groups:
        ax.text(x0, ey + 0.58, caption, fontsize=6.8, color=edge, ha="left", va="bottom")
        for i, text in enumerate(toks):
            if highlight is None or i != highlight:
                face, this_edge = GRAY_F, "#B0B0B0"
            elif key == "first":
                face, this_edge = BLUE_F, BLUE
            else:
                face, this_edge = VERM_F, edge
            box(ax, x0 + i * (bw + gap), ey, bw, 0.42, text, face, this_edge, fs=6.6)
            tops[(key, i)] = (x0 + i * (bw + gap) + bw / 2, ey)
    w = 0.98
    qx = [0.28, 1.32, 2.36, 3.40]
    qy = 0.40
    qtoks = [("<QUERY>", GRAY_F, "#8A8A8A"), (multi["source"], BLUE_F, BLUE), (multi["target"], VERM_F, VERM), ("|", GOLD_F, GOLD)]
    qpos = {}
    for x, (text, face, edge) in zip(qx, qtoks):
        box(ax, x, qy, w, 0.42, text, face, edge, fs=7.0)
        qpos[text] = (x + w / 2, qy + 0.42)
    src, tgt, bar = qpos[multi["source"]], qpos[multi["target"]], qpos["|"]
    first, last, extra = tops[("first", 1)], tops[("last", 2)], tops[("extra", 2)]
    arrow(ax, src, first, BLUE, rad=-0.10)
    arrow(ax, (tgt[0] - 0.06, tgt[1]), last, VERM, rad=-0.02)
    arrow(ax, (tgt[0] + 0.16, tgt[1]), extra, "#C47A4A", rad=0.03)
    tag(ax, 4.58, qy + 0.21, f"L5   {multi['source']} {multi['bar_src']:.2f}", BLUE, ha="left")
    tag(ax, 5.95, qy + 0.21, f"{multi['target']} {multi['bar_tgt']:.2f}", VERM, ha="left")
    ax.text(0.04, 0.12, f"Path {' → '.join(multi['path'])}. Each endpoint reads its own adjacent edges.", fontsize=8, color="#444444")


def _axes_for(fig_w: float, fig_h: float, left: float, bottom: float, width: float, height: float):
    return [left / fig_w, bottom / fig_h, width / fig_w, height / fig_h]


def draw_checkpoints(one: dict, one_rows: list[dict], multi: dict, multi_rows: list[dict]) -> None:
    _style()
    OUT.mkdir(parents=True, exist_ok=True)
    fig_w, fig_h = 11.55, 3.72
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle("One-hop checkpoint", x=0.012, y=0.975, ha="left", fontsize=13, color="#1A1A1A")
    _one_attention(fig.add_axes(_axes_for(fig_w, fig_h, 0.08, 0.18, 5.35, 2.90)), one)
    _degree_panel(fig.add_axes(_axes_for(fig_w, fig_h, 5.62, 0.58, 5.72, 2.42)), one_rows, "Edge present", "Edge absent")
    _save(fig, OUT / "onehop_checkpoint.png")

    fig_w, fig_h = 16.40, 3.38
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle("Three-hop checkpoint", x=0.006, y=0.966, ha="left", fontsize=13, color="#1A1A1A")
    _multi_attention(fig.add_axes(_axes_for(fig_w, fig_h, 0.04, 0.40, 10.92, 2.32)), multi)
    _degree_panel(fig.add_axes(_axes_for(fig_w, fig_h, 11.74, 0.66, 4.52, 2.06)), multi_rows, "Path exists", "No path")
    _save(fig, OUT / "multihop_checkpoint.png")
    for stale in ("attention_schematic.png", "degree_shortcut.png"):
        path = OUT / stale
        if path.exists():
            path.unlink()


def one_example(model, tokens, labels, device) -> dict:
    index = 0
    row = tokens[index]
    assert int(labels[index]) == 1
    attn = attend(model, row, device)
    recs = edges_of(row)
    source, target = int(row[-3]), int(row[-2])
    match = next(rec for rec in recs if rec["u"] == source and rec["v"] == target)
    follower = match["follower"]
    u_slot, v_slot = match["slots"][1], match["slots"][2]
    length = len(row)
    return {
        "index": index,
        "u": name(source),
        "v": name(target),
        "follower_token": name(int(row[follower])),
        "l1_u": float(attn[0, follower, u_slot]),
        "l1_v": float(attn[0, follower, v_slot]),
        "l2_bar": float(attn[1, length - 1, follower]),
        "pred_yes": True,
        "caption": f"Query {name(source)} → {name(target)}. Layer 1 copies the edge onto the next token. Layer 2, | selects that token.",
    }


def multi_example(model, tokens, labels, hops, device) -> dict:
    index = 6669
    row = tokens[index]
    assert int(labels[index]) == 1 and int(hops[index]) == 3
    attn = attend(model, row, device)
    recs = edges_of(row)
    source, target = int(row[-3]), int(row[-2])
    # Unique path of length 3.
    adj = defaultdict(list)
    for rec in recs:
        adj[rec["u"]].append(rec)
    path = [source]
    chosen = []
    node = source
    for _ in range(3):
        nxts = adj[node]
        assert len(nxts) >= 1
        # Prefer the successor that continues toward a 3-hop witness; the logged example has one outgoing edge at the source.
        rec = nxts[0] if len(nxts) == 1 else None
        if rec is None:
            break
        chosen.append(rec)
        node = rec["v"]
        path.append(node)
    # Locate last-hop and a distractor incoming edge by target.
    incoming = [rec for rec in recs if rec["v"] == target]
    last = incoming[0]
    # The witness last hop is the one whose source is on the forward path. Recompute properly.
    dist = {source: 0}
    queue = [source]
    parent = {}
    for cur in queue:
        for rec in adj[cur]:
            if rec["v"] not in dist:
                dist[rec["v"]] = dist[cur] + 1
                parent[rec["v"]] = rec
                queue.append(rec["v"])
    assert dist[target] == 3
    hop3 = parent[target]
    mid_node = hop3["u"]
    hop2 = parent[mid_node]
    hop1 = parent[hop2["u"]]
    extra = next(rec for rec in incoming if rec is not hop3)
    length = len(row)
    src_pos, tgt_pos, bar = length - 3, length - 2, length - 1
    return {
        "index": index,
        "source": name(source),
        "target": name(target),
        "hop1": [name(int(row[i])) for i in hop1["slots"]],
        "mid": [name(int(row[i])) for i in hop2["slots"]],
        "hop3": [name(int(row[i])) for i in hop3["slots"]],
        "extra": [name(int(row[i])) for i in extra["slots"]],
        "src_on_first": float(attn[2, src_pos, hop1["slots"][1]]),
        "src_on_mid": float(attn[2, src_pos, hop2["slots"]].sum()),
        "tgt_on_last": float(attn[2, tgt_pos, hop3["slots"][2]]),
        "tgt_on_extra": float(attn[2, tgt_pos, extra["slots"][2]]),
        "tgt_on_mid": float(attn[2, tgt_pos, hop2["slots"]].sum()),
        "bar_src": float(attn[4, bar, src_pos]),
        "bar_tgt": float(attn[4, bar, tgt_pos]),
        "bar_mid": float(attn[4, bar, hop2["slots"]].sum()),
        "caption": f"Path {name(source)} → {name(hop1['v'])} → {name(hop2['v'])} → {name(target)}. The middle edge is unread. | reads the two endpoints.",
        "path": [name(node) for node in [source, hop1["v"], hop2["v"], target]],
    }


def main() -> None:
    device = torch.device(DEVICE)
    one_model, one_step = load_ckpt(ONE_RUN, "latest.pt", device)
    one_blob = np.load(ONE_DATA)
    one_x = one_blob["test_x"].astype(np.int64)
    one_y = one_blob["test_y"].astype(np.int64)
    one_pred = predict(one_model, one_x, device)
    one_out = np.array([degrees(row)[0] for row in one_x])
    one_in = np.array([degrees(row)[1] for row in one_x])
    one_rows = rate_table(one_out, one_in, one_y, one_pred)
    one_stats = balanced(one_pred, one_y)
    one_ex = one_example(one_model, one_x, one_y, device)
    print("one", one_step, one_stats, flush=True)
    print("one example", one_ex, flush=True)
    print("one cells", one_rows, flush=True)
    del one_model

    multi_model, multi_step = load_ckpt(MULTI_RUN, "best.pt", device)
    blob = np.load(MULTI_DATA)
    x = blob["test_x"].astype(np.int64)
    y = blob["test_y"].astype(np.int64)
    hop = blob["test_hop"].astype(np.int64)
    pred = predict(multi_model, x, device)
    by_hop = {}
    tables = {}
    for value in (1, 2, 3):
        mask = hop == value
        by_hop[str(value)] = balanced(pred[mask], y[mask])
        out = np.array([degrees(row)[0] for row in x[mask]])
        inn = np.array([degrees(row)[1] for row in x[mask]])
        tables[str(value)] = rate_table(out, inn, y[mask], pred[mask])
        print("hop", value, by_hop[str(value)], flush=True)
        print(tables[str(value)], flush=True)
    multi_ex = multi_example(multi_model, x, y, hop, device)
    print("multi example", multi_ex, flush=True)
    del multi_model

    payload = {
        "one_hop": {"step": one_step, "checkpoint": str(ONE_RUN / "checkpoints/latest.pt"), "test": one_stats, "cells": one_rows, "example": one_ex},
        "multi_hop": {"step": multi_step, "checkpoint": str(MULTI_RUN / "checkpoints/best.pt"), "test": by_hop, "cells": tables, "example": multi_ex},
    }
    JSON.write_text(json.dumps(payload, indent=2) + "\n")
    draw_checkpoints(one_ex, one_rows, multi_ex, tables["3"])
    print(JSON)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Where the legal-edge decision enters a PATH step.

Input edits already show that the current node's edge heads control the next
token. These masks keep the tokens and remove attention from candidate
carriers, one carrier at a time.
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
import plot_path_edge_decode as decode  # noqa: E402
import cot_navigation.models.transformer as transformer  # noqa: E402

OUT = ROOT / "figures/hard_dag_v1/path_edge_route.png"
JSON = ROOT / "research/block_membership_probe_20261021/path_edge_route.json"
DEVICE = "cuda:3"

# Shared masks, scored on every intermediate path step at once.
SHARED = (
    ("Path step misses the edge list", "step_misses_list"),
    ("Target misses the edge list", "target_misses_list"),
    ("<QUERY> misses the edge list", "query_misses_list"),
    ("Last edge misses earlier edges", "last_misses_earlier"),
    ("Bar misses the edge list", "bar_misses_list"),
)


def causal(length: int, device) -> torch.Tensor:
    index = torch.arange(length, device=device)
    return index[None, :] <= index[:, None]


def run_mask(model, tokens, device, mask: torch.Tensor) -> torch.Tensor:
    def record_mask(history, positions, keep_records, use_flex=False):
        return mask

    transformer.record_mask = record_mask
    model.swa_records = 1
    model.swa_backend = "sdpa"
    try:
        return decode.logits_of(model, tokens, device)
    finally:
        model.swa_records = 0


def shared_mask(item, kind: str, device) -> torch.Tensor:
    length = len(item["tokens"])
    prompt_len = item["prompt_len"]
    mask = causal(length, device)
    edge_keys = torch.arange(prompt_len - 4, device=device)
    if kind == "step_misses_list":
        queries = torch.arange(prompt_len, length, device=device)
        mask[queries[:, None], edge_keys[None, :]] = False
    elif kind == "target_misses_list":
        mask[prompt_len - 2, edge_keys] = False
    elif kind == "query_misses_list":
        mask[prompt_len - 4, edge_keys] = False
    elif kind == "bar_misses_list":
        mask[prompt_len - 1, edge_keys] = False
    elif kind == "last_misses_earlier":
        start = max(prompt_len - 7, 0)
        queries = torch.arange(start, prompt_len - 4, device=device)
        earlier = torch.arange(start, device=device)
        mask[queries[:, None], earlier[None, :]] = False
    else:
        raise KeyError(kind)
    return mask[None, None]


def direct_mask(item, row, device, only_current: bool) -> torch.Tensor:
    length = len(item["tokens"])
    mask = causal(length, device)
    src, _dst, _inverse = decode.endpoints(item)
    current = torch.tensor(
        [edge * 3 + offset for edge, source in enumerate(src.tolist()) if source == row["cur"] for offset in range(3)],
        device=device, dtype=torch.long,
    )
    if only_current == "self":
        allowed = torch.zeros(length, dtype=torch.bool, device=device)
        allowed[row["receiver"]] = True
        mask[row["receiver"], ~allowed] = False
    elif only_current:
        allowed = torch.zeros(length, dtype=torch.bool, device=device)
        if len(current):
            allowed[current] = True
        allowed[row["receiver"]] = True
        mask[row["receiver"], ~allowed] = False
    elif len(current):
        mask[row["receiver"], current] = False
    return mask[None, None]


def legal_rate(logits, item, tok, rows) -> tuple[float, float]:
    hits, changed = [], []
    base_pred = rows
    for row, pred0 in base_pred:
        pred = int(logits[row["receiver"]].argmax())
        hits.append(float(pred in row["legal"] or pred == row["gold"]))
        changed.append(float(pred != pred0))
    return float(np.mean(hits)), float(np.mean(changed))


def score_size(model, tok, items, device) -> dict:
    shared = {key: {"legal": [], "changed": []} for _label, key in SHARED}
    direct = {key: {"legal": [], "changed": []} for key in ("step_misses_own", "step_sees_only_own", "step_sees_only_self")}
    yes_last, no_last = [], []
    for item in items:
        base = decode.logits_of(model, item["tokens"], device)
        rows = []
        for row in decode.step_rows(item, tok):
            if row["last"]:
                pred = int(base[row["receiver"]].argmax())
                (yes_last if item["label"] == 1 else no_last).append(float(pred == row["gold"]))
                continue
            rows.append((row, int(base[row["receiver"]].argmax())))
        if not rows:
            continue
        for _label, key in SHARED:
            logits = run_mask(model, item["tokens"], device, shared_mask(item, key, device))
            legal, changed = legal_rate(logits, item, tok, rows)
            shared[key]["legal"].append(legal)
            shared[key]["changed"].append(changed)
        for row, pred0 in rows:
            for key, only in (("step_misses_own", False), ("step_sees_only_own", True), ("step_sees_only_self", "self")):
                logits = run_mask(model, item["tokens"], device, direct_mask(item, row, device, only))
                pred = int(logits[row["receiver"]].argmax())
                direct[key]["legal"].append(float(pred in row["legal"] or pred == row["gold"]))
                direct[key]["changed"].append(float(pred != pred0))
    def pack(bucket):
        return {
            key: {"legal": float(np.mean(values["legal"])), "changed": float(np.mean(values["changed"]))}
            for key, values in bucket.items()
        }
    return {
        "shared": pack(shared),
        "direct": pack(direct),
        "yes_last_exact": float(np.mean(yes_last)) if yes_last else None,
        "no_last_exact": float(np.mean(no_last)) if no_last else None,
        "n_yes_last": len(yes_last),
        "n_no_last": len(no_last),
    }


def draw(pred_blocks, route_blocks) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "axes.linewidth": 0.6,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    labels = [block["size"].replace("DAG ", "") for block in pred_blocks]
    x = np.arange(len(pred_blocks))
    width = 0.22
    series = (
        ("Exact next node", [b["intermediate"]["exact"] for b in pred_blocks], "#D55E00"),
        ("Any legal neighbor", [b["intermediate"]["exact"] + b["intermediate"]["other_legal"] for b in pred_blocks], "#009E73"),
        ("Chance", [b["chance_legal"] for b in pred_blocks], "#999999"),
    )
    for offset, (label, values, color) in enumerate(series):
        axes[0].bar(x + (offset - 1) * width, values, width=width, color=color, label=label, linewidth=0)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Share of intermediate steps")
    axes[0].set_title("Next token is a legal neighbor")
    axes[0].yaxis.grid(True, color="#dddddd", linewidth=0.6)
    axes[0].set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=8)

    route_keys = (
        ("Own edges hidden from this step", "step_misses_own"),
        ("This step sees only its own edges", "step_sees_only_own"),
        ("Whole edge list hidden from this step", "step_misses_list"),
        ("Edge list hidden from query target", "target_misses_list"),
        ("Edge list hidden from <QUERY>", "query_misses_list"),
    )
    y = np.arange(len(route_keys))
    height = 0.22
    colors = ("#0072B2", "#E69F00", "#009E73")
    for offset, (block, color) in enumerate(zip(route_blocks, colors)):
        values = []
        for _label, key in route_keys:
            bucket = block["direct"] if key in block["direct"] else block["shared"]
            values.append(bucket[key]["legal"])
        axes[1].barh(y + (offset - 1) * height, values, height=height, color=color, label=block["size"].replace("DAG ", ""), linewidth=0)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([label for label, _key in route_keys], fontsize=8)
    axes[1].set_xlim(0, 1.05)
    axes[1].set_xlabel("Steps still predicting a legal neighbor")
    axes[1].set_title("Which view of the edges is required")
    axes[1].xaxis.grid(True, color="#dddddd", linewidth=0.6)
    axes[1].set_axisbelow(True)
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("How PATH selects a usable edge", fontsize=12)
    fig.subplots_adjust(left=0.28, right=0.98, top=0.84, bottom=0.14, wspace=0.18)
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
    pred = json.loads((ROOT / "research/block_membership_probe_20261021/path_edge_decode.json").read_text())
    saved = []
    for (name, root), pred_block in zip(pathfig.SIZES, pred):
        arrays = pathfig.load_test(root)
        items = decode.choose([pathfig.render_index(arrays, tok, index) for index in range(len(arrays["label"]))])
        block = score_size(model, tok, items, device)
        block["size"] = name
        block["step"] = step
        saved.append(block)
        print(f"\n{name} yes_last_exact {block['yes_last_exact']:.3f} (n={block['n_yes_last']}) "
              f"no_last_exact {block['no_last_exact']:.3f} (n={block['n_no_last']})", flush=True)
        for key, row in {**block["direct"], **block["shared"]}.items():
            print(f"  {key}: legal {row['legal']:.3f} changed {row['changed']:.3f}", flush=True)
    draw(pred, saved)
    JSON.parent.mkdir(parents=True, exist_ok=True)
    JSON.write_text(json.dumps(saved, indent=2))
    print(JSON)


if __name__ == "__main__":
    main()

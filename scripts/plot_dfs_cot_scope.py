#!/usr/bin/env python3
"""Does in-context DFS attention stay on the previous record or cover the CoT?

Traces need at least 12 records so an action can have eight earlier records.
Only those late actions are counted. Mass is split into the prompt, the current
record, the previous record, older records, and the receiver itself.
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

OUT = ROOT / "figures/scale_v1/dfs_context_cot_scope.png"
JSON = ROOT / "research/block_membership_probe_20261021/dfs_context_cot_scope.json"
DEVICE = "cuda:0"
MIN_RECORDS = 12
MAX_LEN = 2400
MIN_HISTORY = 8
BUCKETS = ("Prompt", "Current record", "Previous record", "Older CoT", "Self")
COLORS = {
    "Prompt": "#0072B2",
    "Current record": "#E69F00",
    "Previous record": "#D55E00",
    "Older CoT": "#009E73",
    "Self": "#CC79A7",
}
KINDS = ("PUSH", "Terminal", "Answer")


def long_pool(arrays, tok):
    rows = []
    for index in range(len(arrays["label"])):
        item = scale.render_index(arrays, tok, index)
        n = len(records(item["tokens"], item["prompt_len"], tok))
        if n >= MIN_RECORDS and len(item["tokens"]) <= MAX_LEN:
            rows.append((n, item["label"], item["distance"], item))
    return rows


def record_at(parsed, pos: int) -> int | None:
    for index, record in enumerate(parsed):
        if record["start"] <= pos <= record["end"]:
            return index
    return None


def receivers_of(tokens, tok, prompt_len: int, parsed):
    found = []
    for index, record in enumerate(parsed):
        if index < MIN_HISTORY:
            continue
        action_pos = record["separators"][2] + 1
        action = tok.name(tokens[action_pos])
        kind = "PUSH" if action == "<PUSH>" else "Terminal"
        found.append((kind, action_pos, index))
    if len(parsed) - 1 >= MIN_HISTORY:
        answer = parsed[-1]["end"]
        found.append(("Answer", answer, len(parsed) - 1))
    return found


def row_scope(attn, receiver: int, record_index: int, prompt_len: int, parsed) -> dict:
    row = attn[receiver, : receiver + 1].astype(np.float64)
    total = max(float(row.sum()), 1e-8)
    row = row / total
    mass = {name: 0.0 for name in BUCKETS}
    lag_mass: dict[int, float] = {}
    mass["Self"] = float(row[receiver])
    mass["Prompt"] = float(row[:prompt_len].sum())
    previous_token = float(row[receiver - 1])
    for pos in range(prompt_len, receiver):
        key_index = record_at(parsed, pos)
        if key_index is None:
            continue
        lag = record_index - key_index
        lag_mass[lag] = lag_mass.get(lag, 0.0) + float(row[pos])
        if lag <= 0:
            mass["Current record"] += float(row[pos])
        elif lag == 1:
            mass["Previous record"] += float(row[pos])
        else:
            mass["Older CoT"] += float(row[pos])
    cot = mass["Current record"] + mass["Previous record"] + mass["Older CoT"]
    if lag_mass:
        weights = np.array(list(lag_mass.values()), dtype=np.float64)
        weights = weights / max(weights.sum(), 1e-8)
        entropy = float(-(weights * np.log(weights + 1e-12)).sum())
        n_eff = float(np.exp(entropy))
    else:
        n_eff = 0.0
    return {
        "mass": mass,
        "previous_token": previous_token,
        "cot": cot,
        "n_eff": n_eff,
        "history": record_index,
        "lag_mass": lag_mass,
    }


def aggregate(traces: list[dict]) -> dict:
    out = {}
    for kind in KINDS:
        rows = [item for item in traces if item["kind"] == kind]
        block = {"n": len(rows), "layers": []}
        for layer in range(6):
            chosen = [item["layers"][layer] for item in rows]
            if not chosen:
                continue
            means = {name: float(np.mean([item["mass"][name] for item in chosen])) for name in BUCKETS}
            cot = np.array([item["cot"] for item in chosen])
            prev = np.array([item["mass"]["Previous record"] for item in chosen])
            older = np.array([item["mass"]["Older CoT"] for item in chosen])
            current = np.array([item["mass"]["Current record"] for item in chosen])
            hist = np.array([item["history"] for item in chosen])
            n_eff = np.array([item["n_eff"] for item in chosen])
            lag = np.zeros(13)
            lag_n = np.zeros(13)
            for item in chosen:
                cot_mass = max(item["cot"], 1e-8)
                tail = 0.0
                for key, value in item["lag_mass"].items():
                    if key < 0:
                        continue
                    if key < 12:
                        lag[key] += value / cot_mass
                        lag_n[key] += 1
                    else:
                        tail += value / cot_mass
                if tail > 0:
                    lag[12] += tail
                    lag_n[12] += 1
            block["layers"].append({
                "mass": means,
                "previous_token": float(np.mean([item["previous_token"] for item in chosen])),
                "cot_mean": float(cot.mean()),
                "share_current": float(np.mean(current / np.maximum(cot, 1e-8))),
                "share_previous": float(np.mean(prev / np.maximum(cot, 1e-8))),
                "share_older": float(np.mean(older / np.maximum(cot, 1e-8))),
                "n_eff": float(n_eff.mean()),
                "history": float(hist.mean()),
                "lag_share": (lag / np.maximum(lag_n, 1)).round(4).tolist(),
            })
        out[kind] = block
    return out


def draw(summary: list[dict]) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.linewidth": 0.6,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(3, 3, figsize=(12.4, 8.4), sharey="row")
    x = np.arange(6)
    for row, block in enumerate(summary):
        for column, kind in enumerate(KINDS):
            ax = axes[row, column]
            stats = block["kinds"][kind]
            bottom = np.zeros(6)
            for name in BUCKETS:
                values = np.array([layer["mass"][name] for layer in stats["layers"]])
                ax.bar(x, values, bottom=bottom, width=0.72, color=COLORS[name], label=name, linewidth=0)
                bottom += values
            ax.set_xticks(x)
            ax.set_xticklabels([f"L{i + 1}" for i in range(6)], fontsize=8)
            ax.set_ylim(0, 1.02)
            ax.yaxis.grid(True, color="#dddddd", linewidth=0.6)
            ax.set_axisbelow(True)
            if row == 0:
                ax.set_title(kind, fontsize=11)
            if column == 0:
                ax.set_ylabel(f"{block['size']}\nAttention mass", fontsize=9)
            if row == 2 and column == 2:
                ax.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.02, 1.02))
    fig.suptitle(
        "In-context DFS attention scope  ·  actions with at least 8 earlier records",
        fontsize=12,
    )
    fig.subplots_adjust(left=0.08, right=0.84, top=0.88, bottom=0.08, wspace=0.18, hspace=0.32)
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
    saved = []
    for name, root in scale.SIZES:
        arrays = scale.load_test(root)
        pool = long_pool(arrays, tok)
        print(f"{name} eligible {len(pool)}", flush=True)
        chosen = []
        for label in (1, 0):
            group = sorted((row for row in pool if row[1] == label), key=lambda row: (row[0], row[3]["index"]))
            # Prefer mid-length traces so the forward stays bounded.
            group.sort(key=lambda row: (abs(row[0] - 16), row[3]["index"]))
            chosen.extend(group[:12])
        collected = []
        for n_records, _label, _dist, item in chosen:
            parsed = records(item["tokens"], item["prompt_len"], tok)
            answer = parsed[-1]["end"]
            nxt = tok.name(item["tokens"][answer + 1])
            if nxt not in {"<YES>", "<NO>"}:
                raise RuntimeError(f"answer semicolon is followed by {nxt}")
            targets = receivers_of(item["tokens"], tok, item["prompt_len"], parsed)
            layers = scale.all_layers(model, item["tokens"], device)
            for kind, pos, record_index in targets:
                measured = [
                    row_scope(attn, pos, record_index, item["prompt_len"], parsed)
                    for attn in layers
                ]
                collected.append({"kind": kind, "layers": measured, "label": item["label"]})
            del layers
        block = aggregate(collected)
        saved.append({"size": name, "n_traces": len(chosen), "step": step, "kinds": {
            kind: {
                "n": block[kind]["n"],
                "layers": block[kind]["layers"],
            } for kind in KINDS
        }})
        print(name, flush=True)
        for kind in KINDS:
            print(f"  {kind} n={block[kind]['n']}", flush=True)
            for index, layer in enumerate(block[kind]["layers"]):
                mass = layer["mass"]
                print(
                    f"    L{index + 1}: prompt {mass['Prompt']:.2f} current {mass['Current record']:.2f} "
                    f"prev {mass['Previous record']:.2f} older {mass['Older CoT']:.2f} self {mass['Self']:.2f} "
                    f"| cot {layer['cot_mean']:.2f} share prev {layer['share_previous']:.2f} "
                    f"older {layer['share_older']:.2f} neff {layer['n_eff']:.2f} "
                    f"prev_tok {layer['previous_token']:.2f} hist {layer['history']:.1f}",
                    flush=True,
                )
                print("      lag", " ".join(f"{value:.2f}" for value in layer["lag_share"]), flush=True)
    draw(saved)
    JSON.parent.mkdir(parents=True, exist_ok=True)
    JSON.write_text(json.dumps({"buckets": list(BUCKETS), "min_history": MIN_HISTORY, "summary": saved}, indent=2))
    print(JSON)


if __name__ == "__main__":
    main()

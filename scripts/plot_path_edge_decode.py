#!/usr/bin/env python3
"""What PATH actually predicts, and whether the edge list controls it.

Path steps are teacher-forced positions that predict the next node. Edits
change edge endpoints and rerun the same positions. One head, frozen model.
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
from cot_navigation.evaluation.rollout import generate  # noqa: E402

OUT = ROOT / "figures/hard_dag_v1/path_edge_decode.png"
JSON = ROOT / "studies/block_membership_probe/path_edge_decode.json"
DEVICE = "cuda:3"
EDITS = ("Shuffle heads", "Drop edges from here", "Only edges from here")


def choose(items):
    pool = [item for item in items if 4 <= len(item["path"]) <= 8]
    chosen = []
    for label in (1, 0):
        group = sorted((item for item in pool if item["label"] == label), key=lambda item: item["index"])
        chosen.extend(group[:16])
    return chosen


def endpoints(item):
    tokens, prompt_len = item["tokens"], item["prompt_len"]
    inverse = {token: index for index, token in enumerate(item["node_tokens"])}
    n_edges = (prompt_len - 4) // 3
    src = np.array([inverse[tokens[edge * 3 + 1]] for edge in range(n_edges)])
    dst = np.array([inverse[tokens[edge * 3 + 2]] for edge in range(n_edges)])
    return src, dst, inverse


def dummy_token(item, tok) -> int:
    used = set(item["node_tokens"])
    for raw in range(tok.n_node_tokens):
        token = tok.node_id(raw)
        if token not in used:
            return token
    raise RuntimeError("no unused node token")


@torch.inference_mode()
def logits_of(model, tokens, device) -> torch.Tensor:
    ids = torch.tensor(tokens, device=device, dtype=torch.long)[None]
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model(ids)
    return logits[0].float().cpu()


def edited(tokens, item, kind: str, cur: int | None, dummy: int, rng: np.random.Generator) -> list[int]:
    src, dst, inverse = endpoints(item)
    out = list(tokens)
    n_edges = len(src)
    if kind == "shuffle":
        order = rng.permutation(n_edges)
        for edge, source in enumerate(order):
            out[edge * 3 + 2] = item["node_tokens"][int(dst[source])]
        return out
    for edge in range(n_edges):
        keep = int(src[edge]) == cur
        if kind == "drop" and keep:
            out[edge * 3 + 2] = dummy
        elif kind == "only" and not keep:
            out[edge * 3 + 1] = dummy
            out[edge * 3 + 2] = dummy
    return out


def classify(pred: int, gold: int, legal: set[int], item, cur_token: int) -> str:
    if pred == gold:
        return "exact"
    if pred in legal:
        return "other_legal"
    if pred == item["node_tokens"][item["target"]]:
        return "target"
    if pred == item["node_tokens"][item["source"]]:
        return "source"
    if pred == cur_token:
        return "repeat"
    if pred in item["node_tokens"]:
        return "other_node"
    return "special"


def step_rows(item, tok):
    src, dst, inverse = endpoints(item)
    outgoing = {}
    for u, v in zip(src.tolist(), dst.tolist()):
        outgoing.setdefault(u, set()).add(item["node_tokens"][v])
    rows = []
    for index in range(len(item["path"]) - 1):
        receiver = item["prompt_len"] + index
        cur = inverse[item["tokens"][receiver]]
        rows.append({
            "receiver": receiver,
            "cur": cur,
            "gold": item["tokens"][receiver + 1],
            "legal": outgoing.get(cur, set()),
            "last": index == len(item["path"]) - 2,
        })
    return rows


def rates(flags: list[str]) -> dict:
    n = max(len(flags), 1)
    keys = ("exact", "other_legal", "target", "source", "repeat", "other_node", "special")
    return {key: float(np.mean([flag == key for flag in flags])) if flags else 0.0 for key in keys} | {"n": len(flags)}


def summarize_size(model, tok, items, device, rng) -> dict:
    flags, last_flags, mid_flags = [], [], []
    changed = {name: [] for name in ("shuffle", "drop", "only")}
    legal_after = {name: [] for name in ("shuffle", "drop", "only")}
    chance = []
    for item in items:
        base = logits_of(model, item["tokens"], device)
        dummy = dummy_token(item, tok)
        n_nodes = len(item["node_tokens"])
        for row in step_rows(item, tok):
            pred = int(base[row["receiver"]].argmax())
            flag = classify(pred, row["gold"], row["legal"], item, item["node_tokens"][row["cur"]])
            flags.append(flag)
            (last_flags if row["last"] else mid_flags).append(flag)
            chance.append(len(row["legal"]) / n_nodes)
            for kind in ("shuffle", "drop", "only"):
                alt_tokens = edited(item["tokens"], item, kind, row["cur"], dummy, rng)
                alt = logits_of(model, alt_tokens, device)
                alt_pred = int(alt[row["receiver"]].argmax())
                changed[kind].append(float(alt_pred != pred))
                if kind == "shuffle":
                    _src, dst, inverse = endpoints({"tokens": alt_tokens, "prompt_len": item["prompt_len"], "node_tokens": item["node_tokens"]})
                    legal = {item["node_tokens"][int(v)] for u, v in zip(_src.tolist(), dst.tolist()) if u == row["cur"]}
                elif kind == "drop":
                    legal = set()
                else:
                    legal = row["legal"]
                legal_after[kind].append(float(alt_pred in legal))
    return {
        "n": len(flags),
        "chance_legal": float(np.mean(chance)),
        "all": rates(flags),
        "intermediate": rates(mid_flags),
        "last_step": rates(last_flags),
        "changed": {kind: float(np.mean(values)) for kind, values in changed.items()},
        "legal_after": {kind: float(np.mean(values)) for kind, values in legal_after.items()},
    }


def draw(blocks: list[dict]) -> None:
    copy = json.loads((ROOT / "studies/block_membership_probe/path_edge_copy_offset.json").read_text())
    direct_head = [0.015, 0.007, 0.000]
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "axes.linewidth": 0.6,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3))
    labels = [block["size"].replace("DAG ", "") for block in blocks]
    x = np.arange(len(blocks))
    width = 0.24
    left = (
        ("Exact outgoing head", [block["intermediate"]["exact"] for block in blocks], "#D55E00"),
        ("Any outgoing head", [block["intermediate"]["exact"] + block["intermediate"]["other_legal"] for block in blocks], "#009E73"),
        ("Chance", [block["chance_legal"] for block in blocks], "#BDBDBD"),
    )
    for offset, (label, values, color) in enumerate(left):
        axes[0].bar(x + (offset - 1) * width, values, width=width, color=color, label=label, linewidth=0)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Share of intermediate steps")
    axes[0].set_title("Next token")
    axes[0].yaxis.grid(True, color="#dddddd", linewidth=0.6)
    axes[0].set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=8)

    right = (
        ("The head token", direct_head, "#0072B2"),
        ("Five tokens later", [float(row["5"]) for row in copy], "#E69F00"),
        ("Six tokens later", [float(row["6"]) for row in copy], "#009E73"),
    )
    for offset, (label, values, color) in enumerate(right):
        axes[1].bar(x + (offset - 1) * width, values, width=width, color=color, label=label, linewidth=0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Attention mass")
    axes[1].set_title("Layer 6, at the path step")
    axes[1].yaxis.grid(True, color="#dddddd", linewidth=0.6)
    axes[1].set_axisbelow(True)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("The next node is copied from five and six tokens after the edge head", fontsize=12)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.12, wspace=0.28)
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
    rng = np.random.default_rng(0)
    saved = []
    for name, root in pathfig.SIZES:
        arrays = pathfig.load_test(root)
        items = choose([pathfig.render_index(arrays, tok, index) for index in range(len(arrays["label"]))])
        block = summarize_size(model, tok, items, device, rng)
        block["size"] = name
        block["step"] = step
        saved.append(block)
        print(f"\n{name} steps={block['n']} chance_legal={block['chance_legal']:.3f}", flush=True)
        for part in ("intermediate", "last_step"):
            row = block[part]
            print(
                f"  {part}: exact {row['exact']:.3f} other_legal {row['other_legal']:.3f} "
                f"target {row['target']:.3f} source {row['source']:.3f} repeat {row['repeat']:.3f} "
                f"other {row['other_node']:.3f} special {row['special']:.3f} n={row['n']}",
                flush=True,
            )
        print(
            "  changed shuffle {shuffle:.3f} drop {drop:.3f} only {only:.3f}".format(**block["changed"]),
            flush=True,
        )
        print(
            "  legal_after shuffle {shuffle:.3f} drop {drop:.3f} only {only:.3f}".format(**block["legal_after"]),
            flush=True,
        )
    draw(saved)
    JSON.parent.mkdir(parents=True, exist_ok=True)
    JSON.write_text(json.dumps(saved, indent=2))
    print(JSON)


if __name__ == "__main__":
    main()

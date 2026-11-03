#!/usr/bin/env python3
"""How curr_source_timed_s0 reads a one-hop edge.

Log: studies/direct_learning, branch curr_source_timed_s0.
Stage 0 fixes the source on 16 names and 2 edges. Later stages release the
query, then the node-name pool (16, 64, 256), then the edge count up to 24.
latest.pt is the wall-budget checkpoint, step 1194957. Frozen inference only.
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

from cot_navigation.config import load_config  # noqa: E402
from cot_navigation.models.transformer import build_model  # noqa: E402
from cot_navigation.protocols.tokenizer import Tokenizer  # noqa: E402

RUN = ROOT / "studies/direct_learning/runs/direct_learning_v1/curr_source_timed_s0"
DATA = ROOT / "studies/direct_learning/data_v1/full_v256_m24.npz"
FIG = ROOT / "figures/direct_learning/source_timed_layers.png"
JSON = ROOT / "studies/direct_learning/results_20261021/source_timed_layers.json"
DEVICE = "cuda:0"
YES, NO = 6, 7
N_LAYERS = 6


def reverse_edges(tokens: np.ndarray) -> np.ndarray:
    body, query = tokens[:, :-4], tokens[:, -4:]
    triples = body.reshape(tokens.shape[0], -1, 3)[:, ::-1, :].reshape(tokens.shape[0], -1)
    return np.concatenate([triples, query], axis=1)


def locate(row: np.ndarray) -> dict:
    source, target = int(row[-3]), int(row[-2])
    triples = row[:-4].reshape(-1, 3)
    exact, src_out, tgt_in, other = [], [], [], []
    for index, (_edge, u, v) in enumerate(triples):
        span = [3 * index, 3 * index + 1, 3 * index + 2]
        if int(u) == source and int(v) == target:
            exact.extend(span)
        elif int(u) == source:
            src_out.extend(span)
        elif int(v) == target:
            tgt_in.extend(span)
        else:
            other.extend(span)
    length = len(row)
    return {
        "source": source,
        "target": target,
        "exact": exact,
        "src_out": src_out,
        "tgt_in": tgt_in,
        "other": other,
        "query_src": length - 3,
        "query_tgt": length - 2,
        "marker": length - 4,
        "bar": length - 1,
    }


def mass(row: np.ndarray, positions: list[int]) -> float:
    if not positions:
        return 0.0
    return float(row[positions].sum())


@torch.inference_mode()
def forward_batch(model, tokens: torch.Tensor):
    capture: dict = {}
    logits, weights = model(tokens, return_attn=True, capture=capture)
    lens = []
    for hidden in capture["resid"]:
        scores = model.lm_head(model.final_norm(hidden))
        lens.append(scores[:, -1].float())
    final = logits[:, -1].float()
    attn = torch.stack([w[:, 0, -1].float() for w in weights], dim=1)
    return final, lens, attn


def scores_to_pred(scores: torch.Tensor) -> torch.Tensor:
    return scores.argmax(dim=-1)


def balanced(pred: np.ndarray, labels: np.ndarray) -> dict:
    yes = labels == 1
    no = ~yes
    yes_acc = float((pred[yes] == YES).mean()) if yes.any() else float("nan")
    no_acc = float((pred[no] == NO).mean()) if no.any() else float("nan")
    return {"yes": yes_acc, "no": no_acc, "bacc": 0.5 * (yes_acc + no_acc), "acc": float((pred == np.where(yes, YES, NO)).mean())}


def bucket_means(attn: np.ndarray, rows: list[dict], key: str, mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    vals = [mass(attn[i], rows[i][key]) for i in np.flatnonzero(mask)]
    return float(np.mean(vals))


@torch.inference_mode()
def masked_predict(model, tokens: torch.Tensor, hide: list[np.ndarray], scope: str) -> np.ndarray:
    """Hide key positions. scope is 'bar' or 'downstream'."""
    originals = []

    def wrap(module):
        original = module.forward

        def fwd(x, positions, cache=None, layer_idx=0, cache_mode=None, return_attn=False, record_keep=None):
            batch, length, _ = x.shape
            index = torch.arange(length, device=x.device)
            keep = (index.view(1, -1) <= index.view(-1, 1)).view(1, 1, length, length).expand(batch, 1, length, length).clone()
            for b, keys in enumerate(hide):
                if len(keys) == 0:
                    continue
                key_index = torch.as_tensor(keys, device=x.device, dtype=torch.long)
                if scope == "bar":
                    keep[b, 0, length - 1, key_index] = False
                else:
                    for key in key_index.tolist():
                        if key + 1 < length:
                            keep[b, 0, key + 1 :, key] = False
            eye = torch.arange(length, device=x.device)
            keep[:, 0, eye, eye] = True
            return original(x, positions, cache, layer_idx, cache_mode, True, keep)

        module.forward = fwd
        originals.append((module, original))

    for block in model.blocks:
        wrap(block.attn)
    try:
        logits, _weights = model(tokens, return_attn=True)
        return scores_to_pred(logits[:, -1].float()).cpu().numpy()
    finally:
        for module, original in originals:
            module.forward = original


def edge_records(row: np.ndarray) -> dict:
    source, target = int(row[-3]), int(row[-2])
    triples = row[:-4].reshape(-1, 3)
    exact = src_out = None
    followers = []
    for index, (_edge, u, v) in enumerate(triples):
        rec = {"index": index, "follower": 3 * index + 3, "source": int(u), "target": int(v),
               "body": [3 * index, 3 * index + 1, 3 * index + 2]}
        followers.append(rec)
        if rec["source"] == source and rec["target"] == target:
            exact = rec
        elif rec["source"] == source and src_out is None:
            src_out = rec
    return {"exact": exact, "src_out": src_out, "edges": followers, "length": len(row)}


def follow_masses(attn_bar: np.ndarray, array: np.ndarray, labels: np.ndarray) -> dict:
    """attn_bar is [B, K] at one layer. Mean mass on the next token after an edge."""
    yes = labels == 1
    exact_m, other_m = [], []
    for i in np.flatnonzero(yes):
        info = edge_records(array[i])
        used = set()
        exact = info["exact"]
        if exact is not None and exact["follower"] < info["length"]:
            exact_m.append(float(attn_bar[i, exact["follower"]]))
            used.add(exact["index"])
        others = [rec["follower"] for rec in info["edges"] if rec["index"] not in used and rec["follower"] < info["length"]]
        if others:
            other_m.append(float(np.mean(attn_bar[i, others])))
    return {"yes_exact": float(np.mean(exact_m)), "yes_other": float(np.mean(other_m))}


def one_example(model, row: np.ndarray, device) -> dict:
    tokens = torch.tensor(row, device=device, dtype=torch.long)[None]
    final, _lens, attn = forward_batch(model, tokens)
    info = locate(row)
    pred = int(scores_to_pred(final).item())
    layers = []
    for layer in range(N_LAYERS):
        weights = attn[0, layer].cpu().numpy()
        top = int(weights.argmax())
        layers.append({
            "exact": mass(weights, info["exact"]),
            "src_out": mass(weights, info["src_out"]),
            "tgt_in": mass(weights, info["tgt_in"]),
            "other": mass(weights, info["other"]),
            "query_src": float(weights[info["query_src"]]),
            "query_tgt": float(weights[info["query_tgt"]]),
            "top": top,
            "top_token": int(row[top]),
            "top_mass": float(weights[top]),
        })
    return {
        "source": int(row[-3]) - 14,
        "target": int(row[-2]) - 14,
        "pred": "YES" if pred == YES else ("NO" if pred == NO else str(pred)),
        "exact_edges": [
            [int(row[p + 1]) - 14, int(row[p + 2]) - 14] for p in info["exact"][::3]
        ],
        "layers": layers,
    }


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
    blue, orange, chance, vermillion = "#0072B2", "#E69F00", "0.45", "#D55E00"
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.7), constrained_layout=True)
    layers = np.arange(1, N_LAYERS + 1)

    ax = axes[0]
    ax.plot(layers, payload["canonical"]["layer_bacc"], color=blue, marker="o", label="Canonical order")
    ax.plot(layers, payload["reverse"]["layer_bacc"], color=orange, marker="s", label="Reversed edges")
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
    groups = ["Canonical", "Reversed"]
    exact = [payload["follow"]["canonical"]["yes_exact"], payload["follow"]["reverse"]["yes_exact"]]
    other = [payload["follow"]["canonical"]["yes_other"], payload["follow"]["reverse"]["yes_other"]]
    xpos = np.arange(len(groups))
    ax.bar(xpos - 0.16, exact, width=0.32, color=blue, label="Next token of the matching edge")
    ax.bar(xpos + 0.16, other, width=0.32, color=orange, label="Next token of another edge")
    for dx, vals in ((-0.16, exact), (0.16, other)):
        for x, val in zip(xpos + dx, vals):
            ax.text(x, val + 0.03, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_xticks(xpos)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Layer-2 attention mass at |")
    ax.set_title("YES queries")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)

    ax = axes[2]
    names = ["Baseline", "Block match\ntoken at L2", "Block another\ntoken at L2", "Block the\nL1 copy"]
    keys = ("baseline", "hide_follower_l2", "hide_other_l2", "block_l1_copy")
    yes = [payload["ablation"][k]["yes"] for k in keys]
    no = [payload["ablation"][k]["no"] for k in keys]
    xpos = np.arange(len(names))
    ax.bar(xpos - 0.16, yes, width=0.32, color=blue, label="YES recall")
    ax.bar(xpos + 0.16, no, width=0.32, color=vermillion, label="NO recall")
    ax.axhline(0.5, color=chance, ls="--", lw=1)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(xpos)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Recall")
    ax.set_title("Causal blocks")
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
    assert tok.yes == YES and tok.no == NO and tok.node_lo == 14
    model = build_model(cfg.model, len(tok), tok.pad).to(device).eval()
    state = torch.load(RUN / "checkpoints/latest.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    step = int(state["step"])
    del state

    blob = np.load(DATA)
    tokens_np = blob["test_x"].astype(np.int64)
    labels = blob["test_y"].astype(np.int64)
    assert tokens_np.shape[1] == 76
    orders = {"canonical": tokens_np, "reverse": reverse_edges(tokens_np)}
    summary = {"step": step, "n": int(len(labels)), "yes": int(labels.sum())}

    stored = {}
    for name, array in orders.items():
        tokens = torch.tensor(array, device=device, dtype=torch.long)
        final, lens, attn = forward_batch(model, tokens)
        attn_np = attn.cpu().numpy()
        rows = [locate(array[i]) for i in range(len(array))]
        yes = labels == 1
        layer_bacc, yes_exact, yes_query, yes_src, yes_tgt, yes_other = [], [], [], [], [], []
        no_src, no_tgt, no_other = [], [], []
        top_in_exact = []
        for layer in range(N_LAYERS):
            pred = scores_to_pred(lens[layer]).cpu().numpy()
            layer_bacc.append(balanced(pred, labels)["bacc"])
            yes_exact.append(bucket_means(attn_np[:, layer], rows, "exact", yes))
            yes_query.append(float(np.mean([
                mass(attn_np[i, layer], [rows[i]["query_src"], rows[i]["query_tgt"]]) for i in np.flatnonzero(yes)
            ])))
            yes_src.append(bucket_means(attn_np[:, layer], rows, "src_out", yes))
            yes_tgt.append(bucket_means(attn_np[:, layer], rows, "tgt_in", yes))
            yes_other.append(bucket_means(attn_np[:, layer], rows, "other", yes))
            no_src.append(bucket_means(attn_np[:, layer], rows, "src_out", ~yes))
            no_tgt.append(bucket_means(attn_np[:, layer], rows, "tgt_in", ~yes))
            no_other.append(bucket_means(attn_np[:, layer], rows, "other", ~yes))
            hits = []
            for i in np.flatnonzero(yes):
                top = int(attn_np[i, layer].argmax())
                hits.append(float(top in rows[i]["exact"]))
            top_in_exact.append(float(np.mean(hits)))
        final_stats = balanced(scores_to_pred(final).cpu().numpy(), labels)
        stored[name] = {
            "follow_l2": follow_masses(attn_np[:, 1], array, labels),
            "final": final_stats,
            "layer_bacc": layer_bacc,
            "yes_exact": yes_exact,
            "yes_query": yes_query,
            "yes_src_out": yes_src,
            "yes_tgt_in": yes_tgt,
            "yes_other": yes_other,
            "no_src_out": no_src,
            "no_tgt_in": no_tgt,
            "no_other": no_other,
            "yes_argmax_in_exact": top_in_exact,
        }
        print(name, "final", final_stats, flush=True)
        print("  lens", ["%.3f" % v for v in layer_bacc], flush=True)
        print("  yes exact", ["%.3f" % v for v in yes_exact], "argmax", ["%.3f" % v for v in top_in_exact], flush=True)
        print("  yes query", ["%.3f" % v for v in yes_query], "src", ["%.3f" % v for v in yes_src],
              "tgt", ["%.3f" % v for v in yes_tgt], "other", ["%.3f" % v for v in yes_other], flush=True)
        print("  no src", ["%.3f" % v for v in no_src], "tgt", ["%.3f" % v for v in no_tgt],
              "other", ["%.3f" % v for v in no_other], flush=True)
        del tokens, final, lens, attn

    canon = orders["canonical"]
    tokens = torch.tensor(canon, device=device, dtype=torch.long)
    follower_hide, other_hide, copy_block = [], [], []
    for row in canon:
        info = edge_records(row)
        exact = info["exact"]
        if exact is not None and exact["follower"] < info["length"]:
            follower_hide.append(np.array([exact["follower"]], dtype=np.int64))
            copy_block.append((exact["follower"], exact["body"][1:]))
        else:
            follower_hide.append(np.array([], dtype=np.int64))
            copy_block.append(None)
        used = {exact["index"]} if exact is not None else set()
        others = [rec["follower"] for rec in info["edges"] if rec["index"] not in used and rec["follower"] < info["length"]]
        other_hide.append(np.array(others[:1], dtype=np.int64))

    def predict_hide(hides, layers, pair=False):
        originals = []

        def wrap(module):
            original = module.forward

            def fwd(x, positions, cache=None, layer_idx=0, cache_mode=None, return_attn=False, record_keep=None):
                batch, length, _ = x.shape
                index = torch.arange(length, device=x.device)
                keep = (index.view(1, -1) <= index.view(-1, 1)).view(1, 1, length, length).expand(batch, 1, length, length).clone()
                if layer_idx in layers:
                    for b, spec in enumerate(hides):
                        if spec is None or len(spec) == 0:
                            continue
                        if pair:
                            queries, keys = spec
                            queries = [queries]
                        else:
                            queries, keys = [length - 1], spec
                        key_index = torch.as_tensor(list(keys), device=x.device)
                        for query in queries:
                            if 0 <= query < length:
                                keep[b, 0, query, key_index] = False
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
            return scores_to_pred(logits[:, -1].float()).cpu().numpy()
        finally:
            for module, original in originals:
                module.forward = original

    empty = [np.array([], dtype=np.int64) for _ in range(len(canon))]
    ablation = {
        "baseline": balanced(predict_hide(empty, {1}), labels),
        "hide_follower_l2": balanced(predict_hide(follower_hide, {1}), labels),
        "hide_other_l2": balanced(predict_hide(other_hide, {1}), labels),
        "block_l1_copy": balanced(predict_hide(copy_block, {0}, pair=True), labels),
    }
    for key, stats in ablation.items():
        print("ablation", key, stats, flush=True)

    with torch.inference_mode():
        _logits, weights = model(tokens, return_attn=True)
        layer1 = weights[0][:, 0].float().cpu().numpy()
    copy_src, copy_dst = [], []
    for i, row in enumerate(canon):
        info = edge_records(row)
        exact = info["exact"]
        if labels[i] != 1 or exact is None or exact["follower"] >= len(row):
            continue
        row_attn = layer1[i, exact["follower"]]
        copy_src.append(float(row_attn[exact["body"][1]]))
        copy_dst.append(float(row_attn[exact["body"][2]]))
    copy = {"source": float(np.mean(copy_src)), "destination": float(np.mean(copy_dst)), "n": len(copy_src)}
    print("L1 follower attends to edge", copy, flush=True)

    example_index = int(np.flatnonzero(labels == 1)[0])
    example = one_example(model, canon[example_index], device)
    info = edge_records(canon[example_index])
    exact = info["exact"]
    example["follower_pos"] = exact["follower"]
    example["follower_token"] = int(canon[example_index, exact["follower"]])
    example["edge_tokens"] = [int(v) - 14 for v in canon[example_index, exact["body"][1:]]]
    example["l1_on_source"] = float(layer1[example_index, exact["follower"], exact["body"][1]])
    example["l1_on_destination"] = float(layer1[example_index, exact["follower"], exact["body"][2]])
    example["l2_on_follower"] = float(example["layers"][1]["top_mass"]) if example["layers"][1]["top"] == exact["follower"] else None
    # The stored layer dump is bar-only; record the follower mass directly.
    bar_l2 = None
    with torch.inference_mode():
        _final, _lens, bar = forward_batch(model, torch.tensor(canon[example_index], device=device)[None])
        bar_l2 = float(bar[0, 1, exact["follower"]])
    example["l2_on_follower"] = bar_l2
    print("example", example_index, example, flush=True)

    payload = {
        "checkpoint": str(RUN / "checkpoints/latest.pt"),
        "step": step,
        "task": "one-hop edge lookup, 256 names, 24 edges, held-out graphs",
        "canonical": stored["canonical"],
        "reverse": stored["reverse"],
        "follow": {name: stored[name]["follow_l2"] for name in stored},
        "l1_copy": copy,
        "ablation": ablation,
        "example": {k: v for k, v in example.items() if k != "layers"},
    }
    summary.update(payload)
    draw(payload)
    JSON.parent.mkdir(parents=True, exist_ok=True)
    JSON.write_text(json.dumps(summary, indent=2))
    print(FIG)
    print(JSON)


if __name__ == "__main__":
    main()

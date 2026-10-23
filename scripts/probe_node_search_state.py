#!/usr/bin/env python3
"""Linear probes for latent search state in DFS-node and BFS-node models.

Node traces emit only the first-expansion order. Immediately before each of
those tokens, the residual at the previous position is the activation that
predicts it. Two reference snapshots are read from the same supervised trace
(prepared priority variant 0, including its shuffled neighbor order):

* expansion — the pre-action step at which the upcoming token is current.
  The stack top / queue head and the visited set already contain that node.
  Candidates are its unvisited out-neighbors and do not contain it.
* selection — the PUSH step that chose the node. The frontier, visited set
  and all but the first candidate exclude it. The source node has no PUSH;
  its selection snapshot is the expansion snapshot.

Controls are the multihot of nodes already emitted, the previous-token
embedding, within-graph shuffled labels, and an oracle that sees the source
embedding together with the embedding of the node about to be emitted. On one
fixed graph the expansion state is a function of that pair, so the oracle
separates "the residual names the next node" from "the residual carries
history the node identity does not".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from io import BytesIO
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6")
os.environ.setdefault("WANDB_MODE", "disabled")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))

from cot_navigation.analysis.common import SEED, write_json  # noqa: E402
from cot_navigation.analysis.probes import fit, shuffled_labels  # noqa: E402
from cot_navigation.analysis.scale_followup import make_adapter  # noqa: E402
from cot_navigation.analysis.set_metrics import set_quality, scores  # noqa: E402
from cot_navigation.protocols.bfs import bfs_search, first_expansion_order  # noqa: E402
from cot_navigation.protocols.dfs import dfs_search  # noqa: E402

OUT = ROOT / "research/node_search_state_probe_20261021/results.json"
FIG = ROOT / "figures/scale_v1/node_search_state_probe.png"
FIG_LAYERS = ROOT / "figures/scale_v1/node_search_state_layers.png"
FIG_SELECTION = ROOT / "figures/scale_v1/node_search_state_selection.png"
FIG_10X10 = ROOT / "figures/scale_v1/node_search_state_10x10.png"
FIG_10X10_SELECTION = ROOT / "figures/scale_v1/node_search_state_10x10_selection.png"
FOCUS = (
    ("fixed", "l10", "Fixed 10×10"),
    ("context", "curriculum", "In-context curriculum"),
)

JOBS = [
    ("fixed", "l10", "dfs_nodes", "Fixed N100"),
    ("context", "l10", "dfs_nodes", "In-context N100"),
    ("fixed", "curriculum", "dfs_nodes", "Fixed curriculum"),
    ("context", "curriculum", "dfs_nodes", "In-context curriculum"),
    ("fixed", "l10", "bfs_nodes", "Fixed N100"),
    ("context", "l10", "bfs_nodes", "In-context N100"),
    ("fixed", "curriculum", "bfs_nodes", "Fixed curriculum"),
    ("context", "curriculum", "bfs_nodes", "In-context curriculum"),
]
EXPANSION_TASKS = (
    "expansion_candidates",
    "expansion_visited",
    "expansion_frontier",
    "expansion_frontier_rest",
    "expansion_second",
    "expansion_neighbor_candidates",
)
SELECTION_TASKS = (
    "selection_candidates",
    "selection_candidates_rest",
    "selection_visited",
    "selection_frontier",
)
PREFIX_TASKS = (
    "expansion_candidates",
    "expansion_visited",
    "expansion_frontier",
    "expansion_frontier_rest",
    "selection_candidates_rest",
    "selection_frontier",
)
ORACLE_TASKS = (
    "expansion_candidates",
    "expansion_visited",
    "expansion_frontier_rest",
    "expansion_neighbor_candidates",
    "selection_candidates_rest",
    "selection_frontier",
)
PLOT_TARGETS = (
    ("expansion_candidates", "Candidates"),
    ("expansion_visited", "Visited"),
    ("expansion_frontier_rest", "Rest of stack/queue"),
)


def run_dir(kind: str, scale: str, protocol: str) -> Path:
    base = ROOT / "outputs/scale_v1/launches/scale_v1_s0" / kind
    if kind == "fixed":
        return base / scale / "stage_b" / protocol
    return base / scale / protocol


def shuffled_steps(adj, src: int, dst: int, protocol: str, index: int, variant: int = 0):
    """Replay the offline priority used by ScalePreparedRows variant 0."""
    seed = int(hashlib.sha256(protocol.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed + int(index) * 104729 + variant)
    shuffled = [rng.permutation(list(vs)).astype(int).tolist() for vs in adj]
    search = bfs_search if protocol.startswith("bfs") else dfs_search
    return search(shuffled, src, dst)


def encode(topo_ids, nodes, n_vocab: int, lo: int) -> np.ndarray:
    y = np.zeros(n_vocab, np.float32)
    for raw in topo_ids:
        y[nodes[int(raw)] - lo] = 1.0
    return y


def graph_mask(nodes, n_vocab: int, lo: int) -> np.ndarray:
    mask = np.zeros(n_vocab, np.float32)
    for token in nodes:
        mask[int(token) - lo] = 1.0
    return mask


def empty_identity() -> dict:
    keys = (
        "later", "first", "cand_n", "vis_n", "front_n", "rest_n",
        "nonempty_cand", "nonempty_rest", "has_second",
        "vis_eq_prefix_plus_cur", "rest_eq_latent_visited",
        "sel_cand_ne_exp", "sel_front_ne_exp",
    )
    return {key: 0 for key in keys}


def position_records(ex, tok, protocol: str):
    """One row per expansion node, labeled at the position that predicts it."""
    fifo = protocol.startswith("bfs")
    steps = shuffled_steps(ex.adj, ex.source, ex.target, protocol, ex.index)
    expanded = []
    seen = set()
    for step in steps:
        if step.cur not in seen:
            seen.add(step.cur)
            expanded.append(step)
    order = first_expansion_order(steps)
    if [step.cur for step in expanded] != order:
        raise AssertionError("expansion alignment drifted from first_expansion_order")
    body = ex.row.tokens[ex.row.prompt_len:ex.row.answer_index]
    stop = tok.id("<STOP>")
    if stop in body:
        body = body[:body.index(stop)]
    expected = [ex.nodes[node] for node in order[:len(body)]]
    if body != expected:
        raise AssertionError(
            f"query {ex.index}: trace {tok.decode(body[:8])} != reference {tok.decode(expected[:8])}"
        )
    pushes = {}
    for step in steps:
        if step.action == "PUSH":
            if step.arg in pushes:
                raise AssertionError("node pushed twice")
            pushes[step.arg] = step
    n_vocab, lo = tok.n_node_tokens, tok.node_lo
    mask = graph_mask(ex.nodes, n_vocab, lo)
    rows = []
    stats = empty_identity()
    for k, step in enumerate(expanded[:len(body)]):
        readout = ex.row.prompt_len + k - 1
        if ex.row.tokens[readout + 1] != ex.nodes[step.cur]:
            raise AssertionError("readout does not predict this expansion node")
        head = step.stack[0] if fifo else step.stack[-1]
        if head != step.cur:
            raise AssertionError("frontier head is not the current node")
        if k == 0:
            selected = step
        else:
            selected = pushes[step.cur]
            if selected.cands[0] != step.cur or step.cur in selected.visited or step.cur in selected.stack:
                raise AssertionError("selection step does not choose this node")
        prefix_ids = [earlier.cur for earlier in expanded[:k]]
        neighbor = np.zeros(n_vocab, np.float32)
        for raw in ex.adj[step.cur]:
            neighbor[ex.nodes[int(raw)] - lo] = 1.0
        second = None
        if len(step.stack) >= 2:
            second = step.stack[1] if fifo else step.stack[-2]
        labels = {
            "expansion_candidates": encode(step.cands, ex.nodes, n_vocab, lo),
            "expansion_visited": encode(step.visited, ex.nodes, n_vocab, lo),
            "expansion_frontier": encode(step.stack, ex.nodes, n_vocab, lo),
            "expansion_frontier_rest": encode(
                [raw for raw in step.stack if raw != step.cur], ex.nodes, n_vocab, lo
            ),
            "expansion_neighbor_candidates": encode(step.cands, ex.nodes, n_vocab, lo),
            "selection_candidates": encode(selected.cands, ex.nodes, n_vocab, lo),
            "selection_candidates_rest": encode(
                [raw for raw in selected.cands if raw != step.cur], ex.nodes, n_vocab, lo
            ),
            "selection_visited": encode(selected.visited, ex.nodes, n_vocab, lo),
            "selection_frontier": encode(selected.stack, ex.nodes, n_vocab, lo),
        }
        masks = {name: mask for name in labels}
        masks["expansion_neighbor_candidates"] = neighbor
        if second is not None:
            labels["expansion_second"] = encode([second], ex.nodes, n_vocab, lo)
            masks["expansion_second"] = mask
        rows.append({
            "readout": readout,
            "stage": "first" if k == 0 else "later",
            "query": int(ex.index),
            "prefix": encode(prefix_ids, ex.nodes, n_vocab, lo),
            "source_token": int(ex.nodes[ex.source]),
            "next_token": int(ex.nodes[step.cur]),
            "labels": labels,
            "masks": masks,
        })
        if k == 0:
            stats["first"] += 1
            continue
        prefix, visited, frontier = set(prefix_ids), set(step.visited), set(step.stack)
        stats["later"] += 1
        stats["cand_n"] += len(step.cands)
        stats["vis_n"] += len(visited)
        stats["front_n"] += len(frontier)
        stats["rest_n"] += len(frontier - {step.cur})
        stats["nonempty_cand"] += int(len(step.cands) > 0)
        stats["nonempty_rest"] += int(len(frontier - {step.cur}) > 0)
        stats["has_second"] += int(second is not None)
        stats["vis_eq_prefix_plus_cur"] += int(visited == prefix | {step.cur})
        stats["rest_eq_latent_visited"] += int(frontier - {step.cur} == visited - prefix - {step.cur})
        stats["sel_cand_ne_exp"] += int(set(selected.cands) != set(step.cands))
        stats["sel_front_ne_exp"] += int(set(selected.stack) != frontier)
    return rows, stats


def add_stats(total: dict, part: dict) -> None:
    for key, value in part.items():
        total[key] += value


def identity_report(stats: dict) -> dict:
    later = max(1, stats["later"])
    return {
        "n_first": stats["first"],
        "n_later": stats["later"],
        "mean_candidates": stats["cand_n"] / later,
        "mean_visited": stats["vis_n"] / later,
        "mean_frontier": stats["front_n"] / later,
        "mean_frontier_rest": stats["rest_n"] / later,
        "fraction_nonempty_candidates": stats["nonempty_cand"] / later,
        "fraction_nonempty_frontier_rest": stats["nonempty_rest"] / later,
        "fraction_with_second": stats["has_second"] / later,
        "fraction_visited_equals_prefix_plus_next": stats["vis_eq_prefix_plus_cur"] / later,
        "fraction_frontier_rest_equals_latent_visited": stats["rest_eq_latent_visited"] / later,
        "fraction_selection_candidates_differ": stats["sel_cand_ne_exp"] / later,
        "fraction_selection_frontier_differs": stats["sel_front_ne_exp"] / later,
    }


def collect(am, examples, batch: int = 8):
    grouped = []
    stats = empty_identity()
    for ex in examples:
        rows, part = position_records(ex, am.tok, am.protocol)
        add_stats(stats, part)
        grouped.append((ex, rows))
    flat_rows = [row for _, rows in grouped for row in rows]
    if not flat_rows:
        raise ValueError("no expansion nodes")
    n_layers = len(am.layers)
    width = am.model.d_model + 4
    features = np.empty((len(flat_rows), n_layers + 1, width), np.float32)
    cursor = 0
    for start in range(0, len(grouped), batch):
        chunk = grouped[start:start + batch]
        sequences = [ex.row.tokens[:ex.row.answer_index] for ex, _ in chunk]
        _, residual, _ = am.forward(sequences)
        for b, (ex, rows) in enumerate(chunk):
            for row in rows:
                pos = row["readout"]
                hidden = torch.stack([residual[layer][b, pos].float() for layer in am.layers])
                hidden = F.pad(hidden, (0, 4))
                token = sequences[b][pos]
                embedding = am.model.embed.weight[token].float()
                seen = ex.nodes[ex.target] in sequences[b][ex.row.prompt_len:pos + 1]
                observable = embedding.new_tensor([
                    pos / am.model.max_seq_len,
                    (pos - ex.row.prompt_len) / am.model.max_seq_len,
                    float(seen),
                    float(token == ex.nodes[ex.target]),
                ])
                baseline = torch.cat([embedding, observable])
                features[cursor] = torch.cat([hidden, baseline[None]], 0).cpu().numpy()
                cursor += 1
        del residual
    if cursor != len(flat_rows):
        raise AssertionError("feature rows drifted from labels")
    prefix = np.stack([row["prefix"] for row in flat_rows])
    weight = am.model.embed.weight.detach().float().cpu().numpy()
    source_tok = np.asarray([row["source_token"] for row in flat_rows])
    next_tok = np.asarray([row["next_token"] for row in flat_rows])
    oracle = np.concatenate([weight[source_tok], weight[next_tok]], axis=1).astype(np.float32)
    query = np.asarray([row["query"] for row in flat_rows], np.int64)
    stage = np.asarray([row["stage"] for row in flat_rows])
    packed = {}
    for name in EXPANSION_TASKS + SELECTION_TASKS:
        keep = [i for i, row in enumerate(flat_rows) if name in row["labels"]]
        if not keep:
            continue
        idx = np.asarray(keep)
        packed[name] = {
            "x": features[idx],
            "prefix": prefix[idx, None, :],
            "oracle": oracle[idx, None, :],
            "y": np.stack([flat_rows[i]["labels"][name] for i in keep]),
            "mask": np.stack([flat_rows[i]["masks"][name] for i in keep]),
            "query": query[idx],
            "stage": stage[idx],
        }
    return packed, identity_report(stats)


def slim(quality: dict) -> dict:
    micro = quality.get("micro_ranking") or {}
    kept = {
        "n_rows": quality.get("n_rows"),
        "nontrivial_rows": quality.get("nontrivial_rows"),
        "n_positive": quality.get("n_positive"),
        "pairwise_auc": quality.get("pairwise_auc"),
        "topk_exact": quality.get("topk_exact"),
        "topk_recall": quality.get("topk_recall"),
        "strict_separation": quality.get("strict_separation"),
        "positive_recall_at_half": quality.get("positive_recall_at_half"),
        "micro_auroc": micro.get("auroc"),
        "micro_average_precision": micro.get("average_precision"),
    }
    return kept


def summarize(fitted, data) -> list[dict]:
    logits = scores(fitted, data)
    layers = []
    for layer in range(logits.shape[1]):
        item = {"all": slim(set_quality(logits[:, layer], data["y"], data["mask"]))}
        for stage in ("first", "later"):
            sel = data["stage"] == stage
            if np.any(sel):
                item[stage] = slim(set_quality(logits[sel, layer], data["y"][sel], data["mask"][sel]))
        layers.append(item)
    return layers


def run_task(train, val, test, *, steps: int, shuffled: bool = False):
    seed = SEED + (17 if shuffled else 0)
    fitted = fit(train, val, device="cuda:0", seed=seed, steps=steps, shuffled=shuffled)
    if shuffled:
        evaluated = {**test, "y": shuffled_labels(test, seed + 202)}
    else:
        evaluated = test
    report = summarize(fitted, evaluated)
    del fitted
    return report


def auc_of(report, layer: int, stage: str = "later"):
    if report is None or layer < 0 or layer >= len(report):
        return None
    return report[layer].get(stage, {}).get("pairwise_auc")


def fmt(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def exact_of(report, layer: int, stage: str = "later"):
    if report is None or layer >= len(report):
        return None
    return report[layer].get(stage, {}).get("topk_exact")


def probe_model(am, counts: dict, steps: int):
    splits = {name: am.examples(name, count) for name, count in counts.items()}
    ids = {name: [ex.index for ex in rows] for name, rows in splits.items()}
    if set(ids["train"]) & set(ids["val"]) or set(ids["train"]) & set(ids["test"]) or set(ids["val"]) & set(ids["test"]):
        raise AssertionError("probe splits share queries")
    features = {}
    identity = {}
    for name, rows in splits.items():
        print(f"  collect {name} n={len(rows)}", flush=True)
        features[name], identity[name] = collect(am, rows)
        print(
            f"    later={identity[name]['n_later']} "
            f"visited=prefix+next {identity[name]['fraction_visited_equals_prefix_plus_next']:.3f} "
            f"rest=latent-visited {identity[name]['fraction_frontier_rest_equals_latent_visited']:.3f}",
            flush=True,
        )
    tasks = {}
    names = [name for name in EXPANSION_TASKS + SELECTION_TASKS if name in features["train"]]
    for name in names:
        print(f"  probe {name}", flush=True)
        bundle = {split: features[split][name] for split in splits}
        residual = run_task(bundle["train"], bundle["val"], bundle["test"], steps=steps)
        shuffled = run_task(bundle["train"], bundle["val"], bundle["test"], steps=steps, shuffled=True)
        entry = {"residual": residual, "shuffled": shuffled}
        if name in PREFIX_TASKS:
            prefixed = {
                split: {**bundle[split], "x": bundle[split]["prefix"]}
                for split in splits
            }
            entry["prefix_bag"] = run_task(prefixed["train"], prefixed["val"], prefixed["test"], steps=steps)
        if name in ORACLE_TASKS:
            oracled = {
                split: {**bundle[split], "x": bundle[split]["oracle"]}
                for split in splits
            }
            entry["oracle"] = run_task(oracled["train"], oracled["val"], oracled["test"], steps=steps)
        tasks[name] = entry
        last = len(residual) - 2
        print(
            f"    L6 {fmt(auc_of(residual, last))}  "
            f"input {fmt(auc_of(residual, last + 1))}  "
            f"oracle {fmt(auc_of(entry.get('oracle'), 0))}  "
            f"shuffle {fmt(auc_of(shuffled, last))}  "
            f"exact {fmt(exact_of(residual, last))}",
            flush=True,
        )
    return {
        "checkpoint": am.checkpoint,
        "world": str(am.store.root),
        "query_ids": ids,
        "n_queries": {name: len(rows) for name, rows in splits.items()},
        "identity": identity,
        "layers": [f"L{i + 1}" for i in am.layers] + ["previous_token"],
        "tasks": tasks,
    }


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#EEEEEE", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)


def save_rgb(fig, path: Path):
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=200, facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.open(buffer).convert("RGB").save(path, format="PNG", optimize=True)
    print(path, flush=True)


def plot(doc: dict):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })
    protocols = [("dfs_nodes", "DFS nodes"), ("bfs_nodes", "BFS nodes")]
    settings = []
    for kind, scale, protocol, label in JOBS:
        if protocol == "dfs_nodes":
            settings.append((f"{kind}_{scale}_dfs_nodes", label))
    series = (
        ("l6", "L6 activation", "#0072B2", "-"),
        ("oracle", "Oracle: source + next", "#E69F00", "-."),
        ("prefix", "Emitted-node set", "#009E73", "--"),
        ("token", "Previous token", "#D55E00", ":"),
    )
    fig, axes = plt.subplots(2, 4, figsize=(12.4, 6.2), sharey=True)
    xs = np.arange(len(PLOT_TARGETS))
    for row, (protocol, row_title) in enumerate(protocols):
        for col, ((model_id, label), ax) in enumerate(zip(settings, axes[row])):
            model_id = model_id.replace("dfs_nodes", protocol)
            tasks = doc["models"][model_id]["tasks"]
            ax.axhline(0.5, color="#9A9A9A", lw=0.8, ls="--", zorder=0)
            for kind, title, color, linestyle in series:
                ys = []
                for key, _ in PLOT_TARGETS:
                    if kind == "prefix":
                        report, layer = tasks[key]["prefix_bag"], 0
                    elif kind == "oracle":
                        report, layer = tasks[key]["oracle"], 0
                    elif kind == "token":
                        report = tasks[key]["residual"]
                        layer = len(report) - 1
                    else:
                        report = tasks[key]["residual"]
                        layer = len(report) - 2
                    value = auc_of(report, layer)
                    ys.append(np.nan if value is None else value)
                ax.plot(xs, ys, color=color, marker="o", linestyle=linestyle, linewidth=1.6,
                        markersize=4.5, label=title, zorder=3)
            style_ax(ax)
            ax.set_xlim(-0.2, len(PLOT_TARGETS) - 0.8)
            ax.set_ylim(0.35, 1.05)
            ax.set_xticks(xs, [label for _, label in PLOT_TARGETS], fontsize=7.5, rotation=18, ha="right")
            ax.set_title(f"{row_title.split()[0]} · {label}", fontsize=10, pad=5)
            if col == 0:
                ax.set_ylabel("Pairwise AUC", fontsize=9)
            if row == 0 and col == 3:
                ax.legend(frameon=True, fontsize=7, loc="lower right", framealpha=0.94,
                          edgecolor="none", fancybox=False)
    fig.suptitle("Search state before the next expansion node", fontsize=13)
    fig.tight_layout()
    save_rgb(fig, FIG)

    fig, axes = plt.subplots(2, 4, figsize=(12.4, 6.2), sharey=True)
    colors = {"expansion_candidates": "#0072B2", "expansion_visited": "#009E73",
              "expansion_frontier_rest": "#D55E00"}
    for row, (protocol, _) in enumerate(protocols):
        for col, (model_id, label) in enumerate(settings):
            ax = axes[row, col]
            model_id = model_id.replace("dfs_nodes", protocol)
            tasks = doc["models"][model_id]["tasks"]
            names = doc["models"][model_id]["layers"]
            residual_names = names[:-1]
            ax.axhline(0.5, color="#9A9A9A", lw=0.8, ls="--", zorder=0)
            for key, title in PLOT_TARGETS:
                report = tasks[key]["residual"]
                ys = [np.nan if (value := auc_of(report, i)) is None else value
                      for i in range(len(residual_names))]
                ax.plot(np.arange(1, len(ys) + 1), ys, color=colors[key], marker="o",
                        linewidth=1.6, markersize=4, label=title)
            style_ax(ax)
            ax.set_ylim(0.35, 1.05)
            ax.set_xticks(np.arange(1, len(residual_names) + 1))
            ax.set_title(f"{protocol.split('_')[0].upper()} · {label}", fontsize=10, pad=5)
            if col == 0:
                ax.set_ylabel("Pairwise AUC", fontsize=9)
            if row == 1:
                ax.set_xlabel("Layer", fontsize=9)
            if row == 0 and col == 3:
                ax.legend(frameon=True, fontsize=7, loc="lower right", framealpha=0.94,
                          edgecolor="none", fancybox=False)
    fig.suptitle("Where the expansion-state readout appears", fontsize=13)
    fig.tight_layout()
    save_rgb(fig, FIG_LAYERS)
    plot_selection(doc)


def plot_selection(doc: dict):
    """Choice-set readout. Fixed graphs sit at ceiling; one in-context model does not."""
    labels = []
    l6, bag, token, oracle = [], [], [], []
    for kind, scale, protocol, setting in JOBS:
        model_id = f"{kind}_{scale}_{protocol}"
        short = "DFS" if protocol == "dfs_nodes" else "BFS"
        where = {"l10": "N100", "curriculum": "curriculum"}[scale]
        knowledge = "Fixed" if kind == "fixed" else "In-context"
        labels.append(f"{short}\n{knowledge}\n{where}")
        task = doc["models"][model_id]["tasks"]["selection_candidates_rest"]
        residual = task["residual"]
        l6.append(auc_of(residual, len(residual) - 2))
        token.append(auc_of(residual, len(residual) - 1))
        bag.append(auc_of(task["prefix_bag"], 0))
        oracle.append(auc_of(task["oracle"], 0))
    fig, ax = plt.subplots(figsize=(10.2, 4.4))
    xs = np.arange(len(labels))
    width = 0.18
    series = (
        (l6, -1.5, "L6 activation", "#0072B2"),
        (oracle, -0.5, "Oracle: source + next", "#E69F00"),
        (bag, 0.5, "Emitted-node set", "#009E73"),
        (token, 1.5, "Previous token", "#D55E00"),
    )
    ax.axhline(0.5, color="#9A9A9A", lw=0.8, ls="--", zorder=0)
    for values, offset, title, color in series:
        ax.bar(xs + offset * width, values, width=width, color=color, label=title, zorder=3)
    style_ax(ax)
    ax.set_ylabel("Pairwise AUC", fontsize=10)
    ax.set_ylim(0.35, 1.05)
    ax.set_xticks(xs, labels, fontsize=8)
    ax.set_title("Unchosen candidates when the next node is selected", fontsize=12, pad=8)
    ax.legend(frameon=True, fontsize=8, ncol=4, loc="upper right", framealpha=0.94,
              edgecolor="none", fancybox=False)
    fig.tight_layout()
    save_rgb(fig, FIG_SELECTION)
    plot_focus(doc)


def plot_focus(doc: dict):
    """Fixed 10×10 graph models and in-context curriculum models only."""
    protocols = [("dfs_nodes", "DFS nodes"), ("bfs_nodes", "BFS nodes")]
    series = (
        ("l6", "L6 activation", "#0072B2", "-"),
        ("oracle", "Oracle: source + next", "#E69F00", "-."),
        ("prefix", "Emitted-node set", "#009E73", "--"),
        ("token", "Previous token", "#D55E00", ":"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.8, 5.8), sharey=True)
    xs = np.arange(len(PLOT_TARGETS))
    for row, (protocol, row_title) in enumerate(protocols):
        for col, (kind, scale, label) in enumerate(FOCUS):
            ax = axes[row, col]
            tasks = doc["models"][f"{kind}_{scale}_{protocol}"]["tasks"]
            ax.axhline(0.5, color="#9A9A9A", lw=0.8, ls="--", zorder=0)
            for feature, title, color, linestyle in series:
                ys = []
                for key, _ in PLOT_TARGETS:
                    if feature == "prefix":
                        report, layer = tasks[key]["prefix_bag"], 0
                    elif feature == "oracle":
                        report, layer = tasks[key]["oracle"], 0
                    elif feature == "token":
                        report = tasks[key]["residual"]
                        layer = len(report) - 1
                    else:
                        report = tasks[key]["residual"]
                        layer = len(report) - 2
                    value = auc_of(report, layer)
                    ys.append(np.nan if value is None else value)
                ax.plot(xs, ys, color=color, marker="o", linestyle=linestyle, linewidth=1.6,
                        markersize=5, label=title, zorder=3)
            style_ax(ax)
            ax.set_xlim(-0.2, len(PLOT_TARGETS) - 0.8)
            ax.set_ylim(0.35, 1.05)
            ax.set_xticks(xs, [name for _, name in PLOT_TARGETS], fontsize=8)
            ax.set_title(f"{row_title} · {label}", fontsize=11, pad=6)
            if col == 0:
                ax.set_ylabel("Pairwise AUC", fontsize=10)
            if row == 0 and col == 1:
                ax.legend(frameon=True, fontsize=7.5, loc="center right", framealpha=0.94,
                          edgecolor="none", fancybox=False)
    fig.suptitle("10×10 search state before the next node", fontsize=13)
    fig.tight_layout()
    save_rgb(fig, FIG_10X10)

    labels, l6, bag, token, oracle = [], [], [], [], []
    for protocol, short in (("dfs_nodes", "DFS"), ("bfs_nodes", "BFS")):
        for kind, scale, label in FOCUS:
            labels.append(f"{short}\n{label.replace('10×10', '10x10')}")
            task = doc["models"][f"{kind}_{scale}_{protocol}"]["tasks"]["selection_candidates_rest"]
            residual = task["residual"]
            l6.append(auc_of(residual, len(residual) - 2))
            token.append(auc_of(residual, len(residual) - 1))
            bag.append(auc_of(task["prefix_bag"], 0))
            oracle.append(auc_of(task["oracle"], 0))
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    xs = np.arange(len(labels))
    width = 0.18
    ax.axhline(0.5, color="#9A9A9A", lw=0.8, ls="--", zorder=0)
    for values, offset, title, color in (
        (l6, -1.5, "L6 activation", "#0072B2"),
        (oracle, -0.5, "Oracle: source + next", "#E69F00"),
        (bag, 0.5, "Emitted-node set", "#009E73"),
        (token, 1.5, "Previous token", "#D55E00"),
    ):
        ax.bar(xs + offset * width, values, width=width, color=color, label=title, zorder=3)
    style_ax(ax)
    ax.set_ylabel("Pairwise AUC", fontsize=10)
    ax.set_ylim(0.35, 1.05)
    ax.set_xticks(xs, labels, fontsize=8)
    ax.set_title("Unchosen candidates when the next node is selected", fontsize=12, pad=28)
    ax.legend(frameon=True, fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              framealpha=0.94, edgecolor="none", fancybox=False)
    fig.tight_layout()
    save_rgb(fig, FIG_10X10_SELECTION)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=int, default=256)
    parser.add_argument("--val", type=int, default=96)
    parser.add_argument("--test", type=int, default=128)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--align-only", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        plot(json.loads(OUT.read_text()))
        return
    torch.set_num_threads(4)
    doc = json.loads(OUT.read_text()) if OUT.exists() else {
        "seed": SEED,
        "definition": {
            "readout": "residual after each block at the position that predicts the next expansion node",
            "trace": "prepared priority variant 0, teacher-forced, world token included for fixed graphs",
            "expansion": "pre-action state when that node is current; frontier and visited contain it; candidates do not",
            "selection": "pre-action PUSH state that chose the node; frontier and visited exclude it",
            "frontier_rest": "expansion frontier without the node about to be emitted",
            "controls": "previous-token embedding, multihot of already emitted nodes, within-graph shuffled labels, oracle source+next embeddings",
            "metric": "mean within-position pairwise AUC on later nodes; chance is 0.5",
        },
        "models": {},
    }
    counts = {"train": args.train, "val": args.val, "test": args.test}
    for kind, scale, protocol, _label in JOBS:
        model_id = f"{kind}_{scale}_{protocol}"
        if args.models and model_id not in args.models:
            continue
        if model_id in doc["models"] and not args.align_only:
            print("skip", model_id, flush=True)
            continue
        print("MODEL", model_id, flush=True)
        entry = {"id": model_id, "run": str(run_dir(kind, scale, protocol).relative_to(ROOT))}
        am = make_adapter(entry, ROOT, "cuda:0")
        am.layers = list(range(am.model.n_layers))
        if args.align_only:
            rows = am.examples("test", min(32, args.test))
            stats = empty_identity()
            for ex in rows:
                _records, part = position_records(ex, am.tok, am.protocol)
                add_stats(stats, part)
            print(json.dumps(identity_report(stats), indent=2), flush=True)
            del am
            torch.cuda.empty_cache()
            continue
        doc["models"][model_id] = probe_model(am, counts, args.steps)
        doc["counts"] = counts
        doc["steps"] = args.steps
        write_json(OUT, doc)
        del am
        torch.cuda.empty_cache()
    if not args.align_only and len(doc["models"]) == len(JOBS):
        plot(doc)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

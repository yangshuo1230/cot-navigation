#!/usr/bin/env python3
"""Probe search state using the model's own generated prefix as ground truth.

The residual that predicts the next node has seen only the prompt and the
nodes emitted so far. Labels are the search state implied by that prefix and
the query graph. No reference variant and no future token enters the label.

DFS pops while the stack top has no unused out-neighbor. That backtracking is
fixed by the prefix. The state at the next choice is then unique: the stack,
the visited set (the prefix), and the candidate set (unused neighbors of the
top). The first token is the source itself; its label is the initial state,
and the reported numbers use later positions.

BFS enqueues every unvisited out-neighbor, so the visited set and the queue
set are fixed by the prefix. Sibling order is not. Nodes enqueued together
stay in one cohort, and only the earliest remaining cohort can be written
next. That cohort is the candidate label. The child list of "the" queue head
is not a label unless the cohort has one node, because the prefix does not
decide which member is head.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from io import BytesIO
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ.setdefault("WANDB_MODE", "disabled")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cot_navigation.analysis.common import SEED, write_json  # noqa: E402
from cot_navigation.analysis.probes import fit, shuffled_labels  # noqa: E402
from cot_navigation.analysis.scale_followup import generate, make_adapter  # noqa: E402
from cot_navigation.analysis.set_metrics import scores, set_quality  # noqa: E402
from cot_navigation.protocols.bfs import bfs_search, first_expansion_order  # noqa: E402
from cot_navigation.protocols.dfs import dfs_search  # noqa: E402

OUT = ROOT / "studies/node_search_state_generated/results.json"
FIG = ROOT / "figures/scale_v1/node_search_state_generated_10x10.png"
FIG_LAYERS = ROOT / "figures/scale_v1/node_search_state_generated_layers.png"
FIG_10X10 = ROOT / "figures/scale_v1/node_search_state_10x10.png"

JOBS = (
    ("fixed", "l10", "dfs_nodes", "Fixed 10×10"),
    ("context", "curriculum", "dfs_nodes", "In-context curriculum"),
    ("fixed", "l10", "bfs_nodes", "Fixed 10×10"),
    ("context", "curriculum", "bfs_nodes", "In-context curriculum"),
)
TASKS = ("legal_next", "visited", "frontier", "frontier_rest")
PLOT_TARGETS = (
    ("legal_next", "Legal next"),
    ("visited", "Visited"),
    ("frontier", "Stack / queue"),
    ("frontier_rest", "Frontier without head"),
)


def run_dir(kind: str, scale: str, protocol: str) -> Path:
    base = ROOT / "outputs/scale_v1/launches/scale_v1_s0" / kind
    if kind == "fixed":
        return base / scale / "stage_b" / protocol
    return base / scale / protocol


def _neighbors(adj, node, blocked: set[int]) -> set[int]:
    return {int(v) for v in adj[int(node)] if int(v) not in blocked}


def decision_state(adj, src: int, dst: int, prefix: list[int], fifo: bool):
    """State after `prefix` has been expanded, at the next real choice.

    Returns None when `prefix` is not a legal expansion prefix. Reaching the
    target ends the search, so a prefix that already contains it has no
    current choice. Forced DFS pops are applied. BFS keeps sibling cohorts
    unordered.
    """
    src, dst = int(src), int(dst)
    prefix = [int(v) for v in prefix]
    if any(v == dst for v in prefix):
        return None
    if fifo:
        return _bfs_state(adj, src, prefix)
    return _dfs_state(adj, src, prefix)


def _dfs_state(adj, src: int, prefix: list[int]):
    if not prefix:
        return _pack([src], {src}, {src}, [])
    if prefix[0] != src:
        return None
    stack = [src]
    seen = {src}
    for node in prefix[1:]:
        while stack and not _neighbors(adj, stack[-1], seen):
            stack.pop()
        if not stack or node not in _neighbors(adj, stack[-1], seen):
            return None
        stack.append(node)
        seen.add(node)
    while len(stack) > 1 and not _neighbors(adj, stack[-1], seen):
        stack.pop()
    cands = _neighbors(adj, stack[-1], seen)
    return _pack(stack, seen, cands, stack[:-1])


def _bfs_state(adj, src: int, prefix: list[int]):
    if not prefix:
        return _pack([src], {src}, {src}, [])
    if prefix[0] != src:
        return None
    discovered = {src}
    cohorts: deque[set[int]] = deque()
    for index, node in enumerate(prefix):
        if index:
            if not cohorts or node not in cohorts[0]:
                return None
            cohorts[0].remove(node)
            if not cohorts[0]:
                cohorts.popleft()
        children = _neighbors(adj, node, discovered)
        discovered.update(children)
        if children:
            cohorts.append(children)
    queue = set().union(*cohorts) if cohorts else set()
    front = set(cohorts[0]) if cohorts else set()
    return _pack(sorted(queue), discovered, front, queue - front)


def _pack(frontier, visited: set[int], legal_next: set[int], rest) -> dict:
    frontier = [int(v) for v in frontier]
    return {
        "frontier": frontier,
        "visited": set(visited),
        "legal_next": set(legal_next),
        "frontier_rest": set(rest),
    }


def _state_from_search(steps, order, k: int, fifo: bool):
    """Reference machine state for a known legal expansion prefix."""
    if k == 0:
        step = steps[0]
        return set(step.stack), set(step.visited), {step.cur}
    if not fifo:
        step = next(item for item in steps if item.action == "PUSH" and item.arg == order[k])
        return set(step.stack), set(step.visited), set(step.cands)
    step = next(item for item in steps if item.cur == order[k])
    return set(step.stack), set(step.visited), None


def self_test():
    rng = np.random.default_rng(0)
    for _trial in range(40):
        n = 7
        adj = [[v for v in range(n) if v != u and rng.random() < 0.35] for u in range(n)]
        src, dst = int(rng.integers(0, n)), int(rng.integers(0, n))
        if src == dst:
            dst = (dst + 1) % n
        for fifo, search in ((False, dfs_search), (True, bfs_search)):
            for _variant in range(4):
                shuffled = [rng.permutation(vs).astype(int).tolist() for vs in adj]
                steps = search(shuffled, src, dst)
                order = first_expansion_order(steps)
                other = [rng.permutation(vs).astype(int).tolist() for vs in adj]
                for k in range(len(order)):
                    state = decision_state(shuffled, src, dst, order[:k], fifo)
                    same = decision_state(other, src, dst, order[:k], fifo)
                    assert state is not None and same is not None
                    assert state["visited"] == same["visited"]
                    assert set(state["frontier"]) == set(same["frontier"])
                    assert state["legal_next"] == same["legal_next"]
                    frontier, visited, legal = _state_from_search(steps, order, k, fifo)
                    assert set(state["frontier"]) == frontier
                    assert state["visited"] == visited
                    if not fifo:
                        assert state["frontier_rest"] == set(state["frontier"][:-1])
                    else:
                        assert state["frontier_rest"] == set(state["frontier"]) - state["legal_next"]
                    if k == 0:
                        assert state["legal_next"] == {src}
                        assert state["frontier_rest"] == set()
                    elif not fifo:
                        assert state["legal_next"] == legal
                        assert order[k] in state["legal_next"]
                    else:
                        step = next(item for item in steps if item.cur == order[k])
                        parent = {
                            item.arg: item.cur for item in steps if item.action == "PUSH"
                        }
                        cohort = []
                        for node in step.stack:
                            if parent[node] != parent[order[k]]:
                                break
                            cohort.append(node)
                        assert state["legal_next"] == set(cohort)
                        assert order[k] in state["legal_next"]
                if steps[-1].action == "FOUND":
                    assert decision_state(shuffled, src, dst, order, fifo) is None
                else:
                    done = decision_state(shuffled, src, dst, order, fifo)
                    assert done is not None and done["legal_next"] == set()
    # A top with remaining children cannot be skipped.
    adj = [[1, 2], [], [], []]
    assert decision_state(adj, 0, 3, [0, 3], fifo=False) is None
    assert decision_state(adj, 0, 3, [0], fifo=False)["legal_next"] == {1, 2}
    # A later BFS cohort is not a legal head.
    adj = [[1, 2], [3], [], []]
    state = decision_state(adj, 0, 3, [0], fifo=True)
    assert state["legal_next"] == {1, 2}
    assert state["visited"] == {0, 1, 2}
    assert decision_state(adj, 0, 3, [0, 3], fifo=True) is None
    after = decision_state(adj, 0, 3, [0, 1], fifo=True)
    assert after["legal_next"] == {2}
    assert after["visited"] == {0, 1, 2, 3}
    assert set(after["frontier"]) == {2, 3}


def encode(raw_ids, nodes, n_vocab: int, lo: int) -> np.ndarray:
    target = np.zeros(n_vocab, np.float32)
    for raw in raw_ids:
        target[nodes[int(raw)] - lo] = 1.0
    return target


def graph_mask(nodes, n_vocab: int, lo: int) -> np.ndarray:
    mask = np.zeros(n_vocab, np.float32)
    for token in nodes:
        mask[int(token) - lo] = 1.0
    return mask


def leading_nodes(trace: list[int], inverse: dict[int, int]) -> list[int]:
    nodes = []
    for token in trace:
        if token not in inverse:
            break
        nodes.append(inverse[token])
    return nodes


def rows_for_generation(ex, out, tok, fifo: bool):
    inverse = {int(token): index for index, token in enumerate(ex.nodes)}
    emitted = leading_nodes(out["trace_tokens"], inverse)
    n_vocab, lo = tok.n_node_tokens, tok.node_lo
    mask = graph_mask(ex.nodes, n_vocab, lo)
    rows = []
    illegal_at = None
    for k in range(len(emitted)):
        state = decision_state(ex.adj, ex.source, ex.target, emitted[:k], fifo)
        if state is None:
            illegal_at = k
            break
        labels = {
            "legal_next": encode(state["legal_next"], ex.nodes, n_vocab, lo),
            "visited": encode(state["visited"], ex.nodes, n_vocab, lo),
            "frontier": encode(state["frontier"], ex.nodes, n_vocab, lo),
            "frontier_rest": encode(state["frontier_rest"], ex.nodes, n_vocab, lo),
        }
        if any(np.any(label * (1 - mask) > 0) for label in labels.values()):
            raise AssertionError("label fell outside the graph")
        prefix = encode(emitted[:k], ex.nodes, n_vocab, lo)
        rows.append({
            "readout": ex.row.prompt_len + k - 1,
            "labels": labels,
            "mask": mask,
            "prefix": prefix,
            "query": ex.index,
            "stage": "first" if k == 0 else "later",
            "next_ok": int(emitted[k] in state["legal_next"]),
            "n_legal": len(state["legal_next"]),
            "n_visited": len(state["visited"]),
            "n_frontier": len(state["frontier"]),
            "n_rest": len(state["frontier_rest"]),
            "visited_is_prefix": int(state["visited"] == set(emitted[:k]) or (
                k == 0 and state["visited"] == {ex.source}
            )),
            "unique_head": int(len(state["legal_next"]) == 1),
            "top_is_previous": int(
                (not fifo) and k > 0 and state["frontier"] and state["frontier"][-1] == emitted[k - 1]
            ),
        })
    complete = decision_state(ex.adj, ex.source, ex.target, emitted, fifo)
    if emitted and emitted[-1] == ex.target:
        proper = decision_state(ex.adj, ex.source, ex.target, emitted[:-1], fifo)
        finished = proper is not None and ex.target in proper["legal_next"] and illegal_at is None
    else:
        finished = complete is not None and not complete["legal_next"] and illegal_at is None
    return rows, {
        "n_emitted": len(emitted),
        "illegal_at": -1 if illegal_at is None else illegal_at,
        "finished_legal": int(finished),
        "correct": int(out["pred"] == ex.label),
        "interrupted": int(out["cot_interrupted"]),
    }


def empty_counts():
    return {
        "later": 0, "first": 0, "next_ok": 0, "legal_n": 0.0, "visited_n": 0.0,
        "frontier_n": 0.0, "rest_n": 0.0, "nonempty_legal": 0, "nonempty_rest": 0,
        "visited_is_prefix": 0, "unique_head": 0, "top_is_previous": 0,
    }


def add_row_counts(total, row):
    if row["stage"] != "later":
        total["first"] += 1
        return
    total["later"] += 1
    total["next_ok"] += row["next_ok"]
    total["legal_n"] += row["n_legal"]
    total["visited_n"] += row["n_visited"]
    total["frontier_n"] += row["n_frontier"]
    total["rest_n"] += row["n_rest"]
    total["nonempty_legal"] += int(row["n_legal"] > 0)
    total["nonempty_rest"] += int(row["n_rest"] > 0)
    total["visited_is_prefix"] += row["visited_is_prefix"]
    total["unique_head"] += row["unique_head"]
    total["top_is_previous"] += row["top_is_previous"]


def trace_report(info, counts):
    later = max(1, counts["later"])
    n = max(1, len(info))
    return {
        "n_queries": len(info),
        "n_first": counts["first"],
        "n_later": counts["later"],
        "mean_generated_nodes": sum(item["n_emitted"] for item in info) / n,
        "answer_accuracy": sum(item["correct"] for item in info) / n,
        "fraction_interrupted": sum(item["interrupted"] for item in info) / n,
        "fraction_traces_finished_legal": sum(item["finished_legal"] for item in info) / n,
        "fraction_traces_diverged": sum(item["illegal_at"] >= 0 for item in info) / n,
        "fraction_later_next_legal": counts["next_ok"] / later,
        "mean_legal_next": counts["legal_n"] / later,
        "mean_visited": counts["visited_n"] / later,
        "mean_frontier": counts["frontier_n"] / later,
        "mean_frontier_rest": counts["rest_n"] / later,
        "fraction_nonempty_legal_next": counts["nonempty_legal"] / later,
        "fraction_nonempty_frontier_rest": counts["nonempty_rest"] / later,
        "fraction_visited_equals_prefix": counts["visited_is_prefix"] / later,
        "fraction_unique_head": counts["unique_head"] / later,
        "fraction_frontier_head_is_previous": counts["top_is_previous"] / later,
    }


def collect(am, examples, fifo: bool, batch: int = 8):
    grouped = []
    info = []
    counts = empty_counts()
    outputs = generate(am, [ex.prompt for ex in examples], batch=16)
    if len(outputs) != len(examples):
        raise AssertionError("generation count drifted")
    for ex, out in zip(examples, outputs):
        if out["prompt_len"] != ex.row.prompt_len:
            raise AssertionError("prompt length drifted from the prepared example")
        rows, meta = rows_for_generation(ex, out, am.tok, fifo)
        for row in rows:
            add_row_counts(counts, row)
        grouped.append((ex, out, rows))
        info.append(meta)
    flat = [(ex, out, row) for ex, out, rows in grouped for row in rows]
    if not flat:
        raise ValueError("no legal generated positions")
    n_layers = len(am.layers)
    width = am.model.d_model + 4
    features = np.empty((len(flat), n_layers + 1, width), np.float32)
    cursor = 0
    for start in range(0, len(grouped), batch):
        chunk = grouped[start:start + batch]
        sequences = [ex.prompt + out["trace_tokens"] for ex, out, _rows in chunk]
        _, residual, _ = am.forward(sequences)
        for b, (ex, out, rows) in enumerate(chunk):
            sequence = sequences[b]
            inverse = {int(token): index for index, token in enumerate(ex.nodes)}
            emitted = leading_nodes(out["trace_tokens"], inverse)
            for row in rows:
                pos = row["readout"]
                k = pos - ex.row.prompt_len + 1
                if sequence[pos + 1] != ex.nodes[emitted[k]]:
                    raise AssertionError("readout does not predict the generated node")
                hidden = torch.stack([residual[layer][b, pos].float() for layer in am.layers])
                hidden = F.pad(hidden, (0, 4))
                token = sequence[pos]
                embedding = am.model.embed.weight[token].float()
                seen = ex.nodes[ex.target] in sequence[ex.row.prompt_len:pos + 1]
                observable = embedding.new_tensor([
                    pos / am.model.max_seq_len,
                    (pos - ex.row.prompt_len) / am.model.max_seq_len,
                    float(seen),
                    float(token == ex.nodes[ex.target]),
                ])
                features[cursor] = torch.cat([hidden, torch.cat([embedding, observable])[None]], 0).cpu().numpy()
                cursor += 1
        del residual
    if cursor != len(flat):
        raise AssertionError("feature rows drifted from labels")
    packed = {}
    prefix = np.stack([row["prefix"] for _ex, _out, row in flat])
    query = np.asarray([row["query"] for _ex, _out, row in flat], np.int64)
    stage = np.asarray([row["stage"] for _ex, _out, row in flat])
    for name in TASKS:
        packed[name] = {
            "x": features,
            "prefix": prefix[:, None, :],
            "y": np.stack([row["labels"][name] for _ex, _out, row in flat]),
            "mask": np.stack([row["mask"] for _ex, _out, row in flat]),
            "query": query,
            "stage": stage,
        }
    return packed, trace_report(info, counts)


def slim(quality: dict) -> dict:
    micro = quality.get("micro_ranking") or {}
    return {
        "n_rows": quality.get("n_rows"),
        "nontrivial_rows": quality.get("nontrivial_rows"),
        "n_positive": quality.get("n_positive"),
        "pairwise_auc": quality.get("pairwise_auc"),
        "topk_exact": quality.get("topk_exact"),
        "topk_recall": quality.get("topk_recall"),
        "strict_separation": quality.get("strict_separation"),
        "positive_recall_at_half": quality.get("positive_recall_at_half"),
        "micro_auroc": micro.get("auroc"),
    }


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
    evaluated = {**test, "y": shuffled_labels(test, seed + 202)} if shuffled else test
    report = summarize(fitted, evaluated)
    del fitted
    return report


def auc_of(report, layer: int, stage: str = "later"):
    if report is None or layer < 0 or layer >= len(report):
        return None
    return report[layer].get(stage, {}).get("pairwise_auc")


def fmt(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def probe_model(am, counts: dict, steps: int):
    fifo = am.protocol.startswith("bfs")
    splits = {name: am.examples(name, count) for name, count in counts.items()}
    ids = {name: [ex.index for ex in rows] for name, rows in splits.items()}
    if set(ids["train"]) & set(ids["val"]) or set(ids["train"]) & set(ids["test"]) or set(ids["val"]) & set(ids["test"]):
        raise AssertionError("probe splits share queries")
    features, reports = {}, {}
    for name, rows in splits.items():
        print(f"  generate {name} n={len(rows)}", flush=True)
        features[name], reports[name] = collect(am, rows, fifo)
        replay = reports[name]
        print(
            f"    later={replay['n_later']} next-legal {replay['fraction_later_next_legal']:.3f} "
            f"diverged {replay['fraction_traces_diverged']:.3f} "
            f"visited=prefix {replay['fraction_visited_equals_prefix']:.3f} "
            f"head=previous {replay['fraction_frontier_head_is_previous']:.3f}",
            flush=True,
        )
    tasks = {}
    for name in TASKS:
        print(f"  probe {name}", flush=True)
        bundle = {split: features[split][name] for split in splits}
        residual = run_task(bundle["train"], bundle["val"], bundle["test"], steps=steps)
        shuffled = run_task(bundle["train"], bundle["val"], bundle["test"], steps=steps, shuffled=True)
        prefixed = {split: {**bundle[split], "x": bundle[split]["prefix"]} for split in splits}
        prefix_bag = run_task(prefixed["train"], prefixed["val"], prefixed["test"], steps=steps)
        tasks[name] = {"residual": residual, "shuffled": shuffled, "prefix_bag": prefix_bag}
        last = len(residual) - 2
        print(
            f"    L6 {fmt(auc_of(residual, last))} token {fmt(auc_of(residual, last + 1))} "
            f"bag {fmt(auc_of(prefix_bag, 0))} shuffle {fmt(auc_of(shuffled, last))} "
            f"exact {fmt(residual[last]['later'].get('topk_exact'))}",
            flush=True,
        )
    return {
        "checkpoint": am.checkpoint,
        "world": str(am.store.root),
        "protocol": am.protocol,
        "query_ids": ids,
        "n_queries": {name: len(rows) for name, rows in splits.items()},
        "replay": reports,
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
    protocols = (("dfs_nodes", "DFS"), ("bfs_nodes", "BFS"))
    columns = (("fixed", "l10", "Fixed 10×10"), ("context", "curriculum", "In-context curriculum"))
    series = (
        ("l6", "L6 activation", "#0072B2", "-"),
        ("prefix", "Emitted-node set", "#009E73", "--"),
        ("token", "Previous token", "#D55E00", ":"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.0), sharey=True)
    xs = np.arange(len(PLOT_TARGETS))
    for row, (protocol, row_title) in enumerate(protocols):
        for col, (kind, scale, label) in enumerate(columns):
            ax = axes[row, col]
            tasks = doc["models"][f"{kind}_{scale}_{protocol}"]["tasks"]
            ax.axhline(0.5, color="#9A9A9A", lw=0.8, ls="--", zorder=0)
            for feature, title, color, linestyle in series:
                ys = []
                for key, _name in PLOT_TARGETS:
                    if feature == "prefix":
                        report, layer = tasks[key]["prefix_bag"], 0
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
            ax.set_xlim(-0.25, len(PLOT_TARGETS) - 0.75)
            ax.set_ylim(0.35, 1.05)
            ax.set_xticks(xs, [name for _key, name in PLOT_TARGETS], fontsize=7.5, rotation=18, ha="right")
            ax.set_title(f"{row_title} · {label}", fontsize=11, pad=6)
            if col == 0:
                ax.set_ylabel("Pairwise AUC", fontsize=10)
            if row == 1 and col == 0:
                ax.legend(frameon=True, fontsize=7.5, loc="lower right", framealpha=0.94,
                          edgecolor="none", fancybox=False)
    fig.suptitle("State replayed from the model's own prefix", fontsize=13)
    fig.tight_layout()
    save_rgb(fig, FIG)
    save_rgb_copy(FIG, FIG_10X10)

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.0), sharey=True)
    colors = {
        "legal_next": "#0072B2",
        "visited": "#009E73",
        "frontier": "#D55E00",
        "frontier_rest": "#CC79A7",
    }
    for row, (protocol, row_title) in enumerate(protocols):
        for col, (kind, scale, label) in enumerate(columns):
            ax = axes[row, col]
            model = doc["models"][f"{kind}_{scale}_{protocol}"]
            names = model["layers"][:-1]
            ax.axhline(0.5, color="#9A9A9A", lw=0.8, ls="--", zorder=0)
            for key, title in PLOT_TARGETS:
                report = model["tasks"][key]["residual"]
                ys = [np.nan if (value := auc_of(report, i)) is None else value for i in range(len(names))]
                ax.plot(np.arange(1, len(ys) + 1), ys, color=colors[key], marker="o",
                        linewidth=1.6, markersize=4, label=title)
            style_ax(ax)
            ax.set_ylim(0.35, 1.05)
            ax.set_xticks(np.arange(1, len(names) + 1))
            ax.set_title(f"{row_title} · {label}", fontsize=11, pad=6)
            if col == 0:
                ax.set_ylabel("Pairwise AUC", fontsize=10)
            if row == 1:
                ax.set_xlabel("Layer", fontsize=10)
            if row == 1 and col == 0:
                ax.legend(frameon=True, fontsize=7, loc="lower right", framealpha=0.94,
                          edgecolor="none", fancybox=False)
    fig.suptitle("Generated-prefix state by layer", fontsize=13)
    fig.tight_layout()
    save_rgb(fig, FIG_LAYERS)


def save_rgb_copy(source: Path, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.open(source).convert("RGB").save(dest, format="PNG", optimize=True)
    print(dest, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=int, default=256)
    parser.add_argument("--val", type=int, default=96)
    parser.add_argument("--test", type=int, default=128)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    self_test()
    print("replay self-test passed", flush=True)
    if args.self_test:
        return
    if args.plot_only:
        plot(json.loads(OUT.read_text()))
        return
    torch.set_num_threads(4)
    doc = json.loads(OUT.read_text()) if OUT.exists() else {
        "seed": SEED,
        "definition": {
            "readout": "block residual at the position that predicts the next generated node",
            "generation": "greedy argmax from the training prompt, world token included for fixed graphs",
            "ground_truth": "search state replayed from the generated node prefix and the query edges only",
            "dfs": "after the prefix, pop while the top has no unused neighbor; candidates are the top's unused neighbors",
            "bfs": "visited and queue sets are fixed; the candidate label is the earliest unordered sibling cohort",
            "excluded": "a prefix that is not a legal expansion, or that already expanded the target, ends the label sequence",
            "controls": "previous-token embedding, multihot of emitted nodes, within-graph shuffled labels",
            "metric": "mean within-position pairwise AUC on later positions; chance is 0.5",
        },
        "models": {},
    }
    counts = {"train": args.train, "val": args.val, "test": args.test}
    for kind, scale, protocol, _label in JOBS:
        model_id = f"{kind}_{scale}_{protocol}"
        if args.models and model_id not in args.models:
            continue
        if model_id in doc["models"]:
            print("skip", model_id, flush=True)
            continue
        print("MODEL", model_id, flush=True)
        entry = {"id": model_id, "run": str(run_dir(kind, scale, protocol).relative_to(ROOT))}
        am = make_adapter(entry, ROOT, "cuda:0")
        am.layers = list(range(am.model.n_layers))
        doc["models"][model_id] = probe_model(am, counts, args.steps)
        write_json(OUT, doc)
        del am
        torch.cuda.empty_cache()
    plot(doc)


if __name__ == "__main__":
    main()

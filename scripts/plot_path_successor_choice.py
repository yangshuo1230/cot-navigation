#!/usr/bin/env python3
"""Does in-context PATH pick among successors, or only read the successor set?

Teacher-forced steps on held-out hard DAGs. Layer 6 at the path step reads the
copied head of each outgoing edge (five and six tokens after that head). The
probe compares probability and that attention across successors: uniform, the
successor that still reaches the target, the one on a shorter path, the one
with more outgoing edges, and the first or last edge in the list. Edits then
cut a successor's downstream path while keeping its out-degree, or add edges
to a successor that cannot reach the target.
"""
from __future__ import annotations

import json
import sys
from collections import deque
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
from cot_navigation.protocols.render import context_prompt  # noqa: E402

FIG = ROOT / "figures/hard_dag_v1/path_successor_choice.png"
JSON = ROOT / "studies/block_membership_probe/path_successor_choice.json"
DEVICE = "cuda:0"
EDIT_CAP = 48


def adj_from(edges, n):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
    return adj


def ancestors_of_target(adj, target):
    """seen[u] is true when u can reach target. The target is marked true."""
    radj = [[] for _ in range(len(adj))]
    for u, vs in enumerate(adj):
        for v in vs:
            radj[v].append(u)
    seen = [False] * len(adj)
    seen[target] = True
    stack = [target]
    while stack:
        node = stack.pop()
        for parent in radj[node]:
            if not seen[parent]:
                seen[parent] = True
                stack.append(parent)
    return seen


def dist_to_target(adj, target):
    radj = [[] for _ in range(len(adj))]
    for u, vs in enumerate(adj):
        for v in vs:
            radj[v].append(u)
    dist = [-1] * len(adj)
    dist[target] = 0
    queue = deque([target])
    while queue:
        node = queue.popleft()
        for parent in radj[node]:
            if dist[parent] < 0:
                dist[parent] = dist[node] + 1
                queue.append(parent)
    return dist


def reachable_from(adj, source):
    seen = set()
    stack = [source]
    while stack:
        node = stack.pop()
        for nxt in adj[node]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def edge_table(item):
    inverse = {token: index for index, token in enumerate(item["node_tokens"])}
    n_edges = (item["prompt_len"] - 4) // 3
    tokens = item["tokens"]
    edges = []
    for edge in range(n_edges):
        edges.append((inverse[tokens[edge * 3 + 1]], inverse[tokens[edge * 3 + 2]]))
    return edges


def relay_mass(attn, edges, cur, receiver):
    by_succ = {}
    head = 0.0
    for index, (u, v) in enumerate(edges):
        if u != cur:
            continue
        head_pos = 3 * index + 2
        if head_pos <= receiver:
            head += float(attn[head_pos])
        mass = 0.0
        for offset in (5, 6):
            pos = head_pos + offset
            if pos <= receiver and pos < len(attn):
                mass += float(attn[pos])
        by_succ[v] = by_succ.get(v, 0.0) + mass
    return by_succ, head


def subset_mass(values, members, universe):
    index = {node: i for i, node in enumerate(universe)}
    return float(sum(values[index[node]] for node in members))


@torch.inference_mode()
def forward_row(model, tokens, receiver, device):
    ids = torch.tensor(tokens, device=device, dtype=torch.long)[None]
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits, weights = model(ids, return_attn=True)
    row = logits[0, receiver].float()
    attn = weights[-1][0, 0, receiver].float().cpu().numpy()
    del logits, weights
    return row, attn


def successor_distribution(row, succ, node_tokens):
    prob = torch.softmax(row, dim=-1)
    raw = np.array([float(prob[node_tokens[v]]) for v in succ], dtype=np.float64)
    total = float(raw.sum())
    renorm = raw / total if total > 1e-12 else np.zeros_like(raw)
    pred = int(row.argmax())
    return renorm, total, pred


def fresh():
    keys = ("reach", "short", "direct", "degree", "first", "last", "gold", "gold_open", "target")
    return {
        "steps": 0,
        "branch": 0,
        "legal_mass": [],
        "max_prob": [],
        "uniform": [],
        "relay": [],
        "head": [],
        "top_share": [],
        "neff_over_d": [],
        "attn_is_pred": [],
        "query_target": [],
        "buckets": {key: [] for key in keys},
        "attn_buckets": {key: [] for key in ("reach", "short", "degree", "target")},
        "conflict": [],
        "examples": [],
        "open_max": [],
        "open_uniform": [],
        "open_top_share": [],
        "target_max": [],
        "target_uniform": [],
    }


def add_bucket(store, key, mass, base, hit, attn=None):
    store["buckets"][key].append((mass, base, hit))
    if attn is not None:
        store["attn_buckets"][key].append((attn, base))


def pack(rows):
    if not rows:
        return {"n": 0}
    mass, base, hit = np.array(rows, dtype=np.float64).T
    return {
        "n": int(len(rows)),
        "mass": float(mass.mean()),
        "base": float(base.mean()),
        "lift": float(mass.mean() / base.mean()) if base.mean() > 1e-8 else None,
        "hit": float(hit.mean()),
    }


def pack_attn(rows):
    if not rows:
        return {"n": 0}
    mass, base = np.array(rows, dtype=np.float64).T
    return {
        "n": int(len(rows)),
        "mass": float(mass.mean()),
        "base": float(base.mean()),
        "lift": float(mass.mean() / base.mean()) if base.mean() > 1e-8 else None,
    }


def mean(values):
    return float(np.mean(values)) if values else None


@torch.inference_mode()
def forward_item(model, tokens, device):
    ids = torch.tensor(tokens, device=device, dtype=torch.long)[None]
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits, weights = model(ids, return_attn=True)
    rows = logits[0].float().cpu()
    attn = weights[-1][0, 0].float().cpu().numpy()
    del logits, weights
    return rows, attn


def analyze_size(model, tok, items, device):
    yes, no = fresh(), fresh()
    edits = []
    for item in items:
        edges = edge_table(item)
        n = len(item["node_tokens"])
        adj = adj_from(edges, n)
        target = item["target"]
        can = ancestors_of_target(adj, target)
        dist = dist_to_target(adj, target)
        degree = [len(vs) for vs in adj]
        store = yes if item["label"] == 1 else no
        path = item["path"]
        inverse = {token: index for index, token in enumerate(item["node_tokens"])}
        pending = []
        for step_i in range(len(path) - 1):
            receiver = item["prompt_len"] + step_i
            cur = inverse[item["tokens"][receiver]]
            succ = []
            for u, v in edges:
                if u == cur and v not in succ:
                    succ.append(v)
            store["steps"] += 1
            if len(succ) >= 2:
                pending.append((step_i, receiver, cur, succ))
        if not pending:
            continue
        if item["index"] % 128 == 0:
            print(f"  item {item['index']}", flush=True)
        logits, attention = forward_item(model, item["tokens"], device)
        for step_i, receiver, cur, succ in pending:
            degree_now = len(succ)
            row = logits[receiver]
            attn = attention[receiver]
            renorm, legal, pred = successor_distribution(row, succ, item["node_tokens"])
            by_succ, head = relay_mass(attn, edges, cur, receiver)
            shares = np.array([by_succ.get(v, 0.0) for v in succ], dtype=np.float64)
            relay_total = float(shares.sum())
            attn_share = shares / relay_total if relay_total > 1e-8 else np.zeros_like(shares)
            positive = attn_share[attn_share > 0]
            neff = float(np.exp(-(positive * np.log(positive)).sum())) if len(positive) else 0.0
            pred_topo = inverse.get(pred)
            store["branch"] += 1
            store["legal_mass"].append(legal)
            store["max_prob"].append(float(renorm.max()))
            store["uniform"].append(1.0 / degree_now)
            store["relay"].append(relay_total)
            store["head"].append(head)
            store["top_share"].append(float(attn_share.max()) if relay_total > 1e-8 else 0.0)
            store["neff_over_d"].append(neff / degree_now)
            store["attn_is_pred"].append(float(succ[int(attn_share.argmax())] == pred_topo) if relay_total > 1e-8 else 0.0)
            store["query_target"].append(float(attn[item["prompt_len"] - 2]))

            reach = [v for v in succ if can[v] and v != target]
            short_pool = [v for v in succ if dist[v] >= 0 and v != target]
            short = []
            if short_pool:
                best = min(dist[v] for v in short_pool)
                short = [v for v in short_pool if dist[v] == best]
            direct = [v for v in succ if v != target and target in adj[v]]
            top_degree = max(degree[v] for v in succ)
            high = [v for v in succ if degree[v] == top_degree]
            target_in = target in succ
            if target_in:
                store["target_max"].append(float(renorm.max()))
                store["target_uniform"].append(1.0 / degree_now)
            else:
                store["open_max"].append(float(renorm.max()))
                store["open_uniform"].append(1.0 / degree_now)
                store["open_top_share"].append(float(attn_share.max()) if relay_total > 1e-8 else 0.0)
            gold_node = inverse[item["tokens"][receiver + 1]]
            sets = {
                "reach": reach if item["label"] == 1 and reach and len(reach) < degree_now and not target_in else None,
                "short": short if item["label"] == 1 and short and len(short) < len(short_pool) and not target_in else None,
                "direct": direct if item["label"] == 1 and direct and len(direct) < degree_now and not target_in else None,
                "degree": high if len(high) < degree_now else None,
                "first": [succ[0]],
                "last": [succ[-1]],
                "gold": [gold_node],
                "gold_open": [gold_node] if gold_node != target else None,
                "target": [target] if target_in else None,
            }
            for key, members in sets.items():
                if not members:
                    continue
                mass = subset_mass(renorm, members, succ)
                base = len(members) / degree_now
                hit = float(pred_topo in members)
                attn_on = subset_mass(attn_share, members, succ) if key in store["attn_buckets"] else None
                add_bucket(store, key, mass, base, hit, attn_on)
            if (
                item["label"] == 1
                and not target_in
                and short
                and high
                and set(short).isdisjoint(high)
            ):
                store["conflict"].append((
                    subset_mass(renorm, short, succ),
                    subset_mass(renorm, high, succ),
                    len(short) / degree_now,
                    len(high) / degree_now,
                ))
            if item["label"] == 1 and not target_in and reach and len(edits) < EDIT_CAP:
                edits.append({
                    "tokens": item["tokens"],
                    "prompt_len": item["prompt_len"],
                    "source": item["source"],
                    "target": target,
                    "node_tokens": item["node_tokens"],
                    "edges": edges,
                    "step_i": step_i,
                    "cur": cur,
                    "succ": succ,
                    "reach": reach,
                    "nonreach": [v for v in succ if v not in reach and v != target],
                    "degree": degree,
                })
            if item["label"] == 1 and not target_in and reach and len(store["examples"]) < 1:
                store["examples"].append({
                    "index": item["index"],
                    "cur": cur,
                    "target": target,
                    "pred": pred_topo,
                    "succ": [
                        {
                            "node": v,
                            "degree": degree[v],
                            "dist": int(dist[v]),
                            "prob": float(renorm[i]),
                            "attn": float(attn_share[i]),
                        }
                        for i, v in enumerate(succ)
                    ],
                })
    return yes, no, edits


def choose_dests(origin, count, n, can, banned):
    forward = [w for w in range(origin + 1, n) if not can[w] and w not in banned]
    anywhere = [w for w in range(n) if w != origin and not can[w] and w not in banned]
    pool = forward if len(forward) >= count else anywhere
    if len(pool) < count:
        return None
    return pool[:count]


def retarget(edges, victims, n, can, banned):
    victims = set(victims)
    need = {}
    for u, _v in edges:
        if u in victims:
            need[u] = need.get(u, 0) + 1
    picks = {}
    for node, count in need.items():
        chosen = choose_dests(node, count, n, can, banned)
        if chosen is None:
            return None
        picks[node] = iter(chosen)
    rewritten = []
    for u, v in edges:
        rewritten.append((u, next(picks[u])) if u in victims else (u, v))
    return rewritten


def rebuild(tok, case, edges):
    prompt = context_prompt(tok, case["node_tokens"], edges, case["source"], case["target"])
    tokens = prompt + case["tokens"][case["prompt_len"]:]
    return tokens, len(prompt) + case["step_i"]


def mass_on(row, case, members, node_tokens):
    renorm, _legal, pred = successor_distribution(row, case["succ"], node_tokens)
    return subset_mass(renorm, members, case["succ"]), pred


def run_edits(model, tok, cases, device):
    """Cut downstream paths, or add edges, without changing the successor set."""
    cut_before, cut_after, cut_ctrl = [], [], []
    deg_before, deg_after, deg_ctrl = [], [], []
    used = 0
    for case in cases:
        n = len(case["node_tokens"])
        edges = list(case["edges"])
        adj = adj_from(edges, n)
        can = ancestors_of_target(adj, case["target"])
        if not case["reach"] or not case["nonreach"]:
            continue
        rewritten = retarget(edges, case["reach"], n, can, {case["cur"], case["target"]})
        if rewritten is None:
            continue
        new_adj = adj_from(rewritten, n)
        if [v for u, v in rewritten if u == case["cur"]] != case["succ"]:
            continue
        if any(ancestors_of_target(new_adj, case["target"])[v] for v in case["reach"]):
            continue
        if any(len(new_adj[v]) != len(adj[v]) for v in case["reach"]):
            continue
        outside = reachable_from(adj, case["cur"])
        control_nodes = [z for z in range(n) if z not in outside and z != case["cur"] and adj[z]]
        control = retarget(edges, control_nodes[:1], n, can, {case["cur"], case["target"]}) if control_nodes else None
        base_tokens, base_at = rebuild(tok, case, edges)
        base_row, _attn = forward_row(model, base_tokens, base_at, device)
        before, _pred = mass_on(base_row, case, case["reach"], case["node_tokens"])
        edit_tokens, edit_at = rebuild(tok, case, rewritten)
        edit_row, _attn = forward_row(model, edit_tokens, edit_at, device)
        after, _pred = mass_on(edit_row, case, case["reach"], case["node_tokens"])
        cut_before.append(before)
        cut_after.append(after)
        if control is not None:
            ctrl_tokens, ctrl_at = rebuild(tok, case, control)
            ctrl_row, _attn = forward_row(model, ctrl_tokens, ctrl_at, device)
            ctrl_mass, _pred = mass_on(ctrl_row, case, case["reach"], case["node_tokens"])
            cut_ctrl.append(ctrl_mass)
        w = min(case["nonreach"], key=lambda node: case["degree"][node])
        extra = max(3, max(case["degree"][v] for v in case["reach"]) - case["degree"][w] + 2)
        extra = min(extra, 8)
        dests = choose_dests(w, extra, n, can, {case["cur"], case["target"], *adj[w]})
        if dests is not None:
            grown = edges + [(w, dest) for dest in dests]
            grown_tokens, grown_at = rebuild(tok, case, grown)
            grown_row, _attn = forward_row(model, grown_tokens, grown_at, device)
            grown_mass, _pred = mass_on(grown_row, case, [w], case["node_tokens"])
            base_w, _pred = mass_on(base_row, case, [w], case["node_tokens"])
            deg_before.append(base_w)
            deg_after.append(grown_mass)
            unrelated = [z for z in range(n) if z not in outside and z != case["cur"] and z not in case["succ"]]
            if unrelated:
                z = unrelated[0]
                z_dests = choose_dests(z, extra, n, can, {case["cur"], case["target"], *adj[z]})
                if z_dests is not None:
                    ctrl_edges = edges + [(z, dest) for dest in z_dests]
                    ctrl_tokens, ctrl_at = rebuild(tok, case, ctrl_edges)
                    ctrl_row, _attn = forward_row(model, ctrl_tokens, ctrl_at, device)
                    ctrl_mass, _pred = mass_on(ctrl_row, case, [w], case["node_tokens"])
                    deg_ctrl.append(ctrl_mass)
        used += 1
    return {
        "n": used,
        "cut_reach_mass_before": mean(cut_before),
        "cut_reach_mass_after": mean(cut_after),
        "cut_control": mean(cut_ctrl),
        "n_cut": len(cut_before),
        "n_cut_control": len(cut_ctrl),
        "degree_mass_before": mean(deg_before),
        "degree_mass_after": mean(deg_after),
        "degree_control": mean(deg_ctrl),
        "n_degree": len(deg_before),
        "n_degree_control": len(deg_ctrl),
    }


def summarize(store):
    out = {
        "steps": store["steps"],
        "branch": store["branch"],
        "legal_mass": mean(store["legal_mass"]),
        "max_prob": mean(store["max_prob"]),
        "uniform": mean(store["uniform"]),
        "relay": mean(store["relay"]),
        "head": mean(store["head"]),
        "top_share": mean(store["top_share"]),
        "neff_over_d": mean(store["neff_over_d"]),
        "attn_is_pred": mean(store["attn_is_pred"]),
        "query_target": mean(store["query_target"]),
        "open_max": mean(store["open_max"]),
        "open_uniform": mean(store["open_uniform"]),
        "open_top_share": mean(store["open_top_share"]),
        "open_n": len(store["open_max"]),
        "target_max": mean(store["target_max"]),
        "target_uniform": mean(store["target_uniform"]),
        "target_n": len(store["target_max"]),
        "choice": {key: pack(rows) for key, rows in store["buckets"].items()},
        "attention": {key: pack_attn(rows) for key, rows in store["attn_buckets"].items()},
        "conflict_n": len(store["conflict"]),
        "example": store["examples"][:1],
    }
    if store["conflict"]:
        short, high, short_base, high_base = np.array(store["conflict"], dtype=np.float64).T
        out["conflict"] = {
            "n": len(store["conflict"]),
            "short_mass": float(short.mean()),
            "degree_mass": float(high.mean()),
            "short_base": float(short_base.mean()),
            "degree_base": float(high_base.mean()),
        }
    return out


def draw(payload):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "0.9",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    blue, orange, vermillion, green, gray = "#0072B2", "#E69F00", "#D55E00", "#009E73", "0.65"
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8), constrained_layout=True)
    sizes = [block["size"] for block in payload]
    x = np.arange(len(sizes))

    ax = axes[0]
    width = 0.18
    series = (
        ("YES largest", [block["yes"]["open_max"] for block in payload], blue, -1.5),
        ("YES uniform", [block["yes"]["open_uniform"] for block in payload], gray, -0.5),
        ("NO largest", [block["no"]["open_max"] for block in payload], vermillion, 0.5),
        ("NO uniform", [block["no"]["open_uniform"] for block in payload], orange, 1.5),
    )
    for label, values, color, offset in series:
        ax.bar(x + offset * width, values, width=width, color=color, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Probability among successors")
    ax.set_title("Target is not a successor")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    metrics = (
        ("Reaches target", "reach"),
        ("Shorter path", "short"),
        ("Edge to target", "direct"),
        ("More out-edges", "degree"),
        ("Is the target", "target"),
    )
    width = 0.15
    colors = (blue, green, orange, vermillion, "#CC79A7")
    for index, ((label, key), color) in enumerate(zip(metrics, colors)):
        lifts = []
        for block in payload:
            stat = block["yes"]["choice"][key]
            lifts.append(stat["lift"] if stat.get("lift") is not None else np.nan)
        ax.bar(x + (index - 2) * width, lifts, width=width, color=color, label=label)
    ax.axhline(1.0, color="0.45", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_ylabel("Probability / uniform share")
    ax.set_title("YES, choice among successors")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)

    ax = axes[2]
    width = 0.16
    def num(value):
        return 0.0 if value is None else value

    cut_b = [num(block["edits"]["cut_reach_mass_before"]) for block in payload]
    cut_a = [num(block["edits"]["cut_reach_mass_after"]) for block in payload]
    cut_c = [num(block["edits"]["cut_control"]) for block in payload]
    deg_b = [num(block["edits"]["degree_mass_before"]) for block in payload]
    deg_a = [num(block["edits"]["degree_mass_after"]) for block in payload]
    deg_c = [num(block["edits"]["degree_control"]) for block in payload]
    # Two clusters per size are crowded; plot the change from the unedited mass.
    ax.bar(x - 1.5 * width, [a - b for a, b in zip(cut_a, cut_b)], width=width, color=blue, label="Cut path, keep degree")
    ax.bar(x - 0.5 * width, [c - b for c, b in zip(cut_c, cut_b)], width=width, color=gray, label="Cut an unrelated node")
    ax.bar(x + 0.5 * width, [a - b for a, b in zip(deg_a, deg_b)], width=width, color=vermillion, label="Add edges to a dead end")
    ax.bar(x + 1.5 * width, [c - b for c, b in zip(deg_c, deg_b)], width=width, color=orange, label="Add edges elsewhere")
    ax.axhline(0.0, color="0.45", lw=1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_ylabel("Change in successor probability")
    ax.set_title("Edits")
    ax.legend(frameon=False, fontsize=7)
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
    _cfg, tok, model, step = base.load_model(pathfig.RUN, device)
    print("step", step, flush=True)
    saved = []
    for name, root in pathfig.SIZES:
        arrays = pathfig.load_test(root)
        items = [pathfig.render_index(arrays, tok, index) for index in range(len(arrays["label"]))]
        yes, no, edits = analyze_size(model, tok, items, device)
        edit_stats = run_edits(model, tok, edits, device)
        block = {
            "size": name.replace("DAG ", ""),
            "step": step,
            "n_items": len(items),
            "yes": summarize(yes),
            "no": summarize(no),
            "edits": edit_stats,
        }
        saved.append(block)
        print(f"\n{name} items={len(items)} yes_branch={yes['branch']} no_branch={no['branch']}", flush=True)
        for label, store in (("YES", block["yes"]), ("NO", block["no"])):
            print(
                f"  {label} legal {store['legal_mass']:.3f} max_p {store['max_prob']:.3f} "
                f"uniform {store['uniform']:.3f} open_max {store['open_max']} "
                f"open_uniform {store['open_uniform']} n {store['open_n']} "
                f"target_max {store['target_max']} n {store['target_n']} "
                f"relay {store['relay']:.3f} top_share {store['top_share']:.3f} "
                f"neff/d {store['neff_over_d']:.3f} attn=pred {store['attn_is_pred']:.3f}",
                flush=True,
            )
            for key, stat in store["choice"].items():
                if stat["n"] == 0:
                    continue
                attn = store["attention"].get(key, {})
                attn_txt = ""
                if attn.get("n"):
                    attn_txt = f" attn {attn['mass']:.3f}/{attn['base']:.3f}"
                print(
                    f"    {key:8s} n={stat['n']:4d} mass {stat['mass']:.3f} base {stat['base']:.3f} "
                    f"lift {stat['lift']:.3f} hit {stat['hit']:.3f}{attn_txt}",
                    flush=True,
                )
            if "conflict" in store:
                row = store["conflict"]
                print(
                    f"    conflict n={row['n']} short {row['short_mass']:.3f}/{row['short_base']:.3f} "
                    f"degree {row['degree_mass']:.3f}/{row['degree_base']:.3f}",
                    flush=True,
                )
        print("  edits", edit_stats, flush=True)
        if block["yes"]["example"]:
            print("  example", block["yes"]["example"][0], flush=True)
    JSON.parent.mkdir(parents=True, exist_ok=True)
    JSON.write_text(json.dumps(saved, indent=2))
    draw(saved)
    print(FIG)
    print(JSON)


if __name__ == "__main__":
    main()

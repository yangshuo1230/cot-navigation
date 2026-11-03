#!/usr/bin/env python3
"""Fixed-DAG attention flow, same splits as the in-context figures.

Fixed curriculum checkpoints, the 4×4 world (<WORLD_0>). One held-out trace
for the drawings (test index 16, V6 to V15) plus the backtrack traces
index 18 and index 47. Means use every held-out query on that world.
The gold box is the position whose attention is drawn.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import plot_dfs_cot_attention as base  # noqa: E402
import plot_dfs_field_flow as flow  # noqa: E402
from cot_navigation.analysis.common import records  # noqa: E402
from cot_navigation.data.dataset import ParametricWorld  # noqa: E402

DEVICE = "cuda:0"
WORLD = ROOT / "data/scale_v1/fixed_l4w4_n16_m24_s0/main-seed0"
STATE_RUN = ROOT / "outputs/scale_v1/launches/scale_v1_s0/fixed/curriculum/stage_b/dfs_state"
NODES_RUN = ROOT / "outputs/scale_v1/launches/scale_v1_s0/fixed/curriculum/stage_b/dfs_nodes"
STATE_OUT = ROOT / "figures/scale_v1/fixed_dag/dfs_flow"
NODES_OUT = ROOT / "figures/scale_v1/fixed_dag/dfs_nodes_actions"

W, H, G = flow.W, flow.H, flow.G
BLUE, VERM, GREEN, GOLD = flow.BLUE, flow.VERM, flow.GREEN, flow.GOLD
BLUE_F, VERM_F, GREEN_F, GOLD_F, GRAY_F = (
    flow.BLUE_F, flow.VERM_F, flow.GREEN_F, flow.GOLD_F, flow.GRAY_F,
)
INK, MUTED = flow.INK, flow.MUTED


def save(fig, folder: Path, name: str) -> None:
    flow.OUT = folder
    flow.save(fig, name)


def canvas(xlim, ylim, fig_h, bottom, height, left=0.16):
    fig_w = left + height * (xlim[1] - xlim[0]) / (ylim[1] - ylim[0]) + 0.14
    fig = flow.plt.figure(figsize=(fig_w, fig_h))
    ax = flow.put_axes(fig, xlim, ylim, left, bottom, height)
    return fig, ax


def at(weights, layer, pos, key):
    return float(weights[layer - 1][pos, key])


def load(run: Path, device):
    cfg, tok, model, step = base.load_model(run, device)
    world = ParametricWorld(WORLD)
    wid = base.world_token(cfg, tok, world.root)
    nodes = world.node_vocab_ids(tok)
    inv = {token: index for index, token in enumerate(nodes)}
    return {
        "cfg": cfg, "tok": tok, "model": model, "step": step, "world": world,
        "wid": wid, "nodes": nodes, "inv": inv, "adj": world.adj(),
    }


def forward(pack, protocol, index):
    tok, world = pack["tok"], pack["world"]
    row = world.render_query(int(index), protocol, tok)
    tokens = [pack["wid"], *row.tokens]
    plen = row.prompt_len + 1
    device = next(pack["model"].parameters()).device
    ids = torch.tensor(tokens, device=device)[None]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, weights = pack["model"](ids, return_attn=True)
    parsed = records(tokens, plen, tok) if protocol == "dfs_state" else []
    return {
        "tokens": tokens,
        "plen": plen,
        "W": [w[0, 0].float().cpu() for w in weights],
        "logits": logits[0].float().cpu(),
        "records": parsed,
        "label": int(world.queries["label"][index]),
        "source": int(world.queries["s_topo"][index]),
        "target": int(world.queries["t_topo"][index]),
    }


def nm(pack, token):
    if token in pack["inv"]:
        return f"V{pack['inv'][token]}"
    return pack["tok"].name(token).strip("<>")


def node_body(pack, item):
    inv = pack["inv"]
    body = []
    cursor = item["plen"]
    while cursor < len(item["tokens"]) and item["tokens"][cursor] in inv:
        body.append(cursor)
        cursor += 1
    order = [inv[item["tokens"][pos]] for pos in body]
    parent = {order[0]: None}
    seen = [order[0]]
    for node in order[1:]:
        parent[node] = next((prev for prev in reversed(seen) if node in pack["adj"][prev]), None)
        seen.append(node)
    return body, order, parent


def measure_state(pack):
    tok = pack["tok"]
    acc = {key: [] for key in (
        "push_l2", "push_l4", "push_l6", "push_l2_list", "push_n",
        "vis_l6", "vis_first_arg", "top_l5", "top_after_pop", "pop_l3", "found_l3",
        "first_top_l2_bar", "first_top_l6_query", "vis0_query",
    )}
    for index in pack["world"].indices("test", "full").tolist():
        item = forward(pack, "dfs_state", index)
        W, tokens = item["W"], item["tokens"]
        for ri, rec in enumerate(item["records"]):
            a, b, c = rec["separators"]
            action = tok.name(rec["action"][0]).strip("<>")
            cands = list(range(b + 1, c))
            if action == "PUSH" and len(rec["action"]) > 1:
                arg = next((p for p in cands if tokens[p] == rec["action"][1]), None)
                if arg is not None:
                    acc["push_l2"].append(at(W, 2, c + 1, arg))
                    acc["push_l4"].append(at(W, 4, c + 1, arg))
                    acc["push_l6"].append(at(W, 6, c + 1, arg))
                    acc["push_l2_list"].append(sum(at(W, 2, c + 1, p) for p in cands))
                    acc["push_n"].append(1)
            if ri == 0:
                acc["first_top_l2_bar"].append(at(W, 2, a - 1, item["plen"] - 1))
                acc["first_top_l6_query"].append(at(W, 6, a - 1, 1))
                if a + 1 < b:
                    acc["vis0_query"].append(at(W, 6, a + 1, 1))
                continue
            prev = item["records"][ri - 1]
            prev_act = prev["separators"][2] + 1
            acc["top_l5"].append(at(W, 5, a - 1, prev_act))
            if tok.name(tokens[prev_act]).strip("<>") == "POP":
                acc["top_after_pop"].append(at(W, 5, a - 1, prev_act))
            first = item["records"][0]
            first_arg = first["separators"][2] + 2 if len(first["action"]) > 1 else None
            if first_arg is not None and a + 1 < b:
                acc["vis_first_arg"].append(at(W, 6, a + 1, first_arg))
            if len(prev["action"]) > 1:
                prev_arg = prev_act + 1
                if a + 1 < b:
                    acc["vis_l6"].append(at(W, 6, a + 1, prev_arg))
                if action in {"POP", "FOUND"}:
                    acc[f"{action.lower()}_l3"].append(at(W, 3, c, prev_arg))
    means = {key: (sum(vals) / len(vals), len(vals)) for key, vals in acc.items() if vals}
    example = forward(pack, "dfs_state", 16)
    return means, example


def measure_nodes(pack):
    tok = pack["tok"]
    yes_id = tok.id("<YES>")
    child_hit = gold_hit = descend_n = 0
    back_gold = back_n = 0
    owner_l5, chain_l5, owner_l1 = [], [], []
    stop = {key: {"l1": [], "l6": [], "bar1": [], "bar6": []} for key in ("yes", "no")}
    for index in pack["world"].indices("test", "full").tolist():
        item = forward(pack, "dfs_nodes", index)
        body, order, parent = node_body(pack, item)
        if len(order) < 2:
            continue
        W, tokens, plen = item["W"], item["tokens"], item["plen"]
        for k, pos in enumerate(body):
            if k + 1 < len(order):
                nxt = order[k + 1]
                kind = "descend" if parent.get(nxt) == order[k] else "backtrack"
                gold = tokens[body[k + 1]]
            else:
                kind = "stop"
                gold = yes_id if item["label"] else tok.id("<NO>")
                nxt = None
            pred = int(torch.argmax(item["logits"][pos]))
            if kind == "descend":
                descend_n += 1
                gold_hit += int(pred == gold)
                child_hit += int(pred in pack["inv"] and pack["inv"][pred] in pack["adj"][order[k]])
            elif kind == "backtrack":
                back_n += 1
                back_gold += int(pred == gold)
                chain = []
                cur = order[k]
                while cur is not None:
                    chain.append(body[order.index(cur)])
                    if cur == parent.get(nxt):
                        break
                    cur = parent.get(cur)
                owner_l5.append(at(W, 5, pos, chain[-1]))
                chain_l5.append(sum(at(W, 5, pos, p) for p in chain))
                owner_l1.append(at(W, 1, pos, chain[-1]))
            else:
                bucket = stop["yes" if item["label"] else "no"]
                bucket["l1"].append(at(W, 1, pos, plen - 2))
                bucket["l6"].append(at(W, 6, pos, plen - 2))
                bucket["bar1"].append(at(W, 1, pos, plen - 1))
                bucket["bar6"].append(at(W, 6, pos, plen - 1))
    means = {
        "descend_child": (child_hit, descend_n),
        "descend_gold": (gold_hit, descend_n),
        "back_gold": (back_gold, back_n),
        "owner_l5": (sum(owner_l5) / len(owner_l5), len(owner_l5)),
        "chain_l5": (sum(chain_l5) / len(chain_l5), len(chain_l5)),
        "owner_l1": (sum(owner_l1) / len(owner_l1), len(owner_l1)),
    }
    for label, bucket in stop.items():
        if not bucket["l1"]:
            continue
        means[f"stop_{label}_l1"] = (sum(bucket["l1"]) / len(bucket["l1"]), len(bucket["l1"]))
        means[f"stop_{label}_l6"] = (sum(bucket["l6"]) / len(bucket["l6"]), len(bucket["l6"]))
        means[f"stop_{label}_bar1"] = (sum(bucket["bar1"]) / len(bucket["bar1"]), len(bucket["bar1"]))
        means[f"stop_{label}_bar6"] = (sum(bucket["bar6"]) / len(bucket["bar6"]), len(bucket["bar6"]))
    return means


def chain_masses(pack, index, current, nxt):
    item = forward(pack, "dfs_nodes", index)
    body, order, parent = node_body(pack, item)
    k = order.index(current)
    pos = body[k]
    chain = []
    cur = current
    while cur is not None:
        chain.append(cur)
        if cur == parent.get(nxt):
            break
        cur = parent.get(cur)
    chain = list(reversed(chain))
    rows = []
    for node in chain:
        key = body[order.index(node)]
        rows.append({
            "name": f"V{node}",
            "l1": at(item["W"], 1, pos, key),
            "l5": at(item["W"], 5, pos, key),
            "l6": at(item["W"], 6, pos, key),
            "here": node == current,
        })
    gold = item["tokens"][body[k + 1]]
    return rows, float(item["logits"][pos, gold]), f"V{nxt}"


def descend_step(pack, index, current):
    item = forward(pack, "dfs_nodes", index)
    body, order, parent = node_body(pack, item)
    k = order.index(current)
    pos = body[k]
    nxt = order[k + 1]
    kids = pack["adj"][current]
    scores = sorted(
        ((float(item["logits"][pos, pack["nodes"][kid]]), kid) for kid in kids),
        reverse=True,
    )
    return {
        "name": f"V{current}",
        "nxt": f"V{nxt}",
        "nkids": len(kids),
        "l6_self": at(item["W"], 6, pos, pos),
        "l6_query": at(item["W"], 6, pos, 1),
        "l6_bar": at(item["W"], 6, pos, item["plen"] - 1),
        "l6_world": at(item["W"], 6, pos, 0),
        "l6_dst": at(item["W"], 6, pos, item["plen"] - 2),
        "scores": [(f"V{kid}", score) for score, kid in scores],
    }


def stop_step(pack, index):
    item = forward(pack, "dfs_nodes", index)
    body, order, _parent = node_body(pack, item)
    pos = body[-1]
    plen = item["plen"]
    yes = pack["tok"].id("<YES>")
    return {
        "source": f"V{item['source']}",
        "target": f"V{item['target']}",
        "last": f"V{order[-1]}",
        "l1_dst": at(item["W"], 1, pos, plen - 2),
        "l6_dst": at(item["W"], 6, pos, plen - 2),
        "l1_src": at(item["W"], 1, pos, plen - 3),
        "l6_src": at(item["W"], 6, pos, plen - 3),
        "l1_query": at(item["W"], 1, pos, 1),
        "l1_bar": at(item["W"], 1, pos, plen - 1),
        "l6_bar": at(item["W"], 6, pos, plen - 1),
        "l3_src": at(item["W"], 3, pos, plen - 3),
        "l3_dst": at(item["W"], 3, pos, plen - 2),
        "yes": float(item["logits"][pos, yes]),
    }


def f2(value):
    return f"{value:.2f}"


def candidates(example):
    W = example["W"]
    # Record 0: visited V6 @7, candidate V9 @9. Record 1: visited V9 @20, candidate V13 @22, PUSH @13.
    xlim, ylim = (0, 6.05), (1.05, 3.25)
    fig, ax = canvas(xlim, ylim, 2.85, 0.42, 2.28)
    flow.tag(ax, 0.08, 3.08, "First candidate", fs=10.5, ha="left", color=INK)
    flow.tag(ax, 0.08, 2.84, "It reads the node being expanded.", fs=8, ha="left")
    left = flow.row_boxes(
        ax, 0.12, 1.55,
        ["V6", "V9"],
        [GREEN_F, GOLD_F],
        [GREEN, GOLD],
    )
    flow.arrow(ax, (left[1][0], left[1][1] + H), (left[0][0], left[0][1] + H), BLUE, rad=0.30)
    flow.tag(ax, left[0][0], 2.48, f"L5 {f2(at(W, 5, 9, 7))}", color=BLUE, fs=7.2)
    flow.tag(ax, left[0][0], 2.26, f"L6 {f2(at(W, 6, 9, 7))}", color=VERM, fs=7.2)
    flow.tag(ax, left[0][0], 1.32, "expanded", color=GREEN, fs=7.0)
    flow.tag(ax, left[1][0], 1.32, "this name", fs=7.0)

    flow.tag(ax, 3.15, 3.08, "Later candidate", fs=10.5, ha="left", color=INK)
    flow.tag(ax, 3.15, 2.84, "Layer 1 reads the node being expanded.", fs=8, ha="left")
    right = flow.row_boxes(
        ax, 3.15, 1.55,
        ["PUSH", "V9", "V13"],
        [GRAY_F, GREEN_F, GOLD_F],
        ["#888888", GREEN, GOLD],
    )
    flow.arrow(ax, (right[2][0], right[2][1] + H), (right[0][0], right[0][1] + H), VERM, rad=0.28)
    flow.arrow(ax, (right[2][0], right[2][1] + H), (right[1][0], right[1][1] + H), BLUE, rad=0.16)
    flow.tag(ax, right[0][0], 2.48, f"L6 {f2(at(W, 6, 22, 13))}", color=VERM, fs=7.2)
    flow.tag(ax, right[1][0], 2.26, f"L1 {f2(at(W, 1, 22, 20))}", color=BLUE, fs=7.2)
    flow.tag(ax, right[1][0], 1.32, "expanded", color=GREEN, fs=7.0)
    flow.tag(ax, right[2][0], 1.32, "this name", fs=7.0)
    fig.text(
        0.02, 0.025,
        "Candidates.  Fixed curriculum, 4×4, test index 16, V6 to V15.  Gold is the candidate name.",
        fontsize=7.2, color=MUTED, ha="left", va="bottom",
    )
    save(fig, STATE_OUT, "candidates.png")


def push(example, means):
    W = example["W"]
    # Record 1 candidates V13 @22, V15 @23, V14 @24, PUSH @26. Query target V15 @3.
    names = ["V13", "V15", "V14"]
    poss = [22, 23, 24]
    xlim, ylim = (0, 6.35), (0.72, 3.35)
    fig, ax = canvas(xlim, ylim, 3.15, 0.58, 2.42)
    flow.tag(ax, 0.10, 3.16, "PUSH reads the candidate list", fs=10.5, ha="left", color=INK)
    flow.tag(ax, 0.10, 2.92, "Layer 2 spreads. Layers 4 and 6 select the name that is written.", fs=8, ha="left")
    row = flow.row_boxes(
        ax, 0.12, 1.45,
        names, [VERM_F, GRAY_F, GRAY_F], [VERM, "#888888", "#888888"],
    )
    gold = flow.row_boxes(ax, 2.85, 1.45, ["PUSH"], [GOLD_F], [GOLD])
    src = (gold[0][0], gold[0][1] + H)
    for center, rad in zip(row, (0.22, 0.08, -0.08)):
        flow.arrow(ax, src, (center[0], center[1] + H), BLUE, rad=rad)
    for center, pos in zip(row, poss):
        flow.tag(ax, center[0], 1.20, f"L2 {f2(at(W, 2, 26, pos))}", color=BLUE, fs=7.2)
    flow.tag(ax, row[0][0], 2.42, f"L4 {f2(at(W, 4, 26, 22))}", color=VERM, fs=7.2)
    flow.tag(ax, row[0][0], 2.20, f"L6 {f2(at(W, 6, 26, 22))}", color=VERM, fs=7.2)
    flow.tag(ax, row[0][0], 0.96, "written", color=VERM, fs=7.0)
    target = flow.row_boxes(ax, 4.55, 1.45, ["V15"], [GREEN_F], [GREEN])
    flow.arrow(ax, src, (target[0][0], target[0][1] + H), GREEN, rad=-0.22)
    flow.tag(ax, target[0][0], 2.42, f"L3 {f2(at(W, 3, 26, 3))}", color=GREEN, fs=7.2)
    flow.tag(ax, target[0][0], 1.20, "query target", color=GREEN, fs=7.0)
    l2, n2 = means["push_l2"]
    l4, n4 = means["push_l4"]
    l6, _n6 = means["push_l6"]
    fig.text(
        0.03, 0.055,
        "Push.  Same trace, second PUSH, candidates V13, V15, V14.  Orange is the name that is written.",
        fontsize=7.2, color=MUTED, ha="left", va="bottom",
    )
    fig.text(
        0.03, 0.022,
        f"All {n4} PUSH steps on the 4×4 test.  Layer 2 on the written name {l2:.2f}.  "
        f"Layer 4 {l4:.2f}.  Layer 6 {l6:.2f}.",
        fontsize=7.2, color=MUTED, ha="left", va="bottom",
    )
    save(fig, STATE_OUT, "push.png")


def stack(example, means):
    W = example["W"]
    # First stack name V6 @5 reads prompt bar @4 and QUERY @1.
    # After POP, new top V9 @41 reads POP @38.
    xlim, ylim = (0, 6.35), (0.95, 3.15)
    fig, ax = canvas(xlim, ylim, 2.72, 0.46, 2.12)
    flow.tag(ax, 0.08, 2.98, "First stack name", fs=10.5, ha="left", color=INK)
    flow.tag(ax, 0.08, 2.74, "It reads the query marker and the bar.", fs=8, ha="left")
    left = flow.row_boxes(
        ax, 0.10, 1.45,
        ["QUERY", "V6", "|"],
        [BLUE_F, GOLD_F, GREEN_F],
        [BLUE, GOLD, GREEN],
    )
    flow.arrow(ax, (left[1][0], left[1][1] + H), (left[0][0], left[0][1] + H), BLUE, rad=0.22)
    flow.arrow(ax, (left[1][0], left[1][1] + H), (left[2][0], left[2][1] + H), GREEN, rad=-0.22)
    flow.tag(ax, left[0][0], 1.20, f"L6 {f2(at(W, 6, 5, 1))}", color=BLUE, fs=7.2)
    flow.tag(ax, left[2][0], 1.20, f"L2 {f2(at(W, 2, 5, 4))}", color=GREEN, fs=7.2)

    flow.tag(ax, 3.35, 2.98, "New top after POP", fs=10.5, ha="left", color=INK)
    flow.tag(ax, 3.35, 2.74, "It reads the POP that uncovered it.", fs=8, ha="left")
    right = flow.row_boxes(
        ax, 3.40, 1.45,
        ["POP", "V6", "V9"],
        [VERM_F, GRAY_F, GOLD_F],
        [VERM, "#888888", GOLD],
    )
    flow.arrow(ax, (right[2][0], right[2][1] + H), (right[0][0], right[0][1] + H), VERM, rad=0.22)
    flow.tag(ax, right[0][0], 1.20, f"L5 {f2(at(W, 5, 41, 38))}", color=VERM, fs=7.2)
    flow.tag(ax, right[2][0], 1.20, "new top", fs=7.0)
    top, ntop = means["top_l5"]
    after, nafter = means["top_after_pop"]
    bar, nbar = means["first_top_l2_bar"]
    fig.text(
        0.03, 0.04,
        f"Stack.  Layer 2 on the prompt bar {bar:.2f} ({nbar}).  "
        f"Layer 5 on the previous action {top:.2f} ({ntop}).  After a POP, {after:.2f} ({nafter}).",
        fontsize=7.2, color=MUTED, ha="left", va="bottom",
    )
    save(fig, STATE_OUT, "stack.png")


def stack_middle(example, means):
    W = example["W"]
    # Before the POP the stack is V6 V9 V13. After it, V6 V9, and V9 reads POP.
    xlim, ylim = (0, 7.15), (0.85, 3.15)
    fig, ax = canvas(xlim, ylim, 2.70, 0.46, 2.10)
    flow.tag(ax, 0.08, 2.98, "Frame under the top", fs=10.5, ha="left", color=INK)
    flow.tag(ax, 0.08, 2.74, "POP removes V13. The parent V9 becomes the top and reads that POP.", fs=8, ha="left")
    before = flow.row_boxes(
        ax, 0.10, 1.40,
        ["V6", "V9", "V13", "POP"],
        [GRAY_F, GREEN_F, GRAY_F, VERM_F],
        ["#888888", GREEN, "#888888", VERM],
    )
    flow.tag(ax, before[0][0], 1.16, "bottom", fs=7.0)
    flow.tag(ax, before[1][0], 1.16, "parent", color=GREEN, fs=7.0)
    flow.tag(ax, before[2][0], 1.16, "popped", fs=7.0)
    after = flow.row_boxes(ax, 4.55, 1.40, ["V6", "V9"], [GRAY_F, GOLD_F], ["#888888", GOLD])
    flow.arrow(ax, (after[1][0], after[1][1] + H), (before[3][0], before[3][1] + H), VERM, rad=0.22)
    flow.tag(ax, (after[1][0] + before[3][0]) / 2, 2.22, f"L5 {f2(at(W, 5, 41, 38))}", color=VERM, fs=7.2)
    flow.tag(ax, after[1][0], 1.16, "now the top", fs=7.0)
    after_m, n = means["top_after_pop"]
    fig.text(
        0.03, 0.04,
        f"Same trace, the POP of V13.  After a POP, layer 5 of the new top on that POP is {after_m:.2f} ({n}).",
        fontsize=7.2, color=MUTED, ha="left", va="bottom",
    )
    save(fig, STATE_OUT, "stack_middle.png")


def visited(example, means):
    W = example["W"]
    # Record 0 visited V6 @7 reads QUERY @1. Record 1 visited V6 @19 reads previous arg V9 @14.
    xlim, ylim = (0, 5.55), (0.95, 3.15)
    fig, ax = canvas(xlim, ylim, 2.78, 0.48, 2.15)
    flow.tag(ax, 0.08, 2.98, "First record", fs=10.5, ha="left", color=INK)
    flow.tag(ax, 0.08, 2.74, "It reads the query marker.", fs=8, ha="left")
    left = flow.row_boxes(
        ax, 0.12, 1.45,
        ["QUERY", "V6"],
        [BLUE_F, GOLD_F],
        [BLUE, GOLD],
    )
    flow.arrow(ax, (left[1][0], left[1][1] + H), (left[0][0], left[0][1] + H), BLUE, rad=0.40)
    flow.tag(ax, (left[0][0] + left[1][0]) / 2, 2.32, f"L6 {f2(at(W, 6, 7, 1))}", color=BLUE, fs=7.2)

    flow.tag(ax, 2.85, 2.98, "Later records", fs=10.5, ha="left", color=INK)
    flow.tag(ax, 2.85, 2.74, "It stays on the first PUSH argument.", fs=8, ha="left")
    right = flow.row_boxes(
        ax, 2.90, 1.45,
        ["V9", "V6"],
        [VERM_F, GOLD_F],
        [VERM, GOLD],
    )
    flow.arrow(ax, (right[1][0], right[1][1] + H), (right[0][0], right[0][1] + H), VERM, rad=0.40)
    flow.tag(ax, (right[0][0] + right[1][0]) / 2, 2.32, f"L6 {f2(at(W, 6, 19, 14))}", color=VERM, fs=7.2)
    flow.tag(ax, right[0][0], 1.20, "first argument", color=VERM, fs=7.0)
    first_q, nfirst = means["vis0_query"]
    held, nheld = means["vis_first_arg"]
    fig.text(
        0.03, 0.04,
        f"Visited.  Layer 6 on the query marker {first_q:.2f} ({nfirst}).  "
        f"Later records stay on the first argument, {held:.2f} ({nheld}).",
        fontsize=7.2, color=MUTED, ha="left", va="bottom",
    )
    save(fig, STATE_OUT, "visited.png")


def action(example, means):
    W = example["W"]
    # POP is predicted at bar @37, which reads previous arg V13 @27.
    # FOUND is predicted at bar @62, which reads V15 @51.
    # PUSH @13 reads candidate V9 @9.
    xlim, ylim = (0, 5.35), (0.55, 3.85)
    fig, ax = canvas(xlim, ylim, 3.55, 0.62, 2.78)
    flow.tag(ax, 0.08, 3.66, "POP and FOUND", fs=10.5, ha="left", color=INK)
    flow.tag(ax, 0.08, 3.42, "The bar reads the node just pushed.", fs=8, ha="left")
    pop = flow.row_boxes(ax, 0.12, 2.05, ["V13", "POP"], [VERM_F, GOLD_F], [VERM, GOLD])
    flow.arrow(ax, (pop[1][0], pop[1][1] + H), (pop[0][0], pop[0][1] + H), VERM, rad=0.42)
    flow.tag(ax, (pop[0][0] + pop[1][0]) / 2, 2.95, f"L3 {f2(at(W, 3, 37, 27))}", color=VERM, fs=7.2)
    found = flow.row_boxes(ax, 0.12, 0.85, ["V15", "FOUND"], [GREEN_F, GOLD_F], [GREEN, GOLD])
    flow.arrow(ax, (found[1][0], found[1][1] + H), (found[0][0], found[0][1] + H), GREEN, rad=0.42)
    flow.tag(ax, (found[0][0] + found[1][0]) / 2, 1.72, f"L3 {f2(at(W, 3, 62, 51))}", color=GREEN, fs=7.2)
    flow.tag(ax, found[0][0], 0.62, "the target", color=GREEN, fs=7.0)

    flow.tag(ax, 3.05, 3.66, "PUSH", fs=10.5, ha="left", color=INK)
    flow.tag(ax, 3.05, 3.42, "PUSH reads the name it writes.", fs=8, ha="left")
    push_row = flow.row_boxes(ax, 3.10, 2.05, ["V9", "PUSH"], [VERM_F, GOLD_F], [VERM, GOLD])
    flow.arrow(ax, (push_row[1][0], push_row[1][1] + H), (push_row[0][0], push_row[0][1] + H), VERM, rad=0.42)
    flow.tag(ax, (push_row[0][0] + push_row[1][0]) / 2, 3.02, f"L4 {f2(at(W, 4, 13, 9))}", color=VERM, fs=7.2)
    flow.tag(ax, (push_row[0][0] + push_row[1][0]) / 2, 2.80, f"L6 {f2(at(W, 6, 13, 9))}", color=BLUE, fs=7.2)
    pop_m, npop = means["pop_l3"]
    found_m, nfound = means["found_l3"]
    l4, npush = means["push_l4"]
    fig.text(
        0.03, 0.055,
        "Action.  Gold is the position drawn.  POP and FOUND are the predicting bars.",
        fontsize=7.2, color=MUTED, ha="left", va="bottom",
    )
    fig.text(
        0.03, 0.022,
        f"POP {pop_m:.2f} ({npop}), FOUND {found_m:.2f} ({nfound}) at layer 3.  "
        f"PUSH layer 4 {l4:.2f} ({npush}).",
        fontsize=7.2, color=MUTED, ha="left", va="bottom",
    )
    save(fig, STATE_OUT, "action.png")


def descend(steps, means):
    one, many = steps
    xlim, ylim = (0, 5.85), (0.85, 3.55)
    fig, ax = canvas(xlim, ylim, 3.25, 0.58, 2.52)
    flow.tag(ax, 0.10, 3.36, "Descend", fs=11, ha="left", color=BLUE)
    flow.tag(ax, 0.10, 3.08, "Reads itself", fs=10, ha="left", color=INK)
    flow.tag(ax, 0.10, 2.84, "Layer 6 stays on the current node.", fs=8, ha="left")
    left = flow.row_boxes(ax, 0.15, 1.70, [one["name"]], [GOLD_F], [GOLD])
    flow.tag(ax, left[0][0], 2.32, f"L6 {f2(one['l6_self'])}", color=BLUE, fs=7.4)
    written = next(score for name, score in one["scores"] if name == one["nxt"])
    flow.tag(ax, left[0][0], 1.46, f"writes {one['nxt']}, {written:.1f}", fs=7.2)
    other = ", ".join(f"{name} {score:.1f}" for name, score in one["scores"][1:])
    flow.tag(ax, 0.15, 1.18, f"also {other}", fs=7.2, ha="left")

    flow.tag(ax, 2.85, 3.08, "Reads the query", fs=10, ha="left", color=INK)
    flow.tag(ax, 2.85, 2.84, "Layer 6 moves onto the query marker.", fs=8, ha="left")
    right = flow.row_boxes(
        ax, 2.90, 1.70,
        ["QUERY", many["name"]],
        [BLUE_F, GOLD_F],
        [BLUE, GOLD],
    )
    flow.arrow(ax, (right[1][0], right[1][1] + H), (right[0][0], right[0][1] + H), BLUE, rad=0.0)
    flow.tag(ax, (right[0][0] + right[1][0]) / 2, 2.40, f"L6 {f2(many['l6_query'])}", color=BLUE, fs=7.4)
    score_line = "  ".join(f"{name} {score:.1f}" for name, score in many["scores"])
    flow.tag(ax, 2.90, 1.46, score_line, fs=7.2, ha="left")
    flow.tag(ax, 2.90, 1.22, f"writes {many['nxt']}", fs=7.2, ha="left")
    hit, n = means["descend_child"]
    gold, _ng = means["descend_gold"]
    fig.text(
        0.03, 0.055,
        "Descend.  Left: index 47, V11 to V13.  Right: index 18, V9 to V14.",
        fontsize=7.2, color=MUTED, ha="left", va="bottom",
    )
    fig.text(
        0.03, 0.022,
        f"The next token is a child on {hit} of {n} steps.  The frozen neighbor order matches on {gold} of {n}.",
        fontsize=7.2, color=MUTED, ha="left", va="bottom",
    )
    save(fig, NODES_OUT, "descend.png")


def backtrack(one, three, means):
    xlim, ylim = (0, 6.55), (0.85, 3.85)
    fig, ax = canvas(xlim, ylim, 3.55, 0.72, 2.68)

    def paint(x, title, note, rows):
        flow.tag(ax, x, 3.66, title, fs=10, ha="left", color=INK)
        flow.tag(ax, x, 3.42, note, fs=8, ha="left")
        faces, edges = [], []
        for row in rows:
            if row["here"]:
                faces.append(GOLD_F)
                edges.append(GOLD)
            elif row is rows[0]:
                faces.append(VERM_F)
                edges.append(VERM)
            else:
                faces.append(GRAY_F)
                edges.append("#888888")
        centers = flow.row_boxes(ax, x, 1.85, [row["name"] for row in rows], faces, edges)
        src = next(center for center, row in zip(centers, rows) if row["here"])
        others = [(center, row) for center, row in zip(centers, rows) if not row["here"]]
        for index, (center, row) in enumerate(others):
            rad = 0.08 + 0.08 * (len(others) - 1 - index)
            flow.arrow(ax, (src[0], src[1] + H), (center[0], center[1] + H), VERM, rad=rad)
            flow.tag(ax, center[0], center[1] + H + 0.22 + 0.14 * (len(others) - 1 - index), f2(row["l5"]), color=VERM, fs=7.2)
            flow.tag(ax, center[0], 1.60, f2(row["l1"]), color=BLUE, fs=7.0)
        flow.tag(ax, src[0], 1.60, f2(rows[-1]["l1"]), color=BLUE, fs=7.0)
        flow.tag(ax, centers[0][0], 1.36, "owner", color=VERM, fs=7.0)
        flow.tag(ax, src[0], 1.36, "this step", fs=7.0)

    paint(0.10, "One frame", "Layer 5 reads the parent.", one)
    paint(2.55, "Three frames", "Layer 5 reads the far owner.", three)
    gold, n = means["back_gold"]
    own, _ = means["owner_l5"]
    chain, _ = means["chain_l5"]
    own1, _ = means["owner_l1"]
    fig.text(
        0.03, 0.062,
        f"Backtrack.  Orange is layer 5, blue is layer 1.  Left: index 18.  Right: index 47.  Layer 1 on the owner {own1:.2f}.",
        fontsize=7.2, color=MUTED, ha="left", va="bottom",
    )
    fig.text(
        0.03, 0.018,
        f"Gold next node on {gold} of {n} returns.  Layer 5 on the owner {own:.2f}, on the chain {chain:.2f}.",
        fontsize=7.2, color=MUTED, ha="left", va="bottom",
    )
    save(fig, NODES_OUT, "backtrack.png")


def stop(step, means):
    xlim, ylim = (0, 6.15), (0.95, 3.35)
    fig, ax = canvas(xlim, ylim, 3.15, 0.68, 2.32)
    flow.tag(ax, 0.08, 3.16, "Stop", fs=11, ha="left", color=GREEN)
    flow.tag(ax, 0.08, 2.92, "The last node reads the query target.", fs=8, ha="left")
    prompt = flow.row_boxes(
        ax, 0.10, 1.45,
        ["QUERY", step["source"], step["target"], "|"],
        [GRAY_F, GRAY_F, VERM_F, GRAY_F],
        ["#888888", "#888888", VERM, "#888888"],
    )
    last = flow.row_boxes(ax, 4.85, 1.45, [step["last"]], [GOLD_F], [GOLD])
    flow.arrow(
        ax, (last[0][0], last[0][1] + H),
        (prompt[2][0], prompt[2][1] + H), VERM, rad=0.22,
    )
    flow.tag(ax, (last[0][0] + prompt[2][0]) / 2, 2.42, f"L1 {f2(step['l1_dst'])}", color=VERM, fs=7.2)
    flow.tag(ax, (last[0][0] + prompt[2][0]) / 2, 2.20, f"L6 {f2(step['l6_dst'])}", color=VERM, fs=7.2)
    flow.tag(ax, prompt[3][0], 1.20, f"L1 {f2(step['l1_bar'])}", color=MUTED, fs=7.0)
    flow.tag(ax, prompt[0][0], 1.20, f"L1 {f2(step['l1_query'])}", color=MUTED, fs=7.0)
    flow.tag(ax, prompt[1][0], 1.20, f"L1 {f2(step['l1_src'])}", color=MUTED, fs=7.0)
    flow.tag(ax, last[0][0], 1.20, f"YES {step['yes']:.1f}", fs=7.2)
    y1, ny = means["stop_yes_l1"]
    y6, _ = means["stop_yes_l6"]
    yb1, _ = means["stop_yes_bar1"]
    n1, nn = means["stop_no_l1"]
    n6, _ = means["stop_no_l6"]
    fig.text(
        0.03, 0.062,
        "Stop.  Index 16, V6 to V15.  Gold is the last node.  Orange is its mass on the query target.",
        fontsize=7.2, color=MUTED, ha="left", va="bottom",
    )
    fig.text(
        0.03, 0.016,
        f"YES n={ny}: target layer 1 {y1:.2f}, layer 6 {y6:.2f}, bar {yb1:.2f}.  "
        f"NO n={nn}: target {n1:.2f} and {n6:.2f}.",
        fontsize=7.2, color=MUTED, ha="left", va="bottom",
    )
    save(fig, NODES_OUT, "stop.png")


def main():
    flow.style()
    device = torch.device(DEVICE)
    state = load(STATE_RUN, device)
    print("state step", state["step"])
    state_means, example = measure_state(state)
    for key, val in state_means.items():
        print(f"  {key} {val[0]:.3f} n{val[1]}")
    candidates(example)
    push(example, state_means)
    stack(example, state_means)
    stack_middle(example, state_means)
    visited(example, state_means)
    action(example, state_means)
    del state["model"]
    torch.cuda.empty_cache()

    nodes = load(NODES_RUN, device)
    print("nodes step", nodes["step"])
    node_means = measure_nodes(nodes)
    for key, val in node_means.items():
        print(f"  {key} {val}")
    one = descend_step(nodes, 47, 11)
    many = descend_step(nodes, 18, 9)
    print("descend", one)
    print("descend", many)
    back_one, logit_one, nxt_one = chain_masses(nodes, 18, 13, 15)
    back_three, logit_three, nxt_three = chain_masses(nodes, 47, 14, 7)
    print("back one", nxt_one, logit_one, back_one)
    print("back three", nxt_three, logit_three, back_three)
    stop_info = stop_step(nodes, 16)
    print("stop", stop_info)
    descend((one, many), node_means)
    backtrack(back_one, back_three, node_means)
    stop(stop_info, node_means)


if __name__ == "__main__":
    main()

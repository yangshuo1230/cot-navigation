#!/usr/bin/env python3
"""DFS-node attention as token diagrams, in the dfs_flow style.

Layered 4×4 test index 2, query V209→V191. The gold box is the position
whose attention is drawn. Hard DAG 4×4 test index 9 is the backtrack contrast.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import plot_dfs_cot_attention as base  # noqa: E402
import plot_dfs_field_flow as flow  # noqa: E402
import plot_dfs_nodes_attention_examples as ex  # noqa: E402
import plot_path_dag_attention as pathfig  # noqa: E402

OUT = ROOT / "figures/scale_v1/dfs_nodes_flow"
BLUE, VERM, GREEN, GOLD = flow.BLUE, flow.VERM, flow.GREEN, flow.GOLD
BLUE_F, VERM_F, GREEN_F, GOLD_F, GRAY_F = flow.BLUE_F, flow.VERM_F, flow.GREEN_F, flow.GOLD_F, flow.GRAY_F
INK, MUTED = flow.INK, flow.MUTED
W, H, G = 0.78, 0.42, 0.10


def short(name: str) -> str:
    return name.strip("<>")


def measure(run, data, index, device):
    _cfg, tok, model, step = base.load_model(run, device)
    item = ex.prepare(data, tok, index)
    layers = ex.all_layers(model, item["tokens"], device)
    name = lambda node: short(tok.name(item["node_tokens"][node]))
    order = item["order"]
    parent = item["parent"]
    plen = item["prompt_len"]
    steps = []
    for index_node, node in enumerate(order[:-1]):
        gold = order[index_node + 1]
        dec = plen + index_node
        relays = [
            (3 * edge + 3, dst)
            for edge, (src, dst) in enumerate(item["edges"])
            if src == parent[gold] and 3 * edge + 3 < plen - 4
        ]
        steps.append({
            "cur": name(node),
            "gold": name(gold),
            "parent": name(parent[gold]),
            "outs": len(item["adj"][node]),
            "self": float(layers[0][dec, dec]),
            "relays": [(name(dst), float(layers[3][dec, pos])) for pos, dst in relays],
            "l5": [(name(order[j]), float(layers[4][dec, plen + j])) for j in range(index_node + 1)],
            "l6": [(name(order[j]), float(layers[5][dec, plen + j])) for j in range(index_node + 1)],
            "l6_bar": float(layers[5][dec, plen - 1]),
        })
    edges = []
    for edge, (src, dst) in enumerate(item["edges"]):
        pos = 3 * edge + 3
        if src == order[0] and pos < plen - 4:
            edges.append((name(src), name(dst), float(layers[0][pos, 3 * edge + 2])))
    last = plen + len(order) - 1
    bar = plen - 1
    return {
        "step": step,
        "order": [name(node) for node in order],
        "source": name(order[0]),
        "target": name(item["target"]),
        "edges": edges,
        "steps": steps,
        "bar_l3_target": float(layers[2][bar, plen - 2]),
        "bar_l4_source": float(layers[3][bar, plen - 3]),
        "answer_self": float(layers[0][last, last]),
        "answer_l6_prefix": float(layers[5][last, plen:last].sum()),
        "answer_l6_bar": float(layers[5][last, bar]),
    }


def save(fig, name):
    flow.OUT = OUT
    flow.save(fig, name)


def row(ax, x0, y, names, faces, edges):
    centers = []
    x = x0
    for token, face, edge in zip(names, faces, edges):
        flow.box(ax, x, y, W, H, token, face, edge)
        centers.append((x + W / 2, y))
        x += W + G
    return centers


def edge_figure(trace):
    xlim, ylim = (0, 8.70), (-0.05, 3.85)
    fig_h = 4.35
    height = 3.35
    fig_w = 0.36 + height * (xlim[1] - xlim[0]) / (ylim[1] - ylim[0]) + 0.28
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = flow.put_axes(fig, xlim, ylim, 0.28, 0.62, height)
    flow.tag(ax, 0.08, 3.62, "The next token copies the edge head", fs=10.5, ha="left", color=INK)
    flow.tag(ax, 0.08, 3.36, "Layer 1 writes each destination into the following token. These three edges leave V209.", fs=8, ha="left")
    starts = [0.20, 3.05, 5.90]
    for (src, dst, mass), x0 in zip(trace["edges"], starts):
        centers = row(
            ax, x0, 2.15,
            ["<EDGE>", src, dst],
            [GRAY_F, GREEN_F, BLUE_F],
            ["#888888", GREEN, BLUE],
        )
        cx = centers[2][0]
        cy = 0.85
        flow.box(ax, cx - W / 2, cy, W, H, "+3", BLUE_F, BLUE)
        flow.arrow(ax, (cx, 2.15), (cx, cy + H), BLUE)
        flow.tag(ax, cx - W / 2 - 0.08, cy + H / 2, f"L1 {mass:.2f}", color=BLUE, fs=7.5, ha="right")
    fig.text(
        0.03, 0.045,
        "Layered 4×4, test index 2, V209 to V191.  Layer 1 on the destination at slot +3 is 1.00 on these three edges.",
        fontsize=7.2, color=MUTED, ha="left", va="bottom",
    )
    fig.text(
        0.03, 0.012,
        "Mean over every edge: 1.00 on 4×4, 1.00 on 6×6, 1.00 on 10×10.  Slot +5 stays near 0.  Hard DAG layer 1 is 0.91.",
        fontsize=7.2, color=MUTED, ha="left", va="bottom",
    )
    save(fig, "edge.png")


def continue_figure(trace):
    step = trace["steps"][0]
    xlim, ylim = (0, 4.85), (-0.15, 3.55)
    fig_h = 4.15
    height = 3.15
    fig_w = 0.34 + height * (xlim[1] - xlim[0]) / (ylim[1] - ylim[0]) + 0.28
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = flow.put_axes(fig, xlim, ylim, 0.28, 0.62, height)
    flow.tag(ax, 0.10, 3.32, "Continue reads every outgoing copy", fs=10.5, ha="left", color=INK)
    flow.tag(ax, 0.10, 3.06, "Three unused edges. Layer 4 reads each +3 slot.", fs=8, ha="left")
    slots = row(
        ax, 0.85, 1.85,
        [dst for dst, _mass in step["relays"]],
        [BLUE_F, BLUE_F, BLUE_F],
        [BLUE, BLUE, BLUE],
    )
    for (cx, cy), (_dst, mass), rad in zip(slots, step["relays"], (0.18, 0.0, -0.18)):
        flow.tag(ax, cx, cy + H + 0.18, f"L4 {mass:.2f}", color=BLUE, fs=7.5)
    here = row(ax, slots[1][0] - W / 2, 0.28, [step["cur"]], [GOLD_F], [GOLD])
    src = (here[0][0], here[0][1] + H)
    for (cx, cy), rad in zip(slots, (0.16, 0.0, -0.16)):
        flow.arrow(ax, src, (cx, cy), BLUE, rad=rad)
    flow.tag(ax, here[0][0], 0.08, f"L1 {step['self']:.2f}   writes {step['gold']}", fs=7.5)
    fig.text(
        0.03, 0.045,
        "Same trace, first step. The three slots are nearly even, and V14 is written.",
        fontsize=7.2, color=MUTED, ha="left", va="bottom",
    )
    fig.text(
        0.03, 0.012,
        "Mean on the parent’s slots: 0.59, 0.57, 0.58. Chosen slot: 0.32, 0.31, 0.32.",
        fontsize=7.2, color=MUTED, ha="left", va="bottom",
    )
    save(fig, "continue.png")


def backtrack_figure(trace, hard):
    step = trace["steps"][2]
    hard_step = next(item for item in hard["steps"] if item["outs"] == 0)
    xlim, ylim = (0, 12.6), (-0.05, 3.70)
    fig_h = 4.25
    height = 3.20
    fig_w = 0.32 + height * (xlim[1] - xlim[0]) / (ylim[1] - ylim[0]) + 0.24
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = flow.put_axes(fig, xlim, ylim, 0.24, 0.64, height)
    flow.tag(ax, 0.10, 3.48, "No outgoing edge", fs=10.5, ha="left", color=INK)
    flow.tag(ax, 0.10, 3.22, "V204 has none. Layer 5 reads the nodes already written.", fs=8, ha="left")
    faces = [GREEN_F, VERM_F, GOLD_F]
    edges = [GREEN, VERM, GOLD]
    nodes = row(ax, 0.15, 1.70, [token for token, _mass in step["l5"]], faces, edges)
    src = (nodes[-1][0], nodes[-1][1] + H)
    for (cx, cy), (_token, mass), rad in zip(nodes[:-1], step["l5"][:-1], (0.22, 0.08)):
        flow.arrow(ax, src, (cx, cy + H), VERM, rad=rad)
        flow.tag(ax, cx, cy + H + 0.20, f"L5 {mass:.2f}", color=VERM, fs=7.5)
    flow.tag(ax, nodes[-1][0], nodes[-1][1] - 0.20, f"L5 {step['l5'][-1][1]:.2f}", color=GOLD, fs=7.5)
    flow.tag(ax, nodes[-1][0], 1.28, f"writes {step['gold']}", fs=7.5)
    flow.tag(ax, nodes[1][0], 1.28, "parent", color=VERM, fs=7.5)

    flow.tag(ax, 6.55, 3.48, "Hard DAG", fs=10.5, ha="left", color=INK)
    flow.tag(ax, 6.55, 3.22, "Layer 6 reads the prompt bar. Earlier nodes stay at 0.", fs=8, ha="left")
    earlier = hard_step["l6"][-2][0]
    hard_nodes = row(
        ax, 7.15, 1.70,
        [earlier, hard_step["cur"], "|"],
        [GRAY_F, GOLD_F, BLUE_F],
        ["#888888", GOLD, BLUE],
    )
    flow.arrow(
        ax,
        (hard_nodes[1][0], hard_nodes[1][1] + H),
        (hard_nodes[2][0], hard_nodes[2][1] + H),
        BLUE, rad=-0.28,
    )
    flow.tag(ax, hard_nodes[2][0], hard_nodes[2][1] + H + 0.22, f"L6 {hard_step['l6_bar']:.2f}", color=BLUE, fs=7.5)
    flow.tag(ax, hard_nodes[0][0], 1.28, "0.00", fs=7.5)
    flow.tag(ax, hard_nodes[1][0], 1.28, f"writes {hard_step['gold']}", fs=7.5)
    flow.tag(ax, 7.15, 0.98, "V10 and V36 are 0.00 as well", fs=7.5, ha="left")
    fig.text(
        0.03, 0.045,
        "Layered trace, third step.  V95’s parent is V14.  Mean layer 5 on the nodes already written, on a backtrack step: 0.57, 0.68, 0.76.",
        fontsize=7.2, color=MUTED, ha="left", va="bottom",
    )
    fig.text(
        0.03, 0.012,
        "Hard DAG 4×4, test index 9.  Layer 6 on the prompt bar at a no-edge step.  Mean exact next node in that case: 0.06, 0.04, 0.03.",
        fontsize=7.2, color=MUTED, ha="left", va="bottom",
    )
    save(fig, "backtrack.png")


def answer_figure(trace):
    xlim, ylim = (0, 9.3), (0.15, 3.35)
    fig_h = 3.55
    height = 2.55
    fig_w = 0.32 + height * (xlim[1] - xlim[0]) / (ylim[1] - ylim[0]) + 0.24
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = flow.put_axes(fig, xlim, ylim, 0.26, 0.58, height)
    flow.tag(ax, 0.10, 3.12, "The prompt bar holds the query", fs=10.5, ha="left", color=INK)
    flow.tag(ax, 0.10, 2.86, "Layer 3 reads the target. Layer 4 reads the source.", fs=8, ha="left")
    prompt = row(
        ax, 0.15, 1.70,
        ["<QUERY>", trace["source"], trace["target"], "|"],
        [GRAY_F, GREEN_F, BLUE_F, GOLD_F],
        ["#888888", GREEN, BLUE, GOLD],
    )
    flow.arrow(ax, (prompt[3][0], prompt[3][1] + H + 0.04), (prompt[2][0], prompt[2][1] + H + 0.04), BLUE, rad=0.28)
    flow.arrow(ax, (prompt[3][0], prompt[3][1] + H + 0.04), (prompt[1][0], prompt[1][1] + H + 0.04), GREEN, rad=0.46)
    flow.tag(ax, prompt[2][0], 1.46, f"L3 {trace['bar_l3_target']:.2f}", color=BLUE, fs=7.5)
    flow.tag(ax, prompt[1][0], 1.26, f"L4 {trace['bar_l4_source']:.2f}", color=GREEN, fs=7.5)

    flow.tag(ax, 5.55, 3.12, "Last node", fs=10.5, ha="left", color=INK)
    flow.tag(ax, 5.55, 2.86, "Layer 1 reads the node just written.", fs=8, ha="left")
    last = row(ax, 5.70, 1.70, [trace["target"]], [GOLD_F], [GOLD])
    flow.tag(ax, last[0][0], 1.46, f"L1 {trace['answer_self']:.2f}", color=GREEN, fs=7.5)
    flow.tag(ax, last[0][0], 1.26, "writes YES", fs=7.5)
    fig.text(
        0.03, 0.045,
        "Same trace.  V191 is both the last node and the query target, so the next token is YES.",
        fontsize=7.2, color=MUTED, ha="left", va="bottom",
    )
    fig.text(
        0.03, 0.012,
        "Mean P(YES) on a YES trace is 1.00, and 0.00 after the last node is replaced.  A NO trace is 0.00, and 1.00 after the target is set to that node.",
        fontsize=7.2, color=MUTED, ha="left", va="bottom",
    )
    save(fig, "answer.png")


def sink_figure():
    """V95 is copied beside its edge and kept on V14. V204 reads that token."""
    bw, bh, bg = flow.W, flow.H, flow.G
    xlim, ylim = (0, 13.55), (0.15, 4.72)
    fig_h = 5.05
    height = 3.72
    fig_w = 0.30 + height * (xlim[1] - xlim[0]) / (ylim[1] - ylim[0]) + 0.24
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = flow.put_axes(fig, xlim, ylim, 0.24, 0.78, height)
    flow.tag(ax, 0.12, 4.50, "V95 is kept on V14", fs=10.5, ha="left", color=INK)
    flow.tag(ax, 0.12, 4.24, "Layer 1 copies each child. Layer 4 at V14 reads the copies.", fs=8, ha="left")

    groups = [("V204", "0.22", BLUE_F, BLUE), ("V95", "0.24", VERM_F, VERM), ("V0", "0.22", BLUE_F, BLUE)]
    starts = [0.12, 2.62, 5.12]
    carriers = []
    for (child, mass, face, edge), x0 in zip(groups, starts):
        centers = flow.row_boxes(
            ax, x0, 3.05,
            ["<EDGE>", "V14", child],
            [GRAY_F, GREEN_F, face],
            ["#888888", GREEN, edge],
        )
        cx = centers[2][0]
        cy = 2.05
        flow.box(ax, cx - bw / 2, cy, bw, bh, "+3", face, edge)
        carriers.append((cx, cy))
        flow.arrow(ax, (cx, 3.05), (cx, cy + bh), edge)
        flow.tag(ax, cx - bw / 2 - 0.06, cy + bh * 0.72, "L1 1.00", color=edge, fs=7.2, ha="right")
        flow.tag(ax, cx + bw / 2 + 0.08, cy + bh * 0.28, f"L4 {mass}", color=edge, fs=7.2, ha="left")

    here = flow.row_boxes(ax, 4.22, 0.55, ["V14"], [GOLD_F], [GOLD])
    src = (here[0][0], here[0][1] + bh)
    for (cx, cy), rad in zip(carriers, (0.12, 0.0, -0.12)):
        flow.arrow(ax, src, (cx, cy), BLUE if cx != carriers[1][0] else VERM, rad=rad)
    flow.tag(ax, here[0][0], 0.34, "after L4    V204 8.9    V95 7.0    V0 9.0", fs=7.2)

    flow.tag(ax, 8.55, 4.50, "V204 has no outgoing edge", fs=10.5, ha="left", color=INK)
    flow.tag(ax, 8.55, 4.24, "Layer 6 reads V14. V95 is not in V204.", fs=8, ha="left")
    kept = flow.row_boxes(ax, 8.70, 2.85, ["V14", "V204"], [VERM_F, GOLD_F], [VERM, GOLD])
    flow.arrow(ax, (kept[1][0], kept[1][1] + bh), (kept[0][0], kept[0][1] + bh), VERM, rad=0.28)
    flow.tag(ax, (kept[0][0] + kept[1][0]) / 2, kept[0][1] + bh + 0.28, "L6 0.33", color=VERM, fs=7.5)
    flow.tag(ax, kept[0][0], 2.60, "children", color=VERM, fs=7.2)
    flow.tag(ax, kept[1][0], 2.60, "L1 1.00", color=GREEN, fs=7.2)
    flow.tag(ax, 8.70, 2.15, "Before layer 6, V95 scores 2.4 at V204.", fs=7.5, ha="left")
    flow.tag(ax, 8.70, 1.90, "The read writes V95 to 12.6. V0, the other", fs=7.5, ha="left")
    flow.tag(ax, 8.70, 1.66, "remaining child, is 10.8.", fs=7.5, ha="left")
    flow.tag(ax, 8.70, 1.28, "V14 alone ranks V0 first, 11.8 against 11.4.", fs=7.5, ha="left")
    flow.tag(ax, 8.70, 1.04, "Adding V204 reverses them.", fs=7.5, ha="left")
    fig.text(
        0.03, 0.048,
        "Same trace.  Hiding the V95 copy from V14 at layer 4 changes the next token from V95 to V0.  Hiding that edge from V204, at every layer, leaves V95.",
        fontsize=7.2, color=MUTED, ha="left", va="bottom",
    )
    fig.text(
        0.03, 0.016,
        "4×4 no-edge steps.  Layer 6 on the edge token is 0.00.  One child left: exact 0.98, and 0.00 after the copies are hidden.  Several left: exact 0.45.  Final scores here: V95 18.9, V0 18.1.",
        fontsize=7.2, color=MUTED, ha="left", va="bottom",
    )
    save(fig, "sink.png")


def main():
    flow.style()
    device = torch.device("cuda:0")
    layered = measure(
        ROOT / "outputs/scale_v1/launches/scale_v1_s0/context/curriculum/dfs_nodes",
        pathfig.load_test(ROOT / "data/scale_v1/context_l4w4_n16_m24_train50000_s0"),
        2, device,
    )
    hard = measure(
        ROOT / "outputs/hard_dag_v1/launches/hard_dag_v1_s0/context/curriculum/dfs_nodes",
        pathfig.load_test(pathfig.SIZES[0][1]),
        9, device,
    )
    print("layered", layered["order"], "edges", layered["edges"])
    print("hard", hard["order"])
    for item in hard["steps"]:
        print(item["cur"], item["gold"], "outs", item["outs"], "bar", round(item["l6_bar"], 3), "l6", [(n, round(m, 3)) for n, m in item["l6"]])
    edge_figure(layered)
    continue_figure(layered)
    backtrack_figure(layered, hard)
    answer_figure(layered)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""DFS-node generation, one diagram per action.

Layered 4×4, checkpoint step 185000. Numbers are from frozen forwards:
test index 2 (V209 to V191) for descend, the one-frame backtrack, and stop;
test index 6 (V135 to V246) for the three-frame backtrack.
The gold box is the position whose attention is drawn.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import plot_dfs_field_flow as flow  # noqa: E402

OUT = ROOT / "figures/scale_v1/dfs_nodes_actions"
BLUE, VERM, GREEN, GOLD = flow.BLUE, flow.VERM, flow.GREEN, flow.GOLD
BLUE_F, VERM_F, GREEN_F, GOLD_F, GRAY_F = (
    flow.BLUE_F, flow.VERM_F, flow.GREEN_F, flow.GOLD_F, flow.GRAY_F,
)
INK, MUTED = flow.INK, flow.MUTED
W, H, G = flow.W, flow.H, flow.G


def save(fig, name):
    flow.OUT = OUT
    flow.save(fig, name)


def descend():
    xlim, ylim = (0, 12.30), (0.05, 4.55)
    fig_h = 5.05
    height = 3.72
    fig_w = 0.28 + height * (xlim[1] - xlim[0]) / (ylim[1] - ylim[0]) + 0.22
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = flow.put_axes(fig, xlim, ylim, 0.24, 0.78, height)

    flow.tag(ax, 0.10, 4.32, "Descend", fs=11, ha="left", color=BLUE)
    flow.tag(ax, 0.10, 4.02, "One child left", fs=10, ha="left", color=INK)
    flow.tag(ax, 0.10, 3.74, "Layer 4 reads that copy.", fs=8, ha="left")

    left = flow.row_boxes(
        ax, 0.12, 2.55,
        ["<EDGE>", "V53", "V213"],
        [GRAY_F, GREEN_F, BLUE_F],
        ["#888888", GREEN, BLUE],
    )
    slot = (left[2][0], 1.62)
    flow.box(ax, slot[0] - W / 2, slot[1], W, H, "+3", BLUE_F, BLUE)
    flow.arrow(ax, (left[2][0], 2.55), (slot[0], slot[1] + H), BLUE)
    flow.tag(ax, slot[0] - W / 2 - 0.08, slot[1] + H * 0.55, "L1 1.00", color=BLUE, fs=7.2, ha="right")
    here = flow.row_boxes(ax, slot[0] - W / 2, 0.42, ["V53"], [GOLD_F], [GOLD])
    flow.arrow(ax, (here[0][0], here[0][1] + H), (slot[0], slot[1]), BLUE)
    flow.tag(ax, slot[0] + W / 2 + 0.08, slot[1] + 0.02, "L4 0.32", color=BLUE, fs=7.2, ha="left")
    flow.tag(ax, here[0][0], 0.22, "writes V213, 20.1", fs=7.2)

    flow.tag(ax, 4.05, 4.02, "Several children left", fs=10, ha="left", color=INK)
    flow.tag(ax, 4.05, 3.74, "Layer 4 splits across the copies.", fs=8, ha="left")
    groups = [("V14", "0.25"), ("V53", "0.24"), ("V191", "0.22")]
    starts = [4.05, 6.85, 9.65]
    carriers = []
    for (dst, mass), x0 in zip(groups, starts):
        centers = flow.row_boxes(
            ax, x0, 2.55,
            ["<EDGE>", "V209", dst],
            [GRAY_F, GREEN_F, BLUE_F],
            ["#888888", GREEN, BLUE],
        )
        cx = centers[2][0]
        cy = 1.62
        flow.box(ax, cx - W / 2, cy, W, H, "+3", BLUE_F, BLUE)
        carriers.append((cx, cy))
        flow.arrow(ax, (cx, 2.55), (cx, cy + H), BLUE)
        flow.tag(ax, cx - 0.08, cy + H + 0.18, f"L4 {mass}", color=BLUE, fs=7.0, ha="right")
    gold_x = carriers[1][0] - W / 2
    query = flow.row_boxes(ax, gold_x, 0.42, ["V209"], [GOLD_F], [GOLD])
    src = (query[0][0], query[0][1] + H)
    for (cx, cy), rad in zip(carriers, (0.14, 0.0, -0.14)):
        flow.arrow(ax, src, (cx, cy), BLUE, rad=rad)
    flow.tag(ax, carriers[1][0], 0.22, "V191 17.5     V53 16.5     V14 14.6", fs=7.2)

    fig.text(
        0.03, 0.048,
        "Descend.  Layered 4×4, test index 2, V209 to V191.  Layer 1 has already copied each destination into +3.",
        fontsize=7.2, color=MUTED, ha="left", va="bottom",
    )
    fig.text(
        0.03, 0.016,
        "One child left: exact 1.00 (74 steps).  Several left: exact 0.48 (177).  After layer 4, V209 holds V53 9.0, V191 8.3, V14 6.3.",
        fontsize=7.2, color=MUTED, ha="left", va="bottom",
    )
    save(fig, "descend.png")


def backtrack():
    xlim, ylim = (0, 12.40), (0.15, 4.85)
    fig_h = 5.35
    height = 3.85
    fig_w = 0.26 + height * (xlim[1] - xlim[0]) / (ylim[1] - ylim[0]) + 0.20
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = flow.put_axes(fig, xlim, ylim, 0.22, 0.92, height)

    flow.tag(ax, 0.12, 4.62, "Backtrack", fs=11, ha="left", color=VERM)
    flow.tag(ax, 0.12, 4.30, "One frame", fs=10, ha="left", color=INK)
    flow.tag(ax, 0.12, 4.02, "The unused child is on the parent.", fs=8, ha="left")
    nodes = flow.row_boxes(
        ax, 0.20, 2.85,
        ["V209", "V14", "V204"],
        [GREEN_F, VERM_F, GOLD_F],
        [GREEN, VERM, GOLD],
    )
    origin = (nodes[2][0], nodes[2][1] + H)
    flow.arrow(ax, origin, (nodes[1][0], nodes[1][1] + H), VERM, rad=0.22)
    flow.arrow(ax, (origin[0], origin[1] + 0.02), (nodes[0][0], nodes[0][1] + H), "#888888", rad=0.34, lw=0.9)
    flow.tag(ax, nodes[1][0], nodes[1][1] + H + 0.22, "L6 0.33", color=VERM, fs=7.2)
    flow.tag(ax, nodes[0][0], nodes[0][1] + H + 0.22, "L6 0.03", color=MUTED, fs=7.2)
    flow.tag(ax, nodes[0][0], 2.62, "L5 0.19", color=BLUE, fs=7.0)
    flow.tag(ax, nodes[1][0], 2.62, "L5 0.24", color=BLUE, fs=7.0)
    flow.tag(ax, nodes[2][0], 2.62, "L1 1.00", color=GREEN, fs=7.0)
    flow.tag(ax, nodes[1][0], 2.36, "parent", color=VERM, fs=7.0)
    flow.tag(ax, nodes[2][0], 2.36, "sink", fs=7.0)
    flow.tag(ax, 0.20, 1.95, "Before layer 6, V95 scores 2.4 at V204.", fs=7.4, ha="left")
    flow.tag(ax, 0.20, 1.68, "The read writes V95 to 12.6. V0 is 10.8.", fs=7.4, ha="left")
    flow.tag(ax, 0.20, 1.41, "Final scores: V95 18.9, V0 18.1.", fs=7.4, ha="left")
    flow.tag(ax, 0.20, 1.08, "Layer 4 at V204 is on the edge-list", fs=7.4, ha="left")
    flow.tag(ax, 0.20, 0.82, "names (0.92), not on a child copy.", fs=7.4, ha="left")

    flow.tag(ax, 6.35, 4.30, "Three frames", fs=10, ha="left", color=INK)
    flow.tag(ax, 6.35, 4.02, "Layer 6 reads all three in one step.", fs=8, ha="left")
    chain = flow.row_boxes(
        ax, 6.45, 2.85,
        ["V135", "V24", "V101", "V181"],
        [VERM_F, GRAY_F, GREEN_F, GOLD_F],
        [VERM, "#888888", GREEN, GOLD],
    )
    src = (chain[3][0], chain[3][1] + H)
    for index, rad in ((0, 0.20), (1, 0.12), (2, 0.05)):
        flow.arrow(ax, src, (chain[index][0], chain[index][1] + H), VERM, rad=rad)
    l6 = ["0.17", "0.16", "0.18", "0.08"]
    l5 = ["0.11", "0.10", "0.25", "0.12"]
    for center, mass in zip(chain, l6):
        flow.tag(ax, center[0], center[1] + H + 0.36, mass, color=VERM, fs=7.2)
    for center, mass in zip(chain, l5):
        flow.tag(ax, center[0], 2.62, mass, color=BLUE, fs=7.0)
    flow.tag(ax, chain[0][0], 2.36, "owns V246", color=VERM, fs=7.0)
    flow.tag(ax, chain[2][0], 2.36, "parent", color=BLUE, fs=7.0)
    flow.tag(ax, chain[3][0], 2.36, "this step", fs=7.0)
    flow.tag(ax, 6.45, 1.95, "Layer 5 peaks on the parent.", fs=7.4, ha="left")
    flow.tag(ax, 6.45, 1.68, "Layers 2–4 on this chain are 0.00.", fs=7.4, ha="left")
    flow.tag(ax, 6.45, 1.41, "V181 writes V246. Probability 1.00.", fs=7.4, ha="left")
    flow.tag(ax, 6.45, 1.08, "V135 still holds V246 at 12.2.", fs=7.4, ha="left")
    flow.tag(ax, 6.45, 0.82, "The nearer frames do not.", fs=7.4, ha="left")

    fig.text(
        0.03, 0.055,
        "Backtrack.  Left: test index 2, V204 to V95.  Right: test index 6, V181 to V246.  Orange numbers are layer 6.  Blue numbers are layer 5.",
        fontsize=7.2, color=MUTED, ha="left", va="bottom",
    )
    fig.text(
        0.03, 0.028,
        "4×4 no-edge steps.  One unused child: exact 0.98.  Several: exact 0.45.  Hiding the owner at layer 6 drops exact from 0.79 to 0.09.",
        fontsize=7.2, color=MUTED, ha="left", va="bottom",
    )
    fig.text(
        0.03, 0.008,
        "Three frames stays at exact 0.80 on 4×4 (15 steps).  On 10×10, four frames falls to 0.15.  The same layer 6 read is spread over the nearer frames.",
        fontsize=7.2, color=MUTED, ha="left", va="bottom",
    )
    save(fig, "backtrack.png")


def stop():
    """The last node reads the bar. The bar already holds the query."""
    xlim, ylim = (0, 7.20), (0.10, 4.35)
    fig_h = 5.05
    height = 3.42
    fig_w = 0.26 + height * (xlim[1] - xlim[0]) / (ylim[1] - ylim[0]) + 0.22
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = flow.put_axes(fig, xlim, ylim, 0.22, 1.02, height)
    flow.tag(ax, 0.12, 4.12, "Stop", fs=11, ha="left", color=GREEN)
    flow.tag(ax, 0.12, 3.82, "The last node reads the bar, not the query text.", fs=8, ha="left")
    prompt = flow.row_boxes(
        ax, 0.18, 1.72,
        ["<QUERY>", "V209", "V191", "|"],
        [GRAY_F, GREEN_F, BLUE_F, VERM_F],
        ["#888888", GREEN, BLUE, VERM],
    )
    bar = prompt[3]
    flow.arrow(ax, (bar[0], bar[1] + H + 0.03), (prompt[2][0], prompt[2][1] + H + 0.03), BLUE, rad=0.26)
    flow.arrow(ax, (bar[0], bar[1] + H + 0.03), (prompt[1][0], prompt[1][1] + H + 0.03), GREEN, rad=0.42)
    flow.tag(ax, prompt[2][0], 2.78, "L3 0.93", color=BLUE, fs=7.2)
    flow.tag(ax, prompt[1][0], 3.22, "L4 0.88", color=GREEN, fs=7.2)
    for center in prompt[:3]:
        flow.tag(ax, center[0], 1.46, "0.00", color=VERM, fs=7.2)

    last = flow.row_boxes(ax, 4.55, 1.72, ["V191"], [GOLD_F], [GOLD])
    flow.arrow(ax, (last[0][0] - W * 0.55, last[0][1] + H * 0.55), (bar[0] + W * 0.55, bar[1] + H * 0.55), VERM, rad=0.0)
    flow.tag(ax, (last[0][0] + bar[0]) / 2, 2.48, "L3 0.35", color=VERM, fs=7.2)
    flow.tag(ax, (last[0][0] + bar[0]) / 2, 2.24, "L4 0.34", color=VERM, fs=7.2)
    flow.tag(ax, last[0][0], 1.46, "L1 1.00", color=GREEN, fs=7.2)
    flow.tag(ax, last[0][0], 1.22, "writes YES, 40.2", fs=7.2)
    fig.text(
        0.03, 0.062,
        "Stop.  Same trace.  Orange is the last node.  Blue and green are the bar, which already holds the query.",
        fontsize=7.2, color=MUTED, ha="left", va="bottom",
    )
    fig.text(
        0.03, 0.034,
        "Hiding the bar at layer 3 or 4 changes YES to NO.  Hiding the query text leaves YES, 32 of 32.",
        fontsize=7.2, color=MUTED, ha="left", va="bottom",
    )
    fig.text(
        0.03, 0.008,
        "YES traces, layer 4 on the bar: 0.42.  NO traces: 0.00.  The query text stays at or below 0.02.",
        fontsize=7.2, color=MUTED, ha="left", va="bottom",
    )
    save(fig, "stop.png")


def main():
    flow.style()
    descend()
    backtrack()
    stop()


if __name__ == "__main__":
    main()

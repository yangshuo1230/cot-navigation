#!/usr/bin/env python3
"""How information moves while a DFS record is written.

In-context curriculum DFS, best checkpoint, one held-out 4×4 trace
(query 50520, source V11, target V14). Attention is float32.
The gold box is the position whose attention is drawn. A number under a
box is that position's mass on the box.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/scale_v1/dfs_flow"
BLUE, VERM, GREEN, GOLD = "#0072B2", "#D55E00", "#009E73", "#E69F00"
BLUE_F, VERM_F, GREEN_F, GOLD_F, GRAY_F = "#E6F1F8", "#FDE8DC", "#E5F5EF", "#FFF6DE", "#F3F3F3"
INK, MUTED = "#1A1A1A", "#444444"


def box(ax, x, y, w, h, text, face, edge, fs=8.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.08",
        facecolor=face, edgecolor=edge, linewidth=1.1, mutation_aspect=0.6, zorder=3,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, zorder=4)


def arrow(ax, start, end, color, rad=0.0, lw=1.15):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=9, linewidth=lw, color=color,
        connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2, zorder=2,
    ))


def tag(ax, x, y, text, color=MUTED, fs=7.5, ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color=color, zorder=5)


def frame(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def save(fig, name: str) -> None:
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    Image.open(buf).convert("RGB").save(path)
    print(path, Image.open(path).size)


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.linewidth": 0.6,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def put_axes(fig, xlim, ylim, left, bottom, height):
    width = height * (xlim[1] - xlim[0]) / (ylim[1] - ylim[0])
    ax = fig.add_axes([left / fig.get_figwidth(), bottom / fig.get_figheight(),
                       width / fig.get_figwidth(), height / fig.get_figheight()])
    frame(ax, xlim, ylim)
    return ax


W, H, G = 0.72, 0.42, 0.08


def row_boxes(ax, x0, y, names, faces, edges):
    centers = []
    x = x0
    for name, face, edge in zip(names, faces, edges):
        box(ax, x, y, W, H, name, face, edge)
        centers.append((x + W / 2, y))
        x += W + G
    return centers


def candidates():
    """First list reads shallow destination copies. A later list reads the node and the push."""
    xlim, ylim = (0, 11.55), (-0.18, 4.20)
    fig_h = 4.72
    height = 3.92
    fig_w = 0.42 + height * (xlim[1] - xlim[0]) / (ylim[1] - ylim[0]) + 0.36
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = put_axes(fig, xlim, ylim, 0.36, 0.46, height)

    tag(ax, 0.08, 3.96, "First candidate list", fs=10.5, ha="left", color=INK)
    tag(ax, 0.08, 3.70, "Layer 1 writes each destination three tokens later. Layer 5 reads those copies.", fs=8, ha="left")

    groups = [("V13", "1.00", "0.32"), ("V15", "1.00", "0.29"), ("V14", "1.00", "0.26")]
    starts = [0.08, 2.78, 5.48]
    carriers = []
    for (dest, l1, l5), x0 in zip(groups, starts):
        row_boxes(
            ax, x0, 2.58,
            ["<EDGE>", "V11", dest],
            [GRAY_F, GREEN_F, BLUE_F],
            ["#888888", GREEN, BLUE],
        )
        cx = x0 + 2 * (W + G) + W / 2
        cy = 1.48
        box(ax, cx - W / 2, cy, W, H, "+3", BLUE_F, BLUE)
        carriers.append((cx, cy))
        tag(ax, cx + 0.46, 2.10, f"L1 {l1}", color=BLUE, fs=7.5, ha="left")
        tag(ax, cx - W / 2 - 0.06, cy + H / 2, f"L5 {l5}", color=BLUE, fs=7.5, ha="right")
        arrow(ax, (cx, 2.58), (cx, cy + H), BLUE, rad=0.0)

    rec = row_boxes(
        ax, 1.96, 0.22,
        ["V11", "|", "V11", "|"],
        [GREEN_F, GRAY_F, GREEN_F, GOLD_F],
        [GREEN, "#888888", GREEN, GOLD],
    )
    receiver = (rec[3][0], rec[3][1] + H)
    rads = [0.14, 0.0, -0.16]
    for (cx, cy), rad in zip(carriers, rads):
        arrow(ax, receiver, (cx, cy), BLUE, rad=rad)
    tag(ax, rec[2][0], 0.06, "L4 0.90", color=GREEN, fs=7.5)
    arrow(
        ax,
        (rec[3][0] - W * 0.35, rec[3][1] + H + 0.02),
        (rec[2][0] + W * 0.35, rec[2][1] + H + 0.02),
        GREEN, rad=0.0,
    )

    tag(ax, 8.35, 3.96, "Later list", fs=10.5, ha="left", color=INK)
    tag(ax, 8.35, 3.70, "The edge copies are no longer the main read.", fs=8, ha="left")
    prev = row_boxes(ax, 8.35, 2.58, ["PUSH", "V13"], [GRAY_F, VERM_F], ["#888888", VERM])
    tag(ax, prev[0][0], 3.16, "L5 0.83", color=VERM, fs=7.5)
    tag(ax, prev[1][0], 3.16, "L5 0.15", color=VERM, fs=7.5)
    here = row_boxes(ax, 8.35, 0.22, ["V11", "|"], [GREEN_F, GOLD_F], [GREEN, GOLD])
    tag(ax, here[1][0] + 0.55, 0.43, "this step", fs=7.5, ha="left")
    src = (here[1][0], here[1][1] + H)
    arrow(ax, src, (prev[0][0], prev[0][1]), VERM, rad=0.10)
    arrow(ax, (src[0] + 0.04, src[1]), (prev[1][0], prev[1][1]), VERM, rad=-0.12)
    arrow(
        ax,
        (here[1][0] - W * 0.35, here[1][1] + H + 0.02),
        (here[0][0] + W * 0.35, here[0][1] + H + 0.02),
        GREEN, rad=0.0,
    )
    tag(ax, here[0][0], 0.06, "L4 0.84", color=GREEN, fs=7.5)

    fig.text(
        0.03, 0.055,
        "Query 50520, V11 to V14. Layer 1 copies the destination three tokens later (mean 1.00, 65 edges) and the source two tokens later (0.99).",
        fontsize=7.2, color=MUTED, ha="left", va="bottom",
    )
    fig.text(
        0.03, 0.018,
        "Layer 5 reads those destination copies: 0.84 on a first list (20), 0.37 later (83). Layer 4 reads the current node: 0.94, then 0.82. Layer 5 on an earlier PUSH: 0.52.",
        fontsize=7.2, color=MUTED, ha="left", va="bottom",
    )
    save(fig, "candidates.png")


def stack():
    xlim, ylim = (0, 11.15), (0.95, 3.40)
    fig_h = 3.05
    height = 2.23
    fig_w = 0.40 + height * (xlim[1] - xlim[0]) / (ylim[1] - ylim[0]) + 0.32
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = put_axes(fig, xlim, ylim, 0.32, 0.48, height)

    tag(ax, 0.10, 3.12, "Source node", fs=10.5, ha="left", color=INK)
    tag(ax, 0.10, 2.84, "The prompt bar gathers it, and later records read that bar.", fs=8, ha="left")
    nodes = row_boxes(
        ax, 0.15, 1.55,
        ["<QUERY>", "V11", "V14", "|"],
        [GRAY_F, GREEN_F, GRAY_F, BLUE_F],
        ["#888888", GREEN, "#888888", BLUE],
    )
    semi = row_boxes(ax, 4.15, 1.55, [";"], [GOLD_F], [GOLD])
    arrow(ax, (nodes[3][0], nodes[3][1] + H + 0.06), (nodes[1][0], nodes[1][1] + H + 0.06), GREEN, rad=0.18)
    arrow(ax, (nodes[3][0], nodes[3][1] + H + 0.06), (nodes[1][0], nodes[1][1] + H + 0.06), BLUE, rad=0.42)
    tag(ax, nodes[1][0], 1.32, "L1 0.92", color=GREEN, fs=7.5)
    tag(ax, nodes[1][0], 1.12, "L4 0.79", color=BLUE, fs=7.5)
    arrow(ax, (semi[0][0], semi[0][1] + H + 0.06), (nodes[3][0] + W * 0.45, nodes[3][1] + H + 0.06), BLUE, rad=0.22)
    tag(ax, 3.70, 1.32, "L5 1.00", color=BLUE, fs=7.5)
    tag(ax, semi[0][0], 1.32, "next record", fs=7.5)

    tag(ax, 6.15, 3.12, "New stack top", fs=10.5, ha="left", color=INK)
    tag(ax, 6.15, 2.84, "Copied from the PUSH argument just written.", fs=8, ha="left")
    pushed = row_boxes(ax, 6.15, 1.55, ["PUSH", "V13", ";"], [GRAY_F, VERM_F, GRAY_F], ["#888888", VERM, "#888888"])
    nxt = row_boxes(ax, 9.15, 1.55, ["V11", "V13"], [GOLD_F, VERM_F], [GOLD, VERM])
    arrow(ax, (nxt[0][0], nxt[0][1] + H + 0.06), (pushed[1][0], pushed[1][1] + H + 0.06), VERM, rad=0.20)
    tag(ax, 8.55, 1.32, "L1 0.97", color=VERM, fs=7.5)
    tag(ax, nxt[1][0], 1.32, "written next", fs=7.5)

    fig.text(
        0.03, 0.03,
        "Same trace.  Means: first stack name, layer 1 on the query source 0.85 and layer 4 again 0.77 (20).  "
        "Later records, layer 5 on the prompt bar 1.00 (180).  New top, layer 1 on the PUSH argument 0.99 (59).",
        fontsize=7.2, color=MUTED, ha="left", va="bottom",
    )
    save(fig, "stack.png")


def visited():
    xlim, ylim = (0, 11.20), (0.95, 3.55)
    fig_h = 3.20
    height = 2.36
    fig_w = 0.40 + height * (xlim[1] - xlim[0]) / (ylim[1] - ylim[0]) + 0.32
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = put_axes(fig, xlim, ylim, 0.32, 0.50, height)

    tag(ax, 0.10, 3.32, "First visited name", fs=10.5, ha="left", color=INK)
    tag(ax, 0.10, 3.04, "The bar after the stack reads the prompt bar, which holds the source.", fs=8, ha="left")
    src = row_boxes(
        ax, 0.15, 1.95,
        ["V11", "|"],
        [GREEN_F, BLUE_F],
        [GREEN, BLUE],
    )
    tag(ax, src[0][0], 2.52, "query source", fs=7.5)
    tag(ax, src[1][0], 2.52, "prompt bar", fs=7.5)
    field = row_boxes(ax, 2.55, 1.95, ["V11", "|"], [GREEN_F, GOLD_F], [GREEN, GOLD])
    tag(ax, field[0][0], 2.52, "stack", fs=7.5)
    tag(ax, field[1][0], 2.52, "writes V11", fs=7.5)
    arrow(ax, (src[1][0], src[1][1] + H), (src[0][0], src[0][1] + H), GREEN, rad=0.25)
    tag(ax, src[0][0], 1.72, "L1 0.92", color=GREEN, fs=7.5)
    tag(ax, src[1][0], 1.72, "L4 0.79", color=BLUE, fs=7.5)
    arrow(ax, (field[1][0], field[1][1] + H * 0.72), (src[1][0] + W * 0.2, src[1][1] + H * 0.72), BLUE, rad=0.18)
    tag(ax, 2.15, 2.72, "L5 0.99", color=BLUE, fs=7.5)

    tag(ax, 5.55, 3.32, "Names added later", fs=10.5, ha="left", color=INK)
    tag(ax, 5.55, 3.04, "Layer 5 stays on an earlier V11. It does not read the name being written.", fs=8, ha="left")
    wrote = row_boxes(
        ax, 5.70, 1.95,
        ["V11", "V13", "V15", "V14"],
        [GREEN_F, GRAY_F, GRAY_F, GRAY_F],
        [GREEN, "#888888", "#888888", "#888888"],
    )
    del wrote
    tag(ax, 5.70, 1.68, "While V13, V15, and V14 are written, layer 5 is on an earlier V11", fs=7.5, ha="left")
    tag(ax, 5.70, 1.46, "(0.99 each), not on the name itself.", fs=7.5, ha="left")
    tag(ax, 5.70, 1.16, "The same name in the previous record gets at most 0.03, at every layer.", fs=7.5, ha="left")

    fig.text(
        0.03, 0.03,
        "Last step of query 50520.  Mean layer 5 on the prompt bar, while writing the first visited name: 0.99 (204).  "
        "A later visited name’s earlier copy: ≤ 0.03 (832).",
        fontsize=7.2, color=MUTED, ha="left", va="bottom",
    )
    save(fig, "visited.png")


def action():
    xlim, ylim = (0, 6.05), (0.08, 4.12)
    fig_h = 4.22
    height = 3.22
    fig_w = 0.28 + height * (xlim[1] - xlim[0]) / (ylim[1] - ylim[0]) + 0.22
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = put_axes(fig, xlim, ylim, 0.22, 0.58, height)

    tag(ax, 0.08, 3.96, "Stack", fs=10.5, ha="left", color=GREEN)
    tag(ax, 0.12, 3.64, "POP", fs=10, ha="left", color=INK)
    tag(ax, 3.15, 3.64, "FOUND", fs=10, ha="left", color=INK)
    pop = row_boxes(ax, 0.12, 2.48, ["V11", "V13", "|"], [GREEN_F, VERM_F, GOLD_F], [GREEN, VERM, GOLD])
    arrow(ax, (pop[2][0], pop[2][1] + H + 0.04), (pop[1][0], pop[1][1] + H + 0.04), VERM, rad=0.26)
    tag(ax, pop[0][0], 2.24, "0.04", fs=7.5)
    tag(ax, pop[1][0], 2.24, "L4 0.83", color=VERM, fs=7.5)
    tag(ax, pop[2][0], 2.24, "writes POP", fs=7.2)
    tag(ax, 0.12, 1.98, "mean from the top  0.35, 0.33, 0.19", fs=7.2, ha="left")

    found = row_boxes(ax, 3.15, 2.48, ["V11", "V14", "|"], [GREEN_F, VERM_F, GOLD_F], [GREEN, VERM, GOLD])
    arrow(ax, (found[2][0], found[2][1] + H + 0.04), (found[1][0], found[1][1] + H + 0.04), VERM, rad=0.26)
    tag(ax, found[1][0], 2.24, "L4 0.97", color=VERM, fs=7.5)
    tag(ax, found[2][0], 2.24, "writes FOUND", fs=7.2)
    tag(ax, 3.15, 1.98, "mean on the top  0.85", fs=7.2, ha="left")

    tag(ax, 0.08, 1.64, "Candidates", fs=10.5, ha="left", color=BLUE)
    tag(ax, 0.12, 1.36, "one name, mean 0.96, 0.98", fs=8, ha="left", color=INK)
    tag(ax, 2.55, 1.64, "three names, writes V13", fs=8, ha="left", color=INK)
    one = row_boxes(ax, 0.12, 0.55, ["V14", "PUSH"], [BLUE_F, GOLD_F], [BLUE, GOLD])
    arrow(ax, (one[1][0], one[1][1] + H + 0.03), (one[0][0], one[0][1] + H + 0.03), VERM, rad=0.22)
    tag(ax, one[0][0], 0.32, "L1 0.98", color=VERM, fs=7.5)

    three = row_boxes(
        ax, 2.55, 0.55,
        ["V13", "V15", "V14", "PUSH"],
        [BLUE_F, BLUE_F, BLUE_F, GOLD_F],
        [BLUE, BLUE, BLUE, GOLD],
    )
    arrow(ax, (three[3][0], three[3][1] + H + 0.03), (three[1][0], three[1][1] + H + 0.03), VERM, rad=0.14)
    arrow(ax, (three[3][0], three[3][1] + H + 0.03), (three[0][0], three[0][1] + H + 0.03), "#56B4E9", rad=0.20)
    tag(ax, three[1][0], 0.32, "L1 0.95", color=VERM, fs=7.5)
    tag(ax, three[0][0], 0.32, "L4 0.36", color=BLUE, fs=7.5)

    fig.text(
        0.03, 0.045,
        "Query 50520. Green nodes are the stack. Blue nodes are the candidate list.",
        fontsize=7.2, color=MUTED, ha="left", va="bottom",
    )
    fig.text(
        0.03, 0.012,
        "Three names: layer 1 on the second is 0.98, layer 4 on the first is 0.44 (21).",
        fontsize=7.2, color=MUTED, ha="left", va="bottom",
    )
    save(fig, "action.png")


def push():
    """The PUSH token gathers its candidate list. A later list reads that token."""
    xlim, ylim = (0, 8.55), (0.20, 4.25)
    fig_h = 4.55
    height = 3.55
    fig_w = 0.36 + height * (xlim[1] - xlim[0]) / (ylim[1] - ylim[0]) + 0.28
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = put_axes(fig, xlim, ylim, 0.30, 0.58, height)

    tag(ax, 0.12, 4.02, "PUSH reads the candidate list", fs=10.5, ha="left", color=INK)
    tag(ax, 0.12, 3.74, "Three layers land on three different names.", fs=8, ha="left")
    cands = row_boxes(
        ax, 0.20, 2.35,
        ["V13", "V15", "V14"],
        [VERM_F, BLUE_F, GREEN_F],
        [VERM, BLUE, GREEN],
    )
    tag(ax, cands[0][0], 2.95, "L4 0.36", color=VERM, fs=7.5)
    tag(ax, cands[1][0], 2.95, "L1 0.95", color=BLUE, fs=7.5)
    tag(ax, cands[2][0], 2.95, "L5 0.99", color=GREEN, fs=7.5)
    receiver = row_boxes(ax, cands[1][0] - W / 2, 0.72, ["PUSH"], [GOLD_F], [GOLD])
    src = (receiver[0][0], receiver[0][1] + H)
    arrow(ax, src, (cands[1][0], cands[1][1]), BLUE, rad=0.0)
    arrow(ax, src, (cands[0][0], cands[0][1]), VERM, rad=0.12)
    arrow(ax, src, (cands[2][0], cands[2][1]), GREEN, rad=-0.12)
    tag(ax, receiver[0][0], 0.48, "writes V13", fs=7.5)

    tag(ax, 4.55, 4.02, "A later list reads it", fs=10.5, ha="left", color=INK)
    tag(ax, 4.55, 3.74, "The bar reads this PUSH, not the names.", fs=8, ha="left")
    prev = row_boxes(ax, 4.70, 2.35, ["PUSH", "V13"], [GRAY_F, VERM_F], ["#888888", VERM])
    tag(ax, prev[0][0], 2.95, "L5 0.83", color=VERM, fs=7.5)
    tag(ax, prev[1][0], 2.95, "L5 0.15", color=VERM, fs=7.5)
    here = row_boxes(ax, 4.70, 0.72, ["V11", "|"], [GREEN_F, GOLD_F], [GREEN, GOLD])
    origin = (here[1][0], here[1][1] + H)
    arrow(ax, origin, (prev[0][0], prev[0][1]), VERM, rad=0.10)
    arrow(ax, (origin[0] + 0.04, origin[1]), (prev[1][0], prev[1][1]), VERM, rad=-0.12)
    tag(ax, here[1][0] + 0.58, 0.93, "this step", fs=7.5, ha="left")

    fig.text(
        0.03, 0.055,
        "Query 50520. At this PUSH, layer 1 is on the candidate list (mean 1.00, 94 later lists). The name written is 0.39 of that.",
        fontsize=7.2, color=MUTED, ha="left", va="bottom",
    )
    fig.text(
        0.03, 0.018,
        "Layer 4 stays on the list (0.68). When a later list is written, layer 5 reads this PUSH and its argument (mean 0.52).",
        fontsize=7.2, color=MUTED, ha="left", va="bottom",
    )
    save(fig, "push.png")


def stack_middle():
    """A kept stack name is the second-from-top copied into that record's first visited token."""
    xlim, ylim = (0, 7.15), (0.15, 5.15)
    fig_h = 4.85
    height = 3.85
    fig_w = 0.32 + height * (xlim[1] - xlim[0]) / (ylim[1] - ylim[0]) + 0.28
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = put_axes(fig, xlim, ylim, 0.28, 0.62, height)

    def band(y, names, faces, edges, target, l1, writer, l5, title, note):
        tag(ax, 0.12, y + 1.55, title, fs=10.5, ha="left", color=INK)
        tag(ax, 0.12, y + 1.28, note, fs=8, ha="left")
        nodes = row_boxes(ax, 0.15, y, names, faces, edges)
        vis = nodes[-1]
        goal = nodes[target]
        arrow(ax, (vis[0], vis[1] + H + 0.05), (goal[0], goal[1] + H + 0.05), BLUE, rad=0.16)
        tag(ax, goal[0], y - 0.24, f"L1 {l1}", color=BLUE, fs=7.5)
        tag(ax, vis[0], y - 0.24, "visited", fs=7.5)
        gold = row_boxes(ax, vis[0] + 0.85, y, [writer], [GOLD_F], [GOLD])[0]
        arrow(ax, (gold[0], gold[1] + H + 0.05), (vis[0] + W * 0.35, vis[1] + H + 0.05), VERM, rad=0.22)
        tag(ax, (vis[0] + gold[0]) / 2, y + H + 0.62, f"L5 {l5}", color=VERM, fs=7.5)
        tag(ax, gold[0], y - 0.24, "writes " + writer, fs=7.5)

    gray = ("#888888", GRAY_F)
    band(
        3.05,
        ["V2", "V5", "V9", "|", "V2"],
        [GREEN_F, VERM_F, GRAY_F, GRAY_F, BLUE_F],
        [GREEN, VERM, gray[0], gray[0], BLUE],
        1, "0.99", "V5", "0.99",
        "Writing V5",
        "Record 2 stores its second-from-top in the first visited name.",
    )
    band(
        0.70,
        ["V2", "V5", "V9", "V15", "|", "V2"],
        [GREEN_F, GRAY_F, VERM_F, GRAY_F, GRAY_F, BLUE_F],
        [GREEN, gray[0], VERM, gray[0], gray[0], BLUE],
        2, "1.00", "V9", "0.99",
        "Writing V9",
        "Record 3 stores a different second-from-top. Layer 5 picks that one.",
    )
    fig.text(
        0.03, 0.055,
        "Query 50512, after POP. The first visited name copies the second-from-top at layer 1 (mean 0.99, 458).",
        fontsize=7.2, color=MUTED, ha="left", va="bottom",
    )
    fig.text(
        0.03, 0.018,
        "Layer 5 then selects the visited name whose copy is the stack node being written (257 of 257).",
        fontsize=7.2, color=MUTED, ha="left", va="bottom",
    )
    save(fig, "stack_middle.png")


def main():
    style()
    candidates()
    stack()
    visited()
    action()
    push()
    stack_middle()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""PATH successor choice, drawn in the same two-panel layout as the direct shortcut figures.

Left of the first figure: one 4×4 step where the target is a successor, and one
where it is not. Layer-6 mass on the outgoing edges stays split. The next-token
probability peaks only in the first case. Right: that contrast on 4×4, 6×6 and
10×10, as a ratio to the uniform share of the successor set.

Second figure: free-generation accuracy of the same best checkpoint. No-path
queries are answered correctly; path-exists queries are not.
"""
from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/hard_dag_v1"
CHOICE = ROOT / "studies/block_membership_probe/path_successor_choice.json"
EVAL = ROOT / "outputs/hard_dag_v1/launches/hard_dag_v1_s0/context/curriculum/path/eval"
BLUE, VERM, GREEN, GOLD = "#0072B2", "#D55E00", "#009E73", "#E69F00"
BLUE_F, VERM_F, GREEN_F, GRAY_F = "#E6F1F8", "#FDE8DC", "#E5F5EF", "#F3F3F3"


def box(ax, x, y, w, h, text, face, edge, fs=8.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.08",
        facecolor=face, edgecolor=edge, linewidth=1.1, mutation_aspect=0.6, zorder=3,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="#1A1A1A", zorder=4)


def arrow(ax, start, end, color, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=9, linewidth=1.15, color=color,
        connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2, zorder=2,
    ))


def tag(ax, x, y, text, color, ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=7.4, color=color, zorder=5,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.15})


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
    })


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    buf.seek(0)
    Image.open(buf).convert("RGB").save(path)
    print(path)


def axes_for(fig_w, fig_h, left, bottom, width, height):
    return [left / fig_w, bottom / fig_h, width / fig_w, height / fig_h]


def _case(ax, y, title, current, successors):
    """successors: (name, face, edge, attn, prob, note)."""
    ax.text(0.08, y + 1.38, title, fontsize=8.0, color="#1A1A1A", ha="left", va="bottom")
    w, h = 1.16, 0.46
    xs = [0.15, 2.35, 4.70]
    box(ax, xs[0], y, w, h, current, BLUE_F, BLUE, fs=8)
    for x, (name, face, edge, attn, prob, note) in zip(xs[1:], successors):
        box(ax, x, y, w, h, name, face, edge, fs=8)
        ax.text(x + w / 2, y - 0.12, f"L6  {attn:.2f}", ha="center", va="top", fontsize=7.2, color=edge)
        ax.text(x + w / 2, y - 0.32, f"{prob:.2f}", ha="center", va="top", fontsize=7.4, color=edge)
        ax.text(x + w / 2, y - 0.50, note, ha="center", va="top", fontsize=6.5, color="#666666")
    src = (xs[0] + w / 2, y + h)
    arrow(ax, src, (xs[1] + w * 0.22, y + h), successors[0][2], rad=-0.18)
    arrow(ax, (src[0] + 0.05, src[1]), (xs[2] + w * 0.40, y + h), successors[1][2], rad=-0.28)


def attention_panel(ax):
    ax.set_xlim(0, 6.15)
    ax.set_ylim(0.08, 4.95)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_title("(a)  Attention at the path step", loc="left", fontsize=11, color="#1A1A1A", pad=2)
    _case(
        ax, 3.05, "Target is a successor   ·   query 3",
        "V29",
        [
            ("V62", VERM_F, VERM, 0.56, 0.92, "target"),
            ("V127", GRAY_F, "#8A8A8A", 0.44, 0.08, "other"),
        ],
    )
    _case(
        ax, 0.58, "Otherwise   ·   query 4",
        "V135",
        [
            ("V215", GREEN_F, GREEN, 0.55, 0.47, "reaches V51"),
            ("V204", GRAY_F, "#8A8A8A", 0.45, 0.53, "dead end"),
        ],
    )


def lift_panel(ax, payload):
    sizes = [block["size"] for block in payload]
    x = np.arange(len(sizes))
    series = (
        ("P(is the target)", [block["yes"]["choice"]["target"]["lift"] for block in payload], VERM, -0.26),
        ("P(reaches target)", [block["yes"]["choice"]["reach"]["lift"] for block in payload], GREEN, 0.0),
        ("Attention on target", [block["yes"]["attention"]["target"]["lift"] for block in payload], BLUE, 0.26),
    )
    for label, values, color, offset in series:
        ax.bar(x + offset, values, width=0.24, color=color, label=label, zorder=3)
        for xpos, value in zip(x + offset, values):
            ax.text(xpos, value + 0.04, f"{value:.2f}", ha="center", va="bottom", fontsize=6.4, color=color)
    ax.axhline(1.0, color="#C8C8C8", lw=0.8, ls=(0, (3, 2)), zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_ylim(0, 2.65)
    ax.set_yticks([0, 0.5, 1, 1.5, 2, 2.5])
    ax.yaxis.grid(True, color="#EEEEEE", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylabel("Share / uniform share", fontsize=8.5)
    ax.set_xlabel("Graph size", fontsize=8.5)
    ax.set_title("(b)  Only the target is selected", loc="left", fontsize=11, pad=2)
    ax.legend(frameon=False, fontsize=6.8, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, borderaxespad=0)


def draw_choice(payload):
    style()
    fig_w, fig_h = 12.9, 4.35
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle("PATH successor choice", x=0.01, y=0.972, ha="left", fontsize=13, color="#1A1A1A")
    # Data aspect 6.15 / 4.87 = 1.263. Axes 4.42 x 3.50 inches.
    attention_panel(fig.add_axes(axes_for(fig_w, fig_h, 0.05, 0.16, 4.42, 3.50)))
    lift_panel(fig.add_axes(axes_for(fig_w, fig_h, 5.55, 0.62, 7.05, 2.85)), payload)
    save(fig, OUT / "path_successor_selectivity.png")


def _recall_bars(ax, labels, yes, no, title, xlabel):
    x = np.arange(len(labels))
    ax.bar(x - 0.17, yes, width=0.34, color=BLUE, label="Path exists", zorder=3)
    ax.bar(x + 0.17, no, width=0.34, color=VERM, label="No path", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.4)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.axhline(0.5, color="#C8C8C8", lw=0.8, ls=(0, (3, 2)), zorder=0)
    ax.yaxis.grid(True, color="#EEEEEE", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylabel("Fraction correct", fontsize=8.5)
    ax.set_xlabel(xlabel, fontsize=8.5)
    ax.set_title(title, loc="left", fontsize=11, pad=2)
    ax.legend(frameon=False, fontsize=7.2, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, borderaxespad=0)
    for xpos, value, color in list(zip(x - 0.17, yes, [BLUE] * len(yes))) + list(zip(x + 0.17, no, [VERM] * len(no))):
        if value >= 0.90:
            ax.text(xpos, value - 0.04, f"{value:.2f}", ha="center", va="top", fontsize=6.2, color="white")
        else:
            ax.text(xpos, value + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=6.2, color=color)


def draw_answers():
    style()
    files = [
        ("4×4", "free_test_best-n16.json"),
        ("6×6", "free_test_best-n36.json"),
        ("8×8", "free_test_best-n64.json"),
        ("10×10", "free_test_best.json"),
    ]
    labels, yes, no = [], [], []
    for name, filename in files:
        overall = json.loads((EVAL / filename).read_text())["overall"]
        labels.append(f"{name}\n{overall['n_yes']}/{overall['n_no']}")
        yes.append(overall["yes_recall"])
        no.append(overall["no_recall"])
    distance = json.loads((EVAL / "free_test_best.json").read_text())["by_distance"]
    hops = sorted(distance, key=lambda key: int(key))
    d_labels, d_yes, d_no = [], [], []
    for hop in hops:
        row = distance[hop]
        d_labels.append(f"{hop}\n{row['n_yes']}/{row['n_no']}")
        d_yes.append(row["yes_recall"])
        d_no.append(row["no_recall"])

    fig_w, fig_h = 12.4, 3.85
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle("PATH free generation, best checkpoint", x=0.012, y=0.975, ha="left", fontsize=13, color="#1A1A1A")
    _recall_bars(
        fig.add_axes(axes_for(fig_w, fig_h, 0.62, 0.62, 4.55, 2.55)),
        labels, yes, no, "(a)  By graph size", "Graph size\npositive / negative",
    )
    _recall_bars(
        fig.add_axes(axes_for(fig_w, fig_h, 6.55, 0.62, 5.55, 2.55)),
        d_labels, d_yes, d_no, "(b)  10×10 by distance", "Distance\npositive / negative",
    )
    save(fig, OUT / "path_answer_pos_neg.png")


def main():
    payload = json.loads(CHOICE.read_text())
    draw_choice(payload)
    draw_answers()


if __name__ == "__main__":
    main()

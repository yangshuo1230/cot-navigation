#!/usr/bin/env python3
"""Certified route-blocking NO rates for the scale state models.

The original graph label stays YES. The goal is not inserted into the visited
field. Clean, blocked, and the equal-length duplicate control use the same cases.
"""
from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "research/scale_mechanistic_20261019/results/models"
OUT = ROOT / "figures/scale_v1/route_blocking.png"
SCALES = ["l4", "l6", "l8", "l10", "curriculum"]
TICKS = ["N16", "N36", "N64", "N100", "Curriculum"]
PANELS = [
    ("In-context DFS", "context", "dfs_state"),
    ("In-context BFS", "context", "bfs_state"),
    ("Fixed DFS", "fixed", "dfs_state"),
    ("Fixed BFS", "fixed", "bfs_state"),
]


def rates(kind: str, protocol: str, condition: str) -> list[float]:
    values = []
    for scale in SCALES:
        report = json.loads((MODELS / f"{kind}_{scale}_{protocol}.json").read_text())
        row = report.get("blocking", {}).get("excluding_goal", {}).get("conditions", {}).get(condition, {})
        rate = row.get("counterfactual_NO_rate")
        values.append(np.nan if rate is None else float(rate))
    return values


def main() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.2,
        "ytick.major.size": 3.2,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.6), sharex=True, sharey=True)
    xs = np.arange(len(TICKS))
    for ax, (title, kind, protocol) in zip(axes.ravel(), PANELS):
        ax.plot(xs, rates(kind, protocol, "clean"), color="#7A7A7A", marker="o", linestyle="--",
                linewidth=1.4, markersize=4.5, label="Clean", zorder=2)
        ax.plot(xs, rates(kind, protocol, "duplicate_padding"), color="#D55E00", marker="s",
                linestyle="--", linewidth=1.6, markersize=5, markerfacecolor="white",
                markeredgecolor="#D55E00", markeredgewidth=1.0, label="Control", zorder=3)
        ax.plot(xs, rates(kind, protocol, "all_routes"), color="#0072B2", marker="o", linestyle="-",
                linewidth=1.6, markersize=5, label="Blocked", zorder=4)
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlim(-0.25, len(TICKS) - 0.75)
        ax.set_xticks(xs, TICKS, fontsize=8)
        ax.yaxis.grid(True, color="#EEEEEE", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.set_title(title, fontsize=11, pad=6)
        handles, labels = ax.get_legend_handles_labels()
        order = [labels.index(name) for name in ("Clean", "Blocked", "Control")]
        ax.legend([handles[i] for i in order], [labels[i] for i in order],
                  frameon=True, fontsize=8, loc="best", handlelength=2.2, borderaxespad=0.3,
                  framealpha=0.94, edgecolor="none", fancybox=False)
    axes[0, 0].set_ylabel("NO rate", fontsize=10)
    axes[1, 0].set_ylabel("NO rate", fontsize=10)
    fig.suptitle("Route blocking, goal excluded", fontsize=13)
    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=200, facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUT, format="PNG", optimize=True)
    print(OUT)


if __name__ == "__main__":
    main()

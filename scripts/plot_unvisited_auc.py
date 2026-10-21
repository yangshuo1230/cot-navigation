#!/usr/bin/env python3
"""Unvisited-successor AUC by training scale.

Each cell is the L6 probe AUC and the input-feature control. Rows are N16,
N36, N64, N100, and curriculum through N100.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/scale_v1/unvisited_successor_auc.png"

# (probe, input) for N16, N36, N64, N100, curriculum.
PANELS = [
    ("In-context DFS", [(0.551, 0.520), (0.808, 0.443), (0.677, 0.484), (0.594, 0.482), (0.746, 0.523)]),
    ("In-context BFS", [(0.589, 0.497), (0.833, 0.501), (0.615, 0.511), (0.535, 0.459), (0.656, 0.417)]),
    ("Fixed DFS", [(1.000, 0.571), (0.969, 0.827), (0.948, 0.590), (0.901, 0.699), (0.816, 0.692)]),
    ("Fixed BFS", [(1.000, 0.455), (0.965, 0.641), (0.974, 0.651), (0.958, 0.562), (0.950, 0.570)]),
]
TICKS = ["N16", "N36", "N64", "N100", "Curriculum"]


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
    for ax, (title, rows) in zip(axes.ravel(), PANELS):
        probe = [item[0] for item in rows]
        control = [item[1] for item in rows]
        ax.axhline(0.5, color="#9A9A9A", lw=0.8, ls="--", zorder=0)
        ax.plot(xs, probe, color="#0072B2", marker="o", linestyle="-", linewidth=1.6,
                markersize=5, label="Probe", zorder=3)
        ax.plot(xs, control, color="#D55E00", marker="s", linestyle="--", linewidth=1.6,
                markersize=5, markerfacecolor="white", markeredgecolor="#D55E00",
                markeredgewidth=1.0, label="Control", zorder=3)
        ax.set_ylim(0.35, 1.05)
        ax.set_xlim(-0.25, len(TICKS) - 0.75)
        ax.set_xticks(xs, TICKS, fontsize=8)
        ax.yaxis.grid(True, color="#EEEEEE", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.set_title(title, fontsize=11, pad=6)
        ax.legend(frameon=True, fontsize=8, loc="best", handlelength=2.2, borderaxespad=0.3,
                  framealpha=0.94, edgecolor="none", fancybox=False)
    axes[0, 0].set_ylabel("AUC", fontsize=10)
    axes[1, 0].set_ylabel("AUC", fontsize=10)
    fig.suptitle("Unvisited-successor AUC", fontsize=13)
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

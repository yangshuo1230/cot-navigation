#!/usr/bin/env python3
"""Forward-test accuracy for the fixed 10x10 short-hop run.

Two protocols, direct and path. Each figure compares training on forward
queries of at most 3 hops with the matched all-hop run.
"""
from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/fixed_short3_v1/launches/short3_n100_s0"
OUT = ROOT / "figures/fixed_short3"
ARMS = [
    ("short3", "Train ≤3 hops", "#0072B2", "o", "-"),
    ("all_hops", "Train all hops", "#D55E00", "s", "--"),
]
PROTOCOLS = [("direct", "Direct"), ("path", "Path")]


def hops(path: Path) -> dict[int, float]:
    report = json.loads(path.read_text())
    return {int(key): float(row["answer_accuracy"]) for key, row in report["by_distance"].items()}


def draw(protocol: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.7))
    ticks = set()
    for arm, label, color, marker, linestyle in ARMS:
        values = hops(RUN / arm / protocol / "eval/free_test_best.json")
        xs = np.asarray(sorted(values), dtype=float)
        ys = np.asarray([values[int(hop)] for hop in xs], dtype=float)
        ticks.update(int(hop) for hop in xs)
        ax.plot(
            xs, ys, color=color, marker=marker, linestyle=linestyle,
            linewidth=1.6, markersize=5.5, markerfacecolor="white" if linestyle == "--" else color,
            markeredgecolor=color, markeredgewidth=1.0, label=label, zorder=3,
        )
    ax.axvline(3.5, color="#E0E0E0", lw=0.8, zorder=0)
    ax.axhline(0.5, color="#B3B3B3", lw=0.8, ls=(0, (3, 2)), zorder=0)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(sorted(ticks))
    ax.set_xlim(min(ticks) - 0.35, max(ticks) + 0.35)
    ax.yaxis.grid(True, color="#EEEEEE", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.set_title(f"{title}, fixed 10×10", fontsize=12, pad=8)
    ax.set_xlabel("Query hops", fontsize=10)
    ax.set_ylabel("Test accuracy", fontsize=10)
    ax.legend(frameon=True, fontsize=8, loc="best", handlelength=2.2, borderaxespad=0.3,
              framealpha=0.94, edgecolor="none", fancybox=False)
    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=200, facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB")
    path = OUT / f"{protocol}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)
    print(path)


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
    for protocol, title in PROTOCOLS:
        draw(protocol, title)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""In-context curriculum accuracy and complete-trace legality by hop.

Hops 1-9 come from the 10x10 test. Hops 10-19 come from the 20x5 deep test.
Both series are the curriculum run's final checkpoint, the same rollout as
the trace audit.
"""
from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CONTEXT_AUDIT = ROOT / "research/scale_trace_audit_20261019/results/context/curriculum"
FIXED_AUDIT = ROOT / "research/scale_trace_audit_20261019/results/fixed/curriculum/stage_b"
OUT = ROOT / "figures/scale_v1/curriculum_hops"
PROTOCOLS = [
    ("path", "Path"),
    ("dfs_state", "DFS state"),
    ("dfs_nodes", "DFS nodes"),
    ("bfs_state", "BFS state"),
    ("bfs_nodes", "BFS nodes"),
]


def rates(directory: Path, protocol: str, case: str) -> dict[int, tuple[float, float]]:
    report = json.loads((directory / protocol / f"{case}.json").read_text())
    out = {}
    for key, row in report["by_distance"].items():
        n = int(row["n"])
        if n == 0:
            continue
        out[int(key)] = (row["answer_correct"] / n, row["complete_valid"] / n)
    return out


def draw(protocol: str, title: str) -> None:
    main = rates(CONTEXT_AUDIT, protocol, "main")
    deep = rates(CONTEXT_AUDIT, protocol, "deep")
    hops = list(range(1, 20))
    accuracy, legal = [], []
    for hop in hops:
        source = main if hop <= 9 else deep
        if hop not in source:
            accuracy.append(np.nan)
            legal.append(np.nan)
            continue
        accuracy.append(source[hop][0])
        legal.append(source[hop][1])
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    xs = np.asarray(hops, dtype=float)
    ax.plot(xs, accuracy, color="#0072B2", marker="o", linestyle="-", linewidth=1.6,
            markersize=5, label="Accuracy", zorder=3)
    ax.plot(xs, legal, color="#D55E00", marker="s", linestyle="--", linewidth=1.6,
            markersize=5, markerfacecolor="white", markeredgecolor="#D55E00",
            markeredgewidth=1.0, label="CoT legal", zorder=3)
    ax.axvline(9.5, color="#E0E0E0", lw=0.8, zorder=0)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0.6, 19.4)
    ax.set_xticks(hops)
    ax.yaxis.grid(True, color="#EEEEEE", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.set_title(f"{title}, in-context curriculum", fontsize=12, pad=8)
    ax.set_xlabel("Query hops    1–9: 10×10     10–19: 20×5", fontsize=10)
    ax.set_ylabel("Rate", fontsize=10)
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


def draw_fixed(protocol: str, title: str) -> None:
    main = rates(FIXED_AUDIT, protocol, "main")
    hops = sorted(main)
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    xs = np.asarray(hops, dtype=float)
    ax.plot(xs, [main[hop][0] for hop in hops], color="#0072B2", marker="o", linestyle="-",
            linewidth=1.6, markersize=5, label="Accuracy", zorder=3)
    ax.plot(xs, [main[hop][1] for hop in hops], color="#D55E00", marker="s", linestyle="--",
            linewidth=1.6, markersize=5, markerfacecolor="white", markeredgecolor="#D55E00",
            markeredgewidth=1.0, label="CoT legal", zorder=3)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(min(hops) - 0.4, max(hops) + 0.4)
    ax.set_xticks(hops)
    ax.yaxis.grid(True, color="#EEEEEE", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.set_title(f"{title}, fixed curriculum", fontsize=12, pad=8)
    ax.set_xlabel("Query hops, fixed 10×10", fontsize=10)
    ax.set_ylabel("Rate", fontsize=10)
    ax.legend(frameon=True, fontsize=8, loc="best", handlelength=2.2, borderaxespad=0.3,
              framealpha=0.94, edgecolor="none", fancybox=False)
    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=200, facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB")
    path = OUT / "fixed" / f"{protocol}.png"
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
        draw_fixed(protocol, title)


if __name__ == "__main__":
    main()

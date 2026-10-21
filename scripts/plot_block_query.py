#!/usr/bin/env python3
"""Fixed 10x10 block-query holdout, one figure per protocol.

Training queries are forward gaps 1-4. within_only keeps every query inside
one 5-layer block. cross_allowed uses the same counts and may cross blocks.
"""
from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/block_query_v1/launches/block_n100_s0"
OUT = ROOT / "figures/block_query"
PROTOCOLS = [
    ("direct", "Direct"),
    ("path", "Path"),
    ("dfs_state", "DFS state"),
    ("dfs_nodes", "DFS nodes"),
    ("bfs_state", "BFS state"),
    ("bfs_nodes", "BFS nodes"),
]
SERIES = [
    ("within_only", "within_short", "Within train, in-block", "#0072B2", "o", "-"),
    ("within_only", "cross", "Within train, cross-block", "#0072B2", "o", "--"),
    ("cross_allowed", "within_short", "Cross train, in-block", "#D55E00", "s", "-"),
    ("cross_allowed", "cross", "Cross train, cross-block", "#D55E00", "s", "--"),
]


def job_dir(arm: str, protocol: str) -> Path:
    return RUN / arm if protocol == "direct" else RUN / arm / protocol


def by_hop(path: Path) -> dict[int, float]:
    report = json.loads(path.read_text())
    values = {}
    for key, row in report["by_distance"].items():
        score = row["balanced_accuracy"]
        if score is None:
            continue
        values[int(key)] = float(score)
    return values


def series_hops(arm: str, protocol: str, region: str) -> dict[int, float]:
    folder = job_dir(arm, protocol) / "eval"
    if region == "within_short":
        return by_hop(folder / "free_test_best-within_short.json")
    short = by_hop(folder / "free_test_best-cross_short.json")
    long = by_hop(folder / "free_test_best-cross_long.json")
    overlap = set(short) & set(long)
    if overlap:
        raise ValueError(f"{arm}/{protocol} cross views share hops {sorted(overlap)}")
    return {**short, **long}


def draw(protocol: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    ticks = set()
    for arm, region, label, color, marker, linestyle in SERIES:
        values = series_hops(arm, protocol, region)
        xs = np.asarray(sorted(values), dtype=float)
        ys = np.asarray([values[int(hop)] for hop in xs], dtype=float)
        ticks.update(int(hop) for hop in xs)
        ax.plot(
            xs, ys, color=color, marker=marker, linestyle=linestyle,
            linewidth=1.6, markersize=5, markerfacecolor="white" if linestyle == "--" else color,
            markeredgecolor=color, markeredgewidth=1.0, label=label, zorder=3,
        )
    ax.axvline(4.5, color="#E0E0E0", lw=0.8, zorder=0)
    ax.axhline(0.5, color="#B3B3B3", lw=0.8, ls=(0, (3, 2)), zorder=0)
    ax.set_ylim(0, 1.05)
    ordered = sorted(ticks)
    ax.set_xticks(ordered)
    ax.set_xlim(min(ordered) - 0.35, max(ordered) + 0.35)
    ax.yaxis.grid(True, color="#EEEEEE", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.set_title(f"{title}, fixed 10×10", fontsize=12, pad=8)
    ax.set_xlabel("Query hops", fontsize=10)
    ax.set_ylabel("Balanced accuracy", fontsize=10)
    ax.legend(frameon=True, fontsize=7.5, loc="best", handlelength=2.4, borderaxespad=0.3,
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

#!/usr/bin/env python3
"""Eight separate accuracy plots. The scale-v1 run is not included.

In-context lines use the same conditions as the fixed-DAG plots. Shuffled DFS
was only trained on fixed graphs, so that line is absent from the in-context
DFS figure.
"""
from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EXPLORE = ROOT / "outputs/explore_v1/launches/explore_v1_s012"
OUT = ROOT / "figures/existing_results"

STYLE = {
    "Baseline": ("#0072B2", "s", "-"),
    "Curriculum": ("#CC79A7", "D", "-"),
    "One-hop": ("#E69F00", "^", "--"),
    "Negative path": ("#D55E00", "v", "-"),
    "Full state": ("#0072B2", "o", "-"),
    "Shuffled": ("#D55E00", "s", "-"),
    "Nodes": ("#009E73", "^", "-"),
}


def hops(path: Path) -> dict[int, float]:
    report = json.loads(path.read_text())
    return {int(k): float(v["answer_accuracy"]) for k, v in report["by_distance"].items()}


def mean_sd(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    series = [hops(path) for path in paths]
    keys = sorted(set.intersection(*(set(item) for item in series)))
    mu, sd = [], []
    for hop in keys:
        vals = np.array([item[hop] for item in series], dtype=float)
        mu.append(float(vals.mean()))
        sd.append(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)
    return np.asarray(keys), np.asarray(mu), np.asarray(sd)


def fixed(rel: str) -> list[Path]:
    return [EXPLORE / f"fixed/standard/seed{seed}" / rel / "eval/free_test_best.json" for seed in range(3)]


def context(rel: str) -> list[Path]:
    return [EXPLORE / "context/standard/train50000" / rel / "eval/free_test_best.json"]


def panels() -> list[tuple[str, str, str, list[tuple[str, list[Path]]]]]:
    return [
        ("fixed_direct", "Direct", "Fixed DAG", [
            ("Baseline", fixed("baseline/direct")),
            ("Curriculum", fixed("direct_curriculum/direct")),
            ("One-hop", fixed("direct_onehop/direct")),
        ]),
        ("fixed_path", "Path", "Fixed DAG", [
            ("Baseline", fixed("baseline/path")),
            ("Negative path", fixed("path_negative/path")),
        ]),
        ("fixed_dfs", "DFS", "Fixed DAG", [
            ("Full state", fixed("baseline/dfs_state")),
            ("Shuffled", fixed("dfs_random/dfs_state")),
            ("Nodes", fixed("dfs_nodes/dfs_nodes")),
        ]),
        ("fixed_bfs", "BFS", "Fixed DAG", [
            ("Full state", fixed("bfs_state/bfs_state")),
            ("Nodes", fixed("bfs_nodes/bfs_nodes")),
        ]),
        ("incontext_direct", "Direct", "In-context DAG", [
            ("Baseline", context("baseline/direct")),
            ("Curriculum", context("direct_curriculum/direct")),
            ("One-hop", context("direct_onehop/direct")),
        ]),
        ("incontext_path", "Path", "In-context DAG", [
            ("Baseline", context("baseline/path")),
            ("Negative path", context("path_negative/path")),
        ]),
        ("incontext_dfs", "DFS", "In-context DAG", [
            ("Full state", context("baseline/dfs_state")),
            ("Nodes", context("dfs_nodes/dfs_nodes")),
        ]),
        ("incontext_bfs", "BFS", "In-context DAG", [
            ("Full state", context("bfs_state/bfs_state")),
            ("Nodes", context("bfs_nodes/bfs_nodes")),
        ]),
    ]


def draw(ax, series: list[tuple[str, list[Path]]]) -> None:
    plotted = []
    ticks = set()
    for label, paths in series:
        missing = [path for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(missing[0])
        xs, mu, sd = mean_sd(paths)
        ticks.update(int(h) for h in xs)
        plotted.append(mu)
        color, marker, linestyle = STYLE[label]
        ax.errorbar(
            xs, mu, yerr=sd if np.any(sd > 0) else None,
            color=color, marker=marker, linestyle=linestyle,
            linewidth=1.5, markersize=5.5, capsize=2.5, elinewidth=0.8, capthick=0.8,
            markerfacecolor="white" if linestyle == "--" else color,
            markeredgecolor=color, markeredgewidth=1.0,
            label=label, zorder=3,
        )
    ax.axhline(0.5, color="#B3B3B3", lw=0.8, ls=(0, (3, 2)), zorder=0)
    if plotted and max(float(np.max(np.abs(mu - 0.5))) for mu in plotted) < 0.03:
        ax.text(0.04, 0.90, "all at chance", transform=ax.transAxes, fontsize=9, color="#555555")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(sorted(ticks))
    ax.set_xlim(min(ticks) - 0.35, max(ticks) + 0.35)
    ax.yaxis.grid(True, color="#EEEEEE", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.set_xlabel("Query hops", fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.legend(
        frameon=True, fontsize=8, loc="best", handlelength=1.8, borderaxespad=0.3,
        framealpha=0.94, edgecolor="none", fancybox=False,
    )


def main() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.2,
        "ytick.major.size": 3.2,
    })
    OUT.mkdir(parents=True, exist_ok=True)
    for name, method, task, series in panels():
        fig, ax = plt.subplots(figsize=(4.8, 3.7))
        draw(ax, series)
        ax.set_title(f"{method}, {task}", fontsize=12, pad=8)
        fig.tight_layout()
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=200, facecolor="white")
        plt.close(fig)
        buffer.seek(0)
        image = Image.open(buffer).convert("RGB")
        path = OUT / f"{name}.png"
        image.save(path, format="PNG", optimize=True)
        print(path)


if __name__ == "__main__":
    main()

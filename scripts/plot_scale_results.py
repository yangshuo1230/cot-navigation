#!/usr/bin/env python3
"""One scale-v1 figure per setup and test DAG.

Fixed static lines use the checkpoint trained at that square. In-context
panels use the checkpoint trained at the matching square size, tested on
that run's own test split, plus the curriculum checkpoint on the same test.
Deep, wide, and 12x12 in-context panels use the 10x10 run, which is the
training length those tests belong to.
"""
from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/scale_v1/launches/scale_v1_s0"
OUT = ROOT / "figures/scale_v1"

DAGS = [
    ("4x4", "4×4", "free_test_best-n16.json"),
    ("4x6", "4×6", "free_test_best-n24.json"),
    ("4x8", "4×8", "free_test_best-n32.json"),
    ("6x6", "6×6", "free_test_best-n36.json"),
    ("8x8", "8×8", "free_test_best-n64.json"),
    ("10x10", "10×10", "free_test_best.json"),
    ("20x5", "20×5 deep", "free_test_best-deep.json"),
    ("5x20", "5×20 wide", "free_test_best-wide.json"),
    ("12x12", "12×12", "free_test_best-larger.json"),
]
PROTOCOLS = [
    ("direct", "Direct"),
    ("path", "Negative path"),
    ("dfs_state", "DFS state"),
    ("bfs_state", "BFS state"),
    ("dfs_nodes", "DFS nodes"),
    ("bfs_nodes", "BFS nodes"),
]
def hops(path: Path) -> dict[int, float]:
    report = json.loads(path.read_text())
    return {int(key): float(row["answer_accuracy"]) for key, row in report["by_distance"].items()}


# In-context checkpoints are the run whose main scale is this square.
# 4x6 and 4x8 had no run of their own. Shape tests belong to the 10x10 run.
CONTEXT_SOURCE = {
    "4x4": ("l4", "free_test_best.json", "Trained at 4×4", "free_test_best-n16.json"),
    "6x6": ("l6", "free_test_best.json", "Trained at 6×6", "free_test_best-n36.json"),
    "8x8": ("l8", "free_test_best.json", "Trained at 8×8", "free_test_best-n64.json"),
    "10x10": ("l10", "free_test_best.json", "Trained at 10×10", "free_test_best.json"),
    "20x5": ("l10", "free_test_best-deep.json", "Trained at 10×10", "free_test_best-deep.json"),
    "5x20": ("l10", "free_test_best-wide.json", "Trained at 10×10", "free_test_best-wide.json"),
    "12x12": ("l10", "free_test_best-larger.json", "Trained at 10×10", "free_test_best-larger.json"),
}
# Fixed static lines use the run trained at that square. 4x6 and 4x8 have no such run.
FIXED_SOURCE = {
    "4x4": ("l4", "free_test_best.json", "Static 4×4", "free_test_best-n16.json"),
    "6x6": ("l6", "free_test_best.json", "Static 6×6", "free_test_best-n36.json"),
    "8x8": ("l8", "free_test_best.json", "Static 8×8", "free_test_best-n64.json"),
    "10x10": ("l10", "free_test_best.json", "Static 10×10", "free_test_best.json"),
}


def context_series(protocol: str, dag_slug: str):
    source = CONTEXT_SOURCE.get(dag_slug)
    if source is None:
        return []
    scale, filename, label, curriculum_file = source
    series = []
    path = RUN / "context" / scale / protocol / "eval" / filename
    if path.exists():
        series.append((label, "#0072B2", "o", "-", hops(path)))
    curriculum = RUN / "context" / "curriculum" / protocol / "eval" / curriculum_file
    if curriculum.exists():
        series.append(("Curriculum", "#D55E00", "s", "--", hops(curriculum)))
    return series


def fixed_series(protocol: str, dag_slug: str, filename: str):
    series = []
    source = FIXED_SOURCE.get(dag_slug)
    if source is not None:
        scale, static_file, label, curriculum_file = source
        path = RUN / "fixed" / scale / "stage_b" / protocol / "eval" / static_file
        if path.exists():
            series.append((label, "#0072B2", "o", "-", hops(path)))
        curriculum = RUN / "fixed" / "curriculum" / "stage_b" / protocol / "eval" / curriculum_file
        if curriculum.exists():
            series.append(("Curriculum", "#D55E00", "s", "--", hops(curriculum)))
    elif dag_slug in {"4x6", "4x8"}:
        curriculum = RUN / "fixed" / "curriculum" / "stage_b" / protocol / "eval" / filename
        if curriculum.exists():
            series.append(("Curriculum", "#D55E00", "s", "--", hops(curriculum)))
    return series


def draw(setting: str, protocol: str, title_protocol: str, dag_slug: str, dag_title: str, filename: str) -> bool:
    folder = "incontext" if setting == "context" else setting
    path = OUT / folder / protocol / f"{dag_slug}.png"
    if setting == "context":
        series = context_series(protocol, dag_slug)
    else:
        series = fixed_series(protocol, dag_slug, filename)
    if not series:
        path.unlink(missing_ok=True)
        return False
    ticks = sorted(set().union(*(item.keys() for *_rest, item in series)))
    fig, ax = plt.subplots(figsize=(6.6, 3.7) if len(ticks) > 12 else (4.8, 3.7))
    plotted = []
    for label, color, marker, linestyle, values in series:
        xs = np.asarray(ticks, dtype=float)
        ys = np.asarray([values.get(hop, np.nan) for hop in ticks], dtype=float)
        plotted.append(ys)
        ax.plot(
            xs, ys, color=color, marker=marker, linestyle=linestyle,
            linewidth=1.6, markersize=5.5, markerfacecolor="white" if linestyle == "--" else color,
            markeredgecolor=color, markeredgewidth=1.0, label=label, zorder=3,
        )
    ax.axhline(0.5, color="#B3B3B3", lw=0.8, ls=(0, (3, 2)), zorder=0)
    finite = [ys[np.isfinite(ys)] for ys in plotted if np.any(np.isfinite(ys))]
    if finite and max(float(np.max(np.abs(ys - 0.5))) for ys in finite) < 0.03:
        ax.text(0.04, 0.90, "all at chance", transform=ax.transAxes, fontsize=9, color="#555555")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(ticks)
    ax.set_xlim(min(ticks) - 0.35, max(ticks) + 0.35)
    ax.yaxis.grid(True, color="#EEEEEE", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8 if len(ticks) > 12 else 9)
    task = "Fixed DAG" if setting == "fixed" else "In-context DAG"
    ax.set_title(f"{title_protocol}, {dag_title}, {task}", fontsize=11, pad=8)
    ax.set_xlabel("Query hops", fontsize=10)
    ax.set_ylabel("Test accuracy", fontsize=10)
    ax.legend(
        frameon=True, fontsize=8, loc="best", handlelength=2.2, borderaxespad=0.3,
        framealpha=0.94, edgecolor="none", fancybox=False,
    )
    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=200, facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB")
    folder = "incontext" if setting == "context" else setting
    path = OUT / folder / protocol / f"{dag_slug}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)
    return True


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
    for stale in OUT.glob("*.png"):
        stale.unlink()
    for stale in list(OUT.glob("incontext/*/4x6.png")) + list(OUT.glob("incontext/*/4x8.png")):
        stale.unlink()
    written = 0
    for setting in ("fixed", "context"):
        for protocol, title in PROTOCOLS:
            for slug, dag_title, filename in DAGS:
                written += draw(setting, protocol, title, slug, dag_title, filename)
    print(f"{written} figures in {OUT}")


if __name__ == "__main__":
    main()

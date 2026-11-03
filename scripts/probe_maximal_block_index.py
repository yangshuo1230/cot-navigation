#!/usr/bin/env python3
"""Probe overlapping maximal-block membership from fixed 10x10 endpoint states.

A block is a maximal set in which every shallower member reaches every deeper
member. A node belongs to many blocks, so the target is that membership set,
not a single class id. Single-layer blocks (one entire layer) are left out:
they only repeat the layer id.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import probe_block_membership as base  # noqa: E402

OUT = ROOT / "studies/block_membership_probe/block_index.json"
GRAPH = ROOT / "data/scale_v1/fixed_l10w10_n100_m150_s0/main-seed0/graph.npz"
DEVICE = "cuda:4"


def reachability(graph: Path):
    data = np.load(graph)
    ptr, flat = data["nbr_ptr"], data["nbr_flat"]
    n = len(ptr) - 1
    adj = [flat[int(ptr[i]):int(ptr[i + 1])].astype(int).tolist() for i in range(n)]
    reach = [0] * n
    for layer in range(8, -1, -1):
        for u in range(layer * 10, layer * 10 + 10):
            bits = 0
            for v in adj[u]:
                bits |= (1 << int(v)) | reach[int(v)]
            reach[u] = bits
    reach_ll = [[0] * 10 for _ in range(n)]
    for u in range(n):
        bits = reach[u]
        while bits:
            bit = bits & -bits
            v = bit.bit_length() - 1
            reach_ll[u][v // 10] |= 1 << (v % 10)
            bits ^= bit
    return reach_ll


def maximal_blocks(reach_ll) -> list[tuple[int, ...]]:
    blocks: list[tuple[int, ...]] = []

    def candidates(masks, layer):
        cand = (1 << 10) - 1
        for earlier in range(layer):
            mask = masks[earlier]
            while mask:
                bit = mask & -mask
                node = earlier * 10 + (bit.bit_length() - 1)
                cand &= reach_ll[node][layer]
                mask ^= bit
                if cand == 0:
                    return 0
        return cand

    def rec(layer, masks, nonempty):
        if layer == 10:
            if not nonempty:
                return
            for check in range(10):
                cand = (1 << 10) - 1
                for earlier in range(check):
                    mask = masks[earlier]
                    while mask:
                        bit = mask & -mask
                        node = earlier * 10 + (bit.bit_length() - 1)
                        cand &= reach_ll[node][check]
                        mask ^= bit
                keep = 0
                pending = cand
                while pending:
                    bit = pending & -pending
                    node = check * 10 + (bit.bit_length() - 1)
                    ok = True
                    for later in range(check + 1, 10):
                        need = masks[later]
                        if need and (reach_ll[node][later] & need) != need:
                            ok = False
                            break
                    if ok:
                        keep |= bit
                    pending ^= bit
                if keep != masks[check]:
                    return
            blocks.append(tuple(masks))
            return
        cand = candidates(masks, layer)
        sub = cand
        while True:
            masks[layer] = sub
            rec(layer + 1, masks, nonempty or sub != 0)
            if sub == 0:
                break
            sub = (sub - 1) & cand
        masks[layer] = 0

    rec(0, [0] * 10, False)
    return blocks


def membership_matrix(blocks: list[tuple[int, ...]]) -> np.ndarray:
    """Drop blocks that occupy only one layer."""
    kept = [masks for masks in blocks if sum(mask != 0 for mask in masks) >= 2]
    matrix = np.zeros((100, len(kept)), np.float64)
    for index, masks in enumerate(kept):
        for layer, mask in enumerate(masks):
            pending = mask
            while pending:
                bit = pending & -pending
                matrix[layer * 10 + (bit.bit_length() - 1), index] = 1
                pending ^= bit
    return matrix


def ridge_multi(x_train, y_train, x_val, y_val, x_test) -> np.ndarray:
    x_train = np.asarray(x_train, np.float64)
    mu = x_train.mean(0)
    sig = np.maximum(x_train.std(0), 1e-4)

    def norm(x):
        return (np.asarray(x, np.float64) - mu) / sig

    train, val, test = norm(x_train), norm(x_val), norm(x_test)
    ymu = np.asarray(y_train, np.float64).mean(0)
    ytr = np.asarray(y_train, np.float64) - ymu
    yva = np.asarray(y_val, np.float64) - ymu
    xtx = train.T @ train
    xty = train.T @ ytr
    eye = np.eye(train.shape[1])
    best_mse, best_w = None, None
    for lam in np.logspace(-1, 5, 13):
        weight = np.linalg.solve(xtx + lam * eye, xty)
        mse = float(np.mean((val @ weight - yva) ** 2))
        if best_mse is None or mse < best_mse:
            best_mse, best_w = mse, weight
    return test @ best_w + ymu


def node_aucs(y_true: np.ndarray, scores: np.ndarray) -> float:
    values = []
    for row in range(y_true.shape[0]):
        if len(np.unique(y_true[row])) < 2:
            continue
        values.append(base.auc(y_true[row], scores[row]))
    return float(np.mean(values))


def probe_matrix(features: np.ndarray, labels: np.ndarray, seed: int, *, shuffle=False) -> float:
    y = labels
    if shuffle:
        y = labels[np.random.default_rng(20_000 + seed).permutation(len(labels))]
    level = np.arange(100) // 10
    train, val, test = base.stratified_split(np.arange(100), level, seed)
    pred = ridge_multi(features[train], y[train], features[val], y[val], features[test])
    return node_aucs(y[test], pred)


def main() -> None:
    base.DEVICE = DEVICE
    device = torch.device(DEVICE)
    blocks = maximal_blocks(reachability(GRAPH))
    labels = membership_matrix(blocks)
    # Numerical rank of the centered membership code.
    centered = labels - labels.mean(0)
    singular = np.linalg.svd(centered, compute_uv=False)
    rank = int(np.sum(singular > singular[0] * 1e-8))
    print(f"blocks {labels.shape[1]} rank {rank} "
          f"positives/node {labels.sum(1).mean():.1f}", flush=True)
    results = {}
    for setting, protocol, run in base.jobs():
        print(f"probe {setting}/{protocol}", flush=True)
        pack = base.extract(run, device)
        hidden, _ = base.node_matrix(pack["src"], pack["layers"][-1][:, 0])
        embed, _ = base.node_matrix(pack["src"], pack["e_src"])
        row = {
            "hidden": [probe_matrix(hidden, labels, seed) for seed in base.SEEDS],
            "embedding": [probe_matrix(embed, labels, seed) for seed in base.SEEDS],
            "hidden_shuffled": [probe_matrix(hidden, labels, seed, shuffle=True) for seed in base.SEEDS],
            "embedding_shuffled": [probe_matrix(embed, labels, seed, shuffle=True) for seed in base.SEEDS],
        }
        # Best residual layer, same five splits.
        best_name, best_auc = None, -1.0
        for index, layer in enumerate(pack["layers"]):
            feat, _ = base.node_matrix(pack["src"], layer[:, 0])
            aucs = [probe_matrix(feat, labels, seed) for seed in base.SEEDS]
            mean = float(np.mean(aucs))
            row[f"L{index + 1}"] = aucs
            if mean > best_auc:
                best_name, best_auc = f"L{index + 1}", mean
        row["best_layer"] = best_name
        summary = {key: {"auc": float(np.mean(val)), "auc_std": float(np.std(val))}
                   for key, val in row.items() if key != "best_layer"}
        summary["best_layer"] = best_name
        results.setdefault(setting, {})[protocol] = summary
        print(f"  L6 {summary['hidden']['auc']:.3f} emb {summary['embedding']['auc']:.3f} "
              f"best {best_name} {summary[best_name]['auc']:.3f} "
              f"shuf {summary['hidden_shuffled']['auc']:.3f}", flush=True)
        del pack
    payload = {
        "definition": "maximal sets in which every shallower member reaches every deeper member",
        "excluded": "10 single-layer blocks, one per full layer",
        "n_blocks": int(labels.shape[1]),
        "membership_rank": rank,
        "mean_blocks_per_node": float(labels.sum(1).mean()),
        "metric": "per held-out node, AUC of ranking the block indices it belongs to",
        "results": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(OUT)


if __name__ == "__main__":
    main()

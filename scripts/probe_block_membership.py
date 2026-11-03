#!/usr/bin/env python3
"""Linear probes for DAG layer and 5-layer block membership.

Fixed 10x10 prompt: <WORLD> <QUERY> source target |. The source residual has
not seen the target. Layer is node_index // 10. Block is node_index // 50.
"""
from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cot_navigation.config import load_config  # noqa: E402
from cot_navigation.data.dataset import ParametricWorld  # noqa: E402
from cot_navigation.models.transformer import build_model  # noqa: E402
from cot_navigation.protocols.tokenizer import Tokenizer  # noqa: E402

SCALE = ROOT / "outputs/scale_v1/launches/scale_v1_s0/fixed"
BLOCK = ROOT / "outputs/block_query_v1/launches/block_n100_s0"
OUT = ROOT / "studies/block_membership_probe"
FIG = ROOT / "figures/scale_v1/endpoint_layer_probe.png"
PROTOCOLS = ["direct", "path", "dfs_state", "dfs_nodes", "bfs_state", "bfs_nodes"]
PROTOCOL_LABELS = ["Direct", "Path", "DFS state", "DFS nodes", "BFS state", "BFS nodes"]
SETTINGS = [
    ("static", "Static 10×10", "#0072B2"),
    ("curriculum", "Curriculum", "#009E73"),
    ("within_only", "Within-block train", "#E69F00"),
    ("cross_allowed", "Cross-block train", "#D55E00"),
]
SEEDS = (0, 1, 2, 3, 4)
QUERY_CAP = 1200
DEVICE = "cuda:7"


def jobs() -> list[tuple[str, str, Path]]:
    rows = []
    for protocol in PROTOCOLS:
        rows.append(("static", protocol, SCALE / f"l10/stage_b/{protocol}"))
        rows.append(("curriculum", protocol, SCALE / f"curriculum/stage_b/{protocol}"))
        arm = BLOCK / "within_only"
        rows.append(("within_only", protocol, arm if protocol == "direct" else arm / protocol))
        arm = BLOCK / "cross_allowed"
        rows.append(("cross_allowed", protocol, arm if protocol == "direct" else arm / protocol))
    return rows


def world_dir(cfg) -> Path:
    root = (ROOT / cfg.data.root).resolve()
    hits = sorted(p.parent for p in root.glob("*/world.json"))
    if len(hits) == 1:
        return hits[0]
    listed = {(ROOT / p).resolve() for p in cfg.train.world_roots}
    matched = [p for p in hits if p.resolve() in listed]
    if len(matched) == 1:
        return matched[0]
    raise FileNotFoundError(f"world under {root}: {[str(p) for p in hits]}")


def world_token(cfg, tok: Tokenizer, world: Path) -> int:
    key = world.resolve()
    for index, raw in enumerate(cfg.train.world_roots):
        if (ROOT / raw).resolve() == key:
            return tok.id(f"<WORLD_{index}>")
    raise KeyError(f"{world} is not a trained world of this run")


def components(adj: list[list[int]]) -> list[int]:
    n = len(adj)
    und = [[] for _ in range(n)]
    for u, nbrs in enumerate(adj):
        for v in nbrs:
            und[u].append(v)
            und[v].append(u)
    seen = [-1] * n
    sizes = []
    for start in range(n):
        if seen[start] >= 0:
            continue
        stack = [start]
        seen[start] = len(sizes)
        count = 0
        while stack:
            u = stack.pop()
            count += 1
            for v in und[u]:
                if seen[v] < 0:
                    seen[v] = seen[u]
                    stack.append(v)
        sizes.append(count)
    return sorted(sizes)


def subsample(same, label, dist, cap: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    buckets: dict[tuple, list[int]] = {}
    for i, key in enumerate(zip(same.tolist(), label.tolist(), dist.tolist())):
        buckets.setdefault(key, []).append(i)
    keys = sorted(buckets)
    for key in keys:
        rng.shuffle(buckets[key])
    total = sum(len(v) for v in buckets.values())
    limit = min(cap, total)
    chosen: list[int] = []
    while len(chosen) < limit:
        grew = False
        for key in keys:
            if buckets[key] and len(chosen) < limit:
                chosen.append(buckets[key].pop())
                grew = True
        if not grew:
            break
    return np.asarray(chosen, dtype=np.int64)


def split_by_class(y: np.ndarray, seed: int, fracs=(0.5, 0.25, 0.25)):
    rng = np.random.default_rng(seed)
    parts: list[list[np.ndarray]] = [[], [], []]
    y = np.asarray(y)
    for cls in sorted(np.unique(y).tolist()):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        n = len(idx)
        if n < 4:
            parts[0].append(idx)
            continue
        a = min(max(int(round(n * fracs[0])), 1), n - 2)
        b = min(max(int(round(n * (fracs[0] + fracs[1]))), a + 1), n - 1)
        parts[0].append(idx[:a])
        parts[1].append(idx[a:b])
        parts[2].append(idx[b:])
    if any(len(part) == 0 for part in parts):
        raise RuntimeError("query split left a fold empty")
    return [np.concatenate(part) for part in parts]


def stratified_split(nodes: np.ndarray, labels: np.ndarray, seed: int, fracs=(0.6, 0.2, 0.2)):
    """Split node ids 60/20/20 within each class. labels is indexed by node id."""
    rng = np.random.default_rng(seed)
    parts: list[list[np.ndarray]] = [[], [], []]
    nodes = np.asarray(nodes)
    classes = sorted({int(labels[int(n)]) for n in nodes})
    for cls in classes:
        idx = np.asarray([int(n) for n in nodes if int(labels[int(n)]) == cls])
        rng.shuffle(idx)
        n = len(idx)
        if n < 3:
            parts[0].append(idx)
            continue
        a = min(max(int(round(n * fracs[0])), 1), n - 2)
        b = min(max(int(round(n * (fracs[0] + fracs[1]))), a + 1), n - 1)
        parts[0].append(idx[:a])
        parts[1].append(idx[a:b])
        parts[2].append(idx[b:])
    if any(len(part) == 0 for part in parts):
        raise RuntimeError(f"empty split for classes {classes}")
    return [np.concatenate(part) for part in parts]


def auc(y, score) -> float:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=np.float64)
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float((np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / diff.size)


def balanced_accuracy(y, score) -> float:
    y = np.asarray(y).astype(int)
    pred = np.asarray(score) >= 0
    pos, neg = y == 1, y == 0
    if not pos.any() or not neg.any():
        return float("nan")
    return float(0.5 * (pred[pos].mean() + (~pred[neg]).mean()))


def fit_scores(x_train, y_train, x_val, y_val, x_test, seed: int, steps=250, penalty=1e-3):
    torch.manual_seed(seed)
    device = torch.device(DEVICE)
    x = torch.tensor(x_train, device=device, dtype=torch.float32)
    v = torch.tensor(x_val, device=device, dtype=torch.float32)
    t = torch.tensor(x_test, device=device, dtype=torch.float32)
    mean, scale = x.mean(0), x.std(0).clamp(min=1e-4)
    x, v, t = (x - mean) / scale, (v - mean) / scale, (t - mean) / scale
    y = torch.tensor(y_train, device=device, dtype=torch.float32)
    vy = torch.tensor(y_val, device=device, dtype=torch.float32)
    pos = y.sum().clamp(min=1)
    neg = (1 - y).sum().clamp(min=1)
    pos_weight = (neg / pos).clamp(0.2, 20)
    weight = torch.nn.Parameter(torch.zeros(x.shape[1], device=device))
    bias = torch.nn.Parameter(torch.zeros((), device=device))
    opt = torch.optim.AdamW([weight, bias], lr=0.03, weight_decay=penalty)
    best, best_w, best_b = float("inf"), weight.detach().clone(), bias.detach().clone()
    for step in range(steps):
        loss = F.binary_cross_entropy_with_logits(x @ weight + bias, y, pos_weight=pos_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 10 == 0 or step == steps - 1:
            with torch.no_grad():
                val = F.binary_cross_entropy_with_logits(v @ weight + bias, vy, pos_weight=pos_weight)
            if float(val) < best:
                best = float(val)
                best_w, best_b = weight.detach().clone(), bias.detach().clone()
    with torch.no_grad():
        return (t @ best_w + best_b).cpu().numpy()


def summarize(y, score) -> dict:
    return {"auc": auc(y, score), "balanced_accuracy": balanced_accuracy(y, score),
            "n": int(len(y)), "n_pos": int(np.sum(np.asarray(y) == 1))}


def mean_std(rows: list[dict]) -> dict:
    aucs = [r["auc"] for r in rows]
    bals = [r["balanced_accuracy"] for r in rows]
    return {
        "auc": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "balanced_accuracy": float(np.mean(bals)),
        "n": rows[0]["n"],
        "n_pos": rows[0]["n_pos"],
    }


def regression_metrics(y, pred) -> dict:
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if len(y) < 3 or float(np.std(y)) < 1e-8:
        return {"r": float("nan"), "r2": float("nan"), "mae": float("nan"), "n": int(len(y))}
    r = float(np.corrcoef(y, pred)[0, 1])
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1 - np.sum((y - pred) ** 2) / ss_tot)
    return {"r": r, "r2": r2, "mae": float(np.mean(np.abs(y - pred))), "n": int(len(y))}


def mean_reg(rows: list[dict]) -> dict:
    return {
        "r": float(np.nanmean([row["r"] for row in rows])),
        "r_std": float(np.nanstd([row["r"] for row in rows])),
        "r2": float(np.nanmean([row["r2"] for row in rows])),
        "mae": float(np.nanmean([row["mae"] for row in rows])),
        "n": rows[0]["n"],
    }


def ridge_predict(x_train, y_train, x_val, y_val, x_test) -> np.ndarray:
    """Ridge map onto a scalar. Validation mean squared error chooses the penalty."""
    x_train = np.asarray(x_train, np.float64)
    y_train = np.asarray(y_train, np.float64)
    mu, sig = x_train.mean(0), np.maximum(x_train.std(0), 1e-4)
    ymu = float(y_train.mean())

    def norm(x):
        return (np.asarray(x, np.float64) - mu) / sig

    train, val, test = norm(x_train), norm(x_val), norm(x_test)
    ytr = y_train - ymu
    yva = np.asarray(y_val, np.float64) - ymu
    xtx = train.T @ train
    xty = train.T @ ytr
    eye = np.eye(train.shape[1])
    best_mse, best_w = None, None
    for lam in np.logspace(-2, 4, 13):
        weight = np.linalg.solve(xtx + lam * eye, xty)
        mse = float(np.mean((val @ weight - yva) ** 2))
        if best_mse is None or mse < best_mse:
            best_mse, best_w = mse, weight
    return test @ best_w + ymu


def probe_level(features: np.ndarray, labels: np.ndarray, seed: int, *, permute=False) -> dict:
    """One row per node. Held-out nodes are stratified by the label being predicted."""
    y = np.asarray(labels).copy()
    if permute:
        y = np.random.default_rng(10_000 + seed).permutation(y)
    train, val, test = stratified_split(np.arange(len(y)), y, seed)
    pred = ridge_predict(features[train], y[train], features[val], y[val], features[test])
    return regression_metrics(y[test], pred)


def probe_gap(features: np.ndarray, gap: np.ndarray, seed: int, *, shuffle=False) -> dict:
    gap = np.asarray(gap, np.float64)
    train, val, test = split_by_class(np.rint(gap).astype(int), seed)
    ytr = gap[train]
    if shuffle:
        ytr = np.random.default_rng(seed + 17).permutation(ytr)
    pred = ridge_predict(features[train], ytr, features[val], gap[val], features[test])
    return regression_metrics(gap[test], pred)


def seen_matrix(ids: np.ndarray, hidden: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros((100, hidden.shape[1]), np.float32)
    seen = np.zeros(100, dtype=bool)
    for node in range(100):
        rows = hidden[ids == node]
        if len(rows) == 0:
            continue
        out[node] = rows.mean(0)
        seen[node] = True
    return out, seen


def probe_nodes(features: np.ndarray, labels: np.ndarray, seed: int) -> dict:
    train, val, test = stratified_split(np.arange(len(labels)), labels, seed)
    score = fit_scores(features[train], labels[train], features[val], labels[val], features[test], seed)
    return summarize(labels[test], score)


def probe_rows(features: np.ndarray, labels: np.ndarray, groups: np.ndarray, node_labels: np.ndarray,
               seed: int, *, shuffle=False) -> dict:
    """Hold out entire endpoint ids. node_labels is indexed by node id."""
    train_g, val_g, test_g = stratified_split(np.unique(groups), node_labels, seed)

    def take(nodes):
        mask = np.isin(groups, nodes)
        return features[mask], labels[mask]

    xtr, ytr = take(train_g)
    xva, yva = take(val_g)
    xte, yte = take(test_g)
    if shuffle:
        ytr = np.random.default_rng(seed + 17).permutation(ytr)
    if min(len(ytr), len(yva), len(yte)) == 0 or len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2 or len(np.unique(yva)) < 2:
        return {"auc": float("nan"), "balanced_accuracy": float("nan"),
                "n": int(len(yte)), "n_pos": int(np.sum(yte == 1)) if len(yte) else 0}
    score = fit_scores(xtr, ytr, xva, yva, xte, seed)
    return summarize(yte, score)


def probe_queries(features: np.ndarray, labels: np.ndarray, seed: int, *, shuffle=False) -> dict:
    train, val, test = split_by_class(labels, seed)
    ytr = labels[train]
    if shuffle:
        ytr = np.random.default_rng(seed + 17).permutation(ytr)
    score = fit_scores(features[train], ytr, features[val], labels[val], features[test], seed)
    return summarize(labels[test], score)


def node_matrix(src: np.ndarray, hidden: np.ndarray) -> tuple[np.ndarray, float]:
    dim = hidden.shape[1]
    out = np.zeros((100, dim), np.float32)
    deviation = 0.0
    for node in range(100):
        rows = hidden[src == node]
        if len(rows) == 0:
            raise RuntimeError(f"node {node} never appears as a source")
        out[node] = rows.mean(0)
        deviation = max(deviation, float(np.max(np.abs(rows - out[node]))))
    return out, deviation


@torch.inference_mode()
def extract(run: Path, device: torch.device) -> dict:
    cfg = load_config([run / "resolved_config.yaml"])
    tok = Tokenizer.from_config(cfg)
    world = world_dir(cfg)
    store = ParametricWorld(world)
    token = world_token(cfg, tok, world)
    nodes = store.node_vocab_ids(tok)
    queries = store.queries
    mask = (queries["split"] == 2) & (queries["distance"] >= 2)
    src = queries["s_topo"][mask].astype(np.int64)
    tgt = queries["t_topo"][mask].astype(np.int64)
    label = queries["label"][mask].astype(np.int64)
    dist = queries["distance"][mask].astype(np.int64)
    same = (src // 50 == tgt // 50).astype(np.int64)
    chosen = subsample(same, label, dist, QUERY_CAP, seed=0)
    src, tgt, label, dist, same = src[chosen], tgt[chosen], label[chosen], dist[chosen], same[chosen]
    # Layer-9 nodes are never query sources. The source residual does not see the
    # target, so a synthetic second endpoint still reads that node's state.
    missing = [node for node in range(store.n_nodes) if node not in set(src.tolist())]
    if missing:
        src = np.concatenate([src, np.asarray(missing, np.int64)])
        tgt = np.concatenate([tgt, np.asarray([0 if node else 1 for node in missing], np.int64)])
        same = np.concatenate([same, np.full(len(missing), -1, np.int64)])
    prompts = [[token, tok.id("<QUERY>"), nodes[int(s)], nodes[int(t)], tok.id("|")]
               for s, t in zip(src, tgt)]
    model = build_model(cfg.model, len(tok), tok.pad).to(device).eval()
    state = torch.load(run / "checkpoints/best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    step = int(state["step"])
    del state
    layers = []
    embed_src, embed_tgt = [], []
    for start in range(0, len(prompts), 256):
        batch = prompts[start:start + 256]
        ids = torch.tensor(batch, device=device, dtype=torch.long)
        capture: dict = {}
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            model(ids, capture=capture)
        resid = capture["resid"]
        if not layers:
            layers = [[] for _ in resid]
        for index, hidden in enumerate(resid):
            layers[index].append(hidden[:, 2:5].float().cpu())
        emb = model.embed(ids).float().cpu()
        embed_src.append(emb[:, 2])
        embed_tgt.append(emb[:, 3])
        del resid, emb
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    stacked = [torch.cat(chunks, 0).numpy() for chunks in layers]
    e_src = torch.cat(embed_src, 0).numpy()
    e_tgt = torch.cat(embed_tgt, 0).numpy()
    query = same >= 0
    return {
        "step": step,
        "world": str(world.relative_to(ROOT)),
        "world_token": int(token),
        "n_layers": len(stacked),
        "src": src,
        "tgt": tgt,
        "same": same,
        "query": query,
        "layers": stacked,
        "e_src": e_src,
        "e_tgt": e_tgt,
        "graph_components": components(store.adj()),
        "same_rate": float(same[query].mean()),
        "n_queries": int(query.sum()),
    }


def evaluate_model(pack: dict) -> dict:
    query = pack["query"]
    src, tgt = pack["src"], pack["tgt"]
    block = np.arange(100) // 50
    level = np.arange(100) // 10
    same = pack["same"][query]
    gap = (tgt[query] // 10 - src[query] // 10).astype(np.float64)
    out = {"step": pack["step"], "world": pack["world"], "world_token": pack["world_token"],
           "n_queries": pack["n_queries"], "same_rate": pack["same_rate"],
           "n_target_nodes": int(len(np.unique(tgt[query]))),
           "target_levels": sorted(int(v) for v in np.unique(tgt[query] // 10)),
           "gap_min": int(gap.min()), "gap_max": int(gap.max()),
           "components": pack["graph_components"], "layers": {}}
    e_node, e_dev = node_matrix(src, pack["e_src"])
    e_tgt_node, tgt_seen = seen_matrix(tgt[query], pack["e_tgt"][query])
    out["embedding_source_deviation"] = e_dev
    out["embedding"] = {}
    last = pack["n_layers"] - 1
    for layer, hidden in enumerate(pack["layers"]):
        h_src, h_tgt, h_bar = hidden[:, 0], hidden[:, 1], hidden[:, 2]
        node_h, deviation = node_matrix(src, h_src)
        tgt_h, seen = seen_matrix(tgt[query], h_tgt[query])
        layer_name = f"L{layer + 1}"
        record = {
            "source_deviation": deviation,
            "source_layer": mean_reg([probe_level(node_h, level, seed) for seed in SEEDS]),
            "target_layer": mean_reg([
                probe_level(tgt_h[seen], level[seen], seed) for seed in SEEDS
            ]),
        }
        if layer == last:
            record["source_block"] = mean_std([probe_nodes(node_h, block, seed) for seed in SEEDS])
            record["target_block"] = mean_std([
                probe_rows(h_tgt[query], block[tgt[query]], tgt[query], block, seed) for seed in SEEDS
            ])
            record["same_block"] = mean_std([
                probe_queries(np.concatenate([h_src[query], h_tgt[query]], axis=1), same, seed)
                for seed in SEEDS
            ])
            random_rows = []
            for seed in SEEDS:
                random_labels = np.array([0] * 50 + [1] * 50)
                np.random.default_rng(10_000 + seed).shuffle(random_labels)
                random_rows.append(probe_nodes(node_h, random_labels, seed))
            record["source_block_random"] = mean_std(random_rows)
            record["target_block_shuffled"] = mean_std([
                probe_rows(h_tgt[query], block[tgt[query]], tgt[query], block, seed, shuffle=True)
                for seed in SEEDS
            ])
            record["same_block_shuffled"] = mean_std([
                probe_queries(np.concatenate([h_src[query], h_tgt[query]], axis=1), same, seed, shuffle=True)
                for seed in SEEDS
            ])
            record["same_block_bar"] = mean_std([
                probe_queries(h_bar[query], same, seed) for seed in SEEDS
            ])
            record["source_layer_random"] = mean_reg([
                probe_level(node_h, level, seed, permute=True) for seed in SEEDS
            ])
            pair = np.concatenate([h_src[query], h_tgt[query]], axis=1)
            record["layer_gap"] = mean_reg([probe_gap(pair, gap, seed) for seed in SEEDS])
            record["layer_gap_shuffled"] = mean_reg([
                probe_gap(pair, gap, seed, shuffle=True) for seed in SEEDS
            ])
            forward = gap > 0
            record["same_block_forward"] = mean_std([
                probe_queries(pair[forward], same[forward], seed) for seed in SEEDS
            ])
        out["layers"][layer_name] = record
    embed_pair = np.concatenate([pack["e_src"][query], pack["e_tgt"][query]], axis=1)
    out["embedding"] = {
        "source_layer": mean_reg([probe_level(e_node, level, seed) for seed in SEEDS]),
        "target_layer": mean_reg([
            probe_level(e_tgt_node[tgt_seen], level[tgt_seen], seed) for seed in SEEDS
        ]),
        "source_layer_random": mean_reg([
            probe_level(e_node, level, seed, permute=True) for seed in SEEDS
        ]),
        "layer_gap": mean_reg([probe_gap(embed_pair, gap, seed) for seed in SEEDS]),
        "layer_gap_shuffled": mean_reg([
            probe_gap(embed_pair, gap, seed, shuffle=True) for seed in SEEDS
        ]),
        "same_block_forward": mean_std([
            probe_queries(embed_pair[gap > 0], same[gap > 0], seed) for seed in SEEDS
        ]),
        "source_block": mean_std([probe_nodes(e_node, block, seed) for seed in SEEDS]),
        "target_block": mean_std([
            probe_rows(pack["e_tgt"][query], block[tgt[query]], tgt[query], block, seed) for seed in SEEDS
        ]),
        "same_block": mean_std([probe_queries(embed_pair, same, seed) for seed in SEEDS]),
        "same_block_shuffled": mean_std([
            probe_queries(embed_pair, same, seed, shuffle=True) for seed in SEEDS
        ]),
    }
    random_rows = []
    for seed in SEEDS:
        random_labels = np.array([0] * 50 + [1] * 50)
        np.random.default_rng(10_000 + seed).shuffle(random_labels)
        random_rows.append(probe_nodes(e_node, random_labels, seed))
    out["embedding"]["source_block_random"] = mean_std(random_rows)
    return out


def fmt(cell: dict, key: str) -> str:
    return f"{cell[key]:.3f}"


def print_table(results: dict) -> None:
    print(f"{'setting':<16}{'protocol':<12}{'src r':>8}{'src emb':>8}{'tgt r':>8}{'tgt emb':>8}"
          f"{'gap r':>8}{'gap emb':>8}{'rand':>8}")
    present = [item for item in SETTINGS if item[0] in results]
    for setting, _, _ in present:
        for protocol in PROTOCOLS:
            row = results[setting][protocol]
            last = sorted(row["layers"], key=lambda name: int(name[1:]))[-1]
            layer, emb = row["layers"][last], row["embedding"]
            print(f"{setting:<16}{protocol:<12}"
                  f"{fmt(layer['source_layer'], 'r'):>8}{fmt(emb['source_layer'], 'r'):>8}"
                  f"{fmt(layer['target_layer'], 'r'):>8}{fmt(emb['target_layer'], 'r'):>8}"
                  f"{fmt(layer['layer_gap'], 'r'):>8}{fmt(emb['layer_gap'], 'r'):>8}"
                  f"{fmt(layer['source_layer_random'], 'r'):>8}")
    print(f"{'setting':<16}{'protocol':<12}{'src AUC':>8}{'src emb':>8}{'tgt AUC':>8}{'tgt emb':>8}"
          f"{'same':>8}{'same emb':>9}{'bar':>8}")
    for setting, _, _ in present:
        for protocol in PROTOCOLS:
            row = results[setting][protocol]
            last = sorted(row["layers"], key=lambda name: int(name[1:]))[-1]
            layer, emb = row["layers"][last], row["embedding"]
            print(f"{setting:<16}{protocol:<12}"
                  f"{fmt(layer['source_block'], 'auc'):>8}{fmt(emb['source_block'], 'auc'):>8}"
                  f"{fmt(layer['target_block'], 'auc'):>8}{fmt(emb['target_block'], 'auc'):>8}"
                  f"{fmt(layer['same_block'], 'auc'):>8}{fmt(emb['same_block'], 'auc'):>9}"
                  f"{fmt(layer['same_block_bar'], 'auc'):>8}")


def draw(results: dict) -> None:
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
    panels = [
        ("source_layer", "r", "Source layer", "Test correlation", -0.15, 1.05, 0.0),
        ("target_layer", "r", "Target layer", "Test correlation", -0.15, 1.05, 0.0),
        ("source_block", "auc", "Source block", "Test AUC", 0.35, 1.05, 0.5),
        ("layer_gap", "r", "Layer gap", "Test correlation", -0.15, 1.05, 0.0),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.8))
    x = np.arange(len(PROTOCOLS))
    for ax, (key, field, title, ylabel, lo, hi, chance) in zip(axes.ravel(), panels):
        for setting, label, color in SETTINGS:
            hidden, control = [], []
            for protocol in PROTOCOLS:
                row = results[setting][protocol]
                last = sorted(row["layers"], key=lambda name: int(name[1:]))[-1]
                hidden.append(row["layers"][last][key][field])
                control.append(row["embedding"][key][field])
            ax.plot(x, hidden, color=color, marker="o", ms=4.2, lw=1.6, label=label, zorder=3)
            ax.plot(x, control, color=color, marker="o", ms=4.2, lw=1.05, ls="--",
                    markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.1, zorder=2)
        ax.axhline(chance, color="#9A9A9A", lw=0.8, ls=":", zorder=1)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(PROTOCOL_LABELS, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(lo, hi)
        ax.set_xlim(-0.25, len(PROTOCOLS) - 0.75)
        ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, color="#E6E6E6", lw=0.6)
        ax.set_axisbelow(True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.035))
    fig.suptitle("Layer and block from endpoint states, fixed 10×10", fontsize=12, y=0.995)
    fig.text(0.5, 0.008, "Solid: last-layer residual.   Dashed: token embedding.   Dotted: chance.",
             ha="center", va="bottom", fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=200, facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB")
    FIG.parent.mkdir(parents=True, exist_ok=True)
    tmp = FIG.with_suffix(".png.tmp")
    image.save(tmp, format="PNG")
    tmp.replace(FIG)
    print(FIG)


def main() -> None:
    device = torch.device(DEVICE)
    results = {}
    graph = None
    chosen = set()
    if len(sys.argv) == 3:
        chosen.add((sys.argv[1], sys.argv[2]))
    for setting, protocol, run in jobs():
        if chosen and (setting, protocol) not in chosen:
            continue
        print(f"probe {setting}/{protocol}", flush=True)
        pack = extract(run, device)
        if graph is None:
            graph = {"components": pack["graph_components"], "scale_world": pack["world"],
                     "scale_same_rate": pack["same_rate"]}
        elif setting == "within_only" and protocol == "direct":
            graph["block_world"] = pack["world"]
            graph["block_same_rate"] = pack["same_rate"]
            graph["block_components"] = pack["graph_components"]
        row = evaluate_model(pack)
        row["n_layers"] = pack["n_layers"]
        last = row["layers"][f"L{pack['n_layers']}"]
        print(f"  step {row['step']} layer {last['source_layer']['r']:.3f}/{last['target_layer']['r']:.3f} "
              f"emb {row['embedding']['source_layer']['r']:.3f}/{row['embedding']['target_layer']['r']:.3f} "
              f"block {last['source_block']['auc']:.3f}/{last['same_block']['auc']:.3f} "
              f"dev {last['source_deviation']:.4f}", flush=True)
        results.setdefault(setting, {})[protocol] = row
        del pack
        if not chosen:
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / "results.json").write_text(json.dumps({"partial": True, "results": results}))
    payload = {
        "layer": "node_index // 10; DAG depth 0 through 9",
        "block": "node_index // 50; layers 0-4 versus layers 5-9",
        "layer_metric": "Pearson r of a ridge regression, penalty chosen on held-out nodes",
        "prompt": "<WORLD> <QUERY> source target |",
        "positions": {"source": 2, "target": 3, "bar": 4},
        "seeds": list(SEEDS),
        "query_cap": QUERY_CAP,
        "graph": graph,
        "results": results,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(payload, indent=2))
    print_table(results)
    if not chosen:
        draw(results)


if __name__ == "__main__":
    main()

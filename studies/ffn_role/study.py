"""Does the fixed-DAG Direct FFN store graph facts, process them, or both?

An FFN is a position-wise function of its input. It cannot see another token
unless attention has already mixed that token into the residual. "Stored in
the FFN" therefore means the trained weights implement a node-addressed map
that does not generalize from other nodes' embedding geometry. "Processed by
the FFN" means the facts are already in the residual (often the embedding)
and the FFN is the decoder that turns them into later residuals or YES/NO.

Frozen FP32, math attention, CPU only. No training checkpoints, activations,
or W&B runs are written. Node-held-out fits never train on the node they score.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "studies/fixed100_layer_probes"
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "src"), str(OLD)]

from cot_navigation.config import load_config  # noqa: E402
from cot_navigation.data.dataset import ParametricWorld  # noqa: E402
from cot_navigation.models.transformer import build_model  # noqa: E402
from cot_navigation.protocols.tokenizer import Tokenizer  # noqa: E402
import linear  # noqa: E402

SEEDS = (17, 31, 47)
PUBLISHED = {
    "simple": dict(head_e=0.012, head_l1=0.975, head_l6=1.0, exact=0.941, ba=0.931,
                   ffn1_nodes_exact=0.348, ffn1_nodes_ba=0.477, ffn1_sep_exact=0.153, ffn1_sep_ba=0.901),
    "hard": dict(head_e=0.012, head_l1=0.518, head_l6=0.656, exact=0.975, ba=0.976,
                 ffn1_nodes_exact=0.814, ffn1_nodes_ba=0.819, ffn1_sep_exact=0.595, ffn1_sep_ba=0.751),
}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, allow_nan=False))
    tmp.replace(path)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return None if np.isnan(value) else value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    return value


def checksum(model) -> float:
    total = 0.0
    with torch.no_grad():
        for parameter in model.parameters():
            total += float(parameter.detach().float().sum())
    return total


def load(setting: str):
    manifest = json.loads((OLD / f"{setting}_direct.json").read_text())["manifest"]
    run = Path(manifest["run"])
    checkpoint = run / "checkpoints/best.pt"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != manifest["checkpoint_sha256"]:
        raise RuntimeError(f"checkpoint hash mismatch for {setting}")
    cfg = load_config([run / "resolved_config.yaml"])
    tokenizer = Tokenizer.from_config(cfg)
    world = ParametricWorld(Path(cfg.data.root) / "main-seed0")
    adjacency_lists = world.adj()
    nodes = np.asarray(world.node_vocab_ids(tokenizer), dtype=np.int64)
    n = len(nodes)
    adjacency = np.zeros((n, n), np.float32)
    reach = np.eye(n, dtype=bool)
    for u, neighbors in enumerate(adjacency_lists):
        for v in neighbors:
            if v <= u:
                raise RuntimeError(f"{setting} edge {u}->{v} is not in index order")
            adjacency[u, v] = 1
    for u in reversed(range(n)):
        for v in adjacency_lists[u]:
            reach[u] |= reach[v]
    for u, neighbors in enumerate(adjacency_lists):
        seen = {u}
        stack = [u]
        while stack:
            v = stack.pop()
            for w in adjacency_lists[v]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        if seen != set(np.flatnonzero(reach[u])):
            raise RuntimeError(f"{setting} reachability DP disagrees with BFS at {u}")
    np.fill_diagonal(reach, False)
    ancestors = reach.T.astype(np.float32)
    successors = adjacency.copy()
    rows = [json.loads(line) for line in gzip.open(OLD / f"{setting}_direct_queries.jsonl.gz", "rt")]
    labels = np.asarray([row["label"] for row in rows], np.int64)
    sources = np.asarray([row["source"] for row in rows], np.int64)
    targets = np.asarray([row["target"] for row in rows], np.int64)
    splits = np.asarray([row["split"] for row in rows], np.int64)
    saved = np.asarray([row["pred"] for row in rows], np.int64)
    if not np.array_equal(reach[sources, targets], labels.astype(bool)):
        raise RuntimeError(f"{setting} query labels disagree with the graph")
    header = [tokenizer.id("<WORLD_0>")] if cfg.train.world_roots else []
    if len(header) != manifest["world_prefix_tokens"]:
        raise RuntimeError("world prefix length disagrees with the manifest")
    prompts = [header + [tokenizer.id("<QUERY>"), int(nodes[s]), int(nodes[t]), tokenizer.id("|")]
               for s, t in zip(sources, targets)]
    tokens = torch.tensor(prompts, dtype=torch.long)
    model = build_model(cfg.model, len(tokenizer), tokenizer.pad).eval().requires_grad_(False)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"])
    model.attn_impl = "math"
    for block in model.blocks:
        block.attn.attn_impl = "math"
    return dict(
        setting=setting, manifest=manifest, model=model, tok=tokenizer, nodes=nodes,
        successors=successors, ancestors=ancestors, rows=rows, labels=labels,
        sources=sources, targets=targets, splits=splits, saved=saved, tokens=tokens,
        header=len(header), checksum=checksum(model), sha256=digest,
    )


def positions(length: int) -> torch.Tensor:
    return torch.arange(length).unsqueeze(0)


def forward(model, tokens: torch.Tensor, *, attn: bool = True, ffn: bool = True,
            skip_ffn=(), keep: torch.Tensor | None = None, linear=None):
    """Residual forward. ``keep`` multiplies the FFN delta; 0 drops that position.

    ``linear`` replaces each FFN with an affine map of its LayerNorm input.
    """
    x = model.embed(tokens)
    pos = positions(tokens.shape[1]).expand(tokens.shape[0], -1)
    for index, block in enumerate(model.blocks):
        if attn:
            delta, _ = block.attn(block.norm1(x), pos)
            x = x + delta
        hidden = block.norm2(x)
        if linear is not None:
            raw = apply_affine(hidden, linear[index])
        else:
            raw = block.ffn(hidden)
        if (not ffn) or index in skip_ffn:
            raw = torch.zeros_like(raw)
        elif keep is not None:
            raw = raw * keep[:, :, None]
        x = x + raw
    return model.lm_head(model.final_norm(x)), x


def isolated(model, node_ids: np.ndarray, *, attn: bool = False, vectors: np.ndarray | None = None):
    if vectors is None:
        tokens = torch.tensor(node_ids, dtype=torch.long)[:, None]
        x = model.embed(tokens)
    else:
        # Input vectors only. The tied output head stays the trained embedding.
        x = torch.tensor(np.asarray(vectors, np.float32))[:, None, :]
    posts = [x[:, 0].detach().cpu().numpy()]
    norms, deltas = [], []
    pos = positions(1).expand(x.shape[0], -1)
    for block in model.blocks:
        if attn:
            delta, _ = block.attn(block.norm1(x), pos)
            x = x + delta
        hidden = block.norm2(x)
        raw = block.ffn(hidden)
        norms.append(hidden[:, 0].detach().cpu().numpy())
        deltas.append(raw[:, 0].detach().cpu().numpy())
        x = x + raw
        posts.append(x[:, 0].detach().cpu().numpy())
    return np.stack(posts, 1), np.stack(norms, 1), np.stack(deltas, 1)


def head_scores(model, states: np.ndarray, node_ids: np.ndarray) -> np.ndarray:
    """states: (n, layers, d) -> node-vocab scores (n, layers, n)."""
    weight = model.final_norm
    out = []
    ids = torch.tensor(node_ids, dtype=torch.long)
    for layer in range(states.shape[1]):
        hidden = torch.tensor(states[:, layer])
        logits = model.lm_head(weight(hidden))
        out.append(logits.index_select(1, ids).detach().cpu().numpy())
    return np.stack(out, 1)


def self_mask(n: int) -> np.ndarray:
    mask = np.ones((n, n), bool)
    np.fill_diagonal(mask, False)
    return mask


def rank_one(labels: np.ndarray, scores: np.ndarray, mask: np.ndarray) -> dict:
    aucs, recalls, hits, chance = [], [], [], []
    for row_labels, row_scores, row_mask in zip(labels, scores, mask):
        y = row_labels[row_mask].astype(bool)
        s = row_scores[row_mask]
        k = int(y.sum())
        if not k or k == len(y):
            continue
        aucs.append(linear.auc(y, s))
        recalls.append(float(y[np.argsort(s)[-k:]].mean()))
        hits.append(float(y[int(np.argmax(s))]))
        chance.append(k / len(y))
    columns = []
    for column in range(labels.shape[1]):
        keep = mask[:, column]
        value = linear.auc(labels[keep, column], scores[keep, column])
        if value is not None:
            columns.append(value)
    return dict(
        n=len(aucs),
        auc=None if not aucs else float(np.mean(aucs)),
        recall_at_true_k=None if not recalls else float(np.mean(recalls)),
        top1_hit=None if not hits else float(np.mean(hits)),
        random_top1=None if not chance else float(np.mean(chance)),
        column_auc=None if not columns else float(np.mean(columns)),
        n_columns=len(columns),
    )


def summarize(rows: list[dict]) -> dict:
    keys = ("auc", "recall_at_true_k", "top1_hit", "random_top1", "column_auc")
    out = {"folds": len(rows), "n": rows[0]["n"] if rows else 0}
    for key in keys:
        values = [row[key] for row in rows if row.get(key) is not None]
        out[key] = None if not values else float(np.mean(values))
        out[key + "_std"] = None if len(values) < 2 else float(np.std(values))
    return out


def ridge_one(x: np.ndarray, y: np.ndarray, train: np.ndarray, val: np.ndarray):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    mu = x[train].mean(0)
    sd = np.maximum(x[train].std(0), 0.05)
    z = (x - mu) / sd
    z = np.concatenate([z, np.ones((len(x), 1))], 1)
    zt = z[train]
    gram = zt.T @ zt / len(train)
    rhs = zt.T @ y[train] / len(train)
    eye = np.eye(gram.shape[0])
    eye[-1, -1] = 0
    best, chosen, penalty, coef = np.inf, None, None, None
    for value in (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0):
        solved = np.linalg.solve(gram + value * eye + 1e-6 * np.eye(gram.shape[0]), rhs)
        pred = z @ solved
        loss = float(np.mean((pred[val] - y[val]) ** 2))
        if loss <= best:
            best, chosen, penalty, coef = loss, pred, value, solved
    return chosen, penalty, dict(mu=mu, sd=sd, coef=coef)


def ridge_predict(x: np.ndarray, y: np.ndarray, train: np.ndarray, val: np.ndarray):
    single = x.ndim == 2
    if single:
        x = x[:, None, :]
    pred = np.zeros((x.shape[0], x.shape[1], y.shape[1]), np.float64)
    for layer in range(x.shape[1]):
        pred[:, layer], _, _ = ridge_one(x[:, layer], y, train, val)
    return pred[:, 0] if single else pred


def residualize(source: np.ndarray, target: np.ndarray, train, val, test):
    prediction, penalty, _ = ridge_one(source, target, train, val)
    residual = target - prediction
    center = target[train].mean(0)
    ss_tot = float(np.sum((target[test] - center) ** 2))
    ss_res = float(np.sum((target[test] - prediction[test]) ** 2))
    r2 = None if ss_tot <= 0 else float(1 - ss_res / ss_tot)
    return prediction, residual, r2, penalty


def make_folds(n: int, seed: int):
    order = np.random.default_rng(seed).permutation(n)
    parts = np.array_split(order, 5)
    folds = []
    for index in range(5):
        test = np.sort(parts[index])
        rest = np.concatenate([parts[j] for j in range(5) if j != index])
        val, train = np.sort(rest[:20]), np.sort(rest[20:])
        if not (len(train) == 60 and len(val) == 20 and len(test) == 20):
            raise RuntimeError("expected 60/20/20 node folds")
        folds.append((train, val, test))
    return folds


def metric_of(labels, scores, mask) -> dict:
    if scores.ndim == 2:
        return rank_one(labels, scores, mask)
    return [rank_one(labels, scores[:, layer], mask) for layer in range(scores.shape[1])]


def crossfit_probe(x, y, folds, mask):
    per_seed = []
    for seed_folds in folds:
        pooled = np.zeros((y.shape[0],) + (() if x.ndim == 2 else (x.shape[1],)) + (y.shape[1],), np.float64)
        if x.ndim == 2:
            pooled = np.zeros((y.shape[0], y.shape[1]), np.float64)
        else:
            pooled = np.zeros((y.shape[0], x.shape[1], y.shape[1]), np.float64)
        for train, val, test in seed_folds:
            pred = ridge_predict(x, y, train, val)
            pooled[test] = pred[test]
        per_seed.append(metric_of(y, pooled, mask))
    if x.ndim == 2:
        return summarize(per_seed)
    layers = []
    for layer in range(x.shape[1]):
        layers.append(summarize([seed[layer] for seed in per_seed]))
    return layers


def crossfit_parts(source, target, labels, folds, mask):
    """Probe the linear image of ``source`` and the leftover of ``target``."""
    linear_seeds, resid_seeds, r2s = [], [], []
    for seed_folds in folds:
        lin = np.zeros_like(labels)
        resid = np.zeros_like(labels)
        fold_r2 = []
        for train, val, test in seed_folds:
            prediction, leftover, r2, _ = residualize(source, target, train, val, test)
            lin[test] = ridge_predict(prediction, labels, train, val)[test]
            resid[test] = ridge_predict(leftover, labels, train, val)[test]
            fold_r2.append(r2)
        linear_seeds.append(rank_one(labels, lin, mask))
        resid_seeds.append(rank_one(labels, resid, mask))
        r2s.append(float(np.mean(fold_r2)))
    return dict(linear_part=summarize(linear_seeds), orthogonal=summarize(resid_seeds),
                test_r2=float(np.mean(r2s)), test_r2_std=float(np.std(r2s)))


def recall_only(labels, scores, mask) -> float | None:
    return rank_one(labels, scores, mask)["recall_at_true_k"]


def fit_mlp(x, y, train, val, mask, kind: str, seed: int):
    torch.manual_seed(seed)
    width = 32 if kind == "bottleneck" else 512
    net = nn.Sequential(nn.Linear(x.shape[1], width, bias=False), nn.GELU(), nn.Linear(width, y.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    xt = torch.tensor(x)
    yt = torch.tensor(y)
    pos = yt[train].mean().clamp(min=1e-4)
    weight = torch.where(yt[train] > 0.5, (1 - pos) / pos, torch.ones_like(yt[train]))
    best_state, best_recall, stall = None, -1.0, 0
    train_recall = None
    for step in range(600):
        opt.zero_grad(set_to_none=True)
        pred = net(xt[train])
        (weight * (pred - yt[train]).square()).mean().backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % 5 == 0 or step == 599:
            with torch.no_grad():
                val_scores = net(xt[val]).numpy()
                train_scores = net(xt[train]).numpy()
            val_recall = recall_only(y[val], val_scores, mask[val]) or 0.0
            train_recall = recall_only(y[train], train_scores, mask[train])
            if val_recall > best_recall + 1e-4:
                best_recall = val_recall
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                stall = 0
            else:
                stall += 1
            if step > 80 and stall >= 12:
                break
    net.load_state_dict(best_state)
    with torch.no_grad():
        scores = net(xt).numpy()
    return scores, train_recall


def crossfit_decoder(x, y, folds, mask, kind: str):
    per_seed, train_recalls = [], []
    for seed, seed_folds in zip(SEEDS, folds):
        pooled = np.zeros_like(y)
        fold_train = []
        for fold_index, (train, val, test) in enumerate(seed_folds):
            scores, train_recall = fit_mlp(x, y, train, val, mask, kind, seed + 17 * fold_index)
            pooled[test] = scores[test]
            fold_train.append(train_recall)
        per_seed.append(rank_one(y, pooled, mask))
        train_recalls.append(float(np.nanmean(fold_train)))
    out = summarize(per_seed)
    out["train_recall_at_true_k"] = float(np.mean(train_recalls))
    return out


def knn_scores(x, y, gallery, test, k: int):
    base = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)
    sims = base[test] @ base[gallery].T
    kk = min(k, len(gallery))
    nearest = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
    out = np.zeros((len(test), y.shape[1]), np.float64)
    table = y[gallery]
    for row, cols in enumerate(nearest):
        weights = np.maximum(sims[row, cols], 0)
        weights = weights / weights.sum() if weights.sum() else np.full(kk, 1 / kk)
        out[row] = weights @ table[cols]
    return out


def crossfit_knn(x, y, folds, mask, k: int):
    per_seed = []
    for seed_folds in folds:
        pooled = np.zeros_like(y)
        for train, val, test in seed_folds:
            gallery = np.concatenate([train, val])
            pooled[test] = knn_scores(x, y, gallery, test, k)
        per_seed.append(rank_one(y, pooled, mask))
    return summarize(per_seed)


def answer_metrics(logits: np.ndarray, labels: np.ndarray, yes: int, no: int) -> dict:
    pred = logits.argmax(1)
    desired = np.where(labels == 1, yes, no)
    exact = pred == desired
    constrained = (logits[:, yes] > logits[:, no]) == (labels == 1)
    valid = (pred == yes) | (pred == no)

    def bal(correct):
        return float(0.5 * (correct[labels == 1].mean() + correct[labels == 0].mean()))

    return dict(
        n=int(len(labels)), n_yes=int((labels == 1).sum()), n_no=int((labels == 0).sum()),
        accuracy=float(exact.mean()), balanced_accuracy=bal(exact),
        constrained_accuracy=float(constrained.mean()), constrained_balanced_accuracy=bal(constrained),
        yes_recall=float(constrained[labels == 1].mean()), no_recall=float(constrained[labels == 0].mean()),
        output_is_yes_no=float(valid.mean()),
    )


def batched_logits(model, tokens, idx, **kwargs):
    chunks = []
    keep = kwargs.get("keep")
    for start in range(0, len(idx), 256):
        sl = idx[start:start + 256]
        local = dict(kwargs)
        if keep is not None:
            local["keep"] = keep[sl]
        logits, _ = forward(model, tokens[sl], **local)
        chunks.append(logits[:, -1].detach().cpu().numpy())
    return np.concatenate(chunks)


def sublayer_states(model, tokens: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Separator residual after embedding and after each attention and FFN."""
    x = model.embed(tokens)
    pos = positions(tokens.shape[1]).expand(tokens.shape[0], -1)
    saved = [x[:, -1].detach().cpu().numpy()]
    for block in model.blocks:
        delta, _ = block.attn(block.norm1(x), pos)
        x = x + delta
        saved.append(x[:, -1].detach().cpu().numpy())
        x = x + block.ffn(block.norm2(x))
        saved.append(x[:, -1].detach().cpu().numpy())
    states = np.stack(saved, 1)
    logits, _ = forward(model, tokens)
    return states, logits[:, -1].detach().cpu().numpy()


def binary_probe(x, y, train, val, test) -> dict:
    scores, _, _ = ridge_one(x, y[:, None].astype(np.float64), train, val)
    scores = scores[:, 0]

    def stats(split, threshold):
        yy = y[split] == 1
        pred = scores[split] >= threshold
        yes = float(pred[yy].mean()) if yy.any() else None
        no = float((~pred[~yy]).mean()) if (~yy).any() else None
        return yes, no, None if yes is None or no is None else 0.5 * (yes + no)

    grid = np.unique(np.quantile(scores[val], np.linspace(0, 1, 41)))
    best_threshold, best_ba = 0.0, -1.0
    for threshold in grid:
        _, _, ba = stats(val, threshold)
        if ba is not None and ba >= best_ba:
            best_ba, best_threshold = ba, float(threshold)
    yes, no, ba = stats(test, best_threshold)
    return dict(balanced_accuracy=ba, yes_recall=yes, no_recall=no,
                auc=linear.auc(y[test], scores[test]), threshold=best_threshold)


def fit_query_linear(model, tokens, train, val):
    """One affine replacement of each FFN, fit on train-query positions."""
    maps, fit = [], []
    x = model.embed(tokens)
    pos = positions(tokens.shape[1]).expand(tokens.shape[0], -1)
    length = tokens.shape[1]
    train_flat = (train[:, None] * length + np.arange(length)).ravel()
    val_flat = (val[:, None] * length + np.arange(length)).ravel()
    for block in model.blocks:
        delta, _ = block.attn(block.norm1(x), pos)
        x = x + delta
        hidden = block.norm2(x).detach().cpu().numpy().reshape(-1, model.d_model)
        raw = block.ffn(block.norm2(x)).detach().cpu().numpy().reshape(-1, model.d_model)
        # Recompute norm from the current residual without a second attention pass.
        _, _, params = ridge_one(hidden, raw, train_flat, val_flat)
        maps.append(params)
        prediction = ((hidden - params["mu"]) / params["sd"])
        prediction = np.concatenate([prediction, np.ones((len(hidden), 1))], 1) @ params["coef"]
        center = raw[train_flat].mean(0)
        tot = float(np.sum((raw[val_flat] - center) ** 2))
        res = float(np.sum((raw[val_flat] - prediction[val_flat]) ** 2))
        fit.append(None if tot <= 0 else float(1 - res / tot))
        x = x + block.ffn(block.norm2(x))
    return [{k: v.astype(np.float32) for k, v in item.items()} for item in maps], fit


def apply_affine(hidden: torch.Tensor, params) -> torch.Tensor:
    mu = torch.tensor(params["mu"], device=hidden.device, dtype=hidden.dtype)
    sd = torch.tensor(params["sd"], device=hidden.device, dtype=hidden.dtype)
    coef = torch.tensor(params["coef"], device=hidden.device, dtype=hidden.dtype)
    z = (hidden - mu) / sd
    z = torch.cat([z, torch.ones(*hidden.shape[:-1], 1, device=hidden.device, dtype=hidden.dtype)], -1)
    return z @ coef


def fit_node_linear(norms, deltas, folds):
    """Affine FFN replacement fit on isolated nodes, using every node once as test for R^2."""
    maps = []
    r2 = []
    tr0, va0, te0 = folds[0][0]
    fit_idx = np.concatenate([tr0, te0])
    for layer in range(norms.shape[1]):
        _, _, params = ridge_one(norms[:, layer], deltas[:, layer], fit_idx, va0)
        maps.append({k: v.astype(np.float32) for k, v in params.items()})
        seed_r2 = []
        for seed_folds in folds:
            fold_r2 = []
            for tr, va, te in seed_folds:
                _, _, value, _ = residualize(norms[:, layer], deltas[:, layer], tr, va, te)
                fold_r2.append(value)
            seed_r2.append(float(np.mean(fold_r2)))
        r2.append(dict(mean=float(np.mean(seed_r2)), std=float(np.std(seed_r2))))
    return maps, r2


def keep_mask(tokens: torch.Tensor, drop: np.ndarray) -> torch.Tensor:
    mask = np.ones(tokens.shape, np.float32)
    mask[:, drop] = 0
    return torch.tensor(mask)


def relation_bundle(x, labels, folds, mask, name: str):
    print(f"  probe {name}", flush=True)
    out = {
        "linear": crossfit_probe(x, labels, folds, mask),
        "knn5": crossfit_knn(x, labels, folds, mask, 5),
        "knn1": crossfit_knn(x, labels, folds, mask, 1),
        "bottleneck": crossfit_decoder(x, labels, folds, mask, "bottleneck"),
        "wide": crossfit_decoder(x, labels, folds, mask, "wide"),
    }
    shuffled = labels[np.random.default_rng(0).permutation(len(labels))]
    out["shuffled_label_linear"] = crossfit_probe(x, shuffled, folds[:1], mask)
    random = np.random.default_rng(1).normal(size=x.shape).astype(np.float32)
    random *= x.std() / random.std()
    out["random_vector_linear"] = crossfit_probe(random, labels, folds[:1], mask)
    return out


def with_random_ffn(model, seed: int):
    saved = [(p, p.detach().clone()) for name, p in model.named_parameters() if ".ffn." in name]
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter, _ in saved:
            parameter.normal_(0, 0.02, generator=generator)
    return saved


def restore(saved) -> None:
    with torch.no_grad():
        for parameter, old in saved:
            parameter.copy_(old)


def run(setting: str) -> dict:
    print(f"== {setting}", flush=True)
    pack = load(setting)
    model, tokens, nodes = pack["model"], pack["tokens"], pack["nodes"]
    n = len(nodes)
    mask = self_mask(n)
    reference, _ = forward(model, tokens[:32])
    native = model(tokens[:32])
    error = float((reference - native).abs().max())
    if error > 1e-4:
        raise RuntimeError(f"custom forward disagrees with the model by {error}")
    folds = [make_folds(n, seed) for seed in SEEDS]
    posts, norms, deltas = isolated(model, nodes, attn=False)
    trained_head = head_scores(model, posts, nodes)
    random_heads = []
    for seed in SEEDS:
        saved = with_random_ffn(model, seed)
        random_posts, _, _ = isolated(model, nodes, attn=False)
        random_heads.append(head_scores(model, random_posts, nodes))
        restore(saved)
    if abs(checksum(model) - pack["checksum"]) > 0:
        raise RuntimeError("random FFN was not restored")
    # Embedding permutation must carry the native-head set with the vector.
    base_vectors = model.embed.weight.detach().cpu().numpy()[nodes]
    perm = np.random.default_rng(9).permutation(n)
    moved_posts, _, _ = isolated(model, nodes, attn=False, vectors=base_vectors[perm])
    moved_head = head_scores(model, moved_posts, nodes)
    donor_mask = np.ones((n, n), bool)
    donor_mask[np.arange(n), perm] = False
    donor = pack["successors"][perm]
    own = pack["successors"]
    donor_l1 = rank_one(donor, moved_head[:, 1], donor_mask)
    own_l1 = rank_one(own, moved_head[:, 1], mask)
    base_l1 = rank_one(own, trained_head[:, 1], mask)
    if abs((donor_l1["recall_at_true_k"] or 0) - (base_l1["recall_at_true_k"] or 0)) > 1e-4:
        raise RuntimeError(
            "permuting input vectors did not move the successor set "
            f"({donor_l1['recall_at_true_k']} vs {base_l1['recall_at_true_k']})"
        )
    noise = np.random.default_rng(11).normal(0, float(base_vectors.std()), base_vectors.shape).astype(np.float32)
    noise_posts, _, _ = isolated(model, nodes, attn=False, vectors=noise)
    noise_head = head_scores(model, noise_posts, nodes)

    relations = {}
    part_metrics = {}
    for name, labels in (("successors", pack["successors"]), ("ancestors", pack["ancestors"])):
        relations[name] = relation_bundle(posts[:, 0], labels, folds, mask, name)
        relations[name]["native_head"] = [rank_one(labels, trained_head[:, layer], mask) for layer in range(trained_head.shape[1])]
        relations[name]["random_ffn_head"] = [
            summarize([rank_one(labels, head[:, layer], mask) for head in random_heads])
            for layer in range(trained_head.shape[1])
        ]
        relations[name]["post_residual_probe"] = crossfit_probe(posts, labels, folds, mask)
        relations[name]["ffn_delta_probe"] = crossfit_probe(deltas, labels, folds, mask)
        parts = []
        for layer in range(deltas.shape[1]):
            print(f"  orthogonal {name} L{layer + 1}", flush=True)
            parts.append(crossfit_parts(norms[:, layer], deltas[:, layer], labels, folds, mask))
        part_metrics[name] = parts
        final_parts = []
        for seed_folds in folds:
            lin = np.zeros_like(labels)
            resid = np.zeros_like(labels)
            for train, val, test in seed_folds:
                prediction, leftover, _, _ = residualize(posts[:, 0], posts[:, -1], train, val, test)
                lin[test] = ridge_predict(prediction, labels, train, val)[test]
                resid[test] = ridge_predict(leftover, labels, train, val)[test]
            final_parts.append((rank_one(labels, lin, mask), rank_one(labels, resid, mask)))
        relations[name]["final_linear_in_embedding"] = summarize([item[0] for item in final_parts])
        relations[name]["final_beyond_embedding"] = summarize([item[1] for item in final_parts])
        relations[name]["embedding_permutation"] = dict(
            donor_set_l1=donor_l1 if name == "successors" else rank_one(pack["ancestors"][perm], moved_head[:, 1], donor_mask),
            original_identity_l1=own_l1 if name == "successors" else rank_one(labels, moved_head[:, 1], mask),
            random_embedding_l1=rank_one(labels, noise_head[:, 1], mask),
            random_embedding_l6=rank_one(labels, noise_head[:, -1], mask),
        )

    centered = posts[:, 0] - posts[:, 0].mean(0)
    cosine = centered / np.clip(np.linalg.norm(centered, axis=1, keepdims=True), 1e-8, None)
    off = cosine @ cosine.T
    off = off[~np.eye(n, dtype=bool)]

    yes, no = pack["tok"].yes, pack["tok"].no
    test = np.flatnonzero(pack["splits"] == 2)
    train = np.flatnonzero(pack["splits"] == 0)
    val = np.flatnonzero(pack["splits"] == 1)
    labels = pack["labels"]
    src = pack["header"] + 1
    tgt = pack["header"] + 2
    sep = pack["header"] + 3
    conditions = {
        "intact": {},
        "attention_off": dict(attn=False),
        "ffn_off": dict(ffn=False),
        "ffn_off_nodes": dict(keep=keep_mask(tokens, np.array([src, tgt]))),
        "ffn_off_source": dict(keep=keep_mask(tokens, np.array([src]))),
        "ffn_off_target": dict(keep=keep_mask(tokens, np.array([tgt]))),
        "ffn_off_separator": dict(keep=keep_mask(tokens, np.array([sep]))),
        "ffn1_off_nodes": dict(skip_ffn=(0,), keep=None),
    }
    # FFN1-only at nodes/separator matches the published scope: zero layer 0 at those positions.
    # skip_ffn would zero the whole layer. Build position masks and restrict them to layer 0
    # by running a dedicated forward below.
    ablations = {}
    for name, kwargs in conditions.items():
        if name == "ffn1_off_nodes":
            continue
        print(f"  ablation {name}", flush=True)
        ablations[name] = answer_metrics(batched_logits(model, tokens, test, **kwargs), labels[test], yes, no)
    ablations["ffn1_off_nodes"] = answer_metrics(
        layer_position_logits(model, tokens[test], layer=0, drop=(src, tgt)), labels[test], yes, no)
    ablations["ffn1_off_separator"] = answer_metrics(
        layer_position_logits(model, tokens[test], layer=0, drop=(sep,)), labels[test], yes, no)
    layer_knockout = []
    for layer in range(model.n_layers):
        layer_knockout.append(answer_metrics(
            batched_logits(model, tokens, test, skip_ffn=(layer,)), labels[test], yes, no))
    attn_ba = ablations["attention_off"]["constrained_balanced_accuracy"]
    if not 0.45 <= attn_ba <= 0.55:
        raise RuntimeError(f"attention-off separator still depends on the query: BA {attn_ba}")

    print("  linear FFN replacements", flush=True)
    query_maps, query_r2 = fit_query_linear(model, tokens, train, val)
    node_maps, node_r2 = fit_node_linear(norms, deltas, folds)
    ablations["linear_ffn_fit_on_train_queries"] = answer_metrics(
        batched_logits(model, tokens, test, linear=query_maps), labels[test], yes, no)
    ablations["linear_ffn_fit_on_nodes"] = answer_metrics(
        batched_logits(model, tokens, test, linear=node_maps), labels[test], yes, no)

    print("  sublayer readouts", flush=True)
    states, final_logits = sublayer_states(model, tokens)
    # The dedicated intact forward must agree with the ablation forward.
    intact = answer_metrics(final_logits[test], labels[test], yes, no)
    if abs(intact["accuracy"] - ablations["intact"]["accuracy"]) > 1e-6:
        raise RuntimeError("two intact forwards disagree")
    names = ["E"]
    for layer in range(1, model.n_layers + 1):
        names += [f"L{layer}attn", f"L{layer}ffn"]
    lens, probes = [], []
    y = labels.astype(np.float64)
    for index, name in enumerate(names):
        margin_logits = model.lm_head(model.final_norm(torch.tensor(states[test, index]))).detach().cpu().numpy()
        lens.append(dict(name=name, **answer_metrics(margin_logits, labels[test], yes, no)))
        probes.append(dict(name=name, **binary_probe(states[:, index], y, train, val, test)))
    if not 0.45 <= lens[0]["constrained_balanced_accuracy"] <= 0.55:
        raise RuntimeError("separator embedding logit lens depends on the query")
    embed = posts[:, 0]
    pair = np.concatenate([embed[pack["sources"]], embed[pack["targets"]]], 1)
    pair_probe = binary_probe(pair, y, train, val, test)
    pair_net = train_pair_mlp(pair, y, train, val, test)

    saved_test = pack["saved"][test]
    predicted = np.where(final_logits[test, yes] > final_logits[test, no], 1, 0)
    # Agreement uses the constrained bit only when the saved file stored 0/1.
    agreement = float((predicted == saved_test).mean())
    published = PUBLISHED[setting]
    head = relations["successors"]["native_head"]
    replication = dict(
        head_e=head[0]["recall_at_true_k"], head_l1=head[1]["recall_at_true_k"], head_l6=head[-1]["recall_at_true_k"],
        exact=ablations["intact"]["accuracy"], ba=ablations["intact"]["constrained_balanced_accuracy"],
        ffn1_nodes_exact=ablations["ffn1_off_nodes"]["accuracy"],
        ffn1_nodes_ba=ablations["ffn1_off_nodes"]["constrained_balanced_accuracy"],
        ffn1_sep_exact=ablations["ffn1_off_separator"]["accuracy"],
        ffn1_sep_ba=ablations["ffn1_off_separator"]["constrained_balanced_accuracy"],
        published=published,
        saved_prediction_agreement=agreement,
        forward_max_abs_error=error,
    )
    result = dict(
        setting=setting, sha256=pack["sha256"], step=pack["manifest"]["checkpoint_step"],
        n_nodes=n, n_queries=int(len(labels)), test_queries=int(len(test)),
        world_prefix_tokens=pack["header"], layers=["E"] + [f"L{i}" for i in range(1, model.n_layers + 1)],
        embedding_offdiag_cosine=dict(mean=float(off.mean()), max=float(off.max()), min=float(off.min())),
        replication=replication, relations=relations, ffn_parts=part_metrics,
        ablations=ablations, layer_ffn_knockout=layer_knockout,
        query_linear_ffn_val_r2=query_r2, node_linear_ffn_test_r2=node_r2,
        sublayer_names=names, logit_lens=lens, answer_probe=probes,
        endpoint_embedding_probe=pair_probe, endpoint_embedding_mlp=pair_net,
        checks=dict(attention_off_ba=attn_ba, separator_embedding_lens_ba=lens[0]["constrained_balanced_accuracy"]),
    )
    if abs(checksum(model) - pack["checksum"]) > 0:
        raise RuntimeError("model parameters changed")
    write_json(HERE / f"{setting}.json", result)
    print(f"  wrote {setting}", flush=True)
    return result


def layer_position_logits(model, tokens, layer: int, drop: tuple[int, ...]) -> np.ndarray:
    x = model.embed(tokens)
    pos = positions(tokens.shape[1]).expand(tokens.shape[0], -1)
    for index, block in enumerate(model.blocks):
        delta, _ = block.attn(block.norm1(x), pos)
        x = x + delta
        raw = block.ffn(block.norm2(x))
        if index == layer:
            raw = raw.clone()
            for column in drop:
                raw[:, column] = 0
        x = x + raw
    return model.lm_head(model.final_norm(x))[:, -1].detach().cpu().numpy()


def train_pair_mlp(x, y, train, val, test) -> dict:
    torch.manual_seed(5)
    net = nn.Sequential(nn.Linear(x.shape[1], 32, bias=False), nn.GELU(), nn.Linear(32, 1))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    xt = torch.tensor(np.asarray(x, np.float32))
    yt = torch.tensor(np.asarray(y, np.float32))[:, None]
    best, best_auc, stall = None, -1.0, 0
    for step in range(400):
        opt.zero_grad(set_to_none=True)
        nn.functional.mse_loss(net(xt[train]), yt[train]).backward()
        opt.step()
        if step % 5 == 0:
            with torch.no_grad():
                scores = net(xt[val])[:, 0].numpy()
            auc = linear.auc(y[val], scores) or 0
            if auc > best_auc + 1e-4:
                best_auc = auc
                best = {k: v.detach().clone() for k, v in net.state_dict().items()}
                stall = 0
            else:
                stall += 1
            if step > 40 and stall >= 12:
                break
    net.load_state_dict(best)
    with torch.no_grad():
        scores = net(xt)[:, 0].numpy()
    grid = np.unique(np.quantile(scores[val], np.linspace(0, 1, 41)))
    best_ba, threshold = -1.0, 0.0
    for value in grid:
        pred = scores[val] >= value
        yy = y[val] == 1
        ba = 0.5 * (pred[yy].mean() + (~pred[~yy]).mean())
        if ba >= best_ba:
            best_ba, threshold = float(ba), float(value)
    pred = scores[test] >= threshold
    yy = y[test] == 1
    return dict(
        balanced_accuracy=float(0.5 * (pred[yy].mean() + (~pred[~yy]).mean())),
        auc=linear.auc(y[test], scores[test]), threshold=threshold,
    )


def take(metric, key):
    if metric is None:
        return None
    return metric.get(key)


def plot(results: list[dict]) -> None:
    plt.rcParams.update({
        "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "figure.facecolor": "white",
    })
    fig, axes = plt.subplots(2, 4, figsize=(14.2, 6.6), constrained_layout=True)
    layer_x = np.arange(7)
    for row, result in enumerate(results):
        succ = result["relations"]["successors"]
        anc = result["relations"]["ancestors"]
        ax = axes[row, 0]
        ax.plot(layer_x, [z["recall_at_true_k"] for z in succ["native_head"]], "o-", color="black", label="Native head, attention off")
        ax.plot(layer_x, [z["recall_at_true_k"] for z in succ["random_ffn_head"]], "s--", color="0.45", label="Native head, random FFN")
        ax.plot(layer_x, [z["recall_at_true_k"] for z in succ["post_residual_probe"]], "o-", color="tab:blue", label="Probe of residual")
        orth = [np.nan] + [z["orthogonal"]["recall_at_true_k"] for z in result["ffn_parts"]["successors"]]
        ax.plot(layer_x, orth, "^-", color="tab:orange", label="Probe of FFN residual orthogonal to its input")
        ax.axhline(succ["linear"]["recall_at_true_k"], color="tab:green", ls=":", label="Linear probe of embedding")
        ax.axhline(succ["wide"]["recall_at_true_k"], color="tab:purple", ls=":", label="Wide MLP on embedding")
        ax.set(ylim=(-0.03, 1.05), xticks=layer_x, xticklabels=result["layers"], ylabel="Successor recall@k",
               title=f"{result['setting']}: successor sets")
        ax = axes[row, 1]
        ax.plot(layer_x, [z["column_auc"] for z in anc["native_head"]], "o-", color="black", label="Native head, attention off")
        ax.plot(layer_x, [z["column_auc"] for z in anc["random_ffn_head"]], "s--", color="0.45", label="Native head, random FFN")
        ax.plot(layer_x, [z["column_auc"] for z in anc["post_residual_probe"]], "o-", color="tab:blue", label="Probe of residual")
        orth = [np.nan] + [z["orthogonal"]["column_auc"] for z in result["ffn_parts"]["ancestors"]]
        ax.plot(layer_x, orth, "^-", color="tab:orange", label="Orthogonal FFN residual")
        ax.axhline(anc["linear"]["column_auc"], color="tab:green", ls=":", label="Linear probe of embedding")
        ax.axhline(0.5, color="black", lw=0.6, ls=":")
        ax.set(ylim=(0.30, 1.02), xticks=layer_x, xticklabels=result["layers"], ylabel="Ancestor column AUROC",
               title=f"{result['setting']}: ancestor sets")
        ax = axes[row, 2]
        keys = [
            ("intact", "Intact"), ("attention_off", "No attention"), ("ffn_off", "No FFN"),
            ("ffn_off_source", "FFN off at source"), ("ffn_off_target", "FFN off at target"),
            ("ffn_off_separator", "FFN off at separator"),
            ("linear_ffn_fit_on_train_queries", "Linear FFN"),
        ]
        vals = [result["ablations"][key]["constrained_balanced_accuracy"] for key, _ in keys]
        ax.bar(np.arange(len(keys)), vals, color=["black", "0.65", "tab:orange", "tab:cyan", "tab:blue", "tab:red", "tab:green"])
        ax.axhline(0.5, color="black", lw=0.6, ls=":")
        ax.set(ylim=(0, 1.05), ylabel="Constrained YES/NO BA", title=f"{result['setting']}: answer ablations")
        ax.set_xticks(np.arange(len(keys)), [label for _, label in keys], rotation=35, ha="right")
        ax = axes[row, 3]
        knock = [z["constrained_balanced_accuracy"] for z in result["layer_ffn_knockout"]]
        ax.plot(np.arange(1, 7), knock, "o-", color="tab:red", label="One FFN layer removed")
        ax.axhline(result["ablations"]["intact"]["constrained_balanced_accuracy"], color="black", ls="--", label="Intact")
        ax.axhline(0.5, color="black", lw=0.6, ls=":")
        ax.set(ylim=(0, 1.05), xticks=np.arange(1, 7), xlabel="Removed FFN layer", ylabel="Constrained YES/NO BA",
               title=f"{result['setting']}: single-layer FFN knockout")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.42, 1.08))
    fig.savefig(HERE / "ffn_role.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    torch.set_num_threads(4)
    results = [run(setting) for setting in ("simple", "hard")]
    plot(results)
    print("figure", HERE / "ffn_role.png", flush=True)


if __name__ == "__main__":
    main()

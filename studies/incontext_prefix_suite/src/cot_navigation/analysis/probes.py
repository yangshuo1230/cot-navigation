"""Frozen-residual linear probes, query-disjoint splits and negative controls.

One affine classifier per late layer, shared over intermediate positions.
The fourth classifier uses ONLY the current token/node embedding and simple
observable position/goal-occurrence features. Probe labels are never input to
the frozen transformer. Reference and freely generated prefixes are reported
separately. Features stay in memory and probe weights are not persisted.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .common import SEED, anchors


def extract(am, examples, free_sequences=None, batch_size=12):
    stores = {name: {"x": [], "y": [], "mask": [], "query": [], "stage": [], "goal_seen": []}
              for name in ("answer", "successors", "visited")}
    for begin in range(0, len(examples), batch_size):
        chunk = examples[begin:begin+batch_size]
        seqs = ([free_sequences[ex.index] for ex in chunk] if free_sequences is not None else
                [ex.row.tokens[:ex.row.answer_index] for ex in chunk])
        _, resid, _ = am.forward(seqs)
        for b, (ex, tokens) in enumerate(zip(chunk, seqs)):
            ans, structural = anchors(tokens, ex, am.protocol, am.tok)
            def feature(pos, current=None):
                h = torch.stack([resid[layer][b, pos].float() for layer in am.layers])
                # Activation classifiers are affine in h only. Pad zeros to
                # align dimensions with the deliberately stronger input control.
                h = F.pad(h, (0, 4))
                token = tokens[pos] if current is None else current
                embedding = am.model.embed.weight[token].float()
                seen = ex.nodes[ex.target] in tokens[ex.row.prompt_len:pos+1]
                observable = embedding.new_tensor([pos/4096, (pos-ex.row.prompt_len)/4096,
                                                   float(seen), float(token == ex.nodes[ex.target])])
                baseline = torch.cat([embedding, observable])
                return torch.cat([h, baseline[None]], 0).cpu().numpy(), seen
            for stage, pos in ans:
                x, seen = feature(pos)
                s = stores["answer"]
                for key, value in dict(x=x, y=[ex.label], mask=[1.], query=ex.index,
                                       stage=stage, goal_seen=seen).items():
                    s[key].append(value)
            inverse = {token: i for i, token in enumerate(ex.nodes)}
            for item in structural:
                successors = [ex.nodes[v] for v in ex.adj[inverse[item["current"]]]]
                for task, positives in [("successors", successors), ("visited", item["visited"])]:
                    if task == "visited" and am.protocol in {"direct", "filler", "path"}:
                        continue  # a path history is not a DFS/BFS visited set
                    pos = item["successor_pos"] if task == "successors" and "successor_pos" in item else item["pos"]
                    x, seen = feature(pos, item["current"])
                    y = np.zeros(am.tok.n_node_tokens, dtype=np.float32)
                    mask = np.zeros_like(y)
                    for token in ex.nodes:
                        mask[token-am.tok.node_lo] = 1
                    for token in positives:
                        y[token-am.tok.node_lo] = 1
                    s = stores[task]
                    for key, value in dict(x=x, y=y, mask=mask, query=ex.index,
                                           stage="pre_candidates" if task == "successors" and "successor_pos" in item else item["stage"],
                                           goal_seen=seen).items():
                        s[key].append(value)
        del resid
    result = {}
    for name, s in stores.items():
        if s["x"]:
            result[name] = {k: np.asarray(v) for k, v in s.items()}
    return result


def metrics(logits, y, mask):
    logits, y, mask = np.asarray(logits), np.asarray(y), np.asarray(mask).astype(bool)
    pred, truth = logits >= 0, y >= .5
    tp = int((pred & truth & mask).sum())
    fp = int((pred & ~truth & mask).sum())
    tn = int((~pred & ~truth & mask).sum())
    fn = int((~pred & truth & mask).sum())
    tpr, tnr = tp/max(1, tp+fn), tn/max(1, tn+fp)
    out = {"accuracy": (tp+tn)/max(1, tp+fp+tn+fn), "f1": 2*tp/max(1, 2*tp+fp+fn),
           "balanced_accuracy": ((tpr+tnr)/2 if tp+fn > 0 and tn+fp > 0 else None), "positive_recall": tpr,
           "precision": tp/max(1, tp+fp), "n_rows": int(len(y)), "n_positive": tp+fn,
           "tp": tp, "fp": fp, "tn": tn, "fn": fn}
    if y.shape[-1] > 1:
        out["exact_set_accuracy"] = float(((pred == truth) | ~mask).all(-1).mean())
    return out


def shuffled_labels(data, seed):
    """Break query-label association, preserving all anchors of each query.

    Multilabel controls permute candidate identities WITHIN each graph/row,
    preserving positive count and graph-membership masks.
    """
    rng = np.random.default_rng(seed)
    y = data["y"].copy()
    if y.shape[-1] == 1:
        queries = np.unique(data["query"])
        labels = [y[np.flatnonzero(data["query"] == q)[0], 0] for q in queries]
        random = dict(zip(queries, rng.permutation(labels)))
        for i, q in enumerate(data["query"]):
            y[i, 0] = random[q]
    else:
        for i in range(len(y)):
            valid = np.flatnonzero(data["mask"][i])
            y[i, valid] = rng.permutation(y[i, valid])
    return y


def fit(train, val, *, device, seed, steps=250, penalty=.001, shuffled=False):
    """Fit all layer classifiers together; select each using validation BCE."""
    torch.manual_seed(seed)
    x = torch.tensor(train["x"], device=device, dtype=torch.float32)
    v = torch.tensor(val["x"], device=device, dtype=torch.float32)
    mean, scale = x.mean(0), x.std(0).clamp(min=.05)
    x, v = (x-mean)/scale, (v-mean)/scale
    y_np = shuffled_labels(train, seed) if shuffled else train["y"]
    y = torch.tensor(y_np, device=device, dtype=torch.float32)[:, None]
    mask = torch.tensor(train["mask"], device=device, dtype=torch.float32)[:, None]
    vy_np = shuffled_labels(val, seed+101) if shuffled else val["y"]
    vy = torch.tensor(vy_np, device=device, dtype=torch.float32)[:, None]
    vm = torch.tensor(val["mask"], device=device, dtype=torch.float32)[:, None]
    # Queries, not long trajectories, receive equal total training weight.
    unique, counts = np.unique(train["query"], return_counts=True)
    freq = dict(zip(unique, counts))
    row_weight = torch.tensor([1/freq[q] for q in train["query"]], device=device)[:, None, None]
    positive = (y*mask*row_weight).sum()
    negative = ((1-y)*mask*row_weight).sum()
    pos_weight = (negative/positive.clamp(min=1)).clamp(.2, 20).detach()
    layers, dim, outputs = x.shape[1], x.shape[2], y.shape[-1]
    weight = torch.nn.Parameter(torch.zeros(layers, dim, outputs, device=device))
    bias = torch.nn.Parameter(torch.zeros(layers, outputs, device=device))
    optimizer = torch.optim.AdamW([weight, bias], lr=.03, weight_decay=penalty)
    best_loss = torch.full((layers,), float("inf"), device=device)
    best_w, best_b, epochs = weight.detach().clone(), bias.detach().clone(), [0]*layers
    denom = (mask*row_weight).sum().clamp(min=1)
    for step in range(steps):
        logits = torch.einsum("nld,ldk->nlk", x, weight)+bias
        loss = (F.binary_cross_entropy_with_logits(logits, y.expand_as(logits), reduction="none",
                                                   pos_weight=pos_weight)*mask*row_weight).sum()/denom/layers
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 10 == 0 or step == steps-1:
            with torch.no_grad():
                vl = torch.einsum("nld,ldk->nlk", v, weight)+bias
                scores = (F.binary_cross_entropy_with_logits(vl, vy.expand_as(vl), reduction="none",
                                                             pos_weight=pos_weight)*vm).sum((0, 2))/vm.sum().clamp(min=1)
                better = scores < best_loss
                best_loss = torch.minimum(best_loss, scores)
                best_w[better], best_b[better] = weight.detach()[better], bias.detach()[better]
                for layer in torch.nonzero(better).flatten().tolist():
                    epochs[layer] = step+1
    return dict(weight=best_w, bias=best_b, mean=mean, scale=scale,
                val_bce=best_loss.cpu().tolist(), steps=epochs, positive_weight=float(pos_weight))


def evaluate(fitted, data, *, shuffled=False, seed=0):
    x = torch.tensor(data["x"], device=fitted["weight"].device, dtype=torch.float32)
    with torch.no_grad():
        logits = (torch.einsum("nld,ldk->nlk", (x-fitted["mean"])/fitted["scale"],
                              fitted["weight"])+fitted["bias"]).cpu().numpy()
    labels = shuffled_labels(data, seed+202) if shuffled else data["y"]
    result = []
    for layer in range(logits.shape[1]):
        entry = metrics(logits[:, layer], labels, data["mask"])
        entry["by_stage"] = {}
        for stage in np.unique(data["stage"]):
            sel = data["stage"] == stage
            entry["by_stage"][str(stage)] = metrics(logits[sel, layer], labels[sel], data["mask"][sel])
        if labels.shape[-1] == 1:
            sel = ~data["goal_seen"]
            if sel.any():
                entry["goal_not_yet_in_cot"] = metrics(logits[sel, layer], labels[sel], data["mask"][sel])
        result.append(entry)
    return result


def subset(data, selector):
    return {key: value[selector] for key, value in data.items()}


def run_probes(am, train, val, test, free_test, *, steps=250):
    train_features, val_features = extract(am, train), extract(am, val)
    test_features = extract(am, test)
    free_features = extract(am, test, free_sequences=free_test)
    result = {"layers": [i+1 for i in am.layers]+["input_control"],
              "training": "reference_prefixes", "steps_max": steps, "seed": SEED,
              "n_queries": {"train": len(train), "val": len(val), "test": len(test)},
              "query_ids": {"train": [x.index for x in train], "val": [x.index for x in val],
                            "test": [x.index for x in test]}, "targets": {}}
    for task, train_data in train_features.items():
        if task not in val_features or task not in test_features:
            result["targets"][task] = {"status": "insufficient held-out structural anchors"}
            continue
        fitted = fit(train_data, val_features[task], device=am.device, seed=SEED, steps=steps)
        control = fit(train_data, val_features[task], device=am.device, seed=SEED+17,
                      steps=steps, shuffled=True)
        item = {"train": evaluate(fitted, train_data), "reference_test": evaluate(fitted, test_features[task]),
                "shuffled_control_test": evaluate(control, test_features[task], shuffled=True, seed=SEED+17),
                "selected_steps": fitted["steps"], "validation_bce": fitted["val_bce"],
                "positive_weight": fitted["positive_weight"]}
        if task in free_features:
            item["generated_test"] = evaluate(fitted, free_features[task])
        if task == "answer":
            # A shared direction may miss information represented differently
            # at different progress points. Fit independent affine heads too.
            item["phase_specific"] = {}
            for phase in np.unique(train_data["stage"]):
                vd = val_features[task]
                td = test_features[task]
                if not (vd["stage"] == phase).any() or not (td["stage"] == phase).any():
                    continue
                tr = subset(train_data, train_data["stage"] == phase)
                va = subset(vd, vd["stage"] == phase)
                te = subset(td, td["stage"] == phase)
                specific = fit(tr, va, device=am.device, seed=SEED, steps=steps)
                random = fit(tr, va, device=am.device, seed=SEED+17, steps=steps, shuffled=True)
                phase_result = {"reference_test": evaluate(specific, te),
                                "shuffled_control_test": evaluate(random, te, shuffled=True, seed=SEED+17),
                                "train": evaluate(specific, tr), "selected_steps": specific["steps"]}
                if task in free_features:
                    fd = free_features[task]
                    if (fd["stage"] == phase).any():
                        phase_result["generated_test"] = evaluate(specific, subset(fd, fd["stage"] == phase))
                item["phase_specific"][str(phase)] = phase_result
        result["targets"][task] = item
    return result

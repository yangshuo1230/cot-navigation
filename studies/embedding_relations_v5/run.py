"""Strict focal-node-heldout linear set decoding from raw PATH embeddings.

One embedding per graph node, never repeated QUERY contexts. Fivefold rotations:
48 train /16 validation /16 test nodes, repeated3 balanced node assignments.
Ridge is fit in FP64 on idle GPU0; no model forward, training or weight files.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
import torch
torch.set_num_threads(2)
ROOT = Path.cwd()
HERE = ROOT / 'studies/embedding_relations_v5'
SUITE = ROOT / 'studies/fixed_navigation_v5'
sys.path[:0] = [str(ROOT / 'src'), str(SUITE)]
from data import digest, atomic_json, bfs

REGULARIZATION = (.0001, .001, .01, .1, 1., 10., 100.)
NODE_SEEDS = (17, 23, 31)
RANDOM_SEEDS = (101, 202, 303)


def auc(y, scores):
    p, n = scores[y.astype(bool)], scores[~y.astype(bool)]
    if not len(p) or not len(n):
        return None
    d = p[:, None] - n[None, :]
    return float(((d > 0) + .5*(d == 0)).mean())


def rank_rows(y, scores, mask):
    values = {'auc': [], 'recall_at_true_k': [], 'uniform_recall': [], 'top1_hit': [],
              'mean_positive_minus_negative_score': []}
    for yy, ss, mm in zip(y, scores, mask):
        yy, ss = yy[mm].astype(bool), ss[mm]
        k = int(yy.sum())
        if k == 0 or k == len(yy):
            continue  # no empty / all-positive set can inflate ranking metrics
        values['auc'].append(auc(yy, ss))
        values['recall_at_true_k'].append(float(yy[np.argsort(ss, kind='stable')[-k:]].mean()))
        values['uniform_recall'].append(float(yy.mean()))
        values['top1_hit'].append(float(yy[np.argmax(ss)]))
        values['mean_positive_minus_negative_score'].append(float(ss[yy].mean() - ss[~yy].mean()))
    return {'informative_rows': len(values['auc']),
            **{k: float(np.mean(v)) if v else None for k, v in values.items()}}


def columns_auc(y, scores, mask):
    values = [auc(y[mask[:, c], c], scores[mask[:, c], c]) for c in range(y.shape[1])]
    values = [v for v in values if v is not None]
    return {'eligible_columns': len(values), 'auc': float(np.mean(values)) if values else None}


def validation_threshold(y, score, mask):
    y, score = y[mask].astype(int), score[mask]
    unique = np.r_[np.unique(score), np.nextafter(score.max(), np.inf)]
    order = np.argsort(score)
    cuts = np.searchsorted(score[order], unique, side='left')
    positives = np.r_[0, np.cumsum(y[order])]
    tp = positives[-1] - positives[cuts]
    predicted_n = len(score) - cuts
    f1 = 2*tp / np.maximum(1, predicted_n + positives[-1])
    choices = np.flatnonzero(f1 == f1.max())
    return float(unique[choices[np.argmin(abs(unique[choices] - .5))]])


def set_recovery(y, scores, threshold, mask):
    target = y.astype(bool) & mask
    prediction = (scores >= threshold) & mask
    exact = (prediction == target).all(1)
    nonempty = target.any(1)
    tp = (target & prediction).sum(1)
    f1 = 2*tp / np.maximum(1, target.sum(1) + prediction.sum(1))
    both_empty = ~(target | prediction).any(1)
    f1[both_empty] = 1
    return {'threshold': threshold, 'n_nodes': len(y), 'nonempty_nodes': int(nonempty.sum()),
            'exact_set': float(exact.mean()),
            'nonempty_exact_set': float(exact[nonempty].mean()) if nonempty.any() else None,
            'empty_exact_set': float(exact[~nonempty].mean()) if (~nonempty).any() else None,
            'macro_f1': float(f1.mean())}


def ridge_predictions(features, y, train, validation, test):
    """Batched penalty candidates; task-specific validation MSE, no test tuning.

    The dual solve is48x48 rather than129x129. Bias is unpenalized through
    centering Y. Output is80 candidates per task; scores are not probabilities.
    """
    z = torch.tensor(features, dtype=torch.float64, device='cuda:0')
    labels = torch.tensor(y, dtype=torch.float64, device=z.device)
    mean = z[train].mean(0)
    scale = z[train].std(0, unbiased=False).clamp_min(1e-6)
    z = (z - mean) / scale
    ztrain = z[train]
    ymean = labels[train].mean(0)
    centered = labels[train] - ymean
    gram = ztrain @ ztrain.T / len(train)
    eye = torch.eye(len(train), dtype=z.dtype, device=z.device)
    matrices = torch.stack([gram + r*eye for r in REGULARIZATION])
    dual = torch.linalg.solve(matrices, (centered / len(train)).expand(len(REGULARIZATION), -1, -1))
    weights = ztrain.T[None] @ dual
    val = z[validation][None] @ weights + ymean
    tst = z[test][None] @ weights + ymean
    tasks = y.shape[1] // 80
    validation_scores, test_scores, penalties = [], [], []
    for task in range(tasks):
        sl = slice(task*80, (task+1)*80)
        mse = (val[:, :, sl] - labels[validation, sl]).square().mean((1, 2))
        best = int(mse.argmin())
        validation_scores.append(val[best, :, sl].cpu().numpy())
        test_scores.append(tst[best, :, sl].cpu().numpy())
        penalties.append(REGULARIZATION[best])
    return validation_scores, test_scores, penalties


def evaluate(y, val_score, test_score, validation, test, feasible):
    identity = ~np.eye(len(y), dtype=bool)
    threshold = validation_threshold(y[validation], val_score, identity[validation])
    return {'full_row_ranking': rank_rows(y[test], test_score, identity[test]),
            'feasible_row_ranking': rank_rows(y[test], test_score, feasible[test] & identity[test]),
            'full_candidate_column_auc': columns_auc(y[test], test_score, identity[test]),
            'feasible_column_auc': columns_auc(y[test], test_score, feasible[test] & identity[test]),
            'full_set_recovery': set_recovery(y[test], test_score, threshold, identity[test])}


def node_assignment(seed, n=80, width=8):
    rng = np.random.default_rng(seed)
    # Balanced round-robin across permuted nodes in every layer. Each fold has
    # exactly16 nodes and1–2 nodes/layer; every test node appears once/rotation.
    order = np.concatenate([rng.permutation(np.arange(i, i+width)) for i in range(0, n, width)])
    groups = np.zeros(n, dtype=int)
    groups[order] = np.arange(n) % 5
    assert [int((groups == k).sum()) for k in range(5)] == [16]*5
    return groups


def run(seed):
    world = json.loads((SUITE / 'data/n80_g0.json').read_text())
    path = ROOT / f'outputs/fixed_navigation_v5/pilot/n80/g0_s{seed}/path/latest.pt'
    blob = torch.load(path, map_location='cpu', weights_only=False)
    assert blob['step'] == 50000 and blob['world_hash'] == digest(world)
    embedding = blob['model']['embed.weight'][np.asarray(world['node_names']) + 14].double().numpy()
    # Token indices are verified against the project tokenizer below.
    from cot_navigation.protocols.tokenizer import Tokenizer
    tok = Tokenizer(256)
    assert np.array_equal(np.asarray(world['node_names']) + 14, [tok.node_id(n) for n in world['node_names']])
    del blob
    n = world['n_nodes']
    adjacency = np.zeros((n, n), dtype=float)
    reach = np.zeros((n, n), dtype=float)
    for u, successors in enumerate(world['adj']):
        adjacency[u, successors] = 1
        reach[u, list(bfs(world['adj'], u))] = 1
    np.fill_diagonal(reach, 0)  # Strict ancestors/descendants exclude zero-hop self.
    independent = np.zeros((n, n), dtype=bool)
    for u in reversed(range(n)):
        for v in world['adj'][u]:
            independent[u, v] = True
            independent[u] |= independent[v]
    assert np.array_equal(reach, independent)
    layer = np.arange(n) // world['width']
    tasks = {'ancestors': reach.T, 'reachable_descendants': reach, 'direct_successors': adjacency}
    feasible = {'ancestors': layer[:, None] > layer[None, :],
                'reachable_descendants': layer[:, None] < layer[None, :],
                'direct_successors': layer[:, None] + 1 == layer[None, :]}
    all_labels = np.concatenate(list(tasks.values()), axis=1)
    features = {'embedding': embedding, 'layer_onehot_oracle': np.eye(10)[layer],
                'candidate_prior_only': np.zeros((n, 1))}
    for random_seed in RANDOM_SEEDS:
        features[f'random_identity_{random_seed}'] = np.random.default_rng(random_seed).normal(size=embedding.shape)
    out = {'seed': seed, 'checkpoint_step': 50000, 'world_hash': digest(world),
           'embedding_shape': list(embedding.shape), 'node_split': '48train/16validation/16test; focal-node input held out',
           'candidate_note': 'All80 output candidate identities fixed; candidate memberships may occur in training-node label sets',
           'tasks': {name: {'nonempty_nodes': int(y.any(1).sum()), 'positive_memberships': int(y.sum())} for name, y in tasks.items()},
           'folds': []}
    for node_seed in NODE_SEEDS:
        groups = node_assignment(node_seed)
        for fold in range(5):
            train = np.flatnonzero((groups != fold) & (groups != (fold+1) % 5))
            validation = np.flatnonzero(groups == (fold+1) % 5)
            test = np.flatnonzero(groups == fold)
            assert not (set(train) & set(test) or set(validation) & set(test))
            record = {'node_seed': node_seed, 'test_fold': fold, 'train_nodes': train.tolist(),
                      'validation_nodes': validation.tolist(), 'test_nodes': test.tolist(), 'features': {}}
            for name, x in features.items():
                vs, ts, penalties = ridge_predictions(x, all_labels, train, validation, test)
                record['features'][name] = {task: {'ridge': penalties[i], **evaluate(y, vs[i], ts[i], validation, test, feasible[task])}
                                             for i, (task, y) in enumerate(tasks.items())}
            # Shuffle only training labels WITHIN each focal layer, retaining
            # structural feasibility/candidate marginals. Validation/test true
            # labels are never inserted into this shuffled training target.
            for random_seed in RANDOM_SEEDS:
                shuffled = all_labels.copy()
                rng = np.random.default_rng(node_seed*1000 + fold*10 + random_seed)
                for level in range(10):
                    ids = train[layer[train] == level]
                    shuffled[ids] = all_labels[rng.permutation(ids)]
                vs, ts, penalties = ridge_predictions(embedding, shuffled, train, validation, test)
                record['features'][f'within_layer_shuffled_train_{random_seed}'] = {
                    task: {'ridge': penalties[i], **evaluate(y, vs[i], ts[i], validation, test, feasible[task])}
                    for i, (task, y) in enumerate(tasks.items())}
            out['folds'].append(record)
        print(f'seed{seed} node-assignment{node_seed} complete', flush=True)
    HERE.mkdir(parents=True, exist_ok=True)
    atomic_json(HERE / f'seed{seed}.json', out)
    print(f'seed{seed} COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3])
    for seed in parser.parse_args().seeds:
        run(seed)

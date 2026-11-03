"""Probability and ranking diagnostics for masked multi-label linear probes.

Sigmoids are independent membership scores, not a softmax distribution over
mutually exclusive next tokens. All ranking comparisons stay inside a graph.
"""
from __future__ import annotations

import numpy as np
import torch


def scores(fitted, data):
    x = torch.as_tensor(data['x'], dtype=torch.float32, device=fitted['weight'].device)
    with torch.no_grad():
        z = torch.einsum('nld,ldk->nlk', (x-fitted['mean'])/fitted['scale'], fitted['weight'])+fitted['bias']
    return z.cpu().numpy()


def auc_ap(y, score):
    """Tie-aware AUROC and average precision; None for undefined populations."""
    y, score = np.asarray(y, bool), np.asarray(score, float)
    if not y.any() or y.all():
        return {'auroc': None, 'average_precision': None}
    order = np.argsort(-score, kind='stable')
    yy, ss = y[order], score[order]
    ends = np.r_[np.flatnonzero(ss[1:] != ss[:-1])+1, len(ss)]
    tp = np.r_[0, np.cumsum(yy)[ends-1]]
    fp = np.r_[0, ends-tp[1:]]
    tpr, fpr = tp/y.sum(), fp/(~y).sum()
    precision = tp[1:]/ends
    return {'auroc': float(np.sum(np.diff(fpr)*(tpr[1:]+tpr[:-1])/2)),
            'average_precision': float(np.sum(np.diff(tpr)*precision))}


def set_quality(logits, y, mask):
    """Check positive AND negative membership, and within-row discrimination."""
    z, y, mask = np.asarray(logits), np.asarray(y, bool), np.asarray(mask, bool)
    p = 1/(1+np.exp(-np.clip(z, -80, 80)))
    pos, neg = mask & y, mask & ~y
    rows, empty, full = [], 0, 0
    for zz, pp, yy, mm in zip(z, p, y, mask):
        a, b = zz[mm & yy], zz[mm & ~yy]
        if not len(a): empty += 1; continue
        if not len(b): full += 1; continue
        gap = a[:, None]-b[None, :]
        rank = float(((gap > 0)+.5*(gap == 0)).mean())
        order = np.argsort(-zz[mm], kind='stable')
        truth = yy[mm]
        k = int(truth.sum())
        # Strict separation is tie-safe. Top-k uses a deterministic tie order.
        rows.append({'pairwise_auc': rank, 'strict_separation': float(a.min() > b.max()),
                     'topk_exact': float(truth[order[:k]].all()),
                     'topk_recall': float(truth[order[:k]].sum()/k),
                     'mean_probability_gap': float(pp[mm & yy].mean()-pp[mm & ~yy].mean()),
                     'margin': float(a.min()-b.max()),
                     'ap': auc_ap(truth, zz[mm])['average_precision']})
    out = {'n_rows': len(y), 'nontrivial_rows': len(rows), 'empty_target_rows': empty,
           'full_target_rows': full, 'n_positive': int(pos.sum()), 'n_negative': int(neg.sum()),
           'mean_true_probability': float(p[pos].mean()) if pos.any() else None,
           'mean_false_probability': float(p[neg].mean()) if neg.any() else None,
           'false_positive_rate_at_half': float((p[neg] >= .5).mean()) if neg.any() else None,
           'positive_recall_at_half': float((p[pos] >= .5).mean()) if pos.any() else None,
           'brier': float(((p-y)**2)[mask].mean()),
           'micro_ranking': auc_ap(y[mask], z[mask])}
    if rows:
        out.update({key: float(np.mean([r[key] for r in rows])) for key in rows[0]})
    return out


def evaluate_sets(fitted, data):
    z = scores(fitted, data)
    return [set_quality(z[:, i], data['y'], data['mask']) for i in range(z.shape[1])]


def evaluate_errors(fitted, data):
    from .probes import metrics
    z = scores(fitted, data)
    result = []
    for i in range(z.shape[1]):
        row = metrics(z[:, i], data['y'], data['mask'])
        row.update(auc_ap(data['y'][:, 0], z[:, i, 0]))
        p = 1/(1+np.exp(-np.clip(z[:, i, 0], -80, 80)))
        yy = data['y'][:, 0].astype(bool)
        row['mean_error_probability_on_wrong'] = float(p[yy].mean()) if yy.any() else None
        row['mean_error_probability_on_correct'] = float(p[~yy].mean()) if (~yy).any() else None
        # Every event contains an erroneous prefix and its repaired control.
        groups = {}
        for n, event in enumerate(data['event']): groups.setdefault(int(event), []).append(n)
        wins = []
        for indices in groups.values():
            a = [n for n in indices if yy[n]]; b = [n for n in indices if not yy[n]]
            if a and b:
                gap = float(z[a[0], i, 0]-z[b[0], i, 0])
                wins.append(float(gap > 0)+.5*float(gap == 0))
        row['paired_wrong_higher_rate'] = float(np.mean(wins)) if wins else None
        result.append(row)
    return result

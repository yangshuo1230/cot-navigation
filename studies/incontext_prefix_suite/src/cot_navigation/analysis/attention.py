"""Streaming relative-distance centering of causal attention.

mu[layer, head, d] is the pair-weighted mean of A[i,j] over calibration
examples with i-j=d, j<=i. Padding and final answers are never included.
Held-out residual attention is A[i,j]-mu[layer,head,i-j]. This is a descriptive
distance correction, not a causal removal of every RoPE/position effect.
"""
from __future__ import annotations

import numpy as np
import torch

from .common import ROLES, role_ids


class DistanceBaseline:
    def __init__(self, layers, heads, max_len):
        self.sums = np.zeros((layers, heads, max_len), dtype=np.float64)
        self.counts = np.zeros(max_len, dtype=np.int64)

    def add(self, weights):
        """weights [layers,heads,length,length], with no padding."""
        length = weights.shape[-1]
        for d in range(length):
            self.sums[..., d] += np.diagonal(weights, offset=-d, axis1=-2, axis2=-1).sum(-1)
            self.counts[d] += length-d

    @property
    def mean(self):
        return self.sums / np.maximum(self.counts, 1)[None, None]


def collect_attention(am, calibration, evaluation, *, generated_prefixes=None):
    baseline = DistanceBaseline(am.model.n_layers, am.model.n_heads, am.model.max_seq_len)
    # Calibration on TRAIN queries; this estimate is frozen before held-out maps.
    for ex in calibration:
        tokens = ex.row.tokens[:ex.row.answer_index]
        _, _, w = am.forward([tokens], attention=True)
        baseline.add(torch.stack(w)[:, 0].float().cpu().numpy())
    mu = baseline.mean
    n_roles = len(ROLES)
    shape = (am.model.n_layers, am.model.n_heads, n_roles, n_roles)
    raw, corrected, counts = np.zeros(shape), np.zeros(shape), np.zeros((n_roles, n_roles))
    decision_raw = np.zeros(shape[:2] + (n_roles,))
    decision_corrected = np.zeros_like(decision_raw)
    decision_counts = np.zeros(n_roles)
    unseen, all_pairs = 0, 0
    examples = []
    for ex in evaluation:
        tokens = (ex.row.tokens[:ex.row.answer_index] if generated_prefixes is None
                  else generated_prefixes[ex.index])
        _, _, w = am.forward([tokens], attention=True)
        a = torch.stack(w)[:, 0].float().cpu().numpy()
        roles = role_ids(tokens, ex, am.protocol, am.tok)
        i, j = np.tril_indices(len(tokens))
        d = i-j
        supported = baseline.counts[d] > 0
        unseen += int((~supported).sum())
        all_pairs += len(d)
        pair_role = roles[i] * n_roles + roles[j]
        pair_role = pair_role[supported]
        pair_counts = np.bincount(pair_role, minlength=n_roles*n_roles).reshape(n_roles, n_roles)
        counts += pair_counts
        for layer in range(a.shape[0]):
            for head in range(a.shape[1]):
                values = a[layer, head, i[supported], j[supported]]
                residual = values-mu[layer, head, d[supported]]
                raw[layer, head] += np.bincount(pair_role, weights=values, minlength=n_roles*n_roles).reshape(n_roles, n_roles)
                corrected[layer, head] += np.bincount(pair_role, weights=residual, minlength=n_roles*n_roles).reshape(n_roles, n_roles)
        # Actual attention MASS at the decision input position, with expected
        # distance-conditioned mass subtracted (distinct from per-pair mean).
        last_d = len(tokens)-1-np.arange(len(tokens))
        seen = baseline.counts[last_d] > 0
        for r in range(n_roles):
            mask = (roles == r) & seen
            decision_raw[..., r] += a[..., -1, mask].sum(-1)
            decision_corrected[..., r] += (a[..., -1, mask]-mu[..., last_d[mask]]).sum(-1)
            decision_counts[r] += mask.sum()
        if len(examples) < 3:
            examples.append({"query": ex.index, "label": ex.label, "length": len(tokens)})
    denom = np.maximum(counts, 1)[None, None]
    return {
        "mode": ("reference_prefix_excluding_answer" if generated_prefixes is None
                 else "generated_prefix_excluding_answer"), "roles": list(ROLES),
        "calibration_queries": len(calibration), "evaluation_queries": len(evaluation),
        "distance_support": baseline.counts.tolist(), "distance_mean": mu.tolist(),
        "unsupported_pair_fraction": unseen/max(1, all_pairs),
        "role_pair_counts": counts.astype(int).tolist(),
        "raw_per_pair": (raw/denom).tolist(), "corrected_per_pair": (corrected/denom).tolist(),
        "decision_raw_mass": (decision_raw/max(1, len(evaluation))).tolist(),
        "decision_corrected_mass": (decision_corrected/max(1, len(evaluation))).tolist(),
        "examples": examples,
    }

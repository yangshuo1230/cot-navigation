"""Quota plans and fixed-graph query splits.

Context-graph quotas are exact integer allocations, not expected frequencies
under independent coin flips. Remainder labels go to the smallest distances,
and positive/negative counts inside a shape are equal.
"""

from __future__ import annotations

import math

import numpy as np

from cot_navigation.data.reachability import pair_at_rank, pool_index


def per_distance_quotas(n_samples: int, n_distances: int) -> list[tuple[int, int, int]]:
    """Return ``(distance, n_pos, n_neg)`` for distances ``1 .. n_distances``.

    ``n_samples`` is the per-shape count and must be even. Each label receives
    ``n_samples // 2`` rows, spread as evenly as possible. The first ``rem``
    distances get one extra row of each label.
    """
    if n_samples < 0 or n_samples % 2 != 0:
        raise ValueError(f"per-shape count must be even, got {n_samples}")
    if n_distances < 1:
        raise ValueError("n_distances")
    half = n_samples // 2
    base, rem = divmod(half, n_distances)
    quotas = []
    for index, distance in enumerate(range(1, n_distances + 1)):
        count = base + (1 if index < rem else 0)
        quotas.append((distance, count, count))
    return quotas


def iter_sample_plan(shape_names: list[str], layers_of: dict[str, int], per_shape: int):
    """Yield ``(local_index, shape, distance, label)`` in a stable order.

    ``label`` is 1 for reachable and 0 for unreachable. Order does not depend
    on worker scheduling.
    """
    cursor = 0
    for shape in shape_names:
        n_distances = layers_of[shape] - 1
        for distance, n_pos, n_neg in per_distance_quotas(per_shape, n_distances):
            for _ in range(n_pos):
                yield cursor, shape, distance, 1
                cursor += 1
            for _ in range(n_neg):
                yield cursor, shape, distance, 0
                cursor += 1


def split_small_bucket(n: int, ratios: tuple[float, float, float] = (0.5, 0.2, 0.3)):
    """Split ``n`` items into train/val/test index arrays.

    Buckets smaller than 3 cannot fill three non-empty splits, so every row
    stays in test and the shortfall is recorded. Otherwise ``floor`` assigns
    train and val, and the remainder stays in test. Rows are not copied.
    """
    if n < 3:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.arange(n, dtype=np.int64),
            {"n": n, "insufficient": True, "n_train": 0, "n_val": 0, "n_test": n},
        )
    n_train = int(math.floor(n * ratios[0]))
    n_val = int(math.floor(n * ratios[1]))
    if n_train + n_val > n - 1:
        overflow = n_train + n_val - (n - 1)
        take = min(overflow, n_val)
        n_val -= take
        overflow -= take
        n_train -= overflow
    n_test = n - n_train - n_val
    return (
        np.arange(n_train, dtype=np.int64),
        np.arange(n_train, n_train + n_val, dtype=np.int64),
        np.arange(n_train + n_val, n, dtype=np.int64),
        {
            "n": n,
            "insufficient": False,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
        },
    )


def assign_bucket(n: int, rng, ratios: tuple[float, float, float]):
    """Permute a bucket, then apply :func:`split_small_bucket` on the permutation."""
    perm = rng.permutation(n).astype(np.int64, copy=False) if n else np.zeros(0, dtype=np.int64)
    train_i, val_i, test_i, meta = split_small_bucket(n, ratios)
    return perm[train_i], perm[val_i], perm[test_i], meta


def materialize_split_pairs(
    reach: list[int],
    width: int,
    n_layers: int,
    min_distance: int,
    rng,
    ratios: tuple[float, float, float],
):
    """Sample held-out query pairs for one fixed graph.

    Pairs are addressed by rank inside each ``(distance, label)`` pool. Only
    selected ranks are decoded to ``(s, t)``. Returns rows
    ``(split, distance, label, s, t)`` with split in ``{0,1,2}``.
    """
    rows: list[tuple[int, int, int, int, int]] = []
    notes: list[dict] = []
    for distance in range(min_distance, n_layers):
        for label, positive in ((1, True), (0, False)):
            nodes, _counts, prefix, total = pool_index(reach, width, n_layers, distance, positive)
            train_r, val_r, test_r, meta = assign_bucket(total, rng, ratios)
            meta = dict(meta)
            meta.update({"distance": distance, "label": label, "pool": total})
            notes.append(meta)
            for split, ranks in ((0, train_r), (1, val_r), (2, test_r)):
                for rank in ranks.tolist():
                    src, dst = pair_at_rank(reach, width, distance, positive, nodes, prefix, int(rank))
                    rows.append((split, distance, label, src, dst))
    return rows, notes


def balanced_ids(distances: np.ndarray, labels: np.ndarray, rng) -> np.ndarray:
    """Match the positive distance histogram and balance labels, without copies.

    For each distance, keep ``min(#pos, #neg)`` of each class. Shortfalls are
    the caller's to record from the returned length; dropped rows remain in
    the full split for audit.
    """
    chosen: list[int] = []
    for distance in sorted(set(int(d) for d in distances.tolist())):
        pos = np.flatnonzero((distances == distance) & (labels == 1))
        neg = np.flatnonzero((distances == distance) & (labels == 0))
        take = int(min(pos.size, neg.size))
        if take == 0:
            continue
        pos_pick = rng.choice(pos, size=take, replace=False)
        neg_pick = rng.choice(neg, size=take, replace=False)
        chosen.extend(int(i) for i in pos_pick.tolist())
        chosen.extend(int(i) for i in neg_pick.tolist())
    if not chosen:
        return np.zeros(0, dtype=np.int64)
    return np.asarray(chosen, dtype=np.int64)

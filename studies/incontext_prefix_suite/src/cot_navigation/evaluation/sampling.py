"""Fixed, stratified evaluation subsets independent of training RNG and protocol."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def stratified_ids(ids, shapes, distances, labels, limit: int, seed: int) -> list[int]:
    """Round-robin shape/label, then distance, before taking a second bucket row.

    Stored arrays are sorted by quota. Taking a prefix would evaluate only one
    positive bucket. Small subsets cover shapes/classes first; a sufficiently
    large subset visits every nonempty shape/distance/class bucket once.
    """
    rng = np.random.default_rng(seed)
    buckets = defaultdict(list)
    for i in ids:
        i = int(i)
        buckets[(str(shapes[i]), int(labels[i]), int(distances[i]))].append(i)
    groups = defaultdict(list)
    for key in sorted(buckets):
        rng.shuffle(buckets[key])
        groups[key[:2]].append(key)
    schedule = []
    for level in range(max((len(g) for g in groups.values()), default=0)):
        for group in sorted(groups):
            if level < len(groups[group]):
                schedule.append(groups[group][level])
    target = len(ids) if limit <= 0 else min(limit, len(ids))
    selected = []
    depth = 0
    while len(selected) < target:
        for key in schedule:
            if depth < len(buckets[key]):
                selected.append(buckets[key][depth])
                if len(selected) == target:
                    return selected
        depth += 1
    return selected

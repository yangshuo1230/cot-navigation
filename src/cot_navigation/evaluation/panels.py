"""Deterministic stratified panel selection."""
import random
def panel_subset(world, ids, cap, seed=701):
    """Fixed unique stratified panel, not sampling duplicate queries to fill cap."""
    if len(ids) <= cap:
        return ids
    rng = random.Random(seed)
    cells = {}
    for i in ids:
        r = world['records'][i]
        cells.setdefault((r['y'], r['gap']), []).append(i)
    for values in cells.values():
        rng.shuffle(values)
    selected = []
    while len(selected) < cap and any(cells.values()):
        for cell in sorted(cells):
            if cells[cell] and len(selected) < cap:
                selected.append(cells[cell].pop())
    return selected


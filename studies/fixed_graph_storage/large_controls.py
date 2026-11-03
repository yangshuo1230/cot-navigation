"""Check whether larger regional masks generalize beyond discovery nodes.

128/512 neurons per layer is a large intervention (25%), not a sparse circuit.
Run separately after followup.py; merge compact results into its existing JSON.
"""
import followup as f
import study as s
from study import np, HERE, json

for setting in ('simple', 'hard'):
    p, parts, graph, train, test, discovery, base = f.setup(setting, 'path', 'cuda:1')
    path = HERE/f'{setting}_path_followup.json'; out = json.loads(path.read_text())
    controls = {}
    panel, _ = f.regional_panel(p, train, test, discovery, base, parts['lateral'], k=128, node_heldout=True)
    controls['node_heldout'] = panel
    for seed in (201, 303, 407):
        rng = np.random.default_rng(seed); labels = parts['lateral'].copy()
        for band in range(10):
            ix = np.flatnonzero(np.arange(len(labels))//10 == band)
            labels[ix] = rng.permutation(labels[ix])
        panel, _ = f.regional_panel(p, train, test, discovery, base, labels, k=128)
        controls[f'random_partition_{seed}'] = panel
    out['large_controls'] = controls; s.write(path, out)
    print(setting, {k: v['interaction'] for k, v in controls.items()}, flush=True)

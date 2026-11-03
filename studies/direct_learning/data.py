"""Offline one-hop DAG data with independent input-level label/split audits.

M2 has four visible endpoints. M>2 adds adjacent-layer distractors in a 4x4
DAG, never changing any of the core rectangle's four one-hop labels. The
model vocabulary stays 270; vocab means the names used by this dataset.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

PAD, EOS, EDGE, QUERY, BAR, SEMI, YES, NO = range(8)
NODE_LO = 14
LABELS = np.array([1, 0, 0, 1, 0, 1, 1, 0], np.int64)
VARIANTS = {'full': list(range(8)), 'fixed_source': [0, 1, 4, 5],
            'fixed_target': [0, 2, 4, 6], 'fixed_query': [0, 4],
            'fixed_graph': [0, 1, 2, 3]}
SPLITS = ('train', 'val', 'test')
DATA_SEED = 20261020


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def key(kind, vocab, edges):
    return f'{kind}_v{vocab}_m{edges}'


def group(rng, vocab, anchor=None):
    nodes = rng.choice(vocab, 16, replace=False)+NODE_LO
    # Anchors are fixed token identities, not fixed edges or fixed labels.
    locked = {0: NODE_LO} if anchor in ('fixed_source', 'fixed_query') else {}
    if anchor in ('fixed_target', 'fixed_query'):
        locked[4] = NODE_LO+1
    for pos, token in locked.items():
        where = int(np.where(nodes == token)[0][0]) if token in nodes else None
        if where is not None:
            nodes[pos], nodes[where] = nodes[where], nodes[pos]
        else:
            nodes[pos] = token
    assert len(set(nodes)) == 16
    corners = {(0, 4), (0, 5), (1, 4), (1, 5)}
    possible = [(u, v) for u in range(16) for v in range(16)
                if v//4 == u//4+1 and (u, v) not in corners]
    background = [possible[i] for i in rng.permutation(len(possible))[:22]]
    core = [[(0, 4), (1, 5)], [(0, 5), (1, 4)]]
    edges = [[(int(nodes[u]), int(nodes[v])) for u, v in c+background] for c in core]
    queries = [(int(nodes[u]), int(nodes[v])) for u, v in [(0, 4), (0, 5), (1, 4), (1, 5)]]
    return edges, queries


def graph_keys(g, m=2):
    return [tuple(sorted(e[:m])) for e in g[0]]


def materialize(groups, kind, edges, seed):
    variants = VARIANTS.get(kind, VARIANTS['full'])
    x, y, gids, vids = [], [], [], []
    rng = np.random.default_rng(seed)
    for gi, (graphs, queries) in enumerate(groups):
        chosen = variants
        if kind == 'ordinary':
            # Exactly one query per graph, not a complete label-balanced query block.
            chosen = [int(rng.integers(8))]
        for vi in chosen:
            graph, qi = divmod(vi, 4)
            prompt = [t for u, v in graphs[graph][:edges] for t in (EDGE, u, v)]
            prompt += [QUERY, *queries[qi], BAR]
            x.append(prompt);y.append(int(LABELS[vi]));gids.append(gi);vids.append(vi)
    if kind == 'ordinary':
        # Keep independent queries but remove accidental class imbalance.
        yarr = np.array(y)
        n = min(int((yarr == c).sum()) for c in (0, 1))
        keep = np.concatenate([np.flatnonzero(yarr == c)[:n] for c in (0, 1)])
    else:
        keep = np.arange(len(y))
    return {'x': np.asarray(x, np.int64)[keep], 'y': np.asarray(y, np.int64)[keep],
            'group': np.asarray(gids, np.int64)[keep], 'variant': np.asarray(vids, np.int64)[keep]}


def audit_pool(parts, name, fixed_graph=False):
    summary, seen = {}, {}
    for split, p in parts.items():
        exact = set()
        for prompt, label in zip(p['x'], p['y']):
            triples = prompt[:-4].reshape(-1, 3)
            assert np.all(triples[:, 0] == EDGE)
            pairs = [(int(t[1]), int(t[2])) for t in triples]
            assert len(pairs) == len(set(pairs))
            s, t = map(int, prompt[-3:-1])
            # Independent traversal: a one-hop label must also equal DAG reachability
            # because query endpoints are always in the same adjacent layer pair.
            reached, stack = {s}, [s]
            while stack:
                u = stack.pop()
                for a, b in pairs:
                    if a == u and b not in reached:
                        reached.add(b);stack.append(b)
            assert int((s, t) in pairs) == int(t in reached) == int(label)
            identity = (tuple(sorted(pairs)), s, t)
            exact.add(identity)
        seen[split] = exact
        summary[split] = {'n': len(p['y']), 'yes': int(p['y'].sum()),
                          'unique_queries': len(exact), 'prompt_length': p['x'].shape[1]}
    if not fixed_graph:
        for i, a in enumerate(SPLITS):
            for b in SPLITS[i+1:]:
                assert not seen[a] & seen[b], (name, 'query leakage', a, b)
    return summary


def prepare(folder):
    folder = Path(folder)
    manifest_path = folder/'manifest.json'
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    folder.mkdir(parents=True, exist_ok=True)
    # Global ownership prevents an input graph used by one curriculum pool's
    # training from entering another pool's held-out split. Fixed-world A is
    # an explicit memorization control and is kept out of this registry.
    ownership = {}
    # Reserve the tiny anchored-query universe FIRST; otherwise larger pools
    # can consume its held-out support before the 96/32/32 split is made.
    specs = [('fixed_query', 16)]
    specs += [(kind, v) for v in (16, 64, 256) for kind in ('full',)]
    specs += [(kind, v) for v in (16, 256) for kind in ('fixed_source', 'fixed_target')]
    specs += [('ordinary', 16), ('ordinary', 256)]
    summaries = {}
    for spec_index, (kind, vocab) in enumerate(specs):
        rng = np.random.default_rng(DATA_SEED+spec_index*100)
        pools = {}
        counts = (96, 32, 32) if kind == 'fixed_query' else ((4096, 512, 1024) if kind == 'ordinary' else (512, 128, 128))
        for split, count in zip(SPLITS, counts):
            groups, local_seen, attempts = [], set(), 0
            while len(groups) < count:
                attempts += 1
                if attempts > 1000000:
                    raise RuntimeError(f'Cannot obtain disjoint pool: {kind} {vocab} {split}')
                g = group(rng, vocab, kind)
                keys = graph_keys(g)
                if any(k in local_seen or (k in ownership and ownership[k] != split) for k in keys):
                    continue
                for k in keys:
                    local_seen.add(k);ownership[k] = split
                groups.append(g)
            pools[split] = groups
        sizes = (2, 4, 8, 16, 24) if kind == 'full' else ((2, 24) if kind == 'ordinary' else (2,))
        for m in sizes:
            name = key(kind, vocab, m)
            parts = {s:materialize(pools[s], kind, m, DATA_SEED+spec_index*100+i)
                     for i, s in enumerate(SPLITS)}
            if kind != 'full':
                full = materialize(pools['train'], 'full', m, DATA_SEED)
                trained = set(zip(parts['train']['group'].tolist(), parts['train']['variant'].tolist()))
                trained_graphs = {(g, v//4) for g, v in trained}
                keep = np.array([(int(g), int(v)) not in trained and (int(g), int(v)//4) in trained_graphs
                                 for g, v in zip(full['group'], full['variant'])])
                parts['known_graph_query'] = {k:a[keep] for k,a in full.items()}
            summaries[name] = audit_pool(parts, name)
            arrays = {f'{s}_{k}': a for s, part in parts.items() for k, a in part.items()}
            np.savez_compressed(folder/(name+'.npz'), **arrays)
    for vocab in (16, 256):
        g = group(np.random.default_rng(DATA_SEED+vocab), vocab)
        name = key('fixed_graph', vocab, 2)
        parts = {s:materialize([g], 'fixed_graph', 2, DATA_SEED) for s in SPLITS}
        summaries[name] = audit_pool(parts, name, fixed_graph=True)
        np.savez_compressed(folder/(name+'.npz'), **{f'{s}_{k}':a for s,p in parts.items() for k,a in p.items()})
        summaries[name]['warning'] = 'Same known graph/query inputs: fit control, NOT held-out graph generalization.'
    manifest = {'version': 1, 'data_seed': DATA_SEED, 'pools': summaries,
                'split_contract': 'M2 core graphs owned globally by split; paired query blocks never cross splits.',
                'vocab_size': 270, 'node_token_offset': NODE_LO,
                'files': {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('*.npz')}}
    atomic_json(manifest_path, manifest)
    return manifest

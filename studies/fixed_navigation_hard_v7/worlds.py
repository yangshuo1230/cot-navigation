"""Freeze multiscale DAGs with true-hop positive splits and role holdouts."""
from collections import Counter
import importlib.util
import json
from pathlib import Path
import random
import numpy as np
from hard_graph import sample_hard_edges, graph_statistics, shortest_distances

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('legacy_splits', HERE.parent / 'fixed_navigation_v5/data.py')
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)


def validate(world):
    n = world['n_nodes']
    adj, records, groups = world['adj'], world['records'], world['node_groups']
    role = {u: k for k, nodes in groups.items() for u in nodes}
    assert len(role) == n and sum(map(len, groups.values())) == n
    assert len(set(world['node_names'])) == n
    bits = [0] * n
    for u in reversed(range(n)):
        assert len(adj[u]) == len(set(adj[u]))
        for v in adj[u]:
            assert u < v < n
            bits[u] |= (1 << v) | bits[v]
    seen = set()
    for panel, ids in world['panels'].items():
        assert len(ids) == len(set(ids)) and not seen.intersection(ids)
        seen.update(ids)
        assert {records[q]['y'] for q in ids} == {0, 1}
        for q in ids:
            r = records[q]
            assert (r['gap'] <= 3) == panel.startswith('short_')
            if panel.startswith('long_'):
                pair = panel.split('_')[1]
                pair = 'AA' if pair == 'train' else pair
                assert role[r['s']] + role[r['t']] == pair
    assert len({(r['s'], r['t']) for r in records}) == len(records)
    for s in range(n):
        bfs = legacy.bfs(adj, s)
        dp = shortest_distances(adj, s)
        assert all(dp[t] == bfs.get(t, -1) for t in range(n))
        assert all(bool(bits[s] & (1 << t)) == (t != s and t in bfs) for t in range(n))
    for r in records:
        s, t = r['s'], r['t']
        distance = legacy.bfs(adj, s).get(t)
        assert bool(r['y']) == (distance is not None)
        assert r['shortest_hops'] == distance
        assert r['gap'] == (distance if r['y'] else r['band_gap'])
        assert r['paths'] and len(r['paths']) <= world['variant_limit']
        for path in r['paths']:
            assert path[0] == s and len(set(path)) == len(path)
            assert all(v in adj[u] for u, v in zip(path, path[1:]))
            assert path[-1] == t if r['y'] else t not in path
    for endpoint in ('s', 't'):
        eligible = {r[endpoint] for r in records if r['gap'] <= 3}
        assert eligible <= {records[q][endpoint] for q in world['panels']['short_train']}
    # Every real edge has a positive one-hop QUERY (possibly held out) and ADJ.
    pairs = {(r['s'], r['t']): r for r in records}
    assert all(pairs[u, v]['gap'] == pairs[u, v]['y'] == 1 for u, vs in enumerate(adj) for v in vs)
    return dict(status='PASS', dag=True, independent_bfs_dp=True, pair_disjoint=True,
                true_positive_hop_split=True, long_endpoint_roles_checked=True,
                short_all_eligible_roles_covered=True, all_edges_in_adjacency_knowledge=True,
                queries=len(records), edges=sum(map(len, adj)))


def make_world(cfg):
    width, layers, seed = cfg['graph']['width'], cfg['graph']['layers'], cfg['graph']['seed']
    n, limit = width * layers, cfg['variants']
    for attempt in range(cfg['graph']['max_attempts']):
        graph_rng = np.random.default_rng(7_000_000 + seed * 10000 + attempt)
        rng = random.Random(7_100_000 + seed * 10000 + attempt)
        adj = [[] for _ in range(n)]
        for u, v in sample_hard_edges(layers, width, 2*n, graph_rng):
            adj[u].append(v)
        groups = {'A': [], 'V': [], 'T': []}
        for band in range(layers):
            ids = list(range(band*width, (band+1)*width))
            rng.shuffle(ids)
            groups['A'].extend(ids[:width//2])
            groups['V'].extend(ids[width//2:3*width//4])
            groups['T'].extend(ids[3*width//4:])
        role = {u: key for key, ids in groups.items() for u in ids}
        reach = [legacy.bfs(adj, s) for s in range(n)]
        records = []
        for s in range(n):
            for t in range(s+1, n):
                distance = reach[s].get(t)
                band_gap = t//width - s//width
                # Within-band positives are needed to include all edges. As in
                # the baseline, negatives are forward, different-band queries.
                if distance is None and band_gap == 0:
                    continue
                records.append(dict(s=s, t=t, y=int(distance is not None),
                    gap=distance if distance is not None else band_gap,
                    band_gap=band_gap, shortest_hops=distance))
        short = legacy.split_queries([i for i,r in enumerate(records) if r['gap'] <= 3], records, rng, True)
        aa = legacy.split_queries([i for i,r in enumerate(records) if r['gap'] >= 4
                                   and role[r['s']] == role[r['t']] == 'A'], records, rng)
        panels = {f'short_{k}': v for k,v in short.items()}
        panels.update(long_train=aa['train'], long_AA_val=aa['val'], long_AA_test=aa['test'])
        for pair, split in [('VA','val'),('AV','val'),('VV','val'),('TA','test'),('AT','test'),('TT','test')]:
            panels[f'long_{pair}_{split}'] = [i for i,r in enumerate(records) if r['gap'] >= 4
                                             and role[r['s']]+role[r['t']] == pair]
        minima = {p: 16 for p in panels}
        for p in ('long_AA_val','long_AA_test'):
            minima[p] = 4
        if any(min(sum(records[q]['y'] == y for q in ids) for y in (0,1)) < minima[p]
               for p, ids in panels.items()):
            continue
        longest = [0]*n
        for u in reversed(range(n)):
            longest[u] = max((1+longest[v] for v in adj[u]), default=0)
        for r in records:
            r['paths'] = legacy.make_variants(adj, reach, r, rng, limit, longest)
        world = dict(version=7, graph_family='multiscale_dag_v1', width=width, layers=layers,
                     seed=seed, attempt=attempt, n_nodes=n, adj=adj,
                     node_names=rng.sample(range(256),n), node_groups=groups,
                     records=records, panels=panels, variant_limit=limit,
                     acceptance_minimum_per_label=minima,
                     distance_definition='positive:true shortest path; negative:forward rank-band gap',
                     graph_statistics=graph_statistics(adj,width))
        world['validation'] = validate(world)
        world['panel_counts'] = {p: dict(n=len(ids), cells=dict(Counter(
            f"gap{records[q]['gap']}/y{records[q]['y']}" for q in ids))) for p,ids in panels.items()}
        world['potential_path_exposure'] = legacy.exposure_audit(world)
        return world
    raise RuntimeError('No graph satisfies frozen per-label panel support; do not silently weaken acceptance')


def prepare():
    cfg = json.loads((HERE/'experiment.json').read_text())
    out = HERE/'worlds'
    root = HERE.parent.parent/'outputs/fixed_navigation_hard_v7'
    if root.exists() and any(root.rglob('resolved_job.json')):
        raise RuntimeError('Cannot replace prepared data after formal training starts')
    world = make_world(cfg)
    path = out/'data/n80_g0.json'
    if path.exists():
        assert legacy.digest(json.loads(path.read_text())) == legacy.digest(world)
    legacy.atomic_json(path, world)
    legacy.atomic_json(out/'manifest.json', dict(worlds=[dict(path='data/n80_g0.json',
        sha256=legacy.digest(world), n_nodes=80, seed=0)], graph=cfg['graph']))
    print(json.dumps(dict(attempt=world['attempt'], graph=world['graph_statistics'], panels=world['panel_counts'])), flush=True)

if __name__ == '__main__':
    prepare()

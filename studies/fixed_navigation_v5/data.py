"""Offline graph generation, role splits, frozen trace variants and exposure audit.

No graph search or trajectory construction is performed by the training loop.
Node indices are topological internal indices; only randomized names reach the LM.
"""
from __future__ import annotations
from collections import Counter, deque
import argparse
import hashlib
import json
from pathlib import Path
import random

HERE = Path(__file__).resolve().parent


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.writing')
    temp.write_text(json.dumps(value, separators=(',', ':')), encoding='utf-8')
    temp.replace(path)


def bfs(adj, source):
    distance = {source: 0}
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if v not in distance:
                distance[v] = distance[u] + 1
                queue.append(v)
    return distance


def split_queries(ids, records, rng, cover_roles=False):
    """Unique query split, stratified by label and gap; repair train role support."""
    cells = {}
    for i in ids:
        r = records[i]
        cells.setdefault((r['y'], r['gap']), []).append(i)
    out = {'train': [], 'val': [], 'test': []}
    for cell in sorted(cells):
        values = cells[cell][:]
        rng.shuffle(values)
        n = len(values)
        hold = max(1, int(n * .1)) if n >= 3 else 0
        out['val'].extend(values[:hold])
        out['test'].extend(values[hold:2 * hold])
        out['train'].extend(values[2 * hold:])
    if cover_roles:
        for role in ('s', 't'):
            present = {records[i][role] for i in out['train']}
            for node in sorted({records[i][role] for i in ids} - present):
                for part in ('val', 'test'):
                    candidate = next((i for i in out[part] if records[i][role] == node), None)
                    if candidate is not None:
                        out[part].remove(candidate)
                        out['train'].append(candidate)
                        break
    return {part: sorted(values) for part, values in out.items()}


def make_variants(adj, reach, r, rng, limit, longest):
    s, t, gap = r['s'], r['t'], r['gap']
    variants = set()
    if r['y']:
        # Exact path-count weighted sampling avoids always returning one DFS prefix.
        count = [0] * len(adj)
        count[t] = 1
        for u in reversed(range(t)):
            count[u] = sum(count[v] for v in adj[u])
        target = min(limit, count[s])
        for _ in range(limit * 64):
            path = [s]
            while path[-1] != t:
                choices = [v for v in adj[path[-1]] if count[v]]
                ticket = rng.randrange(sum(count[v] for v in choices))
                for v in choices:
                    ticket -= count[v]
                    if ticket < 0:
                        path.append(v)
                        break
            variants.add(tuple(path))
            if len(variants) >= target:
                break
    else:
        # When possible, negative walks have exactly the corresponding positive
        # gap length. Otherwise retain the query and report the shorter support.
        horizon = min(gap, longest[s])
        for _ in range(limit * 32):
            path = [s]
            while len(path) <= horizon:
                remaining = horizon - len(path)
                choices = [v for v in adj[path[-1]] if longest[v] >= remaining]
                path.append(rng.choice(choices))
            variants.add(tuple(path))
            if len(variants) >= limit:
                break
    return [list(p) for p in sorted(variants)]


def exposure_audit(world, used=None):
    """Potential by default; exact cumulative actual counts can restrict variants.

    Keys q:<query_id>:<variant_id> come from optimizer exposures. Direct queries
    contribute label macros only; ADJ contributes the complete original edges.
    """
    n = world['n_nodes']
    macros = {(u, v) for u, vs in enumerate(world['adj']) for v in vs}
    witnesses, decisions = set(), set()
    train_ids = world['panels']['short_train'] + world['panels']['long_train']
    for qid in train_ids:
        r = world['records'][qid]
        if used is None or any(used.get(f'q:{qid}:{v}', 0) for v in range(len(r['paths']))):
            if r['y']:
                macros.add((r['s'], r['t']))
        for variant, path in enumerate(r['paths']):
            if used is not None and not used.get(f'q:{qid}:{variant}', 0):
                continue
            witnesses.update((u, v) for j, u in enumerate(path) for v in path[j + 1:])
            decisions.update((u, r['t']) for u in path[:-1])
    macros.update(witnesses)
    h = [[] for _ in range(n)]
    for u, v in sorted(macros):
        h[u].append(v)
    distance = [bfs(h, s) for s in range(n)]
    detail, summary = {}, {}
    for panel, ids in world['panels'].items():
        if not (panel.endswith('_test') or panel.endswith('_val')):
            continue
        positive = [i for i in ids if world['records'][i]['y']]
        rows = {}
        for i in positive:
            r = world['records'][i]
            rows[str(i)] = {
                'witness_seen': (r['s'], r['t']) in witnesses,
                'first_decision_seen': (r['s'], r['t']) in decisions,
                'macro_distance': distance[r['s']][r['t']],
            }
        detail[panel] = rows
        summary[panel] = {'positive_n': len(positive),
                          'witness_seen': sum(row['witness_seen'] for row in rows.values()),
                          'macro_distance_counts': dict(Counter(row['macro_distance'] for row in rows.values()))}
    return {'summary': summary, 'by_query': detail}


def validate(world):
    """Independent BFS + reverse-topological bitset DP checks every pair/variant."""
    n, width, adj = world['n_nodes'], world['width'], world['adj']
    bits = [0] * n
    for u in reversed(range(n)):
        assert len(adj[u]) == len(set(adj[u]))
        for v in adj[u]:
            assert v // width == u // width + 1
            bits[u] |= (1 << v) | bits[v]
    distances = [bfs(adj, s) for s in range(n)]
    groups = {k: set(v) for k, v in world['node_groups'].items()}
    assert not (groups['A'] & groups['V'] or groups['A'] & groups['T'] or groups['V'] & groups['T'])
    assert set.union(*groups.values()) == set(range(n))
    assert len(set(world['node_names'])) == n
    seen = set()
    records = world['records']
    assert len(records) == len({(r['s'], r['t']) for r in records})
    for name, ids in world['panels'].items():
        assert not seen.intersection(ids), name
        seen.update(ids)
        assert len(ids) == len(set(ids))
        for i in ids:
            r = records[i]
            assert 1 <= r['gap'] <= 3 if name.startswith('short') else 4 <= r['gap'] <= 9
            if name.startswith('long_'):
                role = name.split('_')[1]
                role = 'AA' if role in ('train', 'AA') else role
                assert r['s'] in groups[role[0]] and r['t'] in groups[role[1]], (name, r)
    for r in records:
        s, t, y, gap = r['s'], r['t'], r['y'], r['gap']
        assert gap == t // width - s // width
        assert bool(y) == (t in distances[s]) == bool(bits[s] & (1 << t))
        if y:
            assert distances[s][t] == gap
        assert r['paths'] and len(r['paths']) <= world['variant_limit']
        for p in r['paths']:
            assert p[0] == s and len(p) == len(set(p))
            assert all(v in adj[u] for u, v in zip(p, p[1:]))
            assert (p[-1] == t) if y else (t not in p and len(p) - 1 <= gap)
    train = world['panels']['short_train']
    for role in ('s', 't'):
        eligible = {r[role] for r in records if r['gap'] <= 3}
        assert eligible <= {records[i][role] for i in train}
    for panel in ('short_train', 'short_val', 'short_test', 'long_train', 'long_AA_val', 'long_AA_test',
                  'long_VA_val', 'long_AV_val', 'long_VV_val', 'long_TA_test', 'long_AT_test', 'long_TT_test'):
        assert {records[i]['y'] for i in world['panels'][panel]} == {0, 1}, panel
        minimum = world.get('acceptance_minimum_per_label', {}).get(panel, 1)
        assert min(sum(records[i]['y'] == y for i in world['panels'][panel]) for y in (0, 1)) >= minimum
    return {'status': 'PASS', 'queries': len(records), 'edges': sum(map(len, adj)),
            'variants': sum(len(r['paths']) for r in records), 'pair_disjoint': True,
            'bfs_equals_dp': True, 'all_edges_in_adjacency_knowledge': True,
            'short_all_eligible_roles_covered': True}


def make_world(width, seed, limit=8):
    n, layers = width * 10, 10
    candidates = [(u, v) for u in range(n - width)
                  for v in range((u // width + 1) * width, (u // width + 2) * width)]
    for attempt in range(200):
        rng = random.Random(500_000 + seed * 1000 + width * 10000 + attempt)
        adj = [[] for _ in range(n)]
        for u, v in sorted(rng.sample(candidates, n * 2)):
            adj[u].append(v)
        groups = {'A': [], 'V': [], 'T': []}
        for layer in range(layers):
            nodes = list(range(layer * width, (layer + 1) * width))
            rng.shuffle(nodes)
            groups['A'].extend(nodes[:width // 2])
            groups['V'].extend(nodes[width // 2:3 * width // 4])
            groups['T'].extend(nodes[3 * width // 4:])
        role = {u: k for k, nodes in groups.items() for u in nodes}
        reach = [bfs(adj, s) for s in range(n)]
        longest = [0] * n
        for u in reversed(range(n)):
            longest[u] = max((1 + longest[v] for v in adj[u]), default=0)
        records = [{'s': s, 't': t, 'y': int(t in reach[s]), 'gap': t // width - s // width}
                   for s in range(n) for t in range(n) if t // width > s // width]
        short = split_queries([i for i, r in enumerate(records) if r['gap'] <= 3], records, rng, True)
        aa = split_queries([i for i, r in enumerate(records) if r['gap'] >= 4
                            and role[r['s']] == role[r['t']] == 'A'], records, rng)
        panels = {f'short_{k}': v for k, v in short.items()}
        panels.update(long_train=aa['train'], long_AA_val=aa['val'], long_AA_test=aa['test'])
        for pair, split in [('VA', 'val'), ('AV', 'val'), ('VV', 'val'),
                            ('TA', 'test'), ('AT', 'test'), ('TT', 'test')]:
            panels[f'long_{pair}_{split}'] = [i for i, r in enumerate(records) if r['gap'] >= 4
                                             and role[r['s']] + role[r['t']] == pair]
        if any({records[i]['y'] for i in ids} != {0, 1} for ids in panels.values()):
            continue
        # Structural acceptance is frozen before model training: avoid a TT
        # panel with only a handful of negatives. AA query-holdout panels are
        # smaller, especially in the pilot; publish their exact denominators.
        minima = {name: (16 if width == 8 else 32) for name in panels}
        for name in ('long_AA_val', 'long_AA_test'):
            minima[name] = 4 if width == 8 else 12
        if any(min(sum(records[i]['y'] == y for i in ids) for y in (0, 1)) < minima[name]
               for name, ids in panels.items()):
            continue
        for r in records:
            r['paths'] = make_variants(adj, reach, r, rng, limit, longest)
        world = dict(version=5, width=width, layers=10, seed=seed, attempt=attempt,
                     n_nodes=n, node_names=rng.sample(range(256), n), node_groups=groups,
                     adj=adj, records=records, panels=panels, variant_limit=limit,
                     acceptance_minimum_per_label=minima)
        world['validation'] = validate(world)
        world['panel_counts'] = {k: {'n': len(ids), 'cells': dict(Counter(
            f"gap{records[i]['gap']}/y{records[i]['y']}" for i in ids))} for k, ids in panels.items()}
        world['negative_length_shortfall'] = sum(
            len(p) - 1 < r['gap'] for r in records if not r['y'] for p in r['paths'])
        world['potential_path_exposure'] = exposure_audit(world)
        return world
    raise ValueError(f'No graph with both labels in every panel: width={width}, seed={seed}')


def prepare(replace=False):
    if replace:
        output = HERE.parent.parent / 'outputs/fixed_navigation_v5'
        if output.exists() and any(output.rglob('resolved_job.json')):
            raise RuntimeError('Cannot replace data after a formal v5 run has started')
    cfg = json.loads((HERE / 'experiment.json').read_text())
    worlds = []
    for width in cfg['widths']:
        seeds = [0] if width == 8 else cfg['graph_seeds']
        for seed in seeds:
            path = HERE / 'data' / f'n{10 * width}_g{seed}.json'
            world = make_world(width, seed, cfg['variants'])
            if path.exists() and digest(json.loads(path.read_text())) != digest(world) and not replace:
                raise FileExistsError(f'Refusing dataset replacement: {path}')
            atomic_json(path, world)
            worlds.append({'path': path.relative_to(HERE).as_posix(), 'sha256': digest(world),
                           'n_nodes': width * 10, 'seed': seed, 'validation': world['validation'],
                           'panel_counts': world['panel_counts']})
            print(json.dumps({'prepared': path.name, **world['validation']}), flush=True)
    atomic_json(HERE / 'manifest.json', {'version': 5, 'config_sha256': digest(cfg), 'worlds': worlds})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replace-pretraining-data', action='store_true',
                        help='Only for correcting preparation before any formal run starts')
    prepare(parser.parse_args().replace_pretraining_data)

"""Legal graph/query counterfactuals, represented by small reproducible recipes.

Renaming is permitted ONLY when the graph is supplied in the prompt. Fixed
worlds keep their learned naming and instead vary queries and search priority.
Every rendered trajectory is replayed by an independent transition checker.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json

import numpy as np

from engine import Snapshot, FIELDS, different_state, digest


def reachable(adj, source, target):
    seen, work = {source}, [source]
    while work:
        u = work.pop()
        if u == target:
            return True
        for v in adj[u]:
            if v not in seen:
                seen.add(v); work.append(v)
    return False


def depth_bin(snapshot):
    n = len(snapshot.record['visited'])
    return 0 if n <= 3 else 1 if n <= 7 else 2 if n <= 15 else 3


def unit(recipe, context):
    return (f"graph:{recipe['base_split']}:{recipe['base_index']}" if context else
            f"query:{recipe['source']}:{recipe['target']}")


def validate(ex, rows, tok, fifo):
    """Validate all records without relying on the trajectory constructor."""
    ff, visited = [ex.nodes[ex.source]], {ex.nodes[ex.source]}
    adj = {ex.nodes[u]: {ex.nodes[v] for v in vs} for u, vs in enumerate(ex.adj)}
    terminal = None
    for row in rows:
        assert terminal is None
        assert row['frontier'] == ff
        assert len(row['visited']) == len(visited) and set(row['visited']) == visited
        current = ff[0 if fifo else -1]
        candidates = adj[current] - visited
        assert len(row['candidates']) == len(candidates) and set(row['candidates']) == candidates
        action = ('FOUND' if current == ex.nodes[ex.target] else 'PUSH' if candidates else
                  'POP' if len(ff) > 1 else 'EXHAUST')
        aa = row['action']
        assert aa[0] == tok.id(f'<{action}>')
        if action == 'PUSH':
            assert len(aa) == 2 and aa[1] in candidates
            assert aa[1] == row['candidates'][0]
            ff.append(aa[1]); visited.add(aa[1])
        elif action == 'POP':
            assert len(aa) == 1
            ff.pop(0 if fifo else -1)
        else:
            assert len(aa) == 1
            terminal = action
    assert terminal == ('FOUND' if ex.label else 'EXHAUST')


class Factory:
    """Cache base graphs, not large activation/trajectory collections."""
    def __init__(self, am):
        self.am = am
        self.base = {}
        self.recipes = {}

    def render(self, recipe):
        from cot_navigation.analysis.common import Example, records
        from cot_navigation.protocols.render import render_task, context_prompt, parametric_prompt
        key = (recipe['base_split'], recipe['base_index'])
        if key not in self.base:
            self.base[key] = self.am.example(key[1], key[0])
        base = self.base[key]
        nodes = recipe.get('node_ids', base.nodes)
        assert len(nodes) == len(base.nodes) and len(set(nodes)) == len(nodes)
        assert all(self.am.tok.node_lo <= v < self.am.tok.node_lo + self.am.tok.n_node_tokens for v in nodes)
        if not self.am.context:
            assert list(nodes) == base.nodes, 'Never rename an implicit learned world'
        rng = np.random.default_rng(recipe['priority_seed'])
        adj = [rng.permutation(vs).astype(int).tolist() for vs in base.adj]
        source, target = recipe['source'], recipe['target']
        if self.am.context:
            inverse = {v: i for i, v in enumerate(base.nodes)}
            edge_tokens = base.prompt[:-4]
            assert len(edge_tokens) % 3 == 0
            edges = [(inverse[edge_tokens[i + 1]], inverse[edge_tokens[i + 2]]) for i in range(0, len(edge_tokens), 3)]
            assert {(u, v) for u, vs in enumerate(adj) for v in vs} == set(edges)
            prompt = context_prompt(self.am.tok, nodes, edges, source, target)
        else:
            prompt = self.am.world_prefix + parametric_prompt(self.am.tok, nodes, source, target)
        label = int(reachable(adj, source, target))
        row = render_task(self.am.tok, self.am.protocol, label, prompt, adj, source, target, nodes)
        tag = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
        ex = Example(int(tag[:14], 16), 'synthetic_analysis', label, adj, list(nodes), source, target, row)
        rows = records(row.tokens, len(prompt), self.am.tok)
        validate(ex, rows, self.am.tok, self.am.protocol == 'bfs_state')
        self.recipes[ex.index] = recipe
        return [Snapshot(ex, row.tokens, r, rows[i + 1], i) for i, r in enumerate(rows[:-1])]

    def recipe(self, snapshot):
        return self.recipes[snapshot.example.index]

    def restore(self, recipe, ordinal):
        return self.render(recipe)[ordinal]


def base_recipes(am, queries, variants, seed):
    """All dataset splits are sources; synthetic query identities stay explicit."""
    rng = np.random.default_rng(seed)
    if am.context:
        ids = []
        for split in ('train', 'val', 'test'):
            pool = am.store.indices(split)
            ids.extend((split, int(i)) for i in rng.permutation(pool)[:max(1, queries // 3)])
        rng.shuffle(ids)
        for split, idx in ids:
            a = am.store.arrays
            for variant in range(variants):
                yield dict(base_split=split, base_index=idx, source=int(a['s_topo'][idx]),
                           target=int(a['t_topo'][idx]), priority_seed=int(rng.integers(2**31)), variant=variant)
    else:
        # Include existing train/val/test pairs, then additional legal pairs on
        # the SAME world. No WORLD token or node identity is changed.
        idx = int(am.store.indices('train', view='full')[0])
        pairs = [(s, t) for s in range(len(am.store.adj())) for t in range(s + 1, len(am.store.adj()))]
        rng.shuffle(pairs)
        for source, target in pairs[:queries]:
            for variant in range(variants):
                yield dict(base_split='train', base_index=idx, source=source, target=target,
                           priority_seed=int(rng.integers(2**31)), variant=variant)


def select_recipients(pool, field, count, context, factory, rng):
    """Balance history size; use one base graph/query per field when possible."""
    bins = defaultdict(list)
    for s in pool:
        bins[depth_bin(s)].append(s)
    for values in bins.values():
        rng.shuffle(values)
    seen, result = set(), []
    while len(result) < count:
        progress = False
        for b in range(4):
            while bins[b]:
                s = bins[b].pop()
                key = unit(factory.recipe(s), context)
                if key in seen:
                    continue
                seen.add(key); result.append(s); progress = True
                break
            if len(result) >= count:
                break
        if not progress:
            break
    return result


def rename_to_match(recipient, donor, field, factory, rng):
    """Create a bijective donor node assignment satisfying an exact field.

    Both graphs use the same set of vocabulary node IDs. Matching is by field
    role/length, NOT by model predictions. Graphs, queries and ALL trace nodes
    are consistently relabeled when the recipe is rendered again.
    """
    assert factory.am.context
    wanted, original = recipient.record[field], donor.record[field]
    if len(wanted) != len(original):
        return None
    assignment = dict(zip(original, wanted))
    if len(assignment) != len(original) or len(set(assignment.values())) != len(assignment):
        return None
    remaining = [v for v in recipient.example.nodes if v not in assignment.values()]
    rng.shuffle(remaining)
    for old in donor.example.nodes:
        if old not in assignment:
            assignment[old] = remaining.pop()
    mapped = {f: [assignment[v] for v in donor.record[f]] for f in FIELDS}
    if not all(different_state(recipient.record, mapped, f) for f in FIELDS if f != field):
        return None
    recipe = dict(factory.recipe(donor), node_ids=[assignment[v] for v in donor.example.nodes])
    renamed = factory.restore(recipe, donor.ordinal)
    assert renamed.record[field] == recipient.record[field]
    assert all(different_state(recipient.record, renamed.record, f) for f in FIELDS if f != field)
    return renamed


def history_donor(recipient, field, factory, rng):
    """Same full trace prefix; different graph suffix OR query target.

    New graphs: freeze every node name already in the current complete trace
    prefix plus both query names. Permute ONLY future names in the graph.
    Fixed graphs: preserve the graph and all trace tokens, choose the next
    expanded node as a new target, after checking it was never expanded.
    """
    r, ex = recipient.record, recipient.example
    recipe = dict(factory.recipe(recipient))
    if factory.am.context:
        frozen = set(recipient.tokens[len(ex.prompt):r['end'] + 1]) | {ex.nodes[ex.source], ex.nodes[ex.target]}
        future = [v for v in ex.nodes if v not in frozen]
        changeable = set(recipient.following['candidates']) - frozen
        if len(future) < 2 or not changeable:
            return None
        # A cyclic permutation is bijective and changes EVERY free node.
        rng.shuffle(future)
        mapping = dict(zip(future, future[1:] + future[:1]))
        recipe['node_ids'] = [mapping.get(v, v) for v in ex.nodes]
    else:
        next_current = recipient.following['frontier'][0 if factory.am.protocol == 'bfs_state' else -1]
        new_target = ex.nodes.index(next_current)
        if new_target == ex.target:
            return None
        # Reject targets that would have stopped the reference before the cut.
        from cot_navigation.analysis.common import records
        past = records(recipient.tokens[:r['end'] + 1], len(ex.prompt), factory.am.tok)
        currents = {row['frontier'][0 if factory.am.protocol == 'bfs_state' else -1] for row in past}
        if next_current in currents:
            return None
        recipe['target'] = new_target
    rows = factory.render(recipe)
    if recipient.ordinal >= len(rows):
        return None
    donor = rows[recipient.ordinal]
    assert recipient.tokens[len(ex.prompt):r['end'] + 1] == donor.tokens[len(donor.example.prompt):donor.record['end'] + 1]
    assert len(recipient.prefix) == len(donor.prefix)
    if all(not different_state(recipient.following, donor.following, f) for f in FIELDS) and recipient.following['action'] == donor.following['action']:
        return None
    return donor


def prepare(am, *, count=128, queries=1800, variants=3, seed=20261021, max_prefix=3000):
    factory = Factory(am)
    rng = np.random.default_rng(seed)
    pool, audit = [], Counter()
    for recipe in base_recipes(am, queries, variants, seed):
        rows = factory.render(recipe)
        pool.extend(s for s in rows if len(s.prefix) <= max_prefix)
        audit['rendered_queries'] += 1
        audit['replayed_records'] += len(rows) + 1
    rng.shuffle(pool)
    audit['eligible_records'] = len(pool)
    # Index all snapshots by exact serialized value and length. Fixed matching
    # has no per-key 64-row truncation; inspect every distinct semantic state.
    exact = {f: defaultdict(list) for f in FIELDS}
    lengths = {f: defaultdict(list) for f in FIELDS}
    for s in pool:
        for f in FIELDS:
            exact[f][tuple(s.record[f])].append(s)
            lengths[f][len(s.record[f])].append(s)
    pairs, coverage = [], {}
    for field in FIELDS:
        # More candidate recipients than final quota to avoid shallow-only
        # first-match behavior and discover difficult matching states.
        recipients = select_recipients(pool, field, min(queries, count * 12), am.context, factory, rng)
        field_pairs, history_pairs = defaultdict(list), defaultdict(list)
        used_field, used_history = set(), set()
        for r in recipients:
            b = depth_bin(r)
            key = unit(factory.recipe(r), am.context)
            if key not in used_field:
                donor = None
                if am.context:
                    candidates = lengths[field][len(r.record[field])]
                    # Structural candidates that cannot differ in the remaining
                    # frontier (e.g. initial singleton VIS) are not fabricated.
                    positions = rng.choice(len(candidates), size=min(80, len(candidates)), replace=False)
                    ranked = sorted((candidates[int(i)] for i in positions), key=lambda d:
                        (d.record['action'][0] != r.record['action'][0], abs(len(d.prefix) - len(r.prefix))))
                    for d in ranked:
                        if unit(factory.recipe(d), True) == key:
                            continue
                        donor = rename_to_match(r, d, field, factory, rng)
                        if donor is not None:
                            break
                else:
                    candidates = [d for d in exact[field].get(tuple(r.record[field]), [])
                                  if unit(factory.recipe(d), False) != key and
                                  all(different_state(r.record, d.record, f) for f in FIELDS if f != field)]
                    if candidates:
                        donor = min(candidates, key=lambda d: (d.record['action'][0] != r.record['action'][0],
                                                              abs(len(d.prefix) - len(r.prefix))))
                if donor is not None:
                    used_field.add(key); field_pairs[b].append((field, r, donor))
            if key not in used_history:
                donor = history_donor(r, field, factory, rng)
                if donor is not None:
                    used_history.add(key); history_pairs[b].append((field, r, donor))
            if sum(map(len, field_pairs.values())) >= count * 2 and sum(map(len, history_pairs.values())) >= count * 2:
                break
        for name, bins in [('exact_field', field_pairs), ('shared_history', history_pairs)]:
            chosen = []
            while len(chosen) < count and any(bins.values()):
                for b in range(4):
                    if bins[b] and len(chosen) < count:
                        chosen.append(bins[b].pop())
            coverage[f'{name}/{field}'] = dict(selected=len(chosen),
                depth_bins=dict(Counter(depth_bin(r) for _, r, _ in chosen)),
                operator_identity=sum(r.record['action'][0] == d.record['action'][0] for _, r, d in chosen))
            pairs.extend((name, f, r, d) for f, r, d in chosen)
    records = []
    for cohort, field, r, d in pairs:
        assert r.record[field] == d.record[field]
        if cohort == 'exact_field':
            assert all(different_state(r.record, d.record, f) for f in FIELDS if f != field)
        else:
            assert len(r.prefix) == len(d.prefix)
            assert r.prefix[len(r.example.prompt):] == d.prefix[len(d.example.prompt):]
        records.append(dict(cohort=cohort, field=field, unit=unit(factory.recipe(r), am.context), depth_bin=depth_bin(r),
            recipient=dict(recipe=factory.recipe(r), ordinal=r.ordinal, sha256=digest(r.prefix)),
            donor=dict(recipe=factory.recipe(d), ordinal=d.ordinal, sha256=digest(d.prefix)),
            operator_identity=r.record['action'][0] == d.record['action'][0],
            prefix_gap=len(d.prefix) - len(r.prefix)))
    return records, dict(audit), coverage


def restore_pairs(am, records, readout):
    factory = Factory(am)
    pairs = []
    for item in records:
        r, d = [factory.restore(item[arm]['recipe'], item[arm]['ordinal']) for arm in ('recipient', 'donor')]
        assert digest(r.prefix) == item['recipient']['sha256']
        assert digest(d.prefix) == item['donor']['sha256']
        if readout == 'current_record':
            field = item['field']
            r = replace(r, cut_at=r.record['separators'][FIELDS.index(field)] + 1, following=r.record)
            d = replace(d, cut_at=d.record['separators'][FIELDS.index(field)] + 1, following=d.record)
        pairs.append((item['field'], r, d))
    return pairs

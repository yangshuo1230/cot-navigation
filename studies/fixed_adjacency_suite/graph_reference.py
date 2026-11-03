"""Prepare 36 exposure tasks and two event-driven one-hop-to-multihop runs."""
from __future__ import annotations
from collections import Counter, deque
import hashlib
import json
from pathlib import Path
import random

HERE = Path(__file__).resolve().parent


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def reachable(adj, s):
    seen = {s}
    todo = [s]
    for u in todo:
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                todo.append(v)
    seen.remove(s)
    return seen


def witness(adj, s, t):
    parent = {s: None}
    todo = deque([s])
    while todo:
        u = todo.popleft()
        if u == t:
            path = [u]
            while parent[path[-1]] is not None:
                path.append(parent[path[-1]])
            return path[::-1]
        for v in adj[u]:
            if v not in parent:
                parent[v] = u
                todo.append(v)
    return []


def split_labeled(rows, rng):
    groups = {}
    for r in rows:
        groups.setdefault((r[2], r[3]), []).append(r)
    out = dict(train=[], monitor=[], final=[])
    for key, values in sorted(groups.items()):
        rng.shuffle(values)
        n = len(values)
        n_train, n_monitor = int(n * .6), int(n * .2)
        if n < 5:
            out['final'].extend(values)
        else:
            out['train'].extend(values[:n_train])
            out['monitor'].extend(values[n_train:n_train + n_monitor])
            out['final'].extend(values[n_train + n_monitor:])
    return out


def balanced_panel(rows, rng, per_label=128):
    yes = [r for r in rows if r[2] == 1]
    no = [r for r in rows if r[2] == 0]
    n = min(per_label, len(yes), len(no))
    panel = rng.sample(yes, n) + rng.sample(no, n)
    rng.shuffle(panel)
    return panel


def fragments(adj, rng, max_hops=4, count=4096):
    paths = set()
    active = [u for u, vs in enumerate(adj) if vs]
    for _ in range(count * 10):
        u = rng.choice(active)
        path = [u]
        for _ in range(max_hops):
            if not adj[path[-1]]:
                break
            path.append(rng.choice(adj[path[-1]]))
        if len(path) > 1:
            paths.add(tuple(path))
        if len(paths) >= count:
            break
    return [list(p) for p in sorted(paths)]


def direct_facts(rows, paths):
    facts = {(r[0], r[1]) for r in rows if r[2]}
    for p in paths:
        facts.update((u, v) for i, u in enumerate(p) for v in p[i+1:])
    return facts


def make_world(seed):
    # Strict adjacent-layer graph: path lengths and exposure bounds are exact.
    width, layers, n_edges = 16, 10, 320
    n = width * layers
    candidates = [(u, v) for u in range(n-width)
                  for v in range((u//width+1)*width, (u//width+2)*width)]
    for attempt in range(100):
        rng = random.Random(10_000 + seed*1000 + attempt)
        edges = sorted(rng.sample(candidates, n_edges))
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
        reach = [reachable(adj, u) for u in range(n)]
        source_B, target_B = set(), set()
        for layer in range(layers):
            source_B.update(rng.sample(range(layer*width, (layer+1)*width), width//2))
            target_B.update(rng.sample(range(layer*width, (layer+1)*width), width//2))
        rows = [[s, t, int(t in reach[s]), t//width-s//width]
                for s in range(n) for t in range(n) if t//width > s//width]
        core = [r for r in rows if r[3] >= 6 and r[0] in source_B and r[1] in target_B]
        if min(sum(r[2] == y for r in core) for y in (0, 1)) < 128:
            continue
        rng.shuffle(core)
        labels = [[r for r in core if r[2] == y] for y in (0, 1)]
        if min(map(len, labels)) < 160:
            continue
        final = balanced_panel([r for group in labels for r in group[:128]], rng)
        monitor = balanced_panel([r for group in labels for r in group[128:]], rng, 32)
        base = [r for r in rows if r[3] <= 2 and not (r[0] in source_B and r[1] in target_B)]
        base_split = split_labeled(base, rng)
        if any(sum(r[2] == y for r in base_split['train']) == 0 for y in (0, 1)):
            continue
        node_names = rng.sample(range(256), n)
        local = [[u, v] for u, v in edges]
        short = fragments(adj, rng)
        positive_witnesses = [witness(adj, r[0], r[1]) for r in final + monitor if r[2]]
        # R3 explicitly exposes both monitor and final positives; it is an upper bound.
        all_pairs = split_labeled(rows, random.Random(20_000 + seed))
        one = {k: [r for r in v if r[3] == 1] for k, v in all_pairs.items()}
        multi = {k: [r for r in v if r[3] >= 2] for k, v in all_pairs.items()}
        allowed_paths = {
            'R0': local,
            'R2': local + short,
            'R3': local + positive_witnesses,
        }
        audits = {}
        # Enumerate every allowed PATH witness, so changing variants cannot leak later.
        train_paths = [witness(adj, r[0], r[1]) for r in base_split['train'] if r[2]]
        for arm, paths in allowed_paths.items():
            facts = direct_facts(base_split['train'], paths + train_paths)
            scores = []
            H = [[] for _ in adj]
            for u, v in facts:
                H[u].append(v)
            disclosed = sum((r[0], r[1]) in facts for r in final if r[2])
            for r in final:
                if r[2]:
                    scores.append(len(witness(H, r[0], r[1])) - 1)
            if arm != 'R3':
                assert disclosed == 0
            audits[arm] = dict(final_positive_directly_disclosed=disclosed,
                macro_composition_distance_counts=dict(Counter(scores)),
                allowed_fragment_count=len(paths),
                max_fragment_hops=max(len(p)-1 for p in paths))
        # Independent reverse-topological DP agrees with traversal for all pairs.
        bitsets = [0]*n
        for u in reversed(range(n)):
            for v in adj[u]:
                bitsets[u] |= (1 << v) | bitsets[v]
        assert all(bool(bitsets[s] & (1 << t)) == bool(y) for s, t, y, gap in rows)
        return dict(seed=seed, attempt=attempt, width=width, layers=layers,
            n_nodes=n, n_edges=n_edges, node_names=node_names, adj=adj, edges=edges,
            source_B=sorted(source_B), target_B=sorted(target_B),
            base=base_split, core_monitor=monitor, core_final=final,
            fragments=allowed_paths, one_hop=one, multihop=multi,
            audit=audits, local_knowledge=local,
            notes=['Knowledge is positive one-edge WALK, no EOS loss or multi-hop WALK in Stage A',
                   'PATH negatives have empty body; no negative random-walk exposure',
                   'R3 is intentionally exposed; do not label its final set unseen',
                   'Core final/monitor length >=6; base QUERY <=2; R2 fragments <=4'])
    raise ValueError(f'No feasible graph for seed {seed}; do not silently shrink holdouts')


def main():
    data = HERE/'data'
    configs = HERE/'configs'
    data.mkdir(exist_ok=True)
    configs.mkdir(exist_ok=True)
    worlds=[]
    for seed in range(3):
        w = make_world(seed)
        path=data/f'graph_{seed}.json'
        text=json.dumps(w, separators=(',', ':'))
        path.write_text(text)
        worlds.append(dict(seed=seed,path=path.relative_to(HERE).as_posix(),sha256=digest(w),
                           audit=w['audit'], n_nodes=w['n_nodes'], n_edges=w['n_edges']))
    jobs=[]
    for graph in range(3):
        for model_seed in range(2):
            knowledge=f'knowledge/g{graph}_s{model_seed}'
            jobs.append(dict(id=knowledge,kind='knowledge',graph=graph,model_seed=model_seed,
                protocol='direct',arm='R0',steps=1000,dependencies=[],gpu_pool=list(range(6))))
            for protocol in ['direct','path']:
                for arm in ['R0','R2','R3']:
                    jobs.append(dict(id=f'fixed/g{graph}_s{model_seed}/{protocol}/{arm}',
                        kind='fixed',graph=graph,model_seed=model_seed,protocol=protocol,
                        arm=arm,steps=125_000,dependencies=[knowledge],gpu_pool=list(range(6))))
    for p in (0.0,0.1):
        jobs.append(dict(id=f'grokking/d{p:g}_w0.1',kind='grokking',graph=0,model_seed=0,
            protocol='direct',arm='R0',steps=None,dropout=p,weight_decay=.1,
            dependencies=['knowledge/g0_s0'],gpu_pool=[6 if p==0 else 7],
            phase1_query='one_hop.train',phase1_monitor='one_hop.monitor',
            phase2_query='multihop.train',phase2_monitor='multihop.monitor',
            gate=dict(train_accuracy=.99,monitor_ba=.95,min_class_recall=.95,consecutive=3),
            maximum_steps=None))
    for job in jobs:
        job.update(lr=1e-4,warmup_steps=1000,query_batch=64,knowledge_transition_budget=64,
            eval_interval=20_000,checkpoint_interval=10_000,log_interval=100,
            dropout=job.get('dropout',0.),weight_decay=job.get('weight_decay',.01))
        (configs/(job['id'].replace('/','__')+'.json')).write_text(json.dumps(job,indent=2))
    manifest=dict(version=2,status='prepared_not_launched',worlds=worlds,jobs=jobs,
        fixed_task_count=36,shared_knowledge_count=6,grokking_count=2,
        model=dict(layers=6,heads=1,d_model=128,d_ff=512,max_seq_len=256),
        criterion_note='Monitor split controls the phase transition; final split never controls it',
        policies=dict(final_fixed_steps=125_000,grokking_no_step_cap=True,
            phase2_no_step_cap=True,wandb='online',gpu_0_5='fixed queue',gpu_6_7='grokking reserved'))
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(status='PREPARED',worlds=worlds,jobs=len(jobs)),indent=2))


if __name__=='__main__':main()

"""Minimal, architecture-preserving query coverage and small-graph experiments.

Run from the remote repository root through launch.sh. No training occurs on
import or with --check-only. Existing datasets, source and runs stay untouched.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from cot_navigation.config import load_config
from cot_navigation.data.dataset import ContextStore, collate_examples
from cot_navigation.data.reachability import bfs_reaches, build_reach, reaches
from cot_navigation.evaluation.rollout import generate
from cot_navigation.protocols.render import context_prompt, render_task
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.training.augmentation import transform
from cot_navigation.training.sampler import IndexSampler
from cot_navigation.training.trainer import Trainer

HERE = Path(__file__).resolve().parent
ROOT = Path.cwd()
BASE = ROOT / 'outputs/explore_v1/launches/explore_v1_s012/context/standard/train50000'
DATA = ROOT / 'data/explore_v1/context_standard_n16_m24_train50000_val10000_test10000_s0'
OLD = ROOT / 'studies/length_generalization'
sys.path.insert(0, str(OLD))
from trace_audit import validate_trace, validate_execution

TOK = Tokenizer()
SEED = 20261017


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def adjacency(record):
    adj = [[] for _ in record['nodes']]
    for u, v in record['edges']:
        adj[u].append(v)
    return adj


def record_from_store(store, index):
    a = store.arrays
    return dict(nodes=store.node_vocab_ids(index, TOK),
                edges=list(map(list, zip(a['edge_src'][index].astype(int).tolist(),
                                         a['edge_dst'][index].astype(int).tolist()))),
                s=int(a['s_topo'][index]), t=int(a['t_topo'][index]),
                h=int(a['distance'][index]), label=int(a['label'][index]), index=int(index))


def render(record, protocol, index=-1):
    row = render_task(TOK, protocol, record['label'],
                      context_prompt(TOK, record['nodes'], record['edges'], record['s'], record['t']),
                      adjacency(record), record['s'], record['t'], record['nodes'])
    row.meta.update(index=index, distance=record['h'], shape=f"n{len(record['nodes'])}")
    return row


def balanced_original(store, split, count):
    """Independent deterministic subset; selection never consults predictions."""
    ids = store.indices(split)
    a = store.arrays
    result = []
    for label in (0, 1):
        eligible = ids[(a['distance'][ids] == 1) & (a['label'][ids] == label)]
        if len(eligible) < count // 2:
            raise ValueError('insufficient one-hop examples')
        result.extend(record_from_store(store, int(i)) for i in eligible[:count // 2])
    return result


def graph_key(record):
    # Pre-rename topology identity, not a general graph-isomorphism test.
    return len(record['nodes']), tuple(sorted(tuple(e) for e in record['edges']))


def make_direct_data(store, n_graphs):
    rng = np.random.default_rng(SEED)
    graphs = balanced_original(store, 'train', n_graphs)
    rng.shuffle(graphs)
    seen_unseen = []
    for gi, graph in enumerate(graphs):
        edges = set(map(tuple, graph['edges']))
        pairs = [(u, v) for u in range(16) for v in range(16) if v // 4 == u // 4 + 1]
        anchor = (graph['s'], graph['t'])
        groups = {y: [p for p in pairs if int(p in edges) == y and p != anchor] for y in (0, 1)}
        queries = [list(anchor)]
        for y in (0, 1):
            count = 4 - int(y == graph['label'])
            queries.extend(list(groups[y][j]) for j in rng.choice(len(groups[y]), count, replace=False))
        assert len(queries) == 8 and len(set(map(tuple, queries))) == 8
        graph['queries'] = queries  # Anchor remains slot 0 for the single arm.
        graph['query_order'] = rng.permutation(8).tolist()
        available = [p for p in pairs if p not in set(map(tuple, queries))]
        # A shared diagnostic holdout excludes ALL eight multi-query labels.
        if gi < min(n_graphs, 256):
            for y in (0, 1):
                s, t = next(p for p in available if int(p in edges) == y)
                seen_unseen.append({**graph, 's': s, 't': t, 'label': y, 'graph_id': gi})
    val = balanced_original(store, 'val', 512)
    test = balanced_original(store, 'test', 1024)
    keys = [set(map(graph_key, part)) for part in (graphs, val, test)]
    assert not (keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2])
    return dict(graphs=graphs, seen_unseen=seen_unseen, val=val, test=test)


def make_small_pool(per_stratum):
    """Four layers throughout; sample each (size, hop, label) explicitly.

    Rejection handles empty positive strata without changing the requested hop
    or label. Node names remain uniformly sampled from the original vocabulary.
    """
    rng = np.random.default_rng(SEED + 7)
    result = {}
    for width, m in ((2, 6), (3, 12)):
        n = 4 * width
        possible = [(u, v) for u in range(n) for v in range(n) if v // width == u // width + 1]
        for hop in (1, 2, 3):
            for label in (0, 1):
                rows = []
                while len(rows) < per_stratum:
                    edges = [possible[i] for i in rng.choice(len(possible), m, replace=False)]
                    rec = dict(nodes=rng.choice(np.arange(TOK.node_lo, len(TOK)), n, replace=False).tolist(),
                               edges=[list(e) for e in edges], h=hop, label=label)
                    adj = adjacency(rec)
                    reach = build_reach(adj)
                    pairs = [(u, v) for u in range(n) for v in range(n)
                             if v // width - u // width == hop and int(reaches(reach, u, v)) == label]
                    if not pairs:
                        continue
                    s, t = pairs[int(rng.integers(len(pairs)))]
                    rec.update(s=s, t=t)
                    assert bfs_reaches(adj, s, t) == bool(label)
                    rows.append(rec)
                result[f'{n}/{hop}/{label}'] = rows
    return result


class DirectStore:
    """Small in-memory adapter to the existing trainer, not a new data format."""
    def __init__(self, data, queries_per_graph):
        self.records = []
        self.parts = {}
        for gi, graph in enumerate(data['graphs']):
            edges = set(map(tuple, graph['edges']))
            for s, t in graph['queries'][:queries_per_graph]:
                self.records.append({**graph, 's': s, 't': t, 'label': int((s, t) in edges), 'graph_id': gi})
        self.parts['train'] = np.arange(len(self.records))
        for name in ('val', 'test', 'seen_unseen'):
            start = len(self.records)
            self.records.extend(data[name])
            self.parts[name] = np.arange(start, len(self.records))
        self.arrays = {k: np.array([r[field] for r in self.records])
                       for k, field in (('distance', 'h'), ('label', 'label'), ('s_topo', 's'), ('t_topo', 't'))}
        self.arrays['shape_id'] = np.zeros(len(self.records), dtype=int)
        self.fingerprint = digest(self.records)

    def indices(self, split):
        return self.parts[split]

    def render(self, index, protocol, tok, **kwargs):
        return render(self.records[index], protocol, index)

    def adj(self, index):
        return adjacency(self.records[index])

    def node_vocab_ids(self, index, tok):
        return self.records[index]['nodes']


class GraphSampler:
    """Identical graph stream across arms; cycle distinct queries per graph.

    Each graph receives exactly the same number of exposures in both arms.
    Multi-query coverage is exhaustive in every eight visits to that graph.
    """
    def __init__(self, graphs, k, batch, seed):
        self.sampler = IndexSampler(len(graphs), batch, seed)
        self.visits = np.zeros(len(graphs), dtype=np.int64)
        self.orders = np.array([g['query_order'] for g in graphs])
        self.k = k

    def next(self):
        result = []
        for gi in self.sampler.next():
            slot = 0 if self.k == 1 else int(self.orders[gi, self.visits[gi] % 8])
            result.append(gi * self.k + slot)
            self.visits[gi] += 1
        return result

    def state_dict(self):
        return dict(sampler=self.sampler.state_dict(), visits=self.visits, k=self.k)

    def load_state_dict(self, state):
        assert state['k'] == self.k
        self.sampler.load_state_dict(state['sampler'])
        self.visits = state['visits'].copy()


class DirectTrainer(Trainer):
    def __init__(self, cfg, data, k, resume=None):
        self.study_data, self.k = data, k
        super().__init__(cfg, resume=resume)

    def _load_data(self):
        self.store = DirectStore(self.study_data, self.k)
        self.train_ids = self.store.indices('train')
        self.val_ids = self.store.indices('val')

    def _preload_training_rows(self):
        self.prepared_tasks = self.prepared_walks = None
        for i in self.train_ids:
            self._render_task(int(i))

    def _init_samplers(self):
        super()._init_samplers()
        self.query_sampler = GraphSampler(self.study_data['graphs'], self.k,
                                         self.cfg.train.effective_query_batch_size,
                                         self.cfg.train.exposure_seed + 3)

    def _evaluate(self, kind):
        super()._evaluate(kind)
        if kind == 'teacher_forced':
            direct_diagnostics(self, final=False)


def log_study(trainer, values):
    """Persist diagnostics and upload explicitly beyond the framework allowlist."""
    path = trainer.out / 'study_metrics.jsonl'
    with path.open('a') as f:
        f.write(json.dumps(dict(step=trainer.completed, **values)) + '\n')
    if trainer.tracker.run:
        trainer.tracker.run.log(dict(optimizer_step=trainer.completed, **values))


def direct_diagnostics(trainer, final):
    trainer.raw.eval()
    trainer.model.eval()
    groups = {'train_seen': trainer._selected_ids('train', 512),
              'seen_graph_unseen_query': trainer._selected_ids('seen_unseen', 512)}
    if final:
        groups['new_graph_test'] = trainer._selected_ids('test', 0)
    reports = {}
    # The exact same original queries are scored in both arms; train_seen above
    # measures each arm's own supervision and therefore has different queries.
    anchors = trainer.study_data['graphs'][:512]
    reports['shared_anchor_queries'] = trainer._teacher_forced(render(r, 'direct') for r in anchors)
    log_study(trainer, {'diagnostic/shared_anchor_queries/accuracy':
                       reports['shared_anchor_queries']['overall']['answer_accuracy']})
    for name, ids in groups.items():
        reports[name] = trainer._teacher_forced(trainer._render_task(i) for i in ids)
        log_study(trainer, {f'diagnostic/{name}/accuracy': reports[name]['overall']['answer_accuracy']})
    if final:
        write_json(trainer.out / 'direct_diagnostics.json', reports)
    trainer.raw.train()
    trainer.model.train()


class SmallGraphTrainer(Trainer):
    def __init__(self, cfg, mode, pool, resume=None):
        self.mix_mode, self.small_pool, self.mix_quota = mode, pool, []
        self.small_rows = {key: [render(r, cfg.protocol.name) for r in rows] for key, rows in pool.items()}
        self.mix_counts = Counter()
        super().__init__(cfg, resume=resume)

    def _init_samplers(self):
        super()._init_samplers()
        self.augmentation_rngs['small_graph'] = np.random.default_rng(SEED + 23)

    def _payload(self):
        blob = super()._payload()
        blob['small_graph_state'] = dict(mode=self.mix_mode, quota=self.mix_quota, counts=dict(self.mix_counts))
        return blob

    def _load_resume(self, path):
        super()._load_resume(path)
        state = torch.load(path, map_location='cpu', weights_only=False)['small_graph_state']
        assert state['mode'] == self.mix_mode
        self.mix_quota, self.mix_counts = state['quota'], Counter(state['counts'])

    def _training_task(self, index):
        # Always consume the same original row/edge RNG stream in both arms.
        original = super()._training_task(index)
        rng = self.augmentation_rngs['small_graph']
        if not self.mix_quota:
            batch = self.cfg.train.effective_query_batch_size
            assert batch % 4 == 0
            self.mix_quota = rng.permutation([16] * (batch // 2) + [12] * (batch // 4)
                                             + [8] * (batch // 4)).tolist()
        planned_size = self.mix_quota.pop()
        size = planned_size if self.mix_mode == 'small_mix' else 16
        self.mix_counts[str(size)] += 1
        if size == 16:
            return original
        # Preserve the sampled row's hop and label, rather than biasing toward
        # easy short-hop or negative samples when smaller graphs are selected.
        key = f"{size}/{original.meta['distance']}/{original.label}"
        j = int(rng.integers(len(self.small_pool[key])))
        rec, row = self.small_pool[key][j], self.small_rows[key][j]
        row = transform(row, self.cfg.protocol.name, TOK, adjacency(rec), rec['nodes'], rec['s'], rec['t'], edge_rng=rng)
        self._check_length(row)
        return row


def length_evaluate(trainer, per_label_hop):
    metadata = json.loads((OLD / 'results_n32/metadata.json').read_text())
    cases = ('id', 'id_disconnected_pad32', 'wide32_branch2', 'deep16_sameprompt', 'deep24', 'deep32')
    path = trainer.out / 'length_predictions.jsonl'
    # Final-only evaluation is deterministic and may be rerun after interruption.
    path.write_text('')
    trainer.raw.eval()
    reports = {}
    for case in cases:
        counts, records = Counter(), []
        for r in metadata['inputs'][case]:
            if counts[(r['h'], r['label'])] < per_label_hop:
                counts[(r['h'], r['label'])] += 1
                records.append(r)
        totals, by_hop = Counter(), {}
        for start in range(0, len(records), 16):
            chunk = records[start:start + 16]
            rows = [render(r, trainer.cfg.protocol.name) for r in chunk]
            output = generate(trainer.raw, [r.tokens[:r.prompt_len] for r in rows], trainer.cfg.protocol.name,
                              TOK, 'bf16', max_new_tokens=0)
            for rec, row, gen in zip(chunk, rows, output):
                token_adj = {node: [] for node in rec['nodes']}
                for u, v in rec['edges']:
                    token_adj[rec['nodes'][u]].append(rec['nodes'][v])
                s, t = rec['nodes'][rec['s']], rec['nodes'][rec['t']]
                valid = validate_trace(gen['trace_tokens'], gen['pred'], trainer.cfg.protocol.name, token_adj, s, t, TOK)
                execution = validate_execution(gen['trace_tokens'], gen['pred'], token_adj, s, t, TOK,
                                               trainer.cfg.protocol.name == 'bfs_state')
                item = dict(case=case, index=rec['index'], hop=rec['h'], label=rec['label'], pred=gen['pred'],
                            correct=gen['pred'] == rec['label'] and not gen['format_fail'],
                            trace_valid=bool(valid['valid']), execution_valid=bool(execution),
                            trace_error=valid['error'], format_fail=gen['format_fail'],
                            limit=gen['generation_limit_reached'], trace_tokens=gen['trace_tokens'])
                with path.open('a') as f:
                    f.write(json.dumps(item) + '\n')
                counters = [totals, by_hop.setdefault(str(rec['h']), Counter())]
                for counter in counters:
                    counter['n'] += 1
                    for metric in ('correct', 'trace_valid', 'execution_valid', 'format_fail', 'limit'):
                        counter[metric] += int(item[metric])
        def rates(counter):
            return {'n': counter['n'], **{k: v / counter['n'] for k, v in counter.items() if k != 'n'}}
        reports[case] = dict(overall=rates(totals), by_hop={h: rates(c) for h, c in by_hop.items()})
        log_study(trainer, {f'length/{case}/{k}': v for k, v in rates(totals).items() if k != 'n'})
        print('LENGTH EVAL', case, reports[case]['overall'], flush=True)
    write_json(trainer.out / 'length_summary.json', reports)


def check_data(direct, pool, store):
    """CPU-only oracle/serialization/mask/sampler/resume checks, no run files."""
    n = 0
    for graph in direct['graphs']:
        adj = adjacency(graph)
        assert len(graph['nodes']) == len(set(graph['nodes'])) == 16
        assert len(set(map(tuple, graph['edges']))) == 24
        assert bfs_reaches(adj, graph['s'], graph['t']) == bool(graph['label'])
        assert sum(bfs_reaches(adj, s, t) for s, t in graph['queries']) == 4
        for s, t in graph['queries']:
            assert t // 4 - s // 4 == 1
        n += 8
    for part in ('val', 'test', 'seen_unseen'):
        for rec in direct[part]:
            assert bfs_reaches(adjacency(rec), rec['s'], rec['t']) == bool(rec['label'])
    for key, records in pool.items():
        size, hop, label = map(int, key.split('/'))
        for rec in records:
            assert len(rec['nodes']) == size <= 16
            assert len(set(rec['nodes'])) == size
            assert len(rec['edges']) == (6 if size == 8 else 12) <= 24
            assert len(set(map(tuple, rec['edges']))) == len(rec['edges'])
            assert rec['t'] // (size // 4) - rec['s'] // (size // 4) == hop <= 3
            assert bfs_reaches(adjacency(rec), rec['s'], rec['t']) == bool(label)
            assert all(v // (size // 4) == u // (size // 4) + 1 for u, v in rec['edges'])
            n += 1
        for protocol in ('bfs_state', 'dfs_state'):
            for rec in records[:4]:
                row = render(rec, protocol)
                row = transform(row, protocol, TOK, adjacency(rec), rec['nodes'], rec['s'], rec['t'],
                                edge_rng=np.random.default_rng(11))
                token_adj = {node: [] for node in rec['nodes']}
                for _, u, v in np.array(row.tokens[:row.prompt_len - 4]).reshape(-1, 3):
                    token_adj[int(u)].append(int(v))
                assert validate_trace(row.tokens[row.prompt_len:row.answer_index], row.label, protocol,
                                      token_adj, rec['nodes'][rec['s']], rec['nodes'][rec['t']], TOK)['valid']
                batch = collate_examples([row], TOK.pad)
                assert batch['answer_mask'][0].sum() == 1
                assert batch['answer_mask'][0, row.answer_index] == 1
                assert batch['trace_mask'][0].sum() == len(row.trace_indices)
                assert not batch['answer_mask'][0, :row.prompt_len].any()
                assert not batch['trace_mask'][0, :row.prompt_len].any()
    a, b = [GraphSampler(direct['graphs'], k, 64, 3) for k in (1, 8)]
    for _ in range(max(10, 8 * len(direct['graphs']) // 64)):
        x, y = a.next(), b.next()
        assert x == [i // 8 for i in y]
    state = b.state_dict()
    clone = GraphSampler(direct['graphs'], 8, 64, 3)
    clone.load_state_dict(state)
    assert b.next() == clone.next()
    # Multiple examples in one batch must not mutate the cached row.
    row = render(direct['graphs'][0], 'direct')
    masks = collate_examples([row], TOK.pad)
    assert masks['answer_mask'].sum() == 1 and masks['trace_mask'].sum() == 0
    # Exercise the actual online replacement method, without constructing a
    # model/optimizer/tracker or running any training step.
    cfg = load_config([BASE / 'bfs_state/bfs_state/resolved_config.yaml'])
    controls = []
    for mode in ('control', 'small_mix'):
        shim = SmallGraphTrainer.__new__(SmallGraphTrainer)
        shim.cfg, shim.store, shim.world, shim.tok = cfg, store, None, TOK
        shim.render_cache, shim.prepared_tasks, shim.path_rng = {}, None, None
        shim.small_pool = pool
        shim.small_rows = {key: [render(r, 'bfs_state') for r in rows] for key, rows in pool.items()}
        shim.mix_mode, shim.mix_quota, shim.mix_counts = mode, [], Counter()
        shim.augmentation_rngs = {name: np.random.default_rng(seed) for name, seed in
                                 (('edges', 701), ('search', 709), ('negative', 719), ('small_graph', SEED + 23))}
        controls.append(shim)
    indices = np.random.default_rng(SEED).choice(store.indices('train'), 64, replace=False)
    for index in indices:
        original, changed = [shim._training_task(int(index)) for shim in controls]
        assert changed.label == original.label
        assert changed.meta['distance'] == original.meta['distance']
        assert changed.prompt_len in (22, 40, 76)
        if changed.prompt_len == 76:
            assert changed.tokens == original.tokens
    assert controls[1].mix_counts == Counter({'16': 32, '12': 16, '8': 16})
    assert controls[0].mix_counts == Counter({'16': 64})
    assert digest(controls[0].augmentation_rngs['edges'].bit_generator.state) == digest(
        controls[1].augmentation_rngs['edges'].bit_generator.state)
    # A partially consumed batch quota and independent RNG restore exactly.
    import copy
    controls[1]._training_task(int(indices[0]))
    restored = copy.copy(controls[1])
    restored.mix_quota = list(controls[1].mix_quota)
    restored.mix_counts = controls[1].mix_counts.copy()
    restored.augmentation_rngs = copy.deepcopy(controls[1].augmentation_rngs)
    for index in indices[:10]:
        assert controls[1]._training_task(int(index)).tokens == restored._training_task(int(index)).tokens
    print(f'CHECK PASSED: {n} query labels; full-state traces/masks; paired graph stream; sampler resume.', flush=True)
    print('CHECK PASSED: online small-graph replacement, 32/16/16 quota, hop/label preservation, RNG/quota restore.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', choices=('direct_single', 'direct_multi', 'bfs_control', 'bfs_small_mix',
                                        'dfs_control', 'dfs_small_mix'))
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--graphs', type=int, default=4096)
    parser.add_argument('--small-per-stratum', type=int, default=256)
    parser.add_argument('--direct-steps', type=int, default=10000)
    parser.add_argument('--cot-steps', type=int, default=5000)
    parser.add_argument('--eval-per-label-hop', type=int, default=16)
    parser.add_argument('--run-name', default='minimal_query_smallgraph_v1')
    parser.add_argument('--tracking', choices=('online', 'offline', 'disabled'), default='online')
    args = parser.parse_args()
    if args.graphs < 2 or args.graphs % 2 or args.small_per_stratum < 1:
        parser.error('--graphs must be positive/even; --small-per-stratum must be positive')
    if min(args.direct_steps, args.cot_steps, args.eval_per_label_hop) < 1:
        parser.error('step/evaluation budgets must be positive')
    if not args.run_name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_name):
        parser.error('--run-name must contain only letters, digits, underscore or hyphen')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    store = ContextStore(DATA)
    direct, pool = make_direct_data(store, args.graphs), make_small_pool(args.small_per_stratum)
    if args.check_only:
        check_data(direct, pool, store)
        return
    if not args.job:
        parser.error('--job is required unless --check-only')
    study = HERE / 'runs' / args.run_name
    out = study / args.job
    if (out / 'completed.json').exists():
        print('SKIP completed', out, flush=True)
        return
    is_direct = args.job.startswith('direct_')
    protocol = 'direct' if is_direct else ('bfs_state' if args.job.startswith('bfs_') else 'dfs_state')
    mode = args.job.split('_', 1)[1]
    suffix = 'direct_onehop/direct' if is_direct else ('bfs_state/bfs_state' if protocol == 'bfs_state' else 'baseline/dfs_state')
    original = BASE / suffix
    steps = args.direct_steps if is_direct else args.cot_steps
    parent = None if is_direct else original / 'checkpoints/latest.pt'
    spec = dict(version=1, job=args.job, graphs=args.graphs, seed=SEED, model_seed=0, steps=steps,
                small_per_stratum=args.small_per_stratum, eval_per_label_hop=args.eval_per_label_hop,
                direct_data_hash=digest(direct), small_pool_hash=digest(pool),
                parent=None if parent is None else str(parent),
                parent_sha256=None if parent is None else hashlib.sha256(parent.read_bytes()).hexdigest(),
                train_sizes=[16] if is_direct or mode == 'control' else [16, 12, 8],
                train_edges=[24] if is_direct or mode == 'control' else [24, 12, 6],
                max_query_hop=1 if is_direct else 3, fresh_optimizer=True,
                architecture='unchanged: 6 layers, 1 head, d128, FFN512', tracking=args.tracking)
    spec_path = out / 'experiment.json'
    if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
        raise ValueError('existing experiment differs; use a new --run-name')
    out.mkdir(parents=True, exist_ok=True)
    write_json(spec_path, spec)
    # Data evidence is shared by arms and allows independent follow-up audits.
    if not (study / 'direct_data.json').exists():
        write_json(study / 'direct_data.json', direct)
        write_json(study / 'small_pool.json', pool)
    overrides = [f'output_dir={out}', f'train.max_steps={steps}', 'train.model_seed=0',
                 'train.eval_interval=1000', 'train.save_interval=1000', 'train.keep_last=1',
                 'train.log_interval=100', 'train.eval_samples=256',
                 f'train.free_gen_samples={256 if is_direct else 0}',
                 'evaluation.final_checkpoints=[]', 'model.compile=true',
                 f'tracking.mode={args.tracking}', f'tracking.group={args.run_name}',
                 f'tracking.name={args.run_name}/{args.job}', f'tracking.variant={args.job}',
                 'tracking.online_failure=error', 'tracking.upload_artifacts=false']
    if not is_direct:
        overrides.extend(['train.lr=0.0001', 'train.min_lr=0.0001', 'train.warmup_steps=100'])
    cfg = load_config([original / 'resolved_config.yaml'], overrides)
    cfg.tracking.notes = json.dumps(spec)
    latest = out / 'checkpoints/latest.pt'
    resume = str(latest) if latest.exists() else None
    if is_direct:
        trainer = DirectTrainer(cfg, direct, 1 if mode == 'single' else 8, resume=resume)
    else:
        trainer = SmallGraphTrainer(cfg, mode, pool, resume=resume)
        if not resume:
            blob = torch.load(parent, map_location='cpu', weights_only=False)
            assert blob['step'] == 50000
            trainer.raw.load_state_dict(blob['model'])
            trainer.parent_run_id = blob.get('wandb_run_id')
            # Tracker has already initialized: record lineage explicitly too.
            if trainer.tracker.run:
                trainer.tracker.run.config.update({'source_checkpoint': str(parent),
                                                   'source_wandb_run_id': trainer.parent_run_id})
            del blob
    initial_hash = hashlib.sha256(b''.join(p.detach().cpu().contiguous().numpy().tobytes()
                                          for p in trainer.raw.parameters())).hexdigest()
    if not resume:
        write_json(out / 'initialization.json', dict(model_sha256=initial_hash, parent=spec['parent']))
    if trainer.tracker.run:
        trainer.tracker.run.define_metric('diagnostic/*', step_metric='optimizer_step', hidden=False)
        trainer.tracker.run.define_metric('length/*', step_metric='optimizer_step', hidden=False)
    finish = trainer.tracker.finish
    trainer.tracker.finish = lambda: None
    start = time.time()
    try:
        trainer.fit()
        trainer.model = trainer.raw  # Avoid evaluation compilation/recompilation.
        if is_direct:
            direct_diagnostics(trainer, final=True)
            trainer.evaluate_split('test', checkpoint_tag='last')
            sampler = trainer.query_sampler
            write_json(out / 'graph_exposure.json', dict(
                queries_seen=trainer.queries_seen, visits=sampler.visits.tolist(),
                graph_order_hash=digest(sampler.sampler.order.tolist()),
                graph_rng_hash=digest(sampler.sampler.rng.bit_generator.state),
                min_graph_visits=int(sampler.visits.min()), max_graph_visits=int(sampler.visits.max())))
        else:
            length_evaluate(trainer, args.eval_per_label_hop)
            write_json(out / 'mix_counts.json', dict(trainer.mix_counts))
        write_json(out / 'completed.json', dict(step=trainer.completed, seconds=time.time() - start))
    finally:
        finish()
    print('COMPLETED', args.job, out, flush=True)


if __name__ == '__main__':
    main()

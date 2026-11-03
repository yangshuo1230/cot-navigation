"""Two pure-Direct runs: paired M24 vs paired edge-count curriculum.

No model changes, auxiliary loss, CoT or extra tokens. Run launch.sh manually;
--check-only checks CPU data/samplers and never constructs a training run.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import copy, deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from cot_navigation.config import load_config
from cot_navigation.data.dataset import ContextStore, collate_examples
from cot_navigation.data.reachability import bfs_reaches, build_reach, reaches
from cot_navigation.protocols.render import RenderedExample, context_prompt
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.training.sampler import IndexSampler
from cot_navigation.training.trainer import Trainer

HERE = Path(__file__).resolve().parent
BASE = Path('outputs/explore_v1/launches/explore_v1_s012/context/standard/train50000/direct_onehop/direct')
DATA = Path('data/explore_v1/context_standard_n16_m24_train50000_val10000_test10000_s0')
TOK = Tokenizer()
SIZES = (2, 4, 8, 16, 24)
SEED = 20261018
LABELS = [1, 0, 0, 1, 0, 1, 1, 0]


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def make_groups(count, seed):
    """Each group is G/G' with four identical corner queries per graph.

    Core endpoints occupy one adjacent-layer 2x2 rectangle. G has its diagonal
    edges; G' has its off-diagonal edges. All other edges are shared. Distractors
    never occupy the other two corners. Larger sizes nest the smaller graphs.
    """
    rng = np.random.default_rng(seed)
    possible = [(u, v) for u in range(16) for v in range(16) if v // 4 == u // 4 + 1]
    groups = []
    for _ in range(count):
        layer = int(rng.integers(3))
        a, c = (rng.choice(4, 2, replace=False) + layer * 4).tolist()
        b, d = (rng.choice(4, 2, replace=False) + (layer + 1) * 4).tolist()
        corners = {(a, b), (a, d), (c, b), (c, d)}
        others = [edge for edge in possible if edge not in corners]
        background = [list(others[i]) for i in rng.permutation(len(others))[:22]]
        groups.append(dict(nodes=rng.choice(np.arange(TOK.node_lo, len(TOK)), 16, replace=False).tolist(),
                           queries=[[a, b], [a, d], [c, b], [c, d]],
                           core=[[[a, b], [c, d]], [[a, d], [c, b]]], background=background))
    return groups


def record(group, m, variant):
    graph, query = divmod(variant, 4)
    s, t = group['queries'][query]
    return dict(nodes=group['nodes'], edges=group['core'][graph] + group['background'][:m - 2],
                s=s, t=t, label=LABELS[variant])


def row(rec, index=-1):
    prefix = context_prompt(TOK, rec['nodes'], rec['edges'], rec['s'], rec['t'])
    return RenderedExample(tokens=prefix + [TOK.label_id(rec['label']), TOK.id('<EOS>')],
                           prompt_len=len(prefix), answer_index=len(prefix),
                           trace_indices=[], walk_indices=[], label=rec['label'],
                           meta=dict(index=index, distance=1, shape='n16', edges=len(rec['edges'])))


def ordinary_records(store, split, n):
    ids, arrays = store.indices(split), store.arrays
    result = []
    for y in (0, 1):
        chosen = ids[(arrays['distance'][ids] == 1) & (arrays['label'][ids] == y)][:n // 2]
        if len(chosen) != n // 2:
            raise ValueError('insufficient ordinary held-out one-hop queries')
        for index in chosen:
            result.append(dict(nodes=store.node_vocab_ids(int(index), TOK),
                edges=list(map(list, zip(arrays['edge_src'][index].tolist(), arrays['edge_dst'][index].tolist()))),
                s=int(arrays['s_topo'][index]), t=int(arrays['t_topo'][index]), label=y))
    return result


def make_data(n_train):
    store = ContextStore(DATA)
    return dict(train=make_groups(n_train, SEED), val=make_groups(128, SEED + 1),
                test=make_groups(128, SEED + 2),
                ordinary_val=ordinary_records(store, 'val', 512),
                ordinary_test=ordinary_records(store, 'test', 1024))


class BlockSampler:
    """Eight independent forward examples per graph pair; batch64=8 groups."""
    def __init__(self, n_groups, seed):
        self.groups = IndexSampler(n_groups, 8, seed)
        self.rng = np.random.default_rng(seed + 1)

    def next(self):
        return [g * 8 + int(v) for g in self.groups.next() for v in self.rng.permutation(8)]

    def state_dict(self):
        return dict(groups=self.groups.state_dict(), rng=self.rng.bit_generator.state)

    def load_state_dict(self, state):
        self.groups.load_state_dict(state['groups'])
        self.rng.bit_generator.state = deepcopy(state['rng'])


def planned_sizes(step, total, rng):
    """Five equal stages; after stage1, 20% previous-size replay over time.

    Quotas operate on complete eight-query groups. 1/2 replay groups alternate
    to realize 20%, instead of breaking within-group label balance.
    """
    stage = min(4, step // (total // 5))
    if not stage:
        return [SIZES[0]] * 8
    replay = ((step + 1) * 8) // 5 - (step * 8) // 5
    sizes = [SIZES[stage]] * (8 - replay)
    sizes += rng.choice(SIZES[:stage], replay, replace=True).tolist()
    return rng.permutation(sizes).tolist()


class PairedTrainer(Trainer):
    def __init__(self, cfg, data, mode, resume=None, *, ordinary_training=None, interleaved=False):
        self.study_data, self.mode = data, mode
        self.ordinary_training, self.interleaved = ordinary_training, interleaved
        self.size_quota, self.edge_counts = [], Counter()
        super().__init__(cfg, resume=resume)

    def _load_data(self):
        ordinary = getattr(self, 'ordinary_training', None)
        n = len(ordinary) if ordinary is not None else len(self.study_data['train']) * 8
        labels = np.array([r['label'] for r in ordinary]) if ordinary is not None else np.tile(LABELS, n // 8)
        fingerprint = sha(self.study_data) if ordinary is None else sha(dict(paired=self.study_data, ordinary=ordinary))
        # Adapter only: all rendering and validation below use the paired data.
        self.store = SimpleNamespace(fingerprint=fingerprint, arrays=dict(
            distance=np.ones(n, dtype=int), label=labels))
        self.train_ids = np.arange(n)
        self.val_ids = np.arange(len(self.study_data['ordinary_val']))

    def _preload_training_rows(self):
        self.prepared_tasks = self.prepared_walks = None
        if getattr(self, 'ordinary_training', None) is not None:
            self.paired_rows = {24: [row(r, i) for i, r in enumerate(self.ordinary_training)]}
            return
        self.paired_rows = {m: [row(record(g, m, v), i * 8 + v)
                               for i, g in enumerate(self.study_data['train']) for v in range(8)] for m in SIZES}

    def _init_samplers(self):
        super()._init_samplers()
        self.query_sampler = BlockSampler(len(self.study_data['train']), self.cfg.train.exposure_seed + 3)
        if getattr(self, 'ordinary_training', None) is not None or getattr(self, 'interleaved', False):
            self.query_sampler = IndexSampler(len(self.train_ids), 64, self.cfg.train.exposure_seed + 3)
        self.augmentation_rngs['difficulty'] = np.random.default_rng(SEED + 11)

    def _training_task(self, index):
        if not self.size_quota:
            planned = planned_sizes(self.completed, self.cfg.train.max_steps, self.augmentation_rngs['difficulty'])
            actual = planned if self.mode == 'curriculum' else [24] * 8
            self.size_quota = [m for m in actual for _ in range(8)][::-1]
        m = self.size_quota.pop()
        original = self.paired_rows[m][index]
        # Consume one RNG seed per exposure regardless of graph size; this
        # keeps matched M24 edge permutations identical across the two arms.
        seed = int(self.augmentation_rngs['edges'].integers(2 ** 63))
        rng = np.random.default_rng(seed)
        triples = np.array(original.tokens[:original.prompt_len - 4]).reshape(-1, 3)
        prefix = triples[rng.permutation(m)].ravel().tolist() + original.tokens[original.prompt_len - 4:original.prompt_len]
        self.edge_counts[str(m)] += 1
        return replace(original, tokens=prefix + original.tokens[original.prompt_len:])

    def _payload(self):
        result = super()._payload()
        result['paired_state'] = dict(mode=self.mode, quota=self.size_quota, counts=dict(self.edge_counts))
        return result

    def _load_resume(self, path):
        super()._load_resume(path)
        saved = torch.load(path, map_location='cpu', weights_only=False)['paired_state']
        assert saved['mode'] == self.mode
        self.size_quota, self.edge_counts = saved['quota'], Counter(saved['counts'])

    def _update(self, collect_metrics=True):
        metrics = super()._update(collect_metrics)
        metrics['curriculum_stage'] = min(5, self.completed // (self.cfg.train.max_steps // 5) + 1) if self.mode == 'curriculum' else 5
        return metrics

    def _evaluate(self, kind):
        if kind == 'teacher_forced':
            evaluate(self, final=False)


@torch.inference_mode()
def score(model, rows):
    predictions, probabilities, losses = [], [], []
    for start in range(0, len(rows), 256):
        chunk = rows[start:start + 256]
        # All conditions have a fixed length; only prompts enter the model.
        tokens = torch.tensor([r.tokens[:r.prompt_len] for r in chunk], device='cuda')
        logits = model(tokens)[:, -1].float()
        labels = torch.tensor([TOK.label_id(r.label) for r in chunk], device='cuda')
        losses.extend((-logits.log_softmax(-1).gather(1, labels[:, None]).squeeze(1)).cpu().tolist())
        probabilities.extend(logits[:, [TOK.no, TOK.yes]].softmax(-1)[:, 1].cpu().tolist())
        picked = logits.argmax(-1).cpu().tolist()
        predictions.extend(1 if p == TOK.yes else 0 if p == TOK.no else -1 for p in picked)
    return predictions, probabilities, losses


def evaluate(trainer, final):
    trainer.raw.eval()
    report, flat = {}, {}
    conditions = {'ordinary_val': [row(r) for r in trainer.study_data['ordinary_val']]}
    groups = trainer.study_data['test' if final else 'val']
    for m in SIZES:
        conditions[f'paired_{"test" if final else "val"}_m{m}'] = [row(record(g, m, v)) for g in groups for v in range(8)]
    if final:
        conditions['ordinary_test'] = [row(r) for r in trainer.study_data['ordinary_test']]
    # Train probes help distinguish failure to fit from failure to generalize.
    if getattr(trainer, 'ordinary_training', None) is not None:
        selected = [r for y in (0, 1) for r in [q for q in trainer.ordinary_training if q['label'] == y][:128]]
        conditions['ordinary_train_probe'] = [row(r) for r in selected]
    else:
        for m in (2, 24):
            conditions[f'train_probe_m{m}'] = [row(record(g, m, v)) for g in trainer.study_data['train'][:32] for v in range(8)]
    for name, rows in conditions.items():
        pred, p, ce = score(trainer.raw, rows)
        truth = np.array([r.label for r in rows])
        pred, p = np.array(pred), np.array(p)
        metrics = dict(n=len(rows), accuracy=float(np.mean(pred == truth)),
                       answer_ce=float(np.mean(ce)), p_yes_std=float(p.std()), yes_rate=float(np.mean(pred == 1)))
        if name.startswith(('paired_', 'train_probe_')):
            qpred, qtruth, qp = pred.reshape(-1, 8), truth.reshape(-1, 8), p.reshape(-1, 8)
            metrics['graph_flip_rate'] = float(np.mean(qpred[:, :4] != qpred[:, 4:]))
            metrics['graph_flip_both_correct'] = float(np.mean((qpred[:, :4] == qtruth[:, :4]) & (qpred[:, 4:] == qtruth[:, 4:])))
            signed = (qp[:, :4] - qp[:, 4:]) * (2 * qtruth[:, :4] - 1)
            metrics['graph_flip_signed_delta_p'] = float(signed.mean())
            query_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
            metrics['target_flip_both_correct'] = float(np.mean([
                (qpred[:, a] == qtruth[:, a]) & (qpred[:, b] == qtruth[:, b]) for a, b in query_pairs]))
            source_pairs = [(0, 2), (1, 3), (4, 6), (5, 7)]
            metrics['source_flip_both_correct'] = float(np.mean([
                (qpred[:, a] == qtruth[:, a]) & (qpred[:, b] == qtruth[:, b]) for a, b in source_pairs]))
        report[name] = metrics
        flat.update({f'direct_check/{name}/{k}': v for k, v in metrics.items() if k != 'n'})
        if final:
            with (trainer.out / f'predictions_{name}.jsonl').open('w') as f:
                for i, (pr, prob, target) in enumerate(zip(pred, p, truth)):
                    f.write(json.dumps(dict(index=i, label=int(target), pred=int(pr), p_yes=float(prob))) + '\n')
    report['step'] = trainer.completed
    folder = trainer.out / 'eval'
    folder.mkdir(exist_ok=True)
    write_json(folder / ('final.json' if final else f'step_{trainer.completed:08d}.json'), report)
    # Preserve standard chart naming while uploading the richer diagnostics.
    trainer.tracker.log(trainer.completed, {'free/overall/answer_accuracy': report['ordinary_val']['accuracy']})
    if trainer.tracker.run:
        trainer.tracker.run.log(dict(optimizer_step=trainer.completed, **flat))
    print('EVAL', trainer.mode, trainer.completed, json.dumps(report), flush=True)
    trainer.raw.train()


def check_data(data):
    graph_sets = {}
    checks = 0
    for split in ('train', 'val', 'test'):
        identities = set()
        for group in data[split]:
            assert len(set(group['nodes'])) == 16
            for m in SIZES:
                records = [record(group, m, v) for v in range(8)]
                assert Counter(r['label'] for r in records) == {0: 4, 1: 4}
                for rec in records:
                    a = [[] for _ in rec['nodes']]
                    for u, v in rec['edges']:
                        assert v // 4 == u // 4 + 1
                        a[u].append(v)
                    assert len(set(map(tuple, rec['edges']))) == len(rec['edges']) == m
                    assert bfs_reaches(a, rec['s'], rec['t']) == bool(rec['label'])
                    assert int(reaches(build_reach(a), rec['s'], rec['t'])) == rec['label']
                    r = row(rec)
                    assert r.prompt_len == 3 * m + 4
                    masks = collate_examples([r], TOK.pad)
                    assert masks['answer_mask'].sum() == 1 and masks['trace_mask'].sum() == 0
                    assert masks['answer_mask'][0, r.answer_index] == 1
                    checks += 1
                p0, p1 = row(records[0]).tokens, row(records[4]).tokens
                # Query and prompt-token histogram identical; labels opposite.
                assert Counter(p0[:-2]) == Counter(p1[:-2])
                assert sum(a != b for a, b in zip(p0[:-2], p1[:-2])) == 2
                for v in (0, 4):
                    rec = records[v]
                    identities.add(tuple(sorted((rec['nodes'][u], rec['nodes'][w]) for u, w in rec['edges'])))
        graph_sets[split] = identities
    assert not (graph_sets['train'] & graph_sets['val'] or graph_sets['train'] & graph_sets['test'] or graph_sets['val'] & graph_sets['test'])
    for split in ('ordinary_val', 'ordinary_test'):
        for rec in data[split]:
            a = [[] for _ in rec['nodes']]
            for u, v in rec['edges']:
                a[u].append(v)
            assert bfs_reaches(a, rec['s'], rec['t']) == bool(rec['label'])
            identity = tuple(sorted((rec['nodes'][u], rec['nodes'][v]) for u, v in rec['edges']))
            assert identity not in graph_sets['train']
    # Grouping, full query coverage, and exact sampler continuation.
    sampler = BlockSampler(len(data['train']), 3)
    for _ in range(10):
        ids = sampler.next()
        assert len(ids) == 64
        for start in range(0, 64, 8):
            block = ids[start:start + 8]
            assert len({i // 8 for i in block}) == 1 and {i % 8 for i in block} == set(range(8))
    saved = deepcopy(sampler.state_dict())
    clone = BlockSampler(len(data['train']), 3)
    clone.load_state_dict(saved)
    assert sampler.next() == clone.next()
    rng = np.random.default_rng(4)
    for stage in range(5):
        quota = Counter(m for step in range(stage * 100, (stage + 1) * 100) for m in planned_sizes(step, 500, rng))
        assert quota[SIZES[stage]] == (800 if stage == 0 else 640)
        assert all(m <= SIZES[stage] for m in quota)
    # Exercise real Trainer sampling/render hooks without a model, optimizer,
    # GPU forward, W&B or persisted outputs.
    cfg = load_config([BASE / 'resolved_config.yaml'], [
        'train.max_steps=500', 'train.warmup_steps=50', 'train.free_gen_samples=0',
        'evaluation.final_checkpoints=[]', 'tracking.mode=disabled'])
    tiny = {**data, 'train': data['train'][:8]}
    shims = []
    for mode in ('paired24', 'curriculum'):
        shim = PairedTrainer.__new__(PairedTrainer)
        shim.cfg, shim.study_data, shim.mode = cfg, tiny, mode
        shim.world, shim.tok = None, TOK
        shim.size_quota, shim.edge_counts = [], Counter()
        shim._load_data()
        shim._preload_training_rows()
        shim._init_samplers()
        shims.append(shim)
    for stage in range(5):
        for shim in shims:
            shim.completed = stage * 100
        indices = [shim.query_sampler.next() for shim in shims]
        assert indices[0] == indices[1]
        for index in indices[0]:
            direct, curriculum = [shim._training_task(index) for shim in shims]
            assert direct.label == curriculum.label
            assert direct.prompt_len == 76
            assert curriculum.prompt_len <= 3 * SIZES[stage] + 4
            assert direct.trace_indices == curriculum.trace_indices == []
            if curriculum.prompt_len == 76:
                assert direct.tokens == curriculum.tokens
    original = shims[1]
    indices = original.query_sampler.next()
    for index in indices[:5]:
        original._training_task(index)
    restored = copy(original)
    restored.size_quota = list(original.size_quota)
    restored.edge_counts = original.edge_counts.copy()
    restored.augmentation_rngs = deepcopy(original.augmentation_rngs)
    for index in indices[5:]:
        assert original._training_task(index).tokens == restored._training_task(index).tokens
    print(f'CHECK PASSED: {checks} structured labels/masks, ordinary holdouts, split separation, sampler resume, stage/replay quotas.', flush=True)
    print('CHECK PASSED: online grouped batches, edge shuffling, matched M24 exposure, partial-batch RNG/quota continuation; no training.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('paired24', 'curriculum'))
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--groups', type=int, default=512)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--run-name', default='direct_minimal_v2')
    parser.add_argument('--tracking', choices=('online', 'offline', 'disabled'), default='online')
    args = parser.parse_args()
    if args.steps < 5 or args.steps % 5 or args.groups < 8 or args.lr <= 0:
        parser.error('--steps must be >=5 and divisible by5; --groups>=8; --lr>0')
    if not args.run_name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_name):
        parser.error('invalid run-name')
    torch.set_num_threads(2)
    data = make_data(args.groups)
    if args.check_only:
        check_data(data)
        return
    if not args.mode:
        parser.error('--mode required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise RuntimeError('this study is restricted to physical GPU0; use launch.sh')
    out = HERE / 'runs' / args.run_name / args.mode
    spec = dict(version=1, mode=args.mode, steps=args.steps, groups=args.groups,
                lr=args.lr, data_sha256=sha(data), model_seed=0, batch_size=64,
                pure_direct=True, curriculum_sizes=list(SIZES), replay_fraction=0.2,
                stage_steps=[args.steps // 5] * 5, tracking=args.tracking)
    old = out / 'experiment.json'
    if old.exists() and json.loads(old.read_text()) != spec:
        raise ValueError('settings differ from existing run; use a new --run-name')
    if (out / 'completed.json').exists():
        print('SKIP completed', out, flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    write_json(old, spec)
    data_path = out.parent / 'data.json'
    if not data_path.exists():
        write_json(data_path, data)
    cfg = load_config([BASE / 'resolved_config.yaml'], [
        f'output_dir={out}', f'train.max_steps={args.steps}', f'train.lr={args.lr}',
        f'train.min_lr={args.lr / 10}', f'train.warmup_steps={min(500, args.steps // 10)}',
        'train.model_seed=0', 'train.effective_query_batch_size=64',
        'train.eval_interval=5000', 'train.free_gen_samples=0',
        'train.save_interval=10000', 'train.keep_last=5', 'train.log_interval=100',
        'evaluation.final_checkpoints=[]', 'model.compile=true', 'runtime.cuda_device=0',
        f'tracking.mode={args.tracking}', f'tracking.group={args.run_name}',
        f'tracking.name={args.run_name}/{args.mode}', f'tracking.variant={args.mode}',
        'tracking.online_failure=error', 'tracking.upload_artifacts=false'])
    cfg.tracking.notes = json.dumps(spec)
    latest = out / 'checkpoints/latest.pt'
    trainer = PairedTrainer(cfg, data, args.mode, resume=str(latest) if latest.exists() else None)
    if not latest.exists():
        model_hash = hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in trainer.raw.parameters())).hexdigest()
        write_json(out / 'initialization.json', dict(model_sha256=model_hash))
    if trainer.tracker.run:
        trainer.tracker.run.define_metric('direct_check/*', step_metric='optimizer_step', hidden=False)
    finish = trainer.tracker.finish
    trainer.tracker.finish = lambda: None
    try:
        trainer.fit()
        evaluate(trainer, final=True)
        write_json(out / 'exposure.json', dict(edge_counts=dict(trainer.edge_counts),
                   queries_seen=trainer.queries_seen, group_sampler=sha(trainer.query_sampler.state_dict()['rng'])))
        write_json(out / 'completed.json', dict(step=trainer.completed))
    finally:
        finish()


if __name__ == '__main__':
    main()

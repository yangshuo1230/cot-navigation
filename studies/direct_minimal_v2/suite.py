"""Six one-factor Direct comparisons, sharing the existing two-run machinery.

Inference/check-only is CPU-safe. Training is manual and restricted to GPU0.
The long arm matches paired24's LR schedule through the first base_steps;
its extra half continues at that schedule's floor, with no optimizer reset.
"""
import argparse
from collections import Counter
import hashlib
import json
import os

import numpy as np
import torch

import train as core

JOBS = ('ordinary24', 'paired24', 'paired24_interleaved', 'curriculum',
        'paired24_low_lr', 'paired24_long')


def settings(job, steps):
    return dict(job=job, steps=steps * (2 if job == 'paired24_long' else 1),
                schedule_horizon=steps, lr=1e-4 if job == 'paired24_low_lr' else 3e-4,
                interleaved=job == 'paired24_interleaved', ordinary=job == 'ordinary24')


def config(args, out, preset):
    return core.load_config([core.BASE / 'resolved_config.yaml'], [
        f'output_dir={out}', f'train.max_steps={preset["steps"]}', f'train.lr={preset["lr"]}',
        f'train.min_lr={preset["lr"] / 10}', f'train.warmup_steps={min(500, args.steps // 10)}',
        'train.model_seed=0', 'train.effective_query_batch_size=64',
        'train.eval_interval=5000', 'train.free_gen_samples=0',
        'train.save_interval=10000', f'train.keep_last={max(5, preset["steps"] // 10000)}',
        'train.log_interval=100', 'evaluation.final_checkpoints=[]',
        'model.compile=true', 'runtime.cuda_device=0',
        f'tracking.mode={args.tracking}', f'tracking.group={args.run_name}',
        f'tracking.name={args.run_name}/{args.job}', f'tracking.variant={args.job}',
        'tracking.online_failure=error', 'tracking.upload_artifacts=false'])


class SuiteTrainer(core.PairedTrainer):
    def __init__(self, cfg, data, mode, base_steps, **kwargs):
        self.base_steps = base_steps
        super().__init__(cfg, data, mode, **kwargs)
        self._configure_schedule(kwargs.get('resume'))

    def _configure_schedule(self, resume):
        # On resume the saved schedule is authoritative; verify its horizon.
        if not resume:
            self.schedule.max_steps = self.base_steps
            self.schedule.set_lr()
        assert self.schedule.max_steps == self.base_steps

    def _save(self):
        super()._save()
        if self.persist and self.completed == self.base_steps:
            model_hash = hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes()
                                                for p in self.raw.parameters())).hexdigest()
            core.write_json(self.out / 'prefix_summary.json', dict(
                step=self.completed, model_sha256=model_hash, queries_seen=self.queries_seen,
                sampler_rng_sha256=core.sha(self.query_sampler.state_dict()['rng']),
                edge_counts=dict(self.edge_counts)))


def check_extra(data, ordinary, args):
    # Independent ordinary examples are from original train only. The number
    # of distinct supervised queries matches the paired dataset exactly.
    assert len(ordinary) == 8 * len(data['train'])
    assert Counter(r['label'] for r in ordinary) == {0: len(ordinary) // 2, 1: len(ordinary) // 2}
    def identity(rec):
        return tuple(sorted((rec['nodes'][u], rec['nodes'][v]) for u, v in rec['edges']))
    forbidden = {identity(r) for part in ('ordinary_val', 'ordinary_test') for r in data[part]}
    forbidden |= {identity(core.record(g, 24, v)) for part in ('val', 'test') for g in data[part] for v in (0, 4)}
    for rec in ordinary:
        adj = [[] for _ in rec['nodes']]
        for u, v in rec['edges']:
            adj[u].append(v)
        assert core.bfs_reaches(adj, rec['s'], rec['t']) == bool(rec['label'])
        assert identity(rec) not in forbidden
    for job in JOBS:
        preset = settings(job, args.steps)
        check_args = core.copy(args)
        check_args.job = job
        cfg = config(check_args, core.HERE / 'unused_check_directory', preset)
        cfg.validate()
        assert cfg.model.layers == 6 and cfg.model.heads == 1 and cfg.model.d_model == 128
        assert cfg.protocol.name == 'direct' and cfg.train.stage == 'context'
        shim = core.PairedTrainer.__new__(core.PairedTrainer)
        shim.cfg, shim.study_data, shim.mode = cfg, data, job
        shim.ordinary_training = ordinary if preset['ordinary'] else None
        shim.interleaved = preset['interleaved']
        shim.world, shim.tok, shim.completed = None, core.TOK, 0
        shim.size_quota, shim.edge_counts = [], Counter()
        shim._load_data()
        shim._preload_training_rows()
        shim._init_samplers()
        chosen = shim.query_sampler.next()
        assert len(chosen) == 64
        rows = [shim._training_task(i) for i in chosen]
        assert all(r.trace_indices == [] for r in rows)
        assert all(r.prompt_len == (10 if job == 'curriculum' else 76) for r in rows)
        state = core.deepcopy(shim.query_sampler.state_dict())
        expected = shim.query_sampler.next()
        shim.query_sampler.load_state_dict(state)
        assert expected == shim.query_sampler.next()
    # Schedule equality over the whole common prefix, then floor continuation.
    from cot_navigation.training.schedule import WarmupCosine
    from types import SimpleNamespace
    def schedule(horizon):
        return WarmupCosine(SimpleOptimizer(), min(500, args.steps // 10), horizon, 3e-4, 3e-5)
    a, b = schedule(args.steps), schedule(args.steps * 2)
    assert a.lr_at(args.steps) != b.lr_at(args.steps)
    shim = SimpleNamespace(schedule=b, base_steps=args.steps)
    SuiteTrainer._configure_schedule(shim, resume=None)
    for step in (0, 499, args.steps // 2, args.steps - 1, args.steps, args.steps * 2):
        assert a.lr_at(step) == b.lr_at(step)
    assert abs(b.lr_at(args.steps * 2) - 3e-5) < 1e-12
    print('CHECK PASSED: ordinary labels/split separation, six configs/render paths, grouped/interleaved samplers, long-run LR horizon; no training.', flush=True)


class SimpleOptimizer:
    param_groups = []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', choices=JOBS)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--steps', type=int, default=50000, help='base budget; long arm uses twice this')
    parser.add_argument('--groups', type=int, default=512)
    parser.add_argument('--run-name', default='direct_minimal_v3')
    parser.add_argument('--tracking', choices=('online', 'offline', 'disabled'), default='online')
    args = parser.parse_args()
    if args.steps < 5 or args.steps % 5 or args.groups < 8:
        parser.error('--steps>=5, divisible by5; --groups>=8')
    if not args.run_name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_name):
        parser.error('invalid run-name')
    if args.list:
        print(json.dumps([settings(j, args.steps) for j in JOBS], indent=2))
        return
    torch.set_num_threads(2)
    data = core.make_data(args.groups)
    ordinary = core.ordinary_records(core.ContextStore(core.DATA), 'train', args.groups * 8)
    if args.check_only:
        core.check_data(data)
        check_extra(data, ordinary, args)
        return
    if not args.job:
        parser.error('--job required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise RuntimeError('physical GPU0 only; use launch.sh')
    preset = settings(args.job, args.steps)
    out = core.HERE / 'runs' / args.run_name / args.job
    spec = dict(version=3, **preset, base_steps=args.steps, groups=args.groups,
                paired_data_sha256=core.sha(data), ordinary_train_sha256=core.sha(ordinary),
                model_seed=0, batch_size=64, pure_direct=True, tracking=args.tracking)
    path = out / 'experiment.json'
    if path.exists() and json.loads(path.read_text()) != spec:
        raise ValueError('existing settings differ; use a new --run-name')
    if (out / 'completed.json').exists():
        print('SKIP completed', args.job, flush=True)
        return
    core.write_json(path, spec)
    for name, content in (('data.json', data), ('ordinary_train.json', ordinary)):
        target = out.parent / name
        if target.exists():
            assert core.sha(json.loads(target.read_text())) == core.sha(content), 'shared data mismatch'
        else:
            core.write_json(target, content)
    cfg = config(args, out, preset)
    cfg.tracking.notes = json.dumps(spec)
    latest = out / 'checkpoints/latest.pt'
    trainer = SuiteTrainer(cfg, data, args.job, args.steps, resume=str(latest) if latest.exists() else None,
        ordinary_training=ordinary if preset['ordinary'] else None, interleaved=preset['interleaved'])
    if not latest.exists():
        model_hash = hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes()
                                            for p in trainer.raw.parameters())).hexdigest()
        core.write_json(out / 'initialization.json', dict(model_sha256=model_hash))
    if trainer.tracker.run:
        trainer.tracker.run.define_metric('direct_check/*', step_metric='optimizer_step', hidden=False)
    finish = trainer.tracker.finish
    trainer.tracker.finish = lambda: None
    try:
        trainer.fit()
        core.evaluate(trainer, final=True)
        core.write_json(out / 'exposure.json', dict(edge_counts=dict(trainer.edge_counts),
                        queries_seen=trainer.queries_seen,
                        sampler_rng_sha256=core.sha(trainer.query_sampler.state_dict()['rng'])))
        core.write_json(out / 'completed.json', dict(step=trainer.completed))
    finally:
        finish()


if __name__ == '__main__':
    main()

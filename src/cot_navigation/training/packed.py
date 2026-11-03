"""Joint hard-DAG Direct / PATH / full DFS state / DFS Nodes training on frozen v5 splits.

Random initialization, one training stage, 48 short+16 long QUERY+16 ADJ.
Length-bucketed microbatches share global denominators and one optimizer step.
--check never creates a training directory, W&B run or checkpoint.
"""
import argparse
from collections import Counter
from contextlib import nullcontext
import json
import os
from pathlib import Path
import random
import time

from cot_navigation.experiments.artifacts import atomic_json, digest


def validate_resume_job(previous, current, *, allowed_hashes=()):
    """Accept an explicitly recorded code transition, never objective changes."""
    previous = dict(previous)
    if previous.get('source_hash') != current.get('source_hash'):
        if previous.get('source_hash') not in allowed_hashes:
            raise ValueError('Resume data/config/code changed: unknown source hash')
        previous['source_hash'] = current['source_hash']
    if previous != current:
        raise ValueError('Resume data/config/code changed')


def main(argv=None, *, backend=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--backend', choices=['v6', 'v7'], default='v7')
    parser.add_argument('--resources', type=Path)
    parser.add_argument('--world-root', type=Path)
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    parser.add_argument('--world', default='n80_g0')
    parser.add_argument('--protocol', choices=['direct','path','dfs_state','dfs_nodes'], required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int, help='Checkpoint and stop after this absolute update; useful for controlled interruption')
    parser.add_argument('--migrate-checkpoint', action='store_true', help='Accept only the recorded pre-refactor code hash; all other fields must match')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--eager', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--tracking', choices=['online', 'disabled'], default='online')
    args = parser.parse_args(argv)
    args.repo_root = args.repo_root.resolve()
    if args.stop_after is not None and args.stop_after < 1:
        parser.error('--stop-after must be positive')
    from cot_navigation.experiments.packed_backend import load_backend
    backend = backend or load_backend(args.backend, args.repo_root, resources=args.resources, world_root=args.world_root)
    HERE = backend.here
    if args.protocol not in backend.protocols:
        parser.error(f'{args.protocol} is not supported by this backend')
    source_hash = backend.source_hash
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import torch
    torch.set_num_threads(2)
    from cot_navigation.evaluation.panels import panel_subset
    from cot_navigation.evaluation.adjacency import evaluate_adjacency
    from cot_navigation.data.adjacency import Samples
    from cot_navigation.models.transformer import TransformerLM
    from cot_navigation.protocols.tokenizer import Tokenizer
    from cot_navigation.data.packed_bank import PackedBank
    from cot_navigation.training.packed_loss import logical_loss
    from cot_navigation.evaluation.packed import evaluate, actual_exposure
    cfg = json.loads((HERE / 'experiment.json').read_text())
    manifest = json.loads((HERE / 'manifest.json').read_text())
    assert manifest['config_sha256'] == digest(cfg), 'Preparation config is stale'
    meta = next(m for m in manifest['worlds'] if m['name'] == args.world)
    bank_meta = meta['banks'][args.protocol]
    job = {**cfg, 'world': args.world, 'world_hash': meta['world_hash'], 'bank_hash': bank_meta['sha256'],
           'protocol': args.protocol, 'model_seed': args.seed, 'initialization': 'random',
           'training_mode': 'joint_short_long_adjacency', 'source_hash': source_hash(args.repo_root),
           'model': {**cfg['model'], 'max_seq_len': bank_meta['max_seq_len']}, 'compile': not args.eager}
    if not args.check and not args.execute:
        print(json.dumps({'job': job, 'training_started': False}, indent=2))
        return
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    torch.manual_seed(args.seed)
    raw = TransformerLM(len(Tokenizer(256)), **job['model'], dropout=0.).to(device)
    opt = torch.optim.AdamW([
        {'params': [p for p in raw.parameters() if p.ndim >= 2], 'weight_decay': cfg['weight_decay']},
        {'params': [p for p in raw.parameters() if p.ndim < 2], 'weight_decay': 0.}],
        lr=cfg['lr'], betas=(.9, .99), fused=device.type == 'cuda')
    compiled = device.type == 'cuda' and not args.eager
    model = torch.compile(raw, dynamic=True) if compiled else raw
    loss_fn = torch.compile(logical_loss, dynamic=True, fullgraph=True) if compiled else logical_loss
    bank = PackedBank(meta, args.protocol, device, bank_root=HERE, world_root=backend.base)
    rng, vrng, counts = random.Random(1901), random.Random(1902), Counter()
    use_trace = args.protocol != 'direct'
    def update(step):
        ids = bank.sample(rng, vrng, cfg)
        denominator = bank.denominators(ids, cfg)
        lr = cfg['lr'] * min(1., (step + 1) / cfg['warmup_steps'])
        for group in opt.param_groups:
            group['lr'] = lr
        opt.zero_grad(set_to_none=True)
        values = torch.zeros(8, device=device)
        microbatches = bank.microbatches(ids, cfg)
        padded = 0
        for micro in microbatches:
            tokens, lengths, answers, groups = bank.batch(micro)
            padded += tokens.numel()
            with torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext():
                value = loss_fn(model(tokens), tokens, lengths, answers, groups, denominator,
                                cfg['trace_weight'], cfg['knowledge_weight'], use_trace)
            value[0].backward()
            values += value.detach()
        norm = torch.nn.utils.clip_grad_norm_(raw.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        counts.update(bank.keys[i] for i in ids)
        return values, norm, lr, {'microbatches': len(microbatches), 'padded_tokens': padded,
                                  'real_tokens': int(sum(bank.lengths[i] for i in ids))}
    if args.check:
        timings = []
        for step in range(3):
            start = time.perf_counter()
            values, norm, lr, packing = update(step)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            timings.append(time.perf_counter() - start)
        # Forward/backward of the longest TRAIN reference tests context/memory;
        # it is never truncated, and the temporary weights are discarded.
        row = max(np_rows(bank), key=lambda i: bank.lengths[i])
        tokens, lengths, answers, groups = bank.batch([row])
        opt.zero_grad(set_to_none=True)
        denominator = torch.tensor([1, 1, 1, 1], dtype=torch.float32, device=device)
        with torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext():
            longest = loss_fn(model(tokens), tokens, lengths, answers, groups, denominator,
                              cfg['trace_weight'], cfg['knowledge_weight'], use_trace)
        longest[0].backward()
        assert bool(torch.isfinite(values).all()) and bool(torch.isfinite(longest).all())
        sample = panel_subset(bank.world, bank.world['panels']['short_val'], 4)
        rollout = evaluate(raw, bank, sample, 4, max_new_tokens=8)
        print(json.dumps({'status': 'PASS', 'protocol': args.protocol, 'files_written': False,
            'warm_logical_update_seconds': sum(timings[1:]) / 2, 'packing': packing,
            'longest_train_reference_tokens': int(bank.lengths[row]), 'max_seq_len': bank_meta['max_seq_len'],
            'generation_queries': rollout['metrics']['all']['n'],
            'peak_gpu_memory_GB': torch.cuda.max_memory_allocated() / 1e9 if device.type == 'cuda' else None}), flush=True)
        return
    if args.output is None:
        raise ValueError('--output is required for a formal run')
    output = args.output.resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (output / '.run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    step, run_id = 0, None
    if args.resume:
        blob = torch.load(output / 'latest.pt', map_location='cpu', weights_only=False)
        allowed = json.loads((args.repo_root / 'configs/checkpoint_migration.json').read_text()).get(HERE.name, []) if args.migrate_checkpoint else []
        validate_resume_job(blob['job'], job, allowed_hashes=allowed)
        raw.load_state_dict(blob['model'])
        opt.load_state_dict(blob['optimizer'])
        rng.setstate(blob['sample_rng'])
        vrng.setstate(blob['variant_rng'])
        torch.set_rng_state(blob['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state(blob['cuda_rng'])
        counts.update(blob['counts'])
        step, run_id = blob['step'], blob.get('wandb_id')
    run = None
    if args.tracking == 'online':
        import wandb
        run = wandb.init(project=cfg['wandb_project'], group=cfg['wandb_group'],
                         name=f"{args.world}/s{args.seed}/{args.protocol}", config=job,
                         id=run_id, resume='must' if run_id else None, mode='online', dir=str(output))
        if run.offline:
            raise RuntimeError('Expected online W&B')
        run_id = run.id
        run.define_metric('optimizer_step')
        run.define_metric('*', step_metric='optimizer_step')
    atomic_json(output / 'resolved_job.json', job)
    started = time.perf_counter()
    def save(name):
        temporary = output / (name + '.writing')
        torch.save({'job': job, 'world_hash': meta['world_hash'],
            'model': raw.state_dict(), 'optimizer': opt.state_dict(), 'step': step,
            'sample_rng': rng.getstate(), 'variant_rng': vrng.getstate(),
            'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state() if device.type == 'cuda' else None,
            'counts': dict(counts), 'wandb_id': run_id}, temporary)
        temporary.replace(output / name)
    try:
        while step < cfg['task_steps']:
            values, norm, lr, packing = update(step)
            step += 1
            if step % cfg['log_interval'] == 0 or step == cfg['task_steps']:
                names = ['total', 'answer', 'trace', 'knowledge', 'short_answer', 'short_trace', 'long_answer', 'long_trace']
                row = {'optimizer_step': step, 'lr': lr, 'grad_norm': float(norm),
                       'elapsed': time.perf_counter() - started, **packing,
                       **{f'loss/{k}': v for k, v in zip(names, values.tolist())}}
                with (output / 'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(row) + '\n')
                print(json.dumps(row), flush=True)
                if run:
                    run.log({k: v for k, v in row.items() if k in
                             ('optimizer_step', 'loss/answer', 'loss/trace', 'loss/knowledge')})
            if step % cfg['eval_interval'] == 0 or step == cfg['task_steps']:
                result = {name: evaluate(raw, bank, panel_subset(bank.world, bank.world['panels'][name],
                          cfg['validation_per_panel']), cfg['evaluation_batch'][args.protocol])
                          for name in cfg['validation_panels']}
                with (output / 'validation.jsonl').open('a') as stream:
                    stream.write(json.dumps({'step': step, 'panels': result}) + '\n')
                if run:
                    charts = {'optimizer_step': step}
                    for name, value in result.items():
                        charts[f'{name}/BA'] = value['metrics']['all']['ba']
                        if use_trace:
                            charts[f'{name}/strict_trace_valid'] = value['metrics']['all']['trace_valid_rate']
                    run.log(charts)
            if step % cfg['adjacency_eval_interval'] == 0 or step == cfg['task_steps']:
                adj = evaluate_adjacency(raw, Samples(bank.world, 'direct'), device)
                adj.pop('predictions', None)
                atomic_json(output / 'adjacency.json', {'step': step, **adj})
            if step % cfg['checkpoint_interval'] == 0 or step == cfg['task_steps']:
                save('latest.pt')
            if step == 20000:
                save('step020000.pt')
            if args.stop_after is not None and step >= args.stop_after and step < cfg['task_steps']:
                save('latest.pt')
                return
        test = {name: evaluate(raw, bank, ids, cfg['evaluation_batch'][args.protocol])
                for name, ids in bank.world['panels'].items() if name.endswith('_test')}
        atomic_json(output / 'final.json', {'step': step, 'protocol': args.protocol, 'panels': test})
        atomic_json(output / 'actual_exposure.json', actual_exposure(bank, counts))
        atomic_json(output / 'exposure_counts.json', dict(counts))
        if run:
            for name, value in test.items():
                for metric in ('yes_recall', 'no_recall', 'ba', 'trace_valid_rate'):
                    run.summary[f'test/{name}/{metric}'] = value['metrics']['all'][metric]
        atomic_json(output / 'COMPLETE.json', {'step': step, 'protocol': args.protocol, 'wandb_id': run_id})
    finally:
        if run:
            run.finish()
        lock.close()


def np_rows(bank):
    return [row for q in bank.world['panels']['short_train'] + bank.world['panels']['long_train']
            for row in range(int(bank.qptr[q]), int(bank.qptr[q + 1]))]



if __name__ == '__main__':
    main()

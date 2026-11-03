"""Isolated v5 runtime: pre-encoded on-device banks, frozen variants, exact resume.

--check is memory-only: no run directory, checkpoint, log file or W&B run.
Formal runs save only latest, fixed step20k (task) and final compact reports.
"""
from __future__ import annotations
from cot_navigation.evaluation.panels import panel_subset
import argparse
from collections import Counter
from contextlib import nullcontext
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time

from data import HERE, atomic_json, digest, exposure_audit, validate


def runtime(repo):
    sys.path.insert(0, str(repo / 'src'))
    path = repo / 'studies/fixed_adjacency_suite/train_suite.py'
    spec = importlib.util.spec_from_file_location('v5_legacy', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_hash(repo):
    """Freeze trainer, data, tokenizer, model, attention and evaluation semantics."""
    paths = list(HERE.glob('*.py')) + [HERE / 'experiment.json']
    paths += list((repo / 'src/cot_navigation/models').glob('*.py'))
    paths += [repo / 'src/cot_navigation/protocols/tokenizer.py',
              repo / 'src/cot_navigation/evaluation/rollout.py',
              repo / 'src/cot_navigation/data/dataset.py',
              repo / 'src/cot_navigation/protocols/render.py',
              repo / 'studies/fixed_adjacency_suite/train_suite.py']
    return digest({str(p.relative_to(repo) if p.is_relative_to(repo) else p.name):
                   hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)})


class Bank:
    def __init__(self, world, protocol, device, max_new_tokens=16):
        import numpy as np
        import torch
        from cot_navigation.protocols.tokenizer import Tokenizer
        self.world, self.protocol, self.device = world, protocol, device
        self.max_new_tokens = max_new_tokens
        self.tok = Tokenizer(256)
        self.nodes = [self.tok.node_id(i) for i in world['node_names']]
        self.variants, self.keys, self.adj_ids = {}, [], []
        encoded = []
        for qid, r in enumerate(world['records']):
            prompt = self.prompt(qid)
            variants = r['paths'] if protocol == 'path' else [None]
            self.variants[qid] = []
            for variant, path in enumerate(variants):
                body = [self.nodes[u] for u in path] if path is not None else []
                tokens = prompt + body + [self.tok.label_id(r['y']), self.tok.id('<EOS>')]
                answer = len(prompt) + len(body)
                trace = list(range(len(prompt), answer)) + [answer + 1] if protocol == 'path' else []
                self.variants[qid].append(len(encoded))
                self.keys.append(f'q:{qid}:{variant}')
                encoded.append((tokens, [answer], trace, [], 0 if r['gap'] <= 3 else 1))
        for u, vs in enumerate(world['adj']):
            body = sorted(self.nodes[v] for v in vs) or [self.tok.id('<EXHAUST>')]
            tokens = [self.tok.id('<EDGE>'), self.nodes[u], self.tok.id('|')] + body + [self.tok.id('<EOS>')]
            self.adj_ids.append(len(encoded))
            self.keys.append(f'adj:{u}')
            encoded.append((tokens, [], [], list(range(3, len(tokens))), 2))
        width = ((max(len(row[0]) for row in encoded) + 15) // 16) * 16
        arrays = [np.full((len(encoded), width), self.tok.pad, dtype=np.int64)]
        arrays += [np.zeros((len(encoded), width), dtype=np.float32) for _ in range(3)]
        for i, row in enumerate(encoded):
            arrays[0][i, :len(row[0])] = row[0]
            for j in range(1, 4):
                arrays[j][i, row[j]] = 1
        self.tensors = [torch.from_numpy(array).to(device) for array in arrays]
        self.groups = torch.tensor([row[4] for row in encoded], device=device)
        self.lengths = [len(row[0]) for row in encoded]
        self.encoded = encoded
        self.cells = {}
        for part in ('short_train', 'long_train'):
            gaps = sorted({world['records'][i]['gap'] for i in world['panels'][part]})
            cells = {(gap, label): [i for i in world['panels'][part]
                                   if world['records'][i]['gap'] == gap and world['records'][i]['y'] == label]
                     for gap in gaps for label in (0, 1)}
            self.cells[part] = {k: v for k, v in cells.items() if v}

    def prompt(self, qid):
        r = self.world['records'][qid]
        return [self.tok.id('<QUERY>'), self.nodes[r['s']], self.nodes[r['t']], self.tok.id('|')]

    def sample(self, rng, cfg, variant_rng=None):
        variant_rng = variant_rng if variant_rng is not None else rng
        indices = []
        n_short = int(cfg['query_batch'] * cfg['short_fraction'])
        for part, count in [('short_train', n_short), ('long_train', cfg['query_batch'] - n_short)]:
            cells = self.cells[part]
            # Balance labels first; distribute across supported gap cells within label.
            for j in range(count):
                label = j % 2
                keys = [k for k in cells if k[1] == label]
                qid = rng.choice(cells[rng.choice(keys)])
                indices.append(variant_rng.choice(self.variants[qid]))
        indices.extend(rng.choices(self.adj_ids, k=cfg['adjacency_rows']))
        return indices

    def batch(self, ids):
        import torch
        index = torch.tensor(ids, dtype=torch.long, device=self.device)
        return [t.index_select(0, index) for t in self.tensors] + [self.groups.index_select(0, index)]


def losses(logits, tokens, answer_mask, trace_mask, knowledge_mask, groups, trace_weight, knowledge_weight):
    import torch
    import torch.nn.functional as F
    ce = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                         tokens[:, 1:].reshape(-1), reduction='none').reshape(tokens.shape[0], -1)
    def average(mask):
        m = mask[:, 1:]
        return (ce * m).sum(-1) / m.sum(-1).clamp_min(1)
    answer, trace = average(answer_mask), average(trace_mask)
    task = groups < 2
    answer_mean = (answer * task).sum() / task.sum().clamp_min(1)
    trace_mean = (trace * task).sum() / task.sum().clamp_min(1)
    km = knowledge_mask[:, 1:]
    knowledge = (ce * km).sum() / km.sum().clamp_min(1)
    values = [answer_mean + trace_weight * trace_mean + knowledge_weight * knowledge,
              answer_mean, trace_mean, knowledge]
    for group in (0, 1):
        selected = groups == group
        values.extend([(v * selected).sum() / selected.sum().clamp_min(1) for v in (answer, trace)])
    return torch.stack(values)





def classify_trace(world, bank, qid, output):
    r, tok = world['records'][qid], bank.tok
    trace = output['trace_tokens']
    inverse = {token: u for u, token in enumerate(bank.nodes)}
    path = [inverse.get(token, -1) for token in trace]
    starts = bool(path) and path[0] == r['s']
    edges = bool(path) and all(u >= 0 for u in path)
    if edges:
        edges = all(v in world['adj'][u] for u, v in zip(path, path[1:]))
    goal = bool(starts and edges and path[-1] == r['t'])
    if output['generation_limit_reached']:
        error = 'truncated'
    elif not path:
        error = 'empty_trace'
    elif not starts:
        error = 'wrong_start'
    elif not edges:
        error = 'illegal_token_or_edge'
    elif r['t'] in path[:-1]:
        error = 'passed_target'
    elif r['y'] and not goal:
        useful = world.get('_reach', [])
        error = 'early_no_or_wrong_stop'
        # First irreversible branch is distinguished from stopping on a useful prefix.
        if useful and any(u != r['t'] and r['t'] not in useful[u] for u in path):
            error = 'wrong_branch'
    else:
        error = 'valid_goal' if r['y'] else 'legal_negative_trace'
    return {'legal_trace': bool(starts and edges), 'valid_goal': goal, 'path': path,
            'trace_hops': max(0, len(path) - 1), 'error': error}


def evaluate(raw, bank, ids, actual=None, keep_examples=2, seen_paths=None):
    from cot_navigation.evaluation.rollout import generate
    from data import bfs
    world, tok = bank.world, bank.tok
    world['_reach'] = [set(bfs(world['adj'], s)) for s in range(world['n_nodes'])]
    buckets, errors, examples = {}, Counter(), []
    potential = world['potential_path_exposure']['by_query']
    exposure = {int(i): value for rows in (actual['by_query'] if actual else potential).values()
                for i, value in rows.items()}
    for start in range(0, len(ids), 128):
        part = ids[start:start + 128]
        outputs = generate(raw, [bank.prompt(i) for i in part], bank.protocol, tok,
                           'bf16' if bank.device.type == 'cuda' else 'fp32', max_new_tokens=bank.max_new_tokens)
        for qid, output in zip(part, outputs):
            r = world['records'][qid]
            correct = output['answer_token'] == tok.label_id(r['y'])
            trace = classify_trace(world, bank, qid, output) if bank.protocol == 'path' else None
            keys = ['all', f"gap{r['gap']}", 'gap4-5' if r['gap'] in (4, 5) else
                    'gap6-7' if r['gap'] in (6, 7) else 'gap8-9' if r['gap'] >= 8 else 'gap1-3']
            if r['y'] and bank.protocol == 'path' and qid in exposure:
                keys.append('witness_seen' if exposure[qid]['witness_seen'] else 'witness_unseen')
            for key in keys:
                c = buckets.setdefault(key, Counter())
                c['n'] += 1
                c[f"n{r['y']}"] += 1
                c[f"correct{r['y']}"] += correct
                c['correct'] += correct
                if trace:
                    c['legal'] += trace['legal_trace']
                    c['positive_valid_goal'] += r['y'] and trace['valid_goal']
                    c['negative_legal'] += not r['y'] and trace['legal_trace']
                    if seen_paths is not None:
                        c['generated_path_seen'] += tuple(trace['path']) in seen_paths
            if trace:
                errors[f"y{r['y']}/{trace['error']}"] += 1
            if (not correct or (trace and r['y'] and not trace['valid_goal'])) and len(examples) < keep_examples:
                examples.append({'query_id': qid, 'query': [r[k] for k in ('s', 't', 'y', 'gap')],
                                 'generated': output['generated'], 'trace': trace})
    world.pop('_reach', None)
    result = {}
    for key, c in buckets.items():
        yes = c['correct1'] / c['n1'] if c['n1'] else None
        no = c['correct0'] / c['n0'] if c['n0'] else None
        result[key] = {**dict(c), 'yes_recall': yes, 'no_recall': no,
                       'ba': (yes + no) / 2 if yes is not None and no is not None else None,
                       'accuracy': c['correct'] / c['n'],
                       'positive_valid_goal_rate': c['positive_valid_goal'] / c['n1']
                       if bank.protocol == 'path' and c['n1'] else None}
    raw.train()
    return {'metrics': result, 'errors': dict(errors), 'examples': examples,
            'exposure_source': 'actual_optimizer_stream' if actual else 'potential_allowed_variants'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root', type=Path, required=True)
    p.add_argument('--world', required=True)
    p.add_argument('--protocol', choices=['path'], default='path')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--output', type=Path)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--check', action='store_true')
    p.add_argument('--eager', action='store_true')
    p.add_argument('--tracking', choices=['online', 'disabled'])
    p.add_argument('--steps', type=int)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import torch
    torch.set_num_threads(2)
    legacy = runtime(args.repo_root)
    cfg = json.loads((HERE / 'experiment.json').read_text())
    manifest = json.loads((HERE / 'manifest.json').read_text())
    assert manifest['config_sha256'] == digest(cfg), 'Config changed; reprepare into an audited dataset'
    meta = next(m for m in manifest['worlds'] if Path(m['path']).name == args.world)
    world = json.loads((HERE / meta['path']).read_text())
    assert digest(world) == meta['sha256'], 'Dataset hash mismatch'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    steps = args.steps or cfg['task_steps']
    protocol = args.protocol
    job = {**cfg, 'model_seed': args.seed, 'protocol': args.protocol, 'world_hash': meta['sha256'],
           'source_hash': source_hash(args.repo_root), 'steps': steps, 'compile': not args.eager,
           'initialization': 'random', 'training_mode': 'joint_short_long_adjacency'}
    if not args.check and not args.execute:
        print(json.dumps({'job': job, 'training_started': False}, indent=2))
        return
    torch.manual_seed(args.seed)
    from cot_navigation.models.transformer import TransformerLM
    from cot_navigation.protocols.tokenizer import Tokenizer
    raw = TransformerLM(len(Tokenizer(256)), **cfg['model'], dropout=cfg['dropout']).to(device)
    for block in raw.blocks:
        block.attn.drop.p = 0.  # Residual dropout only, matching the previous suite.
    opt = torch.optim.AdamW([
        {'params': [v for v in raw.parameters() if v.ndim >= 2], 'weight_decay': cfg['weight_decay']},
        {'params': [v for v in raw.parameters() if v.ndim < 2], 'weight_decay': 0.}],
        lr=cfg['lr'], betas=(.9, .99), fused=device.type == 'cuda')
    model = torch.compile(raw, dynamic=True) if device.type == 'cuda' and not args.eager else raw
    loss_fn = torch.compile(losses, fullgraph=True) if device.type == 'cuda' and not args.eager else losses
    bank = Bank(world, protocol, device, cfg['max_new_tokens'])
    if args.check:
        # A few in-memory updates plus real generation; no persistent artifacts.
        rng = random.Random(123)
        timings = []
        for _ in range(8):
            start = time.perf_counter()
            ids = bank.sample(rng, cfg)
            tensors = bank.batch(ids)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext():
                value = loss_fn(model(tensors[0]), *tensors, cfg['trace_weight'], cfg['knowledge_weight'])
            value[0].backward()
            torch.nn.utils.clip_grad_norm_(raw.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            timings.append(time.perf_counter() - start)
        ids = panel_subset(world, world['panels']['short_val'], 8)
        result = evaluate(raw, bank, ids)
        assert bool(torch.isfinite(value).all()), 'Nonfinite loss in memory check'
        print(json.dumps({'status': 'PASS', 'check_only': True, 'files_written': False,
                          'finite_loss': bool(torch.isfinite(value).all()),
                          'warm_update_seconds': sum(timings[-4:]) / 4,
                          'generation_queries': result['metrics']['all']['n']}))
        return
    if args.output is None:
        raise ValueError('--output required for formal training')
    output = args.output.resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (output / '.run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rng, variant_rng = random.Random(1901), random.Random(1902)
    counts, step, run_id = Counter(), 0, None
    if args.resume:
        blob = torch.load(output / 'latest.pt', map_location='cpu', weights_only=False)
        if blob['job'] != job:
            raise ValueError('Resume config, data, or code changed')
        raw.load_state_dict(blob['model'])
        opt.load_state_dict(blob['optimizer'])
        rng.setstate(blob['sample_rng'])
        variant_rng.setstate(blob['variant_rng'])
        torch.set_rng_state(blob['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state(blob['cuda_rng'])
        counts.update(blob['counts'])
        step, run_id = blob['step'], blob.get('wandb_id')
    run = None
    tracking = args.tracking or cfg['tracking']
    if tracking == 'online':
        import wandb
        run = wandb.init(project=cfg['wandb_project'], group=cfg['wandb_group'],
                         name=f"{Path(args.world).stem}/s{args.seed}/{args.protocol}",
                         config=job, id=run_id, resume='must' if run_id else None,
                         mode='online', dir=str(output))
        if run.offline:
            raise RuntimeError('Online W&B required')
        run_id = run.id
        run.define_metric('optimizer_step')
        run.define_metric('*', step_metric='optimizer_step')
    atomic_json(output / 'resolved_job.json', job)
    start_time = time.perf_counter()
    last_metrics = None
    def save(name):
        legacy.save_atomic(output / name, {'job': job, 'world_hash': meta['sha256'],
            'model': raw.state_dict(), 'optimizer': opt.state_dict(), 'step': step,
            'sample_rng': rng.getstate(), 'variant_rng': variant_rng.getstate(),
            'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state() if device.type == 'cuda' else None,
            'counts': dict(counts), 'wandb_id': run_id})
    try:
        while step < steps:
            ids = bank.sample(rng, cfg, variant_rng)
            counts.update(bank.keys[i] for i in ids)
            tensors = bank.batch(ids)
            lr = cfg['lr'] * min(1., (step + 1) / cfg['warmup_steps'])
            for group in opt.param_groups:
                group['lr'] = lr
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext():
                values = loss_fn(model(tensors[0]), *tensors, cfg['trace_weight'], cfg['knowledge_weight'])
            values[0].backward()
            norm = torch.nn.utils.clip_grad_norm_(raw.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            step += 1
            if step % cfg['log_interval'] == 0 or step == steps:
                names = ['total', 'answer', 'trace', 'knowledge', 'short_answer', 'short_trace', 'long_answer', 'long_trace']
                row = {'optimizer_step': step, 'lr': lr, 'elapsed': time.perf_counter() - start_time,
                       'grad_norm': float(norm), **{f'loss/{k}': v for k, v in zip(names, values.detach().tolist())}}
                with (output / 'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(row) + '\n')
                print(json.dumps(row), flush=True)
                if run:
                    # Keep the visible dashboard small; component detail stays in JSONL.
                    run.log({k: v for k, v in row.items() if k in
                             ('optimizer_step', 'loss/answer', 'loss/trace', 'loss/knowledge')})
            if step % cfg['eval_interval'] == 0 or step == steps:
                panels = {}
                for name in ('short_train', 'long_train', 'short_val', 'long_AA_val',
                             'long_VA_val', 'long_AV_val', 'long_VV_val'):
                    cap = 64 if name.endswith('train') else cfg['validation_per_panel']
                    panels[name] = evaluate(raw, bank, panel_subset(world, world['panels'][name], cap))
                last_metrics = {k: v['metrics']['all'] for k, v in panels.items()}
                with (output / 'validation.jsonl').open('a') as stream:
                    stream.write(json.dumps({'step': step, 'panels': panels}) + '\n')
                if run:
                    visible = {'optimizer_step': step}
                    for name in ('short_val', 'long_AA_val', 'long_VV_val'):
                        visible[f'{name}/BA'] = last_metrics[name]['ba']
                        if protocol == 'path':
                            visible[f'{name}/valid_goal'] = last_metrics[name]['positive_valid_goal_rate']
                    run.log(visible)
            if step % cfg['adjacency_eval_interval'] == 0 or step == steps:
                adjacency = legacy.evaluate_adjacency(raw, legacy.Samples(world, 'direct'), device)
                adjacency.pop('predictions', None)
                atomic_json(output / 'adjacency.json', {'step': step, **adjacency})
                if run:
                    run.log({'optimizer_step': step, 'knowledge/exact_set': adjacency['exact_set_accuracy']})
            if step % cfg['checkpoint_interval'] == 0 or step == steps:
                save('latest.pt')
            if step == 20000:
                save('step020000.pt')
        actual = exposure_audit(world, counts)
        result = {'step': step, 'validation': last_metrics, 'panels': {}}
        seen_paths = set()
        for key, count in counts.items():
            if key.startswith('q:') and count:
                _, qid, variant = key.split(':')
                seen_paths.add(tuple(world['records'][int(qid)]['paths'][int(variant)]))
        for name in ('short_test', 'long_AA_test', 'long_TA_test', 'long_AT_test', 'long_TT_test'):
            result['panels'][name] = evaluate(raw, bank, world['panels'][name], actual, seen_paths=seen_paths)
        atomic_json(output / 'actual_exposure.json', actual)
        atomic_json(output / 'final.json', result)
        atomic_json(output / 'exposure_counts.json', dict(counts))
        if run:
            for name, value in result['panels'].items():
                for metric in ('ba', 'yes_recall', 'no_recall', 'positive_valid_goal_rate'):
                    run.summary[f'test/{name}/{metric}'] = value['metrics']['all'][metric]
        atomic_json(output / 'COMPLETE.json', {'step': step, 'protocol': args.protocol,
                                             'world_hash': meta['sha256'], 'wandb_id': run_id})
    finally:
        if run:
            run.finish()
        lock.close()


if __name__ == '__main__':
    main()

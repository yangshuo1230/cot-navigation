"""Frozen evaluation of same-layer and reversed queries. No training or W&B."""
from pathlib import Path
from collections import Counter
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time

os.environ['WANDB_MODE'] = 'disabled'
import numpy as np
import torch
from cot_navigation.config import load_config
from cot_navigation.data.dataset import ContextStore, ParametricWorld
from cot_navigation.data.reachability import bfs_reaches, build_reach, reaches
from cot_navigation.evaluation.rollout import generate
from cot_navigation.models.transformer import build_model
from cot_navigation.protocols.render import context_prompt, parametric_prompt
from cot_navigation.protocols.tokenizer import Tokenizer

ROOT = Path(str(next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'pyproject.toml').is_file() and (_p / 'studies').is_dir())))
HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'
MAIN = ROOT / 'outputs/explore_v1/launches/explore_v1_s012'
SEED = 20261018
TOK = Tokenizer()
sys.path.insert(0, str(ROOT / 'studies/length_generalization'))
from trace_audit import validate_trace


def save(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def degrees(adj):
    incoming = [0] * len(adj)
    for vs in adj:
        for v in vs:
            incoming[v] += 1
    return incoming, [len(vs) for vs in adj]


def record(adj, nodes, edges, s, t, label, width, index, kind):
    assert s != t
    assert bfs_reaches(adj, s, t) == bool(label)
    assert reaches(build_reach(adj), s, t) == bool(label)
    inc, out = degrees(adj)
    assert inc[s] and out[s] and inc[t] and out[t], 'Both endpoints must be internal and visible'
    delta = t // width - s // width
    assert (delta > 0 if kind.startswith('forward') else delta == 0 if kind == 'same_layer' else delta < 0)
    return dict(adj=adj, nodes=nodes, edges=edges, s=s, t=t, label=label,
                layer_delta=delta, index=index, kind=kind,
                source_in=inc[s], source_out=out[s], target_in=inc[t], target_out=out[t])


def make_context():
    source = ContextStore(ROOT / 'data/explore_v1/context_standard_n16_m24_train50000_val10000_test10000_s0')
    a = source.arrays
    ids = source.indices('test')
    ids = ids[(a['distance'][ids] == 1) & (a['label'][ids] == 1)]
    rng = np.random.default_rng(SEED)
    result = {k: [] for k in ('forward_yes', 'forward_no', 'reverse', 'same_layer')}
    for index in rng.permutation(ids):
        index = int(index)
        adj = source.adj(index)
        s, t = int(a['s_topo'][index]), int(a['t_topo'][index])
        inc, out = degrees(adj)
        if s // 4 != 1 or t // 4 != 2 or not all((inc[s], out[s], inc[t], out[t])):
            continue
        negatives = [v for v in range(8, 12) if v not in adj[s] and inc[v] and out[v]]
        same = [v for v in range(4, 8) if v != s and inc[v] and out[v]]
        if not negatives or not same:
            continue
        nodes = source.node_vocab_ids(index, TOK)
        edges = list(zip(map(int, a['edge_src'][index]), map(int, a['edge_dst'][index])))
        specs = [('forward_yes', s, t, 1), ('forward_no', s, int(rng.choice(negatives)), 0),
                 ('reverse', t, s, 0), ('same_layer', s, int(rng.choice(same)), 0)]
        for kind, start, target, label in specs:
            result[kind].append(record(adj, nodes, edges, start, target, label, 4, index, kind))
        if len(result['forward_yes']) == 64:
            break
    assert all(len(v) == 64 for v in result.values())
    return result


def make_fixed(seed):
    world = ParametricWorld(ROOT / f'data/explore_v1/fixed_standard_n32_m48_onehop_s0/balanced-seed{seed}')
    adj, nodes, a = world.adj(), world.node_vocab_ids(TOK), world.queries
    edges = [(u, v) for u, vs in enumerate(adj) for v in vs]
    inc, out = degrees(adj)
    rng = np.random.default_rng(SEED + seed)
    result = {k: [] for k in ('forward_yes', 'forward_no', 'reverse', 'same_layer')}
    for y, kind in ((1, 'forward_yes'), (0, 'forward_no')):
        candidates = [int(i) for i in world.indices('test', 'full') if int(a['label'][i]) == y
                      and all((inc[int(a['s_topo'][i])], out[int(a['s_topo'][i])],
                               inc[int(a['t_topo'][i])], out[int(a['t_topo'][i])] ))]
        chosen = rng.permutation(candidates)[:32]
        assert len(chosen), (seed, kind)
        for i in chosen:
            i = int(i)
            s, t = int(a['s_topo'][i]), int(a['t_topo'][i])
            result[kind].append(record(adj, nodes, edges, s, t, y, 4, i, kind))
            if y == 1:
                result['reverse'].append(record(adj, nodes, edges, t, s, 0, 4, i, 'reverse'))
    pairs = [(s, t) for s in range(32) for t in range(32) if s != t and s // 4 == t // 4
             and all((inc[s], out[s], inc[t], out[t]))]
    for position in rng.permutation(len(pairs))[:32]:
        s, t = pairs[int(position)]
        result['same_layer'].append(record(adj, nodes, edges, s, t, 0, 4, int(position), 'same_layer'))
    return result


def evaluate(name, folder, data, fixed):
    checkpoint = folder / 'checkpoints/latest.pt'
    before = sha(checkpoint)
    cfg = load_config([folder / 'resolved_config.yaml'])
    assert cfg.protocol.name in ('direct', 'dfs_state', 'bfs_state')
    blob = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert blob['arch']['vocab_hash'] == TOK.hash()
    model = build_model(cfg.model, len(TOK), TOK.pad)
    model.load_state_dict(blob['model'], strict=True)
    model.to('cuda').eval()
    del blob
    reports, raw = {}, {}
    for kind, records in data.items():
        predictions, errors, correct, valid, fmt, limits = [], Counter(), 0, 0, 0, 0
        for begin in range(0, len(records), 32):
            rows = records[begin:begin+32]
            prompts = [(parametric_prompt(TOK, r['nodes'], r['s'], r['t']) if fixed else
                        context_prompt(TOK, r['nodes'], r['edges'], r['s'], r['t'])) for r in rows]
            outputs = generate(model, prompts, cfg.protocol.name, TOK, 'fp32')
            for r, output in zip(rows, outputs):
                ok = output['pred'] == r['label'] and not output['format_fail']
                correct += int(ok)
                fmt += int(output['format_fail'])
                limits += int(output['generation_limit_reached'])
                verdict = None
                if cfg.protocol.name != 'direct':
                    token_adj = {r['nodes'][u]: [r['nodes'][v] for v in vs] for u, vs in enumerate(r['adj'])}
                    verdict = validate_trace(output['trace_tokens'], output['pred'], cfg.protocol.name,
                                             token_adj, r['nodes'][r['s']], r['nodes'][r['t']], TOK)
                    valid += int(verdict['valid'])
                    if not verdict['valid']:
                        errors[str(verdict['error'])] += 1
                predictions.append(dict(index=r['index'], s=r['s'], t=r['t'], label=r['label'],
                    pred=output['pred'], correct=ok, format_fail=output['format_fail'],
                    generation_limit=output['generation_limit_reached'], trace_verdict=verdict,
                    layer_delta=r['layer_delta'], tokens=output['tokens']))
        n = len(records)
        reports[kind] = dict(n=n, correct=correct, accuracy=correct/n, format_fail=fmt/n,
            yes_rate=sum(p['pred'] == 1 for p in predictions)/n,
            generation_limit=limits/n, trajectory_valid=valid/n if cfg.protocol.name != 'direct' else None,
            errors=dict(errors))
        raw[kind] = predictions
        print(name, kind, json.dumps(reports[kind]), flush=True)
    flip = sum(a['pred'] == 1 and b['pred'] == 0
               for a, b in zip(raw['forward_yes'], raw['reverse'])) / len(raw['forward_yes'])
    result = dict(name=name, protocol=cfg.protocol.name, step=cfg.train.max_steps,
                  checkpoint=str(checkpoint), checkpoint_sha256=before, precision='fp32',
                  reports=reports, forward_reverse_both_correct=flip)
    save(OUT / (name.replace('/', '__') + '.json'), result)
    save(OUT / (name.replace('/', '__') + '_predictions.json'), raw)
    assert sha(checkpoint) == before
    del model
    torch.cuda.empty_cache()
    return result


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    lock = open('/tmp/cot-navigation-gpu0.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    memory = int(subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=memory.used',
                                         '--format=csv,noheader,nounits']).decode().strip())
    assert memory < 1000, f'GPU0 busy: {memory} MiB'
    assert not OUT.exists(), 'Use a new evaluation directory; never overwrite results'
    OUT.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = {'context': make_context(), **{f'fixed{i}': make_fixed(i) for i in range(3)}}
    save(OUT / 'inputs.json', data)
    summary = {}
    context_models = [
        ('direct', MAIN / 'context/standard/train50000/baseline/direct'),
        ('direct_onehop', MAIN / 'context/standard/train50000/direct_onehop/direct'),
        ('dfs_state', MAIN / 'context/standard/train50000/baseline/dfs_state'),
        ('bfs_state', MAIN / 'context/standard/train50000/bfs_state/bfs_state'),
        ('ordinary24', ROOT / 'studies/direct_minimal_v2/runs/direct_minimal_v3/ordinary24'),
        ('paired24', ROOT / 'studies/direct_minimal_v2/runs/direct_minimal_v3/paired24'),
    ]
    started = time.time()
    for name, folder in context_models:
        key = 'context/' + name
        summary[key] = evaluate(key, folder, data['context'], False)
        save(OUT / 'summary.json', summary)
    for seed in range(3):
        for name, suffix in [('direct', 'baseline/direct'), ('dfs_state', 'baseline/dfs_state'),
                             ('dfs_random', 'dfs_random/dfs_state'), ('bfs_state', 'bfs_state/bfs_state')]:
            key = f'fixed{seed}/{name}'
            summary[key] = evaluate(key, MAIN / f'fixed/standard/seed{seed}' / suffix, data[f'fixed{seed}'], True)
            save(OUT / 'summary.json', summary)
    save(OUT / 'completed.json', dict(models=len(summary), elapsed_s=time.time()-started,
                                    script_sha256=sha(Path(__file__)), gpu=0,
                                    training_updates=0, wandb=False))


if __name__ == '__main__':
    with torch.inference_mode():
        main()

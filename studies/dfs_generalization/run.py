"""Frozen DFS generalization: shared graph generators, decoder and semantic audit.

Prepare CPU-only test data first; --launch --background queues GPU evaluations.
Never interrupts training. Only compact per-query audits are persisted.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

P = Path(__file__).resolve().parent
ROOT = P.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ['WANDB_MODE'] = 'disabled'
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'TORCHINDUCTOR_COMPILE_THREADS'):
    os.environ.setdefault(key, '2')
from cot_navigation.experiments.scheduler import file_lock, write_json
from cot_navigation.data.shards import file_sha256

NODES = {32: (4, 8), 64: (8, 8), 100: (10, 10), 144: (12, 12),
         169: (13, 13), 196: (14, 14)}
MODELS = ROOT / 'outputs/correction_reproduction_v1/launches/dfs_correction_repro_20261028/dfs_state'
OUT = ROOT / 'outputs/analysis/dfs_generalization_20261031'
DATA = ROOT / 'data/evaluation/dfs_generalization_20261031'
TRAIN_QUEUE = ROOT / 'outputs/correction_depth_rate/launches/dfs_depth_rate_20261031/status.json'
ARMS = ['matched_clean', 'correction']


def save_gzip(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with gzip.open(tmp, 'wt', encoding='utf-8') as handle:
        json.dump(obj, handle, separators=(',', ':'))
    tmp.replace(path)


def read_gzip(path):
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        return json.load(handle)


def prepare():
    import numpy as np
    from cot_navigation.config import load_config
    from cot_navigation.data.generate import _generate_context_row
    from cot_navigation.data.splits import iter_sample_plan
    from cot_navigation.data.reachability import bfs_reaches
    from cot_navigation.data.graph import adj_from_presentation
    from cot_navigation.data.hard_graph import graph_statistics, shortest_distances
    from cot_navigation.training.checkpoint import load_checkpoint

    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if (DATA / 'manifest.json').exists():
        manifest = json.loads((DATA / 'manifest.json').read_text())
        for cell in manifest['cells']:
            assert file_sha256(DATA / cell['file']) == cell['sha256']
        for item in manifest['models'].values():
            assert file_sha256(Path(item['checkpoint'])) == item['sha256']
        print('REUSE VERIFIED', len(manifest['cells']), 'cells', flush=True)
        return manifest

    cfg = load_config([MODELS / 'correction/s0/resolved_config.yaml'])
    # Read only graph hashes, rather than loading all training arrays/trajectories.
    forbidden = set()
    for root in cfg.train.scale_roots:
        for split in ('train', 'val', 'test'):
            for path in (ROOT / root / split).glob('shard-*.npz'):
                with np.load(path, allow_pickle=False) as arrays:
                    forbidden.update(map(int, arrays['struct_hash']))
    original_forbidden = len(forbidden)
    manifest = dict(schema=1, seed=20261031, samples_per_cell=512,
        positive_per_cell=256, primary_population='positive',
        query_policy='equal positive/negative quotas by rank-band distance 1..bands-1; same original sampler',
        graph_identity='pre-rename topological-edge hash; not graph isomorphism',
        forbidden_training_validation_test_graphs=original_forbidden,
        context_length=4096, precision='bf16', batch_size=32, cot_margin=cfg.protocol.cot_margin,
        cells=[], models={})
    for family_id, family in enumerate(('simple', 'complex')):
        for n, (layers, width) in NODES.items():
            m = 3*n//2 if family == 'simple' else 2*n
            payload = dict(split='test', n_nodes=n, n_edges=m,
                graph_family='layered' if family == 'simple' else 'multiscale_dag_v1',
                node_vocab=256, data_seed=20261031 + 1000*family_id + n,
                max_attempts=2000, shapes={'main': (layers, width)},
                shape_names=['main'], forbidden=forbidden)
            rows = []
            for plan in iter_sample_plan(['main'], {'main': layers}, 512):
                row = _generate_context_row(payload, plan)
                edges = list(zip(row['edge_src'], row['edge_dst']))
                adj = adj_from_presentation(n, edges)
                assert bfs_reaches(adj, row['s'], row['t']) == bool(row['label'])
                assert len(set(edges)) == m and all(u < v for u, v in edges)
                assert len(set(row['node_token'])) == n
                assert row['struct_hash'] not in forbidden
                forbidden.add(row['struct_hash'])
                row['shortest_hops'] = shortest_distances(adj, row['s'])[row['t']]
                row['graph_statistics'] = graph_statistics(adj, width)
                # A canonical cost diagnostic only: a different valid DFS can be shorter.
                row['reference_fits'] = 3*m + 4 + row['dfs_trace_tokens'] + 3 <= 4096
                rows.append(row)
            assert Counter(r['label'] for r in rows) == {0: 256, 1: 256}
            name = f'{family}_n{n}'
            path = DATA / f'{name}.json.gz'
            save_gzip(path, rows)
            manifest['cells'].append(dict(name=name, family=family, n=n, m=m,
                layers=layers, width=width, file=path.name, sha256=file_sha256(path),
                reference_fit_rate=sum(r['reference_fits'] for r in rows)/len(rows)))
            print('DATA READY', name, len(rows), flush=True)
    for arm in ARMS:
        for seed in range(4):
            folder = MODELS / arm / f's{seed}'
            path = folder / 'checkpoints/latest.pt'
            checkpoint = load_checkpoint(path, map_location='cpu')
            assert checkpoint['step'] == 200000
            del checkpoint
            mc = load_config([folder / 'resolved_config.yaml'])
            assert mc.model.layers == 6 and mc.model.max_seq_len == 4096
            manifest['models'][f'{arm}_s{seed}'] = dict(arm=arm, seed=seed,
                checkpoint=str(path), sha256=file_sha256(path), step=200000,
                config=str(folder / 'resolved_config.yaml'))
    write_json(DATA / 'manifest.json', manifest)
    return manifest


def worker(name):
    import numpy as np
    import torch
    from cot_navigation.config import load_config
    from cot_navigation.models.transformer import build_model
    from cot_navigation.models.compiled_inference import CompiledInference
    from cot_navigation.protocols.tokenizer import Tokenizer
    from cot_navigation.protocols.render import context_prompt
    from cot_navigation.protocols.correction_replay import replay, self_test
    from cot_navigation.protocols.trace_audit import validate_states
    from cot_navigation.training.checkpoint import load_checkpoint
    from cot_navigation.evaluation.rollout import generate

    torch.set_num_threads(2)
    manifest = json.loads((DATA / 'manifest.json').read_text())
    identity = manifest['models'][name]
    assert file_sha256(Path(identity['checkpoint'])) == identity['sha256']
    cfg = load_config([identity['config']])
    tok = Tokenizer.from_config(cfg)
    self_test(tok)
    raw = build_model(cfg.model, len(tok), pad_id=tok.pad).cuda().eval().requires_grad_(False)
    blob = load_checkpoint(Path(identity['checkpoint']), map_location='cpu')
    raw.load_state_dict(blob['model'], strict=True)
    del blob
    model = CompiledInference(raw).eval()
    for cell in manifest['cells']:
        dest = OUT / 'results' / f'{name}__{cell["name"]}.json.gz'
        if dest.exists():
            previous = read_gzip(dest)
            assert previous['model'] == identity and previous['data_sha256'] == cell['sha256']
            if previous['complete']:
                continue
        rows = read_gzip(DATA / cell['file'])
        # Deterministic mixed-label batches, shared across all receiving models.
        order = np.random.default_rng(1729).permutation(len(rows))
        audits = []
        excerpts = []
        started = time.monotonic()
        for offset in range(0, len(order), manifest['batch_size']):
            chunk = [rows[int(i)] for i in order[offset:offset+manifest['batch_size']]]
            node_ids = [[tok.node_id(v) for v in r['node_token']] for r in chunk]
            prompts = [context_prompt(tok, nodes, list(zip(r['edge_src'], r['edge_dst'])), r['s'], r['t'])
                       for r, nodes in zip(chunk, node_ids)]
            with torch.no_grad():
                outputs = generate(model, prompts, 'dfs_state', tok, 'bf16',
                    max_new_tokens=0, cot_margin=manifest['cot_margin'])
            for row, nodes, prompt, g in zip(chunk, node_ids, prompts, outputs):
                adj = {nodes[u]: [] for u in range(cell['n'])}
                for u, v in zip(row['edge_src'], row['edge_dst']):
                    adj[nodes[u]].append(nodes[v])
                detail = replay(g['trace_tokens'], g['pred'], adj, nodes[row['s']], nodes[row['t']], tok,
                    prompt_len=len(prompt), interrupted=g['cot_interrupted'], label=row['label'])
                if validate_states(g['trace_tokens'], g['pred'], adj, nodes[row['s']], nodes[row['t']], tok, False)['valid']:
                    assert detail['trace_valid']
                assert detail['correct'] == (not g['format_fail'] and g['pred'] == row['label'])
                audits.append(dict(query_id=row['sample_id'], label=row['label'], pred=g['pred'],
                    distance=row['distance'], shortest_hops=row['shortest_hops'],
                    reference_fits=row['reference_fits'], reference_tokens=row['dfs_trace_tokens'],
                    prompt_len=len(prompt), interrupted=g['cot_interrupted'], forced_stop=g['forced_stop'],
                    graph_hash=str(row['struct_hash']),
                    trace_hash=hashlib.sha256(np.asarray(g['trace_tokens'], dtype=np.int32).tobytes()).hexdigest(),
                    **detail))
                if detail['first'] and len(excerpts) < 3:
                    pos=detail['first']['token']-1
                    excerpts.append(dict(query_id=row['sample_id'], first=detail['first'],
                        excerpt=tok.decode(g['trace_tokens'][max(0,pos-12):pos+40])))
            print(name, cell['name'], len(audits), '/', len(rows), round(time.monotonic()-started,1), flush=True)
        save_gzip(dest, dict(complete=True, model=identity, data_sha256=cell['sha256'], cell=cell,
            batch_size=manifest['batch_size'], context_length=4096, precision='bf16',
            rows=audits, excerpts=excerpts, seconds=time.monotonic()-started,
            compile_fallbacks=sorted(model.compile_fallbacks)))
        subprocess.run([sys.executable, str(P/'report.py')], cwd=ROOT, check=True)
    assert file_sha256(Path(identity['checkpoint'])) == identity['sha256']


def coordinator(gpus):
    from cot_navigation.runtime.launch import idle_gpus
    manifest = json.loads((DATA/'manifest.json').read_text())
    pending = list(manifest['models'])
    running = {}
    done = []
    failed = {}
    previous_state = None
    OUT.mkdir(parents=True, exist_ok=True)
    with file_lock(OUT/'.queue.lock', blocking=False):
        while pending or running:
            for gpu, entry in list(running.items()):
                code = entry['proc'].poll()
                if code is not None:
                    entry['log'].close()
                    if code:
                        failed[entry['name']] = dict(exit_code=code, log=str(OUT/f'{entry["name"]}.log'))
                        print('FAILED', entry['name'], code, flush=True)
                    else:
                        done.append(entry['name'])
                    del running[gpu]
            # Let the existing training scheduler finish assigning all queued jobs.
            # Once its pending list is empty it will not claim another idle GPU.
            training = json.loads(TRAIN_QUEUE.read_text()) if TRAIN_QUEUE.exists() else {}
            training_pending = training.get('pending', [])
            training_gpus = set(map(int, training.get('running', {})))
            if pending and not training_pending:
                for item in idle_gpus():
                    gpu = item['index']
                    if not pending or not item['idle'] or gpu not in gpus or gpu in running or gpu in training_gpus:
                        continue
                    name = pending.pop(0)
                    log = (OUT/f'{name}.log').open('a')
                    proc = subprocess.Popen([sys.executable, str(__file__), '--worker', name], cwd=ROOT,
                        env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'PYTHONUNBUFFERED':'1'},
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    running[gpu] = dict(name=name, proc=proc, log=log)
                    print('STARTED', name, 'GPU', gpu, 'PID', proc.pid, flush=True)
            status = dict(completed=done, failed=failed, pending=pending, running={str(g):dict(name=e['name'],pid=e['proc'].pid)
                for g,e in running.items()}, waiting_for_training_queue=bool(training_pending))
            write_json(OUT/'status.json', status)
            state=(len(done),len(pending),len(running),bool(training_pending))
            if state != previous_state:
                print('STATUS', json.dumps(status), flush=True)
                previous_state=state
            if pending or running:
                time.sleep(20)
        subprocess.run([sys.executable,str(P/'report.py')], cwd=ROOT, check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--launch',action='store_true')
    parser.add_argument('--background',action='store_true')
    parser.add_argument('--gpus',type=int,nargs='+',default=list(range(8)))
    parser.add_argument('--worker')
    args=parser.parse_args()
    os.chdir(ROOT)
    if args.worker:
        worker(args.worker)
        return
    if args.background and not args.launch:
        parser.error('--background requires --launch')
    if args.prepare or not (DATA/'manifest.json').exists():
        prepare()
    if args.launch:
        if args.background:
            OUT.mkdir(parents=True,exist_ok=True)
            with (OUT/'launch.log').open('a') as log:
                proc=subprocess.Popen([sys.executable,str(__file__),'--launch','--gpus',*map(str,args.gpus)],
                    cwd=ROOT,env={**os.environ,'PYTHONUNBUFFERED':'1'},stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            print('BACKGROUND PID',proc.pid,'LOG',OUT/'launch.log')
        else:
            coordinator(args.gpus)


if __name__ == '__main__':
    main()

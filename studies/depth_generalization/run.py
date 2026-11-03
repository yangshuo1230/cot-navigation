"""Frozen clean-depth evaluation; reuse native graph sampler, decoder and audit."""
import argparse
from collections import Counter
import csv
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/mnt/ningxuefei/yangshuo/cot-navigation-unified')
sys.path.insert(0, str(ROOT / 'src'))
os.environ['WANDB_MODE'] = 'disabled'
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'TORCHINDUCTOR_COMPILE_THREADS']:
    os.environ.setdefault(key, '2')
from cot_navigation.experiments.scheduler import write_json, file_lock
from cot_navigation.data.shards import file_sha256

OUT = ROOT / 'outputs/analysis/depth_generalization_20261103'
DATA = ROOT / 'data/evaluation/depth_generalization_20261103'
SOURCE = ROOT / 'outputs/depth_sweep_4096_20261101/hard'
SHAPES = {12:(3,4),16:(4,4),24:(4,6),32:(4,8),36:(6,6),48:(6,8),
          64:(8,8),80:(8,10),100:(10,10),120:(10,12),144:(12,12),169:(13,13),196:(14,14)}

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with gzip.open(tmp, 'wt', encoding='utf-8') as f:
        json.dump(value, f, separators=(',', ':'))
    tmp.replace(path)

def read(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return json.load(f)

def prepare():
    import numpy as np
    from cot_navigation.config import load_config
    from cot_navigation.data.generate import _generate_context_row
    from cot_navigation.data.splits import iter_sample_plan
    from cot_navigation.data.reachability import bfs_reaches
    from cot_navigation.data.graph import adj_from_presentation
    from cot_navigation.data.hard_graph import shortest_distances
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = DATA / 'manifest.json'
    if path.exists():
        manifest = json.loads(path.read_text())
        for cell in manifest['cells']:
            assert file_sha256(DATA/cell['file']) == cell['sha256']
        for model in manifest['models']:
            assert file_sha256(Path(model['checkpoint'])) == model['sha256']
        return manifest
    cfg = load_config([SOURCE/'dfs_nodes/L3/s0/resolved_config.yaml'])
    forbidden = set()
    for root in cfg.train.scale_roots:
        for split in ['train', 'val', 'test']:
            for shard in (ROOT/root/split).glob('shard-*.npz'):
                with np.load(shard, allow_pickle=False) as arrays:
                    forbidden.update(map(int, arrays['struct_hash']))
    manifest = dict(seed=20261103, family='multiscale_dag_v1', n_edges='2N',
        samples=512, positives=256, context=4096, precision='bf16', batch_size=32,
        cot_margin=8, training_max_n=100, forbidden_original_graphs=len(forbidden),
        query_policy='Native equal quotas per rank-band distance and label; fresh graphs, shared by all models',
        graph_identity='Pre-rename topological-edge hash, not graph isomorphism',
        cells=[], models=[])
    for n, (layers, width) in SHAPES.items():
        payload = dict(split='test', n_nodes=n, n_edges=2*n, graph_family='multiscale_dag_v1',
            node_vocab=256, data_seed=20261103+n, max_attempts=2000,
            shapes={'main':(layers,width)}, shape_names=['main'], forbidden=forbidden)
        rows=[]
        for plan in iter_sample_plan(['main'], {'main':layers}, 512):
            row=_generate_context_row(payload,plan)
            adj=adj_from_presentation(n,list(zip(row['edge_src'],row['edge_dst'])))
            assert bfs_reaches(adj,row['s'],row['t'])==bool(row['label'])
            assert int(row['struct_hash']) not in forbidden
            forbidden.add(int(row['struct_hash']))
            row['shortest_hops']=shortest_distances(adj,row['s'])[row['t']]
            row['state_reference_fits']=6*n+4+row['dfs_trace_tokens']+3<=4096
            rows.append(row)
        assert Counter(x['label'] for x in rows)=={0:256,1:256}
        dest=DATA/f'hard_n{n}.json.gz';save(dest,rows)
        manifest['cells'].append(dict(n=n,layers=layers,width=width,file=dest.name,
            sha256=file_sha256(dest),state_reference_fit_rate=sum(x['state_reference_fits'] for x in rows)/512))
        print('DATA',n,flush=True)
    for protocol in ['dfs_nodes','dfs_state']:
        for depth in [3,6,9,12]:
            for seed in [0,1]:
                folder=SOURCE/protocol/f'L{depth}'/f's{seed}'
                assert (folder/'COMPLETE.json').exists()
                ckpt=folder/'checkpoints/stage_01000000.pt'
                manifest['models'].append(dict(protocol=protocol,depth=depth,seed=seed,
                    config=str(folder/'resolved_config.yaml'),checkpoint=str(ckpt),sha256=file_sha256(ckpt)))
    write_json(path,manifest)
    return manifest

def metrics(rows):
    result={}
    for population,label in [('all',None),('positive',1),('negative',0)]:
        rr=[r for r in rows if label is None or r['label']==label]
        result[population]=dict(n=len(rr),accuracy=sum(r['correct'] for r in rr)/len(rr),
            legality=sum(r['valid'] for r in rr)/len(rr),
            joint=sum(r['correct'] and r['valid'] for r in rr)/len(rr),
            forced_stop=sum(r['forced_stop'] for r in rr)/len(rr),
            interrupted=sum(r['interrupted'] for r in rr)/len(rr),
            errors=dict(Counter(r['error'] for r in rr if not r['valid'])))
    return result

def _summarize():
    results=[]
    for path in sorted((OUT/'results').glob('*.json.gz')):
        item=read(path)
        results.append(dict(protocol=item['model']['protocol'],depth=item['model']['depth'],
            seed=item['model']['seed'],n=item['cell']['n'],metrics=item['metrics']))
    write_json(OUT/'summary.json',dict(expected=16*len(SHAPES),complete=len(results),results=results))
    with (OUT/'results.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['protocol','depth','seed','n','population','samples','accuracy','legality','forced_stop'])
        for r in results:
            for pop,v in r['metrics'].items():
                writer.writerow([r['protocol'],r['depth'],r['seed'],r['n'],pop,v['n'],v['accuracy'],v['legality'],v['forced_stop']])

def summarize():
    with file_lock(OUT/'.summary.lock'):
        _summarize()

def worker(protocol,depth,seed=None,scales=None):
    import numpy as np
    import torch
    from cot_navigation.config import load_config
    from cot_navigation.models.transformer import build_model
    from cot_navigation.models.compiled_inference import CompiledInference
    from cot_navigation.protocols.tokenizer import Tokenizer
    from cot_navigation.protocols.render import context_prompt,render_task
    from cot_navigation.protocols.trace_audit import validate_trace
    from cot_navigation.training.checkpoint import load_checkpoint
    from cot_navigation.evaluation.rollout import generate
    torch.set_num_threads(2)
    manifest=json.loads((DATA/'manifest.json').read_text())
    identities=[m for m in manifest['models'] if m['protocol']==protocol and m['depth']==depth and (seed is None or m['seed']==seed)]
    cfg=load_config([identities[0]['config']]);tok=Tokenizer.from_config(cfg)
    assert cfg.model.max_seq_len==4096 and not cfg.protocol.enable_mistake
    raw=build_model(cfg.model,len(tok),pad_id=tok.pad).cuda().eval().requires_grad_(False)
    model=CompiledInference(raw).eval()
    for identity in identities:
        checkpoint=Path(identity['checkpoint']);assert file_sha256(checkpoint)==identity['sha256']
        blob=load_checkpoint(checkpoint,map_location='cpu');raw.load_state_dict(blob['model'],strict=True);del blob
        cells=manifest['cells'] if scales is None else [next(c for c in manifest['cells'] if c['n']==n) for n in scales]
        for cell in cells:
            dest=OUT/'results'/f'{protocol}_L{depth}_s{identity["seed"]}_n{cell["n"]}.json.gz'
            if dest.exists():
                previous=read(dest)
                assert previous['model']==identity and previous['data_sha256']==cell['sha256']
                continue
            rows=read(DATA/cell['file']);order=np.random.default_rng(1729).permutation(len(rows));audits=[]
            start=time.monotonic()
            for offset in range(0,len(order),manifest['batch_size']):
                chunk=[rows[int(i)] for i in order[offset:offset+manifest['batch_size']]]
                nodes=[[tok.node_id(v) for v in row['node_token']] for row in chunk]
                prompts=[context_prompt(tok,nn,list(zip(row['edge_src'],row['edge_dst'])),row['s'],row['t']) for row,nn in zip(chunk,nodes)]
                with torch.no_grad():
                    generated=generate(model,prompts,protocol,tok,'bf16',max_new_tokens=0,cot_margin=8)
                for row,nn,prompt,g in zip(chunk,nodes,prompts,generated):
                    adj={nn[u]:[] for u in range(cell['n'])}
                    for u,v in zip(row['edge_src'],row['edge_dst']):adj[nn[u]].append(nn[v])
                    # Validate against truth independently of the predicted answer.
                    check=validate_trace(g['trace_tokens'],row['label'],protocol,adj,nn[row['s']],nn[row['t']],tok)
                    if offset==0 and len(audits)<4:
                        topology=[[] for _ in nn]
                        for u,v in zip(row['edge_src'],row['edge_dst']):topology[u].append(v)
                        gold=render_task(tok,protocol,row['label'],prompt,topology,row['s'],row['t'],nn)
                        gold_check=validate_trace(gold.tokens[gold.prompt_len:gold.answer_index],row['label'],protocol,adj,nn[row['s']],nn[row['t']],tok)
                        assert gold_check['valid'],gold_check
                    audits.append(dict(query_id=row['sample_id'],label=row['label'],pred=g['pred'],
                        correct=not g['format_fail'] and g['pred']==row['label'],valid=bool(check['valid']),error=check.get('error'),
                        forced_stop=bool(g['forced_stop']),interrupted=bool(g['cot_interrupted']),
                        graph_hash=str(row['struct_hash']),distance=row['distance'],shortest_hops=row['shortest_hops'],
                        state_reference_fits=row['state_reference_fits'],trace_tokens=g['trace_tokens']))
            save(dest,dict(model=identity,cell=cell,data_sha256=cell['sha256'],context=4096,
                metrics=metrics(audits),rows=audits,seconds=time.monotonic()-start,compile_fallbacks=sorted(model.compile_fallbacks)))
            summarize()
            print('DONE',protocol,depth,identity['seed'],cell['n'],round(time.monotonic()-start,1),flush=True)
        assert file_sha256(checkpoint)==identity['sha256']

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    with file_lock(OUT/'.run.lock',blocking=False):
        prepare()
        for protocol in ['dfs_nodes','dfs_state']:
            for depth in [3,6,9,12]:
                write_json(OUT/'status.json',dict(protocol=protocol,depth=depth,complete=False))
                subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',protocol,'--depth',str(depth)],check=True,cwd=ROOT)
        summarize();write_json(OUT/'COMPLETE.json',dict(cells=16*len(SHAPES)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',action='store_true')
    parser.add_argument('--launch',action='store_true');parser.add_argument('--worker');parser.add_argument('--depth',type=int);parser.add_argument('--seed',type=int);parser.add_argument('--scales',type=int,nargs='+')
    args=parser.parse_args();os.chdir(ROOT)
    if args.prepare:prepare()
    elif args.worker:worker(args.worker,args.depth,args.seed,args.scales)
    elif args.launch:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'launch.log').open('a') as log:
            proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--run'],cwd=ROOT,
                env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','PYTHONUNBUFFERED':'1'},stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print('PID',proc.pid,'LOG',OUT/'launch.log')
    elif args.run:run()

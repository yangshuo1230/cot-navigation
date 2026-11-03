"""Versioned manifest-driven natural behavior workflow; frozen models, GPU0."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['WANDB_MODE']='disabled'
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','TORCHINDUCTOR_COMPILE_THREADS']:os.environ.setdefault(key,'2')
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
import numpy as np
import torch
from cot_navigation.analysis.error_continuation import audit_continuation
from cot_navigation.analysis.error_dependencies import dependencies
from cot_navigation.config import load_config
from cot_navigation.data.dataset import ContextStore
from cot_navigation.data.scale_prepared import ScalePreparedRows
from cot_navigation.evaluation.rollout import generate
from cot_navigation.experiments.artifacts import atomic_json,sha_file
from cot_navigation.models.compiled_inference import CompiledInference
from cot_navigation.models.transformer import build_model
from cot_navigation.protocols.correction_replay import replay
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.training.checkpoint import load_checkpoint


def write_cache(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
    with gzip.open(temp,'wt',encoding='utf-8',compresslevel=3) as f:json.dump(data,f,separators=(',',':'))
    temp.replace(path)


def resolve(path):return Path(path) if Path(path).is_absolute() else ROOT/path


def checkpoint_step(manifest,entry):
    """v1 fixes 200K; v2 pins each token-matched model's actual final step."""
    return 200000 if manifest['protocol_version']==1 else int(entry['checkpoint_step'])


def prepare(manifest):
    if manifest.get('lifecycle') == 'deprecated':
        raise ValueError('Cohort withdrawn by user; do not run or reuse these models')
    assert manifest['protocol_version'] in (1,2)
    assert manifest['settings']==dict(queries=256,batch=32,query_seed=20261031,
        checkpoint_step=200000 if manifest['protocol_version']==1 else 'per_model_final',
        precision='bf16',population='reachable',max_seq_len=4096)
    out=resolve(manifest['output']);out.mkdir(parents=True,exist_ok=True)
    sources=[Path(__file__),P/'report.py',P/'dashboard_template.html',P/'PROTOCOL.md',P.parent/'correction_reproduction/error_continuation/report.py',P.parent/'correction_reproduction/error_continuation/length_analysis.py',ROOT/'src/cot_navigation/analysis/error_continuation.py',ROOT/'src/cot_navigation/analysis/error_dependencies.py',ROOT/'src/cot_navigation/evaluation/rollout.py']
    plan=dict(manifest=manifest,source_sha256={str(f.relative_to(ROOT)):sha_file(f) for f in sources},models=[],tasks={})
    ids=set()
    for entry in manifest['models']:
        assert entry['id'] not in ids and all(c.isalnum() or c in '_-' for c in entry['id']);ids.add(entry['id'])
        folder=resolve(entry['run']);cfg=load_config([folder/'resolved_config.yaml'])
        assert cfg.protocol.name=='dfs_state' and cfg.model.layers==entry['layers']
        if entry.get('training_arm') == 'matched_clean':
            proof=json.loads((folder/'reproduction_identity.json').read_text())
            assert proof['arm']=='matched_clean' and entry['error_rate']==0
        else:
            assert abs(cfg.protocol.mistake_rate-entry['error_rate'])<1e-8
        assert not (folder/'DEPRECATED.json').exists() and not (folder.parents[2]/'DEPRECATED.json').exists()
        assert cfg.train.model_seed==entry['seed'] and cfg.model.max_seq_len==4096
        if manifest['protocol_version']==2:assert checkpoint_step(manifest,entry)==cfg.train.max_steps
        ckpt=folder/'checkpoints/latest.pt';assert ckpt.exists()
        if manifest['protocol_version']==2:
            blob=load_checkpoint(ckpt,map_location='cpu');assert blob['step']==checkpoint_step(manifest,entry);del blob
        plan['models'].append(dict(entry,checkpoint=str(ckpt),checkpoint_sha256=sha_file(ckpt),config_sha256=sha_file(folder/'resolved_config.yaml')))
        roots=dict(x.split('=',1) for x in cfg.evaluation.extra_sets);roots['main']=cfg.data.root
        for task in manifest['tasks']:
            suffix='' if task=='main' else '-'+task
            oldfile=folder/f'eval/predictions_test_last{suffix}.jsonl'
            old=[json.loads(line) for line in oldfile.read_text().splitlines()]
            positives=[r for r in old if r['label']==1]
            assert len(positives)==256 and len({r['query_id'] for r in positives})==256
            selected=[positives[int(i)] for i in np.random.default_rng(20261031).permutation(256)]
            source=ContextStore(roots[task])
            task_identity=dict(root=roots[task],fingerprint=source.fingerprint,queries=[r['query_id'] for r in selected],distances=[r['distance'] for r in selected])
            if task in plan['tasks']:assert plan['tasks'][task]==task_identity,'Model cohorts do not share graph/query identities'
            else:plan['tasks'][task]=task_identity
    path=out/'plan.json'
    if path.exists():assert json.loads(path.read_text())==plan,'Frozen plan changed; use a new cohort output'
    else:atomic_json(path,plan)
    return out,plan


def run(manifest,out,plan):
    torch.set_num_threads(2)
    total=len(plan['models'])*len(manifest['tasks']);finished=0
    for entry in plan['models']:
        paths=[out/'results'/f"{entry['id']}__{task}.json.gz" for task in manifest['tasks']]
        if all(f.exists() and json.load(gzip.open(f,'rt'))['complete'] for f in paths):finished+=len(paths);continue
        folder=resolve(entry['run']);cfg=load_config([folder/'resolved_config.yaml']);tok=Tokenizer.from_config(cfg)
        raw=build_model(cfg.model,len(tok),pad_id=tok.pad).cuda().eval().requires_grad_(False)
        blob=load_checkpoint(Path(entry['checkpoint']),map_location='cpu');assert blob['step']==checkpoint_step(manifest,entry)
        raw.load_state_dict(blob['model'],strict=True);del blob
        model=CompiledInference(raw).eval()
        for task,path in zip(manifest['tasks'],paths):
            identity=dict(model=entry,task=task,task_identity=plan['tasks'][task],settings=manifest['settings'],sources=plan['source_sha256'])
            data=json.load(gzip.open(path,'rt')) if path.exists() else dict(identity=identity,complete=False,rows=[])
            assert data['identity']==identity
            if data['complete']:finished+=1;continue
            source=ContextStore(plan['tasks'][task]['root']);reference=cfg
            if task=='hard':reference=load_config([ROOT/'outputs/hard_dag_v1/launches/hard_dag_v1_s0/context/curriculum/dfs_state/resolved_config.yaml'])
            reftok=Tokenizer.from_config(reference);assert tok.itos[:len(reftok)]==reftok.itos
            prepared=ScalePreparedRows(source,reference,reftok)
            qs=plan['tasks'][task]['queries'];ds=plan['tasks'][task]['distances']
            corpus_path=out/'probe_corpus'/f"{entry['id']}__test.json.gz"
            capture=bool(manifest.get('capture_probe_test') and task=='main')
            if capture:
                corpus_identity=dict(model=entry,fingerprint=source.fingerprint,split='test',queries=qs)
                corpus=json.load(gzip.open(corpus_path,'rt')) if corpus_path.exists() else dict(identity=corpus_identity,complete=False,rows=[])
                assert corpus['identity']==corpus_identity
                assert len(corpus['rows'])==len(data['rows'])
            assert [r['query'] for r in data['rows']]==qs[:len(data['rows'])]
            for start in range(len(data['rows']),256,32):
                refs=[prepared.row(q,0) for q in qs[start:start+32]]
                with torch.inference_mode():outputs=generate(model,[r.tokens[:r.prompt_len] for r in refs],'dfs_state',tok,'bf16',max_new_tokens=0,cot_margin=cfg.protocol.cot_margin)
                for j,(ref,g) in enumerate(zip(refs,outputs)):
                    q=qs[start+j];nodes=source.node_vocab_ids(q,tok)
                    adj={nodes[u]:[nodes[v] for v in vs] for u,vs in enumerate(source.adj(q))}
                    src=nodes[int(source.arrays['s_topo'][q])];dst=nodes[int(source.arrays['t_topo'][q])]
                    trace=g['trace_tokens'];detail=audit_continuation(trace,adj,src,dst,tok)
                    shared=replay(trace,g['pred'],adj,src,dst,tok,label=1)
                    if capture:
                        edges=sorted(zip(source.arrays['edge_src'][q].tolist(),source.arrays['edge_dst'][q].tolist()))
                        graph_hash=hashlib.sha256(json.dumps([edges,source.arrays['node_token'][q].tolist()]).encode()).hexdigest()
                        marker=tok.stoi.get('<MISTAKE>',-1)
                        corpus['rows'].append(dict(query=q,trace=trace,first=shared['first'],graph_hash=graph_hash,
                            marker_at=trace.index(marker) if marker in trace else None,source=entry['id'],split='test',
                            analysis_stopped=False,generation_limit=g['generation_limit_reached'],forced_stop=g['forced_stop']))
                    first=next((r for r in detail['records'] if r['bad']),None)
                    if first is not None and shared['first'] is not None:assert first['record']==shared['first']['record']
                    data['rows'].append(dict(query=q,distance=ds[start+j],correct=bool(g['pred']==1 and not g['format_fail']),
                        pred=g['pred'],interrupted=g['cot_interrupted'],forced_stop=g['forced_stop'],trace_valid=shared['trace_valid'],
                        first=shared['first'],true_marks=shared['true_marks'],false_marks=shared['false_marks'],repairs=shared['repairs'],
                        trace_tokens=len(trace),prompt_tokens=ref.prompt_len,
                        trace_sha256=hashlib.sha256(np.asarray(trace,np.int32).tobytes()).hexdigest(),
                        dependencies=dependencies(trace,adj,src,dst,tok),**detail))
                if capture:write_cache(corpus_path,corpus)
                write_cache(path,data);print('BEHAVIOR',entry['id'],task,len(data['rows']),flush=True)
            if capture:corpus['complete']=True;write_cache(corpus_path,corpus)
            data['complete']=True;write_cache(path,data);finished+=1
            atomic_json(out/'status.json',dict(phase='running',complete_tasks=finished,total_tasks=total))
            subprocess.run([sys.executable,str(P/'report.py'),'--output',str(out)],check=True)
        assert sha_file(Path(entry['checkpoint']))==entry['checkpoint_sha256']
        del model,raw;torch.cuda.empty_cache()
    subprocess.run([sys.executable,str(P/'report.py'),'--output',str(out)],check=True)
    atomic_json(out/'status.json',dict(phase='complete',complete_tasks=finished,total_tasks=total))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--prepare-only',action='store_true');args=ap.parse_args()
    os.chdir(ROOT);manifest=json.loads(args.manifest.read_text());out,plan=prepare(manifest)
    if args.prepare_only:print('PREPARED',len(plan['models']),'models',len(plan['tasks']),'graph settings');return
    try:run(manifest,out,plan)
    except BaseException as exc:
        atomic_json(out/'failure.json',dict(error=repr(exc)))
        previous=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
        atomic_json(out/'status.json',dict(previous,phase='failed',error=repr(exc)));raise


if __name__=='__main__':main()

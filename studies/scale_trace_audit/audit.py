"""Frozen scale-checkpoint semantic audit. No optimizer, W&B, or training writes.

Main cases use exactly the original final-test query order and batch size.
Alternative valid DFS/BFS priorities are accepted. STOP prefix validity and
complete search validity are deliberately separate quantities.
"""
from pathlib import Path
from collections import Counter
import argparse
import gzip
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(Path(__file__).resolve().parent)]
from cot_navigation.config import load_config
from cot_navigation.data.dataset import ContextStore, ParametricWorld
from cot_navigation.data.scale_prepared import ScalePreparedRows
from cot_navigation.evaluation.rollout import generate
from cot_navigation.evaluation.sampling import stratified_ids
from cot_navigation.models.transformer import build_model
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.protocols.render import render_steps
from cot_navigation.protocols.dfs import dfs_search
from cot_navigation.protocols.bfs import bfs_search, first_expansion_order
from legacy_trace_audit import validate_trace, validate_execution, self_test

RUN = ROOT/'outputs/scale_v1/launches/scale_v1_s0'
OUT = ROOT/'studies/scale_trace_audit/results'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    tmp.replace(path)

def semantic(trace, pred, protocol, adj, src, dst, tok, negative=False):
    body=list(trace)
    stop=tok.stoi.get('<STOP>')
    stopped=stop in body if stop is not None else False
    if stopped:
        if body[-1]!=stop or body.count(stop)!=1:
            return dict(complete_valid=False,prefix_valid=False,error='misplaced_stop',records=0,
                        valid_records=0,valid_prefix_tokens=0,skeleton_valid=False,empty_prefix=False,stop=True)
        body=body[:-1]
    verdict=validate_trace(body,pred,protocol,adj,src,dst,tok,negative)
    complete=bool(verdict['valid'])
    error=verdict.get('error')
    if protocol.endswith('_state'):
        # The legacy replayer reaches this error only after ALL displayed
        # records and legal actions passed; it has not compared any gold order.
        prefix=complete or error=='missing_or_inconsistent_terminal'
        skeleton=validate_execution(body,pred,adj,src,dst,tok,protocol=='bfs_state')
    elif protocol.endswith('_nodes'):
        prefix=complete or error=='missing_or_inconsistent_terminal'
        skeleton=complete
    else:
        # Path negatives are walks, not exhaustive non-reachability proofs.
        prefix=bool(body) and body[0]==src and len(body)==len(set(body))
        prefix=prefix and all(v in adj for v in body) and all(v in adj[u] for u,v in zip(body,body[1:]))
        skeleton=complete
    if stopped and not body:
        prefix=True  # Explicitly report zero-progress prefixes separately.
    return dict(complete_valid=complete,prefix_valid=bool(prefix),error=error,
                records=body.count(tok.id(';')) if protocol.endswith('_state') else len(body),
                valid_records=int(verdict.get('valid_records',0)),
                valid_prefix_tokens=int(verdict.get('valid_prefix_tokens',0)),
                skeleton_valid=bool(skeleton),empty_prefix=not body,stop=stopped)

def make_jobs():
    plan=json.loads((RUN/'launch_plan.json').read_text())
    jobs=[]
    for name,spec in plan.items():
        cfg=spec['config']
        if cfg['protocol']['name']=='direct' or cfg['train']['stage']=='A':
            continue
        cases=[('main',None,256)]
        if name.startswith(('context/l10/','context/curriculum/')) or 'knowledge_transfer' in name:
            cases += [(case,path,64) for case,path in (s.split('=',1) for s in cfg['evaluation']['extra_sets'])
                      if case in {'deep','wide','larger'}]
        jobs.append(dict(name=name,folder=cfg['output_dir'],cases=cases))
    return jobs

def do_job(job):
    folder=Path(job['folder']); cfg=load_config([folder/'resolved_config.yaml'])
    protocol=cfg.protocol.name;tok=Tokenizer.from_config(cfg)
    dest=OUT/job['name'];dest.mkdir(parents=True,exist_ok=True)
    checkpoint=folder/'checkpoints/latest.pt';before=sha(checkpoint)
    model=build_model(cfg.model,len(tok),tok.pad).cuda().eval().requires_grad_(False)
    blob=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(blob['model'],strict=True);step=int(blob['step']);del blob
    assert step==cfg.train.max_steps
    for case,path,limit in job['cases']:
        result=dest/(case+'.json')
        if result.exists():
            saved=json.loads(result.read_text())
            if saved.get('checkpoint_sha256')==before and saved.get('requested_samples')==limit:
                continue
        start=time.time()
        context=cfg.data.knowledge=='context_graph'
        if context:
            source=ContextStore(path or cfg.data.root);a=source.arrays
            ids=source.indices('test');shapes=a['shape_id']
        else:
            source=ParametricWorld(path or str(Path(cfg.data.root)/f'{cfg.data.graph_shape}-seed{cfg.data.graph_seed}'))
            a=source.queries;ids=source.indices('test',cfg.evaluation.fixed_query_view)
            shapes=np.zeros(len(a['label']),dtype=np.int8)
        # Reconstruct exactly the original 256 selection before taking an OOD
        # diagnostic subset; all OOD audit IDs remain in the original test.
        original=stratified_ids(ids,shapes,a['distance'],a['label'],cfg.evaluation.test_samples,cfg.evaluation.seed)
        chosen=original if limit>=len(original) else stratified_ids(np.asarray(original),shapes,a['distance'],a['label'],limit,cfg.evaluation.seed)
        oldcase='' if case=='main' else '-'+case
        previous=folder/'eval'/('predictions_test_last'+oldcase+'.jsonl')
        old={int(r['query_id']):r for r in map(json.loads,previous.read_text().splitlines())} if previous.exists() else {}
        cache=ScalePreparedRows(source,cfg,tok)
        fixed_nodes=None if context else source.node_vocab_ids(tok)
        fixed_adj=None if context else source.adj()
        worldprefix=[]
        if not context and cfg.train.world_roots:
            index=[Path(p).resolve() for p in cfg.train.world_roots].index(source.root.resolve())
            worldprefix=[tok.id(f'<WORLD_{index}>')]
        counts=Counter();errors=Counter();by_hop={};examples=[]
        batch=cfg.evaluation.batch_size
        with gzip.open(dest/(case+'.jsonl.gz'),'wt',encoding='utf-8') as output:
            for offset in range(0,len(chosen),batch):
                subset=chosen[offset:offset+batch]; records=[]
                for idx in subset:
                    idx=int(idx);row=cache.row(idx,0)
                    nodes=source.node_vocab_ids(idx,tok) if context else fixed_nodes
                    raw=source.adj(idx) if context else fixed_adj
                    adj={nodes[u]:[nodes[v] for v in vs] for u,vs in enumerate(raw)}
                    src,dst=nodes[int(a['s_topo'][idx])],nodes[int(a['t_topo'][idx])]
                    # Check the real prepared trajectory, not a new reference.
                    ref=semantic(row.tokens[row.prompt_len:row.answer_index],row.label,protocol,adj,src,dst,tok,cfg.protocol.negative_random_path)
                    assert ref['complete_valid'] or (ref['stop'] and ref['prefix_valid']),(job['name'],case,idx,'invalid prepared reference',ref)
                    # Independent reachability oracle, unrelated to model trace.
                    seen={src};todo=[src]
                    while todo:
                        u=todo.pop()
                        for v in adj[u]:
                            if v not in seen:seen.add(v);todo.append(v)
                    assert int(dst in seen)==int(row.label)
                    records.append((idx,row,worldprefix+row.tokens[:row.prompt_len],adj,src,dst))
                generated=generate(model,[r[2] for r in records],protocol,tok,cfg.train.precision,
                                   max_new_tokens=cfg.evaluation.max_new_tokens,cot_margin=cfg.protocol.cot_margin)
                for rec,gen in zip(records,generated):
                    idx,row,prompt,adj,src,dst=rec
                    v=semantic(gen['trace_tokens'],gen['pred'],protocol,adj,src,dst,tok,cfg.protocol.negative_random_path)
                    good=not gen['format_fail'] and gen['pred']==row.label
                    item=dict(index=idx,label=row.label,pred=gen['pred'],distance=int(a['distance'][idx]),
                              correct=good,format_fail=gen['format_fail'],interrupted=gen['cot_interrupted'],
                              forced_stop=gen['forced_stop'],trace_tokens=gen['trace_tokens'],**v)
                    delta=Counter(n=1,answer_correct=int(good),complete_valid=int(v['complete_valid']),
                                  prefix_valid=int(v['prefix_valid']),joint_correct=int(good and v['complete_valid']),
                                  skeleton_valid=int(v['skeleton_valid']),interrupted=int(gen['cot_interrupted']),
                                  forced_stop=int(gen['forced_stop']),format_fail=int(gen['format_fail']),
                                  interrupted_prefix_valid=int(gen['cot_interrupted'] and v['prefix_valid']),
                                  empty_prefix=int(v['empty_prefix']),valid_records=v['valid_records'],records=v['records'],
                                  old_prediction_matches=int(idx in old and gen['pred']==old[idx]['pred']),
                                  old_prediction_compared=int(idx in old))
                    counts.update(delta);by_hop.setdefault(item['distance'],Counter()).update(delta)
                    if not v['complete_valid']:errors[v['error'] or 'incomplete']+=1
                    if not v['complete_valid'] and len(examples)<8:
                        examples.append({**item,'prompt':tok.decode(prompt),'trace_preview':tok.decode(gen['trace_tokens'][:350])})
                    output.write(json.dumps(item)+'\n')
                print(json.dumps(dict(job=job['name'],case=case,done=min(offset+batch,len(chosen)),total=len(chosen))),flush=True)
        n=counts['n']
        summary=dict(job=job['name'],protocol=protocol,case=case,requested_samples=limit,n=n,
                     counts=dict(counts),errors=dict(errors),by_distance={str(k):dict(v) for k,v in by_hop.items()},
                     answer_accuracy=counts['answer_correct']/n,complete_trace_accuracy=counts['complete_valid']/n,
                     prefix_valid_rate=counts['prefix_valid']/n,skeleton_accuracy=counts['skeleton_valid']/n,
                     joint_accuracy=counts['joint_correct']/n,seconds=time.time()-start,
                     checkpoint_sha256=before,checkpoint_step=step,precision=cfg.train.precision,
                     batch_size=batch,source=str(source.root),sample_ids=list(map(int,chosen)),examples=examples)
        write(result,summary)
        del cache,source
    assert sha(checkpoint)==before,'checkpoint changed during frozen audit'
    write(dest/'complete.json',dict(checkpoint_sha256=before,checkpoint_step=step,cases=[c[0] for c in job['cases']]))

def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',type=int)
    parser.add_argument('--workers',type=int,default=7)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--job-file',type=Path,help='explicit frozen-run jobs for a subsequent training batch')
    parser.add_argument('--output-root',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if args.self_test:
        tok=Tokenizer(enable_stop=True,world_tokens=15)
        count=self_test(tok,dfs_search,bfs_search,render_steps,first_expansion_order)
        # STOP at a record boundary accepts a correct prefix, never a proof.
        nodes=[tok.node_id(i) for i in range(4)];raw=[[1,2],[3],[],[]]
        adj={nodes[i]:[nodes[j] for j in vs] for i,vs in enumerate(raw)}
        for protocol,search in [('dfs_state',dfs_search),('bfs_state',bfs_search)]:
            body=render_steps(search(raw,0,3)[:1],nodes,tok)+[tok.id('<STOP>')]
            v=semantic(body,0,protocol,adj,nodes[0],nodes[3],tok)
            assert v['prefix_valid'] and not v['complete_valid'],v
            body[0]=tok.id('<WORLD_0>')
            assert not semantic(body,0,protocol,adj,nodes[0],nodes[3],tok)['prefix_valid']
        print(json.dumps({'validator_assertions':count,'stop_checks':'passed'}));return
    if args.output_root is not None:
        OUT=args.output_root.resolve()
    if args.job_file is not None:
        for job in json.loads(args.job_file.read_text()):
            do_job(job)
        write(OUT/'finished.json',dict(jobs=len(json.loads(args.job_file.read_text()))))
        return
    jobs=make_jobs()
    if args.worker is None:
        write(OUT.parent/'plan.json',dict(jobs=jobs,main_samples=256,ood_samples=64,workers=args.workers))
        print(json.dumps({'jobs':len(jobs),'conditions':sum(len(j['cases']) for j in jobs)}));return
    for index,job in enumerate(jobs):
        if index%args.workers==args.worker:
            do_job(job)

if __name__=='__main__':
    main()

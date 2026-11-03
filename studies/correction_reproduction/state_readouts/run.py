"""Field identity, search-state readouts and logit lens on a validated cohort.

Natural text is never edited. All receivers see the same seed-0 donor pool for
probe fitting; primary tests use each receiver's own unedited held-out traces.
"""
import argparse
from collections import defaultdict
import gc
import json
import os
from pathlib import Path
import runpy
import time

os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
os.environ['WANDB_MODE']='disabled'
ROOT=Path(__file__).resolve().parents[3]
P=Path(__file__).resolve().parent
import numpy as np
import torch
from cot_navigation.analysis.common import records
from cot_navigation.analysis.state_readout import fit_readout,evaluate_readout
from cot_navigation.analysis.within_token import fit_lens,lens_logits
from cot_navigation.experiments.artifacts import atomic_json,sha_file
R=runpy.run_path(str(P.parent/'natural_history_probes/prepare.py'))
FIELDS=['frontier','visited','candidates','action']
SETS=['frontier','visited','listed_candidates','successors','legal_candidates','successors_pre']


def load_cases(arm,seed,train_count,val_count):
    cases={};sources=[]
    for split in ['train','val','test']:
        rows=[]
        for source in (['matched_clean','correction'] if split!='test' else [arm]):
            path=P.parent/'natural_history_probes/native'/f'{source}_s{0 if split!="test" else seed}_{split}.json.gz'
            d=R['read_cache'](path);assert d['complete']
            count={'train':train_count,'val':val_count,'test':256}[split]
            rows.extend(d['rows'][:count]);sources.append(dict(path=str(path.relative_to(ROOT)),sha256=sha_file(path)))
        cases[split]=rows
    for a,b in [('train','val'),('train','test'),('val','test')]:
        assert not {r['query'] for r in cases[a]}&{r['query'] for r in cases[b]}
        assert not {r['graph_hash'] for r in cases[a]}&{r['graph_hash'] for r in cases[b]}
    return cases,sources


def select(am,cases):
    """At most four records/query; no selection using activations or correctness.

    Node identity: one sampled node per field/record. Bars: same token '|',
    classified by which boundary it closes. State: third bar, with all emitted
    state fields visible. Graph targets are computed from actual prompt edges.
    """
    result={};rng=np.random.default_rng(20261102)
    for split,cc in cases.items():
        rows=[]
        for sid,c in enumerate(cc):
            ex=am.example(c['query'],split);rr=records(c['trace'],0,am.tok)
            selected=sorted(rng.choice(len(rr),size=min(4,len(rr)),replace=False)) if rr else []
            graph={ex.nodes[u]:[ex.nodes[v] for v in vs] for u,vs in enumerate(ex.adj)}
            for ri in selected:
                r=rr[ri];a,b,d=r['separators'];spans=[(r['start'],a),(a+1,b),(b+1,d),(d+1,r['end'])]
                for label,(lo,hi) in enumerate(spans):
                    ps=[p for p in range(lo,hi) if am.tok.node_lo<=c['trace'][p]<am.tok.node_lo+am.tok.n_node_tokens]
                    if ps:rows.append(dict(sid=sid,pos=int(rng.choice(ps)),task='field_node',y=label,field=FIELDS[label],record=int(ri)))
                for label,pos in enumerate([a,b,d]):
                    rows.append(dict(sid=sid,pos=pos,task='field_bar',y=label,field=FIELDS[label],record=int(ri)))
                current=r['frontier'][-1]
                if current not in graph:continue
                successor=graph[current];visited=set(r['visited'])
                targets=[r['frontier'],r['visited'],r['candidates'],successor,[n for n in successor if n not in visited]]
                rows.append(dict(sid=sid,pos=d,task='state',field='candidates',record=int(ri),
                                 targets=targets,nodes=ex.nodes,current=current))
                # Here visited/candidate text of this record has not been read.
                rows.append(dict(sid=sid,pos=a,task='state_pre',field='frontier',record=int(ri),
                                 targets=targets,nodes=ex.nodes,current=current))
        result[split]=rows
    return result


@torch.no_grad()
def extract(am,cases,rows):
    n=len(rows);dim=am.model.d_model;layers=am.model.n_layers
    x=np.zeros((n,layers+2,dim+2),np.float32);nxt=np.zeros(n,np.int64)
    locations=defaultdict(list)
    for i,r in enumerate(rows):locations[r['sid']].append((i,r))
    order=sorted(locations,key=lambda sid:max(r['pos'] for _,r in locations[sid]))
    for start in range(0,len(order),4):
        ids0=order[start:start+4];seqs=[];prompts=[]
        for sid in ids0:
            c=cases[sid];prompt=am.example(c['query'],c['split']).prompt
            prompts.append(len(prompt));end=max(r['pos'] for _,r in locations[sid])+1
            seqs.append(prompt+c['trace'][:end])
        ids=torch.full((len(seqs),max(map(len,seqs))),am.tok.pad,device=am.device,dtype=torch.long)
        for j,seq in enumerate(seqs):ids[j,:len(seq)]=torch.tensor(seq,device=am.device)
        cap={};am.model(ids,capture=cap)
        for j,sid in enumerate(ids0):
            ii=np.array([i for i,_ in locations[sid]]);pp=np.array([r['pos'] for _,r in locations[sid]])
            abspos=pp+prompts[j]
            x[ii,:layers,:dim]=torch.stack([h[j,abspos].float() for h in cap['resid']],1).cpu().numpy()
            emb=am.model.embed(ids[j,abspos]).float().cpu().numpy()
            x[ii,layers:,:dim]=emb[:,None]
            x[ii,layers+1,dim]=abspos/4096;x[ii,layers+1,dim+1]=pp/4096
            nxt[ii]=[cases[sid]['trace'][p+1] if p+1<len(cases[sid]['trace']) else -1 for p in pp]
        del ids,cap
    return dict(x=x,next=nxt)


def probe_data(rows,features,task,target=None,tok=None):
    ids=np.array([i for i,r in enumerate(rows) if r['task']==task],int)
    x=features['x'][ids]
    if task not in {'state','state_pre'}:return dict(x=x,y=np.array([rows[i]['y'] for i in ids],np.int64))
    y=np.zeros((len(ids),tok.n_node_tokens),np.float32);mask=np.zeros_like(y)
    for j,i in enumerate(ids):
        r=rows[i]
        for node in r['nodes']:mask[j,node-tok.node_lo]=1
        for node in r['targets'][target]:
            if tok.node_lo<=node<tok.node_lo+tok.n_node_tokens:y[j,node-tok.node_lo]=1
    return dict(x=x,y=y,mask=mask)


@torch.no_grad()
def lens_scores(am,lens,features,rows):
    out=[];layers=am.model.n_layers;dim=am.model.d_model
    for task in ['field_node','field_bar','state']:
        take=np.array([i for i,r in enumerate(rows) if r['task']==task and features['next'][i]>=0],int)
        if not len(take):continue
        for kind in ['raw','calibrated']:
            for layer in range(layers):
                totals=defaultdict(float)
                for start in range(0,len(take),256):
                    ix=take[start:start+256];h=torch.as_tensor(features['x'][ix,layer,:dim],device=am.device)
                    z=lens_logits(am.model,lens,h,layer,kind)
                    final=lens_logits(am.model,None,torch.as_tensor(features['x'][ix,layers-1,:dim],device=am.device),layers-1,'raw')
                    y=torch.as_tensor(features['next'][ix],device=am.device);pred=z.argmax(-1)
                    totals['correct']+=int((pred==y).sum());totals['agreement']+=int((pred==final.argmax(-1)).sum())
                    totals['nll']+=float(torch.nn.functional.cross_entropy(z,y,reduction='sum'))
                    totals['kl']+=float(torch.nn.functional.kl_div(z.log_softmax(-1),final.softmax(-1),reduction='sum'))
                out.append(dict(task=task,kind=kind,layer=layer+1,n=len(take),accuracy=totals['correct']/len(take),
                    final_agreement=totals['agreement']/len(take),nll=totals['nll']/len(take),kl_to_final=totals['kl']/len(take)))
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',default='studies/correction_behavior/cohorts/legacy6_s0123_supplement_20261102.json')
    p.add_argument('--steps',type=int,default=1500);p.add_argument('--train-queries',type=int,default=512)
    p.add_argument('--val-queries',type=int,default=128);p.add_argument('--limit-models',type=int,default=0)
    args=p.parse_args();os.chdir(ROOT);torch.set_num_threads(2)
    manifest=json.loads(Path(args.manifest).read_text());assert manifest.get('lifecycle')!='deprecated'
    out=ROOT/'outputs/analysis/state_readouts'/manifest['cohort'];out.mkdir(parents=True,exist_ok=True)
    plan=dict(schema='natural_state_readout_v1',args=vars(args),manifest_sha256=sha_file(Path(args.manifest)),
        heads=['L1','L2','L3','L4','L5','L6','token_only','token_position'],tasks=SETS,
        scope='main N100 simple DAG, positive queries, naturally generated prefix windows',
        state_anchor='third bar, after visible state and before action',threshold=.5)
    if (out/'plan.json').exists():assert json.loads((out/'plan.json').read_text())==plan
    atomic_json(out/'plan.json',plan)
    models=manifest['models'][:args.limit_models or None]
    for m in models:
        dest=out/(m['id']+'.json')
        if dest.exists():continue
        started=time.time();print('MODEL',m['id'],flush=True)
        assert not (ROOT/m['run']/'DEPRECATED.json').exists()
        am=R['Model'](m['training_arm'],m['seed']);assert am.run==ROOT/m['run']
        cases,sources=load_cases(m['training_arm'],m['seed'],args.train_queries,args.val_queries)
        rows=select(am,cases);features={s:extract(am,cases[s],rr) for s,rr in rows.items()}
        print('EXTRACTED',m['id'],{s:len(rr) for s,rr in rows.items()},flush=True)
        results=[]
        for task in ['field_node','field_bar']+SETS:
            categorical=task.startswith('field_')
            anchor='state_pre' if task=='successors_pre' else 'state'
            target=3 if task=='successors_pre' else (None if categorical else SETS.index(task))
            dd={s:probe_data(rr,features[s],task if categorical else anchor,target,am.tok) for s,rr in rows.items()}
            for hidden in [0,64]:
                fitted=fit_readout(dd['train'],dd['val'],outputs=(4 if task=='field_node' else 3) if categorical else am.tok.n_node_tokens,
                    categorical=categorical,hidden=hidden,steps=args.steps,seed=20261102)
                result=dict(task=task,architecture='linear' if hidden==0 else 'mlp64',steps=fitted['steps'],
                    validation_loss=fitted['validation_loss'],test=evaluate_readout(fitted,dd['test'],categorical))
                results.append(result);print('PROBE',m['id'],task,hidden,flush=True)
                del fitted
        # Lens calibration sees only final output distributions, no state labels.
        lens_parts={};rng=np.random.default_rng(20261102)
        for split in ['train','val']:
            take=rng.permutation(len(rows[split]))[:4096 if split=='train' else 2048]
            h=features[split]['x'][take,:6,:am.model.d_model].copy()
            with torch.no_grad():
                logits=am.model.lm_head(am.model.final_norm(torch.as_tensor(h[:,-1],device=am.device))).cpu().numpy()
            lens_parts[split]=(h,logits)
        lens=fit_lens(am.model,*lens_parts['train'],*lens_parts['val'],steps=args.steps,seed=20261102)
        ls=lens_scores(am,lens,features['test'],rows['test'])
        assert all(abs(x['final_agreement']-1)<1e-8 for x in ls if x['layer']==6)
        atomic_json(dest,dict(model=m,checkpoint=am.checkpoint,sources=sources,
            rows={s:len(rr) for s,rr in rows.items()},queries={s:len({c['query'] for c in cc}) for s,cc in cases.items()},
            probes=results,logit_lens=ls,lens_validation={k:lens[k] for k in ['steps','validation_kl','identity_kl']},
            elapsed=time.time()-started,complete=True))
        print('DONE',m['id'],round(time.time()-started),flush=True)
        runpy.run_path(str(P/'report.py'))['report'](out)
        del am,features,lens,lens_parts;gc.collect();torch.cuda.empty_cache()
    atomic_json(out/'status.json',dict(complete=all((out/(m['id']+'.json')).exists() for m in manifest['models']),
        finished=[m['id'] for m in manifest['models'] if (out/(m['id']+'.json')).exists()],total=len(manifest['models'])))


if __name__=='__main__':main()

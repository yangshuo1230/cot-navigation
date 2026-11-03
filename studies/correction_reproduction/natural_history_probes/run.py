"""Natural-only probe fitting and primary tests; artificial edits are controls.

Reuses latest-checkpoint adapters, token-position probe banks, semantic replay,
and artificial-control extraction. Neither natural branch is repaired/spliced.
"""
import argparse
from collections import Counter,defaultdict
import gc
import os
from pathlib import Path
import runpy
import time

os.environ.setdefault('CUDA_VISIBLE_DEVICES','0');os.environ['WANDB_MODE']='disabled'
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(k,'2')
P=Path(__file__).resolve().parent;ROOT=P.parents[2]

import numpy as np
import torch
from cot_navigation.analysis.natural_history import annotate,match_readouts,FIELDS
from cot_navigation.analysis.history_probe import fit_positions,position_scores
from cot_navigation.analysis.common import records
from cot_navigation.protocols.correction_replay import replay
from cot_navigation.experiments.artifacts import atomic_json,sha_file

PREP=runpy.run_path(str(P/'prepare.py'));OLD=PREP['OLD'];Model=PREP['Model'];HEADS=OLD['HEADS']


def load_data(tok):
    grouped={s:[] for s in ['train','val','test']};provenance=[];fingerprints=set()
    for path in sorted((P/'native').glob('*.json.gz')):
        d=PREP['read_cache'](path);assert d['complete']
        fingerprints.add(d['identity']['source_fingerprint'])
        split=d['identity']['split'];grouped[split].extend(d['rows'])
        assert [r['query'] for r in d['rows']]==d['identity']['queries']
        assert all(r['split']==split for r in d['rows'])
        provenance.append(dict(file=path.name,sha256=sha_file(path),checkpoint=d['identity']['checkpoint'],
            rows=len(d['rows']),split=split,errors=sum(r['first'] is not None for r in d['rows'])))
    assert len(provenance)==12 and len(fingerprints)==1
    hashes={s:{c['graph_hash'] for c in cc} for s,cc in grouped.items()}
    queries={s:{c['query'] for c in cc} for s,cc in grouped.items()}
    for a,b in [('train','val'),('train','test'),('val','test')]:
        assert not hashes[a]&hashes[b];assert not queries[a]&queries[b]
    cases={s:annotate(cc,tok) for s,cc in grouped.items()}
    plans={};summaries={}
    for j,split in enumerate(['train','val','test']):
        plans[split],summaries[split]=match_readouts(cases[split],tok,20261030+j)
        assert plans[split],split
        for r in plans[split]:
            c=cases[split][r['sid']]
            assert r['pos']<len(c['trace'])
            assert int(c['tau'] is not None and r['pos']>=c['tau'])==r['y']
        print('MATCH',split,summaries[split],flush=True)
    summaries['source_files']=provenance
    summaries['queries_per_split']={s:len(q) for s,q in queries.items()}
    summaries['first_types_raw']={s:dict(Counter(c['first']['kind'] for c in cc if c['first'])) for s,cc in grouped.items()}
    summaries['before_marker_positive_cases']={s:sum(c['has_observed_error'] for c in cc) for s,cc in cases.items()}
    summaries['marker_cases']={s:sum(c['marker_at'] is not None for c in cc) for s,cc in grouped.items()}
    return cases,plans,summaries


def extract(am,cases,rows):
    """Gather frozen FP32 residuals once per unmodified natural sequence."""
    if not rows:return None
    n=len(rows);dim=am.model.d_model;layers=am.model.n_layers
    features=np.zeros((n,layers+2,dim+8),np.float32)
    locations=defaultdict(list)
    for i,r in enumerate(rows):locations[r['sid']].append((i,r))
    table=am.model.embed.weight.detach().cpu().numpy();prompts={}
    order=sorted(locations,key=lambda sid:max(r['pos'] for _,r in locations[sid]))
    for start in range(0,len(order),8):
        chunk=order[start:start+8];seqs=[]
        for sid in chunk:
            q=cases[sid]['query']
            if q not in prompts:prompts[q]=am.example(q,cases[sid]['split']).prompt
            end=max(r['pos'] for _,r in locations[sid])+1
            seqs.append(prompts[q]+cases[sid]['trace'][:end])
        ids=torch.full((len(chunk),max(map(len,seqs))),am.tok.pad,device=am.device,dtype=torch.long)
        for b,seq in enumerate(seqs):ids[b,:len(seq)]=torch.tensor(seq,device=am.device)
        capture={}
        with torch.inference_mode():am.model(ids,capture=capture)
        for b,sid in enumerate(chunk):
            case=cases[sid];prompt_len=len(prompts[case['query']]);items=locations[sid]
            dest=np.array([i for i,_ in items]);pos=np.array([r['pos'] for _,r in items]);absolute=pos+prompt_len
            hidden=torch.stack([h[b,absolute].float() for h in capture['resid']],1).cpu().numpy()
            features[dest,:layers,:dim]=hidden
            chosen=np.array(case['trace'])[pos];features[dest,layers,:dim]=table[chosen]
            bag=np.cumsum(table[np.array(case['trace'])],axis=0)
            features[dest,layers+1,:dim]=bag[pos]/(pos[:,None]+1)
            obs=np.zeros((len(dest),8),np.float32);obs[:,0]=absolute/4096;obs[:,1]=pos/4096
            for j,field in enumerate(FIELDS):obs[:,2+j]=[r['field']==field for _,r in items]
            obs[:,7]=(chosen>=am.tok.node_lo)&(chosen<am.tok.node_lo+am.tok.n_node_tokens)
            features[dest,layers:,-8:]=obs[:,None]
        del capture,ids
        if start%256==0:print('FEATURES',start,'/',len(order),flush=True)
    data={k:np.array([r[k] for r in rows]) for k in ['y','query','event','kind','field','phase','offset','cell','exact_token','positive_query','source']}
    data.update(x=features,y=data['y'].astype(np.float32),after=np.ones(n,bool))
    return data


def subset(data,mask):
    return {k:(v[mask] if isinstance(v,np.ndarray) and v.ndim>0 and len(v)==len(mask) else v)
            for k,v in data.items()}


def compact(scores):
    grouped=defaultdict(list)
    for r in scores:grouped[(r['record'],r['head'])].append(r)
    out=[]
    for (phase,head),rr in sorted(grouped.items()):
        out.append(dict(record=phase,head=head,cells=len(rr),
            **{k:float(np.mean([r[k] for r in rr if r.get(k) is not None]))
               for k in ['auroc','paired_rank','positive_p','negative_p','bce','brier','tpr','fpr']}))
    return out


def artificial_controls(am,observed,limit=64):
    """Only post-hoc controls; error categories must occur in natural training."""
    rng=np.random.default_rng(20261101);events=defaultdict(list)
    supported={'candidate_nonedge','candidate_out_of_graph','candidate_visited','candidate_duplicate',
        'candidate_omission','visited_omission','visited_extra','frontier_wrong_order','frontier_wrong_node'}&set(observed)
    for q in am.sample_ids('test',256):
        ex=am.example(q,'test');good=ex.row.tokens[:ex.row.answer_index]
        rr=records(good,len(ex.prompt),am.tok)
        common=[ri for ri,r in enumerate(rr[:-2]) if len(r['candidates'])>=2 and len(r['visited'])>=2 and len(r['frontier'])>=2]
        if not common:continue
        shared_record=int(rng.choice(common))
        for kind in sorted(supported):
            if len(events[kind])>=limit:continue
            choices=[shared_record]
            for ri in choices:
                r=rr[ri];bars=r['separators'];end=rr[ri+2]['end']+1
                bad=list(good[:end]);lo=bars[1]+1;hi=bars[2];vl=bars[0]+1;vh=bars[1]
                cands=bad[lo:hi];visited=bad[vl:vh];nodes=ex.nodes
                if kind.startswith('candidate_'):
                    if len(cands)<2:continue
                    if kind=='candidate_omission':bad=bad[:hi-1]+bad[hi:]
                    else:
                        if kind=='candidate_duplicate':pool=[cands[0]]
                        elif kind=='candidate_visited':pool=visited
                        elif kind=='candidate_out_of_graph':pool=[am.tok.node_id(i) for i in range(am.tok.n_node_tokens) if am.tok.node_id(i) not in nodes]
                        else:
                            cur=nodes.index(r['frontier'][-1]);pool=[n for n in nodes if n not in visited and n not in {nodes[v] for v in ex.adj[cur]}]
                        if not pool:continue
                        bad[hi-1]=int(rng.choice(pool))
                elif kind=='visited_omission':
                    if len(visited)<2:continue
                    bad=bad[:vl]+bad[vl+1:]
                elif kind=='visited_extra':
                    pool=[n for n in nodes if n not in visited]
                    if not pool:continue
                    bad=bad[:vh]+[int(rng.choice(pool))]+bad[vh:]
                elif kind=='frontier_wrong_order':
                    if len(r['frontier'])<2:continue
                    bad[r['start']],bad[r['start']+1]=bad[r['start']+1],bad[r['start']]
                else:
                    pool=[n for n in nodes if n not in r['frontier']]
                    if not pool:continue
                    bad[r['start']]=int(rng.choice(pool))
                graph={nodes[u]:[nodes[v] for v in vs] for u,vs in enumerate(ex.adj)}
                audit=replay(bad[len(ex.prompt):],None,graph,nodes[ex.source],nodes[ex.target],am.tok)
                first=audit['first']
                if first is None or first['kind']!=kind:continue
                assert replay(good[len(ex.prompt):end],None,graph,nodes[ex.source],nodes[ex.target],am.tok)['first'] is None
                events[kind].append(dict(query=q,correct=good[:end],wrong=bad,prompt_length=len(ex.prompt),
                    focus_start=r['start'],focus_record=ri,first_error=len(ex.prompt)+first['token']-1,
                    kind=kind,graph_control=False))
                break
        if all(len(events[k])>=limit for k in supported):break
    # All reported type controls use the same original queries and records.
    shared=set.intersection(*[{e['query'] for e in events[k]} for k in supported]) if supported else set()
    events={k:[e for e in v if e['query'] in shared] for k,v in events.items()}
    return {k:OLD['extract'](am,ev) for k,ev in events.items() if ev}, {k:len(v) for k,v in events.items()}


def fit_model(arm,seed,args):
    dest=P/'results'/f'{arm}_s{seed}.json'
    import json
    if dest.exists() and json.loads(dest.read_text()).get('complete'):return
    started=time.time();am=Model(arm,seed)
    cases,plans,info=load_data(am.tok)
    own=f'{arm}_s{seed}'
    own_rows,own_info=match_readouts(cases['test'],am.tok,20261102,positive_source=own)
    datasets={s:extract(am,cases[s],plans[s]) for s in ['train','val','test']}
    datasets['own_test']=extract(am,cases['test'],own_rows)
    controls,control_counts=artificial_controls(am,info['first_types_raw']['train'])
    result=dict(complete=False,arm=arm,model_seed=seed,checkpoint=am.checkpoint,heads=HEADS,
        target='past-or-current semantic error in an unedited natural prefix, BEFORE any MISTAKE',
        natural_sampling=info,own_sampling=own_info,artificial_control_counts=control_counts,
        steps=args.steps,probe_seeds=[0,1],sharing='none: independent cell/layer parameters and normalization',runs=[])
    # Dataset is fixed across all frozen checkpoints; no artificial data enters
    # probe training, validation, checkpoint selection, or the primary test.
    del am;gc.collect();torch.cuda.empty_cache()
    for hidden,arch in [(0,'linear'),(64,'mlp64')]:
        for ps in [0,1]:
            fitted=fit_positions(datasets['train'],datasets['val'],args.steps,ps,hidden=hidden)
            if fitted is None:raise RuntimeError('Not enough natural train/validation examples; do not fill with synthetic errors.')
            item=dict(architecture=arch,probe_seed=ps,selected_steps=fitted['selected_steps'],
                val_bce=fitted['validation_bce'],tests={},by_first_type={},by_field={},artificial_controls={},control_by_field={})
            for name in ['test','own_test']:
                if datasets[name] is not None:item['tests'][name]=position_scores(fitted,datasets[name])
            exact=subset(datasets['test'],datasets['test']['exact_token'].astype(bool))
            item['tests']['exact_token_test']=compact(position_scores(fitted,exact))
            for field in FIELDS:
                dd=subset(datasets['test'],datasets['test']['field']==field)
                item['by_field'][field]=compact(position_scores(fitted,dd))
            for kind in sorted(set(datasets['test']['kind'])):
                dd=subset(datasets['test'],datasets['test']['kind']==kind)
                item['by_first_type'][str(kind)]=compact(position_scores(fitted,dd))
            for kind,dd in controls.items():
                item['artificial_controls'][kind]=compact(position_scores(fitted,dd))
                item['control_by_field'][kind]={field:compact(position_scores(fitted,subset(dd,dd['field']==field))) for field in FIELDS}
            result['runs'].append(item);atomic_json(dest,result)
            print('NATURAL_FIT',arm,seed,arch,ps,'seconds',round(time.time()-started),flush=True)
            del fitted;gc.collect();torch.cuda.empty_cache()
    shuffled={}
    for j,split in enumerate(['train','val','test']):
        d=datasets[split];rng=np.random.default_rng(4477+j)
        flips={int(e):int(rng.integers(2)) for e in np.unique(d['event'])}
        yy=np.array([int(y)^flips[int(e)] for y,e in zip(d['y'],d['event'])],np.float32)
        shuffled[split]={**d,'y':yy}
    fitted=fit_positions(shuffled['train'],shuffled['val'],args.steps,0,hidden=64)
    result['shuffle_control']=compact(position_scores(fitted,shuffled['test']))
    result.update(complete=True,seconds=time.time()-started,
        checkpoint_unchanged=sha_file(Path(result['checkpoint']['path']))==result['checkpoint']['sha256'])
    assert result['checkpoint_unchanged']
    atomic_json(dest,result);print('NATURAL_MODEL_FINISHED',arm,seed,round(result['seconds']),flush=True)
    del fitted,datasets,controls,shuffled;gc.collect();torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--steps',type=int,default=5000)
    parser.add_argument('--model-seeds',type=int,nargs='+',default=[0,1,2,3])
    args=parser.parse_args();os.chdir(ROOT);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.32)
    assert (P/'prepared.json').exists(),'Finish natural prepare.py first'
    (P/'results').mkdir(exist_ok=True)
    atomic_json(P/'probe_plan.json',dict(vars(args),gpu=0,
        train='natural-only from matched/correction seed0',test='natural shared + checkpoint own unedited trajectories',
        source_sha256={str(p.relative_to(ROOT)):sha_file(p) for p in [Path(__file__),ROOT/'src/cot_navigation/analysis/natural_history.py',ROOT/'src/cot_navigation/analysis/history_probe.py']}))
    for seed in args.model_seeds:
        for arm in ['matched_clean','correction']:fit_model(arm,seed,args)


if __name__=='__main__':main()

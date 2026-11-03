"""All-layer, all-token history-error probes on the latest matched checkpoints.

Frozen navigation checkpoints; existing canonical searches and paired graph
controls; no MISTAKE token, activations, or probe checkpoints are persisted.
"""
import argparse
from collections import Counter
from dataclasses import replace
import gc
import hashlib
import json
import os
from pathlib import Path
import runpy
import time

P=Path(__file__).resolve().parent
ROOT=P.parents[2]
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
os.environ['WANDB_MODE']='disabled'
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(k,'2')

import numpy as np
import torch
from cot_navigation.analysis.common import AnalysisModel, records
from cot_navigation.analysis.history_probe import history_target, score, fit_positions, position_scores
from cot_navigation.config import load_config
from cot_navigation.data.dataset import ContextStore
from cot_navigation.data.scale_prepared import ScalePreparedRows
from cot_navigation.models.transformer import build_model
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.protocols.correction_replay import replay
from cot_navigation.experiments.artifacts import atomic_json,sha_file
from cot_navigation.training.checkpoint import load_checkpoint

LEGACY=runpy.run_path(str(ROOT/'studies/history_error/run.py'))
HEADS=['L1','L2','L3','L4','L5','L6','token_position_control','history_bag_control']


class Model(AnalysisModel):
    def __init__(self,arm,seed):
        self.run=ROOT/f'outputs/correction_reproduction_v1/launches/dfs_correction_repro_20261028/dfs_state/{arm}/s{seed}'
        self.cfg=load_config([self.run/'resolved_config.yaml']);self.tok=Tokenizer.from_config(self.cfg)
        self.protocol='dfs_state';self.filler=0;self.context=True;self.device=torch.device('cuda:0')
        self.root=ROOT;self.store=ContextStore(ROOT/self.cfg.data.root)
        checkpoint=self.run/'checkpoints/latest.pt';state=load_checkpoint(checkpoint,map_location='cpu')
        assert state['step']==200000
        self.model=build_model(self.cfg.model,len(self.tok),self.tok.pad).to(self.device).eval().requires_grad_(False)
        self.model.load_state_dict(state['model'],strict=True)
        self.checkpoint=dict(path=str(checkpoint),sha256=sha_file(checkpoint),step=state['step'])
        self.layers=list(range(self.model.n_layers));del state

    def sample_ids(self,split,count):
        ids=self.indices(split);ids=ids[self.store.arrays['label'][ids]==1]
        return np.random.default_rng(20261030+{'train':0,'val':1,'test':2}[split]).permutation(ids)[:count].tolist()


def graph_id(am,q):
    # Exclude query endpoints: even two different queries on the same graph
    # must never land in different probe splits. Include node relabeling.
    a=am.store.arrays
    edges=sorted(zip(a['edge_src'][q].tolist(),a['edge_dst'][q].tolist()))
    return hashlib.sha256(json.dumps([edges,a['node_token'][q].tolist()]).encode()).hexdigest()


def windows(am,events,make_graph=False):
    out=[];graphs=[]
    shape=am.cfg.data.shapes[am.cfg.data.graph_shape or 'main']
    width=int(shape.width if hasattr(shape,'width') else shape['width'])
    for e in events:
        recs=records(e['correct'],e['prompt_length'],am.tok)
        ri=next(i for i,r in enumerate(recs) if r['start']<=e['error_position']<=r['end'])
        if ri+2>=len(recs):continue
        end=recs[ri+2]['end']+1
        event={**e,'correct':e['correct'][:end],'wrong':e['wrong'][:end],
               'focus_start':recs[ri]['start'],'focus_record':ri,'first_error':e['error_position'],
               'kind':'in_graph_candidate_replacement','graph_id':graph_id(am,e['query'])}
        out.append(event)
        if make_graph:
            inv={n:i for i,n in enumerate(am.store.node_vocab_ids(e['query'],am.tok))}
            used={inv[r['frontier'][-1]] for r in recs[:ri+3]}
            ge={**event,'positions':{'record_1_vis':end-1},'expanded_through_next_vis':sorted(used)}
            # The original control stopped before the next candidate field.
            # A longer window can revisit the edited source, invalidating its
            # legal branch. Retain only controls passing its independent oracle.
            try:control=LEGACY['graph_pair'](am,am.example(e['query'],'test'),ge,width)
            except AssertionError:control=None
            if control:graphs.append({**control,'focus_start':event['focus_start'],
                'focus_record':ri,'first_error':event['first_error'],'kind':'identical_cot_graph_control',
                'graph_id':event['graph_id']})
    return out,graphs


def extract(am,events):
    """One causal forward per paired sequence; gather every focus-window token."""
    if not events:return None
    store={k:[] for k in ['x','y','query','event','kind','field','phase','offset','cell','after','branch']}
    embedding=am.model.embed.weight.detach().cpu().numpy()
    max_pre=0.;n=0
    ordered=sorted(enumerate(events),key=lambda r:len(r[1]['wrong']))
    for eid,e in ordered:
        seqs=[e['correct'],e['wrong']];length=max(map(len,seqs))
        ids=torch.full((2,length),am.tok.pad,device=am.device,dtype=torch.long)
        for b,seq in enumerate(seqs):ids[b,:len(seq)]=torch.tensor(seq,device=am.device)
        capture={}
        with torch.inference_mode():am.model(ids,capture=capture)
        if not e.get('graph_control') and e.get('kind')=='in_graph_candidate_replacement':
            p=e['first_error']-1
            max_pre=max(max_pre,max(float((h[0,p]-h[1,p]).abs().max()) for h in capture['resid']))
        for branch,seq in enumerate(seqs):
            rs=records(seq,e['prompt_length'],am.tok)
            selected=rs[e['focus_record']:e['focus_record']+3]
            for phase,r in enumerate(selected):
                positions=np.arange(r['start'],r['end']+1);nrows=len(positions)
                hs=torch.stack([h[branch,positions].float() for h in capture['resid']],1).cpu().numpy()
                features=np.zeros((nrows,len(HEADS),am.model.d_model+8),np.float32)
                features[:,:6,:am.model.d_model]=hs
                fields=np.array(['frontier' if p<r['separators'][0] else
                    'visited' if p<r['separators'][1] else 'candidates' if p<r['separators'][2] else
                    'semicolon' if p==r['end'] else 'action' for p in positions])
                obs=np.zeros((nrows,8),np.float32)
                obs[:,0]=positions/4096;obs[:,1]=(positions-e['prompt_length'])/4096
                for i,f in enumerate(['frontier','visited','candidates','action','semicolon']):obs[:,2+i]=(fields==f)
                obs[:,7]=np.array([am.tok.node_lo<=seq[p]<am.tok.node_lo+am.tok.n_node_tokens for p in positions])
                features[:,6,:am.model.d_model]=embedding[np.array(seq)[positions]]
                bag=np.cumsum(embedding[np.array(seq[e['prompt_length']:])],axis=0)
                rel=positions-e['prompt_length']
                features[:,7,:am.model.d_model]=bag[rel]/(rel[:,None]+1)
                features[:,6:,-8:]=obs[:,None]
                after=history_target(positions,e['first_error']).astype(bool)
                values=dict(x=features,y=after.astype(np.float32)*branch,
                    query=np.full(nrows,e['query']),event=np.full(nrows,eid),
                    kind=np.full(nrows,e['kind']),field=fields,phase=np.full(nrows,phase),
                    offset=np.arange(nrows),cell=phase*10000+np.arange(nrows),after=after,
                    branch=np.full(nrows,branch))
                for k,v in values.items():store[k].append(v)
        del capture,ids
        n+=1
        if n%256==0:print('EXTRACT',n,'/',len(events),flush=True)
    assert max_pre<1e-5,max_pre
    result={k:np.concatenate(v) for k,v in store.items()}
    result['pre_error_max_difference']=max_pre
    return result


def predict_chunked(fitted,data,batch=4096):
    model,mean,scale,_=fitted;parts=[]
    with torch.no_grad():
        for i in range(0,len(data['y']),batch):
            x=torch.as_tensor(data['x'][i:i+batch],device=mean.device)
            parts.append(model((x-mean)/scale).cpu().numpy())
    return np.concatenate(parts)


def summarize(data,z):
    rows=[]
    for h,name in enumerate(HEADS):
        for phase in range(3):
            for field in ['all','frontier','visited','candidates','action','semicolon']:
                take=(data['phase']==phase)&data['after']
                if field!='all':take&=(data['field']==field)
                if take.any():rows.append(dict(head=name,phase=phase,field=field,**score(data['y'][take],z[take,h])))
    return rows


def natural_test(am):
    """Held-out donor errors, not synthetic edits or training-pool episodes."""
    from cot_navigation.data.natural_mistakes import pool_path
    a=np.load(pool_path(am.store,am.cfg,am.tok),allow_pickle=False)
    prepared=ScalePreparedRows(am.store,am.cfg,am.tok);out=[]
    for i in np.flatnonzero(a['split']==2):
        meta=json.loads(str(a['metadata'][i]));q=meta['index']
        if not am.store.arrays['label'][q]:continue
        row=prepared.row(q,int(meta['natural_variant']));seq=row.tokens[:row.answer_index]
        event=meta['natural_events'][0];rs=records(seq,row.prompt_len,am.tok)
        ri=next((j for j,r in enumerate(rs) if r['start']==event['original_start']),None)
        if ri is None or ri+2>=len(rs):continue
        r=rs[ri];lo=r['separators'][1]+1;hi=r['separators'][2]
        clean=seq[:rs[ri+2]['end']+1];wrong=clean[:lo]+event['predicted']+clean[hi:]
        adj=am.store.adj(q);nodes=am.store.node_vocab_ids(q,am.tok)
        a_ids={nodes[u]:[nodes[v] for v in vs] for u,vs in enumerate(adj)}
        src=nodes[int(am.store.arrays['s_topo'][q])];dst=nodes[int(am.store.arrays['t_topo'][q])]
        audit=replay(wrong[row.prompt_len:],None,a_ids,src,dst,am.tok)
        first=audit.get('first')
        if first is None:continue
        out.append(dict(query=q,correct=clean,wrong=wrong,prompt_length=row.prompt_len,
            focus_start=r['start'],focus_record=ri,first_error=row.prompt_len+first['token']-1,
            kind=first['kind'],graph_control=False,graph_id=graph_id(am,q)))
    return out


def run(arm,seed,args):
    dest=args.output_dir/f'{arm}_s{seed}.json'
    if dest.exists() and json.loads(dest.read_text()).get('finished'):return
    start=time.time();am=Model(arm,seed);result=dict(finished=False,arm=arm,seed=seed,
        checkpoint=am.checkpoint,heads=HEADS,steps=args.steps,probe_seeds=[0,1],
        target='history through consumed input position p contains an error; activation predicts p+1',
        probe_sharing='NONE: separate weights, biases, standardization, validation selection per layer/record/token offset',
        evaluation_schema='paired_after_error_v2: both branches retained only after first error is observable; excludes onset/field shortcut',
        precision='float32 frozen forward; TF32 disabled',gpu=0,
        population='positive N100 in-context; one controlled event per original query',runs=[])
    data={};graph={};splits={};graphs={}
    for split,target in [('train',args.train_pairs),('val',args.val_pairs),('test',args.test_pairs)]:
        events,_,inventory=LEGACY['build_events'](am,split,target,min(50000,target*30))
        events,ge=windows(am,events,True)
        splits[split]={e['query'] for e in events};graphs[split]={e['graph_id'] for e in events}
        result.setdefault('sampling',{})[split]=dict(inventory,window_pairs=len(events),graph_pairs=len(ge),
            query_ids=sorted(splits[split]),graph_hashes=sorted(graphs[split]))
        assert len(events)>=min(target,64), (split,len(events))
        print('DATA',arm,seed,split,len(events),'graph',len(ge),flush=True)
        data[split]=extract(am,events);graph[split]=extract(am,ge)
    for a,b in [('train','val'),('train','test'),('val','test')]:
        assert not splits[a]&splits[b];assert not graphs[a]&graphs[b]
    result['split_check']='query AND graph hashes disjoint; all edits/positions of a query stay together'
    # Donor natural test remains external; exclude any graph used for probe fitting.
    natural=natural_test(am);natural=[e for e in natural if e['graph_id'] not in graphs['train']|graphs['val']]
    if natural:data['natural_test']=extract(am,natural)
    result['natural_pairs']=len(natural)
    result['pre_error_max_difference']={s:d['pre_error_max_difference'] for s,d in data.items()}
    del am;gc.collect();torch.cuda.empty_cache()
    for hidden,architecture in [(0,'linear_independent'),(64,'mlp64_independent')]:
        for ps in [0,1]:
            fitted=fit_positions(data['train'],data['val'],args.steps,ps,hidden=hidden)
            item=dict(architecture=architecture,probe_seed=ps,selected_steps=fitted['selected_steps'],
                      val_bce=fitted['validation_bce'],tests={})
            for name in ['test','natural_test']:
                if name in data:item['tests'][name]=position_scores(fitted,data[name])
            if graph['test'] is not None:
                item['tests']['identical_cot_graph_zero_shot']=position_scores(fitted,graph['test'])
            pre=~data['test']['after']
            item['before_error']=dict(n=int(pre.sum()),all_labels_zero=bool((data['test']['y'][pre]==0).all()))
            result['runs'].append(item);del fitted;gc.collect();torch.cuda.empty_cache()
            atomic_json(dest,result);print('FIT',arm,seed,architecture,ps,round(time.time()-start),flush=True)
    # Train an additional context-only control: COT text is identical in both labels.
    if all(graph[s] is not None for s in ['train','val','test']):
        fitted=fit_positions(graph['train'],graph['val'],args.steps,0,hidden=64)
        result['graph_trained']=dict(scores=position_scores(fitted,graph['test']),
            selected_steps=fitted['selected_steps'] if fitted else [])
        del fitted;gc.collect();torch.cuda.empty_cache()
    # Label randomization flips the error/clean branch per whole query; onset
    # remains unchanged. Never shuffle individual tokens across a split.
    shuffled={}
    for j,s in enumerate(['train','val','test']):
        d=data[s];rng=np.random.default_rng(444+j)
        flips={int(q):int(rng.integers(2)) for q in np.unique(d['query'])}
        yy=np.array([(int(b)^flips[int(q)])*int(a) for q,b,a in zip(d['query'],d['branch'],d['after'])],np.float32)
        shuffled[s]={**d,'y':yy}
    fitted=fit_positions(shuffled['train'],shuffled['val'],args.steps,0,hidden=64)
    result['shuffle_control']=position_scores(fitted,shuffled['test'])
    del fitted,shuffled;gc.collect();torch.cuda.empty_cache()
    del data,graph;gc.collect();torch.cuda.empty_cache()
    assert sha_file(Path(result['checkpoint']['path']))==result['checkpoint']['sha256']
    result.update(finished=True,seconds=time.time()-start)
    atomic_json(dest,result);print('FINISHED',arm,seed,round(result['seconds']),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=5000)
    parser.add_argument('--output-dir',type=Path,default=P/'results_paired_v2')
    parser.add_argument('--train-pairs',type=int,default=768)
    parser.add_argument('--val-pairs',type=int,default=192)
    parser.add_argument('--test-pairs',type=int,default=256)
    parser.add_argument('--arms',nargs='+',default=['matched_clean','correction'])
    parser.add_argument('--model-seeds',type=int,nargs='+',default=[0,1,2,3])
    args=parser.parse_args();os.chdir(ROOT)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(.32)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    args.output_dir=args.output_dir.resolve()
    atomic_json(args.output_dir/'plan.json',dict({**vars(args),'output_dir':str(args.output_dir)},gpu=0,created=time.time(),
        source_sha256={str(p.relative_to(ROOT)):sha_file(p) for p in [Path(__file__),ROOT/'src/cot_navigation/analysis/history_probe.py']}))
    for seed in args.model_seeds:
        for arm in args.arms:run(arm,seed,args)


if __name__=='__main__':main()

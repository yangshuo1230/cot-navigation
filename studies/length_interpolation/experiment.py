"""Controlled prompt-length interpolation with unchanged query/search targets.

Three continuations of each frozen 50k model: short(76), long(220), equal
endpoints(76/220). Hold out148 for interpolation,292 for extrapolation. Each
adds5000 updates with the SAME query order and original neighbor priorities.
No main framework files or original outputs are modified.
"""
from dataclasses import replace
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
import numpy as np
import torch
from cot_navigation.config import load_config
from cot_navigation.data.dataset import collate_examples
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.training.trainer import Trainer
from cot_navigation.evaluation.rollout import generate

BASE=Path('outputs/explore_v1/launches/explore_v1_s012/context/standard/train50000')
STUDY=Path('studies/length_interpolation')
OLDSTUDY=Path('studies/length_generalization')
sys.path.insert(0,str(OLDSTUDY))
from trace_audit import validate_trace,validate_execution

TOK=Tokenizer()
POOL=np.array([(u,v) for layer in range(3) for u in range(layer*4,layer*4+4)
               for v in range((layer+1)*4,(layer+1)*4+4)],dtype=int)


def add_disconnected(row,nodes,components,rng):
    """Preserve the relative order of ALL original edges and every target token.

    Extra components each have16fresh node names and24edges, with no cross edge.
    Return a NEW row; never mutate the cached training row. Nodes are token IDs.
    """
    if components==0:return row
    available=np.setdiff1d(np.arange(TOK.node_lo,len(TOK)),nodes)
    if 16*components>len(available):raise ValueError('insufficient distinct node names')
    extra_nodes=rng.choice(available,16*components,replace=False).reshape(components,16)
    additions=[]
    for mapping in extra_nodes:
        edges=POOL[rng.choice(len(POOL),24,replace=False)]
        additions.append(np.column_stack((np.full(24,TOK.id('<EDGE>')),mapping[edges[:,0]],mapping[edges[:,1]])))
    additions=np.concatenate(additions)
    additions=additions[rng.permutation(len(additions))]
    original=np.asarray(row.tokens[:row.prompt_len-4]).reshape(-1,3)
    total=len(original)+len(additions)
    slots=np.sort(rng.choice(total,len(original),replace=False))
    other=np.ones(total,dtype=bool);other[slots]=False
    triples=np.empty((total,3),dtype=int);triples[slots]=original;triples[other]=additions
    prompt=triples.ravel().tolist()+row.tokens[row.prompt_len-4:row.prompt_len]
    delta=len(prompt)-row.prompt_len
    return replace(row,tokens=prompt+row.tokens[row.prompt_len:],prompt_len=len(prompt),
        answer_index=row.answer_index+delta,trace_indices=[p+delta for p in row.trace_indices],
        meta={**row.meta,'padding_components':components,'padding_delta':delta})


class PaddingTrainer(Trainer):
    """Reuse original optimized trainer/loss/checkpoint/online-W&B machinery."""
    def __init__(self,cfg,mode,parent,resume=None):
        self.padding_mode=mode;self.padding_seen=0;self.padding_quota=[]
        super().__init__(cfg,resume=resume)
        if not resume:
            blob=torch.load(parent,map_location='cpu',weights_only=False)
            assert blob['step']==50000
            self.raw.load_state_dict(blob['model'])
            self.parent_run_id=blob.get('wandb_run_id')
            del blob

    def _init_samplers(self):
        super()._init_samplers()
        self.augmentation_rngs['padding']=np.random.default_rng(self.cfg.train.exposure_seed+12301)

    def _load_resume(self,path):
        super()._load_resume(path)
        blob=torch.load(path,map_location='cpu',weights_only=False)
        state=blob['padding_state']
        if state['mode']!=self.padding_mode:raise ValueError('padding mode resume mismatch')
        self.padding_seen=state['seen'];self.padding_quota=state['quota']

    def _payload(self):
        payload=super()._payload()
        payload['padding_state']={'mode':self.padding_mode,'seen':self.padding_seen,'quota':self.padding_quota}
        return payload

    def _training_task(self,index):
        row=super()._training_task(index)
        rng=self.augmentation_rngs['padding']
        if self.padding_mode=='endpoints':
            if not self.padding_quota:
                batch=self.cfg.train.effective_query_batch_size
                assert batch%2==0
                self.padding_quota=rng.permutation([0]*(batch//2)+[2]*(batch//2)).tolist()
            components=self.padding_quota.pop()
        else:components=0 if self.padding_mode=='short' else 2
        self.padding_seen+=1
        nodes=self.store.node_vocab_ids(index,self.tok)
        changed=add_disconnected(row,nodes,components,rng)
        self._check_length(changed)
        return changed


def evaluate(trainer,out):
    """All arms use the exact previous held-out conditions; add endpoint tests."""
    old=json.loads((OLDSTUDY/'results_n32/metadata.json').read_text())
    cases={k:old['inputs'][k] for k in ('id','id_disconnected_pad32','wide32_branch2','wide32_densityhalf',
                  'deep16_sameprompt','deep24','deep32','deep48','large64')}
    # Keep the OOD computation modest without using answers to select rows:
    # first16 of each pre-existing hop/label stratum, versus32 in original audit.
    for case,records in list(cases.items()):
        counts={};selected=[]
        for r in records:
            key=r['h'],r['label']
            if counts.get(key,0)<16:selected.append(r);counts[key]=counts.get(key,0)+1
        cases[case]=selected
    cases['train_length220']=cases['id']
    cases['extrap_length292']=cases['id']
    path=out/'predictions.jsonl'
    if path.exists():raise FileExistsError(f'evaluation already exists: {path}')
    model=trainer.raw.eval()
    for case,records in cases.items():
        start=time.time();correct=valid=0
        for offset in range(0,len(records),32):
            chunk=records[offset:offset+32];rows=[];graphs=[]
            for r in chunk:
                nodes=r['nodes'];adj=[[] for _ in nodes]
                for u,v in r['edges']:adj[u].append(v)
                from cot_navigation.protocols.render import context_prompt,render_task
                prompt=context_prompt(TOK,nodes,r['edges'],r['s'],r['t'])
                row=render_task(TOK,trainer.cfg.protocol.name,r['label'],prompt,adj,r['s'],r['t'],nodes)
                if case in ('train_length220','extrap_length292'):
                    components=2 if case=='train_length220' else 3
                    row=add_disconnected(row,nodes,components,np.random.default_rng(20261018+r['index']))
                triples=np.asarray(row.tokens[:row.prompt_len-4]).reshape(-1,3)
                tokenadj={int(v):[] for triple in triples for v in triple[1:]}
                tokenadj.setdefault(nodes[r['s']],[]);tokenadj.setdefault(nodes[r['t']],[])
                for _,u,v in triples:tokenadj[int(u)].append(int(v))
                rows.append(row);graphs.append(tokenadj)
            gens=generate(model,[r.tokens[:r.prompt_len] for r in rows],trainer.cfg.protocol.name,TOK,'bf16',max_new_tokens=0)
            with path.open('a') as handle:
                for r,row,adj,g in zip(chunk,rows,graphs,gens):
                    s,t=r['nodes'][r['s']],r['nodes'][r['t']]
                    v=validate_trace(g['trace_tokens'],g['pred'],trainer.cfg.protocol.name,adj,s,t,TOK)
                    skeleton=validate_execution(g['trace_tokens'],g['pred'],adj,s,t,TOK,trainer.cfg.protocol.name=='bfs_state')
                    item={'case':case,'index':r['index'],'hop':r['h'],'label':r['label'],'pred':g['pred'],
                          'correct':g['pred']==r['label'],'trace_valid':v['valid'],'trace_error':v['error'],
                          'execution_valid':skeleton,'format_fail':g['format_fail'],'limit':g['generation_limit_reached'],
                          'trace_tokens':g['trace_tokens'],'prompt_len':row.prompt_len,
                          'gold_len':row.answer_index-row.prompt_len,'valid_records':v.get('valid_records')}
                    assert not v['valid'] or skeleton
                    handle.write(json.dumps(item)+'\n');correct+=item['correct'];valid+=item['trace_valid']
            print('EVAL',trainer.cfg.protocol.name,trainer.padding_mode,case,min(offset+32,len(records)),
                  '/',len(records),'correct',correct,'valid',valid,flush=True)
        print('EVAL DONE',case,time.time()-start,flush=True)
        if trainer.tracker.run:
            trainer.tracker.run.log({'optimizer_step':trainer.completed,
                f'length_eval/{case}/answer_accuracy':correct/len(records),
                f'length_eval/{case}/trace_valid':valid/len(records)})
    model.train()


def preflight():
    """No training, W&B, checkpoints or persisted smoke outputs."""
    from cot_navigation.data.dataset import ContextStore
    from cot_navigation.data.reachability import bfs_reaches
    store=ContextStore('data/explore_v1/context_standard_n16_m24_train50000_val10000_test10000_s0')
    count=0
    for protocol in ('dfs_state','bfs_state'):
        for i in (0,20,8334,17000,26000,40000):
            row=store.render(i,protocol,TOK);nodes=store.node_vocab_ids(i,TOK)
            for n in (0,1,2,3):
                changed=add_disconnected(row,nodes,n,np.random.default_rng(97))
                assert changed.tokens[changed.prompt_len:]==row.tokens[row.prompt_len:]
                assert changed.prompt_len==76+72*n
                assert changed.answer_index-row.answer_index==72*n
                assert changed.trace_indices==[p+72*n for p in row.trace_indices]
                triples=np.asarray(changed.tokens[:changed.prompt_len-4]).reshape(-1,3)
                adj={int(v):[] for triple in triples for v in triple[1:]}
                s,t=map(int,changed.tokens[changed.prompt_len-3:changed.prompt_len-1])
                adj.setdefault(s,[]);adj.setdefault(t,[])
                for _,u,v in triples:adj[int(u)].append(int(v))
                verdict=validate_trace(changed.tokens[changed.prompt_len:changed.answer_index],changed.label,protocol,adj,s,t,TOK)
                assert verdict['valid']
                data=collate_examples([changed],TOK.pad,1)
                assert int(np.flatnonzero(data['answer_mask'][0])[0])==changed.answer_index
                # Every added node is outside the original query component.
                old=set(nodes)
                assert all((int(u) in old)==(int(v) in old) for _,u,v in triples)
                count+=1
    print('PREFLIGHT PASSED',count,'rows; no smoke outputs persisted')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',choices=('bfs_state','dfs_state'),default='bfs_state')
    parser.add_argument('--mode',choices=('short','long','endpoints'),default='endpoints')
    parser.add_argument('--steps',type=int,default=5000)
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--parent-only',action='store_true',help='frozen50k reference; no training or W&B')
    parser.add_argument('--resume')
    args=parser.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if args.check_only:preflight();return
    suffix='bfs_state/bfs_state' if args.protocol=='bfs_state' else 'baseline/dfs_state'
    if args.parent_only:
        from types import SimpleNamespace
        from cot_navigation.models.transformer import build_model
        cfg=load_config([BASE/suffix/'resolved_config.yaml'])
        model=build_model(cfg.model,len(TOK),TOK.pad).cuda().eval()
        blob=torch.load(BASE/suffix/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(blob['model']);del blob
        out=STUDY/'runs'/args.protocol/'parent';out.mkdir(parents=True,exist_ok=True)
        shim=SimpleNamespace(raw=model,cfg=cfg,padding_mode='parent',tracker=SimpleNamespace(run=None))
        evaluate(shim,out)
        (out/'completed.json').write_text(json.dumps({'step':50000,'additional_steps':0}))
        return
    original=BASE/suffix;out=STUDY/'runs'/args.protocol/args.mode
    cfg=load_config([original/'resolved_config.yaml'],[
        f'output_dir={out}',f'train.max_steps={args.steps}',
        'train.lr=0.0001','train.min_lr=0.0001','train.warmup_steps=100',
        'train.eval_interval=0','train.save_interval=1000','train.keep_last=1','train.log_interval=100',
        'evaluation.final_checkpoints=[]','model.compile=true',
        'tracking.mode=online','tracking.group=length_interp_v1_20261017',
        f'tracking.name=context_{args.protocol}_{args.mode}_plus{args.steps}',
        f'tracking.variant=length_interp_{args.mode}'])
    spec={'protocol':args.protocol,'mode':args.mode,'additional_steps':args.steps,
          'parent_checkpoint':str(original/'checkpoints/latest.pt'),
          'parent_sha256':hashlib.sha256((original/'checkpoints/latest.pt').read_bytes()).hexdigest(),
          'train_components':[0] if args.mode=='short' else [2] if args.mode=='long' else [0,2],
          'train_prompt_lengths':[76] if args.mode=='short' else [220] if args.mode=='long' else [76,220],
          'heldout_interpolation_prompt_length':148,'heldout_extrapolation_prompt_length':292,
          'same_query_exposure':True,'same_original_neighbor_order':True,
          'fresh_optimizer':True,'lr':0.0001,'seed':cfg.train.model_seed}
    cfg.tracking.notes=json.dumps(spec)
    out.mkdir(parents=True,exist_ok=True)
    specpath=out/'experiment.json'
    if specpath.exists() and json.loads(specpath.read_text())!=spec:raise ValueError('experiment mismatch')
    if (out/'checkpoints/latest.pt').exists() and not args.resume:raise FileExistsError('use --resume')
    specpath.write_text(json.dumps(spec,indent=2))
    trainer=PaddingTrainer(cfg,args.mode,original/'checkpoints/latest.pt',resume=args.resume)
    start=time.time()
    # Original fit finishes its W&B tracker. Keep it open until length metrics
    # have also uploaded; no changes to Tracker or the original training loop.
    finish=trainer.tracker.finish;trainer.tracker.finish=lambda:None
    trainer.fit()
    trainer.model=trainer.raw
    print('TRAIN DONE',args.protocol,args.mode,'seconds',time.time()-start,flush=True)
    evaluate(trainer,out)
    finish()
    (out/'completed.json').write_text(json.dumps({'step':trainer.completed,'seconds':time.time()-start}))
    print('ALL DONE',args.protocol,args.mode,flush=True)


if __name__=='__main__':main()

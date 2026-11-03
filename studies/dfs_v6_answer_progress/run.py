"""Answer decodability across CoT progress; frozen v6 final checkpoints.

Inference: physical GPU1. All affine heads: physical GPU0. No checkpoints,
activations or W&B runs. Reference prefixes train probes; natural prefixes test
the SAME heads. A separate active-only fit removes terminal/revealed states.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
os.environ['WANDB_MODE']='disabled'
import sys,json,hashlib,importlib.util
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
V6=ROOT/'studies/fixed_navigation_protocols_v6'
sys.path[:0]=[str(V6),str(ROOT/'src')]
spec=importlib.util.spec_from_file_location('v6_state_probe',ROOT/'studies/dfs_v6_state_probes/run.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
from packed_bank import PackedBank
from semantics import replay_nodes
from cot_navigation.analysis.common import AnalysisModel
from cot_navigation.analysis.probes import fit,shuffled_labels
from cot_navigation.analysis.set_metrics import scores
from cot_navigation.models.transformer import TransformerLM
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.evaluation.rollout import generate

OUT=Path(__file__).resolve().parent
HEADS=['L1','L2','L3','L4','L5','L6','current_embedding','query_embeddings','observable_control']
PHASES=[('r0',0.),('r25',.25),('r50',.5),('r75',.75),('r100',1.)]+[(f't{k}',k) for k in [1,2,4,8,16,32,64]]
PANELS=previous.PANELS
SEED=20261027


def locations(length):
    result=[]
    for name,value in PHASES:
        if name[0]=='r':
            # Number of generated CoT tokens already consumed, never the answer.
            k=int(np.floor(value*length))
        else:
            k=int(value)
            if k>length:continue
        result.append((name,k))
    return result


def prefix_meta(qid,record,body,k,tok,inverse,adj):
    prefix=body[:k]
    assert tok.yes not in prefix and tok.no not in prefix
    if not prefix:
        active=True;valid=True;terminal=False
    else:
        check=replay_nodes(adj,record['s'],record['t'],prefix,inverse)
        terminal=check['valid']
        active=check['error']=='premature_stop'
        valid=terminal or active
    return dict(query=qid,label=record['y'],consumed=k,length=len(body),
                goal_seen=tok.node_id(int(WORLD['node_names'][record['t']])) in prefix,
                active=active,valid=valid,terminal=terminal)


def collect(model,tok,bank,ids,free=False):
    """Extract only selected positions, keeping tensors in memory transiently."""
    world=bank.world;nodes=[tok.node_id(v) for v in world['node_names']];inverse={t:i for i,t in enumerate(nodes)}
    sequences=[];native={};compiled=None
    if free:
        compiled=torch.compile(model,dynamic=True)
        for at in range(0,len(ids),128):
            qids=ids[at:at+128]
            prompts=[[tok.id('<QUERY>'),nodes[world['records'][q]['s']],nodes[world['records'][q]['t']],tok.id('|')] for q in qids]
            outputs=generate(compiled,prompts,'dfs_nodes',tok,'bf16',max_new_tokens=bank.info['max_new_tokens'])
            for q,prompt,out in zip(qids,prompts,outputs):
                r=world['records'][q];body=out['trace_tokens']
                assert tok.yes not in body and tok.no not in body
                sequences.append((q,prompt,body))
                native[q]=dict(label=r['y'],correct=out['answer_token']==tok.label_id(r['y']),
                               truncated=out['generation_limit_reached'])
    else:
        for q in ids:
            for row in range(int(bank.qptr[q]),int(bank.qptr[q+1])):
                a=int(bank.cpu['offsets'][row]);b=int(bank.cpu['offsets'][row+1]);answer=int(bank.cpu['answers'][row])
                tokens=bank.cpu['tokens'][a:b].astype(int).tolist()
                assert tokens[answer]==tok.label_id(world['records'][q]['y'])
                sequences.append((q,tokens[:4],tokens[4:answer]))
    storage={p:dict(x=[],meta=[]) for p,_ in PHASES}
    class Adapter:
        forward=torch.inference_mode()(AnalysisModel.forward)
    am=Adapter();am.model=model;am.tok=tok;am.device=torch.device('cuda:1')
    for start in range(0,len(sequences),24):
        chunk=sequences[start:start+24]
        _,residual,_=am.forward([prompt+body for _,prompt,body in chunk])
        for batch,(q,prompt,body) in enumerate(chunk):
            r=world['records'][q];query=np.concatenate([model.embed.weight[nodes[u]].detach().float().cpu().numpy() for u in [r['s'],r['t']]])
            for phase,k in locations(len(body)):
                pos=len(prompt)+k-1
                mm=prefix_meta(q,r,body,k,tok,inverse,world['adj'])
                feature=np.zeros((len(HEADS),256),np.float32)
                for layer in range(6):feature[layer,:128]=residual[layer][batch,pos].float().cpu().numpy()
                current=(prompt+body)[pos]
                feature[6,:128]=model.embed.weight[current].detach().float().cpu().numpy()
                feature[7]=query
                feature[8,:4]=[k/256.,float(mm['goal_seen']),float(current==nodes[r['t']]),float(current in inverse)]
                storage[phase]['x'].append(feature);storage[phase]['meta'].append(mm)
        del residual
    for phase,s in storage.items():
        s['x']=np.asarray(s['x'],np.float32)
        s['query']=np.asarray([m['query'] for m in s['meta']],np.int64)
        s['y']=np.asarray([[m['label']] for m in s['meta']],np.float32)
        s['mask']=np.ones_like(s['y'])
    return storage,native


def subset(data,gate):
    choose=np.asarray([gate(m) for m in data['meta']],bool)
    return {k:v[choose] for k,v in data.items() if k in ['x','query','y','mask']}


def weights(query):
    keys,counts=np.unique(query,return_counts=True);freq=dict(zip(keys,counts))
    return np.asarray([1/freq[q] for q in query])


def thresholds(z,data):
    y=data['y'][:,0].astype(bool);w=weights(data['query']);out=[]
    for h in range(z.shape[1]):
        zz=z[:,h,0]
        if not y.any() or y.all():out.append(0.);continue
        unique=np.unique(zz);candidates=np.r_[unique[0]-1.,(unique[1:]+unique[:-1])/2.,unique[-1]+1.,0.]
        pred=zz[:,None]>=candidates[None,:]
        ba=.5*((pred*(w*y)[:,None]).sum(0)/(w*y).sum()+((~pred)*(w*~y)[:,None]).sum(0)/(w*~y).sum())
        best=np.flatnonzero(np.abs(ba-ba.max())<1e-12)
        out.append(float(candidates[best[np.argmin(np.abs(candidates[best]))]]))
    return out


def metrics(z,data,cutoffs):
    if not len(data['query']):return dict(n_rows=0,n_queries=0,heads=[])
    y=data['y'][:,0].astype(bool);w=weights(data['query']);out=[]
    yes=float((w*y).sum());no=float((w*~y).sum())
    for h,threshold in enumerate(cutoffs):
        pred=z[:,h,0]>=threshold
        yr=float((w*(pred&y)).sum()/yes) if yes else None
        nr=float((w*((~pred)&~y)).sum()/no) if no else None
        out.append(dict(accuracy=float((w*(pred==y)).sum()/w.sum()),
                        balanced_accuracy=.5*(yr+nr) if yr is not None and nr is not None else None,
                        yes_recall=yr,no_recall=nr))
    return dict(n_rows=len(y),n_queries=len(set(data['query'].tolist())),n_positive_queries=int(round(yes)),
                n_negative_queries=int(round(no)),heads=out)


def evaluate(fitted,data,cutoffs,shuffled=False):
    result={}
    for name,gate in [('all',lambda m:True),('no_target_yet',lambda m:not m['goal_seen']),
                      ('active_legal',lambda m:m['active'])]:
        b=subset(data,gate)
        if not len(b['query']):result[name]=dict(n_rows=0,n_queries=0,heads=[]);continue
        if shuffled:b['y']=shuffled_labels(b,SEED+219)
        result[name]=metrics(scores(fitted,b),b,cutoffs)
    return result


def main():
    global WORLD
    torch.set_num_threads(3);tok=Tokenizer(256)
    manifest=json.loads((V6/'manifest.json').read_text());meta=next(m for m in manifest['worlds'] if m['name']=='n80_g0')
    bank=PackedBank(meta,'dfs_nodes',torch.device('cpu'));WORLD=bank.world
    state=json.loads((ROOT/'studies/dfs_v6_state_probes/results.json').read_text())
    ids={'train':state['train_ids'],'val':state['val_ids'],**state['test_ids']}
    doc={'heads':HEADS,'phases':PHASES,'query_ids':ids,'probe_steps':250,
         'metric':'Query-macro confusion rates; BA averages yes/no recall; validation-only thresholds.',
         'families':'all and active-only phase-specific affine heads fit on reference train/val, same heads on free test.',
         'positions':'After consumed CoT tokens; prompt end at0; never include YES/NO/EOS.',
         'models':{}}
    path=OUT/'results.json'
    if path.exists():doc=json.loads(path.read_text())
    for seed in range(4):
        name=f'seed{seed}'
        if name in doc['models']:continue
        run=ROOT/f'outputs/fixed_navigation_protocols_v6/pilot/n80/g0_s{seed}/dfs_nodes'
        blob=torch.load(run/'latest.pt',map_location='cpu',weights_only=False);assert blob['step']==50000
        model=TransformerLM(len(tok),**blob['job']['model'],dropout=0.).to('cuda:1');model.load_state_dict(blob['model']);model.eval()
        for p in model.parameters():p.requires_grad_(False)
        reference={};free={};native={}
        for panel,qids in ids.items():
            print('REFERENCE',seed,panel,len(qids),flush=True)
            reference[panel],_=collect(model,tok,bank,qids)
            if panel in PANELS:
                print('FREE',seed,panel,flush=True)
                free[panel],native[panel]=collect(model,tok,bank,qids,True)
        result={'checkpoint_sha256':hashlib.sha256((run/'latest.pt').read_bytes()).hexdigest(),'phases':{},'native_final':{}}
        for panel,items in native.items():
            yes=[v['correct'] for v in items.values() if v['label']];no=[v['correct'] for v in items.values() if not v['label']]
            result['native_final'][panel]=dict(accuracy=float(np.mean([v['correct'] for v in items.values()])),
                    balanced_accuracy=.5*(float(np.mean(yes))+float(np.mean(no))),n=len(items))
        for phase,_ in PHASES:
            result['phases'][phase]={}
            for family,gate in [('all',lambda m:True),('active',lambda m:m['active'])]:
                tr=subset(reference['train'][phase],gate);va=subset(reference['val'][phase],gate)
                if len(tr['query'])<2 or len(va['query'])<2 or len(np.unique(tr['y']))<2 or len(np.unique(va['y']))<2:
                    result['phases'][phase][family]={'status':'insufficient active samples or classes'};continue
                print('FIT',seed,phase,family,len(tr['query']),len(va['query']),flush=True)
                fitted=fit(tr,va,device='cuda:0',seed=SEED,steps=250)
                cutoffs=thresholds(scores(fitted,va),va)
                item={'selected_steps':fitted['steps'],'thresholds':cutoffs,'reference':{},'free':{}}
                for panel in PANELS:
                    item['reference'][panel]=evaluate(fitted,reference[panel][phase],cutoffs)
                    item['free'][panel]=evaluate(fitted,free[panel][phase],cutoffs)
                del fitted
                random_head=fit(tr,va,device='cuda:0',seed=SEED+17,steps=250,shuffled=True)
                vv=dict(va,y=shuffled_labels(va,SEED+118))
                random_cut=thresholds(scores(random_head,vv),vv)
                item['shuffled_reference']={panel:evaluate(random_head,reference[panel][phase],random_cut,True) for panel in PANELS}
                result['phases'][phase][family]=item
                del random_head
        doc['models'][name]=result
        path.write_text(json.dumps(doc,ensure_ascii=False,indent=2))
        del model,blob,reference,free,native
        torch.cuda.empty_cache()
    print('DONE',flush=True)


if __name__=='__main__':main()

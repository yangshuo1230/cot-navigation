"""All-query probe split, percentage progress only; no navigation training.

Reuses prior phase-specific answer probes. Query identities are freshly split
60/20/20 within role/gap/label strata, including the original V/T queries.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
os.environ['WANDB_MODE']='disabled'
import sys,json,random,hashlib,importlib.util
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('old_answer_progress',ROOT/'studies/dfs_v6_answer_progress/run.py')
o=importlib.util.module_from_spec(spec);spec.loader.exec_module(o)
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
OUT=Path(__file__).resolve().parent
PHASES=[(f'r{p}',p/100) for p in range(0,101,10)]
o.PHASES=PHASES
SEED=20261027


def make_split(world):
    bucket={}
    for name,ids in world['panels'].items():
        role='short' if name.startswith('short_') else 'AA' if name=='long_train' else name.split('_')[1]
        for q in ids:
            assert q not in bucket,'Original query panel overlap'
            bucket[q]=role
    cells={}
    for q,role in bucket.items():
        r=world['records'][q];cells.setdefault((role,r['gap'],r['y']),[]).append(q)
    rng=random.Random(SEED);splits={s:[] for s in ['train','val','test']}
    for key in sorted(cells):
        ids=sorted(cells[key]);rng.shuffle(ids);n=len(ids)
        if n>=3:nv=max(1,int(.2*n));nt=max(1,int(.2*n))
        else:nv=0;nt=int(n==2)
        ntr=n-nv-nt
        splits['train']+=ids[:ntr];splits['val']+=ids[ntr:ntr+nv];splits['test']+=ids[ntr+nv:]
    for ids in splits.values():ids.sort()
    assert set().union(*map(set,splits.values()))==set(bucket)
    assert not(set(splits['train'])&set(splits['test']) or set(splits['train'])&set(splits['val']) or set(splits['val'])&set(splits['test']))
    lm_train=set(world['panels']['short_train'])|set(world['panels']['long_train'])
    panels={'all':splits['test']}
    for role in ['short','AA','VA','AV','VV','TA','AT','TT']:
        panels[role]=[q for q in splits['test'] if bucket[q]==role]
    panels['original_query_seen']=[q for q in splits['test'] if q in lm_train]
    panels['original_query_unseen']=[q for q in splits['test'] if q not in lm_train]
    coverage={role:{split:sum(bucket[q]==role for q in ids) for split,ids in splits.items()}
              for role in sorted(set(bucket.values()))}
    return splits,panels,coverage


def fast_thresholds(z,data):
    """Exact validation BA scan with tied scores, O(n log n) rather than n²."""
    y=data['y'][:,0].astype(bool);w=o.weights(data['query']);out=[]
    for h in range(z.shape[1]):
        zz=z[:,h,0]
        if not y.any() or y.all():out.append(0.);continue
        order=np.argsort(zz,kind='stable');a=zz[order];yp=y[order];ww=w[order]
        cuts=np.r_[0,np.flatnonzero(a[1:]!=a[:-1])+1,len(a)]
        pos=np.r_[0,np.cumsum(ww*yp)];neg=np.r_[0,np.cumsum(ww*~yp)]
        ba=.5*((pos[-1]-pos[cuts])/pos[-1]+neg[cuts]/neg[-1])
        mid=np.asarray([(a[c-1]+a[c])/2 for c in cuts[1:-1]])
        threshold=np.r_[a[0]-1.,mid,a[-1]+1.]
        at_zero=.5*((w*(y&(zz>=0))).sum()/(w*y).sum()+(w*(~y&(zz<0))).sum()/(w*~y).sum())
        ba=np.r_[ba,at_zero];threshold=np.r_[threshold,0.]
        best=np.flatnonzero(np.abs(ba-ba.max())<1e-12)
        out.append(float(threshold[best[np.argmin(np.abs(threshold[best]))]]))
    return out


def collect(model,tok,bank,ids,free=False):
    """Same positions/features, with batched CPU transfers for efficiency."""
    world=bank.world;nodes=[tok.node_id(v) for v in world['node_names']];inverse={t:i for i,t in enumerate(nodes)}
    sequences=[];native={}
    if free:
        compiled=torch.compile(model,dynamic=True)
        for at in range(0,len(ids),128):
            qids=ids[at:at+128]
            prompts=[[tok.id('<QUERY>'),nodes[world['records'][q]['s']],nodes[world['records'][q]['t']],tok.id('|')] for q in qids]
            outputs=o.generate(compiled,prompts,'dfs_nodes',tok,'bf16',max_new_tokens=bank.info['max_new_tokens'])
            for q,prompt,out in zip(qids,prompts,outputs):
                r=world['records'][q];body=out['trace_tokens']
                assert tok.yes not in body and tok.no not in body
                sequences.append((q,prompt,body))
                native[q]=dict(label=r['y'],correct=out['answer_token']==tok.label_id(r['y']),truncated=out['generation_limit_reached'])
    else:
        for q in ids:
            for row in range(int(bank.qptr[q]),int(bank.qptr[q+1])):
                a=int(bank.cpu['offsets'][row]);b=int(bank.cpu['offsets'][row+1]);ans=int(bank.cpu['answers'][row])
                tokens=bank.cpu['tokens'][a:b].astype(int).tolist()
                assert tokens[ans]==tok.label_id(world['records'][q]['y'])
                sequences.append((q,tokens[:4],tokens[4:ans]))
    storage={p:dict(x=[],meta=[]) for p,_ in PHASES}
    class Adapter:
        forward=torch.inference_mode()(o.AnalysisModel.forward)
    am=Adapter();am.model=model;am.tok=tok;am.device=torch.device('cuda:1')
    embedding=model.embed.weight.detach().float().cpu().numpy()
    for start in range(0,len(sequences),48):
        chunk=sequences[start:start+48]
        _,residual,_=am.forward([prompt+body for _,prompt,body in chunk])
        cpu=[residual[layer].float().cpu().numpy() for layer in range(6)]
        for batch,(q,prompt,body) in enumerate(chunk):
            r=world['records'][q];query=np.concatenate([embedding[nodes[u]] for u in [r['s'],r['t']]])
            for phase,k in o.locations(len(body)):
                pos=len(prompt)+k-1;mm=o.prefix_meta(q,r,body,k,tok,inverse,world['adj'])
                x=np.zeros((len(o.HEADS),256),np.float32)
                for layer in range(6):x[layer,:128]=cpu[layer][batch,pos]
                current=(prompt+body)[pos];x[6,:128]=embedding[current];x[7]=query
                x[8,:4]=[k/256.,float(mm['goal_seen']),float(current==nodes[r['t']]),float(current in inverse)]
                storage[phase]['x'].append(x);storage[phase]['meta'].append(mm)
        del cpu,residual
    for s in storage.values():
        s['x']=np.asarray(s['x'],np.float32);s['query']=np.asarray([m['query'] for m in s['meta']],np.int64)
        s['y']=np.asarray([[m['label']] for m in s['meta']],np.float32);s['mask']=np.ones_like(s['y'])
    return storage,native


def take(data,ids):
    ids=set(ids);choose=np.asarray([q in ids for q in data['query']],bool)
    return {k:([m for m,c in zip(v,choose) if c] if k=='meta' else v[choose]) for k,v in data.items()}


def main():
    torch.set_num_threads(3);tok=o.Tokenizer(256)
    manifest=json.loads((o.V6/'manifest.json').read_text());meta=next(m for m in manifest['worlds'] if m['name']=='n80_g0')
    bank=o.PackedBank(meta,'dfs_nodes',torch.device('cpu'));world=bank.world;o.WORLD=world
    splits,panels,coverage=make_split(world)
    print('SPLIT',{k:len(v) for k,v in splits.items()},coverage,flush=True)
    doc={'heads':o.HEADS,'phases':PHASES,'probe_steps':250,'query_ids':splits,'test_panels':panels,'coverage':coverage,
         'split':'Fresh role/gap/label-stratified ~60/20/20 probe split over ALL original query panels, query variants grouped.',
         'note':'External probe split; some probe-test queries were seen by the navigation model. Original seen/unseen reported separately.',
         'models':{}}
    path=OUT/'results.json'
    if path.exists():doc=json.loads(path.read_text())
    for seed in range(4):
        name=f'seed{seed}'
        if name in doc['models']:continue
        run=ROOT/f'outputs/fixed_navigation_protocols_v6/pilot/n80/g0_s{seed}/dfs_nodes'
        blob=torch.load(run/'latest.pt',map_location='cpu',weights_only=False);assert blob['step']==50000
        model=o.TransformerLM(len(tok),**blob['job']['model'],dropout=0.).to('cuda:1');model.load_state_dict(blob['model']);model.eval()
        for p in model.parameters():p.requires_grad_(False)
        reference={}
        for part,qids in splits.items():
            print('REFERENCE',seed,part,len(qids),flush=True);reference[part],_=collect(model,tok,bank,qids)
        print('FREE',seed,flush=True);free,native=collect(model,tok,bank,splits['test'],True)
        result={'checkpoint_sha256':hashlib.sha256((run/'latest.pt').read_bytes()).hexdigest(),'phases':{},'native_final':{}}
        for panel,qids in panels.items():
            items=[native[q] for q in qids];yes=[v['correct'] for v in items if v['label']];no=[v['correct'] for v in items if not v['label']]
            result['native_final'][panel]=dict(n=len(items),accuracy=float(np.mean([v['correct'] for v in items])) if items else None,
                balanced_accuracy=.5*(float(np.mean(yes))+float(np.mean(no))) if yes and no else None)
        for phase,_ in PHASES:
            print('PHASE',seed,phase,flush=True);result['phases'][phase]={}
            testref={panel:take(reference['test'][phase],qids) for panel,qids in panels.items()}
            testfree={panel:take(free[phase],qids) for panel,qids in panels.items()}
            for family,gate in [('all',lambda m:True),('active',lambda m:m['active'])]:
                tr=o.subset(reference['train'][phase],gate);va=o.subset(reference['val'][phase],gate)
                if len(tr['query'])<2 or len(va['query'])<2 or len(np.unique(tr['y']))<2 or len(np.unique(va['y']))<2:
                    result['phases'][phase][family]={'status':'insufficient active samples/classes'};continue
                # Before the reference endpoint all frozen prefixes are active.
                # Reuse the identical fit; evaluation already includes all gates.
                if (family=='active' and len(tr['query'])==len(reference['train'][phase]['query'])
                        and len(va['query'])==len(reference['val'][phase]['query'])):
                    result['phases'][phase][family]=dict(result['phases'][phase]['all'],
                        fit_note='Same train/val rows as all; reused identical fitted head.')
                    continue
                fitted=o.fit(tr,va,device='cuda:0',seed=SEED,steps=250)
                cut=fast_thresholds(o.scores(fitted,va),va)
                item={'selected_steps':fitted['steps'],'thresholds':cut,'reference':{},'free':{}}
                for panel in panels:
                    item['reference'][panel]=o.evaluate(fitted,testref[panel],cut)
                    item['free'][panel]=o.evaluate(fitted,testfree[panel],cut)
                del fitted
                shuffled=o.fit(tr,va,device='cuda:0',seed=SEED+17,steps=250,shuffled=True)
                vv=dict(va,y=o.shuffled_labels(va,SEED+118));scut=fast_thresholds(o.scores(shuffled,vv),vv)
                item['shuffled_reference']={panel:o.evaluate(shuffled,testref[panel],scut,True) for panel in panels}
                result['phases'][phase][family]=item
                del shuffled
            del testref,testfree
        doc['models'][name]=result;path.write_text(json.dumps(doc,ensure_ascii=False,indent=2))
        del model,blob,reference,free,native
        torch.cuda.empty_cache()
    print('DONE',flush=True)


if __name__=='__main__':main()

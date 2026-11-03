"""First-candidate-error diagnostics, strictly before the erroneous token.

Reuse cached natural trajectories and early-record linear-probe training.
Matched correct prefixes are held-out, from other queries. No causal claims.
"""
import argparse, gzip, importlib.util, json, os, sys
from collections import Counter,defaultdict
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
sys.path.insert(0,str(ROOT/'src'));os.environ['WANDB_MODE']='disabled'
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(k,'2')
BASE=ROOT/'studies/correction_state_localization'
SOURCE=ROOT/'outputs/analysis/correction_state_localization_20261102'
OUT=ROOT/'outputs/analysis/correction_first_error_20261102'
spec=importlib.util.spec_from_file_location('localization',BASE/'run.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
from cot_navigation.experiments.scheduler import write_json,file_lock


def recipes(cases,store,tok,split):
    import numpy as np
    from cot_navigation.analysis.within_token import candidate_slots
    out=[];rng=np.random.default_rng(20261102)
    for sid,case in enumerate(cases):
        q=case['query'];nodes=store.node_vocab_ids(q,tok);adj={nodes[u]:[nodes[v] for v in vs] for u,vs in enumerate(store.adj(q))}
        c={**case,'tau':case['first']['token']-1 if case['first'] else None}
        bank=defaultdict(list)
        for slot in candidate_slots(c,adj,tok,include_closure=True):
            pos=slot['pos'];rec=1+case['trace'][:pos].count(tok.id(';'))
            if rec>16 or (split!='test' and rec>4):break
            # All targets are defined from the consumed, still-correct prefix.
            prefix=case['trace'][:pos];start=len(prefix)-1-prefix[::-1].index(tok.id(';'))+1 if tok.id(';') in prefix else 0
            groups=[[]]
            for token in prefix[start:]:
                if token==tok.id('|'):groups.append([])
                else:groups[-1].append(token)
            stack,vis,printed=groups[:3];succ=adj[stack[-1]];eligible=set(succ)-set(vis)
            row=dict(query=q,sid=sid,record=rec,pos=pos,field=2,location='candidate_token',
                target=case['trace'][pos],allowed=slot['allowed'],requires_node=slot['requires_node'],
                depth=len(stack),visited_size=len(vis),printed=len(printed),
                prev=case['trace'][pos-1] if pos else case['prompt'][-1],
                stack=stack,visited=vis,successors=succ,eligible=list(eligible),nodes=nodes,
                is_onset=bool(case['first'] and pos==c['tau']),kind=case['first']['kind'] if case['first'] else None)
            bank[rec].append(row)
        for rec,rr in bank.items():
            if split=='test':out.extend(rr)
            elif not case['first'] or rec<case['first']['record']:
                # Cap per-record positions, not per-label: keep entry/closure and a seeded middle.
                indices=sorted(set([0,len(rr)-1,int(rng.integers(len(rr)))]));out.extend(rr[i] for i in indices)
    return out


def matched(rows,cases):
    errors=[r for r in rows if r['is_onset']]
    healthy=[r for r in rows if not cases[r['sid']]['first'] or r['record']<cases[r['sid']]['first']['record']]
    pairs=[];unused=set(range(len(healthy)))
    for onset in errors:
        for role in ['onset','previous_record_1','previous_record_2']:
            event=onset
            if role!='onset':
                gap=int(role[-1]);rr=[r for r in rows if r['sid']==onset['sid'] and r['record']==onset['record']-gap]
                if not rr:continue
                event=min(rr,key=lambda r:r['pos'])
            available=[i for i in unused if healthy[i]['query']!=event['query'] and healthy[i]['requires_node']==event['requires_node']]
            if not available:continue
            def cost(i):
                c=healthy[i]
                return (abs(c['record']-event['record'])+abs(c['depth']-event['depth'])+
                        .5*abs(c['visited_size']-event['visited_size'])+abs(c['printed']-event['printed'])+
                        .5*abs(len(c['allowed'])-len(event['allowed']))+.25*(c['prev']!=event['prev']))
            best=min(available,key=lambda i:(cost(i),healthy[i]['query'],healthy[i]['pos']))
            # Do not disguise distant controls as matched. Record every excluded case.
            if cost(best)>4:continue
            control=healthy[best];unused.remove(best)
            pairs.append(dict(role=role,error=event,control=control,cost=cost(best),
                first_error_kind=onset['kind'],category=('premature_close' if role=='onset' and onset['kind']=='candidate_omission' else 'node_required') if event['requires_node'] else 'close_list',
                same_prev_token=event['prev']==control['prev']))
    return pairs


def extract(model,tok,cases,rows):
    import numpy as np
    import torch
    from cot_navigation.evaluation.rollout import pad_rows
    keys=['stack','visited','successors','eligible'];x=[];y={k:[] for k in keys};mask=[];logits=[];metadata=[]
    groups=defaultdict(list)
    for r in rows:groups[r['sid']].append(r)
    ids=list(groups)
    for at in range(0,len(ids),8):
        chunk=ids[at:at+8];seq=[]
        for sid in chunk:
            c=cases[sid];end=max(r['pos'] for r in groups[sid]);seq.append(c['prompt']+c['trace'][:end])
        tokens,lens=pad_rows(seq,tok.pad,'cuda:0');cap={}
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):z=model(tokens,capture=cap)
        for j,sid in enumerate(chunk):
            c=cases[sid]
            for r in groups[sid]:
                pos=len(c['prompt'])+r['pos']-1
                h=torch.stack([v[j,pos].float() for v in cap['resid']]).cpu().numpy()
                h=np.pad(h,((0,0),(0,4)))
                emb=model.embed(tokens[j,pos]).detach().float().cpu().numpy()
                control=np.r_[emb,pos/4096,(pos-len(c['prompt']))/4096,0,0]
                x.append(np.r_[h,control[None]]);logits.append(z[j,pos].float().cpu().numpy())
                for k in keys:y[k].append([int(tok.node_id(v) in r[k]) for v in range(256)])
                mask.append([int(tok.node_id(v) in r['nodes']) for v in range(256)]);metadata.append(r)
        if at%128==0:print('FEATURES',at,len(ids),flush=True)
    return dict(x=np.array(x,np.float32),y={k:np.array(v,np.float32) for k,v in y.items()},mask=np.array(mask,np.float32),logits=np.array(logits),meta=metadata)


def run():
    import numpy as np
    import torch
    from cot_navigation.config import load_config
    from cot_navigation.data.dataset import ContextStore
    from cot_navigation.models.transformer import build_model
    from cot_navigation.protocols.tokenizer import Tokenizer
    from cot_navigation.training.checkpoint import load_checkpoint
    from cot_navigation.analysis.within_token import fit_lens,lens_logits
    from cot_navigation.data.shards import file_sha256
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.2)
    cfg=load_config([old.MODELS/'l6_p60/s0/resolved_config.yaml']);store=ContextStore(cfg.data.root)
    for depth in [6,9]:
        for seed in [0,1]:
            name=f'l{depth}_p60_s{seed}';dest=OUT/f'{name}.json'
            if dest.exists() and json.loads(dest.read_text()).get('complete'):continue
            folder=old.MODELS/f'l{depth}_p60'/f's{seed}';cfg=load_config([folder/'resolved_config.yaml']);tok=Tokenizer.from_config(cfg)
            ckpt=folder/'checkpoints/latest.pt';sha=file_sha256(ckpt);blob=load_checkpoint(ckpt,map_location='cpu');assert blob['step']==200000
            model=build_model(cfg.model,len(tok),pad_id=tok.pad).cuda().eval().requires_grad_(False);model.load_state_dict(blob['model']);del blob
            cases={};rows={}
            for split in ['train','val','test']:
                with gzip.open(SOURCE/'natural'/f'{name}_{split}.json.gz','rt') as f:d=json.load(f)
                assert d['checkpoint_sha256']==sha;cases[split]=d['rows'];rows[split]=recipes(cases[split],store,tok,split)
            queries={s:{r['query'] for r in rr} for s,rr in rows.items()}
            assert all(not queries[a]&queries[b] for a,b in [('train','val'),('train','test'),('val','test')])
            pairs=matched(rows['test'],cases['test'])
            assert all(p['error']['target'] not in p['error']['allowed'] for p in pairs if p['role']=='onset')
            assert all(p['control']['target'] in p['control']['allowed'] for p in pairs)
            testrows=[]
            for pair_id,pair in enumerate(pairs):
                for side in ['error','control']:
                    testrows.append(dict(pair[side],pair=pair_id,side=side,role=pair['role'],category=pair['category']))
            features={s:extract(model,tok,cases[s],rr) for s,rr in rows.items() if s!='test'}
            features['test']=extract(model,tok,cases['test'],testrows)
            tq=set(sorted(queries['train'])[:512]);vq=set(sorted(queries['val'])[:128])
            lens_tr=old.sub(features['train'],[r['query'] in tq for r in features['train']['meta']]);lens_va=old.sub(features['val'],[r['query'] in vq for r in features['val']['meta']])
            lens=fit_lens(model,lens_tr['x'][:,:depth,:128],lens_tr['logits'],lens_va['x'][:,:depth,:128],lens_va['logits'],steps=2000)
            train=old.sub(features['train'],[r['query'] not in tq for r in features['train']['meta']]);val=old.sub(features['val'],[r['query'] not in vq for r in features['val']['meta']]);test=features['test']
            result=dict(complete=False,name=name,checkpoint_sha256=sha,source='unchanged cached natural rollouts',
                first_error_types=dict(Counter(c['first']['kind'] for c in cases['test'] if c['first'])),
                first_error_n=sum(c['first'] is not None for c in cases['test']),candidate_onsets=sum(r['is_onset'] for r in rows['test']),
                matching=[{k:v for k,v in p.items() if k not in ['error','control']} for p in pairs],
                training='natural correct complete records1..4; separate queries for lens/probes; test never selects checkpoints',lens=[],probes=[])
            for kind in ['raw','calibrated']:
                for layer in range(depth):
                    preds=[]
                    for at in range(0,len(test['x']),512):
                        h=torch.as_tensor(test['x'][at:at+512,layer,:128],device='cuda');preds+=lens_logits(model,lens,h,layer,kind).argmax(-1).cpu().tolist()
                    for role in ['onset','previous_record_1','previous_record_2']:
                        for category in ['node_required','premature_close','close_list']:
                            for side in ['error','control']:
                                ii=[i for i,r in enumerate(test['meta']) if r['role']==role and r['category']==category and r['side']==side]
                                if ii:result['lens'].append(dict(kind=kind,layer=layer+1,role=role,category=category,side=side,n=len(ii),legal_accuracy=float(np.mean([preds[i] in test['meta'][i]['allowed'] for i in ii])),
                                    observed_token_agreement=float(np.mean([preds[i]==test['meta'][i]['target'] for i in ii]))))
            for task in ['stack','visited','successors','eligible']:
                fitted=old.fit(train,val,task,steps=3000)
                for role in ['onset','previous_record_1','previous_record_2']:
                    for category in ['node_required','premature_close','close_list']:
                        for side in ['error','control']:
                            d=old.sub(test,[r['role']==role and r['category']==category and r['side']==side for r in test['meta']])
                            if len(d['meta']):result['probes'].append(dict(task=task,role=role,category=category,side=side,scores=old.evaluate(fitted,d,task)))
                write_json(dest,result);print('PROBE',name,task,flush=True)
            result['complete']=True;assert file_sha256(ckpt)==sha;write_json(dest,result)
            del model,features,lens,train,val,test,fitted;torch.cuda.empty_cache()
    write_json(OUT/'status.json',dict(complete=True,models=4))


def main():
    import subprocess
    p=argparse.ArgumentParser();p.add_argument('--background',action='store_true');a=p.parse_args();os.chdir(ROOT);OUT.mkdir(parents=True,exist_ok=True)
    if a.background:
        with (OUT/'run.log').open('a') as log:
            proc=subprocess.Popen([sys.executable,str(__file__)],cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','PYTHONUNBUFFERED':'1'},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write_json(OUT/'process.json',dict(pid=proc.pid));print('STARTED',proc.pid);return
    with file_lock(OUT/'.lock',blocking=False):
        write_json(OUT/'status.json',dict(complete=False,state='running'))
        try:run()
        except Exception as exc:write_json(OUT/'status.json',dict(complete=False,state='failed',error=str(exc)));raise


if __name__=='__main__':main()

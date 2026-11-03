"""All-layer search-variable probes on early, naturally correct DFS records.

Frozen l6/l9 p60 s0/s1. Shared generation, parsing, replay and lens calibration.
Only compact natural-prefix recipes and scalar scores are saved, never activations.
"""
import argparse, copy, gzip, json, os, sys, time
from pathlib import Path
P=Path(__file__).resolve().parent; ROOT=P.parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ['WANDB_MODE']='disabled'
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','TORCHINDUCTOR_COMPILE_THREADS']:
    os.environ.setdefault(k,'2')
from cot_navigation.experiments.scheduler import write_json,file_lock
from cot_navigation.data.shards import file_sha256
OUT=ROOT/'outputs/analysis/correction_state_localization_20261102'
MODELS=ROOT/'outputs/correction_depth_rate/launches/dfs_depth_rate_20261031/dfs_state'
FIELDS=['stack','visited','candidates','action']


def read(path):
    with gzip.open(path,'rt') as f:return json.load(f)


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    with gzip.open(tmp,'wt',compresslevel=3) as f:json.dump(value,f,separators=(',',':'))
    tmp.replace(path)


def generate_cases(model,tok,store,queries,split,name,checkpoint_sha):
    import torch
    from cot_navigation.models.compiled_inference import CompiledInference
    from cot_navigation.evaluation.rollout import generate
    from cot_navigation.protocols.render import context_prompt
    from cot_navigation.protocols.correction_replay import replay
    path=OUT/'natural'/f'{name}_{split}.json.gz'
    if path.exists():
        old=read(path);assert old['queries']==queries and old['checkpoint_sha256']==checkpoint_sha
        if old['complete']:return old['rows']
    else:old=dict(queries=queries,checkpoint_sha256=checkpoint_sha,complete=False,rows=[])
    compiled=CompiledInference(model)
    for at in range(len(old['rows']),len(queries),32):
        qs=queries[at:at+32]; graphs=[];prompts=[]
        for q in qs:
            nodes=store.node_vocab_ids(q,tok);a=store.adj(q)
            adj={nodes[u]:[nodes[v] for v in vs] for u,vs in enumerate(a)}
            s=nodes[int(store.arrays['s_topo'][q])];t=nodes[int(store.arrays['t_topo'][q])]
            edges=list(zip(store.arrays['edge_src'][q].tolist(),store.arrays['edge_dst'][q].tolist()))
            prompts.append(context_prompt(tok,nodes,edges,int(store.arrays['s_topo'][q]),int(store.arrays['t_topo'][q])))
            graphs.append((adj,s,t))
        counts=[0]*len(qs)
        def stop(i,trace):
            counts[i]+=1
            return counts[i]>=(16 if split=='test' else 4) or replay(trace,None,*graphs[i],tok)['first'] is not None or tok.id('<MISTAKE>') in trace
        with torch.no_grad():outs=generate(compiled,prompts,'dfs_state',tok,'bf16',cot_margin=8,record_stop=stop)
        for q,p,g,graph in zip(qs,prompts,outs,graphs):
            first=replay(g['trace_tokens'],g['pred'],*graph,tok)['first']
            old['rows'].append(dict(query=q,prompt=p,trace=g['trace_tokens'],first=first))
        save(path,old);print('NATURAL',name,split,len(old['rows']),flush=True)
    old['complete']=True;save(path,old)
    return old['rows']


def extract(model,tok,store,cases):
    import numpy as np
    import torch
    from cot_navigation.analysis.common import records
    from cot_navigation.evaluation.rollout import pad_rows
    xs=[];cs=[];meta=[];ys={k:[] for k in ['stack','visited','successors','eligible','top']+[f'slot{i}' for i in range(8)]};masks=[];final=[]
    for at in range(0,len(cases),8):
        chunk=cases[at:at+8];seq=[];banks=[]
        for case in chunk:
            prompt=case['prompt'];trace=case['trace'];q=case['query'];nodes=store.node_vocab_ids(q,tok)
            adj={nodes[u]:[nodes[v] for v in vs] for u,vs in enumerate(store.adj(q))}
            rr=records(prompt+trace,len(prompt),tok); anchors=[]
            for ri,r in enumerate(rr):
                if ri>=16 or (case['first'] and ri+1>=case['first']['record']):break
                if r['action']==[tok.id('<MISTAKE>')]:break
                a,b,c=r['separators']; stack=r['frontier'];vis=r['visited'];cur=stack[-1]
                succ=adj[cur];eligible=set(succ)-set(vis)
                assert len(vis)==len(set(vis)) and set(r['candidates'])==eligible
                starts=[r['start'],a+1,b+1,c+1];ends=[a,b,c,r['end']]
                for fi,(lo,hi) in enumerate(zip(starts,ends)):
                    for location,target in [('entry',lo),('end',hi)]:
                        # h[target-1] never consumes the next token under study.
                        anchors.append(dict(query=q,record=ri+1,field=fi,location=location,pos=target-1,
                            target=(prompt+trace)[target],depth=len(stack),stack=stack,visited=vis,
                            successors=succ,eligible=list(eligible),top=[cur],nodes=nodes,
                            **{f'slot{i}':[stack[-1-i]] if len(stack)>i else [] for i in range(8)}))
            end=max((r['pos']+1 for r in anchors),default=len(prompt));seq.append((prompt+trace)[:end]);banks.append(anchors)
        ids,lengths=pad_rows(seq,tok.pad,'cuda:0');cap={}
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):z=model(ids,capture=cap)
        for j,bank in enumerate(banks):
            for r in bank:
                pos=r['pos'];xs.append(torch.stack([h[j,pos].float() for h in cap['resid']]).cpu().numpy())
                emb=model.embed(ids[j,pos]).detach().float().cpu().numpy()
                control=np.r_[emb,pos/4096,(pos-len(chunk[j]['prompt']))/4096]
                cs.append(np.pad(control,(0,2)));masks.append([int(tok.node_id(i) in r['nodes']) for i in range(256)])
                for key in ys:ys[key].append([int(tok.node_id(i) in r[key]) for i in range(256)])
                final.append(z[j,pos].float().cpu().numpy())
                m={k:r[k] for k in ['query','record','field','location','target','depth']}
                if r['location']=='end':m['allowed']=[r['target']]
                elif r['field']==0:m['allowed']=[r['stack'][0]]
                elif r['field']==1:m['allowed']=r['visited']
                elif r['field']==2:m['allowed']=r['eligible'] or [tok.id('|')]
                else:m['allowed']=[r['target']]
                meta.append(m)
        if at%128==0:print('EXTRACT',at,len(cases),flush=True)
    # Pad activation with four zeros so the input+position baseline has same dim.
    x=np.pad(np.array(xs,np.float32),((0,0),(0,0),(0,4)))
    x=np.concatenate([x,np.array(cs,np.float32)[:,None]],1)
    return dict(x=x,y={k:np.array(v,np.float32) for k,v in ys.items()},mask=np.array(masks,np.float32),meta=meta,logits=np.array(final))


def fit(train,val,task,steps=3000):
    """Independent linear heads; class/threshold semantics fixed before test."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    torch.manual_seed(1709);rng=np.random.default_rng(1709)
    x=torch.as_tensor(train['x'],device='cuda');v=torch.as_tensor(val['x'],device='cuda')
    mean=x.mean(0);std=x.std(0).clamp_min(.05);x=(x-mean)/std;v=(v-mean)/std
    mode='set' if task not in ['field','record','depth'] else 'ce' if task=='field' else 'reg'
    def target(d):
        if mode=='set':return torch.as_tensor(d['y'][task],device='cuda'),torch.as_tensor(d['mask'],device='cuda')
        y=torch.tensor([r[task] for r in d['meta']],device='cuda')
        return y,None
    y,mask=target(train);vy,vm=target(val);heads=x.shape[1];outputs=256 if mode=='set' else 4 if mode=='ce' else 1
    w=torch.nn.Parameter(torch.zeros(heads,x.shape[2],outputs,device='cuda'));b=torch.nn.Parameter(torch.zeros(heads,outputs,device='cuda'))
    opt=torch.optim.AdamW([w,b],lr=.003,weight_decay=.01)
    if mode=='set':pw=((mask*(1-y)).sum()/((mask*y).sum().clamp_min(1))).detach()
    def loss(z,yy,mm):
        if mode=='set':return (F.binary_cross_entropy_with_logits(z,yy[:,None].expand_as(z),reduction='none',pos_weight=pw)*mm[:,None]).sum((0,2))/mm.sum().clamp_min(1)
        if mode=='ce':return F.cross_entropy(z.permute(0,2,1),yy[:,None].expand(-1,heads),reduction='none').mean(0)
        return ((z[:,:,0]-yy[:,None])**2).mean(0)
    best=torch.full((heads,),float('inf'),device='cuda');bw=w.detach().clone();bb=b.detach().clone();chosen=[0]*heads
    for step in range(1,steps+1):
        ids=torch.as_tensor(rng.integers(len(x),size=min(256,len(x))),device='cuda')
        z=torch.einsum('nld,ldk->nlk',x[ids],w)+b
        l=loss(z,y[ids],mask[ids] if mask is not None else None).mean();opt.zero_grad(set_to_none=True);l.backward();opt.step()
        if step%100==0 or step==steps:
            with torch.no_grad():
                scores=torch.zeros(heads,device='cuda')
                for at in range(0,len(v),256):
                    zz=torch.einsum('nld,ldk->nlk',v[at:at+256],w)+b
                    scores+=loss(zz,vy[at:at+256],vm[at:at+256] if vm is not None else None)*len(zz)/len(v)
                better=scores<best;best=torch.minimum(best,scores);bw[better]=w.detach()[better];bb[better]=b.detach()[better]
                for h in torch.where(better)[0].tolist():chosen[h]=step
    return dict(mean=mean,std=std,w=bw,b=bb,mode=mode,steps=chosen)


def evaluate(fitted,d,task):
    import numpy as np
    import torch
    from cot_navigation.analysis.probes import metrics
    out=[]
    for at in range(0,len(d['x']),256):
        x=(torch.as_tensor(d['x'][at:at+256],device='cuda')-fitted['mean'])/fitted['std']
        out.append((torch.einsum('nld,ldk->nlk',x,fitted['w'])+fitted['b']).detach().cpu().numpy())
    z=np.concatenate(out);scores=[]
    for h in range(z.shape[1]):
        if fitted['mode']=='set':
            score=metrics(z[:,h],d['y'][task],d['mask'])
            if task=='top' or task.startswith('slot'):
                truth=d['y'][task].argmax(-1);score['top1_accuracy']=float((z[:,h].argmax(-1)==truth).mean())
        elif fitted['mode']=='ce':score=dict(accuracy=float((z[:,h].argmax(-1)==np.array([r[task] for r in d['meta']])).mean()),n_rows=len(z))
        else:
            truth=np.array([r[task] for r in d['meta']]);pred=z[:,h,0]
            score=dict(accuracy=float((np.rint(pred)==truth).mean()),mae=float(abs(pred-truth).mean()),n_rows=len(z))
        scores.append(dict(layer=h+1 if h<z.shape[1]-1 else 'input_position_control',**score))
    return scores


def sub(d,keep):
    import numpy as np
    ix=np.flatnonzero(keep)
    return dict(x=d['x'][ix],y={k:v[ix] for k,v in d['y'].items()},mask=d['mask'][ix],meta=[d['meta'][i] for i in ix],logits=d['logits'][ix])


def lens_report(model,tok,features,steps):
    import numpy as np
    import torch
    from cot_navigation.analysis.within_token import fit_lens,lens_logits
    def split(d,lens):return sub(d,[((r['query'] in lens)) for r in d['meta']])
    trainq=sorted({r['query'] for r in features['train']['meta']})[:512]
    valq=sorted({r['query'] for r in features['val']['meta']})[:128]
    tr=split(features['train'],trainq);va=split(features['val'],valq)
    layers=model.n_layers
    fitted=fit_lens(model,tr['x'][:,:layers,:128],tr['logits'],va['x'][:,:layers,:128],va['logits'],steps=steps)
    result=[]
    test=features['test']
    for kind in ['raw','calibrated']:
        for layer in range(layers):
            pred=[]
            for at in range(0,len(test['x']),512):
                h=torch.as_tensor(test['x'][at:at+512,layer,:128],device='cuda')
                pred.extend(lens_logits(model,fitted,h,layer,kind).argmax(-1).cpu().tolist())
            for field in range(4):
                for location in ['entry','end']:
                    for phase in ['early','later']:
                        ids=[i for i,r in enumerate(test['meta']) if r['field']==field and r['location']==location and (r['record']<=4)==(phase=='early')]
                        if ids:result.append(dict(kind=kind,layer=layer+1,field=FIELDS[field],location=location,phase=phase,n=len(ids),
                            exact_next_token_accuracy=float(np.mean([pred[i]==test['meta'][i]['target'] for i in ids])),
                            legal_next_token_accuracy=float(np.mean([pred[i] in test['meta'][i]['allowed'] for i in ids])),
                            agreement_with_final=float(np.mean([pred[i]==int(test['logits'][i].argmax()) for i in ids]))))
    return dict(scores=result,selected_steps=fitted['steps'],validation_kl=fitted['validation_kl'],identity_kl=fitted['identity_kl'],
                note='下一token精确匹配仅用于诊断，合法候选可以有不同顺序；最终答案YES/NO不作为本次lens目标。')


def work():
    import numpy as np
    import torch
    from cot_navigation.config import load_config
    from cot_navigation.data.dataset import ContextStore
    from cot_navigation.evaluation.sampling import stratified_ids
    from cot_navigation.models.transformer import build_model
    from cot_navigation.protocols.tokenizer import Tokenizer
    from cot_navigation.training.checkpoint import load_checkpoint
    torch.set_num_threads(2)
    # Bound this analysis process while sharing an 80GB GPU with another job.
    torch.cuda.set_per_process_memory_fraction(.20)
    cfg=load_config([MODELS/'l6_p60/s0/resolved_config.yaml']);store=ContextStore(cfg.data.root)
    queries={}
    for split,count in [('train',1536),('val',256),('test',256)]:
        ids=store.indices(split);a=store.arrays;ids=ids[a['label'][ids]==1]
        queries[split]=stratified_ids(ids,a['shape_id'],a['distance'],a['label'],count,20261102)
    hashes={s:set(map(int,store.arrays['struct_hash'][qs])) for s,qs in queries.items()}
    assert all(not hashes[a]&hashes[b] for a,b in [('train','val'),('train','test'),('val','test')])
    write_json(OUT/'plan.json',dict(queries=queries,probe_steps=3000,lens_steps=2000,train_records=[1,4],test_records=[1,16],
        checkpoint_policy='latest200k; l6/l9 p60 seeds0/1',variables=['field','record','depth','stack','visited','successors','eligible','top','top-relative stack slots0..7'],
        threshold=.5,precision='bf16 extraction; fp32 probes',split='graph/query disjoint official splits; separate query subsets for lens/probe',
        population='positive N100 simple graphs; naturally correct complete records only; no outcome filtering'))
    for depth in [6,9]:
        for seed in [0,1]:
            name=f'l{depth}_p60_s{seed}';dest=OUT/f'{name}.json'
            if dest.exists() and json.loads(dest.read_text()).get('complete'):continue
            folder=MODELS/f'l{depth}_p60'/f's{seed}';cfg=load_config([folder/'resolved_config.yaml']);tok=Tokenizer.from_config(cfg)
            ckpt=folder/'checkpoints/latest.pt';sha=file_sha256(ckpt);blob=load_checkpoint(ckpt,map_location='cpu');assert blob['step']==200000
            model=build_model(cfg.model,len(tok),pad_id=tok.pad).cuda().eval().requires_grad_(False);model.load_state_dict(blob['model']);del blob
            features={}
            for split,qs in queries.items():
                cases=generate_cases(model,tok,store,qs,split,name,sha)
                features[split]=extract(model,tok,store,cases)
            result=dict(complete=False,name=name,checkpoint_sha256=sha,selection={},probes=[],lens=lens_report(model,tok,features,2000))
            # Reserve lens-query subsets; actual surviving query sets determine the split.
            tq=set(sorted({r['query'] for r in features['train']['meta']})[:512]);vq=set(sorted({r['query'] for r in features['val']['meta']})[:128])
            features['train']=sub(features['train'],[r['query'] not in tq for r in features['train']['meta']])
            features['val']=sub(features['val'],[r['query'] not in vq for r in features['val']['meta']])
            result['selection']={s:dict(rows=len(d['meta']),queries=len(set(r['query'] for r in d['meta']))) for s,d in features.items()}
            for location in ['entry','end']:
                for field in [-1,0,1,2,3]:
                    tasks=['field'] if field==-1 else ['record','depth','stack','visited','successors','eligible','top']+[f'slot{i}' for i in range(8)]
                    for task in tasks:
                        bank={s:sub(d,[r['location']==location and (field==-1 or r['field']==field) and (not task.startswith('slot') or r['depth']>int(task[4:])) for r in d['meta']]) for s,d in features.items()}
                        if min(len(bank[s]['meta']) for s in ['train','val'])<64:continue
                        fitted=fit(bank['train'],bank['val'],task)
                        for split in ['val','test']:
                            for phase in ['early','later']:
                                d=sub(bank[split],[((r['record']<=4)==(phase=='early')) for r in bank[split]['meta']])
                                if len(d['meta']):result['probes'].append(dict(task=task,field='pooled' if field==-1 else FIELDS[field],location=location,split=split,phase=phase,scores=evaluate(fitted,d,task),selected_steps=fitted['steps']))
                        del fitted;write_json(dest,result)
                        print('PROBE',name,location,field,task,flush=True)
            result['complete']=True;assert file_sha256(ckpt)==sha;write_json(dest,result)
            del model,features;torch.cuda.empty_cache()
    write_json(OUT/'status.json',dict(complete=True,models=4))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--background',action='store_true');p.add_argument('--execute',action='store_true');p.add_argument('--share-gpu',action='store_true',help='Allow GPU0 sharing if at least 24GiB is free; analysis allocator capped at 20%%');a=p.parse_args()
    os.chdir(ROOT);OUT.mkdir(parents=True,exist_ok=True)
    if a.background:
        import subprocess
        with (OUT/'run.log').open('a') as log:
            child=subprocess.Popen([sys.executable,str(__file__),'--execute']+(['--share-gpu'] if a.share_gpu else []),cwd=ROOT,env={**os.environ,'PYTHONUNBUFFERED':'1','CUDA_VISIBLE_DEVICES':'0'},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write_json(OUT/'process.json',dict(pid=child.pid));print('QUEUED PID',child.pid);return
    if not a.execute:p.error('Use --background or --execute')
    from cot_navigation.runtime.launch import idle_gpus
    with file_lock(OUT/'.lock',blocking=False):
        write_json(OUT/'status.json',dict(complete=False,state='waiting_for_idle_gpu0'))
        while not any(r['index']==0 and (r['idle'] or (a.share_gpu and r['memory_free_mb']>=24576)) for r in idle_gpus()):time.sleep(30)
        os.environ['CUDA_VISIBLE_DEVICES']='0';write_json(OUT/'status.json',dict(complete=False,state='running'))
        try:work()
        except Exception as exc:
            write_json(OUT/'status.json',dict(complete=False,state='failed',error=str(exc)));raise


if __name__=='__main__':main()

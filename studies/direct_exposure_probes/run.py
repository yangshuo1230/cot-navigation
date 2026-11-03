"""Direct graph-relation probes for completed exposure-controlled models.

Frozen FP32 forward on GPU1; all ridge fitting on GPU0. Reuse the suite's
original train/monitor/final query splits AND hold out focal-node identities.
No main-model training, activation files, fitted weights or new checkpoints.
"""
from pathlib import Path
import os, sys, json, hashlib, time
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import numpy as np
import torch
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False
ROOT=Path.cwd(); HERE=ROOT/'studies/direct_exposure_probes'
SUITE=ROOT/'studies/fixed_adjacency_suite'
MODELS=ROOT/'outputs/exposure_grokking_adj_v3'
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'studies/fixed100_layer_probes')]
from cot_navigation.models.transformer import TransformerLM
from cot_navigation.protocols.tokenizer import Tokenizer
import linear
TASKS=('predecessors','successors','ancestors','descendants')
NODE_SEEDS=(17,31,47); VECTOR_SEEDS=(201,303,407)

def write(path,value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False))
    tmp.replace(path)

def graph(world):
    n=world['n_nodes'];a=np.zeros((n,n),bool);r=np.eye(n,dtype=bool)
    for u,vs in enumerate(world['adj']):
        for v in vs:assert v>u;a[u,v]=True
    for u in reversed(range(n)):
        for v in np.flatnonzero(a[u]):r[u]|=r[v]
    for u in range(n):
        seen={u};queue=[u]
        for v in queue:
            for w in np.flatnonzero(a[v]):
                if w not in seen:seen.add(w);queue.append(w)
        assert seen==set(np.flatnonzero(r[u]))
    np.fill_diagonal(r,False)
    return a,np.stack([a.T,a,r.T,r]).astype(np.float32)

@torch.inference_mode()
def collect(run,world,records):
    job=json.loads((run/'resolved_job.json').read_text())
    assert json.loads((run/'COMPLETE.json').read_text())['step']==125000
    checkpoint=run/'latest.pt';digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    blob=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert blob['step']==125000 and blob['job']==job
    graph_hash=hashlib.sha256(json.dumps(world,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert blob['world_hash']==graph_hash==job['_world_hash']
    tok=Tokenizer(256);nodes=np.array([tok.node_id(v) for v in world['node_names']])
    ids=torch.tensor([[tok.id('<QUERY>'),nodes[s],nodes[t],tok.id('|')] for s,t,y,g in records])
    model=TransformerLM(len(tok),6,1,128,512,256,dropout=job['dropout']).eval().requires_grad_(False)
    model.load_state_dict(blob['model']);del blob
    # Exact CPU/GPU consistency check before using faster frozen inference.
    reference=model(ids[:16]).numpy();model.to('cuda:1')
    difference=float(np.max(np.abs(model(ids[:16].to('cuda:1')).cpu().numpy()-reference)))
    assert difference<1e-3,difference
    features=[];next_tokens=[]
    for start in range(0,len(ids),256):
        tokens=ids[start:start+256].to('cuda:1');cap={};logits=model(tokens,capture=cap)
        hidden=torch.stack([model.embed(tokens)]+cap['resid'])[:,:,-3:-1]
        features.append(hidden.permute(1,2,0,3).cpu().numpy())
        next_tokens.extend(logits[:,-1].argmax(-1).cpu().tolist())
    final=json.loads((run/'final.json').read_text());lookup={tuple(v):i for i,v in enumerate(records)}
    agree=sum(next_tokens[lookup[tuple(v['query'])]]==v['predicted_token'] for v in final['predictions'])
    manifest=dict(job=job['id'],checkpoint=str(checkpoint),checkpoint_step=125000,
        checkpoint_sha256=digest,graph_sha256=graph_hash,n_nodes=len(nodes),n_edges=world['n_edges'],
        source_code_sha256=job['_code_sha256'],inference='FP32 GPU1',fitting='GPU0',
        CPU_GPU_max_logit_error=difference,official_final={k:final[k] for k in ('n','accuracy','balanced_accuracy','yes_recall','no_recall')},
        FP32_vs_official_prediction_agreement=agree/len(final['predictions']))
    del model;torch.cuda.empty_cache()
    return np.concatenate(features),manifest

def fit(features,labels,split):
    """Multi-task batching preserves separate val-MSE lambda for each task/layer.

    labels: [query, task, candidate]. Random identity features have exactly
    the same feature dimension and train-only scaling/regularization path.
    """
    tr,va=split[:2];device='cuda:0';n,k=labels.shape[1:]
    z=torch.tensor(features,device=device).transpose(0,1)
    mean=z[:,tr].mean(1,keepdim=True);std=z[:,tr].std(1,keepdim=True).clamp_min(.05)
    z=(z-mean)/std;z=torch.cat([z,torch.ones((*z.shape[:2],1),device=device)],-1)
    yy=torch.tensor(labels.reshape(len(labels),-1),device=device)
    train=z[:,tr];gram=train.transpose(1,2)@train/len(tr);rhs=train.transpose(1,2)@yy[tr]/len(tr)
    eye=torch.eye(z.shape[-1],device=device);eye[-1,-1]=0
    jitter=1e-6*torch.eye(z.shape[-1],device=device)
    best=np.full((z.shape[0],n),np.inf);chosen=np.zeros_like(best)
    weights=torch.empty((z.shape[0],z.shape[-1],n*k),device=device)
    for penalty in (.001,.01,.1,1.,10.):
        w=torch.linalg.solve(gram+penalty*eye+jitter,rhs)
        p=(z[:,va]@w).reshape(z.shape[0],len(va),n,k)
        loss=((p-yy[va].reshape(1,len(va),n,k))**2).mean((1,3)).cpu().numpy()
        for f,task in np.argwhere(loss<=best):
            best[f,task]=loss[f,task];chosen[f,task]=penalty
            weights[f,:,task*k:(task+1)*k]=w[f,:,task*k:(task+1)*k]
    pred=(z@weights).transpose(0,1).reshape(len(labels),z.shape[0],n,k).cpu().numpy()
    return pred,chosen.tolist()

def thresholds(y,p,mask):
    values=[]
    truth=y.astype(bool)&mask
    for f in range(p.shape[1]):
        scores=p[:,f];cuts=np.unique(np.r_[0,np.quantile(scores[mask],np.linspace(0,1,41))])
        quality=[]
        for cut in cuts:
            guess=(scores>=cut)&mask;tp=(guess&truth).sum();fp=(guess&~truth).sum();fn=(~guess&truth).sum()
            quality.append(2*tp/max(2*tp+fp+fn,1))
        values.append(float(cuts[int(np.argmax(quality))]))
    return values

def evaluate(y,p,indices,focal,other,forward,cuts):
    """Fixed-candidate AUC after grouping repeated focal nodes, not pooled folds.

    Direction-valid candidates remove trivial topological-order negatives.
    Current query endpoints are excluded, so copying them cannot earn credit.
    F1 uses an all-other-node validation threshold; it is not a probability.
    """
    n=y.shape[1];q=np.arange(n)[None,:];f=focal[indices];o=other[indices]
    mask=(q!=f[:,None])&(q!=o[:,None])
    valid=mask&((q>f[:,None]) if forward else (q<f[:,None]))
    groups=np.unique(f);gy=[];gp=[];gm=[]
    for node in groups:
        loc=np.flatnonzero(f==node);m=valid[loc];count=m.sum(0)
        gp.append(np.round((p[indices[loc]].astype(np.float64)*m[:,None,:]).sum(0)/np.maximum(count,1)[None,:],9))
        gy.append(y[indices[loc[0]]]);gm.append(count>0)
    gp=np.array(gp);gy=np.array(gy);gm=np.array(gm)
    out=[];truth=y[indices].astype(bool)&mask
    for feature in range(p.shape[1]):
        aucs=[]
        for candidate in range(n):
            take=gm[:,candidate]
            value=linear.auc(gy[take,candidate],gp[take,feature,candidate])
            if value is not None:aucs.append(value)
        guess=(p[indices,feature]>=cuts[feature])&mask
        tp=int((guess&truth).sum());fp=int((guess&~truth).sum());fn=int((~guess&truth).sum())
        # Rank within direction-valid candidates, excluding empty/trivial sets.
        recalls=[];random=[]
        for yy,score,eligible in zip(y[indices],p[indices,feature],valid):
            yy=yy[eligible].astype(bool);score=score[eligible];k=int(yy.sum())
            if not k or k==len(yy):continue
            recalls.append(float(yy[np.argsort(score)[-k:]].mean()));random.append(k/len(yy))
        out.append(dict(auc=float(np.mean(aucs)) if aucs else None,n_candidates=len(aucs),
            focal_nodes=len(groups),queries=len(indices),f1=2*tp/max(2*tp+fp+fn,1),
            precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),exact_set=float((guess==truth).all(1).mean()),
            recall_at_true_k=float(np.mean(recalls)) if recalls else None,
            random_recall=float(np.mean(random)) if random else None,threshold=cuts[feature]))
    return out

def run(g,seed,arm):
    run_dir=MODELS/f'fixed/g{g}_s{seed}/direct/{arm}';out=HERE/f'g{g}_s{seed}_{arm}.json'
    if out.exists():
        result=json.loads(out.read_text())
        assert result['manifest']['checkpoint_sha256']==hashlib.sha256((run_dir/'latest.pt').read_bytes()).hexdigest()
        print('verified existing',out.name,flush=True);return
    world=json.loads((SUITE/f'data/graph_{g}.json').read_text());a,mats=graph(world);n=len(a)
    records=[];parts=[];seen=set()
    for part,rows in enumerate((world['base']['train'],world['base']['monitor'],world['base']['final'],world['core_final'])):
        for record in rows:
            s,t,y,gap=record
            assert (s,t) not in seen;seen.add((s,t))
            assert bool(mats[3,s,t])==bool(y)
            records.append(record);parts.append(part)
    records=np.array(records);parts=np.array(parts)
    x,manifest=collect(run_dir,world,records)
    result=dict(manifest=manifest,layers=['E']+[f'L{i}' for i in range(1,7)],roles={},
        query_counts={str(i):int(sum(parts==i)) for i in range(4)},
        method='Frozen original query prompts. Base query split AND focal-node holdout; core_final evaluated separately with same fitted decoder. '
               'Fixed-candidate AUROC over heldout focal nodes, restricted to valid topological direction, excluding query endpoints. '
               'Three node partitions; three fixed random identity controls; std is not a confidence interval.')
    random={rs:np.random.default_rng(rs).normal(size=(n,128)).astype(np.float32) for rs in VECTOR_SEEDS}
    for role,name in enumerate(('source','target')):
        focal=records[:,role];other=records[:,1-role];features=x[:,role]
        labels=mats[:,focal].transpose(1,0,2)
        entries={task:[] for task in TASKS}
        for node_seed in NODE_SEEDS:
            order=np.random.default_rng(node_seed).permutation(n);fold=np.zeros(n,int)
            fold[order[int(.6*n):int(.8*n)]]=1;fold[order[int(.8*n):]]=2
            split=[np.flatnonzero((parts==i)&(fold[focal]==i)) for i in range(3)]
            core=np.flatnonzero((parts==3)&(fold[focal]==2))
            assert all(len(v)>0 for v in split) and len(core)>0
            assert not set(focal[split[0]])&set(focal[split[2]])
            mean=features[split[0]].mean(0);std=features[split[0]].std(0)
            controls=[random[rs][focal,None,:]*std[None,:,:]+mean[None,:,:] for rs in VECTOR_SEEDS]
            xx=np.concatenate([features]+controls,axis=1)
            pred,penalties=fit(xx,labels,split)
            for task_i,task in enumerate(TASKS):
                y=labels[:,task_i];p=pred[:,:,task_i];tr,va,te=split
                mask=np.ones((len(va),n),bool);mask[np.arange(len(va)),focal[va]]=False;mask[np.arange(len(va)),other[va]]=False
                cuts=thresholds(y[va],p[va],mask)
                values={}
                for panel,ids in [('base_final',te),('core_final',core)]:
                    metrics=evaluate(y,p,ids,focal,other,task in ('successors','descendants'),cuts)
                    values[panel]=dict(actual=metrics[:7],random={str(rs):metrics[(j+1)*7:(j+2)*7] for j,rs in enumerate(VECTOR_SEEDS)})
                entries[task].append(dict(node_seed=node_seed,query_counts=list(map(len,split)),core_queries=len(core),
                    focal_node_counts=[len(set(focal[v])) for v in split],penalties=np.array(penalties)[:,task_i].tolist(),panels=values))
            print(manifest['job'],name,'node split',node_seed,'complete',flush=True)
            del xx,pred;torch.cuda.empty_cache()
        result['roles'][name]=entries
    write(out,result);print(manifest['job'],'COMPLETE',flush=True)

if __name__=='__main__':
    HERE.mkdir(exist_ok=True)
    for g in range(3):
        for seed in range(2):
            for arm in ('R0','R2','R3'):run(g,seed,arm)

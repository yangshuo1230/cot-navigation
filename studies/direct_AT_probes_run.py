"""Node-held-out A/T set probes on final v6 Direct; all fits on GPU0.

One vector per focal node: average frozen states over ALL forward query
contexts (same other-endpoint pool for nodes in each layer). No repeated
contexts of a held-out focal node enter probe training. No tensors saved.
"""
from pathlib import Path
import sys, os, json, hashlib, importlib.util
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import torch
torch.set_num_threads(2)
ROOT=Path.cwd();OUT=ROOT/'studies/direct_AT_probes'
sys.path.insert(0,str(ROOT/'src'))
from cot_navigation.models.transformer import TransformerLM
from cot_navigation.protocols.tokenizer import Tokenizer
spec=importlib.util.spec_from_file_location('sets',ROOT/'studies/embedding_relations_v5/run.py')
sets=importlib.util.module_from_spec(spec);spec.loader.exec_module(sets)
STAGES=['E','L1A','L1F','L2F','L3F','L4F','L5F','L6F']
PENALTIES=(.0001,.001,.01,.1,1.,10.,100.)
TASKS=['ancestors','descendants','successors']

@torch.inference_mode()
def collect(world,folder):
    path=folder/'latest.pt';blob=torch.load(path,map_location='cpu',weights_only=False)
    assert blob['step']==50000
    model=TransformerLM(270,**blob['job']['model'],dropout=0).eval().requires_grad_(False)
    tok=Tokenizer(256);assert len(tok)==270
    model.load_state_dict(blob['model']);model.to('cuda:0');del blob
    rows=world['records'];nodes=[tok.node_id(v) for v in world['node_names']]
    ids=torch.tensor([[tok.id('<QUERY>'),nodes[r['s']],nodes[r['t']],tok.id('|')] for r in rows],device='cuda:0')
    observed={};handles=[]
    def before(i):
        def f(m,a):observed[i,'input']=a[0]
        return f
    def after(i):
        def f(m,a,o):observed[i,'A']=observed[i,'input']+o[0]
        return f
    with torch.autocast('cuda',dtype=torch.bfloat16):reference=model(ids[:128])
    for i,b in enumerate(model.blocks):handles.extend([b.register_forward_pre_hook(before(i)),b.attn.register_forward_hook(after(i))])
    sums=np.zeros((2,80,8,128),np.float64);counts=np.zeros((2,80),int);hook_error=0
    for start in range(0,len(ids),128):
        tokens=ids[start:start+128];cap={}
        with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(tokens,capture=cap)
        if start==0:hook_error=float((reference-logits).abs().max());assert hook_error==0
        x=torch.stack([model.embed(tokens),observed[0,'A']]+cap['resid'],1)[:,:,-3:-1].float().cpu().numpy()
        for role in range(2):
            focal=np.array([r['s' if role==0 else 't'] for r in rows[start:start+128]])
            np.add.at(sums[role],focal,x[:,:,role]);np.add.at(counts[role],focal,1)
    for h in handles:h.remove()
    features=(sums/np.maximum(counts[:,:,None,None],1)).astype(np.float32)
    # E is raw embedding, including nodes without forward-context support.
    embedding=model.embed(torch.tensor(nodes,device='cuda:0')).float().cpu().numpy()
    features[:,:,0]=embedding[None]
    del model;torch.cuda.empty_cache()
    return features,counts,dict(checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),hook_max_error=hook_error)

def fit(x,y,tr,va,te):
    """FP64 dual ridge; separate validation MSE choice per feature/task."""
    z=torch.tensor(x.transpose(1,0,2),dtype=torch.float64,device='cuda:0')
    yy=torch.tensor(y.reshape(80,-1),dtype=torch.float64,device='cuda:0')
    mean=z[:,tr].mean(1,keepdim=True);std=z[:,tr].std(1,unbiased=False,keepdim=True).clamp_min(.05)
    z=(z-mean)/std
    # Center features again so the intercept is strictly unpenalized.
    z=z-z[:,tr].mean(1,keepdim=True);ym=yy[tr].mean(0);target=yy[tr]-ym
    a=z[:,tr];gram=a@a.transpose(1,2)/len(tr);eye=torch.eye(len(tr),device=z.device,dtype=z.dtype)
    best=np.full((x.shape[1],3),np.inf);pen=np.zeros_like(best)
    vp=np.zeros((len(va),x.shape[1],3,80));tp=np.zeros((len(te),x.shape[1],3,80))
    for lam in PENALTIES:
        dual=torch.linalg.solve(gram+lam*eye,target.expand(x.shape[1],-1,-1)/len(tr))
        v=(z[:,va]@a.transpose(1,2)@dual+ym).reshape(x.shape[1],len(va),3,80)
        t=(z[:,te]@a.transpose(1,2)@dual+ym).reshape(x.shape[1],len(te),3,80)
        loss=(v-yy[va].reshape(1,len(va),3,80)).square().mean((1,3)).cpu().numpy()
        vn=v.cpu().numpy();tn=t.cpu().numpy()
        for h,k in np.argwhere(loss<=best):
            best[h,k]=loss[h,k];pen[h,k]=lam;vp[:,h,k]=vn[h,:,k];tp[:,h,k]=tn[h,:,k]
    return vp,tp,pen.tolist()

def assignment(nodes,layer,seed):
    rng=np.random.default_rng(seed);fold=np.full(80,-1,int);offset=0
    for level in range(10):
        group=rng.permutation(np.array([v for v in nodes if layer[v]==level],dtype=int))
        fold[group]=(np.arange(len(group))+offset)%5;offset+=len(group)
    return fold

def metrics(y,v,t,va,te,mask):
    identity=~np.eye(80,dtype=bool)
    cut=sets.validation_threshold(y[va],v,identity[va])
    return dict(column=sets.columns_auc(y[te],t,mask[te]),
                row=sets.rank_rows(y[te],t,mask[te]),
                recovery=sets.set_recovery(y[te],t,cut,identity[te]))

def main():
    OUT.mkdir(exist_ok=True)
    world=json.loads((ROOT/'studies/fixed_navigation_v5/data/n80_g0.json').read_text())
    layer=np.arange(80)//8;adj=np.zeros((80,80),bool);reach=np.zeros_like(adj)
    for u in reversed(range(80)):
        for v in world['adj'][u]:adj[u,v]=True;reach[u,v]=True;reach[u]|=reach[v]
    for u in range(80):
        seen=set();q=list(world['adj'][u])
        for v in q:
            if v in seen:continue
            seen.add(v);q.extend(world['adj'][v])
        assert seen==set(np.flatnonzero(reach[u]))
    y=np.stack([reach.T,reach,adj]).astype(float)
    masks=[layer[:,None]>layer[None,:],layer[:,None]<layer[None,:],layer[:,None]+1==layer[None,:]]
    groups={k:np.array(world['node_groups'][k]) for k in ('A','T')}
    for seed in range(4):
        folder=ROOT/f'outputs/fixed_navigation_protocols_v6/pilot/n80/g0_s{seed}/direct'
        states,counts,manifest=collect(world,folder)
        result=dict(seed=seed,manifest=manifest,stages=STAGES,records=[],context_counts=counts.tolist(),
                    method='One context-mean vector per node; focal-node holdout; all candidate identities fixed. Primary masks only allow correct layer direction (successors: next layer).')
        for role in range(2):
            eligible={k:vs[counts[role,vs]>0] for k,vs in groups.items()}
            # Controls have identical 128-dimensional input size except padded one-hot/prior.
            xs=[states[role]];feature_names=STAGES.copy()
            for rs in (101,202,303):
                xs.append(np.random.default_rng(rs).normal(size=(80,1,128)).astype(np.float32));feature_names.append(f'random{rs}')
            oracle=np.zeros((80,1,128),np.float32);oracle[np.arange(80),0,layer]=1
            xs.extend([oracle,np.zeros_like(oracle)]);feature_names+=['layer_oracle','candidate_prior']
            x=np.concatenate(xs,axis=1)
            for ns in (17,31,47):
                folds={k:assignment(vs,layer,ns) for k,vs in eligible.items()}
                for fold in range(5):
                    tr={k:vs[(folds[k][vs]!=fold)&(folds[k][vs]!=(fold+1)%5)] for k,vs in eligible.items()}
                    va={k:vs[folds[k][vs]==(fold+1)%5] for k,vs in eligible.items()}
                    te={k:vs[folds[k][vs]==fold] for k,vs in eligible.items()}
                    rng=np.random.default_rng(ns*100+fold)
                    setups={'pooled':(np.r_[tr['A'],tr['T']],np.r_[va['A'],va['T']]),
                            'A_full':(tr['A'],va['A']), 'T':(tr['T'],va['T']),
                            'A_matched':(rng.choice(tr['A'],len(tr['T']),replace=False),rng.choice(va['A'],len(va['T']),replace=False))}
                    for setup,(train,val) in setups.items():
                        test=np.r_[te['A'],te['T']]
                        assert not(set(train)&set(test) or set(val)&set(test) or set(train)&set(val))
                        vp,tp,pen=fit(x,y,train,val,test)
                        rec=dict(role=('source','target')[role],node_seed=ns,fold=fold,setup=setup,
                                 train=train.tolist(),validation=val.tolist(),test_A=te['A'].tolist(),test_T=te['T'].tolist(),
                                 penalties=pen,features={})
                        for h,name in enumerate(feature_names):
                            rec['features'][name]={}
                            for k,task in enumerate(TASKS):
                                rec['features'][name][task]={}
                                for group in ('A','T'):
                                    loc=np.flatnonzero(np.isin(test,te[group]))
                                    rec['features'][name][task][group]=metrics(y[k],vp[:,h,k],tp[loc,h,k],val,test[loc],masks[k])
                        # Same-layer training-label permutation is a semantic control.
                        shuffled=y.copy();order=np.arange(80)
                        for level in range(10):
                            ix=train[layer[train]==level];order[ix]=rng.permutation(ix)
                        shuffled[:,train]=y[:,order[train]]
                        sv,st,sp=fit(x[:,:8],shuffled,train,val,test)
                        rec['shuffled_features']={}
                        for h,name in enumerate(STAGES):
                            rec['shuffled_features'][name]={task:{group:metrics(y[k],sv[:,h,k],st[np.isin(test,te[group]),h,k],val,te[group],masks[k])
                                for group in ('A','T')} for k,task in enumerate(TASKS)}
                        result['records'].append(rec)
                print(f'seed{seed} role{role} split{ns} complete',flush=True)
        (OUT/f'seed{seed}.json').write_text(json.dumps(result,separators=(',',':')))
        print(f'seed{seed} COMPLETE',flush=True)

if __name__=='__main__':main()

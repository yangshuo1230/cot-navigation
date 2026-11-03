"""Frozen Path next-node study. Run from repo root; only affine fits use GPU 0.

Reuse native free generations, keep original query splits, exclude target/end
positions and illegal prefixes. No activations/checkpoints are saved. A linear
score is not a calibrated probability, and attention is descriptive evidence.
"""
from pathlib import Path
import os, sys, json, gzip, hashlib, itertools
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import numpy as np
import torch
torch.set_num_threads(2)
ROOT=Path.cwd(); HERE=ROOT/'studies/path_direction'
OLD=ROOT/'studies/fixed100_layer_probes'
sys.path[:0]=[str(ROOT/'src'),str(OLD)]
import linear
from cot_navigation.config import load_config
from cot_navigation.data.dataset import ParametricWorld
from cot_navigation.models.transformer import build_model
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.analysis.attention import DistanceBaseline
from cot_navigation.models.attention import apply_rope
ROLES=['syntax','query source','query target','separator','earlier path','current node']

def write(path,obj):
    p=path.with_suffix('.tmp');p.write_text(json.dumps(obj,indent=2,allow_nan=False));p.replace(path)

def fit(x,y,split):
    """Multi-output affine ridge; one validation-MSE lambda per layer, GPU 0.

    No gradient descent or frozen-model changes. Targets are 100-dimensional
    incidence vectors. A single lambda avoids per-output selection overfitting.
    """
    if x.ndim==2:x=x[:,None,:]
    tr,va,te=split; dev='cuda:0'
    z=torch.tensor(x,device=dev).transpose(0,1)
    mu=z[:,tr].mean(1,keepdim=True);sd=z[:,tr].std(1,keepdim=True).clamp(min=.05)
    z=(z-mu)/sd;z=torch.cat([z,torch.ones((*z.shape[:2],1),device=dev)],-1)
    yt=torch.tensor(y,device=dev);zt=z[:,tr]
    gram=zt.transpose(1,2)@zt/len(tr);rhs=zt.transpose(1,2)@yt[tr]/len(tr)
    eye=torch.eye(z.shape[-1],device=dev);eye[-1,-1]=0
    best=np.full(z.shape[0],np.inf);out=np.zeros((len(y),z.shape[0],y.shape[1]),np.float32);chosen=[]
    choices=np.zeros(z.shape[0])
    for penalty in (.001,.01,.1,1.,10.):
        w=torch.linalg.solve(gram+penalty*eye+1e-6*torch.eye(z.shape[-1],device=dev),rhs)
        p=(z@w).transpose(0,1).cpu().numpy()
        loss=((p[va]-y[va,None,:])**2).mean((0,2))
        for l in range(z.shape[0]):
            if loss[l]<=best[l]:best[l]=loss[l];out[:,l]=p[:,l];choices[l]=penalty
    return out,choices.tolist()

def ranking(y,s,mask=None):
    """Macro per-decision AUROC, top-|true| recall, and top-1 hit.

    Every metric here requires >=1 true and >=1 false eligible candidate.
    Thus all-zero vectors and trivial all-neighbor-positive decisions cannot
    inflate performance. Random ranking has AUROC .5, top1=k/candidates.
    """
    if mask is None:mask=np.ones_like(y,bool)
    auc=[];rec=[];hit=[];chance=[];margin=[]
    for yy,ss,mm in zip(y,s,mask):
        yy=yy[mm].astype(bool);ss=ss[mm];k=int(yy.sum())
        if not k or k==len(yy):continue
        auc.append(linear.auc(yy,ss));rec.append(float(yy[np.argsort(ss)[-k:]].mean()))
        hit.append(float(yy[np.argmax(ss)]));chance.append(k/len(yy))
        margin.append(float(ss[yy].mean()-ss[~yy].mean()))
    return dict(n=len(auc),auc=float(np.mean(auc)) if auc else None,
        recall_at_true_k=float(np.mean(rec)) if rec else None,
        top1_hit=float(np.mean(hit)) if hit else None,
        random_top1=float(np.mean(chance)) if chance else None,
        mean_positive_minus_negative_score=float(np.mean(margin)) if margin else None)

@torch.inference_mode()
def collect(setting):
    previous=json.loads((OLD/f'{setting}_path.json').read_text())['manifest']
    run=Path(previous['run']);ck=run/'checkpoints/best.pt'
    assert hashlib.sha256(ck.read_bytes()).hexdigest()==previous['checkpoint_sha256']
    cfg=load_config([run/'resolved_config.yaml']);tok=Tokenizer.from_config(cfg)
    world=ParametricWorld(Path(cfg.data.root)/'main-seed0');adj=world.adj();n=len(adj)
    nodes=world.node_vocab_ids(tok);inv={int(v):i for i,v in enumerate(nodes)}
    adjacency=np.zeros((n,n),np.float32);reach=np.eye(n,dtype=bool)
    for u in range(n):adjacency[u,adj[u]]=1
    for u in reversed(range(n)):
        for v in adj[u]:reach[u]|=reach[v]
    header=[tok.id('<WORLD_0>')] if cfg.train.world_roots else []
    rows=[json.loads(l) for l in gzip.open(OLD/f'{setting}_path_queries.jsonl.gz','rt')]
    seqs=[header+[tok.id('<QUERY>'),nodes[r['source']],nodes[r['target']],tok.id('|')]+r['trace'] for r in rows]
    plen=len(header)+4;maximum=max(map(len,seqs));baseline=DistanceBaseline(6,1,maximum)
    blob=torch.load(ck,map_location='cpu',weights_only=False)
    model=build_model(cfg.model,len(tok),tok.pad).to('cuda:1').eval().requires_grad_(False)
    model.load_state_dict(blob['model']);del blob
    features=[];meta=[];test_att=[];lens=[];lens_scores=[];next_logits=[]
    # Observe native SDPA without replacing its attention implementation. The
    # built-in return_attn=True changes BF16 arithmetic and can flip close logits.
    # Hooks reconstruct QK weights from the ACTUAL native-layer input, and also
    # capture the residual immediately after attention, before the FFN.
    observed={};record=[True]
    def preblock(index):
        def hook(module,args):
            if record[0]:observed['input',index]=args[0]
        return hook
    def postattn(index):
        def hook(module,args,output):
            if not record[0]:return
            observed['after_attn',index]=observed['input',index]+output[0]
            xx,positions=args[:2];b,length,_=xx.shape
            qkv=module.qkv(xx).view(b,length,3,module.n_heads,module.head_dim)
            q,k,_=qkv.unbind(2)
            q=apply_rope(q.transpose(1,2),positions,module.rope_base)
            k=apply_rope(k.transpose(1,2),positions,module.rope_base)
            with torch.autocast('cuda',enabled=False):
                score=q.float()@k.float().transpose(-2,-1)/module.head_dim**.5
                causal=torch.ones(length,length,device=xx.device,dtype=torch.bool).tril()
                observed['weights',index]=score.masked_fill(~causal,float('-inf')).softmax(-1)
        return hook
    handles=[]
    for l,block in enumerate(model.blocks):
        handles.extend([block.register_forward_pre_hook(preblock(l)),block.attn.register_forward_hook(postattn(l))])
    max_error=0;argmax_equal=True
    for start in range(0,len(rows),64):
        seq=seqs[start:start+64];length=max(map(len,seq))
        ids=torch.full((len(seq),length),tok.pad,dtype=torch.long,device='cuda:1')
        for b,s in enumerate(seq):ids[b,:len(s)]=torch.tensor(s,device='cuda:1')
        capture={}
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(ids,capture=capture)
            if start==0:
                record[0]=False
                native=model(ids)
                record[0]=True
                max_error=float((native-logits).abs().max());argmax_equal=bool((native.argmax(-1)==logits.argmax(-1)).all())
            hidden=torch.stack([model.embed(ids)]+capture['resid'])
            sublayers=[model.embed(ids)]
            for l in range(6):sublayers.extend([observed['after_attn',l],capture['resid'][l]])
            ll=model.lm_head(model.final_norm(torch.stack(sublayers).float()))
        h=hidden.float().cpu().numpy();lln=ll[...,nodes].float().cpu().numpy()
        at=torch.stack([observed['weights',l] for l in range(6)]).float().cpu().numpy()
        for b,s in enumerate(seq):
            r=rows[start+b];a=at[:,b,:,:len(s),:len(s)]
            if r['split']==0:baseline.add(a)
            for j,token in enumerate(r['trace'][:-1]):
                if token not in inv:break
                u=inv[token];v=inv.get(r['trace'][j+1],-1)
                if j==0 and u!=r['source']:break
                if u==r['target'] or v<0:break
                pos=plen+j
                features.append(h[:,b,pos]);meta.append(dict(query=r['query_ids'],split=r['split'],
                    source=r['source'],target=r['target'],current=u,next=v,k=j+1,label=r['label']))
                lens_scores.append(lln[:,b,pos])
                # Actual native next-token distribution, including non-node tokens.
                next_logits.append(logits[b,pos].float().cpu().numpy())
                if r['split']==2:
                    roles=np.zeros(pos+1,int);roles[plen-3]=1;roles[plen-2]=2;roles[plen-1]=3
                    roles[plen:pos]=4;roles[pos]=5
                    test_att.append((len(meta)-1,a[:,:,pos,:pos+1].copy(),roles))
                if v not in adj[u]:break
        if start%512==0:print(setting,'capture',start,'/',len(rows),flush=True)
    x=np.asarray(features);m={k:np.array([r[k] for r in meta]) for k in meta[0]}
    ys=adjacency[m['current']];yu=ys*reach[:,m['target']].T
    yn=np.eye(n,dtype=np.float32)[m['next']]
    mixed=(ys.sum(1)>yu.sum(1))&(yu.sum(1)>0)
    # Report attention per decision. Baseline uses train full causal token pairs,
    # never validation/test, padding or answers. Unsupported distances excluded.
    att={};mu=baseline.mean;unsupported=0;allpairs=0
    for group,selection in [('all',np.ones(len(x),bool)),('mixed_useful_dead',mixed),('first_node',m['k']==1)]:
        raw=np.zeros((6,6));center=np.zeros_like(raw);count=0
        for ix,a,roles in test_att:
            if not selection[ix]:continue
            distances=len(roles)-1-np.arange(len(roles));valid=baseline.counts[distances]>0
            if group=='all':unsupported+=int((~valid).sum());allpairs+=len(valid)
            for role in range(6):
                keep=(roles==role)&valid
                raw[:,role]+=a[:,0,keep].sum(-1)
                center[:,role]+=(a[:,0,keep]-mu[:,0,distances[keep]]).sum(-1)
            count+=1
        att[group]=dict(n=count,raw_mass=(raw/max(count,1)).tolist(),distance_centered_mass=(center/max(count,1)).tolist())
    ns=np.asarray(next_logits);node_logits=ns[:,nodes];best_node=node_logits.argmax(-1)
    first=np.flatnonzero((m['split']==2)&(m['k']==1)&mixed&(m['label']==1))
    pairs=[];examples=[]
    for u in np.unique(m['current'][first]):
        ids=first[m['current'][first]==u]
        # All unordered test target pairs with disjoint useful-successor sets.
        for i,j in itertools.combinations(ids,2):
            if (yu[i]*yu[j]).sum():continue
            # Native observed generations define behavioral choices; SDPA
            # replay/uncalibrated lenses are analyzed separately below.
            p=int(m['next'][i]);q=int(m['next'][j])
            row=dict(current=tok.name(nodes[u]),target_a=tok.name(nodes[m['target'][i]]),target_b=tok.name(nodes[m['target'][j]]),
                useful_a=[tok.name(nodes[v]) for v in np.flatnonzero(yu[i])],useful_b=[tok.name(nodes[v]) for v in np.flatnonzero(yu[j])],
                next_a=tok.name(nodes[p]),next_b=tok.name(nodes[q]),both_useful=bool(yu[i,p] and yu[j,q]),changes=bool(p!=q))
            pairs.append(row)
            if len(examples)<5 and row['both_useful']:examples.append(row)
    out=dict(manifest=previous,scope='native generated node positions that predict another node; local prefix checked; terminal excluded',
        n_queries=len(rows),n_decisions=len(x),split_decisions={str(i):int((m['split']==i).sum()) for i in range(3)},
        observation_hooks_vs_native=dict(max_abs_logit_difference=max_error,all_argmax_equal=argmax_equal),
        attention=dict(roles=ROLES,groups=att,unsupported_distance_fraction=unsupported/max(allpairs,1)),
        target_pairs=dict(n=len(pairs),unique_sources=len(set(p['current'] for p in pairs)),
            fraction_choice_changes=float(np.mean([p['changes'] for p in pairs])) if pairs else None,
            fraction_both_useful=float(np.mean([p['both_useful'] for p in pairs])) if pairs else None,examples=examples),
        native=dict(test_node_argmax_matches_saved_next=float((best_node[m['split']==2]==m['next'][m['split']==2]).mean())))
    # Lens uses the trained final norm/head at earlier layers (uncalibrated).
    scores=np.asarray(lens_scores);te=np.flatnonzero(m['split']==2)
    out['sublayer_logit_lens']=[]
    for l in range(13):
        s=scores[:,l];prob=np.exp(s-s.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True)
        out['sublayer_logit_lens'].append(dict(successor=ranking(ys[te],s[te]),useful_within_successors=ranking(yu[te],s[te],ys[te].astype(bool)),
            successor_node_conditional_mass=float((prob[te]*ys[te]).sum(1).mean())))
    out['logit_lens']=[out['sublayer_logit_lens'][2*l] for l in range(7)]
    out['sublayer_names']=['E']+[name for l in range(1,7) for name in (f'L{l} attention',f'L{l} FFN')]
    emb=model.embed.weight.detach().float().cpu().numpy();del model;torch.cuda.empty_cache()
    return out,x,m,ys,yu,yn,mixed,emb,nodes

def analyze(setting):
    out,x,m,ys,yu,yn,mixed,emb,nodes=collect(setting)
    splits=[np.flatnonzero(m['split']==i) for i in range(3)]
    rng=np.random.default_rng(17);order=rng.permutation(100)
    # Both query and current-node held out. Graph nodes were seen in main training.
    nodefold=np.zeros(100,int);nodefold[order[60:80]]=1;nodefold[order[80:]]=2
    held=[np.flatnonzero((m['split']==i)&(nodefold[m['current']]==i)) for i in range(3)]
    curr=emb[np.asarray(nodes)[m['current']]];target=emb[np.asarray(nodes)[m['target']]]
    random_embed=np.random.default_rng(2026).normal(0,emb.std(),(100,emb.shape[1])).astype(np.float32)
    controls={'current_embedding':curr,'target_embedding':target,
        'current_target_embedding_concat':np.concatenate([curr,target],1),
        'random_current_embedding':random_embed[m['current']],
        'random_target_embedding':random_embed[m['target']],
        'current_target_onehot_concat':np.concatenate([np.eye(100)[m['current']],np.eye(100)[m['target']]],1).astype(np.float32)}
    out['probes']={}
    for splitname,sp in [('query_split',splits),('query_and_current_heldout',held)]:
        res={};te=sp[2]
        for fname,xx in [('residual',x)]+list(controls.items()):
            if splitname!='query_split' and fname not in ('residual','current_embedding','target_embedding','current_target_embedding_concat'):continue
            entry={}
            for task,y in [('successor',ys),('useful',yu),('native_next',yn)]:
                pred,ridge=fit(xx,y,sp);layers=[]
                for l in range(pred.shape[1]):
                    ss=pred[:,l];mte=te[mixed[te]]
                    metric=ranking(y[te],ss[te])
                    if task=='useful':metric['within_successors']=ranking(y[mte],ss[mte],ys[mte].astype(bool))
                    if task=='native_next':
                        metric['mixed_top1_exact']=float((ss[mte].argmax(1)==m['next'][mte]).mean()) if len(mte) else None
                    layers.append(metric)
                entry[task]=dict(ridge=ridge,layers=layers)
            res[fname]=entry;print(setting,splitname,fname,'done',flush=True)
        out['probes'][splitname]=dict(split_counts=list(map(len,sp)),mixed_test=int(mixed[te].sum()),features=res)
    # Label-shuffle sanity control preserves each original split's marginals.
    shuffled=yu.copy();rng=np.random.default_rng(181)
    for ids in splits:shuffled[ids]=shuffled[rng.permutation(ids)]
    p,_=fit(x,shuffled,splits);te=splits[2]
    out['shuffled_useful_control']=[ranking(shuffled[te],p[te,l]) for l in range(7)]
    write(HERE/f'{setting}.json',out)
    print(setting,'COMPLETE',flush=True)

if __name__=='__main__':
    HERE.mkdir(parents=True,exist_ok=True)
    for setting in sys.argv[1:] or ['simple','hard']:analyze(setting)

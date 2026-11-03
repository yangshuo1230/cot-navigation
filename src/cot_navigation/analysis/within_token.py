"""Before-emission candidate legality and reusable within-token interventions.

The read position is immediately BEFORE the token being predicted. Error labels
describe a layer's provisional node, never a token already consumed in the CoT.
"""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .error_awareness import ParallelProbe


def candidate_slots(case,graph,tok,include_closure=False):
    """Legal-prefix slots requiring another unprinted, unvisited successor.

    Any remaining successor is valid; no canonical traversal order is imposed.
    A premature separator at such a position is invalid. The first natural
    erroneous token can be a target, but is NEVER included in its input prefix.
    """
    fields=[[],[],[]];field=0;trace=case['trace'];tau=case.get('tau')
    for i,token in enumerate(trace):
        if tau is not None and i>tau:break
        if field==2 and fields[0]:
            allowed=set(graph.get(fields[0][-1],[]))-set(fields[1])-set(fields[2])
            if allowed or include_closure:
                yield dict(pos=i,allowed=sorted(allowed) if allowed else [tok.id('|')],requires_node=bool(allowed),depth=len(fields[0]),
                    visited=len(fields[1]),printed=len(fields[2]),natural_onset=i==tau)
        if tau is not None and i==tau:break
        if token==tok.id(';'):fields=[[],[],[]];field=0
        elif token==tok.id('|'):field+=1
        elif field<3:fields[field].append(token)


def cached_tail(model,cache,h,positions,layer):
    """Recompute only blocks AFTER layer at the SAME current input position.

    Earlier prefix KV entries are immutable. Each used layer rewrites its own
    current-position KV before attention, so the same cache can be reused for
    many counterfactuals without cloning. Never reuse it for a different prefix
    or current position. layer is zero-based, h is post-block residual [B,D].
    """
    x=h[:,None]
    for j in range(layer+1,model.n_layers):
        x,_=model.blocks[j](x,positions,cache=cache,cache_mode='decode',layer_idx=j)
    return model.lm_head(model.final_norm(x[:,0]))


def lens_logits(model,lens,h,layer,kind):
    if kind=='calibrated' and layer<model.n_layers-1:
        h=h@lens['a'][layer]+lens['b'][layer]
    return model.lm_head(model.final_norm(h))


def fit_lens(model,train_h,train_logits,val_h,val_logits,steps=2000,seed=0):
    """Independent affine maps, distilled ONLY from the frozen final logits.

    Correct-node labels are not used. Each layer selects its own validation KL;
    the last layer is always the exact unmodified model output head.
    """
    torch.manual_seed(seed);device=next(model.parameters()).device
    x=torch.as_tensor(train_h,device=device);vx=torch.as_tensor(val_h,device=device)
    targets=torch.as_tensor(train_logits,device=device).softmax(-1)
    vt=torch.as_tensor(val_logits,device=device).softmax(-1)
    layers=x.shape[1]-1;dim=x.shape[2]
    a=nn.Parameter(torch.eye(dim,device=device).expand(layers,-1,-1).clone())
    b=nn.Parameter(torch.zeros(layers,dim,device=device))
    optimizer=torch.optim.AdamW([a,b],lr=.001,weight_decay=.0001)
    def logits(h):return model.lm_head(model.final_norm(torch.einsum('nld,ldk->nlk',h[:,:layers],a)+b))
    @torch.no_grad()
    def validation():
        sums=torch.zeros(layers,device=device)
        for start in range(0,len(vx),256):
            z=logits(vx[start:start+256]).log_softmax(-1)
            sums+=F.kl_div(z,vt[start:start+256,None].expand_as(z),reduction='none').sum((0,2))
        return sums/len(vx)
    best=validation();initial=best.clone();ba=a.detach().clone();bb=b.detach().clone();chosen=np.zeros(layers,int)
    rng=np.random.default_rng(seed)
    for step in range(1,steps+1):
        ids=torch.as_tensor(rng.integers(len(x),size=min(256,len(x))),device=device)
        z=logits(x[ids]).log_softmax(-1)
        loss=F.kl_div(z,targets[ids,None].expand_as(z),reduction='none').sum(-1).mean()
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        if step%100==0 or step==steps:
            v=validation();better=v<best;best=torch.minimum(best,v)
            ba[better]=a.detach()[better];bb[better]=b.detach()[better];chosen[better.cpu().numpy()]=step
    return dict(a=ba,b=bb,steps=chosen.tolist(),validation_kl=best.cpu().tolist(),identity_kl=initial.cpu().tolist())


def fit_binary_heads(train,val,hidden=0,steps=5000,seed=0):
    """Independent balanced binary heads; labels AND masks differ by layer.

    Inputs are [N,H,D], targets/masks [N,H]. No head shares weights or selection.
    Training/validation support is counted in distinct original queries.
    """
    torch.manual_seed(seed);device='cuda:0';rng=np.random.default_rng(seed)
    xx=train['x'];heads=xx.shape[1];valid=[];supports=[]
    for head in range(heads):
        counts=[]
        for data in [train,val]:
            counts.append([len(np.unique(data['query'][data['mask'][:,head]&(data['y'][:,head]==label)])) for label in [0,1]])
        supports.append(counts);valid.append(min(counts[0])>=30 and min(counts[1])>=8)
    active=torch.tensor(valid,device=device,dtype=torch.float32)
    if not any(valid):raise ValueError('No head has enough natural legal and illegal provisional-node queries')
    mean=torch.as_tensor(xx.mean(0),device=device);scale=torch.as_tensor(xx.std(0).clip(.05),device=device)
    def pack(d):
        x=(torch.as_tensor(d['x'],device=device)-mean)/scale
        y=torch.as_tensor(d['y'],device=device,dtype=torch.float32)
        mask=torch.as_tensor(d['mask'],device=device,dtype=torch.float32)
        freq=(y*mask).sum(0)/mask.sum(0).clamp_min(1)
        weights=torch.where(y.bool(),.5/freq.clamp_min(.0001),.5/(1-freq).clamp_min(.0001))*mask*active
        return x,y,weights
    x,y,w=pack(train);vx,vy,vw=pack(val)
    net=ParallelProbe(heads,xx.shape[-1],hidden).to(device)
    optimizer=torch.optim.AdamW(net.parameters(),lr=.001 if hidden else .01,weight_decay=.01)
    best=torch.full((heads,),float('inf'),device=device);best_state=copy.deepcopy(net.state_dict());chosen=np.zeros(heads,int)
    for step in range(1,steps+1):
        net.train();ids=torch.as_tensor(rng.integers(len(x),size=min(512,len(x))),device=device)
        loss=(F.binary_cross_entropy_with_logits(net(x[ids]),y[ids],reduction='none')*w[ids]).sum()/w[ids].sum().clamp_min(1)
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        if step%100==0 or step==steps:
            net.eval()
            with torch.no_grad():
                sums=torch.zeros(heads,device=device)
                for start in range(0,len(vx),256):
                    sums+=(F.binary_cross_entropy_with_logits(net(vx[start:start+256]),vy[start:start+256],reduction='none')*vw[start:start+256]).sum(0)
                v=sums/vw.sum(0).clamp_min(1);better=(v<best)&active.bool();best=torch.minimum(best,torch.where(active.bool(),v,best))
                for k,t in net.state_dict().items():best_state[k][better]=t[better]
                chosen[better.cpu().numpy()]=step
    net.load_state_dict(best_state);net.eval().requires_grad_(False)
    return dict(net=net,mean=mean,scale=scale,valid=valid,supports=supports,steps=chosen.tolist())


@torch.no_grad()
def predict_binary(fitted,x):
    device=fitted['mean'].device;out=[]
    for start in range(0,len(x),512):
        batch=(torch.as_tensor(x[start:start+512],device=device)-fitted['mean'])/fitted['scale']
        out.append(fitted['net'](batch).cpu().numpy())
    return np.concatenate(out)


def binary_accuracy(y,z,queries):
    y=np.asarray(y,bool);pred=np.asarray(z)>=0
    if not len(y) or not y.any() or y.all():return None
    tpr=float(pred[y].mean());tnr=float((~pred[~y]).mean())
    return dict(accuracy=float((pred==y).mean()),balanced_accuracy=(tpr+tnr)/2,
        tpr=tpr,tnr=tnr,threshold=.5,n=len(y),positive=int(y.sum()),negative=int((~y).sum()),
        queries=len(np.unique(queries)),majority_accuracy=max(float(y.mean()),1-float(y.mean())))


@torch.no_grad()
def protected_pair(model,lens,h,layer,kind,signal_delta,random_delta,node_lo,node_count):
    """Joint backtracking preserves BOTH provisional full-token and node argmax.

    Both paired perturbations receive the same multiplier, including zero if
    needed. No sample is discarded; final-layer effects are then zero by design.
    """
    base=lens_logits(model,lens,h,layer,kind)
    token=base.argmax(-1);node=base[:,node_lo:node_lo+node_count].argmax(-1)
    factor=torch.zeros(len(h),device=h.device);pending=torch.ones(len(h),device=h.device,dtype=torch.bool)
    for amount in [1.,.5,.25,.125,.0625,.03125,.015625]:
        ok=torch.ones_like(pending)
        for delta in [signal_delta,random_delta]:
            z=lens_logits(model,lens,h+amount*delta,layer,kind)
            ok&=(z.argmax(-1)==token)&(z[:,node_lo:node_lo+node_count].argmax(-1)==node)
        accepted=pending&ok;factor[accepted]=amount;pending&=~accepted
    return h+factor[:,None]*signal_delta,h+factor[:,None]*random_delta,factor

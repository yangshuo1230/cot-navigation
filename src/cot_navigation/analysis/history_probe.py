"""Indexed token-position probes built on the existing parallel probe heads.

Input position p consumes token p and predicts p+1. Targets describe only the
consumed prefix, including p. All standardization uses probe-training rows.
"""
import copy
import numpy as np
import torch
from torch.nn import functional as F
from .error_awareness import ParallelProbe
from .set_metrics import auc_ap


def history_target(positions, first_error):
    return (np.asarray(positions) >= first_error).astype(np.float32)


def score(y, z):
    y = np.asarray(y, bool); z = np.asarray(z, float)
    p = 1 / (1 + np.exp(-np.clip(z, -60, 60)))
    result = dict(n=len(y), positive=int(y.sum()), **auc_ap(y, z))
    result.update(bce=float(np.mean(np.logaddexp(0, z)-y*z)),
                  brier=float(np.mean((p-y)**2)),
                  positive_p=float(p[y].mean()) if y.any() else None,
                  negative_p=float(p[~y].mean()) if (~y).any() else None,
                  tpr=float((z[y]>=0).mean()) if y.any() else None,
                  fpr=float((z[~y]>=0).mean()) if (~y).any() else None)
    return result


def paired_after_error(data):
    """Keep both members of each event/cell only after its error is observable.

    Mixing earlier positions (both labels zero) with post-error positions
    (one positive, one negative) creates an onset/field shortcut even under
    branch-label shuffling. Binary error discrimination must use matched pairs.
    """
    if 'after' not in data:return data
    keys=np.stack([data['event'],data['cell']],axis=1)
    _,inverse=np.unique(keys,axis=0,return_inverse=True)
    counts=np.bincount(inverse,weights=data['after'].astype(int))
    keep=data['after'] & (counts[inverse]==2)
    return {k:(v[keep] if isinstance(v,np.ndarray) and v.ndim>0 and len(v)==len(keep) else v)
            for k,v in data.items()}


def fit_positions(train, val, steps=5000, seed=0, device='cuda:0', hidden=0):
    """Independent classifiers for EVERY (record, token offset, layer).

    Sparse positions share no fitted weights; each cell chooses its own best
    validation BCE. Cells lacking both labels / sufficient queries are omitted.
    This reuses ParallelProbe's parameterization and AdamW, without creating
    thousands of Python estimators or saving activation/checkpoint caches.
    """
    torch.manual_seed(seed)
    train=paired_after_error(train);val=paired_after_error(val)
    keys = sorted(set(train['cell'].tolist()) & set(val['cell'].tolist()))
    valid = []
    for key in keys:
        tr = train['cell']==key; va = val['cell']==key
        if (len(np.unique(train['query'][tr & (train['y']==1)]))>=30
                and len(np.unique(val['query'][va & (val['y']==1)]))>=8
                and len(np.unique(train['y'][tr]))==2 and len(np.unique(val['y'][va]))==2):
            valid.append(int(key))
    mapping = {v:i for i,v in enumerate(valid)}
    if not valid: return None
    def pack(data):
        take = np.isin(data['cell'], valid)
        return (torch.as_tensor(data['x'][take], device=device),
                torch.as_tensor(data['y'][take], device=device),
                torch.tensor([mapping[int(c)] for c in data['cell'][take]], device=device),
                data['query'][take])
    x,y,g,queries = pack(train); vx,vy,vg,_ = pack(val)
    heads=x.shape[1]; dim=x.shape[2]
    mean=torch.zeros((len(valid),heads,dim),device=device)
    square=torch.zeros_like(mean)
    count=torch.bincount(g,minlength=len(valid)).float()[:,None,None]
    mean.index_add_(0,g,x);square.index_add_(0,g,x*x)
    mean=mean/count
    scale=(square/count-mean*mean).clamp_min(.0025).sqrt()
    x=(x-mean[g])/scale[g];vx=(vx-mean[vg])/scale[vg]
    probe=ParallelProbe(len(valid)*heads,dim,hidden).to(device)
    optimizer=torch.optim.AdamW(probe.parameters(),lr=.001 if hidden else .01,weight_decay=.01)
    best=torch.full((len(valid)*heads,),float('inf'),device=device)
    best_state=copy.deepcopy(probe.state_dict()); chosen=np.zeros(len(best),int)
    head_ids=torch.arange(heads,device=device)
    def forward(xx,gg):
        ids=gg[:,None]*heads+head_ids[None]
        h=(xx[...,None]*probe.w[ids]).sum(-2)+probe.b[ids]
        if not hidden:return h[...,0]
        return (probe.dropout(F.gelu(h))*probe.v[ids]).sum(-1)+probe.c[ids]
    rng=np.random.default_rng(seed)
    for step in range(1,steps+1):
        probe.train()
        idx=torch.as_tensor(rng.integers(len(x),size=min(512,len(x))),device=device)
        loss=F.binary_cross_entropy_with_logits(forward(x[idx],g[idx]),y[idx,None].expand(-1,heads))
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        if step%100==0 or step==steps:
            probe.eval()
            with torch.no_grad():
                sums=torch.zeros_like(best); counts=torch.zeros_like(best)
                for begin in range(0,len(vx),256):
                    gg=vg[begin:begin+256]
                    ids=(gg[:,None]*heads+head_ids[None]).flatten()
                    losses=F.binary_cross_entropy_with_logits(forward(vx[begin:begin+256],gg),
                        vy[begin:begin+256,None].expand(-1,heads),reduction='none')
                    sums.scatter_add_(0,ids,losses.flatten());counts.scatter_add_(0,ids,torch.ones_like(losses).flatten())
                bce=sums/counts.clamp_min(1);better=bce<best;best=torch.minimum(best,bce)
                current=probe.state_dict()
                for k in best_state:best_state[k][better]=current[k][better]
                chosen[better.cpu().numpy()]=step
    probe.load_state_dict(best_state);probe.eval()
    return dict(probe=probe,mean=mean,scale=scale,mapping=mapping,heads=heads,
                selected_steps=chosen.tolist(),validation_bce=best.cpu().tolist(),hidden=hidden)


def position_scores(fitted, data):
    if fitted is None:return []
    data=paired_after_error(data)
    probe=fitted['probe'];device=fitted['mean'].device;heads=fitted['heads']
    output=[]
    for cell,group in fitted['mapping'].items():
        take=data['cell']==cell
        if not take.any():continue
        y=data['y'][take]
        if len(np.unique(data['query'][take]))<8 or len(np.unique(y))<2:continue
        with torch.no_grad():
            x=torch.as_tensor(data['x'][take],device=device)
            x=(x-fitted['mean'][group])/fitted['scale'][group]
            lo=group*heads;hi=lo+heads
            h=torch.einsum('nld,ldk->nlk',x,probe.w[lo:hi])+probe.b[lo:hi]
            z=(F.gelu(h)*probe.v[lo:hi]).sum(-1)+probe.c[lo:hi] if fitted['hidden'] else h[...,0]
            z=z.cpu().numpy()
        for layer in range(heads):
            pair_values={}
            if 'event' in data:
                for event,truth,zz in zip(data['event'][take],y,z[:,layer]):
                    pair_values.setdefault(int(event),{})[int(truth)]=float(zz)
            margins=[v[1]-v[0] for v in pair_values.values() if 0 in v and 1 in v]
            output.append(dict(cell=int(cell),record=int(cell//10000),token_offset=int(cell%10000),
                               head=layer,queries=len(np.unique(data['query'][take])),
                               paired_rank=float(np.mean([float(m>0)+.5*float(m==0) for m in margins])) if margins else None,
                               **score(y,z[:,layer])))
    return output

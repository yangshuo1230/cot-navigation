"""Independent layerwise categorical/set probes, with query-held-out selection.

No navigation gradients or artifacts are retained. Set decisions use logit >= 0
(probability .5); validation loss selects each layer independently.
"""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class LayerReadout(nn.Module):
    def __init__(self, layers, dim, outputs, hidden=0):
        super().__init__()
        width = hidden or outputs
        self.w = nn.Parameter(torch.randn(layers, dim, width) / dim**.5)
        self.b = nn.Parameter(torch.zeros(layers, width))
        self.hidden = hidden
        if hidden:
            self.v = nn.Parameter(torch.randn(layers, hidden, outputs) / hidden**.5)
            self.c = nn.Parameter(torch.zeros(layers, outputs))

    def forward(self, x):
        z = torch.einsum('nld,ldk->nlk', x, self.w) + self.b
        return torch.einsum('nlk,lko->nlo', F.gelu(z), self.v) + self.c if self.hidden else z


def fit_readout(train, val, *, outputs, categorical=False, hidden=0, steps=1500, seed=0):
    """Inputs x[N,L,D], y[N] or y[N,K], optional graph-node mask[N,K]."""
    torch.manual_seed(seed)
    device = 'cuda:0'
    mean = torch.as_tensor(train['x'].mean(0), device=device)
    scale = torch.as_tensor(train['x'].std(0).clip(.05), device=device)
    def pack(d):
        x = (torch.as_tensor(d['x'], device=device)-mean)/scale
        y = torch.as_tensor(d['y'], device=device, dtype=torch.long if categorical else torch.float32)
        mask = None if categorical else torch.as_tensor(d['mask'], device=device, dtype=torch.float32)
        return x, y, mask
    x, y, mask = pack(train); vx, vy, vm = pack(val)
    net = LayerReadout(x.shape[1], x.shape[2], outputs, hidden).to(device)
    if categorical:
        counts = torch.bincount(y, minlength=outputs).float().clamp_min(1)
        weight = len(y)/counts/outputs
    else:
        pos = (y*mask).sum().clamp_min(1); neg = ((1-y)*mask).sum().clamp_min(1)
        weight = neg/pos
    def losses(z, target, m):
        if categorical:
            yy = target[:, None].expand(-1, z.shape[1])
            loss = F.cross_entropy(z.flatten(0, 1), yy.flatten(), weight=weight, reduction='none')
            return loss.reshape(z.shape[:2]).mean(0)
        loss = F.binary_cross_entropy_with_logits(z, target[:, None].expand_as(z), pos_weight=weight, reduction='none')
        return (loss*m[:, None]).sum((0, 2))/m.sum().clamp_min(1)
    opt = torch.optim.AdamW(net.parameters(), lr=.001 if hidden else .005, weight_decay=.01)
    best = torch.full((x.shape[1],), float('inf'), device=device)
    state = copy.deepcopy(net.state_dict()); chosen = np.zeros(x.shape[1], int)
    rng = np.random.default_rng(seed)
    for step in range(1, steps+1):
        ids = torch.as_tensor(rng.integers(len(x), size=min(256, len(x))), device=device)
        loss = losses(net(x[ids]), y[ids], None if categorical else mask[ids]).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 100 == 0 or step == steps:
            with torch.no_grad():
                parts=[]
                for start in range(0, len(vx), 256):
                    end=min(len(vx), start+256)
                    parts.append(losses(net(vx[start:end]), vy[start:end], None if categorical else vm[start:end])*(end-start))
                score=sum(parts)/len(vx); better=score<best; best=torch.minimum(best,score)
                for key, value in net.state_dict().items(): state[key][better]=value[better]
                chosen[better.cpu().numpy()]=step
    net.load_state_dict(state); net.eval().requires_grad_(False)
    return dict(net=net,mean=mean,scale=scale,steps=chosen.tolist(),validation_loss=best.cpu().tolist())


@torch.no_grad()
def evaluate_readout(fitted, data, categorical=False):
    zs=[]
    for start in range(0,len(data['x']),256):
        x=torch.as_tensor(data['x'][start:start+256],device=fitted['mean'].device)
        zs.append(fitted['net']((x-fitted['mean'])/fitted['scale']).cpu().numpy())
    z=np.concatenate(zs);y=data['y'];result=[]
    for layer in range(z.shape[1]):
        if categorical:
            pred=z[:,layer].argmax(-1);classes=np.unique(y)
            item=dict(accuracy=float((pred==y).mean()),balanced_accuracy=float(np.mean([(pred[y==c]==c).mean() for c in classes])),
                      majority_accuracy=float(np.bincount(y).max()/len(y)),class_counts=np.bincount(y).tolist())
        else:
            m=data['mask'].astype(bool);yy=y.astype(bool);pred=z[:,layer]>=0
            tp=int((pred&yy&m).sum());tn=int((~pred&~yy&m).sum())
            fp=int((pred&~yy&m).sum());fn=int((~pred&yy&m).sum())
            item=dict(accuracy=(tp+tn)/max(1,tp+tn+fp+fn),balanced_accuracy=.5*(tp/max(1,tp+fn)+tn/max(1,tn+fp)),
                      f1=2*tp/max(1,2*tp+fp+fn),exact_set_accuracy=float(((pred==yy)|~m).all(-1).mean()),
                      tp=tp,tn=tn,fp=fp,fn=fn,positive_probability=float((1/(1+np.exp(-np.clip(z[:,layer],-60,60))))[yy&m].mean()) if (yy&m).any() else None,
                      negative_probability=float((1/(1+np.exp(-np.clip(z[:,layer],-60,60))))[~yy&m].mean()) if (~yy&m).any() else None)
        result.append(dict(head=layer,n=len(y),**item))
    return result

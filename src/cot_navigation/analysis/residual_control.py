"""Bounded, one-position residual steering with norm-matched null controls.

All edits occur after a transformer block during prefill. Higher layers rebuild
their KV entries from the edited state; decode is unhooked. No token is changed.
Axes and target class means must be estimated on probe-training data only.
"""
from contextlib import contextmanager
import numpy as np
import torch


def probe_axis(fitted,train,cell,head):
    """Convert a standardized linear probe's weights to a raw residual axis."""
    if fitted['hidden']:raise ValueError('Use a linear probe to define the intervention axis')
    group=fitted['mapping'][cell];index=group*fitted['heads']+head
    direction=(fitted['probe'].w[index,:,0]/fitted['scale'][group,head]).detach().cpu().numpy()
    direction=direction/np.linalg.norm(direction)
    take=train['cell']==cell;x=train['x'][take,head];y=train['y'][take]
    projected=x@direction
    means=[float(projected[y==label].mean()) for label in [0,1]]
    assert means[1]>means[0],'Probe axis must point toward the error class on training data'
    return dict(direction=direction,means=means,projection_std=float(projected.std()),
                train_queries=[len(np.unique(train['query'][take][y==label])) for label in [0,1]])


def orthogonal_axis(direction,seed):
    rng=np.random.default_rng(seed);axis=rng.normal(size=len(direction))
    axis-=np.dot(axis,direction)*direction
    return (axis/np.linalg.norm(axis)).astype(np.float32)


def steer(h,direction,target,*,dose=1.,random_direction=None,max_relative_norm=.2):
    """Target one coordinate; random null receives the identical signed norm.

    Clipping limits every perturbation to 20% of the recipient residual norm.
    A class-mean target does not claim to erase all nonlinear error information.
    """
    amount=dose*(target-(h*direction).sum(-1))
    cap=max_relative_norm*h.norm(dim=-1)
    amount=amount.clamp(-cap,cap)
    axis=direction if random_direction is None else random_direction
    return h+amount[...,None]*axis


@contextmanager
def residual_edits(model,edits):
    """edits: layer index -> list of (row, position, axis, target, dose, null)."""
    handles=[]
    try:
        for layer,jobs in edits.items():
            if not jobs:continue
            rows=torch.tensor([j[0] for j in jobs],device=model.embed.weight.device)
            positions=torch.tensor([j[1] for j in jobs],device=rows.device)
            axis=torch.as_tensor(np.stack([j[2] for j in jobs]),device=rows.device,dtype=torch.float32)
            target=torch.tensor([j[3] for j in jobs],device=rows.device)
            dose=torch.tensor([j[4] for j in jobs],device=rows.device)
            null=torch.as_tensor(np.stack([j[5] if j[5] is not None else j[2] for j in jobs]),device=rows.device,dtype=torch.float32)
            def hook(_module,_args,output,rows=rows,positions=positions,axis=axis,target=target,dose=dose,null=null):
                x,weights=output;x=x.clone()
                x[rows,positions]=steer(x[rows,positions].float(),axis,target,dose=dose,random_direction=null).to(x.dtype)
                return x,weights
            handles.append(model.blocks[layer].register_forward_hook(hook))
        yield
    finally:
        for handle in handles:handle.remove()

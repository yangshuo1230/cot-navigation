"""Regularized affine readouts with validation-only hyperparameter selection.

Inputs are [examples, layers, dimensions]. No frozen-model optimizer exists.
Ridge classification uses class-balanced least squares and a validation-only
threshold. Scores are NOT calibrated probabilities. All fitting runs on GPU 0.
"""
import numpy as np
import torch

RIDGES=(.001,.01,.1,1.,10.)


def auc(y,s):
    y=np.asarray(y).astype(bool);s=np.asarray(s)
    p=int(y.sum());n=len(y)-p
    if not p or not n:return None
    order=np.argsort(s,kind='stable');sorted_s=s[order];ranks=np.empty(len(s),float)
    start=0
    while start<len(s):
        end=start+1
        while end<len(s) and sorted_s[end]==sorted_s[start]:end+=1
        ranks[order[start:end]]=(start+1+end)/2
        start=end
    return float((ranks[y].sum()-p*(p+1)/2)/(p*n))


def classification(y,s,threshold=0):
    y=np.asarray(y).astype(int);s=np.asarray(s);pred=(s>=threshold).astype(int)
    if not len(y):return dict(n=0,accuracy=None,balanced_accuracy=None,auc=None)
    yes=y==1;no=~yes
    yp=float((pred[yes]==1).mean()) if yes.any() else None
    np_=float((pred[no]==0).mean()) if no.any() else None
    return dict(n=len(y),n_positive=int(yes.sum()),accuracy=float((pred==y).mean()),
                yes_recall=yp,no_recall=np_,balanced_accuracy=(yp+np_)/2 if yp is not None and np_ is not None else None,
                auc=auc(y,s))


def regression(y,p):
    y=np.asarray(y);p=np.asarray(p)
    mse=float(np.mean((y-p)**2));var=float(np.var(y))
    return dict(n=len(y),r2=1-mse/var if var>1e-10 else None,
                mae=float(np.mean(abs(y-p))),rmse=float(mse**.5),
                rounded_accuracy=float((np.rint(p)==y).mean()))


def fit(x,y,split,*,binary=False,seed=0):
    """Return full-data predictions, one validation-selected lambda per head/target."""
    x=np.asarray(x,dtype=np.float32)
    if x.ndim==2:x=x[:,None,:]
    y=np.asarray(y,dtype=np.float32)
    if y.ndim==1:y=y[:,None]
    tr,va,te=[np.asarray(i,dtype=int) for i in split]
    if min(map(len,(tr,va,te)))<2:raise ValueError('insufficient disjoint probe split')
    device='cuda:0'
    xt=torch.as_tensor(x[tr],device=device).transpose(0,1)
    mean=xt.mean(1,keepdim=True);scale=xt.std(1,keepdim=True).clamp(min=.05)
    allx=torch.as_tensor(x,device=device).transpose(0,1)
    z=(allx-mean)/scale
    z=torch.cat((z,torch.ones(z.shape[0],z.shape[1],1,device=device)),dim=-1)
    yt=torch.as_tensor(y[tr],device=device);center=yt.mean(0);ys=yt.std(0).clamp(min=1)
    if binary:
        target=yt*2-1;positive=yt[:,0]>.5
        weights=torch.where(positive,.5/positive.sum(),.5/(~positive).sum())*len(tr)
        center=torch.zeros_like(center);ys=torch.ones_like(ys)
    else:
        target=(yt-center)/ys;weights=torch.ones(len(tr),device=device)
    zt=z[:,tr];gram=torch.einsum('hnd,n,hne->hde',zt,weights,zt)/len(tr)
    rhs=torch.einsum('hnd,n,nk->hdk',zt,weights,target)/len(tr)
    eye=torch.eye(z.shape[-1],device=device);eye[-1,-1]=0
    best=np.full((z.shape[0],y.shape[1]),np.inf)
    chosen=np.zeros_like(best);predictions=np.zeros((len(y),z.shape[0],y.shape[1]),np.float32)
    thresholds=np.zeros_like(best)
    for penalty in RIDGES:
        w=torch.linalg.solve(gram+penalty*eye+1e-6*torch.eye(z.shape[-1],device=device),rhs)
        pred=(torch.einsum('hnd,hdk->nhk',z,w)*ys+center).cpu().numpy()
        for h in range(z.shape[0]):
            for k in range(y.shape[1]):
                threshold=0.
                if binary:
                    v=pred[va,h,k];vy=y[va,k]
                    if len(np.unique(vy))<2:raise ValueError('binary validation missing a class')
                    candidates=np.unique(np.r_[0,np.quantile(v,np.linspace(0,1,31))])
                    threshold=max(candidates,key=lambda t:(classification(vy,v,t)['balanced_accuracy'],-abs(t)))
                    score=auc(vy,v)
                    loss=-(score if score is not None else .5)
                else:loss=float(np.mean((pred[va,h,k]-y[va,k])**2))
                # Prefer stronger regularization on exact ties.
                if loss<=best[h,k]+1e-10:
                    best[h,k]=loss;chosen[h,k]=penalty;thresholds[h,k]=threshold
                    predictions[:,h,k]=pred[:,h,k]
    return predictions,dict(ridge=chosen.tolist(),threshold=thresholds.tolist(),
        selection='validation AUROC; validation balanced-accuracy threshold' if binary else 'validation MSE',
        n_train=len(tr),n_val=len(va),n_test=len(te))

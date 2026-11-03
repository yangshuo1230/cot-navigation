"""Small nonlinear readout robustness check, with equally sized input controls.

One hidden GELU layer (32 units), three seeds. Train on query-train only;
early stopping on validation AUROC, validation-only BA threshold. All training
is decoder-only GPU0; frozen main-model inference CPU. No saved weights.
"""
import run as study
import numpy as np
import torch
from torch import nn
import copy

def fit(x,y,split,seed):
    tr,va,te=split;torch.manual_seed(seed)
    z=torch.tensor(x,device='cuda:0');yy=torch.tensor(y,device='cuda:0',dtype=torch.float32)
    mean=z[tr].mean(0);std=z[tr].std(0).clamp_min(.05);z=(z-mean)/std
    model=nn.Sequential(nn.Linear(z.shape[1],32),nn.GELU(),nn.Linear(32,1)).to('cuda:0')
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.001)
    positive=yy[tr].sum();negative=len(tr)-positive
    weights=torch.where(yy[tr]>.5,.5/positive,.5/negative)*len(tr)
    best=-np.inf;best_state=None;best_step=0;stale=0
    for step in range(1,601):
        optimizer.zero_grad(set_to_none=True)
        logits=model(z[tr])[:,0]
        loss=(nn.functional.binary_cross_entropy_with_logits(logits,yy[tr],reduction='none')*weights).mean()
        loss.backward();optimizer.step()
        if step%5:continue
        with torch.no_grad():pred=model(z[va])[:,0].cpu().numpy()
        value=study.linear.auc(y[va],pred)
        if value>best+1e-6:
            best=value;best_step=step;best_state=copy.deepcopy(model.state_dict());stale=0
        else:stale+=5
        if stale>=100:break
    model.load_state_dict(best_state)
    with torch.no_grad():pred=model(z)[:,0].cpu().numpy()
    candidates=np.unique(np.r_[0,np.quantile(pred[va],np.linspace(0,1,31))])
    threshold=max(candidates,key=lambda t:(study.linear.classification(y[va],pred[va],t)['balanced_accuracy'],-abs(t)))
    result=dict(seed=seed,hidden=32,best_step=best_step,validation_auc=best,threshold=float(threshold),
        test=study.linear.classification(y[te],pred[te],threshold))
    del model,optimizer,z;torch.cuda.empty_cache()
    return result

def run(setting):
    pack=study.base.load(setting,'path','cpu');features,_,_,_=study.collect(pack)
    m={key:np.array([r[key] for r in pack['rows']]) for key in ('source','target','split','label')}
    n=len(pack['nodes']);embedding=pack['model'].embed.weight.detach().numpy()[pack['nodes']]
    inputs={key:features[key][:,-1] for key in ('prompt_separator','prompt_joint')}
    inputs['endpoint_embeddings']=np.concatenate([embedding[m['source']],embedding[m['target']]],1)
    # Match the endpoint-embedding control. One random vector per pair would
    # be a query-identity table, so use two fixed endpoint vectors instead.
    for vector_seed in (201,303,407):
        vectors=np.random.default_rng(vector_seed).normal(size=(n,128)).astype(np.float32)
        inputs[f'random_endpoints_{vector_seed}']=np.concatenate([vectors[m['source']],vectors[m['target']]],1)
    split=[np.flatnonzero(m['split']==i) for i in range(3)]
    result={}
    for name,x in inputs.items():
        result[name]=[fit(x,m['label'],split,seed) for seed in (11,23,37)]
        print(setting,name,[round(v['test']['balanced_accuracy'],4) for v in result[name]],flush=True)
    target=study.OUT/f'{setting}.json';out=study.json.loads(target.read_text())
    out['nonlinear_readouts']=result
    out['nonlinear_note']=('One hidden GELU layer32; balanced BCE; train-only scaling, val-AUC early stop, val-BA threshold. '
        'Three initialization seeds. Query holdout only. Adds external nonlinear computation; not proof the original model has already computed its answer. '
        'Feature dimensions differ: separator128, endpoint pair256, joint384; compare as information-access diagnostics, not equal parameter-count tests.')
    study.base.write(target,out)

if __name__=='__main__':
    for setting in ('simple','hard'):run(setting)

"""Direct answer readouts at E and after every attention/FFN residual.

Reuse original checkpoints, query splits, BF16 inference and the existing
validation-selected balanced ridge classifier. Frozen inference GPU1; fits
GPU0. No activation, probe-weight, model-checkpoint or W&B output is saved.
"""
from pathlib import Path
import os, sys, json, csv, importlib.util
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import torch
torch.set_num_threads(2)
ROOT=Path.cwd();HERE=ROOT/'studies/direct_exposure_probes'
spec=importlib.util.spec_from_file_location('storage',ROOT/'studies/fixed_graph_storage/study.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
import linear

@torch.inference_mode()
def run(setting):
    p=base.load(setting,'direct','cpu');model=p['model'].to('cuda:1');tok=p['tok']
    rows=p['rows'];ids=torch.tensor([p['header']+[tok.id('<QUERY>'),int(p['nodes'][r['source']]),
        int(p['nodes'][r['target']]),tok.id('|')] for r in rows],device='cuda:1')
    observed={};handles=[]
    def before(index):
        def hook(module,args):observed[index,'input']=args[0]
        return hook
    def attention(index):
        def hook(module,args,output):observed[index,'attention']=observed[index,'input']+output[0]
        return hook
    # Verify hooks observe exactly the native BF16 computation.
    with torch.autocast('cuda',dtype=torch.bfloat16):reference=model(ids[:64])
    for i,block in enumerate(model.blocks):
        handles.extend([block.register_forward_pre_hook(before(i)),block.attn.register_forward_hook(attention(i))])
    features=[];max_error=0
    for start in range(0,len(ids),64):
        tokens=ids[start:start+64];cap={}
        with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(tokens,capture=cap)
        if start==0:max_error=float((reference-logits).abs().max());assert max_error==0
        states=[model.embed(tokens)[:, -1]]
        for i in range(6):states.extend([observed[i,'attention'][:,-1],cap['resid'][i][:,-1]])
        features.append(torch.stack(states,1).float().cpu().numpy())
    for handle in handles:handle.remove()
    x=np.concatenate(features);y=np.array([r['label'] for r in rows]);partition=np.array([r['split'] for r in rows])
    split=[np.flatnonzero(partition==i) for i in range(3)]
    pred,fit=linear.fit(x,y,split,binary=True)
    names=['E']+[f'L{i}_{stage}' for i in range(1,7) for stage in ('attention','FFN')]
    metrics=[dict(stage=name,**linear.classification(y[split[2]],pred[split[2],i,0],fit['threshold'][i][0])) for i,name in enumerate(names)]
    shuffled=y.copy();rng=np.random.default_rng(181)
    for subset in split:shuffled[subset]=rng.permutation(shuffled[subset])
    control,control_fit=linear.fit(x,shuffled,split,binary=True)
    controls=[dict(stage=name,**linear.classification(shuffled[split[2]],control[split[2],i,0],control_fit['threshold'][i][0])) for i,name in enumerate(names)]
    with (ROOT/'studies/fixed100_layer_probes/answer_readouts.csv').open() as stream:
        old={r['layer']:r for r in csv.DictReader(stream) if r['model']==setting+'_direct' and r['phase']=='prompt'}
    comparisons=[]
    for i in range(7):
        current=metrics[2*i];previous=old['E' if i==0 else f'L{i}']
        comparisons.append(dict(layer=i,accuracy_delta=current['accuracy']-float(previous['accuracy']),
            balanced_accuracy_delta=current['balanced_accuracy']-float(previous['balanced_accuracy'])))
    result=dict(manifest=p['manifest'],position='QUERY prompt separator |; no generated tokens or answers',
        method='Class-balanced affine ridge independently at each stage. Original query train/val/test only; no node holdout. '
               'Training-only feature scaling; val AUROC selects ridge; val BA selects threshold. '
               'Attention/FFN are residual AFTER addition, not isolated sublayer outputs.',
        layers=names,metrics=metrics,shuffled_control=controls,fit=fit,
        verification=dict(observation_hook_max_logit_error=max_error,previous_FFN_readout_comparisons=comparisons),
        inference='native GPU1 BF16 as previous study; residual stored in-memory FP32',probe_fit='GPU0')
    base.write(HERE/f'{setting}_query_sublayers.json',result)
    print(setting,[(v['stage'],round(v['balanced_accuracy'],4)) for v in metrics],flush=True)
    del model,p,x;torch.cuda.empty_cache()

if __name__=='__main__':
    for setting in ('simple','hard'):run(setting)

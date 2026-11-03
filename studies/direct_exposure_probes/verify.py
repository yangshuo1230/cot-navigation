"""Small, meaningful checks for multi-task batching and metric controls.

No smoke training/checkpoints. Compare batched regression with the previous
single-task implementation and verify oracle/frequency metrics on a toy DAG.
"""
import importlib.util
from pathlib import Path
import numpy as np
import run as current

spec=importlib.util.spec_from_file_location('previous_fit',Path.cwd()/'studies/path_direction/analyze.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
rng=np.random.default_rng(120)
x=rng.normal(size=(60,3,5)).astype(np.float32)
y=rng.normal(size=(60,4,12)).astype(np.float32)
split=[np.arange(36),np.arange(36,48),np.arange(48,60)]
p,penalty=current.fit(x,y,split);errors=[]
for task in range(4):
    old,old_penalty=previous.fit(x,y[:,task],split)
    errors.append(float(np.max(np.abs(p[:,:,task]-old))))
    assert errors[-1]<1e-4
    assert np.array_equal(np.array(penalty)[:,task],old_penalty)

n=12
world=dict(n_nodes=n,adj=[[v for v in range(u+1,n) if rng.random()<.3] for u in range(n)])
a,mats=current.graph(world)
records=np.array([(s,t) for s in range(n) for t in range(n) if s!=t])
focal,other=records.T
for i,task in enumerate(current.TASKS):
    labels=mats[i,focal];oracle=labels[:,None]
    m=current.evaluate(labels,oracle,np.arange(len(labels)),focal,other,i in (1,3),[.5])[0]
    assert m['auc']==1 and m['f1']==1 and m['exact_set']==1,m
    frequency=np.broadcast_to(labels.mean(0),(len(labels),1,n)).copy()
    m=current.evaluate(labels,frequency,np.arange(len(labels)),focal,other,i in (1,3),[.5])[0]
    assert m['auc']==.5,m
current.write(current.HERE/'verification.json',dict(batched_vs_previous_max_errors=errors,
    all_selected_regularizers_match=True,toy_DAG_BFS_matches=True,oracle_AUC_F1_exact_set=1,
    constant_candidate_frequency_AUC=.5,no_smoke_training=True))
print('VERIFIED',errors)

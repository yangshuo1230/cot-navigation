"""Disposable CUDA integration check: no formal run, tracking or checkpoint."""
from pathlib import Path
import json
import os
import random
import sys

ROOT=Path(str(next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'pyproject.toml').is_file() and (_p / 'studies').is_dir())))
HERE=ROOT/'studies/fixed_adjacency_suite'
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['WANDB_MODE']='disabled'
sys.path.insert(0,str(HERE))
from train_suite import load_runtime,Samples,new_model,update,balanced,evaluate_adjacency
load_runtime(ROOT)
import torch
torch.set_num_threads(2)
world=json.loads((HERE/'data/graph_0.json').read_text())
samples=Samples(world,'direct')
job=dict(model_seed=0,dropout=.1,weight_decay=.1,lr=1e-4)
device=torch.device('cuda:0')
raw,model,opt,loss_fn=new_model(job,device)
for i in range(5):
    rows=balanced(world['one_hop']['train'],64,random.Random(42+i))
    knowledge=samples.knowledge_batch(random.Random(43+i),'R0')
    assert all(p[0]=='adj' for p in knowledge)
    assert sum(len(samples.knowledge(p).walk_indices) for p in knowledge)==64
    values,norm,_,_=update(raw,model,opt,loss_fn,samples,rows,knowledge,device,1e-4)
    assert torch.isfinite(values[0]) and torch.isfinite(norm)
evaluation=evaluate_adjacency(raw,samples,device)
assert evaluation['n_sources']==160 and evaluation['empty_sources']==29
report=dict(status='PASS',cuda_updates=5,adjacency_sources=160,
    empty_sources=evaluation['empty_sources'],checkpoint_saved=False,wandb_run_created=False,
    formal_training_started=False)
(HERE/'gpu_verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))

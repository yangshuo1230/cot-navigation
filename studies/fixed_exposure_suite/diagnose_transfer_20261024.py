"""Frozen diagnostics on an idle GPU; do not alter live training or checkpoints."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(str(next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'pyproject.toml').is_file() and (_p / 'studies').is_dir())))
HERE=ROOT/'studies/fixed_exposure_suite'
OUT=ROOT/'outputs/exposure_grokking_v2'
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['WANDB_MODE']='disabled'
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'src'))
import torch
from train_suite import Samples,evaluate,new_model
torch.set_num_threads(2)
world=json.loads((HERE/'data/graph_0.json').read_text())
samples=Samples(world,'direct')
device=torch.device('cuda:0')
results=[]
for folder in sorted((OUT/'grokking').iterdir()):
    checkpoint=folder/'latest.pt'
    # Read exactly one atomic checkpoint version into RAM before inference.
    import io
    content=checkpoint.read_bytes()
    ckpt_hash=hashlib.sha256(content).hexdigest()
    blob=torch.load(io.BytesIO(content),map_location='cpu',weights_only=False)
    raw,unused,opt,loss=new_model(blob['job'],device,compile_model=False)
    raw.load_state_dict(blob['model']);raw.eval()
    train=evaluate(raw,samples,world['one_hop']['train'],device)
    monitor=evaluate(raw,samples,world['one_hop']['monitor'],device)
    active=[u for u,vs in enumerate(world['adj']) if vs]
    prompts=torch.tensor([[samples.tok.id('<WALK>'),samples.nodes[u]] for u in active],device=device)
    raw.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        probabilities=raw(prompts)[:,-1].float().softmax(-1).cpu()
    edge_hits=0;total_edges=0;exact=0;mass=[];by_source={}
    for u,prob in zip(active,probabilities):
        true={samples.nodes[v] for v in world['adj'][u]}
        picked=set(prob.topk(len(true)).indices.tolist())
        hits=len(picked&true)
        edge_hits+=hits;total_edges+=len(true);exact+=picked==true
        mass.append(float(prob[list(true)].sum()))
        by_source[u]=picked
    monitor_yes=[r for r in world['one_hop']['monitor'] if r[2]]
    known=sum(samples.nodes[r[1]] in by_source[r[0]] for r in monitor_yes)
    results.append(dict(id=folder.name,checkpoint_step=blob['step'],phase=blob['phase'],
        checkpoint_sha256=ckpt_hash,
        full_train={k:v for k,v in train.items() if k!='predictions'},
        monitor={k:v for k,v in monitor.items() if k!='predictions'},
        knowledge=dict(nonempty_sources=len(active),edges=total_edges,
            recall_at_true_outdegree=edge_hits/total_edges,
            exact_successor_set_rate=exact/len(active),
            mean_true_successor_probability_mass=sum(mass)/len(mass),
            monitor_positive_edges_retrieved=known,monitor_positive_edges=len(monitor_yes))))
    del raw,unused,opt,blob
    torch.cuda.empty_cache()
report=dict(frozen_inference_only=True,training_modified=False,gpu=0,results=results)
(HERE/'transfer_diagnostic_20261024.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))

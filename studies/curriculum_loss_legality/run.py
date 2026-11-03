"""Paired frozen replay of lambda=1/4/16, validation-best and final 200k.

Only reads completed checkpoints; no optimizer, W&B or rollout bulk caches.
Reuses the prior tested semantic oracle and identical per-gap/label TEST IDs.
"""
from pathlib import Path
import argparse
import hashlib
import sys
import torch

ROOT=Path.cwd()
sys.path.insert(0,str(ROOT/'studies/curriculum_trace_errors'))
import audit

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--lambda',dest='lam',type=int,choices=[1,4,16],required=True)
    p.add_argument('--per-cell',type=int,default=8)
    p.add_argument('--batch',type=int,default=64)
    args=p.parse_args()
    torch.set_num_threads(4)
    audit.OUT=ROOT/'studies/curriculum_loss_legality'
    folder=ROOT/f'outputs/curriculum_loss_swa_v1/launches/curriculum_dfs_loss_swa_s0/context/dfs_state/full_l{args.lam}'
    for name,file in [('best','best'),('last','latest')]:
        ck=folder/'checkpoints'/(file+'.pt')
        entry={'id':f'lambda{args.lam}_{name}','run':str(folder),'kind':'context',
               'protocol':'dfs_state','scale':'curriculum','lambda':args.lam,
               'checkpoint_name':file,'checkpoint_label':name,
               'checkpoint_sha256':hashlib.sha256(ck.read_bytes()).hexdigest()}
        audit.run(entry,args.per_cell,args.batch)

if __name__=='__main__':main()

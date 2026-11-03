"""Fill missing batch16 DFS full-trace metrics for scale/shape plots.

Only frozen existing checkpoints and prepared test queries are used. Each
case writes compact audit metrics, and verifies answers against the saved
original batch16 predictions. Independent cases/arms can use separate GPUs
without changing batch shapes or model numerics.
"""
import argparse,os,runpy
from pathlib import Path
os.environ['WANDB_MODE']='disabled'
os.environ['AUDIT_BATCH']='16'
os.environ['AUDIT_CAPTURE']='0'
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
P=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--case',required=True,choices=['n64','deep','wide','larger'])
parser.add_argument('--gpu',type=int,default=0)
parser.add_argument('--arm',nargs='+',choices=['clean','error'],default=['clean','error'])
args=parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
D=runpy.run_path(str(P.parent/'accuracy_source/run.py'))
import torch
torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.15)
out=P/'task_comparison/batch16_audits';out.mkdir(parents=True,exist_ok=True)
D['run'].__globals__['OUT']=out
for arm in args.arm:
    D['run']('dfs_'+arm,[args.case])
D['write'](out/('finished_'+args.case+'_'+'_'.join(args.arm)+'.json'),dict(finished=True,case=args.case,arms=args.arm))

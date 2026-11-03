"""Ephemeral updates through the production pipeline; no W&B/checkpoint/run."""
from pathlib import Path
import os,sys,argparse,json,time,statistics,gc
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--kind',choices=['single','serial'],required=True);args=p.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu);sys.path.insert(0,str(HERE/'src'))
import torch
from cot_navigation.config import load_config
from runtime import PrefixTrainer
torch.set_num_threads(2)
manifest=json.loads((HERE/'manifest.json').read_text())
def build(protocol):
    j=next(j for j in manifest['jobs'] if j['seed']==0 and j['protocol']==protocol and j['recipe']=='standard' and j['mask']=='query' and j['order']==args.kind)
    cfg=load_config([j['config']],['tracking.mode=disabled','train.persist_outputs=false','runtime.smoke=true','runtime.cuda_device=0'])
    t=PrefixTrainer(cfg,j);t.loss_fn=torch.compile(t.loss_fn,dynamic=True,fullgraph=True)
    t._evaluate=lambda *a,**k:None;t._save=lambda:None
    return t
def measure(t,name,completed,mainstep=None):
    t.completed=completed
    if mainstep is not None:t.query_sampler.step=mainstep
    # Never use validation/test data to benchmark optimization.
    values=[]
    for i in range(18):
        torch.cuda.synchronize();start=time.perf_counter();m=t._update(collect_metrics=True);torch.cuda.synchronize()
        elapsed=time.perf_counter()-start;t.completed+=1
        if i>=8:values.append(elapsed)
    r=dict(name=name,median_update_s=statistics.median(values),min_update_s=min(values),max_update_s=max(values),
           median_basis_updates=len(values),last_metrics={k:m.get(k) for k in ['update_time','step_time','data_time','microbatches','gpu_mem_mb','answer_loss','trace_loss']})
    print(json.dumps(r),flush=True);return r
report=[]
if args.kind=='single':
    direct=build('direct');report.append(measure(direct,'direct_boot_standard_m24',0))
    # Exercises the real optimizer-reset boundary, including a checkpoint reload
    # round-trip of samplers/RNG in memory (no checkpoint is written).
    payload=direct._payload();assert not payload['prefix_state']['optimizer_reset']
    direct.completed=550000;direct._update(False);assert direct.optimizer_reset
    direct.tracker.finish();del direct;gc.collect();torch.cuda.empty_cache();torch._dynamo.reset()
t=build('path')
for name,point in [('simple16',0),('simple100',120000),('hard16',160000),('hard100',280000)]:
    report.append(measure(t,name,point,point))
t.tracker.finish()
(HERE/f'benchmark_{args.kind}.json').write_text(json.dumps(dict(status='PASS',kind=args.kind,gpu=args.gpu,measurements=report,formal_runs_created=0),indent=2))

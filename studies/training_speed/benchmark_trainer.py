"""Compare original/current Trainer._update on identical real cached batches.

No persisted training or W&B runs; both scratch optimizers stay in memory.
Includes collation, H2D, objective/backward, clipping, optimizer, and metric
reduction. Does not include query rendering/augmentation or validation.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import statistics
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu',type=int,default=7)
    parser.add_argument('--protocol',choices=['dfs_state','bfs_state'],default='bfs_state')
    args=parser.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu);os.environ['WANDB_MODE']='disabled'
    import numpy as np
    import torch
    from cot_navigation.analysis.scale_followup import make_adapter
    from cot_navigation.training.trainer import Trainer
    from cot_navigation.training.loss import combine_losses
    from cot_navigation.training.schedule import WarmupCosine
    torch.set_num_threads(2)
    root=Path.cwd();folder=root/'studies/training_speed'
    spec=importlib.util.spec_from_file_location('original_trainer',folder/'original_trainer.py')
    original=importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name]=original;spec.loader.exec_module(original)
    am=make_adapter(dict(run=f'outputs/hard_dag_v1/launches/hard_dag_v1_s0/context/curriculum/{args.protocol}'),root,'cuda:0')
    examples=am.examples('train',64);lookup={e.index:e.row for e in examples}
    initial={k:v.detach().clone() for k,v in am.model.state_dict().items()}
    output=[]
    for name,cls in [('before',original.Trainer),('after',Trainer)]:
        trainer=object.__new__(cls);trainer.cfg=am.cfg
        trainer.cfg.train.max_tokens_per_microbatch=16384 if name=='before' else 65536
        trainer.cfg.train.max_padding_fraction=1. if name=='before' else .30
        trainer.device=am.inference_device;trainer.tok=am.tok
        am.model.load_state_dict(initial);trainer.raw=am.model.requires_grad_(True).train()
        trainer.model=torch.compile(trainer.raw,dynamic=True)
        trainer.loss_fn=combine_losses if name=='before' else torch.compile(combine_losses,dynamic=True,fullgraph=True)
        trainer.opt=torch.optim.AdamW(trainer.raw.parameters(),lr=1e-4,fused=True)
        trainer.schedule=WarmupCosine(trainer.opt,100,200000,1e-3,1e-4)
        trainer.completed=150001;trainer.scale_members=[];trainer.walk_sampler=None
        trainer.store=am.store;trainer.loss_balance_ema=None
        trainer._sync_scale_weights=lambda:None
        trainer.train_ids=np.array([e.index for e in examples])
        from types import SimpleNamespace
        trainer.query_sampler=SimpleNamespace(next=lambda:np.arange(64))
        trainer._training_task=lambda index:lookup[index]
        trainer.queries_seen=trainer.walks_seen=trainer.tokens_seen=0
        for collect in (False,True):
            for _ in range(5):trainer._update(collect_metrics=collect)
            torch.cuda.synchronize();timings=[]
            for _ in range(30):
                torch.cuda.synchronize();start=time.perf_counter()
                result=trainer._update(collect_metrics=collect)
                torch.cuda.synchronize();timings.append(time.perf_counter()-start)
            row=dict(version=name,collect_metrics=collect,median_s=statistics.median(timings),mean_s=statistics.mean(timings))
            if collect:row['metrics']=result
            output.append(row);print('TRAINER',args.protocol,row,flush=True)
        del trainer
    (folder/f'trainer_{args.protocol}.json').write_text(json.dumps(dict(query_ids=[e.index for e in examples],results=output),indent=2)+'\n')


if __name__=='__main__':main()

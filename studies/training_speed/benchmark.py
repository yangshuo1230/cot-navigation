"""Short, disposable training benchmark on real held training rows.

No Trainer/Tracker, checkpoints, or W&B runs. Model/optimizer changes exist
only in RAM. All variants use the same 64 examples and logical objective.
"""
import argparse
import json
import itertools
import os
from pathlib import Path
import statistics
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--gpu',type=int,default=7)
    p.add_argument('--protocol',choices=['dfs_state','bfs_state'],default='bfs_state')
    p.add_argument('--budgets',nargs='+',type=int,default=[16384,65536,131072,262144])
    p.add_argument('--padding-fractions',nargs='+',type=float,default=[1.0])
    p.add_argument('--repeats',type=int,default=20)
    p.add_argument('--tag',default='before')
    p.add_argument('--compile-loss',action='store_true')
    args=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu);os.environ['WANDB_MODE']='disabled'
    import torch
    from cot_navigation.analysis.scale_followup import make_adapter
    from cot_navigation.data.dataset import collate_examples
    from cot_navigation.training.trainer import _chunk
    from cot_navigation.training.loss import combine_losses
    torch.set_num_threads(2)
    root=Path.cwd();folder=root/'studies/training_speed'
    run=f'outputs/hard_dag_v1/launches/hard_dag_v1_s0/context/curriculum/{args.protocol}'
    entry=dict(id=args.protocol,run=run)
    am=make_adapter(entry,root,'cuda:0')
    am.model.requires_grad_(True).train()
    examples=am.examples('train',64);rows=[e.row for e in examples]
    model=torch.compile(am.model,dynamic=True)
    loss_fn=torch.compile(combine_losses,dynamic=True,fullgraph=True) if args.compile_loss else combine_losses
    opt=torch.optim.AdamW(am.model.parameters(),lr=1e-4,fused=True)
    results=[]
    for budget,padding in itertools.product(args.budgets,args.padding_fractions):
        chunks=_chunk(rows,0,budget,64,padding)
        arrays=[collate_examples(c,am.tok.pad,64) for c in chunks]
        prepared=[{k:torch.from_numpy(v) for k,v in a.items() if k in ['tokens','answer_mask','trace_mask','walk_mask']} for a in arrays]
        def step():
            opt.zero_grad(set_to_none=True)
            for batch in prepared:
                device={k:v.pin_memory().to('cuda',non_blocking=True) for k,v in batch.items()}
                with torch.autocast('cuda',dtype=torch.bfloat16): logits=model(device['tokens'])
                parts=loss_fn(logits,device['tokens'],device['answer_mask'],device['trace_mask'],device['walk_mask'],64,0)
                parts['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(am.model.parameters(),1.)
            opt.step()
        warm=time.perf_counter()
        for _ in range(5):step()
        torch.cuda.synchronize();warm=time.perf_counter()-warm
        torch.cuda.reset_peak_memory_stats();timings=[]
        for _ in range(args.repeats):
            torch.cuda.synchronize();start=time.perf_counter();step();torch.cuda.synchronize()
            timings.append(time.perf_counter()-start)
        real=sum(len(r.tokens) for r in rows);padded=sum(a['tokens'].size for a in arrays)
        item=dict(budget=budget,max_padding_fraction=padding,microbatches=len(chunks),median_s=statistics.median(timings),mean_s=statistics.mean(timings),
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            effective_tokens=real,padding_fraction=1-real/padded,warmup_s=warm,tokens_per_s=real/statistics.median(timings))
        results.append(item);print('RESULT',args.tag,args.protocol,item,flush=True)
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'{args.tag}_{args.protocol}.json'
    path.write_text(json.dumps(dict(settings=vars(args),query_ids=[e.index for e in examples],
        lengths=[len(r.tokens) for r in rows],results=results),indent=2)+'\n')


if __name__=='__main__':main()

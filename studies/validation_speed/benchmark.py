"""Frozen real-query generation benchmark; no training/W&B/checkpoints."""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--gpu',type=int,default=7)
    p.add_argument('--protocol',default='dfs_state')
    p.add_argument('--batch',nargs='+',type=int,default=[16,32,64])
    p.add_argument('--count',type=int,default=64)
    p.add_argument('--cap',type=int,default=512)
    p.add_argument('--compiled',action='store_true')
    p.add_argument('--tag',default='before')
    args=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu);os.environ['WANDB_MODE']='disabled'
    import torch
    from cot_navigation.analysis.scale_followup import make_adapter
    from cot_navigation.evaluation.rollout import generate
    torch.set_num_threads(2)
    root=Path.cwd();folder=root/'studies/validation_speed';folder.mkdir(parents=True,exist_ok=True)
    am=make_adapter(dict(run=f'outputs/hard_dag_v1/launches/hard_dag_v1_s0/context/curriculum/{args.protocol}'),root,'cuda:0')
    examples=am.examples('val',args.count);prompts=[e.prompt for e in examples]
    model=torch.compile(am.model,dynamic=True,fullgraph=True) if args.compiled else am.model
    print('WARM',args.protocol,args.compiled,flush=True)
    generate(model,prompts[:4],am.protocol,am.tok,'bf16',max_new_tokens=8,cot_margin=am.cfg.protocol.cot_margin)
    torch.cuda.synchronize();results=[]
    for batch in args.batch:
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();outs=[]
        for i in range(0,len(prompts),batch):
            outs.extend(generate(model,prompts[i:i+batch],am.protocol,am.tok,'bf16',max_new_tokens=args.cap,
                                 cot_margin=am.cfg.protocol.cot_margin))
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        item=dict(batch=batch,seconds=elapsed,generated_tokens=sum(len(o['generated']) for o in outs),
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,outputs=outs)
        results.append(item);print('RESULT',args.tag,args.protocol,{k:v for k,v in item.items() if k!='outputs'},flush=True)
    (folder/f'{args.tag}_{args.protocol}.json').write_text(json.dumps(dict(settings=vars(args),
        query_ids=[e.index for e in examples],results=results),indent=2)+'\n')


if __name__=='__main__':main()

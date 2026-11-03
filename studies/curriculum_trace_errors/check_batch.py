"""Diagnose BF16 batching sensitivity on historical evaluation batches.

Select up to one batch per predefined model/condition, maximizing the number
of discrepant overlapping predictions. This is a diagnostic, not a rate sample.
Never replace selected rows in the main analysis with these reruns.
"""
from pathlib import Path
import gzip
import hashlib
import json
import os
import sys

ROOT=Path.cwd()
OUT=ROOT/'studies/curriculum_trace_errors'
sys.path.insert(0,str(OUT))
from audit import *

CASES=[('context_curriculum_path','n100'),('context_curriculum_dfs_state','deep'),
       ('context_curriculum_bfs_state','deep'),('context_curriculum_bfs_state','n100'),
       ('context_curriculum_dfs_nodes','n24')]

def main():
    torch.set_num_threads(4)
    results=[]
    for model_id,case in CASES:
        dest=OUT/'models'/model_id
        metadata=json.loads((dest/(case+'.json')).read_text())
        entry=metadata['model']
        folder=ROOT/entry['run']
        cfg=load_config([folder/'resolved_config.yaml'])
        tok=Tokenizer.from_config(cfg)
        source=ContextStore(metadata['dataset']);a=source.arrays
        with gzip.open(dest/(case+'.jsonl.gz'),'rt') as f: panel={r['query']:r for r in map(json.loads,f)}
        old_path=folder/'eval'/('predictions_test_best'+('' if case=='n100' else '-'+case)+'.jsonl')
        original=[json.loads(line) for line in old_path.read_text().splitlines()]
        chunks=[original[i:i+16] for i in range(0,len(original),16)]
        chunk=max(chunks,key=lambda cc:sum(r['query_id'] in panel and r['pred']!=panel[r['query_id']]['pred'] for r in cc))
        model=build_model(cfg.model,len(tok),tok.pad).cuda().eval().requires_grad_(False)
        checkpoint=folder/'checkpoints/best.pt';assert sha(checkpoint)==entry['checkpoint_sha256']
        blob=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model.load_state_dict(blob['model'],strict=True);del blob
        prompts=[]
        for old in chunk:
            idx=int(old['query_id']);nodes=source.node_vocab_ids(idx,tok)
            edges=list(zip(a['edge_src'][idx].astype(int).tolist(),a['edge_dst'][idx].astype(int).tolist()))
            prompts.append(context_prompt(tok,nodes,edges,int(a['s_topo'][idx]),int(a['t_topo'][idx])))
        outputs=generate(model,prompts,entry['protocol'],tok,cfg.train.precision,
                         max_new_tokens=cfg.evaluation.max_new_tokens,cot_margin=cfg.protocol.cot_margin)
        comparisons=[]
        for old,out in zip(chunk,outputs):
            q=int(old['query_id']);p=panel.get(q)
            comparisons.append({'query':q,'historical_pred':old['pred'],'rerun_original_batch16_pred':out['pred'],
                'panel_batch64_pred':p['pred'] if p else None,'in_panel':p is not None,
                'original_reproduced':old['pred']==out['pred'],'label':old['label'],
                'trace_sha256':hashlib.sha256(json.dumps(out['trace_tokens']).encode()).hexdigest()})
        results.append({'model':model_id,'case':case,'checkpoint_sha256':sha(checkpoint),
                        'original_batch16_reproduced':sum(r['original_reproduced'] for r in comparisons),
                        'n':len(comparisons),'comparisons':comparisons})
        write(OUT/'batch_sensitivity.json',{'selection':'one original batch per predefined case; maximize overlap mismatches; diagnostic only','results':results})
        print(model_id,case,results[-1]['original_batch16_reproduced'],len(comparisons),flush=True)
        del model,source;torch.cuda.empty_cache()

if __name__=='__main__':main()

"""Frozen candidate-only comparison on identical graphs and legal histories.
No training, W&B, checkpoint output, or whole-search inference.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['WANDB_MODE']='disabled'
from pathlib import Path
from collections import Counter
import sys,json,time
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from cot_navigation.config import load_config
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.protocols.dfs import dfs_search
from cot_navigation.protocols.bfs import bfs_search
from cot_navigation.protocols.render import render_steps
from cot_navigation.models.transformer import build_model
from cot_navigation.evaluation.rollout import generate
from cot_navigation.analysis.common import sha256
BASE=ROOT/'outputs/hard_dag_v1/launches/hard_dag_v1_s0/context/curriculum'
DUAL=ROOT/'outputs/dual_attention_v1/launches/dual_attention_v1_s0_20261023_r1'
models=[('baseline_dfs',BASE/'dfs_state'),('dual_dfs_parallel_causal',DUAL/'dfs_state/parallel/causal'),
        ('dual_dfs_serial_causal',DUAL/'dfs_state/serial/causal'),('baseline_bfs',BASE/'bfs_state'),
        ('dual_bfs_serial_bidirectional',DUAL/'bfs_state/serial/bidirectional')]
torch.set_num_threads(2);results=[]
for name,run in models:
    t0=time.monotonic();cfg=load_config([run/'resolved_config.yaml']);tok=Tokenizer.from_config(cfg)
    protocol=cfg.protocol.name;reference=DUAL/protocol/'serial/bidirectional/eval/predictions_test_last.jsonl'
    rows=[json.loads(x) for x in reference.read_text().splitlines()];cases=[]
    for row in rows:
        prompt=row['prompt_tokens'];qi=prompt.index(tok.id('<QUERY>'));adj=[[] for _ in range(cfg.data.node_vocab)]
        nodes=[tok.node_id(i) for i in range(cfg.data.node_vocab)]
        for at in range(0,qi,3):adj[prompt[at+1]-tok.node_lo].append(prompt[at+2]-tok.node_lo)
        src,dst=prompt[qi+1]-tok.node_lo,prompt[qi+2]-tok.node_lo
        steps=(bfs_search if protocol=='bfs_state' else dfs_search)(adj,src,dst)
        assert int(steps[-1].action=='FOUND')==row['label']
        # Test initial graph lookup and a later legal state independently.
        for stage,index in [('first',0),('later',4)]:
            if index>=len(steps):continue
            step=steps[index];cur=step.stack[0 if protocol=='bfs_state' else -1]
            neighbors={nodes[v] for v in adj[cur]};visited={nodes[v] for v in step.visited}
            expected=neighbors-visited;assert expected=={nodes[v] for v in step.cands}
            prefix=prompt+render_steps(steps[:index],nodes,tok)
            prefix += [nodes[v] for v in step.stack]+[tok.id('|')]+[nodes[v] for v in step.visited]+[tok.id('|')]
            cases.append(dict(query_id=row['query_id'],stage=stage,prefix=prefix,
                neighbors=neighbors,visited=visited,expected=expected))
    checkpoint=run/'checkpoints/latest.pt';blob=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=build_model(cfg.model,len(tok),tok.pad).cuda().eval();model.load_state_dict(blob['model']);model.requires_grad_(False)
    step=int(blob['step']);del blob;observations=[]
    # Common eager/BF16 backend and batches; generation capped only for this field probe.
    for at in range(0,len(cases),16):
        chunk=cases[at:at+16]
        outputs=generate(model,[x['prefix'] for x in chunk],protocol,tok,'bf16',max_new_tokens=24,cot_margin=cfg.protocol.cot_margin)
        for c,o in zip(chunk,outputs):
            generated=o['tokens'][len(c['prefix']):];closed=tok.id('|') in generated
            emitted=generated[:generated.index(tok.id('|'))] if closed else generated
            chosen=set(emitted);tp=len(chosen&c['expected'])
            observations.append(dict(query_id=c['query_id'],stage=c['stage'],closed=closed,
                exact=closed and chosen==c['expected'] and len(chosen)==len(emitted),
                eligible_nonempty=bool(c['expected']),neighbor_nonempty=bool(c['neighbors']),
                nonedge=bool(chosen-c['neighbors']),visited_included=bool(chosen&c['visited']),
                omitted=bool(c['expected']-chosen),filter_opportunity=bool(c['neighbors']&c['visited']),
                tp=tp,pred_count=len(emitted),gold_count=len(c['expected'])))
    aggregates={}
    for stage in ['first','later']:
        for subset in ['all','nonempty']:
            selected=[x for x in observations if x['stage']==stage and (subset=='all' or x['eligible_nonempty'])]
            if not selected:continue
            n=len(selected);filterable=[x for x in selected if x['filter_opportunity']]
            aggregates[stage+'_'+subset]={'n':n,**{key:sum(x[key] for x in selected)/n for key in ['exact','closed','nonedge','visited_included','omitted']},
                'micro_precision':sum(x['tp'] for x in selected)/max(1,sum(x['pred_count'] for x in selected)),
                'micro_recall':sum(x['tp'] for x in selected)/max(1,sum(x['gold_count'] for x in selected)),
                'filter_opportunity_n':len(filterable),
                'visited_included_when_filterable':sum(x['visited_included'] for x in filterable)/len(filterable) if filterable else None}
    result={'model':name,'run':str(run),'step':step,'checkpoint_sha256':sha256(checkpoint),
            'seconds':time.monotonic()-t0,'metrics':aggregates,'by_query':observations}
    results.append(result)
    Path(__file__).with_name('baseline_candidate_probe.json').write_text(json.dumps({'models':results,
        'complete':len(results)==len(models),'method':'Identical fixed main-test query IDs, edge order and gold histories per protocol. Candidate field only, eager BF16 batch16 cap24; no full rollouts. Later is record index4 and only included when reference has >=5 records.'},indent=2)+'\n')
    print(name,step,json.dumps(aggregates),'seconds',round(result['seconds'],1),flush=True)
    del model;torch.cuda.empty_cache()

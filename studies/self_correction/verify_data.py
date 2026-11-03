"""Audit paired masks and benchmark the data splice, without GPU/training."""
import json
from pathlib import Path
import time

import numpy as np

from cot_navigation.config import load_config
from cot_navigation.data.dataset import ContextStore
from cot_navigation.data.scale_prepared import ScalePreparedRows
from cot_navigation.data.mistakes import FIELDS, records, choose_plan, apply_plan
from cot_navigation.protocols.tokenizer import Tokenizer

configs=Path('configs/generated/self_correction/self_correction_s0')
study=Path('studies/self_correction')
examples=[];timings=[]
for protocol in ('dfs_state','bfs_state'):
    cfg=load_config([configs/f'context__{protocol}__p30_masked.yaml'])
    tok=Tokenizer.from_config(cfg)
    for root in (cfg.train.scale_roots[0],cfg.data.root):
        source=ContextStore(root);prepared=ScalePreparedRows(source,cfg,tok)
        ids=source.indices('train')[:256]
        pool=[prepared.row(int(i),0) for i in ids]
        plans=[]
        for i,row in enumerate(pool):
            recs=records(row,tok)
            plans.append([choose_plan(row,recs,tok,field,np.random.default_rng(i*7919+f),cfg)
                          for f,field in enumerate(FIELDS)])
        for mask in (True,False):
            # Includes both dataset sizes, both protocols and both error fields.
            for row, pair in zip(pool,plans):
                for plan in pair:
                    masked=apply_plan(row,plan,tok,cfg,mask_errors=True)
                    full=apply_plan(row,plan,tok,cfg,mask_errors=False)
                    assert masked.tokens==full.tokens
                    assert set(masked.trace_indices)<=set(full.trace_indices)
        for rate in (.15,.30):
            for mask in (True,False):
                rng=np.random.default_rng(739)
                exposure=12000;injected=0;extra=0
                start=time.perf_counter()
                for t in range(exposure):
                    j=t%len(pool);u=float(rng.random());f=int(rng.integers(2))
                    if u<rate:
                        out=apply_plan(pool[j],plans[j][f],tok,cfg,mask_errors=mask)
                        injected+=1;extra+=len(out.tokens)-len(pool[j].tokens)
                seconds=time.perf_counter()-start
                result=dict(protocol=protocol,source=root,rate=rate,mask_errors=mask,
                    exposures=exposure,observed_rate=injected/exposure,
                    mean_original_tokens=float(np.mean([len(row.tokens) for row in pool])),
                    mean_extra_tokens_per_exposure=extra/exposure,
                    milliseconds_per_64_queries=seconds/exposure*64000)
                timings.append(result)
                print(json.dumps(result),flush=True)
        if root==cfg.train.scale_roots[0]:
            row=pool[0]
            for f,field in enumerate(FIELDS):
                plan=plans[0][f];bad=apply_plan(row,plan,tok,cfg)
                a,b=map(int,plan[:2])
                examples.append(dict(protocol=protocol,field=field,query_id=int(row.meta['index']),
                    original_record=tok.decode(row.tokens[a:b]),
                    wrong_then_repair=tok.decode(bad.tokens[a:bad.meta['repair_end']]),
                    masked_tokens=tok.decode(bad.tokens[slice(*bad.meta['mistake_mask_span'])]),
                    error_kind=bad.meta['mistake_edit']))
(study/'data_benchmark.json').write_text(json.dumps(timings,indent=2)+'\n')
(study/'examples.json').write_text(json.dumps(examples,indent=2)+'\n')
print('Paired token/mask checks passed: 2048 events across two scales and protocols.',flush=True)

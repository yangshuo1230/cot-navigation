"""Read-only depth/checkpoint/feature compatibility checks; no generation."""
from pathlib import Path
import json
import os
import runpy
import torch
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
os.chdir(ROOT)
R=runpy.run_path(str(Path(__file__).with_name('run.py')))
manifest=json.loads((ROOT/'studies/correction_behavior/cohorts/all_positions_tokenmatched_20261103.json').read_text())
plan=json.loads((ROOT/manifest['output']/'plan.json').read_text())
assert len(plan['models'])==12 and len({e['checkpoint_sha256'] for e in plan['models']})==12
behavior=runpy.run_path(str(ROOT/'studies/correction_behavior/run.py'))
assert behavior['checkpoint_step']({'protocol_version':1},{'checkpoint_step':1})==200000
assert behavior['checkpoint_step']({'protocol_version':2},{'checkpoint_step':262370})==262370
torch.set_num_threads(2)
for depth,rate in [(6,0),(9,.6)]:
    entry=next(e for e in plan['models'] if e['layers']==depth and e['error_rate']==rate and e['seed']==0)
    am=R['Model'](entry);q=am.sample_ids('val',1)[0];example=am.example(q,'val')
    trace=example.row.tokens[len(example.prompt):len(example.prompt)+3]
    cases=[dict(query=q,split='val',trace=trace)]
    rr=[dict(sid=0,pos=p,y=0,query=q,event=0,kind='verification',field='frontier',phase=0,offset=p,cell=p,
        exact_token=True,positive_query=q,source=entry['id']) for p in range(2)]
    features=R['STATE']['extract'](am,cases,rr)
    assert features['x'].shape==(2,depth+2,am.model.d_model+2)
    features=R['NH']['extract'](am,cases,rr)
    assert features['x'].shape==(2,depth+2,am.model.d_model+8)
    embeddings=am.model.embed(torch.tensor(trace[:2],device=am.device)).detach().cpu().numpy()
    assert np.array_equal(features['x'][:,depth,:am.model.d_model],embeddings)
    assert am.checkpoint['step']==entry['checkpoint_step']
    assert 'all_positions_v2' in am.cfg.protocol.mistake_cache_root
    del am,features;torch.cuda.empty_cache()
print('PASS: 12 unique pinned checkpoints; v1/v2 steps; real 6/9-layer state/history heads and control indexing; no generation')

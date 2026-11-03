"""All-query split integrity and fast threshold equivalence checks."""
import json
from pathlib import Path
import numpy as np
import run as study

rng=np.random.default_rng(37)
for tied in [False,True]:
    q=np.repeat(np.arange(40),3);y=rng.integers(0,2,40)[q,None].astype(np.float32)
    z=rng.normal(size=(len(q),4,1)).astype(np.float32)
    if tied:z=np.round(z)
    data={'query':q,'y':y}
    fast=study.fast_thresholds(z,data);slow=study.o.thresholds(z,data)
    a=study.o.metrics(z,data,fast);b=study.o.metrics(z,data,slow)
    for h in range(4):assert abs(a['heads'][h]['balanced_accuracy']-b['heads'][h]['balanced_accuracy'])<1e-10
manifest=json.loads((study.o.V6/'manifest.json').read_text());meta=next(m for m in manifest['worlds'] if m['name']=='n80_g0')
bank=study.o.PackedBank(meta,'dfs_nodes',study.torch.device('cpu'))
splits,panels,coverage=study.make_split(bank.world)
assert all(v['train']>0 and v['val']>0 and v['test']>0 for v in coverage.values())
assert len(splits['test'])==sum(len(panels[p]) for p in ['short','AA','VA','AV','VV','TA','AT','TT'])
assert set(panels['original_query_seen']).isdisjoint(panels['original_query_unseen'])
check={'threshold_equivalence':'PASS','query_disjoint':'PASS','all_roles_covered':'PASS',
       'split_counts':{k:len(v) for k,v in splits.items()},'coverage':coverage}
root=Path(__file__).resolve().parent
if (root/'results.json').exists():
    d=json.loads((root/'results.json').read_text());assert d['query_ids']==splits
    prior=json.loads((study.ROOT/'studies/dfs_v6_state_probes/results.json').read_text())
    for name,m in d['models'].items():
        assert m['checkpoint_sha256']==prior['models'][name]['checkpoint_sha256']
        assert m['phases']['r100']['active'].get('status')
    check.update(models=len(d['models']),checkpoint_sha='PASS',endpoint_active_empty='PASS')
(root/'verification.json').write_text(json.dumps(check,indent=2))
print(check)

"""Check query weighting, threshold calibration, positions, and split integrity."""
import json
from pathlib import Path
import numpy as np
import run as study

assert study.locations(3)[:5]==[('r0',0),('r25',0),('r50',1),('r75',2),('r100',3)]
assert ('t4',4) not in study.locations(3)
data={'query':np.array([0,0,1,2]),'y':np.array([[1],[1],[0],[0]],np.float32)}
z=np.array([1,1,-1,-1],np.float32)[:,None,None]
cut=study.thresholds(z,data);m=study.metrics(z,data,cut)
assert m['heads'][0]['balanced_accuracy']==1 and m['n_queries']==3
# Duplicating a query's identical anchors must not alter query-macro accuracy.
bad=z.copy();bad[2]=2
score=study.metrics(bad,data,[0])['heads'][0]
assert score['balanced_accuracy']==.75
assert abs(score['accuracy']-2/3)<1e-10
root=Path(__file__).resolve().parent
check={'phase_positions':'PASS','query_macro_weighting':'PASS','validation_thresholds':'PASS'}
if (root/'results.json').exists():
    d=json.loads((root/'results.json').read_text())
    train=set(d['query_ids']['train']);val=set(d['query_ids']['val']);assert not train&val
    for panel in study.PANELS:assert not(set(d['query_ids'][panel])&(train|val))
    prior=json.loads((study.ROOT/'studies/dfs_v6_state_probes/results.json').read_text())
    for name,model in d['models'].items():
        assert model['checkpoint_sha256']==prior['models'][name]['checkpoint_sha256']
        assert model['phases']['r100']['active'].get('status')
        for p in model['phases'].values():
            for item in p.values():
                if 'status' in item:continue
                for source in ['reference','free']:
                    for panel in study.PANELS:
                        for population in item[source][panel].values():
                            assert population['n_queries']<=population['n_rows']
                            for h in population['heads']:
                                for value in h.values():assert value is None or 0<=value<=1
    check.update(models=len(d['models']),query_disjoint='PASS',checkpoint_sha='PASS',reference_endpoint_active_empty='PASS')
(root/'verification.json').write_text(json.dumps(check,indent=2))
print(check)

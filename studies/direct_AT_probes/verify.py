"""Independent primal solve checks batched dual fitting and label layout."""
import json
import numpy as np
from run import fit, OUT, ROOT, sets
import hashlib
import torch

rng=np.random.default_rng(718)
x=rng.normal(size=(80,2,128));y=rng.integers(0,2,size=(3,80,80)).astype(float)
tr=np.arange(24);va=np.arange(24,32);te=np.arange(32,40)
vp,tp,pen=fit(x,y,tr,va,te)
errors=[]
for h in range(2):
    z=(x[:,h]-x[tr,h].mean(0))/np.maximum(x[tr,h].std(0),.05)
    z-=z[tr].mean(0)
    for k in range(3):
        mean=y[k,tr].mean(0);target=y[k,tr]-mean;lam=pen[h][k]
        w=np.linalg.solve(z[tr].T@z[tr]/len(tr)+lam*np.eye(128),z[tr].T@target/len(tr))
        errors.append(float(abs(z[te]@w+mean-tp[:,h,k]).max()))
        errors.append(float(abs(z[va]@w+mean-vp[:,h,k]).max()))
assert max(errors)<1e-8,max(errors)
OUT.mkdir(exist_ok=True)
result={'independent_primal_vs_dual_max_error':max(errors),'label_layout_verified':True}
world=json.loads((ROOT/'studies/fixed_navigation_v5/data/n80_g0.json').read_text())
checked=0
for seed in range(4):
    output=OUT/f'seed{seed}.json'
    if not output.exists():continue
    d=json.loads(output.read_text())
    path=ROOT/f'outputs/fixed_navigation_protocols_v6/pilot/n80/g0_s{seed}/direct/latest.pt'
    blob=torch.load(path,map_location='cpu',weights_only=False)
    assert blob['job']['world_hash']==sets.digest(world)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==d['manifest']['checkpoint_sha256']
    for r in d['records']:
        train,val,test=set(r['train']),set(r['validation']),set(r['test_A']+r['test_T'])
        assert not(train&val or train&test or val&test)
        assert set(r['test_A'])<=set(world['node_groups']['A'])
        assert set(r['test_T'])<=set(world['node_groups']['T'])
    checked+=1
result['checkpoint_world_and_node_splits_verified_models']=checked
(OUT/'verification.json').write_text(json.dumps(result,indent=2))
print('PASS',max(errors))

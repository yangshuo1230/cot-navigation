"""Check exact counts, strict node folds, and rotation-invariant reconstruction."""
import json
import numpy as np
from exposure import counts
import run

state=dict(n_items=3,epoch=2,cursor=2,order=[2,0,1])
assert np.array_equal(counts(state),[3,2,3])
assert np.array_equal(counts(dict(sampler=state,pending=[0,2])),[2,2,2])
for seed in run.SPLITS:
    splits=run.node_split(seed)
    assert list(map(len,splits))==[60,20,20]
    assert len(set(np.concatenate(splits)))==100
assert np.allclose(run.rank_values(np.array([2,1,2,4])),[1.5,0,1.5,3])
rng=np.random.default_rng(104)
x=rng.normal(size=(100,4)).astype(np.float32)
y=(x@rng.normal(size=(4,8))+.1*rng.normal(size=(100,8))).astype(np.float32)
q=np.linalg.qr(rng.normal(size=(8,8)))[0].astype(np.float32)
split=run.node_split(17)
w,b,penalty=run.C.old.coefficient_fit(x[:,None],y,split)
rw,rb,rpenalty=run.C.old.coefficient_fit(x[:,None],y@q,split)
error=float(np.max(np.abs((x@w[0]+b[0])@q-(x@rw[0]+rb[0]))))
assert penalty==rpenalty and error<1e-4
actual={}
for setting in ('simple','hard'):
    pack=run.C.old.load(setting,device='cpu')
    attrs,audit=run.recover(pack)
    actual[setting]={k:audit[k] for k in ('query_actual_count','task_walk_actual_count','knowledge_walk_actual_count')}
    del pack
run.C.old.write(run.HERE/'verification.json',dict(epoch_prefix_counts=True,pending_buffer_corrected=True,
    node_folds_disjoint=True,tie_ranks_correct=True,reconstruction_rotation_error=error,
    reconstruction_regularizer_unchanged=True,actual_counts_and_stratum_membership=actual))
print('VERIFIED',error)

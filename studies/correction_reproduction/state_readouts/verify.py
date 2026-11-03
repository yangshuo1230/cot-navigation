"""Small correctness checks for set metrics and held-out probe fitting."""
import numpy as np
import torch
from cot_navigation.analysis.state_readout import fit_readout,evaluate_readout

torch.set_num_threads(2)
rng=np.random.default_rng(17)
def data(n):
    x=rng.normal(size=(n,2,8)).astype('float32')
    # A different recoverable label at each independent layer is not assumed:
    # same truth, distinct invertible coordinates in the second layer.
    x[:,1]=x[:,0,::-1]
    y=(x[:,0,:3]>0).astype('float32')
    return dict(x=x,y=y,mask=np.ones_like(y))
tr,va,te=data(800),data(200),data(200)
fit=fit_readout(tr,va,outputs=3,steps=500)
scores=evaluate_readout(fit,te)
assert min(s['balanced_accuracy'] for s in scores)>.94,scores
assert all(s['tp']+s['tn']+s['fp']+s['fn']==600 for s in scores)
# The metric must penalize false positives, even when every positive is found.
with torch.no_grad():
    fit['net'].w.zero_();fit['net'].b.fill_(20)
scores=evaluate_readout(fit,te)
assert all(abs(s['balanced_accuracy']-.5)<1e-8 for s in scores)
assert all(s['exact_set_accuracy']<.25 for s in scores)
for d in [tr,va,te]:d['y']=(d['x'][:,0,0]>0).astype('int64')
fit=fit_readout(tr,va,outputs=2,categorical=True,steps=500)
assert min(s['accuracy'] for s in evaluate_readout(fit,te,True))>.94
print('PASS: independent layers, held-out fitting, false-positive penalty, categorical accuracy')

# Test-stratum boundaries: an error late in a record excludes that WHOLE
# record; later correct positions must not be accidentally called early.
import runpy
from pathlib import Path
from types import SimpleNamespace
mod=runpy.run_path(str(Path(__file__).with_name('early_clean_lens.py')))
tok=SimpleNamespace(id=lambda s: {'|':2,';':3}[s])
trace=[10,2,10,2,11,2,4,11,3]*5
cases=[dict(trace=trace,first=dict(token=11)),dict(trace=trace,first=None)]
rows=[dict(sid=0,record=0,pos=2),dict(sid=0,record=1,pos=11),
      dict(sid=0,record=4,pos=38),dict(sid=1,record=4,pos=38)]
parts=mod['test_partitions'](SimpleNamespace(tok=tok),cases,rows)
assert parts['early_correct'].tolist()==[True,False,False,False]
assert parts['early_all'].tolist()==[True,True,False,False]
assert parts['later_correct'].tolist()==[False,False,False,True]
assert parts['correct_record'].tolist()==[True,False,False,True]
print('PASS: early/correct test partitions and whole-record error boundary')

leg=runpy.run_path(str(Path(__file__).with_name('legality_transitions.py')))
symbols={'|':2,';':3,'<PUSH>':4,'<POP>':5,'<FOUND>':6,'<EXHAUST>':7,'<MISTAKE>':8,'<STOP>':9}
tok=SimpleNamespace(id=lambda s:symbols[s],stoi=symbols,node_lo=10,n_node_tokens=16)
a,b,c=10,11,12;graph={a:[b,c],b:[],c:[]};prefix=[a,2,a,2]
legal=lambda p,t:leg['next_legal'](p,t,graph,a,b,tok)
assert legal(prefix,b) and legal(prefix,c)  # Both alternatives, not exact-match.
assert not legal(prefix,2)  # Candidate omission at premature closure.
assert not legal(prefix+[b],b)  # Duplicate.
assert legal(prefix+[b],c)
assert not legal(prefix+[b,c,2],2)  # Fourth separator, unfinished action.
assert legal(prefix+[b,c,2],symbols['<PUSH>'])
cell=leg['counts']([False,True,False,True],[True,False,False,True])
assert cell['illegal_legal']==cell['legal_illegal']==1
assert cell['illegal_to_legal_fraction']==cell['legal_to_illegal_fraction']==.25
assert cell['final_legal_given_l5_illegal']==cell['final_illegal_given_l5_legal']==.5
print('PASS: noncanonical semantic alternatives, omission, duplicate, action punctuation, paired denominators')

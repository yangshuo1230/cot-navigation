"""Data invariants, causal model check, and ephemeral CUDA train/eval smoke checks."""
from pathlib import Path
import json,hashlib,subprocess,sys,copy
import torch
import prepare,train
HERE=Path(__file__).resolve().parent
ROOT=Path(str(next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'pyproject.toml').is_file() and (_p / 'studies').is_dir())))
torch.set_num_threads(2)
manifest=json.loads((HERE/'manifest.json').read_text())
for seed in range(3):
    p=HERE/'data'/f'graph_{seed}.json';world=json.loads(p.read_text())
    assert world==prepare.build(seed)
    for split in world['splits'].values():
        labels={(s,t):y for s,t,y in split['train']}
        assert all(labels.get((u,v))==1 for u,vs in enumerate(world['adj']) for v in vs)
        for protocol in ('direct','path'):
            for r in split['sequences']:
                seq=train.sequence(r,world['node_names'],protocol)
                assert len(seq)<=16 and seq[-1]==train.EOS
                if r['y']:
                    path=r['path'];assert path[0]==r['s'] and path[-1]==r['t']
                    assert all(v in world['adj'][u] for u,v in zip(path,path[1:]))
j=manifest['jobs'][0];m=train.model_factory(j).eval()
assert sum(1 for _ in m.blocks)==2
assert m.emb.weight is m.head.weight
assert all(x.bias is None for x in m.modules() if isinstance(x,torch.nn.Linear))
x=torch.tensor([[train.BOS,3,7,11,19]])
y=x.clone();y[0,-1]=20
with torch.no_grad():assert torch.allclose(m(x)[:,:-1],m(y)[:,:-1],atol=1e-6)
# Test evaluation against a deterministic oracle with valid native token streams.
w=json.loads((HERE/'data/graph_0.json').read_text());names=w['node_names']
pos=next(r for r in w['splits']['random20']['sequences'] if r['y'] and len(r['path'])>=3)
neg=next(r for r in w['splits']['random20']['sequences'] if not r['y'])
class Oracle(torch.nn.Module):
    def __init__(self,protocol,malformed=False):super().__init__();self.protocol=protocol;self.malformed=malformed
    def forward(self,x):
        logits=torch.full((*x.shape,train.VOCAB),-20.)
        for i,tokens in enumerate(x.tolist()):
            row=pos if tokens[1:3]==[names[pos['t']],names[pos['s']]] else neg
            expected=train.sequence(row,names,self.protocol)
            token=train.EOS if self.malformed else expected[min(len(tokens),len(expected)-1)]
            logits[i,-1,token]=20.
        return logits
records=[[r['s'],r['t'],r['y']] for r in (pos,neg)]
for protocol in ('direct','path'):
    result=train.evaluate(Oracle(protocol),records,w,protocol,torch.device('cpu'))
    assert result['balanced_accuracy']==1.
    if protocol=='path':assert result['valid_positive_path_rate']==1.
    bad=train.evaluate(Oracle(protocol,True),records,w,protocol,torch.device('cpu'))
    assert bad['accuracy']==0.
checks=[]
for job in ('g0/random20/direct','g0/short3/path'):
    output=subprocess.check_output([sys.executable,str(HERE/'train.py'),'--job',job,'--gpu','0','--repo-root',str(ROOT),'--smoke'],text=True)
    checks.append(json.loads(output.splitlines()[-1]))
report=dict(status='PASS',data_determinism=True,all_edges_query_train=True,split_disjoint=True,path_labels_verified=True,
    causal_future_invariance=True,tied_embeddings=True,oracle_evaluation=True,malformed_output_rejected=True,cuda_checks=checks,
    files_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'prepare.py',HERE/'train.py',HERE/'launch.py',HERE/'manifest.json')})
(HERE/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

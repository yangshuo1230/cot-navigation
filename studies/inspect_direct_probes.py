import json, torch
from pathlib import Path
w=json.loads(Path('studies/fixed_adjacency_suite/data/graph_0.json').read_text())
print('WORLD',[(k,len(v) if isinstance(v,(list,dict)) else v) for k,v in w.items()])
print('BASE',[(k,len(v)) for k,v in w['base'].items()])
print('TRAIN',w['base']['train'][:3]); print('VAL',w['core_monitor'][:3]);print('TEST',w['core_final'][:3])
p=torch.load('outputs/exposure_grokking_adj_v3/fixed/g0_s0/direct/R0/latest.pt',map_location='cpu',weights_only=False)
print('CHECKPOINT',p.keys(),p['step']);print('MODEL',list(p['model'])[:10])
for s in ('source','target'):
    i=0 if s=='source' else 1
    print(s,[(name,len(set(r[i] for r in rows))) for name,rows in [('train',w['base']['train']),('val',w['core_monitor']),('test',w['core_final'])]])

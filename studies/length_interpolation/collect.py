"""Read the eight completed study arms; never touch existing training runs."""
from pathlib import Path
import json
import sys

root=Path('studies/length_interpolation')
result={'runs':{},'complete':True}
for protocol in ('bfs_state','dfs_state'):
    for mode in ('parent','short','long','endpoints'):
        path=root/'runs'/protocol/mode
        run={'protocol':protocol,'mode':mode,'predictions':[],'train_curve':[]}
        for name in ('completed','experiment','wandb_status'):
            f=path/(name+'.json')
            if f.exists():run[name]=json.loads(f.read_text())
        if 'completed' not in run:result['complete']=False
        f=path/'predictions.jsonl'
        if f.exists():
            for line in f.open():
                try:run['predictions'].append(json.loads(line))
                except json.JSONDecodeError:pass
        f=path/'metrics.jsonl'
        if f.exists():
            for line in f.open():run['train_curve'].append(json.loads(line))
        result['runs'][protocol+'/'+mode]=run
if len(sys.argv)>1:
    if result['complete']:
        import torch,hashlib
        verification={}
        for key,run in result['runs'].items():
            if run['mode']=='parent':continue
            blob=torch.load(root/'runs'/run['protocol']/run['mode']/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
            encode=lambda v:json.dumps(v,sort_keys=True,default=lambda x:x.tolist()).encode()
            verification[key]={'step':blob['step'],'queries_seen':blob['queries_seen'],
                'query_sampler_sha256':hashlib.sha256(encode(blob['query_sampler'])).hexdigest(),
                'edge_rng_sha256':hashlib.sha256(encode(blob['augmentation_rngs']['edges'])).hexdigest(),
                'padding_seen':blob['padding_state']['seen'],'remaining_quota':blob['padding_state']['quota']}
            del blob
        result['continuation_verification']=verification
    Path(sys.argv[1]).write_text(json.dumps(result))
else:
    print(json.dumps({k:{'completed':v.get('completed'),'n':len(v['predictions']),
                         'step':v['train_curve'][-1]['step'] if v['train_curve'] else None,
                         'interpolation148':{z:sum(bool(r[z]) for r in v['predictions'] if r['case']=='id_disconnected_pad32')
                            for z in ('correct','trace_valid','execution_valid')}}
                      for k,v in result['runs'].items()},indent=2))

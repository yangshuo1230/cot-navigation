"""Review exposure sampler metadata without printing optimizer/RNG state."""
from pathlib import Path
import json,torch
ROOT=Path.cwd()
def describe(state,indent=''):
    if state is None:print(indent,'None');return
    if isinstance(state,list):
        for i,v in enumerate(state):print(indent,'member',i);describe(v,indent+'  ')
        return
    print(indent,{k:v for k,v in state.items() if k in ('kind','step','batch_size','n_items','epoch','cursor','fraction','balance_labels')})
    print(indent,'keys',list(state))
    if 'groups' in state:print(indent,'groups',({k:len(v) for k,v in state['groups'].items()} if isinstance(state['groups'],dict) else str(state['groups'])[:900]))
    for key in ('samplers','sampler','delegate'):
        if key in state:
            if isinstance(state[key],dict) and key=='samplers':
                for name,s in state[key].items():print(indent,name);describe(s,indent+'  ')
            else:describe(state[key],indent+'  ')
    if 'pending' in state:print(indent,'pending count',len(state['pending']))
for setting in ('simple','hard'):
    manifest=json.loads((ROOT/f'studies/fixed100_layer_probes/{setting}_direct.json').read_text())['manifest']
    run=Path(manifest['run']);parent=json.loads((run/'init_from.json').read_text())
    for name,path in [('task',run/'checkpoints/best.pt'),('knowledge',Path(parent['init_from']))]:
        blob=torch.load(path,map_location='cpu',weights_only=False)
        print(setting,name,'step',blob['step'],'checkpoint keys',list(blob))
        print('query');describe(blob.get('query_sampler'));print('walk');describe(blob.get('walk_sampler'))
        print('signatures',blob.get('exposure_signature'),blob.get('data_signature'))

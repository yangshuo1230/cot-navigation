"""Answer-readout controls for additive endpoint identity memorization."""
from pathlib import Path
import gzip
import json
import numpy as np
import torch
import linear
from study import PHASES,phase_positions,summarize_predictions,load_config,ParametricWorld

OUT=Path.cwd()/'studies/fixed100_layer_probes'
for name in ('simple_direct','simple_path','hard_direct','hard_path'):
    result=json.loads((OUT/f'{name}.json').read_text())
    with gzip.open(OUT/f'{name}_queries.jsonl.gz','rt') as f:rows=[json.loads(x) for x in f]
    y=np.asarray([r['label'] for r in rows]);source=np.asarray([r['source'] for r in rows]);target=np.asarray([r['target'] for r in rows])
    splits=np.asarray([r['split'] for r in rows]);hops=np.asarray([r['hops'] for r in rows])
    eye=np.eye(100,dtype=np.float32);rng=np.random.default_rng(22)
    random=rng.normal(size=(100,128)).astype(np.float32)*.05
    features={'node_identity_pair':np.concatenate((eye[source],eye[target]),axis=1),
              'random_embedding_pair':np.concatenate((random[source],random[target]),axis=1)}
    for phase in result['answers']['linear_readout']:
        selected=np.asarray([phase in (phase_positions(len(r['trace'])) if name.endswith('path') else {'prompt':0}) for r in rows])
        ids=np.flatnonzero(selected);split=[np.flatnonzero(splits[selected]==s) for s in (0,1,2)]
        m=dict(hops=hops[selected]);labels=y[selected]
        for control,x in features.items():
            pred,fit=linear.fit(x[selected],labels,split,binary=True)
            result['answers']['linear_readout'][phase][control]=dict(fit=fit,layers=[summarize_predictions(labels,pred[:,0,0],fit['threshold'][0][0],m,split[2])])
    (OUT/f'{name}.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(name,[(c,result['answers']['linear_readout']['prompt'][c]['layers'][0]['all']['balanced_accuracy']) for c in features],flush=True)

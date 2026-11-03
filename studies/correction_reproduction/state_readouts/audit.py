"""Audit compact results against pinned source/checkpoint files after fitting."""
import argparse
import gzip
import json
from pathlib import Path
from cot_navigation.experiments.artifacts import sha_file,atomic_json

ROOT=Path(__file__).resolve().parents[3]


def audit(out):
    out=Path(out);plan=json.loads((out/'plan.json').read_text())
    manifest=ROOT/plan['args']['manifest']
    assert sha_file(manifest)==plan['manifest_sha256']
    mm=json.loads(manifest.read_text());assert mm.get('lifecycle')!='deprecated'
    checked=[]
    for model in mm['models']:
        path=out/(model['id']+'.json')
        if not path.exists():continue
        d=json.loads(path.read_text());assert d['complete']
        ck=ROOT/model['run']/'checkpoints/latest.pt'
        assert not (ck.parent.parent/'DEPRECATED.json').exists()
        assert sha_file(ck)==d['checkpoint']['sha256']
        splits={s:set() for s in ['train','val','test']}
        for src in d['sources']:
            p=ROOT/src['path'];assert sha_file(p)==src['sha256']
            with gzip.open(p,'rt') as f:cache=json.load(f)
            split=cache['identity']['split']
            count={'train':plan['args']['train_queries'],'val':plan['args']['val_queries'],'test':256}[split]
            splits[split].update(r['graph_hash'] for r in cache['rows'][:count])
            if split=='test':assert cache['identity']['checkpoint']['sha256']==d['checkpoint']['sha256']
        assert all(not splits[a]&splits[b] for a,b in [('train','val'),('train','test'),('val','test')])
        assert len(d['probes'])==16
        for probe in d['probes']:
            assert len(probe['test'])==8
            for h in probe['test']:
                assert 0<=h['accuracy']<=1 and 0<=h['balanced_accuracy']<=1
                if 'tp' in h:
                    total=h['tp']+h['tn']+h['fp']+h['fn']
                    assert abs(h['accuracy']-(h['tp']+h['tn'])/total)<1e-8
        assert all(abs(p['final_agreement']-1)<1e-8 for p in d['logit_lens'] if p['layer']==6)
        checked.append(model['id'])
    result=dict(checked=checked,total=len(mm['models']),complete=len(checked)==len(mm['models']),
                checks=['source hashes','receiver checkpoint hashes','test generator identity',
                        'graph split disjointness','metric consistency','final-layer lens identity'])
    atomic_json(out/'audit.json',result);print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');args=p.parse_args();audit(args.output)

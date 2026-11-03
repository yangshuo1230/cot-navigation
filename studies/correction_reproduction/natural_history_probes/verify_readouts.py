"""CPU-only checks of natural matched endpoints and observation coverage."""
from collections import defaultdict
from pathlib import Path
import runpy
from cot_navigation.config import load_config
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.experiments.artifacts import atomic_json

P=Path(__file__).resolve().parent;ROOT=P.parents[2]


def audit(cases,rows):
    pairs=defaultdict(list);phases={}
    for r in rows:
        c=cases[r['sid']]
        assert c['source']==r['source']
        assert r['pos']<len(c['trace'])
        assert c['marker_at'] is None or r['pos']<c['marker_at']
        assert r['field']==c['fields'][r['pos']]
        start=next(lo for lo,hi in c['spans'] if lo<=r['pos']<hi)
        assert r['offset']==r['pos']-start
        assert r['y']==int(c['tau'] is not None and r['pos']>=c['tau'])
        pairs[(r['event'],r['cell'])].append(r)
    for rr in pairs.values():
        assert len(rr)==2 and {r['y'] for r in rr}=={0,1}
        assert len({r['source'] for r in rr})==1 and len({r['field'] for r in rr})==1
        assert rr[0]['query']!=rr[1]['query']
        actual_same=cases[rr[0]['sid']]['trace'][rr[0]['pos']]==cases[rr[1]['sid']]['trace'][rr[1]['pos']]
        assert all(r['exact_token']==actual_same for r in rr)
    for phase in range(3):
        rr=[r for r in rows if r['phase']==phase and r['y']==1]
        phases[str(phase)]=dict(positive_queries=len({r['query'] for r in rr}),pairs=len(rr),
                              exact_token_pairs=sum(r['exact_token'] for r in rr))
    return dict(rows=len(rows),pairs=len(pairs),phases=phases)


def main():
    module=runpy.run_path(str(P/'run.py'))
    cfg=load_config([ROOT/'outputs/correction_reproduction_v1/launches/dfs_correction_repro_20261028/dfs_state/matched_clean/s0/resolved_config.yaml'])
    tok=Tokenizer.from_config(cfg)
    cases,plans,info=module['load_data'](tok)
    result=dict(passed=True,shared={s:audit(cases[s],plans[s]) for s in ['train','val','test']},own={})
    for seed in range(4):
        for arm in ['matched_clean','correction']:
            source=f'{arm}_s{seed}'
            rows,_=module['match_readouts'](cases['test'],tok,20261102,positive_source=source)
            result['own'][source]=audit(cases['test'],rows)
    atomic_json(P/'readout_verification.json',result)
    print('NATURAL READOUT AUDIT PASSED',result['shared'],flush=True)


if __name__=='__main__':main()

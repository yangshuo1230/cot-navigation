"""Independently verify every natural-stream endpoint on CPU."""
import gzip
import importlib.util
import json
from pathlib import Path


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def main():
    from cot_navigation.analysis.common import write_json,sha256
    from cot_navigation.analysis.scale_followup import make_adapter
    root=Path.cwd(); folder=root/'studies/natural_history_error'
    stream=load('stream',folder/'stream.py'); verifier=load('verifier',folder/'verify.py')
    reports=[]
    for path in sorted((folder/'stream_results/models').glob('*.json')):
        result=json.loads(path.read_text()); entry=result['entry']
        assert result['complete'] and result['source_sha256']==sha256(folder/'stream.py')
        am=make_adapter(entry,root,'cpu'); count=0; query_sets=[]
        for split in ('train','val','test'):
            with gzip.open(folder/'rollouts'/f"{entry['id']}_{split}.json.gz",'rt') as f: cache=json.load(f)
            examples=[am.example(q,split) for q in cache['query_ids']]
            query_sets.append(set(cache['query_ids'])); lookup={ex.index:ex for ex in examples}
            rows,sequences,info=stream.build(am,examples,cache['outputs'])
            assert rows==result['endpoints'][split] and info==result['dataset'][split]
            for anchor,items in rows.items():
                for label in (0,1):
                    queries=[r['query'] for r in items if r['y']==label]
                    assert len(queries)==len(set(queries))
                for r in items:
                    ex=lookup[r['query']]; seq=sequences[r['query']][:r['pos']+1]
                    bad=verifier.first_bad(ex,am.protocol,am.tok,seq)
                    assert (bad is not None)==bool(r['y'])
                    if r['y']: assert bad==r['error']
                    assert stream.grammar_field(seq,len(ex.prompt),len(seq)-1,am.tok)==r['field']
                    count+=1
        assert not any(query_sets[i]&query_sets[j] for i in range(3) for j in range(i))
        reports.append(dict(model=entry['id'],verified_endpoints_all_splits=count))
        print('VERIFIED STREAM',entry['id'],count,flush=True)
        del am
    assert len(reports)==4
    write_json(folder/'stream_verification.json',dict(status='PASS',models=reports))


if __name__=='__main__': main()

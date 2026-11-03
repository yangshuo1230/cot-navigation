"""Immediate evidence from the existing unedited N100 first-error+2 corpus.

This short window does not estimate the probability of a later return.
"""
from collections import Counter
import gzip
import json
from pathlib import Path

from cot_navigation.analysis.error_dependencies import dependencies
from cot_navigation.config import load_config
from cot_navigation.data.dataset import ContextStore
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.experiments.artifacts import sha_file

P=Path(__file__).resolve().parent;ROOT=P.parents[3]


def main():
    run=ROOT/'outputs/correction_reproduction_v1/launches/dfs_correction_repro_20261028/dfs_state'
    cfg=load_config([run/'matched_clean/s0/resolved_config.yaml'])
    tok=Tokenizer.from_config(cfg);store=ContextStore(cfg.data.root)
    out=dict(population='ordinary N100 existing natural test corpus, 4 seeds each',
        stop='first error record plus at most 2 later complete records; no edits',groups={})
    for arm in ['matched_clean','correction']:
        results=[];provenance=[];examples=[]
        for seed in range(4):
            f=P.parents[1]/'natural_history_probes/native'/f'{arm}_s{seed}_test.json.gz'
            b=json.load(gzip.open(f,'rt'));assert b['complete']
            provenance.append(dict(file=str(f),sha256=sha_file(f),seed=seed))
            for r in b['rows']:
                q=r['query'];nodes=store.node_vocab_ids(q,tok)
                adj={nodes[u]:[nodes[v] for v in vs] for u,vs in enumerate(store.adj(q))}
                src=nodes[int(store.arrays['s_topo'][q])];dst=nodes[int(store.arrays['t_topo'][q])]
                detailed=dependencies(r['trace'],adj,src,dst,tok)
                results.append(detailed)
                if len(examples)<2:
                    from cot_navigation.analysis.error_dependencies import parse_records
                    for revisit in detailed['revisits']:
                        if not revisit['repeated_unused_extras']:continue
                        records=parse_records(r['trace'],tok)
                        previous=records[revisit['anchor_record']-1]['groups'];returned=records[revisit['return_record']-1]['groups']
                        examples.append(dict(seed=seed,query=q,current=tok.name(revisit['node']),
                            earlier_candidates=tok.decode(previous[2]),chosen_action=tok.decode(previous[3]),
                            returned_candidates=tok.decode(returned[2]),same_extra=tok.decode(revisit['repeated_unused_extras'])))
                        break
        local=[]
        for offset in [1,2]:
            rs=[r for d in results for r in d['after_illegal'] if r['offset']==offset]
            supported=[r for r in rs if r['follows_executed'] is not None]
            local.append(dict(offset=offset,observed=len(rs),supported=len(supported),
                **{field:sum(r['follows_executed'][field] for r in supported) for field in ['frontier','visited','candidate','action','all']},
                rejected_all=sum(r['follows_rejected']['all'] for r in rs if r['follows_rejected'] is not None)))
        revisits=[r for d in results for r in d['revisits']]
        extra=[r for r in revisits if r['original_unused_extras']]
        following=[r for d in results for r in d['annotation_transitions']]
        out['groups'][arm]=dict(trajectories=len(results),local=local,
            unused_extra_anchors=sum(d['unused_extra_anchors'] for d in results),observed_extra_returns=len(extra),
            same_extra_repeated=sum(bool(r['repeated_unused_extras']) for r in extra),
            same_extra_types=Counter('same_extra' if r['repeated_unused_extras'] else 'no_same_extra' for r in extra),
            following={field:dict(Counter(r['following'] for r in following if r['field']==field and not r['cancel'])) for field in ['frontier','visited']},
            provenance=provenance,examples=examples)
    (P/'native_window_summary.json').write_text(json.dumps(out,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({a:{k:v for k,v in g.items() if k!='provenance'} for a,g in out['groups'].items()},separators=(',',':')))


if __name__=='__main__':main()

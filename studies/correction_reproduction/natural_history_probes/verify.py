"""CPU-only independent first-error audit of every primary natural prefix."""
from collections import Counter
from pathlib import Path
import runpy
import hashlib
import json
from types import SimpleNamespace
from cot_navigation.analysis.error_detection import semantic_slots
from cot_navigation.config import load_config
from cot_navigation.data.dataset import ContextStore
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.experiments.artifacts import atomic_json,sha_file

P=Path(__file__).resolve().parent;ROOT=P.parents[2]


def main():
    prep=runpy.run_path(str(P/'prepare.py'))
    folder=ROOT/'outputs/correction_reproduction_v1/launches/dfs_correction_repro_20261028/dfs_state/matched_clean/s0'
    cfg=load_config([folder/'resolved_config.yaml']);tok=Tokenizer.from_config(cfg);store=ContextStore(ROOT/cfg.data.root)
    counts=Counter();differences=[];graphs={s:set() for s in ['train','val','test']};sources=[]
    for path in sorted((P/'native').glob('*.json.gz')):
        data=prep['read_cache'](path);assert data['complete']
        assert sha_file(Path(data['identity']['checkpoint']['path']))==data['identity']['checkpoint']['sha256']
        source=[]
        for row in data['rows']:
            q=row['query'];split=row['split'];graphs[split].add(row['graph_hash'])
            assert int(store.arrays['split_id'][q])=={'train':0,'val':1,'test':2}[split]
            a=store.arrays
            graph_hash=hashlib.sha256(json.dumps([
                sorted(zip(a['edge_src'][q].tolist(),a['edge_dst'][q].tolist())),
                a['node_token'][q].tolist()]).encode()).hexdigest()
            assert graph_hash==row['graph_hash']
            nodes=store.node_vocab_ids(q,tok)
            ex=SimpleNamespace(nodes=nodes,source=int(store.arrays['s_topo'][q]),
                               target=int(store.arrays['t_topo'][q]),adj=store.adj(q))
            trace=row['trace'][:row['marker_at']] if row['marker_at'] is not None else row['trace']
            bad=next((i for i,allowed,_ in semantic_slots(ex,'dfs_state',tok,trace) if trace[i] not in allowed),None)
            first=row['first'];claimed=first['token']-1 if first is not None else None
            if claimed is not None and claimed>=len(trace):claimed=None
            counts['prefixes']+=1;counts['natural_error_prefixes']+=int(claimed is not None)
            if bad==claimed:counts['exact_independent_agreement']+=1
            elif first and first['kind'] in {'action_candidate_priority','format_after_terminal'}:
                # Older semantic_slots accepts any eligible PUSH and stops at
                # terminal; the current replay additionally enforces protocol order.
                counts['documented_protocol_definition_difference']+=1
                source.append(dict(query=q,oracle=bad,replay=claimed,kind=first['kind']))
            else:differences.append(dict(file=path.name,query=q,oracle=bad,replay=claimed,kind=first['kind'] if first else None))
        sources.extend(source)
    for a,b in [('train','val'),('train','test'),('val','test')]:assert not graphs[a]&graphs[b]
    result=dict(passed=not differences,counts=dict(counts),unexpected_differences=differences,
                documented_definition_examples=sources[:8],graphs_disjoint=True,checkpoints_unchanged=True)
    atomic_json(P/'verification.json',result)
    print(result)
    assert not differences


if __name__=='__main__':main()

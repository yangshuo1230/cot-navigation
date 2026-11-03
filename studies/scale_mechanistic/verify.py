"""Read-only CPU audit of split isolation, hashes and saved blocking controls."""
import hashlib
import json
from pathlib import Path
import sys


def main(root, output, allow_partial=False):
    sys.path.insert(0,str(root/'src'))
    from cot_navigation.config import load_config
    from cot_navigation.data.dataset import ContextStore,ParametricWorld
    from cot_navigation.protocols.tokenizer import Tokenizer
    def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    def reachable(adj,roots,target,blocked):
        seen=set();todo=list(roots)
        while todo:
            v=todo.pop()
            if v in blocked or v in seen:continue
            if v==target:return True
            seen.add(v);todo.extend(adj[v])
        return False
    results=[];checked=0;replay_n=0;replay_equal=0
    for f in sorted((output/'models').glob('*.json')):
        d=json.loads(f.read_text());e=d['entry']
        if allow_partial and 'elapsed_seconds' not in d:continue
        assert 'elapsed_seconds' in d,(e['id'],'incomplete')
        assert all(stage in d for stage in ['probes','blocking','patching','errors'])
        assert d['checkpoint_unchanged_after']
        assert digest(root/e['run']/'checkpoints/best.pt')==e['checkpoint_sha256']
        ids={s:set(v) for s,v in d['query_ids'].items()}
        assert all(not ids[a]&ids[b] for a,b in [('train','val'),('train','test'),('val','test')])
        cfg=load_config([root/e['run']/'resolved_config.yaml']);tok=Tokenizer.from_config(cfg)
        is_context=e['kind']=='context'
        source=ContextStore(root/cfg.data.root) if is_context else ParametricWorld(root/cfg.data.root/f'{cfg.data.graph_shape}-seed{cfg.data.graph_seed}')
        for split,ix in ids.items():
            assert ix<=set(map(int,source.indices(split) if is_context else source.indices(split,cfg.evaluation.fixed_query_view)))
        previous=root/e['run']/'eval/predictions_test_best.jsonl'
        old={int(r['query_id']):r['pred'] for r in map(json.loads,previous.read_text().splitlines())}
        matched=[r for r in d['rollout']['by_query'] if r['query'] in old]
        n=len(matched);equal=sum(r['pred']==old[r['query']] for r in matched)
        replay_n+=n;replay_equal+=equal
        if cfg.protocol.name.endswith('_state'):
            if not allow_partial:assert d.get('conditional_probe',{}).get('checkpoint_unchanged'),(e['id'],'conditional probe incomplete')
            for variant in ['including_goal','excluding_goal']:
                b=d['blocking'].get(variant,{})
                for case in b.get('interesting_examples',[]):
                    index=case['query'];adj=source.adj(index) if is_context else source.adj()
                    nodes=source.node_vocab_ids(index,tok) if is_context else source.node_vocab_ids(tok)
                    inv={v:i for i,v in enumerate(nodes)}
                    a=source.arrays if is_context else source.queries
                    target=int(a['t_topo'][index]);detail=case['detail']
                    frontier={inv[v] for v in detail['frontier']};visited={inv[v] for v in detail['visited']}
                    inserted={inv[v] for v in detail['inserted']}
                    assert reachable(adj,frontier,target,visited-frontier)
                    assert not reachable(adj,frontier,target,(visited-frontier)|inserted)
                    if variant=='excluding_goal':assert target not in inserted
                    prefix=case['prefix_tokens'].get('dead_distinct_matched')
                    if prefix:
                        end=len(prefix)-1;start=end
                        while start>0 and prefix[start-1]!=tok.id('|'):start-=1
                        dead={inv[v] for v in prefix[start:end]}-visited
                        assert len(dead)==len(inserted)
                        assert reachable(adj,frontier,target,(visited-frontier)|dead)
                        assert all(not reachable(adj,[v],target,set()) for v in dead)
                    checked+=1
        results.append({'id':e['id'],'checkpoint_unchanged':True,'split_disjoint':True,
                        'previous_test_overlap':n,'previous_test_prediction_matches':equal,
                        'result_sha256':digest(f)})
    assert allow_partial or len(results)==60,('expected 60 models',len(results))
    report={'models':len(results),'complete':len(results)==60,'saved_blocking_cases_independently_checked':checked,
            'prior_best_test_prediction_overlap':replay_n,'prior_best_test_matches':replay_equal,
            'note':'BF16 batch composition can alter greedy trajectories; actual clean arm is used for paired comparisons.',
            'checks':results}
    (output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='checks'}))


if __name__=='__main__':main(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve(),'--partial' in sys.argv[3:])

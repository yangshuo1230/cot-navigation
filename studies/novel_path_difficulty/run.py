"""How difficult are PATH's no-complete-training-witness successes?

Read-only CPU audit of both cached PATH and WALK exposure. Distinguish true
graph search difficulty from the number of already-exposed fragments needed.
No model inference, training, original data writes, or activation files.
"""
from pathlib import Path
import json,gzip,sys
import numpy as np
ROOT=Path.cwd();OUT=ROOT/'studies/novel_path_difficulty'
sys.path.insert(0,str(ROOT/'src'))
from cot_navigation.config import load_config
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.data.dataset import ParametricWorld
from cot_navigation.data.scale_prepared import ScalePreparedRows

def distances(adj,s):
    d=np.full(len(adj),-1,int);d[s]=0;queue=[s]
    for u in queue:
        for v in adj[u]:
            if d[v]<0:d[v]=d[u]+1;queue.append(v)
    return d

def run(setting):
    previous=ROOT/'studies/fixed100_layer_probes'
    manifest=json.loads((previous/f'{setting}_path.json').read_text())['manifest']
    cfg=load_config([Path(manifest['run'])/'resolved_config.yaml']);tok=Tokenizer.from_config(cfg)
    world=ParametricWorld(Path(cfg.data.root)/'main-seed0');adj=world.adj();n=len(adj)
    nodes=world.node_vocab_ids(tok);inv={v:i for i,v in enumerate(nodes)}
    a=np.zeros((n,n),bool);reach=np.eye(n,dtype=bool)
    for u,vs in enumerate(adj):a[u,vs]=True
    for u in reversed(range(n)):
        for v in adj[u]:assert v>u;reach[u]|=reach[v]
    cached=ScalePreparedRows(world,cfg,tok);sequences=[];pairs=set();goal_policy=set()
    arrays=cached.arrays
    for q in world.indices('train').tolist():
        lo,hi=map(int,arrays['query_offsets'][q:q+2])
        for row in range(lo,hi):
            start=int(arrays['offsets'][row]);prompt,answer=map(int,arrays['fields'][row,:2])
            path=tuple(inv[int(v)] for v in arrays['tokens'][start+prompt:start+answer] if int(v) in inv)
            assert all(v in adj[u] for u,v in zip(path,path[1:]))
            sequences.append(path);goal=int(world.queries['t_topo'][q]);goal_policy.update((u,goal) for u in path[:-1])
    for i in range(world.n_walks()):sequences.append(tuple(world.walk_nodes_at(i)))
    for path in sequences:
        for i,u in enumerate(path):pairs.update((u,v) for v in path[i+1:])
    macro=[[] for _ in range(n)]
    for u,v in pairs:macro[u].append(v)
    rows=[json.loads(v) for v in gzip.open(previous/f'{setting}_path_queries.jsonl.gz','rt')]
    positive=[r for r in rows if r['split']==2 and r['label']]
    strict=[r for r in positive if (r['source'],r['target']) not in pairs]
    expected=json.loads((ROOT/f'studies/path_exposure_audit/{setting}.json').read_text())['strict_no_complete_witness']['n']
    assert len(strict)==expected
    # Only index desired strict-generated fragments, not millions of cache subpaths.
    wanted=set()
    for row in strict:
        path=tuple(inv[v] for v in row['trace'])
        for i in range(len(path)):
            for j in range(i+2,len(path)+1):wanted.add(path[i:j])
    wanted_by_start={}
    for fragment in wanted:wanted_by_start.setdefault(fragment[0],set()).add(len(fragment))
    found={}
    for path in sequences:
        for i,u in enumerate(path):
            for length in wanted_by_start.get(u,[]):
                fragment=path[i:i+length]
                if fragment in wanted and fragment not in found:found[fragment]=path
    random={};counts={}
    for t in set(r['target'] for r in positive):
        prob=np.zeros(n);prob[t]=1.;count=[0]*n;count[t]=1
        for u in reversed(range(n)):
            if u!=t and adj[u]:prob[u]=prob[adj[u]].mean();count[u]=sum(count[v] for v in adj[u])
        assert np.array_equal(prob>0,reach[:,t]);random[t]=prob;counts[t]=count
    def describe(row):
        s,t=row['source'],row['target'];path=[inv.get(v,-1) for v in row['trace']]
        legal=bool(path and path[0]==s and all(u>=0 for u in path))
        if legal:legal=all(v in adj[u] for u,v in zip(path,path[1:])) and path[-1]==t
        branches=[];specific=1.
        if legal:
            for u,v in zip(path,path[1:]):
                good=[w for w in adj[u] if reach[w,t]];dead=[w for w in adj[u] if not reach[w,t]]
                specific/=len(adj[u])
                branches.append(dict(node=u,out_degree=len(adj[u]),useful=len(good),dead=len(dead),goal_conditioned_seen=(u,t) in goal_policy))
        return dict(query=row['query_ids'],source=s,target=t,source_token=tok.name(nodes[s]),target_token=tok.name(nodes[t]),
            shortest_hops=int(distances(adj,s)[t]),generated_edges=len(path)-1,valid_success=legal,answer_correct=row['pred']==1,
            random_goal_probability=float(random[t][s]),successful_graph_paths=int(counts[t][s]),
            mixed_branch_decisions=sum(b['useful']>0 and b['dead']>0 for b in branches),
            unseen_mixed_branch_decisions=sum(b['useful']>0 and b['dead']>0 and not b['goal_conditioned_seen'] for b in branches),
            all_successor_branch_decisions=sum(b['out_degree']>1 for b in branches),
            maximum_out_degree=max((b['out_degree'] for b in branches),default=0),
            specific_generated_path_random_probability=specific if legal else None,
            source_indegree=int(a[:,s].sum()),target_outdegree=len(adj[t]),
            path=path,path_tokens=[tok.name(nodes[u]) if u>=0 else 'INVALID' for u in path],branches=branches)
    strict_records=[]
    for row in strict:
        r=describe(row);path=tuple(r['path']);L=len(path)-1
        dp=[n+1]*(L+1);dp[0]=0;parent={}
        for i in range(L):
            for j in range(i+1,L+1):
                if path[i:j+1] in found and dp[i]+1<dp[j]:dp[j]=dp[i]+1;parent[j]=i
        assert dp[L]>1 and dp[L]<=L
        segments=[];j=L
        while j:
            i=parent[j];segments.append(list(path[i:j+1]));j=i
        segments.reverse()
        seen_suffix=[i for i in range(L) if path[i:] in found]
        novel_prefix=min(seen_suffix) if seen_suffix else L
        r.update(min_seen_fragments_for_generated_path=dp[L],generated_path_fragment_decomposition=segments,
            macro_graph_min_composition=int(distances(macro,row['source'])[row['target']]),
            novel_prefix_edges_before_seen_suffix=novel_prefix,
            longest_seen_fragment_edges=max((j-i for i in range(L) for j in range(i+1,L+1) if path[i:j+1] in found),default=0),
            unseen_goal_conditioned_decisions=sum((u,row['target']) not in goal_policy for u in path[:-1]))
        strict_records.append(r)
    other=[describe(row) for row in positive if (row['source'],row['target']) in pairs]
    matched=[]
    for r in strict_records:
        pool=[o for o in other if o['shortest_hops']==r['shortest_hops']]
        if pool:matched.append(dict(hops=r['shortest_hops'],n=len(pool),random_goal_probability=float(np.mean([o['random_goal_probability'] for o in pool]))))
    summary=dict(strict_n=len(strict_records),shortest_hops=[r['shortest_hops'] for r in strict_records],
        generated_edges=[r['generated_edges'] for r in strict_records],
        random_goal_probability_mean=float(np.mean([r['random_goal_probability'] for r in strict_records])),
        random_goal_probability_range=[float(min(r['random_goal_probability'] for r in strict_records)),float(max(r['random_goal_probability'] for r in strict_records))],
        mixed_branch_range=[min(r['mixed_branch_decisions'] for r in strict_records),max(r['mixed_branch_decisions'] for r in strict_records)],
        strict_with_no_mixed_branch=sum(r['mixed_branch_decisions']==0 for r in strict_records),
        strict_with_no_unseen_mixed_branch=sum(r['unseen_mixed_branch_decisions']==0 for r in strict_records),
        unseen_mixed_branch_hist={str(k):sum(r['unseen_mixed_branch_decisions']==k for r in strict_records) for k in sorted(set(r['unseen_mixed_branch_decisions'] for r in strict_records))},
        initial_single_successor_count=sum(r['branches'][0]['out_degree']==1 for r in strict_records),
        initial_all_successors_useful_count=sum(r['branches'][0]['dead']==0 for r in strict_records),
        generated_fragment_count_hist={str(k):sum(r['min_seen_fragments_for_generated_path']==k for r in strict_records) for k in sorted(set(r['min_seen_fragments_for_generated_path'] for r in strict_records))},
        macro_composition_hist={str(k):sum(r['macro_graph_min_composition']==k for r in strict_records) for k in sorted(set(r['macro_graph_min_composition'] for r in strict_records))},
        one_new_edge_before_seen_suffix=sum(r['novel_prefix_edges_before_seen_suffix']==1 for r in strict_records),
        source_root_count=sum(r['source_indegree']==0 for r in strict_records),target_sink_count=sum(r['target_outdegree']==0 for r in strict_records),
        hop_matched_other_random_rate=float(np.mean([r['random_goal_probability'] for r in matched])) if matched else None)
    result=dict(manifest=manifest,summary=summary,queries=strict_records,
        method='No complete cached PATH/WALK witness. Actual test greedy trajectories. Exact DAG random-walk hitting probability and exact path counts. '
            'Fragments from train cached positive/negative PATH + WALK; min composition in exposed-pair macro graph and along actual output separately. '
            'Cache availability is not verified optimizer exposure. No inference/training/GPU.')
    (OUT/f'{setting}.json').write_text(json.dumps(result,indent=2))
    print(setting,json.dumps(summary),flush=True)

if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    for setting in ('simple','hard'):run(setting)

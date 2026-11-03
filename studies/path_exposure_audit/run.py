"""Read-only PATH cache/walk witness coverage, not optimizer exposure replay.

Recheck both N100 models from their saved test generations. No model forward,
data generation, training, mutation of original caches, or GPU use.
"""
from pathlib import Path
import os,json,gzip,hashlib,sys
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
ROOT=Path.cwd();OUT=ROOT/'studies/path_exposure_audit'
sys.path.insert(0,str(ROOT/'src'))
from cot_navigation.config import load_config
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.data.dataset import ParametricWorld
from cot_navigation.data.scale_prepared import ScalePreparedRows

def run(setting):
    previous=ROOT/'studies/fixed100_layer_probes'
    manifest=json.loads((previous/f'{setting}_path.json').read_text())['manifest']
    folder=Path(manifest['run']);cfg=load_config([folder/'resolved_config.yaml']);tok=Tokenizer.from_config(cfg)
    assert hashlib.sha256((folder/'checkpoints/best.pt').read_bytes()).hexdigest()==manifest['checkpoint_sha256']
    world=ParametricWorld(Path(cfg.data.root)/'main-seed0');adj=world.adj();node_ids=world.node_vocab_ids(tok)
    inverse={v:i for i,v in enumerate(node_ids)};cache=ScalePreparedRows(world,cfg,tok)
    rows=[json.loads(s) for s in gzip.open(previous/f'{setting}_path_queries.jsonl.gz','rt')]
    rows=[r for r in rows if r['split']==2]
    reach_path=set();reach_walk=set();exact_path=set();exact_walk=set();policy=set();found_examples={}
    queries={r['query_ids']:tuple(inverse.get(v,-1) for v in r['trace']) for r in rows if r['label']}
    wanted=set(queries.values());by_start={}
    for sequence in wanted:
        if sequence:by_start.setdefault(sequence[0],set()).add(len(sequence))
    def scan(nodes,pairs,exact,origin):
        assert all(v in adj[u] for u,v in zip(nodes,nodes[1:]))
        for i,u in enumerate(nodes):
            for j,v in enumerate(nodes[i+1:],i+1):
                pairs.add((u,v))
                if (u,v) not in found_examples:found_examples[u,v]=dict(origin=origin,path=nodes[i:j+1])
            for length in by_start.get(u,[]):
                candidate=tuple(nodes[i:i+length])
                if candidate in wanted:exact.add(candidate)
    count=0;train_ids=world.indices('train').tolist()
    for q in train_ids:
        for variant in range(cache.variants(q)):
            row=cache.row(q,variant);body=row.tokens[row.prompt_len:row.answer_index]
            nodes=[inverse[v] for v in body if v in inverse]
            assert nodes and nodes[0]==int(world.queries['s_topo'][q])
            goal=int(world.queries['t_topo'][q])
            scan(nodes,reach_path,exact_path,dict(query=q,variant=variant,label=int(row.label)))
            policy.update((u,goal) for u in nodes[:-1]);count+=1
    for i in range(world.n_walks()):scan(world.walk_nodes_at(i),reach_walk,exact_walk,dict(walk=i))
    output_rows=[]
    for r in rows:
        s,t=r['source'],r['target'];path=tuple(inverse.get(v,-1) for v in r['trace'])
        legal=bool(path and path[0]==s and all(v>=0 for v in path))
        if legal:legal=all(v in adj[u] for u,v in zip(path,path[1:]))
        success=bool(r['label'] and legal and path[-1]==t)
        path_seen=(s,t) in reach_path;walk_seen=(s,t) in reach_walk
        assert r['label'] or not (path_seen or walk_seen)
        output_rows.append(dict(query=r['query_ids'],source=s,target=t,label=r['label'],pred=r['pred'],
            correct=r['pred']==r['label'],valid_successful_witness=success,
            any_training_PATH_witness=path_seen,any_training_WALK_witness=walk_seen,
            exact_generated_PATH_seen=success and path in exact_path,exact_generated_WALK_seen=success and path in exact_walk,
            last_node_is_goal=bool(path and path[-1]==t),goal_ever_output=t in path,
            all_generated_goal_conditioned_decisions_seen=success and all((u,t) in policy for u in path[:-1])))
    positives=[r for r in output_rows if r['label']];success=[r for r in positives if r['valid_successful_witness']]
    strict=[r for r in positives if not r['any_training_PATH_witness'] and not r['any_training_WALK_witness']]
    novel=[r for r in success if not r['exact_generated_PATH_seen'] and not r['exact_generated_WALK_seen']]
    lookup=[bool(r['any_training_PATH_witness'] or r['any_training_WALK_witness']) for r in output_rows]
    result=dict(manifest=manifest,training_queries=len(train_ids),training_cached_PATH_variants=count,walks=world.n_walks(),
        test_queries=len(rows),positive_queries=len(positives),negative_queries=len(rows)-len(positives),
        positive_with_PATH_witness=sum(r['any_training_PATH_witness'] for r in positives),
        positive_with_WALK_witness=sum(r['any_training_WALK_witness'] for r in positives),
        positive_with_either_witness=sum(r['any_training_PATH_witness'] or r['any_training_WALK_witness'] for r in positives),
        valid_successful_outputs=len(success),successful_exact_generated_PATH_seen=sum(r['exact_generated_PATH_seen'] for r in success),
        successful_exact_generated_either_seen=sum(r['exact_generated_PATH_seen'] or r['exact_generated_WALK_seen'] for r in success),
        novel_successful_outputs=len(novel),strict_no_complete_witness=dict(n=len(strict),correct=sum(r['correct'] for r in strict),legal_success=sum(r['valid_successful_witness'] for r in strict)),
        successful_all_goal_conditioned_decisions_seen=sum(r['all_generated_goal_conditioned_decisions_seen'] for r in success),
        strict_all_goal_conditioned_decisions_seen=sum(r['all_generated_goal_conditioned_decisions_seen'] for r in strict),
        lookup_available_path_else_NO=dict(accuracy=float(np.mean([int(v)==r['label'] for v,r in zip(lookup,output_rows)])),
            yes_recall=float(np.mean([r['any_training_PATH_witness'] or r['any_training_WALK_witness'] for r in positives])),no_recall=1.),
        lookup_agreement_with_model=float(np.mean([int(v)==r['pred'] for v,r in zip(lookup,output_rows)])),
        final_node_equals_goal_rule=dict(agreement_with_model=float(np.mean([int(r['last_node_is_goal'])==r['pred'] for r in output_rows])),
            ground_truth_accuracy=float(np.mean([int(r['last_node_is_goal'])==r['label'] for r in output_rows]))),
        model_answer_accuracy=float(np.mean([r['correct'] for r in output_rows])),
        strict_examples=[dict(**r,trace=queries[r['query']]) for r in strict[:4]],
        novel_examples=[dict(**r,trace=queries[r['query']]) for r in novel[:4]],
        caveat='Prepared train cache availability, not actual optimizer sample exposure. Only train query IDs; positive and negative cached paths both audited. '
               'Witness may be a contiguous SUBPATH of a longer training example. Does not imply the same query prompt was trained. '
               'Lookup is a non-neural diagnostic, not identified model mechanism.')
    (OUT/f'{setting}.json').write_text(json.dumps(result,indent=2))
    print(setting,json.dumps({k:v for k,v in result.items() if k not in ('manifest','strict_examples','novel_examples','caveat')}),flush=True)

if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    for setting in ('simple','hard'):run(setting)

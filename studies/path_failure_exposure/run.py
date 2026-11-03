"""Expose three different meanings of 'seen' for PATH dead-branch failures.

Read-only CPU cache audit: correct witness, exact wrong generated path, and
goal-conditioned next-edge supervision at the first irreversible error.
"""
from pathlib import Path
import json,sys
ROOT=Path.cwd();OUT=ROOT/'studies/path_failure_exposure'
sys.path.insert(0,str(ROOT/'src'))
from cot_navigation.config import load_config
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.data.dataset import ParametricWorld
from cot_navigation.data.scale_prepared import ScalePreparedRows

def run(setting):
    previous=json.loads((ROOT/f'studies/path_failures/{setting}.json').read_text())
    manifest=previous['manifest'];cfg=load_config([Path(manifest['run'])/'resolved_config.yaml']);tok=Tokenizer.from_config(cfg)
    world=ParametricWorld(Path(cfg.data.root)/'main-seed0');nodes=world.node_vocab_ids(tok);inv={v:i for i,v in enumerate(nodes)}
    cache=ScalePreparedRows(world,cfg,tok);a=cache.arrays;training=[]
    def token(v):return tok.name(nodes[int(v)])
    failures=[f for f in previous['failures'] if f['kind']=='legal_edge_to_dead_branch']
    out=[]
    for f in failures:
        r=dict(f);r['internal_source']=inv[tok.id(f['source'])];r['internal_target']=inv[tok.id(f['target'])]
        r['path']=tuple(inv[tok.id(v)] for v in f['trace'])
        b=f['first_bad_branch'];r['current']=inv[tok.id(b['current'])];r['chosen']=inv[tok.id(b['chosen'])]
        r['useful']={inv[tok.id(v)] for v in b['useful']}
        r['correct_witness']=None;r['wrong_exact_PATH']=None;r['wrong_exact_WALK']=None
        r['correct_same_goal_cached_rows']=0;r['wrong_same_goal_cached_rows']=0;r['same_goal_correct_examples']=[]
        r['wrong_edge_ever_seen']=False;out.append(r)
    for q in world.indices('train').tolist():
        lo,hi=map(int,a['query_offsets'][q:q+2]);goal=int(world.queries['t_topo'][q]);label=int(world.queries['label'][q])
        for variant,row in enumerate(range(lo,hi)):
            offset=int(a['offsets'][row]);prompt,answer=map(int,a['fields'][row,:2])
            path=tuple(inv[int(v)] for v in a['tokens'][offset+prompt:offset+answer] if int(v) in inv)
            info=dict(training_query=q,variant=variant,label=label,training_goal=token(goal))
            for r in out:
                s,t=r['internal_source'],r['internal_target'];u,v=r['current'],r['chosen']
                if r['correct_witness'] is None and s in path and t in path and path.index(s)<path.index(t):
                    i,j=path.index(s),path.index(t);r['correct_witness']=dict(**info,witness=[token(w) for w in path[i:j+1]])
                if r['wrong_exact_PATH'] is None:
                    length=len(r['path'])
                    if any(path[i:i+length]==r['path'] for i in range(len(path)-length+1)):
                        r['wrong_exact_PATH']=info
                for i,w in enumerate(path[:-1]):
                    if w!=u:continue
                    if path[i+1]==v:r['wrong_edge_ever_seen']=True
                    if goal==t:
                        if path[i+1] in r['useful']:
                            r['correct_same_goal_cached_rows']+=1
                            if len(r['same_goal_correct_examples'])<2:r['same_goal_correct_examples'].append(dict(**info,next_node=token(path[i+1])))
                        if path[i+1]==v:r['wrong_same_goal_cached_rows']+=1
    for i in range(world.n_walks()):
        path=tuple(world.walk_nodes_at(i))
        for r in out:
            length=len(r['path'])
            if r['wrong_exact_WALK'] is None and any(path[j:j+length]==r['path'] for j in range(len(path)-length+1)):
                r['wrong_exact_WALK']=dict(walk=i)
            if any((u,v)==(r['current'],r['chosen']) for u,v in zip(path,path[1:])):r['wrong_edge_ever_seen']=True
    for r in out:
        assert r['correct_witness'] is not None
        assert r['wrong_same_goal_cached_rows']==0
        r['useful']=sorted(r['useful']);r.pop('path')
    summary=dict(n=len(out),with_correct_complete_witness=sum(r['correct_witness'] is not None for r in out),
        wrong_generated_path_seen_PATH=sum(r['wrong_exact_PATH'] is not None for r in out),
        wrong_generated_path_seen_PATH_or_WALK=sum(r['wrong_exact_PATH'] is not None or r['wrong_exact_WALK'] is not None for r in out),
        correct_next_choice_seen_with_same_goal=sum(r['correct_same_goal_cached_rows']>0 for r in out),
        wrong_next_choice_seen_with_same_goal=sum(r['wrong_same_goal_cached_rows']>0 for r in out),
        wrong_edge_seen_under_other_context=sum(r['wrong_edge_ever_seen'] for r in out))
    (OUT/f'{setting}.json').write_text(json.dumps(dict(manifest=manifest,summary=summary,queries=out,
        caveat='Training cache availability, not verified optimizer exposure. Complete witnesses and exact generated paths may be SUBPATHS. '
               'Same-goal state supervision can occur inside another source query. Zero wrong goal-conditioned choices follows valid supervision semantics, not a discovery of actual optimization behavior.'),indent=2))
    print(setting,json.dumps(summary),flush=True)

if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    for setting in ('simple','hard'):run(setting)

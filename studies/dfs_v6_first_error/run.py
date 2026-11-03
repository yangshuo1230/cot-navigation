"""Streaming first-error audit of v6 DFS nodes on all original query panels.

Only the first temporally observable violation is attributed. Implicit legal
backtracking is reconstructed before testing an edge. Causes here are behavioral
signatures, not claims about the model's internal causal mechanism.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
os.environ['WANDB_MODE']='disabled'
import sys,json,gzip,hashlib
from pathlib import Path
from collections import Counter,defaultdict,deque
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
V6=ROOT/'studies/fixed_navigation_protocols_v6'
sys.path[:0]=[str(V6),str(ROOT/'src')]
from semantics import replay_nodes
OUT=Path(__file__).resolve().parent


def distances(adj):
    result=[]
    for start in range(len(adj)):
        dist={start:0};todo=deque([start])
        while todo:
            u=todo.popleft()
            for v in adj[u]:
                if v not in dist:dist[v]=dist[u]+1;todo.append(v)
        result.append(dist)
    return result


def analyze(adj,source,target,generated,inverse,tok,limited=False,dist=None):
    """Return the first error, its genuine pre-error state and opportunities.

    The event stream includes the sampled answer/EOS, unlike a node-only parser.
    This avoids scanning future non-node tokens before an earlier search error.
    """
    stack=[];seen=set();order=[];opportunities=Counter();last=None
    yes,no,eos=tok.yes,tok.no,tok.id('<EOS>')

    def kind(token):
        if token is None:return 'missing'
        if token in inverse:return 'node'
        if token==yes:return 'yes'
        if token==no:return 'no'
        if token==eos:return 'eos'
        if tok.node_lo<=token<tok.node_lo+tok.n_node_tokens:return 'off_graph_node'
        return 'syntax'

    def finish(error,position,token=None,decision=None,candidates=None,pops=0,terminal=None,**detail):
        semantic_valid=error is None or error in {
            'wrong_answer_after_goal','wrong_answer_after_exhaustion',
            'missing_answer_after_goal','missing_answer_after_exhaustion'}
        if error=='budget_limit' and terminal is not None:semantic_valid=True
        if error:
            detail['actual_is_target']=inverse.get(token)==target
        return dict(error=error or 'valid',protocol_valid=error is None,semantic_valid=semantic_valid,
                    first_error_position=position if error else None,
                    first_error_fraction=(position/max(1,len(generated))) if error else None,
                    generated_length=len(generated),legal_expansions_before_error=len(order),visited_count=len(seen),
                    visited_fraction=len(seen)/len(adj),stack_depth=len(stack),required_pops=pops,
                    decision_top=decision[-1] if decision else None,n_legal_candidates=len(candidates or []),
                    actual_token=token,actual_node=inverse.get(token),expected_terminal=terminal,
                    opportunities=dict(opportunities),detail=detail,
                    _stack=stack[:],_decision_stack=list(decision or []),_visited=sorted(seen),
                    _candidates=list(candidates or []),_prefix_tail=order[-12:])

    for i,token in enumerate(generated):
        event=kind(token)
        if not order:
            opportunities['initial_source']+=1
            if event!='node':return finish('missing_start' if event in ['yes','no','eos'] else 'non_node_at_start',i,token)
            if inverse[token]!=source:return finish('wrong_start',i,token)
            stack=[source];seen={source};order=[source];last=source
            continue
        if stack[-1]==target:
            opportunities['goal_reached']+=1
            if event=='yes':
                opportunities['goal_stopped_correctly']+=1
                return finish(None,i,token,stack,[],terminal='FOUND')
            if event=='no':return finish('wrong_answer_after_goal',i,token,stack,[],terminal='FOUND')
            if event=='eos':return finish('missing_answer_after_goal',i,token,stack,[],terminal='FOUND')
            # Does the next token follow ordinary DFS if the goal test is
            # deliberately ignored? This isolates a concrete behavioral pattern.
            ignore_stack=stack[:];ignore_pops=0
            while len(ignore_stack)>1 and not any(v not in seen for v in adj[ignore_stack[-1]]):
                ignore_stack.pop();ignore_pops+=1
            ignore_candidates=[v for v in adj[ignore_stack[-1]] if v not in seen]
            return finish('continued_after_goal',i,token,stack,[],terminal='FOUND',event_kind=event,
                          follows_DFS_if_goal_ignored=(event=='node' and inverse[token] in ignore_candidates),
                          ignored_goal_required_pops=ignore_pops,
                          ignored_goal_has_remaining_candidates=bool(ignore_candidates))
        decision=stack[:];pops=0
        while len(decision)>1 and not any(v not in seen for v in adj[decision[-1]]):
            decision.pop();pops+=1
        candidates=[v for v in adj[decision[-1]] if v not in seen]
        if not candidates:
            opportunities['search_exhausted']+=1
            if event=='no':
                opportunities['exhaust_stopped_correctly']+=1
                return finish(None,i,token,decision,[],pops,'EXHAUST')
            if event=='yes':return finish('wrong_answer_after_exhaustion',i,token,decision,[],pops,'EXHAUST')
            if event=='eos':return finish('missing_answer_after_exhaustion',i,token,decision,[],pops,'EXHAUST')
            return finish('continued_after_exhaustion',i,token,decision,[],pops,'EXHAUST',event_kind=event)
        opportunities['active_decision']+=1
        if pops:opportunities['decision_after_required_backtrack']+=1
        else:opportunities['decision_without_backtrack']+=1
        if event in ['yes','no']:
            return finish('premature_yes' if event=='yes' else 'premature_no',i,token,decision,candidates,pops)
        if event=='eos':return finish('premature_eos',i,token,decision,candidates,pops)
        if event=='off_graph_node':return finish('off_graph_node',i,token,decision,candidates,pops)
        if event!='node':return finish('non_node_token',i,token,decision,candidates,pops)
        node=inverse[token]
        if node in seen:
            subtype=('printed_implicit_return_top' if pops and node==decision[-1] else
                     'repeated_current' if node==last else
                     'repeated_stack_ancestor' if node in stack else 'revisited_closed_node')
            return finish('revisited_node',i,token,decision,candidates,pops,repeat_subtype=subtype,
                          edge_from_decision_top=node in adj[decision[-1]])
        if node not in candidates:
            ancestors=[u for u in reversed(decision[:-1]) if node in adj[u]]
            if ancestors:
                return finish('skipped_live_frontier',i,token,decision,candidates,pops,
                              alternative_ancestor=ancestors[0])
            distance=dist[decision[-1]].get(node) if dist is not None else None
            subtype='reachable_descendant_skip' if distance is not None and distance>=2 else 'other_branch_or_unreachable'
            closed_parents=[u for u in seen if u not in decision and node in adj[u]]
            return finish('non_edge_jump',i,token,decision,candidates,pops,
                          jump_subtype=subtype,graph_distance_from_top=distance,
                          has_edge_from_closed_visited_node=bool(closed_parents))
        opportunities['legal_next_expansion']+=1
        stack=decision+[node];seen.add(node);order.append(node);last=node
    # No sampled answer: represent the missing next event, never invent a NO.
    terminal=None;decision=stack[:];pops=0;candidates=[]
    if stack:
        if stack[-1]==target:terminal='FOUND'
        else:
            while len(decision)>1 and not any(v not in seen for v in adj[decision[-1]]):decision.pop();pops+=1
            candidates=[v for v in adj[decision[-1]] if v not in seen]
            if not candidates:terminal='EXHAUST'
    if terminal=='FOUND':opportunities['goal_reached']+=1
    elif terminal=='EXHAUST':opportunities['search_exhausted']+=1
    if limited:return finish('budget_limit',len(generated),None,decision,candidates,pops,terminal)
    error=('missing_start' if not order else 'missing_answer_after_goal' if terminal=='FOUND' else
           'missing_answer_after_exhaustion' if terminal=='EXHAUST' else 'missing_answer_before_completion')
    return finish(error,len(generated),None,decision,candidates,pops,terminal)


def groups(panel):
    role='short' if panel.startswith('short_') else 'AA' if panel=='long_train' else panel.split('_')[1]
    split='train' if panel.endswith('_train') else 'val' if panel.endswith('_val') else 'test'
    return role,split


def main():
    import torch
    from packed_bank import PackedBank
    from cot_navigation.models.transformer import TransformerLM
    from cot_navigation.protocols.tokenizer import Tokenizer
    from cot_navigation.evaluation.rollout import generate
    torch.set_num_threads(2);tok=Tokenizer(256)
    manifest=json.loads((V6/'manifest.json').read_text());meta=next(m for m in manifest['worlds'] if m['name']=='n80_g0')
    bank=PackedBank(meta,'dfs_nodes',torch.device('cpu'));world=bank.world
    nodes=[tok.node_id(v) for v in world['node_names']];inverse={t:i for i,t in enumerate(nodes)}
    dist=distances(world['adj'])
    for r in world['records']:
        assert (r['t'] in dist[r['s']])==bool(r['y'])
    probe=json.loads((ROOT/'studies/dfs_v6_answer_all_queries/results.json').read_text())
    probe_split={q:split for split,ids in probe['query_ids'].items() for q in ids}
    doc={'analysis_version':2,'setting':'v6 N80 graph0; original13 query panels; all2712 query IDs x4 final50k initializations',
         'definition':'First temporally observable protocol violation; true graph replay with implicit DFS pops, no repairs.',
         'seeds':[]}
    path=OUT/'results.json'
    if path.exists():
        previous=json.loads(path.read_text())
        if previous.get('analysis_version')==2:doc=previous
    examples=json.loads((OUT/'examples.json').read_text()) if doc['seeds'] and (OUT/'examples.json').exists() else {}
    for seed in range(4):
        if any(s['seed']==seed for s in doc['seeds']):continue
        run=ROOT/f'outputs/fixed_navigation_protocols_v6/pilot/n80/g0_s{seed}/dfs_nodes'
        blob=torch.load(run/'latest.pt',map_location='cpu',weights_only=False);assert blob['step']==50000
        model=TransformerLM(len(tok),**blob['job']['model'],dropout=0.).to('cuda:1');model.load_state_dict(blob['model']);model.eval()
        for p in model.parameters():p.requires_grad_(False)
        compiled=torch.compile(model,dynamic=True)
        result={'seed':seed,'checkpoint_sha256':hashlib.sha256((run/'latest.pt').read_bytes()).hexdigest(),
                'panels':{},'legacy_semantic_agreements':0,'legacy_error_cross_tab':Counter()}
        original=json.loads((run/'final.json').read_text())
        with gzip.open(OUT/f'seed{seed}_queries.jsonl.gz','wt',encoding='utf-8') as stream:
            for panel,ids in world['panels'].items():
                print('PANEL',seed,panel,len(ids),flush=True)
                role,main_split=groups(panel)
                rows=[];counter=Counter();opps=Counter()
                for at in range(0,len(ids),128):
                    part=ids[at:at+128]
                    prompts=[[tok.id('<QUERY>'),nodes[world['records'][q]['s']],nodes[world['records'][q]['t']],tok.id('|')] for q in part]
                    outputs=generate(compiled,prompts,'dfs_nodes',tok,'bf16',max_new_tokens=bank.info['max_new_tokens'])
                    for q,out in zip(part,outputs):
                        r=world['records'][q]
                        a=analyze(world['adj'],r['s'],r['t'],out['generated'],inverse,tok,out['generation_limit_reached'],dist)
                        legacy=replay_nodes(world['adj'],r['s'],r['t'],out['trace_tokens'],inverse)
                        assert a['semantic_valid']==legacy['valid'],(seed,panel,q,a,legacy)
                        result['legacy_semantic_agreements']+=1
                        result['legacy_error_cross_tab'][f"{legacy['error'] or 'valid'} -> {a['error']}"]+=1
                        a.update(seed=seed,query=q,panel=panel,role=role,main_split=main_split,probe_split=probe_split[q],
                                 label=r['y'],gap=r['gap'],shortest_path=dist[r['s']].get(r['t']),
                                 answer_correct=out['answer_token']==tok.label_id(r['y']),
                                 answer_token=out['answer_token'],truncated=out['generation_limit_reached'])
                        counter[a['error']]+=1;opps.update(a['opportunities'])
                        key=f"{panel}/{a['error']}"
                        if a['error']!='valid' and key not in examples:
                            def names(values):return [tok.name(nodes[v]) for v in values]
                            example={k:v for k,v in a.items() if not k.startswith('_')}
                            example.update(source=tok.name(nodes[r['s']]),target=tok.name(nodes[r['t']]),
                                actual=tok.name(a['actual_token']) if a['actual_token'] is not None else None,
                                stack=names(a['_stack']),decision_stack=names(a['_decision_stack']),visited=names(a['_visited']),
                                candidates=names(a['_candidates']),legal_prefix_tail=names(a['_prefix_tail']),
                                generated_tail=[tok.name(t) for t in out['generated'][max(0,a['first_error_position']-3):a['first_error_position']+4]])
                            examples[key]=example
                        compact={k:v for k,v in a.items() if not k.startswith('_')}
                        stream.write(json.dumps(compact,ensure_ascii=False)+'\n');rows.append(compact)
                old=original['panels'].get(panel)
                comparisons={}
                if old:
                    expected=old['metrics']['all']
                    current={'n':len(rows),'correct':sum(r['answer_correct'] for r in rows),
                             'trace_valid':sum(r['semantic_valid'] for r in rows),'truncated':sum(r['truncated'] for r in rows)}
                    comparisons={k:dict(now=v,previous=expected.get(k,0)) for k,v in current.items() if v!=expected.get(k,0)}
                result['panels'][panel]={'n':len(rows),'role':role,'split':main_split,'errors':dict(counter),
                    'opportunities':dict(opps),'answer_correct':sum(r['answer_correct'] for r in rows),
                    'protocol_valid':sum(r['protocol_valid'] for r in rows),'semantic_valid':sum(r['semantic_valid'] for r in rows),
                    'native_count_differences':comparisons}
                print('RESULT',seed,panel,dict(counter),flush=True)
        result['legacy_error_cross_tab']=dict(result['legacy_error_cross_tab']);doc['seeds'].append(result)
        path.write_text(json.dumps(doc,ensure_ascii=False,indent=2));(OUT/'examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2))
        del compiled,model,blob
        torch.cuda.empty_cache()
    print('DONE',flush=True)


if __name__=='__main__':main()

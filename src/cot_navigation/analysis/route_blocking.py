"""Block EVERY remaining goal route, auditing the entire stack/queue.

Original graph reachability stays YES. The *counterfactual search machine*
must instead exhaust if it obeys the manipulated discovered/visited set.
These are distinct labels and are reported separately.
"""
from __future__ import annotations

import numpy as np

from .common import SEED, records
from .patching import generate_prefixes, reachable, summarize_pairs


def search_from_state(adj, frontier, visited, target, fifo):
    frontier, visited = list(frontier), set(visited)
    for _ in range(4*len(adj)+4):
        if not frontier: return False
        u = frontier[0 if fifo else -1]
        if u == target: return True
        candidates = [v for v in adj[u] if v not in visited]
        if candidates:
            v = candidates[0]; visited.add(v); frontier.append(v)
        elif len(frontier) == 1: return False
        elif fifo: frontier.pop(0)
        else: frontier.pop()
    raise AssertionError('finite DAG search exceeded the proven linear action bound')


def blocking_case(am, ex, tokens, *, exclude_goal=False):
    if ex.label != 1: return None
    inv = {v:i for i,v in enumerate(ex.nodes)}
    parsed = records(tokens, ex.row.prompt_len, am.tok)
    for index in sorted(range(len(parsed)), key=lambda i: abs(i-len(parsed)/2)):
        r = parsed[index]; frontier, visited = r['frontier'], r['visited']
        if not frontier or any(v not in inv for v in frontier+visited): continue
        if len(set(frontier)) != len(frontier) or len(set(visited)) != len(visited): continue
        if not set(frontier) <= set(visited) or ex.nodes[ex.target] in frontier: continue
        if r['action'][0] in {am.tok.id('<FOUND>'),am.tok.id('<EXHAUST>')}: continue
        ff, vv = [inv[v] for v in frontier], [inv[v] for v in visited]
        fifo = am.protocol == 'bfs_state'
        # Do not attribute a pre-existing doomed state to our intervention.
        if not search_from_state(ex.adj, ff, vv, ex.target, fifo): continue
        available = [v for v in ex.nodes if v not in set(visited)]
        useful = [v for v in available if (not exclude_goal or v != ex.nodes[ex.target])
                  and reachable(ex.adj, inv[v], ex.target)
                  and any(reachable(ex.adj, root, inv[v]) for root in ff)]
        if not useful: continue
        if search_from_state(ex.adj, ff, vv+[inv[v] for v in useful], ex.target, fifo): continue
        # Independent reachability audit: keep active frontier roots traversable,
        # remove visited non-frontier and all inserted nodes.
        blocked = (set(vv)-set(ff)) | {inv[v] for v in useful}
        assert not any(reachable(ex.adj,u,ex.target,blocked) for u in ff)
        cut = r['separators'][1]; clean = tokens[:cut+1]
        def insert(nodes): return tokens[:cut]+nodes+tokens[cut:cut+1]
        dead = [v for v in available if not reachable(ex.adj,inv[v],ex.target)]
        # Always-available length control; duplicate VIS entries are OOD, so
        # this is explicitly NOT claimed to be a valid-state semantic control.
        padding = [visited[0]]*len(useful)
        prefixes = {'clean':clean,'all_routes':insert(useful),'duplicate_padding':insert(padding),
                    'all_unvisited':insert(available)}
        if len(dead) >= len(useful): prefixes['dead_distinct_matched'] = insert(dead[:len(useful)])
        if ex.nodes[ex.target] not in visited:
            prefixes['goal_only'] = insert([ex.nodes[ex.target]])
        return {'query':ex.index,'label':1,'prefix_source':'generated','prompt_length':ex.row.prompt_len,
                'prefixes':prefixes,'detail':{'record':index,'frontier':frontier,'visited':visited,
                'inserted':useful,'inserted_count':len(useful),'excludes_goal':exclude_goal,
                'oracle_clean_found':True,'oracle_all_routes_found':False,
                'all_frontier_reachability_blocked':True,
                'distinct_dead_control_available':'dead_distinct_matched' in prefixes}}
    return None


def run_blocking(am, examples, free, *, max_cases=40, exclude_goal=False):
    cases=[]
    for ex in examples:
        case=blocking_case(am,ex,free[ex.index],exclude_goal=exclude_goal)
        if case: cases.append(case)
        if len(cases)>=max_cases: break
    if not cases: return {'status':'no certified eligible states','n_cases':0}
    flat=[]; keys=[]
    for i,case in enumerate(cases):
        for name,prefix in case['prefixes'].items(): flat.append(prefix); keys.append((i,name))
    outs=generate_prefixes(am,flat,max_new=0)
    grouped=[{} for _ in cases]
    for (i,name),out in zip(keys,outs): grouped[i][name]=out
    base=summarize_pairs(cases,grouped,am.tok)
    rng=np.random.default_rng(SEED)
    for name in base['conditions']:
        indices=[i for i,c in enumerate(cases) if name in c['prefixes']]
        rows=[grouped[i][name] for i in indices]
        condition=base['conditions'][name]
        condition['clean_accuracy_same_cases']=float(np.mean([grouped[i]['clean']['pred']==1 for i in indices]))
        condition['counterfactual_NO_rate']=float(np.mean([o['pred']==0 for o in rows]))
        if name in {'all_routes','all_unvisited'}:
            violations=[];retained=[]
            for i,out in zip(indices,rows):
                prefix=cases[i]['prefixes'][name]
                inserted=set(cases[i]['detail']['inserted']) if name=='all_routes' else set(cases[i]['detail']['inserted'])
                generated=out['generated']
                violations.append(any(a==am.tok.id('<PUSH>') and b in inserted for a,b in zip(generated,generated[1:])))
                future=[r for r in records(out['tokens'],cases[i]['prompt_length'],am.tok) if r['start']>=len(prefix)]
                if future: retained.append(float(inserted<=set(future[0]['visited'])))
            condition['pushed_blocked_route_node_rate']=float(np.mean(violations))
            condition['all_inserted_retained_next_record']=float(np.mean(retained)) if retained else None
        if name not in {'clean','all_routes'}:
            delta=np.asarray([int(grouped[i]['all_routes']['pred']==1)-int(grouped[i][name]['pred']==1) for i in indices])
            boots=delta[rng.integers(len(delta),size=(2000,len(delta)))].mean(-1)
            base['paired_contrasts']['all_routes_minus_'+name]={'n':len(indices),'original_answer_accuracy_difference':float(delta.mean()),
                                                                'ci95':np.quantile(boots,[.025,.975]).tolist()}
    return {'status':'complete','n_cases':len(cases),'query_ids':[c['query'] for c in cases],
            'oracle_blocking_verified_count':len(cases),'exclude_goal':exclude_goal,
            'mean_inserted_nodes':float(np.mean([c['detail']['inserted_count'] for c in cases])),
            'distinct_dead_control_n':sum(c['detail']['distinct_dead_control_available'] for c in cases),
            **base}

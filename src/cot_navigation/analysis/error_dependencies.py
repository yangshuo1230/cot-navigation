"""Structural links in unedited DFS text, including locally followed bad actions.

Local consistency after an illegal action is NOT global graph correctness.
Revisits are scored only while graph-legal execution remains reconstructible.
"""


def parse_records(trace,tok):
    begin=0;out=[]
    for end,t in enumerate(trace):
        if t!=tok.id(';'):continue
        groups=[[]]
        for v in trace[begin:end]:
            if v==tok.id('|'):groups.append([])
            else:groups[-1].append(v)
        out.append(dict(groups=groups,start=begin,end=end+1));begin=end+1
    return out


def dependencies(trace,adj,src,dst,tok):
    rr=parse_records(trace,tok)
    marker=tok.stoi.get('<MISTAKE>',-1)
    def isnode(t):return tok.node_lo<=t<tok.node_lo+tok.n_node_tokens
    def declared(groups):
        if len(groups)!=4 or not groups[0] or not all(isnode(t) for seq in groups[:2] for t in seq):return None
        return (list(groups[0]),set(groups[1]))
    def literal(state,action):
        if state is None:return None
        stack,visited=state;stack=stack[:];visited=set(visited)
        if action==[marker]:return stack,visited
        if len(action)==2 and action[0]==tok.id('<PUSH>') and isnode(action[1]):
            stack.append(action[1]);visited.add(action[1]);return stack,visited
        if action==[tok.id('<POP>')] and len(stack)>1:
            stack.pop();return stack,visited
        return None
    def compare(groups,state):
        if state is None or len(groups)!=4:return None
        stack,visited=state;eligible=set(adj.get(stack[-1],[]))-visited
        wanted='<FOUND>' if stack[-1]==dst else '<PUSH>' if eligible else '<POP>' if len(stack)>1 else '<EXHAUST>'
        action=groups[3]
        legal=action==[marker] or (len(action)==(2 if wanted=='<PUSH>' else 1) and action[0]==tok.id(wanted) and (wanted!='<PUSH>' or action[1] in eligible))
        result=dict(frontier=groups[0]==stack,visited=set(groups[1])==visited and len(groups[1])==len(set(groups[1])),
                    candidate=set(groups[2])==eligible and len(groups[2])==len(set(groups[2])),action=legal)
        result['all']=all(result.values());return result

    state=([src],{src});pending={};revisits=[];transitions=[];bad_action=None;links=[]
    extra_anchors=0
    for i,record in enumerate(rr):
        groups=record['groups']
        if len(groups)!=4:
            bad_action=dict(record=i+1,kind='format',local=None,reject=state,printed=None);break
        stack,visited=state;cur=stack[-1];candidate=groups[2];action=groups[3]
        eligible=set(adj.get(cur,[]))-visited
        checks=compare(groups,state)
        if cur in pending:
            previous=pending.pop(cur)
            # Pending was created only on a legal PUSH. Same stack means a
            # genuine return to that ancestor, not another MISTAKE retry.
            if stack==previous['stack']:
                extras=set(candidate)-eligible
                still_missing=previous['missing']&eligible
                revisits.append(dict(node=cur,anchor_record=previous['record'],return_record=i+1,
                    anchor_token=previous['token'],return_token=record['start'],
                    original_candidate_bad=bool(previous['extras'] or previous['missing'] or previous['duplicates']),
                    original_unused_extras=sorted(previous['extras']),returned_extras=sorted(extras),
                    repeated_unused_extras=sorted(previous['extras']&extras),
                    missing_still_needed=len(still_missing),repeated_missing=len(still_missing-set(candidate)),
                    candidate_bad=not checks['candidate'],
                    previously_clean=not previous['extras'] and not previous['missing'] and not previous['duplicates']))
        if len(action)==2 and action[0]==tok.id('<PUSH>'):
            links.append(dict(record=i+1,node=cur,chosen=action[1],chosen_illegal=action[1] not in eligible,
                chosen_in_printed_candidates=action[1] in candidate,printed_candidate_bad=not checks['candidate']))
        graph_legal=checks['action']
        after=literal(state,action)
        if not graph_legal:
            bad_action=dict(record=i+1,kind='push' if action and action[0]==tok.id('<PUSH>') else 'operator',
                local=after,reject=state,printed=literal(declared(groups),action));break
        if i+1<len(rr) and after is not None:
            own_after=literal(declared(groups),action)
            next_groups=rr[i+1]['groups']
            if own_after is not None and len(next_groups)==4:
                for field,fi in [('frontier',0),('visited',1)]:
                    if checks[field]:continue
                    def matches(state):
                        expected=state[fi]
                        return next_groups[fi]==expected if fi==0 else set(next_groups[fi])==expected and len(next_groups[fi])==len(set(next_groups[fi]))
                    true_match=matches(after);own_match=matches(own_after)
                    transitions.append(dict(record=i+1,field=field,cancel=action==[marker],
                        following=('both' if true_match and own_match else 'true_state' if true_match else 'printed_wrong_state' if own_match else 'neither')))
        if action==[marker]:continue
        if action and action[0]==tok.id('<PUSH>') and graph_legal:
            extra_anchors+=bool(set(candidate)-eligible)
            pending[cur]=dict(record=i+1,stack=stack[:],token=record['start'],
                extras=set(candidate)-eligible,missing=eligible-set(candidate),duplicates=len(candidate)-len(set(candidate)))
        if after is None:break  # natural FOUND/EXHAUST
        state=after
    local=[]
    if bad_action is not None:
        executed=bad_action['local'];own=bad_action['printed'];reject=bad_action['reject']
        for offset in [1,2]:
            j=bad_action['record']-1+offset
            if j>=len(rr):break
            groups=rr[j]['groups']
            local.append(dict(offset=offset,first_illegal_kind=bad_action['kind'],
                follows_executed=compare(groups,executed),follows_printed=compare(groups,own),
                follows_rejected=compare(groups,reject)))
            action=groups[3] if len(groups)==4 else []
            executed=literal(executed,action);own=literal(own,action);reject=literal(reject,action)
    return dict(schema=1,revisits=revisits,annotation_transitions=transitions,
        after_illegal=local,candidate_action_links=links,
        first_illegal_record=None if bad_action is None else bad_action['record'],
        unavailable_revisits=len(pending),unused_extra_anchors=extra_anchors,
        unused_extra_without_observed_return=sum(bool(x['extras']) for x in pending.values()))

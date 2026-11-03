"""Token-resolved DFS/BFS replay with cancellable MISTAKE records.

Field order: frontier | visited | candidates | action ;. Visited/candidate
permutations are allowed. After an illegal true-graph action, later state
annotations are not independently scored. Position is 1-based in the CoT.
"""
from collections import Counter

STEP_BINS = [(1,1),(2,2),(3,4),(5,8),(9,16),(17,32),(33,10**9)]


def bucket(step):
    return next(i for i,(lo,hi) in enumerate(STEP_BINS) if lo<=step<=hi)


def replay(trace, pred, adj, src, dst, tok, *, prompt_len=0, interrupted=False,
           fifo=False, label=None):
    """Audit token IDs, without requiring a canonical traversal or future trace.

    label defaults to independent graph reachability. ``stack_depth`` denotes
    frontier size for BFS. ``risk_steps_by_bin`` counts distinct true execution
    states before the first error; retries do not inflate the denominator.
    """
    body=list(trace)
    stop=tok.stoi.get('<STOP>')
    if body and body[-1]==stop:body.pop()
    nodes=set(adj)
    for vs in adj.values():nodes.update(vs)
    nodes.update([src,dst])
    reverse={u:[] for u in nodes}
    for u,vs in adj.items():
        for v in vs:reverse[v].append(u)
    useful={dst};todo=[dst]
    while todo:
        for u in reverse[todo.pop()]:
            if u not in useful:useful.add(u);todo.append(u)
    bar=tok.id('|');semi=tok.id(';');mistake=tok.stoi.get('<MISTAKE>',-999)
    stack=[src];visited={src};terminal=None;exec_ok=True;effective=True
    first=None;first_bad=None;pending=None;first_detected=False;first_repaired=False
    actions=0;records=0;complete_records=0;counts=Counter();risk=set()

    def event(kind,pos,field):
        return dict(kind=kind,token=int(pos+1),field=field,record=records,
                    search_step=actions+1,stack_depth=len(stack),visited_size=len(visited),
                    current=stack[0 if fifo else -1] if stack else None,
                    at_goal=bool(stack and stack[0 if fifo else -1]==dst),
                    context_fraction=(prompt_len+pos+1)/4096)

    segments=[];begin=0
    for i,t in enumerate(body):
        if t==semi:segments.append((begin,i,True));begin=i+1
    if begin<len(body):segments.append((begin,len(body),False))
    for begin,end,complete in segments:
        records+=1;complete_records+=int(complete)
        if not exec_ok:continue
        if first is None:risk.add(actions+1)
        if terminal or not stack:
            e=event('format_after_terminal',begin,'format')
            if first is None:first=e
            if first_bad is None:first_bad=e
            effective=exec_ok=False;continue
        cur=stack[0 if fifo else -1];eligible=set(adj.get(cur,[]))-visited
        groups=[];start=begin
        for pos in range(begin,end):
            if body[pos]==bar:
                groups.append((body[start:pos],start,pos,True));start=pos+1
        groups.append((body[start:end],start,end,False))
        errors=[]
        # Each field's closing | makes omissions observable; a valid unfinished
        # prefix without | is censored, not falsely labelled as an omission.
        for fi,(seq,start,endpos,closed) in enumerate(groups[:3]):
            field=['frontier','visited','candidate'][fi]
            seen=set();expected=stack if fi==0 else visited if fi==1 else eligible
            for j,t in enumerate(seq):
                if not tok.node_lo<=t<tok.node_lo+tok.n_node_tokens:
                    errors.append(event(field+'_format',start+j,field));break
                kind=None
                if fi==0:
                    if j>=len(stack):kind='frontier_extra'
                    elif t!=stack[j]:kind='frontier_wrong_order' if t in stack else 'frontier_wrong_node'
                elif t in seen:kind=field+'_duplicate'
                elif t not in expected:
                    if t not in nodes:kind=field+'_out_of_graph'
                    elif fi==2 and t in visited:kind='candidate_visited'
                    elif fi==2:kind='candidate_nonedge'
                    else:kind='visited_extra'
                if kind:errors.append(event(kind,start+j,field));break
                seen.add(t)
            if closed:
                missing=(len(seq)<len(stack)) if fi==0 else bool(set(expected)-set(seq))
                if missing:errors.append(event(field+'_omission',endpos,field))
        cand=groups[2][0] if len(groups)>2 else []
        cand_closed=bool(len(groups)>2 and groups[2][3])
        candidate_features=dict(missing=len(eligible-set(cand)) if cand_closed else None,
            missing_useful=len((eligible-set(cand))&useful) if cand_closed else None,
            missing_deadend=len((eligible-set(cand))-useful) if cand_closed else None,
            extra=len(set(cand)-eligible),extra_visited=len(set(cand)&visited),
            extra_nonedge=len(set(cand)-set(adj.get(cur,[]))),
            duplicate=len(cand)-len(set(cand)),eligible=len(eligible),
            useful_eligible=len(eligible&useful),closed=cand_closed)
        for e in errors:
            if e['field']=='candidate':e['candidate_features']=candidate_features
        fields_ok=not errors and len(groups)>=4
        action=groups[3][0] if len(groups)>=4 else []
        action_start=groups[3][1] if len(groups)>=4 else end
        cancel=complete and len(groups)==4 and action==[mistake]
        legal=False;priority=True;wanted=None
        if complete:
            if len(groups)<4:errors.append(event('format_missing_separator',end,'format'))
            elif len(groups)>4:errors.append(event('format_extra_separator',groups[3][2],'format'))
            elif cancel:pass
            else:
                wanted='<FOUND>' if cur==dst else '<PUSH>' if eligible else '<POP>' if len(stack)>1 else '<EXHAUST>'
                op=tok.id(wanted);required=2 if wanted=='<PUSH>' else 1
                if not action:errors.append(event('action_missing',end,'action'))
                elif action[0]!=op:errors.append(event('action_wrong_operator',action_start,'action'))
                elif len(action)<required:errors.append(event('action_missing_argument',end,'action'))
                else:
                    legal=True
                    if wanted=='<PUSH>':
                        v=action[1]
                        if v not in eligible:
                            legal=False
                            kind='action_push_out_of_graph' if v not in nodes else 'action_push_visited' if v in visited else 'action_push_nonedge'
                            errors.append(event(kind,action_start+1,'action'))
                        priority=bool(cand) and v==cand[0]
                        if legal and not priority:errors.append(event('action_candidate_priority',action_start+1,'action'))
                    if len(action)>required:
                        legal=False;errors.append(event('action_extra_tokens',action_start+required,'action'))
        elif len(groups)>=4 and action:
            # Do not overlook an already wrong action token just because the
            # model answered/stopped before emitting this record's semicolon.
            wanted='<FOUND>' if cur==dst else '<PUSH>' if eligible else '<POP>' if len(stack)>1 else '<EXHAUST>'
            if len(groups)>4:errors.append(event('format_extra_separator',groups[3][2],'format'))
            if action[0] not in (tok.id(wanted),mistake):
                errors.append(event('action_wrong_operator',action_start,'action'))
            elif action[0]==tok.id('<PUSH>') and len(action)>1:
                v=action[1]
                if v not in eligible:
                    kind='action_push_out_of_graph' if v not in nodes else 'action_push_visited' if v in visited else 'action_push_nonedge'
                    errors.append(event(kind,action_start+1,'action'))
                elif not cand or v!=cand[0]:errors.append(event('action_candidate_priority',action_start+1,'action'))
            required=2 if action[0]==tok.id('<PUSH>') else 1
            if len(action)>required:errors.append(event('action_extra_tokens',action_start+required,'action'))
        # Incomplete records can still already contain illegal emitted nodes.
        if errors:
            e=min(errors,key=lambda e:e['token'])
            if first is None:first=e
            if not cancel and first_bad is None:first_bad=e
            if not cancel:effective=False
        if not complete:
            effective=False
            continue
        if cancel:
            if errors:
                counts['true_marks']+=1
                if first and first['record']==records:first_detected=True
                if pending is None:pending=dict(record=records,first=bool(first and first['record']==records))
            else:counts['false_marks']+=1
            counts['auditable_marks']+=1
            continue
        if pending is not None:
            if fields_ok and legal and priority:
                counts['repairs']+=1
                if pending['first']:first_repaired=True
            else:counts['failed_repairs']+=1
            pending=None
        if not legal:
            exec_ok=False;effective=False
        else:
            if wanted=='<PUSH>':stack.append(action[1]);visited.add(action[1])
            elif wanted=='<POP>':stack.pop(0 if fifo else -1)
            else:terminal=wanted[1:-1]
            actions+=1
    partial=bool(segments and not segments[-1][2])
    valid=bool(effective and exec_ok and terminal in ('FOUND','EXHAUST') and not partial and pending is None)
    execution_complete=bool(exec_ok and terminal in ('FOUND','EXHAUST') and not partial)
    label=int(src in useful) if label is None else label
    correct=(pred==label)
    answer_consistent=(terminal=='FOUND' and pred==1) or (terminal=='EXHAUST' and pred==0)
    valid_answer=valid and answer_consistent
    literal=valid and first is None
    if not correct:outcome='wrong_answer'
    elif literal:outcome='correct_clean_valid'
    elif valid:outcome='correct_repaired_valid'
    elif execution_complete and answer_consistent:outcome='correct_execution_fields_invalid'
    else:outcome='correct_incomplete_or_illegal'
    if first is not None:first_outcome=first['kind']
    elif valid:first_outcome='no_error_complete'
    elif interrupted:first_outcome='no_error_prefix_budget_cutoff'
    else:first_outcome='no_error_prefix_early_end'
    risks=[0]*len(STEP_BINS)
    for step in risk:risks[bucket(step)]+=1
    return dict(first=first,first_uncancelled=first_bad,first_outcome=first_outcome,
        first_detected=first_detected,first_repaired=first_repaired,
        trace_valid=valid,literal_valid=literal,answer_and_trace_valid=bool(valid_answer),
        execution_complete=execution_complete,terminal=terminal,correct=correct,outcome=outcome,
        true_marks=counts['true_marks'],false_marks=counts['false_marks'],repairs=counts['repairs'],
        failed_repairs=counts['failed_repairs'],raw_mistake_tokens=list(trace).count(mistake),
        records=records,complete_records=complete_records,actions=actions,
        risk_steps_by_bin=risks,partial_record=partial,trace_length=len(trace))


def self_test(tok, old_audit=None):
    """Small deterministic cases; optional independent compatibility oracle."""
    from cot_navigation.protocols.trace_audit import validate_states
    a,b,c,d=[tok.node_id(i) for i in range(4)]
    bar=tok.id('|');semi=tok.id(';')
    def rec(s,v,cs,op):return s+[bar]+v+[bar]+cs+[bar]+op+[semi]
    push=rec([a],[a],[b],[tok.id('<PUSH>'),b]);found=rec([a,b],[a,b],[],[tok.id('<FOUND>')])
    cases=[]
    cases.append((push+found,'no_error_complete'))
    for cand,kind in [([c],'candidate_nonedge'),([a],'candidate_visited'),([b,b],'candidate_duplicate'),([], 'candidate_omission')]:
        cases.append((rec([a],[a],cand,[tok.id('<PUSH>'),b])+found,kind))
    cases.append((rec([a],[a,a],[b],[tok.id('<PUSH>'),b])+found,'visited_duplicate'))
    cases.append((rec([a],[a,c],[b],[tok.id('<PUSH>'),b])+found,'visited_extra'))
    cases.append((rec([a],[],[b],[tok.id('<PUSH>'),b])+found,'visited_omission'))
    cases.append((rec([c],[a],[b],[tok.id('<PUSH>'),b])+found,'frontier_wrong_node'))
    cases.append((rec([a],[a],[b],[tok.id('<POP>')]),'action_wrong_operator'))
    if '<MISTAKE>' in tok.stoi:
        wrong=rec([a],[a],[c],[tok.id('<MISTAKE>')])
        x=replay(wrong+push+found,1,{a:[b],b:[],c:[]},a,b,tok)
        assert x['trace_valid'] and x['first_detected'] and x['first_repaired'] and x['outcome']=='correct_repaired_valid'
        cases.append((wrong+push+found,'candidate_nonedge'))
    for trace,kind in cases:
        adj={a:[b],b:[],c:[]}
        x=replay(trace,1,adj,a,b,tok)
        assert x['first_outcome']==kind,(x,kind)
        if old_audit is not None:
            assert x['answer_and_trace_valid']==old_audit(trace,1,adj,a,b,tok,False)['effective_complete']
        elif tok.stoi.get('<MISTAKE>',-999) not in trace:
            assert x['answer_and_trace_valid']==validate_states(trace,1,adj,a,b,tok,False)['valid']
    # Arbitrary legal candidate ordering is accepted; pushing the second printed
    # node instead of the first is a protocol priority error, not an invalid edge.
    alt=rec([a],[a],[c,b],[tok.id('<PUSH>'),c])+rec([a,c],[a,c],[],[tok.id('<FOUND>')])
    assert replay(alt,1,{a:[b,c],b:[],c:[]},a,c,tok)['literal_valid']
    pri=rec([a],[a],[b,c],[tok.id('<PUSH>'),c])
    assert replay(pri,1,{a:[b,c],b:[],c:[]},a,c,tok)['first_outcome']=='action_candidate_priority'
    # A valid incomplete prefix is censored; a prematurely closed candidate set is not.
    prefix=[a,bar,a,bar]
    assert replay(prefix,1,{a:[b],b:[]},a,b,tok,interrupted=True)['first_outcome']=='no_error_prefix_budget_cutoff'
    assert replay(prefix+[bar],1,{a:[b],b:[]},a,b,tok)['first_outcome']=='candidate_omission'
    return len(cases)+5

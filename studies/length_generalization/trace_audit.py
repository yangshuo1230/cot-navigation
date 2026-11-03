"""Semantic validators independent of reference DFS/BFS traversal order.

Full states must describe the actual graph and evolve under their own actions.
Alternative legal neighbor priorities are accepted. Node-only outputs denote
first expansions, NOT a path: DFS backtracking and BFS siblings need not be
joined by an edge. Answer correctness is reported separately by the caller.
"""
from collections import deque


def validate_trace(tokens, answer, protocol, adj, src, dst, tok, negative_path=False):
    """All graph nodes and trace node IDs use tokenizer IDs, not topology IDs."""
    if protocol == 'direct':
        return {'valid': not tokens, 'error': None if not tokens else 'unexpected_trace'}
    if protocol in ('dfs_state','bfs_state'):
        return validate_states(tokens,answer,adj,src,dst,tok,protocol=='bfs_state')
    if protocol in ('dfs_nodes','bfs_nodes'):
        return validate_nodes(tokens,answer,adj,src,dst,tok,protocol=='bfs_nodes')
    if protocol != 'path':
        raise ValueError(protocol)
    def fail(reason):return {'valid':False,'error':reason}
    if not tokens:
        return {'valid': answer==0 and not negative_path,
                'error':None if answer==0 and not negative_path else 'empty_path'}
    if any(v not in adj for v in tokens):return fail('non_node_token')
    if tokens[0]!=src:return fail('wrong_start')
    if len(tokens)!=len(set(tokens)):return fail('repeated_node')
    if any(v not in adj[u] for u,v in zip(tokens,tokens[1:])):return fail('non_edge')
    if answer==1 and tokens[-1]!=dst:return fail('wrong_positive_endpoint')
    if answer==0:
        if not negative_path:return fail('unexpected_negative_path')
        if adj[tokens[-1]]:return fail('negative_path_not_at_sink')
    if answer not in (0,1):return fail('missing_answer')
    return {'valid':True,'error':None}


def validate_states(tokens,answer,adj,src,dst,tok,fifo):
    frontier,visited=[src],[src]
    start,records=0,0
    terminal=None
    for end,token in enumerate(tokens):
        if token!=tok.id(';'):continue
        record=tokens[start:end];start=end+1
        def fail(reason):return {'valid':False,'error':reason,'valid_records':records,
                                 'valid_prefix_tokens':end-len(record)}
        if terminal is not None:return fail('record_after_terminal')
        fields=[[]]
        for x in record:
            if x==tok.id('|'):fields.append([])
            else:fields[-1].append(x)
        if len(fields)!=4:return fail('malformed_fields')
        shown,vis,cands,action=fields
        if shown!=frontier:return fail('frontier_mismatch')
        if vis!=visited:return fail('visited_mismatch')
        cur=frontier[0] if fifo else frontier[-1]
        eligible=set(adj[cur])-set(visited)
        if len(cands)!=len(set(cands)) or set(cands)!=eligible:return fail('candidate_mismatch')
        expected='FOUND' if cur==dst else 'PUSH' if eligible else 'POP' if len(frontier)>1 else 'EXHAUST'
        if not action or action[0]!=tok.id('<'+expected+'>'):return fail('wrong_action')
        if len(action)!=(2 if expected=='PUSH' else 1):return fail('malformed_action')
        if expected=='PUSH':
            if action[1]!=cands[0]:return fail('push_not_first_candidate')
            frontier.append(action[1]);visited.append(action[1])
        elif expected=='POP':frontier.pop(0 if fifo else -1)
        else:terminal=expected
        records+=1
    if start!=len(tokens):
        return {'valid':False,'error':'unfinished_record','valid_records':records,'valid_prefix_tokens':start}
    valid=(terminal=='FOUND' and answer==1) or (terminal=='EXHAUST' and answer==0)
    return {'valid':valid,'error':None if valid else 'missing_or_inconsistent_terminal',
            'valid_records':records,'valid_prefix_tokens':len(tokens)}


def validate_nodes(tokens,answer,adj,src,dst,tok,fifo):
    def fail(reason):return {'valid':False,'error':reason}
    if not tokens:return fail('empty_nodes')
    if any(v not in adj for v in tokens):return fail('non_node_token')
    if tokens[0]!=src:return fail('wrong_start')
    if len(tokens)!=len(set(tokens)):return fail('repeated_node')
    if dst in tokens[:-1]:return fail('continued_after_target')
    if fifo:
        # Choose each sibling enqueue order by the claimed future expansion
        # ranks. If ANY BFS order realizes the trace, this order realizes it.
        rank={v:i for i,v in enumerate(tokens)}
        queue,discovered=deque([src]),{src}
        for node in tokens:
            if not queue or queue.popleft()!=node:return fail('not_bfs_order')
            if node==dst:break
            cands=set(adj[node])-discovered
            queue.extend(sorted(cands,key=lambda v:(rank.get(v,len(tokens)),v)))
            discovered.update(cands)
        exhausted=not queue and tokens[-1]!=dst
    else:
        stack,seen=[src],{src}
        for node in tokens[1:]:
            while stack and not (set(adj[stack[-1]])-seen):stack.pop()
            if not stack or node not in adj[stack[-1]]:return fail('not_dfs_order')
            stack.append(node);seen.add(node)
        while stack and not (set(adj[stack[-1]])-seen):stack.pop()
        exhausted=not stack and tokens[-1]!=dst
    if answer==1 and tokens[-1]==dst:return {'valid':True,'error':None}
    if answer==0 and exhausted:return {'valid':True,'error':None}
    return fail('missing_or_inconsistent_terminal')


def validate_execution(tokens,answer,adj,src,dst,tok,fifo):
    """Replay only actions; ignore displayed frontier/visited/candidate fields.

    Separates wrong state annotations (including irrelevant goal candidates)
    from actual invalid search moves. An executable skeleton still need not
    match the particular reference priority order.
    """
    frontier,seen=[src],{src}
    start,terminal=0,None
    actions={tok.id('<'+a+'>'):a for a in ('PUSH','POP','FOUND','EXHAUST')}
    for end,x in enumerate(tokens):
        if x!=tok.id(';'):continue
        rec=tokens[start:end];start=end+1
        if terminal:return False
        ids=[i for i,v in enumerate(rec) if v in actions]
        if len(ids)!=1:return False
        tail=rec[ids[0]:];name=actions[tail[0]]
        if len(tail)!=(2 if name=='PUSH' else 1):return False
        cur=frontier[0] if fifo else frontier[-1]
        candidates=set(adj[cur])-seen
        expected='FOUND' if cur==dst else 'PUSH' if candidates else 'POP' if len(frontier)>1 else 'EXHAUST'
        if name!=expected:return False
        if name=='PUSH':
            if tail[1] not in candidates:return False
            seen.add(tail[1]);frontier.append(tail[1])
        elif name=='POP':frontier.pop(0 if fifo else -1)
        else:terminal=name
    return start==len(tokens) and ((terminal=='FOUND' and answer==1) or (terminal=='EXHAUST' and answer==0))


def self_test(tok,dfs_search,bfs_search,render_steps,first_expansion_order):
    """Exhaustive 4-node DAGs, both searches, every distinct endpoint pair."""
    import itertools
    pairs=list(itertools.combinations(range(4),2));count=0
    nodes=[tok.node_id(i) for i in range(4)]
    for mask in range(64):
        raw=[[] for _ in nodes]
        for i,(u,v) in enumerate(pairs):
            if mask>>i&1:raw[u].append(v)
        adj={nodes[u]:[nodes[v] for v in vs] for u,vs in enumerate(raw)}
        for s,t in itertools.permutations(range(4),2):
            for fifo,search in ((False,dfs_search),(True,bfs_search)):
                steps=search(raw,s,t);answer=int(steps[-1].action=='FOUND')
                trace=render_steps(steps,nodes,tok)
                assert validate_states(trace,answer,adj,nodes[s],nodes[t],tok,fifo)['valid']
                expansion=[nodes[v] for v in first_expansion_order(steps)]
                assert validate_nodes(expansion,answer,adj,nodes[s],nodes[t],tok,fifo)['valid']
                # A wrong terminal answer must fail semantic validation.
                assert not validate_states(trace,1-answer,adj,nodes[s],nodes[t],tok,fifo)['valid']
                assert validate_execution(trace,answer,adj,nodes[s],nodes[t],tok,fifo)
                assert not validate_execution(trace,1-answer,adj,nodes[s],nodes[t],tok,fifo)
                count+=5
    return count

from cot_navigation.analysis.error_continuation import audit_continuation


class Tok:
    def __init__(self):
        self.stoi={s:i for i,s in enumerate(['|',';','<MISTAKE>','<PUSH>','<POP>','<FOUND>','<EXHAUST>'])}
        self.node_lo=10;self.n_node_tokens=10
    def id(self,s):return self.stoi[s]


def test_natural_wrong_fields_do_not_corrupt_reference():
    t=Tok()
    def rec(s,v,c,a):return s+[t.id('|')]+v+[t.id('|')]+c+[t.id('|')]+a+[t.id(';')]
    # Wrong candidates, legal graph action, then correct state: no MISTAKE needed.
    x=rec([10],[10],[12],[t.id('<PUSH>'),11])+rec([10,11],[10,11],[],[t.id('<FOUND>')])
    r=audit_continuation(x,{10:[11],11:[],12:[]},10,11,t)['records']
    assert len(r)==2 and r[0]['bad'] and r[0]['action_legal']
    assert r[1]['prior_bad_records']==1 and not r[1]['bad'] and r[1]['unmarked_recovery']


def test_cancel_and_illegal_action_censoring():
    t=Tok()
    def rec(c,a):return [10,t.id('|'),10,t.id('|')]+c+[t.id('|')]+a+[t.id(';')]
    x=rec([12],[t.id('<MISTAKE>')])+rec([11],[t.id('<PUSH>'),11])
    r=audit_continuation(x,{10:[11],11:[],12:[]},10,11,t)['records']
    assert r[0]['true_cancel'] and r[1]['prior_bad_records']==1 and r[1]['prior_unresolved']==0
    x=rec([12],[t.id('<PUSH>'),12])+rec([11],[t.id('<PUSH>'),11])
    r=audit_continuation(x,{10:[11],11:[],12:[]},10,11,t)
    assert len(r['records'])==1 and r['stopped_on_illegal_action']
    assert r['records'][0]['illegal_push_from_current_candidates']


def test_multiple_errors_and_legal_reordering():
    t=Tok()
    bar,semi=t.id('|'),t.id(';')
    def rec(c):return [10,bar,10,bar]+c+[bar,t.id('<MISTAKE>'),semi]
    x=rec([12,12])+rec([12])+[10,bar,10,bar,11,bar,t.id('<PUSH>'),11,semi]
    r=audit_continuation(x,{10:[11],11:[],12:[]},10,11,t)['records']
    assert r[1]['copied_bad_nodes']==1 and r[2]['prior_bad_records']==2
    x=[10,bar,10,bar,12,11,bar,t.id('<PUSH>'),12,semi]
    assert not audit_continuation(x,{10:[11,12],11:[],12:[]},10,11,t)['records'][0]['bad']


def test_first_error_agrees_with_existing_semantic_replayer():
    from cot_navigation.protocols.correction_replay import replay
    from cot_navigation.protocols.tokenizer import Tokenizer
    tok=Tokenizer(32,enable_mistake=True)
    a,b,c,out=[tok.node_id(i) for i in range(4)]
    adj={a:[b],b:[c],c:[]}
    def rec(stack,visited,candidates,action):
        return stack+[tok.id('|')]+visited+[tok.id('|')]+candidates+[tok.id('|')]+action+[tok.id(';')]
    clean=[rec([a],[a],[b],[tok.id('<PUSH>'),b]),
           rec([a,b],[a,b],[c],[tok.id('<PUSH>'),c]),
           rec([a,b,c],[a,b,c],[],[tok.id('<FOUND>')])]
    for i in [0,1]:
        stack=[a] if i==0 else [a,b]
        visited=stack[:];candidate=b if i==0 else c
        for wrong in [[],[out],[candidate,candidate],[a], [candidate,out]]:
            for cancel in [False,True]:
                bad=rec(stack,visited,wrong,[tok.id('<MISTAKE>')] if cancel else [tok.id('<PUSH>'),candidate])
                trace=sum(clean[:i],[])+bad+(clean[i] if cancel else [])+sum(clean[i+1:],[])
                observed=audit_continuation(trace,adj,a,c,tok)
                reference=replay(trace,1,adj,a,c,tok)
                first=next(r for r in observed['records'] if r['bad'])
                assert first['record']==reference['first']['record']==i+1
                assert not observed['stopped_on_illegal_action']

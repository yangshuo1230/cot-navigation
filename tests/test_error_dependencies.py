from cot_navigation.analysis.error_dependencies import dependencies
from cot_navigation.protocols.tokenizer import Tokenizer


def fixture():
    t=Tokenizer(16,enable_mistake=True)
    a,b,c,d=[t.node_id(i) for i in range(4)]
    def rec(s,v,cs,action):return s+[t.id('|')]+v+[t.id('|')]+cs+[t.id('|')]+action+[t.id(';')]
    return t,a,b,c,d,rec


def test_unused_bad_candidate_reappears_after_backtrack():
    t,a,b,c,d,r=fixture();adj={a:[b,c],b:[],c:[]}
    trace=r([a],[a],[b,c,d],[t.id('<PUSH>'),b])+r([a,b],[a,b],[],[t.id('<POP>')])+r([a],[a,b],[c,d],[t.id('<PUSH>'),c])
    x=dependencies(trace,adj,a,c,t)['revisits']
    assert len(x)==1 and x[0]['repeated_unused_extras']==[d]
    assert x[0]['return_record']==3 and x[0]['anchor_record']==1


def test_used_missing_candidate_not_counted_as_recurrent_omission():
    t,a,b,c,d,r=fixture();adj={a:[b,c],b:[],c:[]}
    trace=r([a],[a],[c],[t.id('<PUSH>'),b])+r([a,b],[a,b],[],[t.id('<POP>')])+r([a],[a,b],[c],[t.id('<PUSH>'),c])
    x=dependencies(trace,adj,a,c,t)['revisits'][0]
    assert x['missing_still_needed']==0 and x['repeated_missing']==0


def test_after_bad_push_local_consistency_is_not_global_correctness():
    t,a,b,c,d,r=fixture();adj={a:[b],b:[]}
    trace=r([a],[a],[d],[t.id('<PUSH>'),d])+r([a,d],[a,d],[],[t.id('<POP>')])+r([a],[a,d],[b],[t.id('<PUSH>'),b])
    x=dependencies(trace,adj,a,b,t)
    assert x['first_illegal_record']==1
    assert x['after_illegal'][0]['follows_executed']['all']
    assert not x['after_illegal'][0]['follows_rejected']['frontier']
    assert x['after_illegal'][1]['follows_executed']['visited']


def test_wrong_printed_stack_is_followed_or_recovered():
    t,a,b,c,d,r=fixture();adj={a:[b],b:[c],c:[]}
    prefix=r([d],[a],[b],[t.id('<PUSH>'),b])
    wrong=prefix+r([d,b],[a,b],[c],[t.id('<PUSH>'),c])
    fixed=prefix+r([a,b],[a,b],[c],[t.id('<PUSH>'),c])
    assert dependencies(wrong,adj,a,c,t)['annotation_transitions'][0]['following']=='printed_wrong_state'
    assert dependencies(fixed,adj,a,c,t)['annotation_transitions'][0]['following']=='true_state'


def test_mistake_retry_is_not_backtrack():
    t,a,b,c,d,r=fixture();adj={a:[b],b:[]}
    trace=r([a],[a],[d],[t.id('<MISTAKE>')])+r([a],[a],[b],[t.id('<PUSH>'),b])
    assert not dependencies(trace,adj,a,b,t)['revisits']

"""Regression for seed-independent standard reporting, no model generation."""
import importlib.util
from pathlib import Path
from cot_navigation.analysis.error_continuation import audit_continuation
from cot_navigation.analysis.error_dependencies import dependencies
from cot_navigation.protocols.correction_replay import replay
from cot_navigation.protocols.tokenizer import Tokenizer


def test_standard_report_supports_arbitrary_seed_and_censored_history():
    path=Path(__file__).resolve().parents[1]/'studies/correction_behavior/report.py'
    spec=importlib.util.spec_from_file_location('behavior_report_test',path)
    report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)
    t=Tokenizer(8,enable_mistake=True);a,b,c=[t.node_id(i) for i in range(3)]
    adj={a:[b],b:[],c:[]}
    def rec(s,v,cs,op):return s+[t.id('|')]+v+[t.id('|')]+cs+[t.id('|')]+op+[t.id(';')]
    trace=rec([a],[a],[c],[t.id('<MISTAKE>')])+rec([a],[a],[b],[t.id('<PUSH>'),b])+rec([a,b],[a,b],[],[t.id('<FOUND>')])
    shared=replay(trace,1,adj,a,b,t)
    row=dict(seed=7,query=1,distance=1,correct=True,interrupted=False,trace_valid=shared['trace_valid'],first=shared['first'],
        true_marks=shared['true_marks'],false_marks=shared['false_marks'],repairs=shared['repairs'],trace_tokens=len(trace),
        dependencies=dependencies(trace,adj,a,b,t),**audit_continuation(trace,adj,a,b,t))
    result=report.summarize([row])
    assert result['seeds']==[7] and result['correct']==1 and result['repairs']==1
    assert result['by_history']['0']['bad']==1
    assert result['by_history']['1']['bad']==0 and result['by_history']['1']['n']==2
    assert result['first_error_lags'][0]['n']==1
    assert sum(x['events'] for x in result['hazard'])==1

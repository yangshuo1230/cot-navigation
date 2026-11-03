"""Natural readouts must stay unedited, label-correct, and source-matched."""
from collections import defaultdict
from cot_navigation.analysis.natural_history import annotate,match_readouts
from cot_navigation.protocols.tokenizer import Tokenizer


def sample_cases():
    tok=Tokenizer(16)
    a,b,c=[tok.node_id(i) for i in range(3)];bar=tok.id('|');semi=tok.id(';')
    record=[a,bar,a,bar,b,c,bar,tok.id('<PUSH>'),b,semi]
    trace=record*3
    first=dict(token=6,record=1,kind='candidate_nonedge')
    cases=[dict(query=1,source='donorA',trace=trace,marker_at=None,first=first),
           dict(query=2,source='donorA',trace=trace,marker_at=None,first=None),
           dict(query=3,source='donorB',trace=trace,marker_at=None,first=None)]
    return tok,cases


def test_natural_matching_does_not_repair_or_cross_sources():
    tok,raw=sample_cases();cases=annotate(raw,tok);rows,info=match_readouts(cases,tok)
    assert rows and info['exact_token_fraction']==1
    pairs=defaultdict(list)
    for r in rows:
        case=cases[r['sid']]
        assert case['trace']==raw[r['sid']]['trace']
        assert case['source']=='donorA'
        assert r['y']==int(case['tau'] is not None and r['pos']>=case['tau'])
        pairs[(r['event'],r['cell'])].append(r)
    for pair in pairs.values():
        assert len(pair)==2 and {r['y'] for r in pair}=={0,1}
        assert len({r['field'] for r in pair})==1
        assert len({r['offset'] for r in pair})==1


def test_marker_cut_never_creates_primary_post_marker_readouts():
    tok,raw=sample_cases();raw[0]['marker_at']=7
    cases=annotate(raw,tok);rows,_=match_readouts(cases,tok)
    assert len(cases[0]['trace'])==7
    assert all(r['pos']<7 for r in rows if r['sid']==0)


def test_current_checkpoint_test_can_filter_generating_source():
    tok,raw=sample_cases();cases=annotate(raw,tok)
    rows,_=match_readouts(cases,tok,positive_source='donorB')
    assert rows==[]


def test_error_offsets_cross_records_with_true_record_offset_matching():
    tok,raw=sample_cases();cases=annotate(raw,tok)
    rows,_=match_readouts(cases,tok,error_offsets=range(6))
    positives=[r for r in rows if r['y']]
    assert [r['offset'] for r in positives]==list(range(6))
    assert [r['pos'] for r in positives]==list(range(5,11))
    # Offset 5 lies in the next record; negative must match that record's offset 0.
    last=[r for r in rows if r['offset']==5]
    assert len(last)==2
    for r in last:
        c=cases[r['sid']]
        begin=next(a for a,b in c['spans'] if a<=r['pos']<b)
        assert r['pos']-begin==0 and r['field']=='frontier'
        assert r['y']==int(c['tau'] is not None and r['pos']>=c['tau'])


def test_error_offset_window_stops_before_marker():
    tok,raw=sample_cases();raw[0]['marker_at']=7
    rows,_=match_readouts(annotate(raw,tok),tok,error_offsets=range(6))
    assert [r['offset'] for r in rows if r['y']]==[0,1]


def test_action_boundary_is_after_error_and_matched_by_real_record_offset():
    tok,raw=sample_cases();cases=annotate(raw,tok)
    rows,_=match_readouts(cases,tok,error_offsets=[0],anchor='action_boundary')
    assert len(rows)==2
    assert {r['field'] for r in rows}=={'action'}
    for r in rows:
        assert cases[r['sid']]['trace'][r['pos']]==tok.id('|')
        assert r['y']==int(cases[r['sid']]['tau'] is not None and r['pos']>=cases[r['sid']]['tau'])
        assert r['pos']%10==6


def test_field_subsets_preserve_scalar_audit_metadata():
    import runpy
    import numpy as np
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    module=runpy.run_path(str(root/'studies/correction_reproduction/natural_history_probes/run.py'))
    data=dict(y=np.array([0,1]),pre_error_max_difference=0.)
    result=module['subset'](data,np.array([True,False]))
    assert result['y'].tolist()==[0] and result['pre_error_max_difference']==0.

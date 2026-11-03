"""Analysis stopping must preserve actual sampled prefixes and normal decode."""
from test_rollout_speedup import Scheduled
from cot_navigation.evaluation.rollout import generate
from cot_navigation.protocols.tokenizer import Tokenizer


def test_record_stop_is_not_a_budget_stop_and_does_not_edit_tokens():
    tok=Tokenizer(8,enable_stop=True);prompts=[[tok.node_id(2)]*2]
    native=generate(Scheduled(tok),prompts,'dfs_state',tok,cot_margin=8)
    stopped=generate(Scheduled(tok),prompts,'dfs_state',tok,cot_margin=8,
                     record_stop=lambda i,trace:trace.count(tok.id(';'))==1)
    result=stopped[0]
    assert result['analysis_stopped'] and not result['generation_limit_reached']
    assert not result['forced_stop'] and result['pred'] is None
    assert result['trace_tokens'][-1]==tok.id(';')
    assert result['trace_tokens']==native[0]['trace_tokens'][:len(result['trace_tokens'])]


def test_never_stopping_callback_preserves_all_default_fields():
    tok=Tokenizer(8,enable_stop=True)
    prompts=[[tok.node_id(0)]*4,[tok.node_id(1)]*7,[tok.node_id(2)]*2]
    native=generate(Scheduled(tok),prompts,'dfs_state',tok,cot_margin=8)
    actual=generate(Scheduled(tok),prompts,'dfs_state',tok,cot_margin=8,
                    record_stop=lambda i,trace:False)
    for a,b in zip(actual,native):
        assert not a.pop('analysis_stopped')
        assert a==b

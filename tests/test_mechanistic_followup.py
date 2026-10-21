"""Tests of semantic labels and counterfactual coverage, independent of GPUs."""
from types import SimpleNamespace
import numpy as np
import pytest
from cot_navigation.analysis.common import Example,records
from cot_navigation.analysis.error_detection import events,semantic_slots
from cot_navigation.analysis.route_blocking import blocking_case,search_from_state
from cot_navigation.analysis.set_metrics import set_quality,auc_ap
from cot_navigation.protocols.render import render_task,parametric_prompt
from cot_navigation.protocols.tokenizer import Tokenizer


def example(adj,source,target,protocol='dfs_state'):
    tok=Tokenizer(32);nodes=[tok.node_id(i) for i in range(len(adj))]
    row=render_task(tok,protocol,1,parametric_prompt(tok,nodes,source,target),adj,source,target,nodes)
    return tok,Example(3,'test',1,adj,nodes,source,target,row)


@pytest.mark.parametrize('protocol',['dfs_state','bfs_state','path','bfs_nodes'])
def test_reference_has_no_semantic_error(protocol):
    tok,ex=example([[1,2],[3],[3],[]],0,3,protocol)
    assert events(ex,protocol,tok,ex.row.tokens) is None
    e=events(ex,protocol,tok,ex.row.tokens,synthetic=True)
    assert e and e['wrong_token'] in ex.nodes and e['correct_token'] in ex.nodes
    assert e['wrong'][:-1]==e['correct'][:-1]
    assert events(ex,protocol,tok,e['wrong']) is not None


def test_path_accepts_alternative_witness_and_reports_first_bad_edge():
    tok,ex=example([[1,2],[3],[3],[]],0,3,'path')
    prompt=ex.row.tokens[:ex.row.prompt_len]
    alternate=prompt+[ex.nodes[i] for i in [0,2,3]]
    assert events(ex,'path',tok,alternate) is None
    bad=prompt+[ex.nodes[i] for i in [0,1,2,3]]
    e=events(ex,'path',tok,bad)
    assert e['position']==len(prompt)+2 and e['next_token']==ex.nodes[3]


def test_state_accepts_any_push_and_set_order():
    tok,ex=example([[1,2],[3],[3],[]],0,3)
    prompt=ex.row.tokens[:ex.row.prompt_len];n=ex.nodes;b=tok.id('|');s=tok.id(';')
    # Reverse candidate order and take the SECOND displayed valid candidate.
    body=[n[0],b,n[0],b,n[2],n[1],b,tok.id('<PUSH>'),n[1],s]
    assert events(ex,'dfs_state',tok,prompt+body) is None
    bad=body.copy();bad[-2]=n[3]
    e=events(ex,'dfs_state',tok,prompt+bad)
    assert e['field']=='argument'


def test_next_token_flags_a_repair_that_creates_a_new_error():
    tok,ex=example([[1,2],[3],[3],[]],0,3)
    prompt=ex.row.tokens[:ex.row.prompt_len];n=ex.nodes;b=tok.id('|');s=tok.id(';')
    # Candidate 2 omitted. Repairing the premature bar to 2 makes the next
    # original PUSH illegal until a new bar is inserted. Must NOT call this
    # a clean +1 control, even though the designated previous token is fixed.
    body=[n[0],b,n[0],b,n[1],b,tok.id('<PUSH>'),n[1],s]
    e=events(ex,'dfs_state',tok,prompt+body)
    assert e['field']=='candidates' and e['next_token']==tok.id('<PUSH>')
    assert not e['repair_next_valid']


def test_premature_terminal_action_is_not_hidden_from_error_probe():
    tok,ex=example([[1,2],[3],[3],[]],0,3)
    prompt=ex.row.tokens[:ex.row.prompt_len];n=ex.nodes;b=tok.id('|');s=tok.id(';')
    body=[n[0],b,n[0],b,n[1],n[2],b,tok.id('<FOUND>'),s]
    e=events(ex,'dfs_state',tok,prompt+body+[tok.yes])
    assert e['field']=='action' and e['wrong_token']==tok.id('<FOUND>')


@pytest.mark.parametrize('fifo',[False,True])
def test_block_all_alternative_frontier_routes(fifo):
    adj=[[1,2],[3],[4],[5],[5],[]]
    # Blocking just branch 3 leaves branch 4 available from the other frontier.
    assert search_from_state(adj,[1,2],{0,1,2,3},5,fifo)
    assert not search_from_state(adj,[1,2],{0,1,2,3,4},5,fifo)
    # Already queued target cannot be blocked by marking it visited.
    assert search_from_state(adj,[1,5],set(range(6)),5,fifo)


def test_no_goal_variant_is_certified_and_direct_edge_rejected():
    tok,ex=example([[1,2],[3],[3],[]],0,3)
    am=SimpleNamespace(tok=tok,protocol='dfs_state')
    c=blocking_case(am,ex,ex.row.tokens,exclude_goal=True)
    assert c and ex.nodes[ex.target] not in c['detail']['inserted']
    assert c['detail']['all_frontier_reachability_blocked']
    tok,ex=example([[1,2],[2],[]],0,2)
    am=SimpleNamespace(tok=tok,protocol='dfs_state')
    assert blocking_case(am,ex,ex.row.tokens,exclude_goal=True) is None
    assert blocking_case(am,ex,ex.row.tokens) is not None


def test_set_ranking_penalizes_high_wrong_scores_and_ignores_absent_nodes():
    y=np.array([[1,0,0],[1,0,0]],float);mask=np.array([[1,1,0],[1,1,0]],float)
    good=set_quality(np.array([[2,-2,100],[3,-3,100]]),y,mask)
    bad=set_quality(np.array([[2,3,100],[3,4,100]]),y,mask)
    assert good['pairwise_auc']==1 and good['strict_separation']==1
    assert bad['pairwise_auc']==0 and bad['strict_separation']==0
    assert bad['mean_true_probability']>.8 and bad['mean_false_probability']>.9
    tie=set_quality(np.zeros((2,3)),y,mask)
    assert tie['pairwise_auc']==.5 and tie['strict_separation']==0
    assert auc_ap([0,1],[1,1])['auroc']==.5


def test_empty_and_full_sets_have_no_artificial_perfect_ranking():
    out=set_quality(np.zeros((2,3)),np.array([[0,0,0],[1,1,1]]),np.ones((2,3)))
    assert out['nontrivial_rows']==0
    assert out['empty_target_rows']==out['full_target_rows']==1
    assert 'pairwise_auc' not in out

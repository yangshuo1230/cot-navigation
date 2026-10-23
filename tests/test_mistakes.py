"""Membership errors, unchanged search state, repair boundaries and loss masks."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cot_navigation.data.dataset import collate_examples
from cot_navigation.data.mistakes import records,audit_reference,choose_plan,apply_plan
from cot_navigation.protocols.render import render_task,context_prompt
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.training.loss import combine_losses


def sample(protocol='dfs_state', max_len=256):
    tok = Tokenizer(16,enable_stop=True,enable_mistake=True)
    adj = [[1,2],[3],[3,4],[5],[5],[]]
    nodes = [tok.node_id(i) for i in range(len(adj))]
    prompt = context_prompt(tok,nodes,[(u,v) for u,vs in enumerate(adj) for v in vs],0,5)
    row = render_task(tok,protocol,1,prompt,adj,0,5,nodes)
    row.meta['index'] = 0
    cfg = SimpleNamespace(model=SimpleNamespace(max_seq_len=max_len),
        protocol=SimpleNamespace(cot_margin=8,mistake_mask_errors=True))
    return tok,row,cfg


def test_optional_token_append_preserves_all_legacy_ids():
    old=Tokenizer(256,enable_stop=True)
    new=Tokenizer(256,enable_stop=True,enable_mistake=True)
    assert old.itos==new.itos[:-1]
    assert new.id('<MISTAKE>')==len(old)
    assert old.hash()!=new.hash()


@pytest.mark.parametrize('protocol',['dfs_state','bfs_state'])
@pytest.mark.parametrize('field',['cands','visited'])
def test_plans_are_genuine_single_field_errors_and_repair_without_transition(protocol,field):
    tok,row,cfg=sample(protocol)
    recs=records(row,tok)
    audit_reference(row,recs,tok,fifo=protocol=='bfs_state')
    seen=set()
    for seed in range(100):
        plan=choose_plan(row,recs,tok,field,np.random.default_rng(seed),cfg)
        got=apply_plan(row,plan,tok,cfg)
        seen.add(got.meta['mistake_edit'])
        start,end,lo,hi,action,op,offset,node=map(int,plan)
        extra=got.meta['repair_start']-start
        assert got.tokens[:start]==row.tokens[:start]
        assert got.tokens[got.meta['repair_start']:]==row.tokens[start:]
        assert got.tokens[got.meta['repair_start']:got.meta['repair_end']]==row.tokens[start:end]
        assert got.tokens.count(tok.id('<MISTAKE>'))==1
        marker=got.meta['mistake_action_index']
        assert got.tokens[marker:marker+2]==[tok.id('<MISTAKE>'),tok.id(';')]
        changed=got.tokens[lo:got.meta['mistake_mask_span'][1]-1]
        assert set(changed)!=set(row.tokens[lo:hi])
        assert len(changed)==len(set(changed))
        assert got.meta['mistake_field']==field
        assert got.tokens[marker+2:marker+2+end-start]==row.tokens[start:end]
        assert got.tokens[got.answer_index]==tok.yes
    assert seen=={'add','drop','replace'}


@pytest.mark.parametrize('mask',[True,False])
def test_target_masks_have_correct_shift_and_supervise_mistake_and_repair(mask):
    tok,row,cfg=sample()
    plan=choose_plan(row,records(row,tok),tok,'visited',np.random.default_rng(42),cfg)
    got=apply_plan(row,plan,tok,cfg,mask_errors=mask)
    lo,hi=got.meta['mistake_mask_span']
    arrays=collate_examples([got],tok.pad,1)
    tokens=torch.from_numpy(arrays['tokens'])
    logits=torch.randn(1,tokens.shape[1],len(tok),requires_grad=True)
    parts=combine_losses(logits,tokens,torch.from_numpy(arrays['answer_mask']),
                         torch.from_numpy(arrays['trace_mask']),torch.from_numpy(arrays['walk_mask']),1,0)
    parts['total_loss'].backward()
    grad=logits.grad[0].abs().sum(-1)
    assert bool((grad[lo-1:hi-1]==0).all()) if mask else bool((grad[lo-1:hi-1]>0).all())
    marker=got.meta['mistake_action_index']
    assert grad[marker-1]>0
    assert grad[got.meta['repair_start']-1]>0
    assert grad[got.answer_index-1]>0


def test_repair_is_complete_even_when_later_canonical_tail_is_truncated():
    tok,row,cfg=sample(max_len=100)
    recs=records(row,tok)
    for seed in range(30):
        plan=choose_plan(row,recs,tok,'cands',np.random.default_rng(seed),cfg)
        got=apply_plan(row,plan,tok,cfg)
        assert len(got.tokens)<=92
        assert got.meta['repair_end']<=got.answer_index
        assert got.tokens[got.meta['repair_end']-1]==tok.id(';')
        assert got.tokens[got.answer_index:]==[tok.yes,tok.id('<EOS>')]


@pytest.mark.parametrize('field',['cands','visited'])
def test_auditor_rejects_wrong_canonical_membership(field):
    tok,row,cfg=sample()
    recs=records(row,tok)
    recs[0][field]=recs[0][field]+[tok.node_id(5)]
    with pytest.raises(ValueError):audit_reference(row,recs,tok,fifo=False)

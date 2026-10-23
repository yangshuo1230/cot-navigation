"""Unit fixtures are not training data: verify immutable observed payloads."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from cot_navigation.data.mistakes import records,audit_reference
from cot_navigation.data.natural_mistakes import describe_error,build_episode,write_pool,NaturalCorrectionPool
from cot_navigation.data.mine_natural_candidates import complete_fields
from cot_navigation.models.attention import KVCache
from cot_navigation.protocols.render import render_task,context_prompt
from cot_navigation.protocols.tokenizer import Tokenizer


def fixture(protocol='dfs_state',limit=512):
    tok=Tokenizer(16,enable_stop=True,enable_mistake=True)
    adj=[[1,2],[3],[4],[5],[5],[]];nodes=[tok.node_id(i) for i in range(6)]
    prompt=context_prompt(tok,nodes,[(u,v) for u,vs in enumerate(adj) for v in vs],0,5)
    row=render_task(tok,protocol,1,prompt,adj,0,5,nodes)
    row.meta.update(index=0,split=0)
    cfg=SimpleNamespace(model=SimpleNamespace(max_seq_len=limit),
        data=SimpleNamespace(node_vocab=16,graph_family='layered'),train=SimpleNamespace(world_roots=[]),
        protocol=SimpleNamespace(name=protocol,cot_margin=8,trajectory_variants=4,enable_mistake=True))
    return tok,row,cfg


def event(rec,predicted):
    lo,hi=rec['spans']['cands']
    return dict(start=rec['start'],end=rec['end'],lo=lo,hi=hi,action=rec['action_start'],
        predicted=predicted,features=dict(edits_lower_bound=2))


def test_error_measurement_does_not_call_reordering_wrong():
    assert describe_error([1,2],[2,1],[0],{0,1,2}) is None
    measured=describe_error([1,2,3],[4,4,5],[0,4],{0,1,2,3,4,5})
    assert measured['missing']==3 and measured['extra']==2 and measured['duplicates']==1
    assert measured['extra_visited']==1 and measured['edits_lower_bound']==4


@pytest.mark.parametrize('protocol',['dfs_state','bfs_state'])
def test_multiple_observed_records_are_preserved_and_repairs_restore_machine(protocol):
    tok,row,cfg=fixture(protocol)
    recs=records(row,tok);audit_reference(row,recs,tok,fifo=protocol=='bfs_state')
    events=[event(recs[0],[tok.node_id(4),tok.node_id(4)]),event(recs[2],[])]
    original=row.tokens.copy()
    got=build_episode(row,events,tok,cfg)
    assert got.meta['natural_complex'] and len(got.meta['natural_events'])==2
    assert row.tokens==original
    recovered=[];cursor=0
    for e in got.meta['natural_events']:
        assert got.tokens[e['candidate_start']:e['candidate_end']]==e['predicted']
        assert got.tokens[e['repair_start']:e['repair_end']]==row.tokens[e['original_start']:e['original_end']]
        recovered.extend(got.tokens[cursor:e['bad_start']]);cursor=e['repair_start']
    recovered.extend(got.tokens[cursor:])
    assert recovered==row.tokens


def test_repair_boundary_and_label_survive_context_truncation():
    tok,row,cfg=fixture(limit=100)
    recs=records(row,tok)
    got=build_episode(row,[event(r,[tok.node_id(4),tok.node_id(4)]) for r in recs],tok,cfg)
    assert got is not None and len(got.tokens)<=92
    assert all(e['repair_end']<=got.answer_index for e in got.meta['natural_events'])
    assert got.tokens[got.answer_index:]==[tok.yes,tok.id('<EOS>')]


def test_pool_sampling_never_uses_held_out_episodes(tmp_path):
    tok,row,cfg=fixture();cfg.protocol.mistake_cache_root=str(tmp_path/'pool')
    base=build_episode(row,[event(records(row,tok)[0],[tok.node_id(4)])],tok,cfg)
    rows=[replace(base,label=label,meta={**base.meta,'split':split,'index':split*2+label})
          for split in range(3) for label in (0,1)]
    source=SimpleNamespace(fingerprint='a'*64,root=tmp_path/'source')
    write_pool(source,cfg,tok,rows,[0,0,1,1,2,2],{}, {})
    pool=NaturalCorrectionPool(source,cfg,tok);rng=np.random.default_rng(42)
    for label in (0,1):
        for _ in range(50):
            got=pool.sample(label,rng,.5)
            assert got.label==label and got.meta['split']==0
    assert len(pool.cases('val',64,0))==2 and len(pool.cases('test',64,0))==2
    with pytest.raises(ValueError):pool.cases('train',64,0)


class Scripted(nn.Module):
    def __init__(self,tok,script):
        super().__init__();self.anchor=nn.Parameter(torch.zeros(1));self.tok=tok
        self.script=script;self.step=0;self.max_seq_len=16
    def new_cache(self,batch,device,dtype):return KVCache(1,batch,1,16,4,device,dtype)
    def forward(self,tokens,**kwargs):
        logits=torch.full((*tokens.shape,len(self.tok)),-30.)
        logits[:,-1,self.script[min(self.step,len(self.script)-1)]]=30
        self.step+=1;return logits


def test_candidate_completion_requires_an_emitted_delimiter():
    tok=Tokenizer(16,enable_stop=True)
    node=tok.node_id(1)
    got=complete_fields(Scripted(tok,[node,tok.id('|')]),[[tok.node_id(0)]],tok)
    assert got==[([node],'closed')]
    capped=complete_fields(Scripted(tok,[node]),[[tok.node_id(0)]],tok,cap=1)
    assert capped==[([node],'unclosed_at_cap')]

import numpy as np
import torch
from cot_navigation.analysis.within_token import candidate_slots,cached_tail,protected_pair,lens_logits,binary_accuracy
from cot_navigation.analysis.residual_control import residual_edits
from cot_navigation.models.transformer import TransformerLM
from cot_navigation.protocols.tokenizer import Tokenizer


def test_slot_is_before_current_error_and_all_remaining_nodes_are_valid():
    tok=Tokenizer(8);a,b,c=[tok.node_id(i) for i in range(3)];bar=tok.id('|')
    trace=[a,bar,a,bar,b,b]
    rows=list(candidate_slots(dict(trace=trace,tau=5),{a:[b,c],b:[],c:[]},tok))
    assert rows[0]['pos']==4 and rows[0]['allowed']==[b,c]
    assert rows[1]['pos']==5 and rows[1]['allowed']==[c]
    assert rows[1]['natural_onset'] and trace[:rows[1]['pos']]==trace[:5]
    # The duplicate b at index5 is not part of either prediction's input.
    assert len(rows)==2


def test_tail_matches_full_patch_and_can_reuse_current_position_cache():
    torch.manual_seed(7);torch.set_num_threads(1)
    m=TransformerLM(32,3,1,16,32,32).eval();ids=torch.tensor([[1,2,3,4],[4,3,2,1]])
    cap={};cache=m.new_cache(2,'cpu',torch.float32);cache.valid_length=4
    with torch.no_grad():
        original=m(ids,cache=cache,cache_mode='prefill',capture=cap)[:,-1]
        pos=torch.tensor([[3],[3]])
        for layer in [0,1,2,0]:
            h=cap['resid'][layer][:,-1];edit=h+torch.randn_like(h)*.05
            unit=np.zeros(16,np.float32);unit[0]=1
            # Full-replay reference uses a replacement hook, not cached-tail code.
            def hook(_m,_args,out):
                x,w=out;x=x.clone();x[:,-1]=edit;return x,w
            handle=m.blocks[layer].register_forward_hook(hook)
            try:full=m(ids)[:,-1]
            finally:handle.remove()
            tail=cached_tail(m,cache,edit,pos,layer)
            torch.testing.assert_close(full,tail,rtol=1e-5,atol=1e-6)
            sham=cached_tail(m,cache,h,pos,layer)
            torch.testing.assert_close(original,sham,rtol=1e-5,atol=1e-6)


def test_preserved_final_layer_prediction_is_structural_zero_not_a_repair():
    torch.manual_seed(8);m=TransformerLM(30,3,1,16,32,32).eval()
    h=torch.randn(10,16);d=torch.randn_like(h)*10;v=torch.randn_like(h)*10
    with torch.no_grad():
        a,b,factor=protected_pair(m,{},h,2,'raw',d,v,5,20)
        base=lens_logits(m,{},h,2,'raw')
        for edited in [a,b]:
            z=lens_logits(m,{},edited,2,'raw')
            assert torch.equal(z.argmax(-1),base.argmax(-1))
            assert torch.equal(z[:,5:25].argmax(-1),base[:,5:25].argmax(-1))
    assert ((factor>=0)&(factor<=1)).all()


def test_accuracy_reports_majority_and_balanced_threshold_point():
    y=np.array([1]*9+[0]);z=np.ones(10)
    r=binary_accuracy(y,z,np.arange(10))
    assert r['accuracy']==.9 and r['balanced_accuracy']==.5 and r['threshold']==.5

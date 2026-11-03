"""Check intervention geometry, causal timing and cached propagation."""
import numpy as np
import pytest
import torch
from cot_navigation.analysis.residual_control import steer,orthogonal_axis,residual_edits
from cot_navigation.models.transformer import TransformerLM


def test_null_is_orthogonal_and_norm_matched():
    h=torch.tensor([[1.,2.,3.],[-1.,4.,2.]])
    u=torch.tensor([1.,0.,0.]);v=torch.tensor(orthogonal_axis(u.numpy(),123))
    edited=steer(h,u,0.);control=steer(h,u,0.,random_direction=v)
    torch.testing.assert_close((edited-h).norm(dim=-1),(control-h).norm(dim=-1))
    torch.testing.assert_close(control@u,h@u)
    assert torch.all((edited-h).norm(dim=-1)<=h.norm(dim=-1)*.200001)


def test_prefill_patch_routes_to_later_cache_and_does_not_edit_tokens():
    torch.manual_seed(11);torch.set_num_threads(1)
    m=TransformerLM(24,3,1,16,32,32).eval();ids=torch.tensor([[1,2,3,4]])
    before=ids.clone();u=np.zeros(16,np.float32);u[0]=1
    def jobs(layer,dose=1):return {layer:[(0,2,u,5.,dose,None)]}
    with torch.no_grad():
        base=m(ids)
        with residual_edits(m,jobs(0,0)):identity=m(ids)
        torch.testing.assert_close(base,identity,rtol=0,atol=0)
        with residual_edits(m,jobs(2)):last=m(ids)
        torch.testing.assert_close(base[:,3],last[:,3],rtol=0,atol=0)
        with residual_edits(m,jobs(0)):early=m(ids)
        assert (early[:,3]-base[:,3]).abs().max()>1e-6
        torch.testing.assert_close(early[:,:2],base[:,:2],rtol=0,atol=0)
        extended=torch.tensor([[1,2,3,4,5]])
        with residual_edits(m,jobs(0)):full=m(extended)[:,-1]
        cache=m.new_cache(1,'cpu',torch.float32)
        with residual_edits(m,jobs(0)):m(ids,cache=cache,cache_mode='prefill')
        cached=m(extended[:,-1:],positions=torch.tensor([[4]]),cache=cache,cache_mode='decode')[:,-1]
        torch.testing.assert_close(full,cached,rtol=1e-5,atol=1e-6)
    assert torch.equal(before,ids)
    assert all(not b._forward_hooks for b in m.blocks)
    with pytest.raises(RuntimeError):
        with residual_edits(m,jobs(0)):raise RuntimeError('intentional')
    assert all(not b._forward_hooks for b in m.blocks)

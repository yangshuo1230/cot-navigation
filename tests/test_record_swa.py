"""Boundary, cache rollback, leakage, gradient and lambda regression tests."""
import pytest
import torch
from cot_navigation.models.record_attention import record_mask
from cot_navigation.models.transformer import TransformerLM
from cot_navigation.evaluation.rollout import cached_logits,full_logits
from cot_navigation.training.loss import combine_losses,token_nll,masked_mean

PROMPT=[2,14,15,3,14,19,4]
RECORDS=[[14,4,14,4,15,4,8,15,5],
         [14,15,4,14,15,4,16,4,8,16,5],
         [14,15,16,4,14,15,16,4,4,9,5],
         [14,15,4,14,15,16,4,4,9,5]]

def sequence():return PROMPT+sum(RECORDS,[])+[11,5,7,1]

def model(window,backend='sdpa',attn='math'):
    return TransformerLM(80,2,1,32,64,128,attn_impl=attn,swa_records=window,swa_backend=backend).eval()

@pytest.mark.parametrize('window',[1,2])
def test_record_boundaries_prompt_current_record_and_no_future(window):
    x=torch.tensor([sequence()]);pos=torch.arange(x.shape[1])[None]
    mask=record_mask(x,pos,window)[0,0]
    starts=[len(PROMPT)]
    for r in RECORDS:starts.append(starts[-1]+len(r))
    q=starts[3]+3  # inside fourth record
    assert mask[q,:len(PROMPT)].all()
    assert mask[q,starts[3]:q+1].all()  # current is extra, not quota
    assert mask[q,starts[3-window]:starts[3]].all()
    assert not mask[q,len(PROMPT):starts[3-window]].any()
    assert not mask[q,q+1:].any()
    q=starts[3]-1  # third semicolon now predicts fourth record
    assert mask[q,starts[3-window]:q+1].all()
    assert not mask[q,len(PROMPT):starts[3-window]].any()
    assert not torch.triu(mask,diagonal=1).any()

@pytest.mark.parametrize('window',[1,2])
def test_cached_full_and_right_padding_match(window):
    torch.manual_seed(1);m=model(window)
    seq=sequence();tokens=torch.tensor([seq,seq]);lengths=torch.tensor([len(seq)-11,len(seq)])
    tokens[0,lengths[0]:]=0
    a=full_logits(m,tokens);b=cached_logits(m,tokens,lengths)
    for i in range(2):assert torch.allclose(a[i,:lengths[i]],b[i,:lengths[i]],atol=1e-5,rtol=1e-5)

def test_walk_rows_still_have_full_causal_attention():
    torch.manual_seed(2);a=model(0);b=model(1);b.load_state_dict(a.state_dict())
    x=torch.tensor([[12,14,15,16,17,1]])
    assert torch.allclose(a(x),b(x),atol=1e-6,rtol=1e-6)

def test_future_record_delimiters_do_not_change_prefix_mask():
    x=torch.tensor([sequence()]);y=x.clone();cut=len(PROMPT)+len(RECORDS[0])+3
    y[:,cut+1:]=18
    pos=torch.arange(x.shape[1])[None]
    a=record_mask(x,pos,1);b=record_mask(y,pos,1)
    assert torch.equal(a[:,:,:cut+1,:],b[:,:,:cut+1,:])

def test_cache_rollback_matches_truncated_prefix():
    torch.manual_seed(3);m=model(1)
    x=torch.tensor([sequence()]);cache=m.new_cache(1,'cpu',torch.float32)
    pos=torch.arange(x.shape[1])[None]
    m(x,positions=pos,cache=cache,cache_mode='prefill')
    # Simulate native forced STOP rewinding to an earlier completed record.
    cut=len(PROMPT)+len(RECORDS[0])+len(RECORDS[1])
    stop=torch.tensor([[79]])
    actual=m(stop,positions=torch.tensor([[cut]]),cache=cache,cache_mode='decode')
    expected=m(torch.cat([x[:,:cut],stop],dim=1))[:,-1:]
    assert torch.allclose(actual,expected,atol=1e-5,rtol=1e-5)

def test_sdpa_math_outputs_and_gradients_match():
    torch.manual_seed(4);a=model(2,attn='math');b=model(2,attn='sdpa');b.load_state_dict(a.state_dict())
    x=torch.tensor([sequence()])
    aa=a(x);bb=b(x)
    assert torch.allclose(aa,bb,atol=1e-5,rtol=1e-5)
    aa.square().mean().backward();bb.square().mean().backward()
    for pa,pb in zip(a.parameters(),b.parameters()):assert torch.allclose(pa.grad,pb.grad,atol=1e-5,rtol=1e-4)

@pytest.mark.parametrize('lam',[1.,4.,16.])
def test_loss_formula_and_microbatch_accumulation(lam):
    torch.manual_seed(5)
    x=torch.randint(1,20,(3,9));logits=torch.randn(3,9,20,requires_grad=True)
    answer=torch.zeros_like(x,dtype=torch.float);answer[:,7]=1
    trace=torch.zeros_like(answer);trace[:,2:7]=1
    zero=torch.zeros_like(answer)
    result=combine_losses(logits,x,answer,trace,zero,3,0,trace_lambda=lam)
    nll=token_nll(logits,x);a,_=masked_mean(nll,answer);t,_=masked_mean(nll,trace)
    expected=(2*(a+lam*t)/(1+lam)).mean()
    assert torch.allclose(result['total_loss'],expected)
    pieces=[combine_losses(logits[i:i+1],x[i:i+1],answer[i:i+1],trace[i:i+1],zero[i:i+1],3,0,trace_lambda=lam)['total_loss'] for i in range(3)]
    assert torch.allclose(sum(pieces),expected)
    actual_grad=torch.autograd.grad(result['total_loss'],logits,retain_graph=True)[0]
    expected_grad=torch.autograd.grad(expected,logits)[0]
    assert torch.allclose(actual_grad,expected_grad)
    # Direct, Filler and WALK objectives are unchanged by trace lambda.
    direct=combine_losses(logits,x,answer,zero,zero,3,0,trace_lambda=lam)
    assert torch.allclose(direct['total_loss'],a.mean())

def test_default_and_explicit_lambda_one_are_identical():
    x=torch.tensor([[14,15,16,6]])
    logits=torch.randn(1,4,80)
    a=torch.tensor([[0.,0.,0.,1.]])
    t=torch.tensor([[0.,1.,1.,0.]])
    z=torch.zeros_like(a)
    assert torch.equal(combine_losses(logits,x,a,t,z,1,0)['total_loss'],combine_losses(logits,x,a,t,z,1,0,trace_lambda=1)['total_loss'])

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_cuda_reference_path_matches_forward_backward_and_decode():
    torch.manual_seed(6)
    ref=model(1,attn='sdpa').cuda();flex=model(1,backend='flex',attn='sdpa').cuda()
    flex.load_state_dict(ref.state_dict())
    x=torch.tensor([sequence()],device='cuda')
    a=ref(x);b=flex(x)
    assert torch.allclose(a,b,atol=2e-4,rtol=2e-3)
    a.square().mean().backward();b.square().mean().backward()
    for pa,pb in zip(ref.parameters(),flex.parameters()):assert torch.allclose(pa.grad,pb.grad,atol=3e-4,rtol=4e-3)
    full=full_logits(flex,x);cached=cached_logits(flex,x,torch.tensor([x.shape[1]],device='cuda'))
    assert torch.allclose(full,cached,atol=3e-4,rtol=4e-3)

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_bf16_non_aligned_prompt_prefill_and_cached_decode():
    torch.manual_seed(8)
    m=TransformerLM(80,2,1,128,512,1024,attn_impl='sdpa',swa_records=1,swa_backend='flex').cuda().eval()
    prompt=[2,14,15]*150+[3,14,19,4]  # real N100 prompt length 454
    seq=prompt+(sum(RECORDS,[])*3)[:58]
    x=torch.tensor([seq],device='cuda')
    cache=m.new_cache(1,'cuda',torch.bfloat16)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        full=m(x)
        m(x[:,:454],positions=torch.arange(454,device='cuda')[None],cache=cache,cache_mode='prefill')
        for i in range(454,len(seq)):
            logits=m(x[:,i:i+1],positions=torch.tensor([[i]],device='cuda'),cache=cache,cache_mode='decode')
            assert torch.allclose(logits[:,0],full[:,i],atol=.006,rtol=.03)

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('window',[1,2])
def test_flex_bf16_actual_head_size_and_full_context(window):
    torch.manual_seed(7)
    ref=TransformerLM(80,2,1,128,512,4096,attn_impl='sdpa',swa_records=window,swa_backend='sdpa').cuda()
    flex=TransformerLM(80,2,1,128,512,4096,attn_impl='sdpa',swa_records=window,swa_backend='flex').cuda()
    flex.load_state_dict(ref.state_dict())
    for batch,length in [(2,128),(1,256),(1,4096)]:
        body=(sum(RECORDS,[])*200)[:length-len(PROMPT)]
        x=torch.tensor([PROMPT+body]*batch,device='cuda')
        ref.zero_grad(set_to_none=True);flex.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):a=ref(x);b=flex(x)
        assert torch.allclose(a,b,atol=.006,rtol=.03)
        a.float().square().mean().backward();b.float().square().mean().backward()
        for pa,pb in zip(ref.parameters(),flex.parameters()):
            assert torch.allclose(pa.grad,pb.grad,atol=.0006,rtol=.06)

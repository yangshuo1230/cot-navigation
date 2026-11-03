"""Check the reused length policy against direct per-example CE and gradients."""
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'studies/correction_reproduction'))
from trace_weight import combine_length_losses,weights_from_masks
from cot_navigation.training.loss import token_nll,masked_mean

def example():
    lengths=[0,64,256,2304]
    tokens=torch.arange(4*2310).reshape(4,2310)%7
    answer=torch.zeros_like(tokens,dtype=torch.float32);trace=answer.clone()
    for i,length in enumerate(lengths):
        trace[i,2:2+length]=1
        answer[i,2+length]=1
        trace[i,3+length]=1  # existing EOS supervision must not increase CoT length
    return tokens,answer,trace,torch.zeros_like(answer)

def test_actual_length_normalization_and_gradient():
    tokens,answer,trace,walk=example();torch.manual_seed(7)
    logits=torch.randn(4,2310,7,requires_grad=True)
    lengths,weights=weights_from_masks(answer,trace)
    torch.testing.assert_close(lengths,torch.tensor([0.,64.,256.,2304.]))
    torch.testing.assert_close(weights,torch.tensor([1.,1.,2.,16.]))
    nll=token_nll(logits,tokens);a,_=masked_mean(nll,answer);t,_=masked_mean(nll,trace)
    expected=(2*(a+weights*t)/(1+weights)).mean()
    actual=combine_length_losses(logits,tokens,answer,trace,walk,4,0)['total_loss']
    torch.testing.assert_close(actual,expected)
    first=torch.autograd.grad(actual,logits,retain_graph=True)[0]
    second=torch.autograd.grad(expected,logits)[0]
    torch.testing.assert_close(first,second)

def test_logical_batch_gradient_is_unchanged_by_microbatching():
    tokens,answer,trace,walk=example();torch.manual_seed(8)
    logits=torch.randn(4,2310,7,requires_grad=True)
    whole=combine_length_losses(logits,tokens,answer,trace,walk,4,0)['total_loss']
    chunks=sum(combine_length_losses(logits[i:i+2],tokens[i:i+2],answer[i:i+2],trace[i:i+2],walk[i:i+2],4,0)['total_loss']for i in [0,2])
    torch.testing.assert_close(whole,chunks)
    torch.testing.assert_close(torch.autograd.grad(whole,logits,retain_graph=True)[0],torch.autograd.grad(chunks,logits)[0])

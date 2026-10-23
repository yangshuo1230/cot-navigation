"""Throughput changes must preserve the logical weighted training objective."""
from types import SimpleNamespace
import pytest
import torch

from cot_navigation.training.trainer import Trainer, _chunk, _reduce_logged
from cot_navigation.training.loss import combine_losses


def test_length_bucketing_keeps_rows_and_bounds_extra_padding():
    rows=[SimpleNamespace(tokens=list(range(n))) for n in [9,17,31,70,85,150,160,300,500]]
    chunks=_chunk(rows,0,1024,16,.15)
    assert sorted(map(id,rows))==sorted(id(r) for c in chunks for r in c)
    for chunk in chunks:
        lengths=[((len(r.tokens)+15)//16)*16 for r in chunk]
        allocated=max(lengths)*len(chunk)
        assert len(chunk)==1 or allocated<=1024
        assert (allocated-sum(lengths))/allocated<=.15


def test_changed_chunking_keeps_weighted_loss_and_gradients():
    torch.manual_seed(0)
    logits=torch.randn(7,48,19,requires_grad=True)
    tokens=torch.randint(19,(7,48))
    answer=torch.zeros(7,48);trace=torch.zeros_like(answer);walk=torch.zeros_like(answer)
    answer[:5,30]=1;trace[:5,3:30]=1;walk[5:,1:24]=1
    full=combine_losses(logits,tokens,answer,trace,walk,5,2,walk_weight=.7,trace_lambda=4.)['total_loss']
    full_grad=torch.autograd.grad(full,logits)[0]
    rows=[SimpleNamespace(tokens=list(range(32+i)),index=i) for i in range(7)]
    chunks=_chunk(rows[:5],0,80,1,.15)+_chunk(rows[5:],0,80,1,.15)
    split=0
    for chunk in chunks:
        ids=[r.index for r in chunk]
        split=split+combine_losses(logits[ids],tokens[ids],answer[ids],trace[ids],walk[ids],5,2,
                                  walk_weight=.7,trace_lambda=4.)['total_loss']
    torch.testing.assert_close(full,split)
    torch.testing.assert_close(full_grad,torch.autograd.grad(split,logits)[0])


def test_logging_reduction_uses_example_counts_and_actual_weighted_objective():
    parts=[]
    for n,answers,traces,walks,total in [(3,[True,True,False],[True,False,False],[False,False,True],2.),
                                       (2,[True,False],[True,False],[False,True],4.)]:
        parts.append(dict(answer=torch.arange(n,dtype=torch.float32)+1,
                          trace=torch.arange(n,dtype=torch.float32)+3,
                          walk=torch.arange(n,dtype=torch.float32)+5,
                          has_answer=torch.tensor(answers),has_trace=torch.tensor(traces),has_walk=torch.tensor(walks),
                          backward_total=torch.tensor(total)))
    result=_reduce_logged(parts,3,2)
    assert result['answer_loss']==pytest.approx(4/3)
    assert result['trace_loss']==3.
    assert result['walk_loss']==6.5
    assert result['total_loss']==6.


@pytest.mark.parametrize('free_interval,expected',[(0,[2,4,6]),(3,[3,6])])
def test_free_evaluation_can_be_decoupled_without_changing_update_count(free_interval,expected):
    trainer=object.__new__(Trainer)
    trainer.cfg=SimpleNamespace(train=SimpleNamespace(max_steps=6,log_interval=1,eval_interval=2,
        free_eval_interval=free_interval,free_gen_samples=4,stage='context',save_interval=0))
    trainer.completed=0;trainer.persist=False;events=[]
    trainer.raw=trainer.model=torch.nn.Identity()
    trainer._update=lambda **kwargs: {'lr':.01}
    trainer._touch_rng=lambda:None
    trainer._save=lambda:None
    trainer._evaluate=lambda kind:events.append((kind,trainer.completed))
    trainer.tracker=SimpleNamespace(log=lambda *args:None,finish=lambda:None)
    trainer.fit()
    assert trainer.completed==6
    assert [s for k,s in events if k=='teacher_forced']==[2,4,6]
    assert [s for k,s in events if k=='free']==expected


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA compilation equivalence check')
def test_compiled_loss_matches_reference_values_and_gradients_cuda():
    torch.manual_seed(7)
    fn=torch.compile(combine_losses,dynamic=True,fullgraph=True)
    for length in (17,33):
        x=torch.randn(5,length,29,device='cuda',requires_grad=True)
        y=x.detach().clone().requires_grad_()
        tokens=torch.randint(29,(5,length),device='cuda')
        a=torch.zeros(5,length,device='cuda');t=torch.zeros_like(a);w=torch.zeros_like(a)
        a[:3,-2]=1;t[:3,3:-2]=1;w[3:,1:]=1
        eager=combine_losses(x,tokens,a,t,w,3,2,walk_weight=.7,trace_lambda=4.)
        compiled=fn(y,tokens,a,t,w,3,2,walk_weight=.7,trace_lambda=4.)
        for key in ['total_loss','answer','trace','walk']:
            torch.testing.assert_close(eager[key],compiled[key],rtol=2e-6,atol=2e-6)
        eager['total_loss'].backward();compiled['total_loss'].backward()
        torch.testing.assert_close(x.grad,y.grad,rtol=2e-5,atol=2e-7)

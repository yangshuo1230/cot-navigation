"""Leakage, cache, gradient, sampling and GPU compiler checks; no formal run."""
from pathlib import Path
import os,sys,json,copy,time,hashlib
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'src'))
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
import torch,numpy as np
from types import SimpleNamespace
from prefix_model import PrefixLM,visibility
from runtime import MainSampler,MAIN_ENDS,Schedule
from cot_navigation.models.transformer import TransformerLM
from cot_navigation.models.compiled_inference import CompiledInference
from cot_navigation.data.dataset import collate_examples
from cot_navigation.protocols.render import RenderedExample
from cot_navigation.training.loss import combine_losses

torch.set_num_threads(2)
CFG=SimpleNamespace(layers=2,d_model=32,d_ff=128,max_seq_len=128,rope_base=10000.)
MODES=[('causal','single'),('graph','single'),('query','single'),('graph','parallel'),('query','parallel'),('graph','serial'),('query','serial')]
def make(mode,order,device='cpu'):
    torch.manual_seed(17);return PrefixLM(CFG,271,0,mode,order).to(device).eval()
def cache_test(model,device):
    prompts=[[2,14,15,2,15,16,3,14,16,4],[2,17,18,3,17,18,4]];lens=torch.tensor([10,7],device=device)
    x=torch.tensor([prompts[0],prompts[1]+[0]*3],device=device)
    cache=model.new_cache(2,device,torch.float32);cache.valid_length=10
    with torch.no_grad():model(x,positions=torch.arange(10,device=device)[None].expand(2,-1),cache=cache,cache_mode='prefill')
    tails=[[],[]]
    for step in range(3):
        vals=[14+step,17+step]
        for i in range(2):tails[i].append(vals[i])
        pos=(lens+step)[:,None];cache.valid_length=11+step
        with torch.no_grad():out=model(torch.tensor(vals,device=device)[:,None],positions=pos,cache=cache,cache_mode='decode')
        for i in range(2):
            full=torch.tensor([prompts[i]+tails[i]],device=device)
            with torch.no_grad():expected=model(full)[:,-1]
            torch.testing.assert_close(out[i,0],expected[0],atol=3e-5,rtol=3e-5)
    indices=torch.tensor([1],device=device);cache.select_batch(indices)
    assert cache.prompt_end.tolist()==[7]
    pos=torch.tensor([[10]],device=device);cache.valid_length=11
    with torch.no_grad():out=model(torch.tensor([[20]],device=device),positions=pos,cache=cache,cache_mode='decode')
    for j,(i,v) in enumerate([(1,20)]):
        with torch.no_grad():expected=model(torch.tensor([prompts[i]+tails[i]+[v]],device=device))[:,-1]
        torch.testing.assert_close(out[j,0],expected[0],atol=3e-5,rtol=3e-5)

def main():
    checks=[]
    for mode,order in MODES:
        m=make(mode,order);x=torch.tensor([[2,14,15,2,15,16,3,14,16,4,14,15,16,6,1]])
        y=x.clone();y[0,11:]=torch.tensor([17,18,7,1])
        cap1={};cap2={}
        with torch.no_grad():a=m(x,capture=cap1);b=m(y,capture=cap2)
        torch.testing.assert_close(a[:,:11],b[:,:11],atol=1e-6,rtol=1e-6)
        q=x.clone();q[0,7:9]=torch.tensor([17,18]);c={}
        with torch.no_grad():m(q,capture=c)
        delta=max(float((u[:,:6]-v[:,:6]).abs().max()) for u,v in zip(cap1['resid'],c['resid']))
        assert delta==0 if mode!='query' else delta>1e-7
        cache_test(m,'cpu')
        padded=torch.cat([x,torch.zeros((1,5),dtype=torch.long)],1)
        with torch.no_grad():torch.testing.assert_close(m(x),m(padded)[:,:x.shape[1]],atol=1e-6,rtol=1e-6)
        checks.append(dict(mode=mode,order=order,no_future_output_leak=True,cache_and_compaction=True,query_to_graph_delta=delta))
    # Identical single-route parameters and equality to native causal behavior.
    models=[make(mode,'single') for mode in ('causal','graph','query')]
    for p,q in zip(models[0].parameters(),models[1].parameters()):assert torch.equal(p,q)
    for p,q in zip(models[0].parameters(),models[2].parameters()):assert torch.equal(p,q)
    native=TransformerLM(271,2,1,32,128,128);native.load_state_dict(models[0].state_dict())
    with torch.no_grad():torch.testing.assert_close(native(x),models[0](x),atol=1e-6,rtol=1e-6)
    assert len({sum(p.numel() for p in make(*cell).parameters()) for cell in MODES})==1
    members=[SimpleNamespace(train_ids=np.arange(100),store=SimpleNamespace(arrays={'label':np.arange(100)%2})) for _ in range(8)]
    s=MainSampler(members,7)
    for step in (0,40000,80000,160000,280000,359999):
        s.step=step;batch=s.next();stage=min(int(np.searchsorted(MAIN_ENDS,step,side='right')),7)
        assert len(batch)==64 and sum(i%2 for m,i in batch)==32
        assert sum(m==stage for m,i in batch)==(64 if stage==0 else 48)
        assert all(m<=stage for m,i in batch)
    state=copy.deepcopy(s.state_dict());s2=MainSampler(members,7);s2.load_state_dict(state);assert s.next()==s2.next()
    gpu=[]
    for mode,order in MODES:
        torch._dynamo.reset();m=make(mode,order,'cuda').train();compiled=torch.compile(m,dynamic=True,fullgraph=True)
        row=RenderedExample(x[0].tolist(),10,13,[10,11,12,14],[],1,meta={})
        arrays=collate_examples([row,row],0,16)
        batch={k:torch.as_tensor(arrays[k],device='cuda') for k in ('tokens','answer_mask','trace_mask','walk_mask')}
        with torch.autocast('cuda',dtype=torch.bfloat16):
            eager=m(batch['tokens']);target=combine_losses(eager,**batch,n_task=2,n_walk=0)['total_loss']
        ge=torch.autograd.grad(target,tuple(m.parameters()))
        with torch.autocast('cuda',dtype=torch.bfloat16):
            actual=compiled(batch['tokens']);loss=combine_losses(actual,**batch,n_task=2,n_walk=0)['total_loss']
        gc=torch.autograd.grad(loss,tuple(m.parameters()))
        torch.testing.assert_close(target,loss,atol=.005,rtol=.005)
        maxgrad=max(float((p-q).abs().max()) for p,q in zip(ge,gc));assert maxgrad<.04,maxgrad
        cache_test(m.eval(),'cuda')
        # Separate production inference compiler, including immutable prefix KV.
        infer=CompiledInference(m.eval())
        prompt=x[:,:10].cuda();cache=infer.new_cache(1,torch.device('cuda'),torch.float32);cache.valid_length=10
        with torch.no_grad():
            eager=m(prompt);full=infer(prompt);prefill=infer(prompt,positions=torch.arange(10,device='cuda')[None],cache=cache,cache_mode='prefill')
            torch.testing.assert_close(full,eager,atol=3e-5,rtol=3e-5);torch.testing.assert_close(prefill,eager,atol=3e-5,rtol=3e-5)
            cache.valid_length=11;out=infer(torch.tensor([[14]],device='cuda'),positions=torch.tensor([[10]],device='cuda'),cache=cache,cache_mode='decode')
            expected=m(torch.cat([prompt,torch.tensor([[14]],device='cuda')],1))[:,-1:]
            torch.testing.assert_close(out,expected,atol=3e-5,rtol=3e-5)
        gpu.append(dict(mode=mode,order=order,compiled_training=True,compiled_prefill_decode=True,max_gradient_difference=maxgrad))
        print(json.dumps(gpu[-1]),flush=True)
    report=dict(status='PASS',cpu_checks=checks,gpu_checks=gpu,matched_parameters=True,identical_single_initialization=True,
                exact_label_and_replay_quotas=True,sampler_resume_exact=True,native_causal_equivalent=True,formal_runs_created=0)
    (HERE/'verification.json').write_text(json.dumps(report,indent=2));print('ALL CHECKS PASS',flush=True)
if __name__=='__main__':main()

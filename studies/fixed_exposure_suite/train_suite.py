"""Isolated training runtime. Preparation and --benchmark never start formal runs."""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import nullcontext
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time

HERE=Path(__file__).resolve().parent


def load_runtime(repo=None):
    location=Path(repo)/'src' if repo else HERE/'runtime/src' if (HERE/'runtime/src').exists() else Path.cwd()/'src'
    sys.path.insert(0,str(location))


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def balanced(rows, count, rng):
    groups=[[r for r in rows if r[2]==y] for y in (0,1)]
    if not all(groups):raise ValueError('Both labels required')
    result=[rng.choice(g) for g in groups for _ in range(count//2)]
    rng.shuffle(result)
    return result


def fixed_panel(rows, count=128):
    rng=random.Random(8821)
    n=min(count//2,*[sum(r[2]==y for r in rows) for y in (0,1)])
    return [r for y in (0,1) for r in rng.sample([r for r in rows if r[2]==y],n)]


def gate(history, train, monitor, rule):
    ok=(train['accuracy']>=rule['train_accuracy'] and
        monitor['balanced_accuracy']>=rule['monitor_ba'] and
        min(monitor['yes_recall'],monitor['no_recall'])>=rule['min_class_recall'])
    history.append(bool(ok))
    return len(history)>=rule['consecutive'] and all(history[-rule['consecutive']:])


class Samples:
    def __init__(self, world, protocol):
        from cot_navigation.protocols.tokenizer import Tokenizer
        from cot_navigation.protocols.render import RenderedExample
        self.world=world; self.protocol=protocol; self.tok=Tokenizer(256)
        self.nodes=[self.tok.node_id(n) for n in world['node_names']]
        self.row_type=RenderedExample; self.cache={}
        self.transition_cache={}

    def query(self, record):
        from cot_navigation.protocols.render import render_task,parametric_prompt
        key=tuple(record)
        if key not in self.cache:
            s,t,y,gap=record
            row=render_task(self.tok,self.protocol,y,
                parametric_prompt(self.tok,self.nodes,s,t),self.world['adj'],s,t,self.nodes)
            self.cache[key]=row
        return self.cache[key]

    def knowledge(self, path):
        key=tuple(path)
        if key not in self.transition_cache:
            tokens=[self.tok.id('<WALK>')]+[self.nodes[v] for v in path]+[self.tok.id('<EOS>')]
            self.transition_cache[key]=self.row_type(tokens,2,-1,[],list(range(2,len(path)+1)),-1)
        return self.transition_cache[key]

    def knowledge_batch(self,rng,arm,budget=64):
        # Same transition-loss target budget, rather than unequal per-walk means.
        paths=[]
        base=self.world['local_knowledge']; pool=self.world['fragments'][arm]
        remaining=budget
        for _ in range(min(16,budget)):
            paths.append(rng.choice(base));remaining-=1
        while remaining:
            path=rng.choice(pool)
            path=path[:remaining+1]
            paths.append(path)
            remaining-=len(path)-1
        return paths


def logical_loss(logits,tokens,answer_mask,trace_mask,walk_mask,n_task,n_transitions):
    import torch
    import torch.nn.functional as F
    nll=F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),
                        tokens[:,1:].reshape(-1),reduction='none').reshape(tokens.shape[0],-1)
    am=answer_mask[:,1:];tm=trace_mask[:,1:];wm=walk_mask[:,1:]
    answer=(nll*am).sum(-1)/am.sum(-1).clamp_min(1)
    trace=(nll*tm).sum(-1)/tm.sum(-1).clamp_min(1)
    task=(answer+trace).sum()/max(n_task,1)
    knowledge=(nll*wm).sum()/max(n_transitions,1)
    return task+knowledge,answer.sum()/max(n_task,1),knowledge


def new_model(job,device,compile_model=True):
    import torch
    from cot_navigation.models.transformer import TransformerLM
    from cot_navigation.protocols.tokenizer import Tokenizer
    torch.manual_seed(job['model_seed'])
    raw=TransformerLM(len(Tokenizer(256)),6,1,128,512,256,dropout=job['dropout']).to(device)
    # Residual-only dropout. No attention-output double masking.
    for b in raw.blocks:b.attn.drop.p=0.;b.drop.p=job['dropout']
    if job['model_seed']<0:raise ValueError('Model seed must be nonnegative')
    decay=[p for p in raw.parameters() if p.ndim>=2]
    nodecay=[p for p in raw.parameters() if p.ndim<2]
    opt=torch.optim.AdamW([dict(params=decay,weight_decay=job['weight_decay']),
                          dict(params=nodecay,weight_decay=0.)],lr=job['lr'],
                         betas=(.9,.99),fused=device.type=='cuda')
    compiled=torch.compile(raw,dynamic=True) if compile_model and device.type=='cuda' else raw
    loss_fn=torch.compile(logical_loss,dynamic=True,fullgraph=True) if compile_model and device.type=='cuda' else logical_loss
    return raw,compiled,opt,loss_fn


def tensors(rows,tok,device):
    import torch
    from cot_navigation.data.dataset import collate_examples
    batch=collate_examples(rows,tok.pad,16)
    return {k:torch.from_numpy(batch[k]).to(device) for k in ('tokens','answer_mask','trace_mask','walk_mask')}


def update(raw,model,opt,loss_fn,samples,records,paths,device,lr):
    import torch
    rows=[samples.query(r) for r in records]+[samples.knowledge(p) for p in paths]
    if not rows:raise ValueError('Empty logical batch')
    n_transitions=sum(len(p)-1 for p in paths)
    batch=tensors(rows,samples.tok,device)
    for group in opt.param_groups:group['lr']=lr
    opt.zero_grad(set_to_none=True)
    context=torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else nullcontext()
    with context:
        values=loss_fn(model(batch['tokens']),**batch,n_task=len(records),n_transitions=n_transitions)
    values[0].backward()
    norm=torch.nn.utils.clip_grad_norm_(raw.parameters(),1.,error_if_nonfinite=True)
    opt.step()
    return values,norm,len(rows),sum(len(r.tokens) for r in rows)


def evaluate(raw,samples,records,device,batch_size=128):
    from cot_navigation.evaluation.rollout import generate
    tok=samples.tok
    raw.eval()
    counts=Counter(); by_gap={};predictions=[]
    precision='bf16' if device.type=='cuda' else 'fp32'
    for start in range(0,len(records),batch_size):
        part=records[start:start+batch_size]
        prompts=[samples.query(r).tokens[:samples.query(r).prompt_len] for r in part]
        outputs=generate(raw,prompts,samples.protocol,tok,precision,max_new_tokens=32)
        for r,out in zip(part,outputs):
            # Native argmax; malformed/non-answer tokens remain errors.
            pred=out['answer_token']
            y=r[2]
            correct=pred==tok.label_id(y)
            counts['n']+=1;counts[f'n{y}']+=1;counts['correct']+=correct;counts[f'correct{y}']+=correct
            g=by_gap.setdefault(str(r[3]),Counter())
            g['n']+=1;g['correct']+=correct;g[f'n{y}']+=1;g[f'correct{y}']+=correct
            predictions.append(dict(query=r,predicted_token=pred,correct=correct,
                                    generated=out.get('generated',out.get('tokens',[]))))
    raw.train()
    if not counts['n0'] or not counts['n1']:raise ValueError('Evaluation needs both labels')
    yes=counts['correct1']/counts['n1'];no=counts['correct0']/counts['n0']
    return dict(n=counts['n'],accuracy=counts['correct']/counts['n'],balanced_accuracy=(yes+no)/2,
                yes_recall=yes,no_recall=no,by_gap={k:dict(v) for k,v in by_gap.items()},predictions=predictions)


def save_atomic(path,payload):
    import torch
    temp=path.with_suffix('.writing')
    torch.save(payload,temp);temp.replace(path)


def train(job,world,output,device,resume=False,tracking='online'):
    import torch
    if output.exists() and not resume:raise FileExistsError('Refusing to replace existing run')
    output.mkdir(parents=True,exist_ok=True)
    lock=(output/'.run.lock').open('a+b')
    if os.name=='nt':
        import msvcrt
        if lock.tell()==0:lock.write(b'0');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    else:
        import fcntl
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    raw,model,opt,loss_fn=new_model(job,device)
    phase='knowledge' if job['kind']=='knowledge' else 'one_hop' if job['kind']=='grokking' else 'fixed'
    step=0;phase_step=0;history=[];counts=Counter();q_rng=random.Random(42);k_rng=random.Random(43)
    run_id=None
    parent=None
    if not resume and job['dependencies']:
        # Resolve from the explicit common output root, not the nesting depth.
        root=Path(job['_output_root'])
        parent=root/job['dependencies'][0]/'latest.pt'
        if not parent.is_file():raise FileNotFoundError(parent)
        blob=torch.load(parent,map_location='cpu',weights_only=False)
        if blob['world_hash']!=job['_world_hash']:raise ValueError('Parent graph mismatch')
        raw.load_state_dict(blob['model'])
    if resume:
        blob=torch.load(output/'latest.pt',map_location=device,weights_only=False)
        if blob['job']!=job or blob['world_hash']!=job['_world_hash']:raise ValueError('Resume identity changed')
        raw.load_state_dict(blob['model']);opt.load_state_dict(blob['optimizer'])
        step=blob['step'];phase_step=blob['phase_step'];phase=blob['phase'];history=blob['history']
        counts.update(blob['counts']);q_rng.setstate(blob['query_rng']);k_rng.setstate(blob['knowledge_rng'])
        torch.set_rng_state(blob['torch_rng'].cpu())
        if device.type=='cuda':torch.cuda.set_rng_state(blob['cuda_rng'].cpu())
        run_id=blob.get('wandb_id')
    if (output/'COMPLETE.json').exists():return
    (output/'resolved_job.json').write_text(json.dumps(job,indent=2))
    run=None
    if tracking=='online':
        import wandb
        run=wandb.init(project='cot-graph-reasoning',group='fixed_exposure_grokking_v2',
            name=job['id'],config={k:v for k,v in job.items() if not k.startswith('_')},
            id=run_id,resume='allow' if run_id else None,dir=str(output),mode='online')
        if run.offline:raise RuntimeError('Expected online tracking')
        run_id=run.id
        run.define_metric('optimizer_step')
        run.define_metric('*',step_metric='optimizer_step')
    samples=Samples(world,job['protocol'])
    complete=False
    def payload():
        return dict(model=raw.state_dict(),optimizer=opt.state_dict(),job=job,
            world_hash=job['_world_hash'],step=step,phase_step=phase_step,phase=phase,
            history=history,counts=dict(counts),query_rng=q_rng.getstate(),knowledge_rng=k_rng.getstate(),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state() if device.type=='cuda' else None,
            wandb_id=run_id,parent_sha256=sha_file(parent) if parent else None)
    start=time.perf_counter()
    try:
        while not complete:
            if phase=='knowledge':records=[]
            elif phase=='fixed':records=balanced(world['base']['train'],64,q_rng)
            elif phase=='one_hop':records=balanced(world['one_hop']['train'],64,q_rng)
            else:
                records=balanced(world['one_hop']['train'],16,q_rng)+balanced(world['multihop']['train'],48,q_rng)
                q_rng.shuffle(records)
            arm=job['arm'] if phase=='fixed' else 'R0'
            paths=samples.knowledge_batch(k_rng,arm,job['knowledge_transition_budget'])
            # Query/fragment identities and transition counts are exact, not inferred from caches.
            for r in records:counts['q:'+','.join(map(str,r))]+=1
            for p in paths:counts['k:'+','.join(map(str,p))]+=1
            lr=job['lr']*min(1.,(phase_step+1)/job['warmup_steps']) if phase!='multi' else job['lr']
            values,norm,nrows,ntokens=update(raw,model,opt,loss_fn,samples,records,paths,device,lr)
            step+=1;phase_step+=1
            if step%job['log_interval']==0:
                if device.type=='cuda':torch.cuda.synchronize()
                row=dict(step=step,phase=phase,phase_step=phase_step,loss=float(values[0]),
                    answer_loss=float(values[1]),knowledge_loss=float(values[2]),grad_norm=float(norm),
                    lr=lr,elapsed=time.perf_counter()-start,rows=nrows,effective_tokens=ntokens)
                with (output/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
                if run:run.log({**{k:v for k,v in row.items() if k!='phase'},'optimizer_step':step})
            if phase=='knowledge' and phase_step>=job['steps']:
                complete=True
            elif phase!='knowledge' and (phase_step%job['eval_interval']==0 or
                    (phase=='fixed' and phase_step>=job['steps'])):
                torch_rng=torch.get_rng_state()
                cuda_rng=torch.cuda.get_rng_state() if device.type=='cuda' else None
                training=world['base']['train'] if phase=='fixed' else world['one_hop']['train'] if phase=='one_hop' else world['multihop']['train']
                monitor=world['core_monitor'] if phase=='fixed' else world['multihop' if phase=='multi' else 'one_hop']['monitor']
                # Long multi-hop monitor uses a fixed balanced 256-query panel.
                if phase=='multi':monitor=fixed_panel(monitor,256)
                train_result=evaluate(raw,samples,fixed_panel(training),device)
                monitor_result=evaluate(raw,samples,monitor,device)
                report=dict(step=step,phase=phase,phase_step=phase_step,train=train_result,monitor=monitor_result)
                (output/f'eval_{step:09d}.json').write_text(json.dumps(report))
                if run:run.log({'monitor/ba':monitor_result['balanced_accuracy'],
                    'monitor/yes_recall':monitor_result['yes_recall'],'monitor/no_recall':monitor_result['no_recall'],
                    'train/accuracy':train_result['accuracy'],'optimizer_step':step})
                torch.set_rng_state(torch_rng)
                if cuda_rng is not None:torch.cuda.set_rng_state(cuda_rng)
                if phase=='fixed':complete=phase_step>=job['steps']
                elif gate(history,train_result,monitor_result,job['gate']):
                    save_atomic(output/f'{phase}_grokking.pt',payload())
                    if phase=='one_hop':
                        (output/'PHASE1_GROKKED.json').write_text(json.dumps(dict(step=step,monitor=monitor_result)))
                        one_final=evaluate(raw,samples,world['one_hop']['final'],device)
                        (output/'one_hop_final.json').write_text(json.dumps(one_final))
                        phase='multi';phase_step=0;history=[]
                    else:complete=True
            if step%job['checkpoint_interval']==0 or complete:
                save_atomic(output/'latest.pt',payload())
            if complete and phase!='knowledge':
                final=world['core_final'] if phase=='fixed' else fixed_panel(world['multihop']['final'],256)
                result=evaluate(raw,samples,final,device)
                (output/'final.json').write_text(json.dumps(result))
        (output/'COMPLETE.json').write_text(json.dumps(dict(step=step,phase=phase)))
    finally:
        if run:run.finish()
        lock.close()


def benchmark(world,device,output,iterations=100):
    import torch
    results=[]
    specifications=[('fixed',p,a,0.) for p in ('direct','path') for a in ('R0','R2','R3')]
    specifications += [('one_hop','direct','R0',p) for p in (0.,.1)]
    for phase,protocol,arm,dropout in specifications:
        job=dict(model_seed=0,dropout=dropout,weight_decay=.1,lr=1e-4)
        raw,model,opt,loss_fn=new_model(job,device)
        samples=Samples(world,protocol)
        q_rng=random.Random(42);k_rng=random.Random(43)
        pool=world['one_hop' if phase=='one_hop' else 'base']['train']
        exposure_counts=Counter()
        times=[];cold=time.perf_counter()
        for i in range(iterations+10):
            if device.type=='cuda':torch.cuda.synchronize()
            start=time.perf_counter()
            records=balanced(pool,64,q_rng)
            paths=samples.knowledge_batch(k_rng,arm)
            for r in records:exposure_counts['q:'+','.join(map(str,r))]+=1
            for path in paths:exposure_counts['k:'+','.join(map(str,path))]+=1
            update(raw,model,opt,loss_fn,samples,records,paths,device,1e-4)
            if device.type=='cuda':torch.cuda.synchronize()
            if i>=10:times.append(time.perf_counter()-start)
        compile_warmup=time.perf_counter()-cold-sum(times)
        panel=fixed_panel(world['one_hop' if phase=='one_hop' else 'base']['train'],128)
        start=time.perf_counter();evaluate(raw,samples,panel,device)
        if device.type=='cuda':torch.cuda.synchronize()
        result=dict(phase=phase,protocol=protocol,arm=arm,dropout=dropout,
            median_update_seconds=statistics.median(times),mean_update_seconds=statistics.mean(times),
            compile_warmup_seconds=compile_warmup,eval_128_seconds=time.perf_counter()-start,
            iterations=iterations,gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu')
        results.append(result);print(json.dumps(result),flush=True)
        del raw,model,opt,loss_fn
        if device.type=='cuda':torch.cuda.empty_cache()
    output.write_text(json.dumps(results,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job');p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--repo-root',type=Path);p.add_argument('--output-root',type=Path)
    p.add_argument('--execute',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--benchmark',action='store_true');p.add_argument('--iterations',type=int,default=100)
    args=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    load_runtime(args.repo_root)
    import torch
    torch.set_num_threads(2)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    manifest=json.loads((HERE/'manifest.json').read_text())
    job=next((j for j in manifest['jobs'] if j['id']==args.job),None)
    if not args.benchmark and job is None:raise ValueError('Specify an exact prepared --job')
    graph=job['graph'] if job else 0
    meta=manifest['worlds'][graph]
    world=json.loads((HERE/meta['path']).read_text())
    from prepare_suite import digest
    if digest(world)!=meta['sha256']:raise ValueError('Dataset changed')
    if args.benchmark:
        benchmark(world,device,HERE/'benchmark.json',args.iterations);return
    output_root=(args.output_root or (args.repo_root/'outputs/exposure_grokking_v2' if args.repo_root else HERE/'outputs')).resolve()
    job=copy.deepcopy(job);job['_world_hash']=meta['sha256'];job['_output_root']=str(output_root)
    job['_code_sha256']=sha_file(__file__)
    if not args.execute:
        print(json.dumps(dict(job=job,device=str(device),training_started=False),indent=2));return
    train(job,world,output_root/job['id'],device,args.resume)


if __name__=='__main__':main()

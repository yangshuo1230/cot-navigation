"""Isolated paper-inspired reference, no imports or edits to active experiments."""
from pathlib import Path
from contextlib import nullcontext
import os,json,random,time,argparse,hashlib
HERE=Path(__file__).resolve().parent
PAD,BOS,EOS,YES,NO=200,201,202,203,204
VOCAB=205

def sequence(row,names,protocol):
    # Paper: goal: target start [intermediate ... target] label EOS.
    body=[names[u] for u in row['path'][1:]] if protocol=='path' and row['y'] else []
    return [BOS,names[row['t']],names[row['s']]]+body+[YES if row['y'] else NO,EOS]

def model_factory(job):
    import torch
    from torch import nn
    import torch.nn.functional as F
    d=job['width'];h=job['heads']
    class Attention(nn.Module):
        def __init__(self):
            super().__init__();self.qkv=nn.Linear(d,3*d,bias=False);self.proj=nn.Linear(d,d,bias=False)
        def forward(self,x):
            b,t,_=x.shape
            q,k,v=[a.view(b,t,h,d//h).transpose(1,2) for a in self.qkv(x).chunk(3,-1)]
            a=F.scaled_dot_product_attention(q,k,v,is_causal=True)
            return self.proj(a.transpose(1,2).contiguous().view(b,t,d))
    class Block(nn.Module):
        def __init__(self):
            super().__init__();self.pre=nn.LayerNorm(d,bias=False);self.post=nn.LayerNorm(d,bias=False)
            self.attn=Attention();self.mlp=nn.Sequential(nn.Linear(d,4*d,bias=False),nn.GELU(),nn.Linear(4*d,d,bias=False))
        def forward(self,x):
            x=x+self.attn(self.pre(x));return x+self.mlp(self.post(x))
    class Model(nn.Module):
        def __init__(self):
            super().__init__();self.emb=nn.Embedding(VOCAB,d);self.pos=nn.Embedding(32,d)
            self.blocks=nn.Sequential(*[Block() for _ in range(job['layers'])]);self.norm=nn.LayerNorm(d,bias=False)
            self.head=nn.Linear(d,VOCAB,bias=False);self.head.weight=self.emb.weight
            for m in self.modules():
                if isinstance(m,(nn.Embedding,nn.Linear)):nn.init.normal_(m.weight,std=.02)
        def forward(self,x):
            return self.head(self.norm(self.blocks(self.emb(x)+self.pos(torch.arange(x.shape[1],device=x.device)))))
    return Model()

def evaluate(model,records,w,protocol,device):
    import torch
    model.eval();predictions=[];names=w['node_names'];inv={v:i for i,v in enumerate(names)}
    with torch.no_grad():
        for offset in range(0,len(records),256):
            part=records[offset:offset+256]
            x=torch.tensor([[BOS,names[t],names[s]] for s,t,y in part],device=device)
            generated=[[] for _ in part];done=[False]*len(part)
            with torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else nullcontext():
                for _ in range(12 if protocol=='path' else 1):
                    nxt=model(x)[:,-1].argmax(-1)
                    for i,v in enumerate(nxt.tolist()):
                        if not done[i]:
                            generated[i].append(v)
                            if v in (YES,NO,EOS):done[i]=True
                    x=torch.cat([x,nxt[:,None]],1)
                    if all(done):break
            for (s,t,y),tokens in zip(part,generated):
                pred=tokens[-1] if tokens and tokens[-1] in (YES,NO) else -1
                p=[s]+[inv[v] for v in tokens[:-1] if v in inv]
                valid=(pred==YES and p[-1]==t and all(v in w['adj'][u] for u,v in zip(p,p[1:])) and all(v in inv for v in tokens[:-1])) if protocol=='path' else None
                predictions.append(dict(s=s,t=t,y=y,gap=t//20-s//20,pred=pred,correct=pred==(YES if y else NO),valid_path=valid,generated=tokens))
    model.train()
    def stats(rows):
        yes=[r for r in rows if r['y']];no=[r for r in rows if not r['y']]
        yr=sum(r['correct'] for r in yes)/len(yes) if yes else None
        nr=sum(r['correct'] for r in no)/len(no) if no else None
        return dict(n=len(rows),accuracy=sum(r['correct'] for r in rows)/max(1,len(rows)),yes_recall=yr,no_recall=nr,
            balanced_accuracy=(yr+nr)/2 if yr is not None and nr is not None else None,
            valid_positive_path_rate=sum(bool(r['valid_path']) for r in yes)/max(1,len(yes)) if protocol=='path' else None)
    return dict(**stats(predictions),forward_only=stats([r for r in predictions if r['gap']>0]),
        by_gap={str(g):stats([r for r in predictions if r['gap']==g]) for g in sorted({r['gap'] for r in predictions})},predictions=predictions)

def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--repo-root',type=Path,required=True);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2)
    manifest=json.loads((HERE/'manifest.json').read_text());j=next(j for j in manifest['jobs'] if j['id']==args.job)
    dp=HERE/'data'/f"graph_{j['graph']}.json"
    assert hashlib.sha256(dp.read_bytes()).hexdigest()==j['data_sha256']
    w=json.loads(dp.read_text());split=w['splits'][j['mode']];device=torch.device('cuda')
    torch.manual_seed(j['model_seed']);torch.cuda.manual_seed_all(j['model_seed']);rng=random.Random(9000+j['model_seed'])
    raw=model_factory(j).to(device);opt=torch.optim.Adam(raw.parameters(),lr=j['lr'],betas=tuple(j['betas']),fused=True)
    rows=[sequence(r,w['node_names'],j['protocol']) for r in split['sequences']]
    assert max(map(len,rows))<=32
    dataset=torch.full((len(rows),16),PAD,dtype=torch.long,device=device)
    for i,row in enumerate(rows):dataset[i,:len(row)]=torch.tensor(row,device=device)
    out=args.repo_root/'outputs/stepwise_reference'/j['id']
    if not args.smoke:
        out.mkdir(parents=True,exist_ok=True)
        import fcntl
        lock=(out/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'COMPLETE.json').exists():return
        if (out/'latest.pt').exists():raise RuntimeError('Explicit resume required; refusing silent restart')
        (out/'resolved_job.json').write_text(json.dumps(j,indent=2))
        import wandb
        run=wandb.init(project='cot-graph-reasoning',entity='yangshuo1230-sjtu-hpc-center',group=manifest['group'],name=j['id'],config=j,dir=str(out),mode='online')
        if run.offline:raise RuntimeError('Online W&B required')
    model=raw if args.smoke else torch.compile(raw,dynamic=False)
    start=time.monotonic();last=None
    for step in range(1,(5 if args.smoke else j['steps'])+1):
        indices=torch.tensor([rng.randrange(len(rows)) for _ in range(j['batch_size'])],device=device)
        x=dataset[indices];opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(x[:,:-1]);loss=F.cross_entropy(logits.float().reshape(-1,VOCAB),x[:,1:].reshape(-1),ignore_index=PAD)
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(raw.parameters(),1.,error_if_nonfinite=True);opt.step()
        if args.smoke:continue
        if step%100==0:
            metric=dict(step=step,loss=loss.item(),grad_norm=float(norm),elapsed=time.monotonic()-start)
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(metric)+'\n')
            run.log(metric,step=step,commit=not (step%j['eval_interval']==0 or step==j['steps']))
        if step%j['eval_interval']==0 or step==j['steps']:
            # All-edge query recall detects the intended interface alignment.
            tr=random.Random(77).sample(split['train'],min(512,len(split['train'])))
            last=dict(step=step,train=evaluate(raw,tr,w,j['protocol'],device),
                monitor=evaluate(raw,split['monitor'],w,j['protocol'],device),
                edge_queries=evaluate(raw,[[u,v,1] for u,vs in enumerate(w['adj']) for v in vs],w,j['protocol'],device))
            (out/f'eval_{step:09d}.json').write_text(json.dumps(last))
            run.log({'monitor/ba':last['monitor']['balanced_accuracy'],'monitor/forward_ba':last['monitor']['forward_only']['balanced_accuracy'],
                'train/accuracy':last['train']['accuracy'],'edges/accuracy':last['edge_queries']['accuracy']},step=step)
        if step%j['checkpoint_interval']==0 or step==j['steps']:
            temp=out/'latest.writing';torch.save(dict(step=step,model=raw.state_dict(),optimizer=opt.state_dict(),job=j,
                python_rng=rng.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),temp);temp.replace(out/'latest.pt')
    if args.smoke:
        report=evaluate(raw,split['monitor'][:16]+split['monitor'][-16:],w,j['protocol'],device)
        print(json.dumps(dict(smoke='PASS',job=j['id'],updates=5,loss=loss.item(),parameters=sum(p.numel() for p in raw.parameters()),evaluation_n=report['n'],max_sequence=max(map(len,rows)))))
    else:
        final=evaluate(raw,split['final'],w,j['protocol'],device)
        (out/'final.json').write_text(json.dumps(final));run.summary['final_ba']=final['balanced_accuracy'];run.finish()
        (out/'COMPLETE.json').write_text(json.dumps(dict(step=step)))
if __name__=='__main__':main()

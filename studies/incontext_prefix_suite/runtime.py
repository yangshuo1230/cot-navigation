"""Adapters over the current optimized Trainer; shared source is never changed."""
from pathlib import Path
from dataclasses import replace
import hashlib,json,math,time,copy
import numpy as np
import torch
import torch.nn.functional as F
from cot_navigation.training import trainer as engine
from cot_navigation.training.sampler import IndexSampler
from cot_navigation.data.dataset import ContextStore
from cot_navigation.data.prepared import PreparedRows
from cot_navigation.data.shards import load_npz,is_complete
from cot_navigation.protocols.render import RenderedExample
from prefix_model import PrefixLM

HERE=Path(__file__).resolve().parent
BOOT_ENDS=np.cumsum([200000,30000,30000,30000,50000,30000,50000,40000,30000,60000])
BOOT_POOLS=[('full_v16_m2',.8),('full_v16_m2',.5),('full_v16_m2',.2),('full_v16_m2',0),
            ('full_v64_m2',0),('full_v256_m2',0),('full_v256_m4',0),('full_v256_m8',0),('full_v256_m16',0),('full_v256_m24',0)]
MAIN_ENDS=np.cumsum([40000]*7+[80000])

def read_prepared(store,cfg,tok):
    from cot_navigation.data.scale_prepared import identity
    expected=identity(store,cfg,tok)
    matches=[]
    for p in (store.root/'prepared').glob(f"scale_{cfg.protocol.name}_*.json"):
        spec=json.loads(p.read_text())
        if all(spec.get(k)==v for k,v in expected.items() if k!='max_seq_len') and spec.get('max_seq_len',0)>=cfg.model.max_seq_len:
            matches.append(p)
    if not matches:
        if cfg.protocol.name=='direct':return PreparedRows(store,'direct',tok)
        raise FileNotFoundError(f'No compatible immutable prepared cache: {store.root}')
    p=sorted(matches,key=lambda p:json.loads(p.read_text())['max_seq_len'])[0]
    binary=p.with_suffix('.npz');assert is_complete(binary),binary
    obj=PreparedRows.__new__(PreparedRows);obj.protocol=cfg.protocol.name;obj.arrays=load_npz(binary)
    assert int(np.diff(obj.arrays['offsets']).max())<=cfg.model.max_seq_len-cfg.protocol.cot_margin
    obj.source_path=str(binary)
    return obj

class MainSampler:
    def __init__(self,members,seed):
        self.step=0;self.rng=np.random.default_rng(seed+31);self.groups=[];self.samplers=[]
        for m in members:
            g=[np.flatnonzero(m.store.arrays['label'][m.train_ids]==y) for y in (0,1)]
            assert all(len(a) for a in g);self.groups.append(g)
            self.samplers.append([IndexSampler(len(g[y]),64,seed+1000+2*len(self.groups)+y) for y in (0,1)])
    def next(self):
        stage=min(int(np.searchsorted(MAIN_ENDS,self.step,side='right')),7)
        quotas={stage:64 if stage==0 else 48}
        if stage:
            for i in range(16):
                k=(self.step*16+i)%stage;quotas[k]=quotas.get(k,0)+1
        result=[]
        consumed=0
        for m,count in sorted(quotas.items()):
            # Alternating odd quota assignments keep each stage balanced over time.
            z=count//2+int(count%2 and (self.step+consumed)%2==0)
            consumed+=count
            for y,n in ((0,z),(1,count-z)):
                ids=self.samplers[m][y].take(n)
                result.extend((m,int(self.groups[m][y][i])) for i in ids)
        self.rng.shuffle(result);self.last_batch=result;self.step+=1
        return result
    def state_dict(self):
        return dict(kind='prefix_exact_main_v1',step=self.step,rng=self.rng.bit_generator.state,
                    samplers=[[s.state_dict() for s in row] for row in self.samplers])
    def load_state_dict(self,state):
        assert state['kind']=='prefix_exact_main_v1'
        self.step=state['step'];self.rng.bit_generator.state=state['rng']
        for row,saved in zip(self.samplers,state['samplers']):
            for s,v in zip(row,saved):s.load_state_dict(v)

class Schedule:
    def __init__(self,opt,boot,updates=0):self.optimizer=opt;self.boot=boot;self.updates=updates
    def lr_at(self,n):
        t=n if n<self.boot else n-self.boot;duration=self.boot if n<self.boot else 360000
        hi,lo=(3e-4,3e-5) if n<self.boot else (1e-4,1e-5)
        if t<1000:return hi*(t+1)/1000
        progress=min(1,max(0,(t-1000)/(.9*duration-1000)))
        return lo+(hi-lo)*.5*(1+math.cos(math.pi*progress))
    def set_lr(self):
        value=self.lr_at(self.updates)
        for g in self.optimizer.param_groups:g['lr']=value
        return value
    def step(self):self.updates+=1;return self.set_lr()
    def state_dict(self):
        return dict(updates=self.updates,warmup_steps=1000,max_steps=self.boot+360000,base_lr=1e-4,min_lr=1e-5,prefix_schedule=True,boot=self.boot)

class PrefixTrainer(engine.Trainer):
    def __init__(self,cfg,job,**kw):
        self.job=job;self.boot=550000 if job['protocol']=='direct' else 0
        self.boot_rng=torch.Generator(device='cuda').manual_seed(83000+job['seed'])
        self.boot_pools={};self.optimizer_reset=not bool(self.boot);self.in_evaluation=False
        engine.build_model=lambda c,v,pad_id=0:PrefixLM(c,v,pad_id,job['mask'],job['order'])
        super().__init__(cfg,**kw)
        self.schedule=Schedule(self.opt,self.boot,self.completed);self.schedule.set_lr()
        self.bootstrap_loss=torch.compile(lambda logits,labels:F.cross_entropy(logits.float(),labels),dynamic=True,fullgraph=True) if cfg.model.compile else lambda a,b:F.cross_entropy(a.float(),b)
        self.snapshot['prefix_experiment']=dict(job=job,contract=hashlib.sha256((HERE/'manifest.json').read_bytes()).hexdigest())
    def _load_resume(self,path):
        blob=torch.load(path,map_location='cpu',weights_only=False)
        old=blob['arch'].get('prefix_experiment');assert old and old['job']==self.job
        self.optimizer_reset=blob['prefix_state']['optimizer_reset'];self.boot_rng.set_state(blob['prefix_state']['boot_rng'])
        self.snapshot['prefix_experiment']=old
        super()._load_resume(path)
    def _preload_member(self,member):
        # Keep flat immutable caches; materialize rows in the existing Trainer cache
        # as used. Avoid constructing 1.6M Python rows in every process at startup.
        member.prepared_tasks=read_prepared(member.store,self.cfg,self.tok)
    def _init_samplers(self):
        super()._init_samplers();self.query_sampler=MainSampler(self.scale_members,self.cfg.train.exposure_seed)
    def _scale_weights(self):
        n=max(0,self.completed-self.boot-int(self.in_evaluation)) if hasattr(self,'completed') else 0
        stage=min(int(np.searchsorted(MAIN_ENDS,n,side='right')),7)
        return [1. if i==stage else 0. for i in range(8)]
    def _boot_stage(self):return min(int(np.searchsorted(BOOT_ENDS,self.completed,side='right')),9)
    def _pool(self,key,split='train'):
        cache_key=(key,split)
        if cache_key not in self.boot_pools:
            if key=='standard':
                member=self.scale_members[0];ids=member.store.indices(split)
                rows=[member.prepared_tasks.row(int(i)) for i in ids]
                x=np.asarray([r.tokens[:r.prompt_len] for r in rows],np.int64);y=np.asarray([r.label for r in rows],np.int64)
            else:
                path=Path('studies/direct_learning/data_v1')/(key+'.npz')
                with np.load(path) as data:x=data[split+'_x'];y=data[split+'_y']
            pool=dict(x=torch.as_tensor(x,device=self.device),y=torch.as_tensor(y,device=self.device))
            pool['target']=torch.where(pool['y'].bool(),self.tok.yes,self.tok.no)
            pool['groups']=[torch.where(pool['y']==y)[0] for y in (0,1)]
            self.boot_pools[cache_key]=pool
        return self.boot_pools[cache_key]
    def _sample_boot(self,key,count):
        p=self._pool(key)
        ids=torch.cat([g[torch.randint(len(g),(count//2,),generator=self.boot_rng,device=self.device)] for g in p['groups']])
        return p['x'][ids],p['target'][ids]
    def _bootstrap_update(self,collect):
        t0=time.perf_counter();stage=self._boot_stage()
        if self.job['recipe']=='standard':x,y=self._sample_boot('standard',64)
        else:
            key,share=BOOT_POOLS[stage]
            # Match fractional quotas cumulatively in label-balanced pairs.
            previous=0 if stage==0 else int(BOOT_ENDS[stage-1]);local=self.completed-previous
            count=2*(int((local+1)*32*share+1e-9)-int(local*32*share+1e-9))
            parts=[self._sample_boot('fixed_source_v16_m2',count)] if count else []
            if count<64:parts.append(self._sample_boot(key,64-count))
            x=torch.cat([p[0] for p in parts]);y=torch.cat([p[1] for p in parts])
        edges=x[:,:-4].reshape(64,-1,3)
        perm=torch.rand(edges.shape[:2],device=self.device,generator=self.boot_rng).argsort(-1)
        x=torch.cat([edges.gather(1,perm[:,:,None].expand(-1,-1,3)).flatten(1),x[:,-4:]],1)
        self.opt.zero_grad(set_to_none=True);lr=self.schedule.set_lr()
        with torch.autocast('cuda',dtype=torch.bfloat16):logits=self.model(x)[:,-1]
        loss=self.bootstrap_loss(logits,y);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(self.raw.parameters(),1.,error_if_nonfinite=True)
        self.opt.step();self.schedule.step();self.queries_seen+=64;self.tokens_seen+=x.numel()
        if not collect:return dict(lr=lr)
        torch.cuda.synchronize()
        return dict(answer_loss=float(loss),total_loss=float(loss),grad_norm=float(norm),lr=lr,
                    phase='bootstrap',bootstrap_stage=stage,update_time=time.perf_counter()-t0,
                    effective_tokens=x.numel(),queries_seen=self.queries_seen,tokens_seen=self.tokens_seen,
                    microbatches=1,gpu_mem_mb=torch.cuda.max_memory_allocated()/1024**2)
    def _update(self,collect_metrics=True):
        boundary=self.completed in set(map(int,BOOT_ENDS)) if self.boot and self.completed<=self.boot else self.completed-self.boot in set(map(int,MAIN_ENDS))
        if self.completed and boundary and self.completed%20000:
            self._evaluate('teacher_forced');self._evaluate('free');self._save()
        if self.completed<self.boot:return self._bootstrap_update(collect_metrics)
        if not self.optimizer_reset:
            self.opt,self.opt_info=engine._optimizer(self.raw,self.cfg,self.device)
            self.schedule=Schedule(self.opt,self.boot,self.completed);self.optimizer_reset=True
            self.boot_pools.clear();torch.cuda.empty_cache()
        result=super()._update(collect_metrics)
        if collect_metrics:
            result['phase']='main';result['main_step']=self.completed-self.boot+1
            result['exposure_batch_sha256']=hashlib.sha256(json.dumps(self.query_sampler.last_batch).encode()).hexdigest()
        return result
    def _evaluate(self,kind):
        if self.completed<=self.boot and self.boot:
            if kind!='free':return
            # Completed-update stage, not the next not-yet-trained stage.
            stage=min(int(np.searchsorted(BOOT_ENDS,max(0,self.completed-1),side='right')),9)
            key=BOOT_POOLS[stage][0] if self.job['recipe']=='stepwise' else 'standard'
            p=self._pool(key,'val');n=min(256,len(p['y']));ids=torch.arange(n,device=self.device)
            self.raw.eval();m=self._inference_model()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):logits=m(p['x'][ids])[:,-1].float()
            pred=logits.argmax(-1);correct=pred==p['target'][ids]
            recalls=[float(correct[p['y'][ids]==y].float().mean()) for y in (0,1)]
            report=dict(step=self.completed,phase='bootstrap',stage=stage,pool=key,accuracy=float(correct.float().mean()),
                        balanced_accuracy=sum(recalls)/2,yes_recall=recalls[1],no_recall=recalls[0],ce=float(F.cross_entropy(logits,p['target'][ids])))
            if self.persist:
                (self.out/'eval').mkdir(exist_ok=True)
                (self.out/'eval'/f'bootstrap_{self.completed:08d}.json').write_text(json.dumps(report,indent=2))
            self.tracker.log(self.completed,{'free/overall/balanced_accuracy':report['balanced_accuracy'],'bootstrap/ce':report['ce']})
            self.raw.train();return
        self.in_evaluation=True
        try:super()._evaluate(kind)
        finally:self.in_evaluation=False
    def _payload(self):
        p=super()._payload();p['prefix_state']=dict(optimizer_reset=self.optimizer_reset,boot_rng=self.boot_rng.get_state());return p
    def _evaluate_final_checkpoints(self):
        if self.completed%20000:
            self._evaluate('teacher_forced');self._evaluate('free')
        self.raw.eval();reports={}
        for name,member in [('simple100',self.scale_members[3]),('hard100',self.scale_members[7])]:
            self._activate(member);ids=self._selected_ids('test',512)
            rows=[self._render_task(i) for i in ids]
            for view in ('native','reverse','shuffle'):
                rng=np.random.default_rng(20261024);changed=[]
                for r in rows:
                    e=np.asarray(r.tokens[:r.prompt_len-4]).reshape(-1,3)
                    order=np.arange(len(e))[::-1] if view=='reverse' else rng.permutation(len(e)) if view=='shuffle' else np.arange(len(e))
                    changed.append(replace(r,tokens=e[order].ravel().tolist()+r.tokens[r.prompt_len-4:]))
                report=self._free(changed,record_name=f'final_{name}_{view}')
                reports[name+'/'+view]=report
                self._semantic(name,view,member.store)
            reports[name+'/counterfactual']=self._counterfactuals(name,rows,member.store)
        manifest=json.loads((HERE/'manifest.json').read_text())
        for name,root in manifest['ood'].items():
            store=ContextStore(root);prepared=read_prepared(store,self.cfg,self.tok)
            self.store=store;self.prepared_tasks=prepared;self._active_member=root
            ids=self._selected_ids('test',256);rows=[self._render_task(i) for i in ids]
            reports[name]=self._free(rows,record_name=f'final_{name}_native');self._semantic(name,'native',store)
        if self.persist:(self.out/'final.json').write_text(json.dumps(reports,indent=2))
        self.raw.train()
    def _counterfactuals(self,name,rows,store):
        native={r['query_id']:r for r in [json.loads(l) for l in (self.out/'eval'/f'predictions_final_{name}_native.jsonl').read_text().splitlines()]}
        rng=np.random.default_rng(419);reports={}
        def reachable(adj,s,t):
            seen={s};todo=[s]
            for u in todo:
                for v in adj[u]:
                    if v not in seen:seen.add(v);todo.append(v)
            return t in seen
        for view in ('source_flip','target_flip','graph_flip'):
            changed=[]
            for row in rows:
                index=int(row.meta['index']);adj=store.adj(index);nodes=store.node_vocab_ids(index,self.tok)
                s,t=int(store.arrays['s_topo'][index]),int(store.arrays['t_topo'][index])
                desired=1-row.label;ns,nt=s,t;edges=[(u,v) for u,vs in enumerate(adj) for v in vs]
                if view!='graph_flip':
                    candidates=rng.permutation(len(nodes));found=False
                    for v in candidates:
                        a,b=(int(v),t) if view=='source_flip' else (s,int(v))
                        if a!=b and int(reachable(adj,a,b))==desired:ns,nt=a,b;found=True;break
                    if not found:continue
                elif not row.label:
                    # Only add a forward edge in the hidden topological DAG.
                    if s>=t:continue
                    edges.append((s,t))
                else:
                    found=False
                    for k in rng.permutation(len(edges)):
                        candidate=edges[:int(k)]+edges[int(k)+1:];other=[[] for _ in adj]
                        for a,b in candidate:other[a].append(b)
                        if not reachable(other,s,t):edges=candidate;found=True;break
                    if not found:continue
                modified=[[] for _ in adj]
                for a,b in edges:modified[a].append(b)
                assert int(reachable(modified,ns,nt))==desired
                # Keep input edge order for endpoint flips; graph edits get a
                # fresh deterministic permutation rather than a position cue.
                if view=='graph_flip':
                    rng.shuffle(edges);prompt=[z for a,b in edges for z in (self.tok.id('<EDGE>'),nodes[a],nodes[b])]
                else:prompt=row.tokens[:row.prompt_len-4]
                prompt=prompt+[self.tok.id('<QUERY>'),nodes[ns],nodes[nt],self.tok.id('|')]
                meta=dict(row.meta,distance=-1)
                changed.append(replace(row,tokens=prompt+[self.tok.label_id(desired),self.tok.id('<EOS>')],
                    prompt_len=len(prompt),answer_index=len(prompt),label=desired,trace_indices=[],meta=meta))
            if changed:
                report=self._free(changed,record_name=f'final_{name}_{view}')
                generated=[json.loads(l) for l in (self.out/'eval'/f'predictions_final_{name}_{view}.jsonl').read_text().splitlines()]
                report['eligible_pairs']=len(generated)
                report['pair_both_correct_rate']=sum(r['correct'] and native[r['query_id']]['correct'] for r in generated)/len(generated)
                self._semantic(name,view,store);reports[view]=report
            else:reports[view]=dict(eligible_pairs=0)
        return reports
    def _semantic(self,name,view,store):
        if not self.persist or self.cfg.protocol.name!='path':return
        path=self.out/'eval'/f'predictions_final_{name}_{view}.jsonl'
        records=[json.loads(line) for line in path.read_text().splitlines()];results=[]
        for r in records:
            i=r['query_id'];prompt=r['prompt_tokens'];s,t=prompt[-3:-1];p=r['trace_tokens']
            edges={(prompt[k+1],prompt[k+2]) for k in range(0,len(prompt)-4,3)}
            valid=bool(p) and p[0]==s and p[-1]==t and len(p)==len(set(p)) and all((u,v) in edges for u,v in zip(p,p[1:])) and all(self.tok.node_lo<=x<self.tok.node_lo+self.tok.n_node_tokens for x in p)
            results.append(dict(query_id=i,label=r['label'],valid_witness=valid,correct=r['correct']))
        positives=[r for r in results if r['label']]
        report=dict(n=len(results),n_positive=len(positives),valid_positive_path_rate=sum(r['valid_witness'] for r in positives)/max(1,len(positives)),
                    correct_positive_witness_rate=sum(r['valid_witness'] and r['correct'] for r in positives)/max(1,len(positives)),records=results)
        (self.out/'eval'/f'semantic_{name}_{view}.json').write_text(json.dumps(report,indent=2))

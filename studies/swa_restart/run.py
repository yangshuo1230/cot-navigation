"""Clean 6/9-layer FULL/SWA-1/SWA-2 study; shared native trainer and scheduler."""
import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
os.chdir(ROOT)
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS','2')
os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR',str(ROOT/'.cache/inductor'))
os.environ.setdefault('TRITON_CACHE_DIR',str(ROOT/'.cache/triton'))
import numpy as np
import torch
import yaml
from cot_navigation.config import load_config
from cot_navigation.experiments import scheduler
from cot_navigation.experiments.fine_node_curriculum import FineNodeTrainer,flat_rows
from cot_navigation.training.trainer import Trainer as NativeTrainer
from cot_navigation.data.dataset import ContextStore
from cot_navigation.data.shards import is_complete,file_sha256
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.protocols.trace_audit import validate_trace
from cot_navigation.models.transformer import build_model

spec=importlib.util.spec_from_file_location('swa_length_policy',ROOT/'studies/correction_reproduction/trace_weight.py')
length_policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(length_policy)
NAME='swa_length_320k_20261103'
GEN=ROOT/'configs/generated'/NAME
OUT=ROOT/'outputs'/NAME
ARMS={'full':0,'swa1':1,'swa2':2}

def source_identity():
    names=[HERE/'run.py',HERE/'base.yaml',Path(length_policy.__file__),
           ROOT/'src/cot_navigation/models/record_attention.py',ROOT/'src/cot_navigation/training/trainer.py']
    return {str(p.relative_to(ROOT)):file_sha256(p)for p in names}

def configure():
    GEN.mkdir(parents=True,exist_ok=True)
    base=yaml.safe_load((HERE/'base.yaml').read_text());jobs=[]
    for seed in [0,1,2]:
        for depth in [6,9]:
            for arm,window in ARMS.items():
                cfg=json.loads(json.dumps(base));name=f'L{depth}/s{seed}/{arm}'
                cfg['output_dir']=str(OUT/name)
                cfg['model'].update(layers=depth,max_seq_len=4096,swa_records=window,
                    swa_backend='flex',compile=window==0)
                cfg['protocol'].update(name='dfs_state',enable_mistake=False,mistake_rate=0,
                    mistake_source='synthetic',trajectory_variants=4)
                cfg['train'].update(max_steps=320000,scale_stage_ends=[80000,160000,240000,320000],
                    model_seed=seed,exposure_seed=0,trace_loss_lambda=1.,
                    eval_samples=256,free_gen_samples=256,eval_interval=20000,free_eval_interval=20000,
                    save_interval=10000,log_interval=100,keep_last=1,
                    max_tokens_per_microbatch=65536,max_padding_fraction=.3,compile_loss=True,
                    persist_outputs=True)
                cfg['evaluation'].update(test_samples=256,batch_size=64,free_batch_size=32,
                    free_max_cache_mb=8192,free_max_prefill_tokens=65536,
                    compile_generation=False,final_checkpoints=['last'],selection_min_step=320001,
                    save_trace_tokens=True,trace_diagnostics=False,mistake_samples=0)
                cfg['tracking'].update(mode='disabled',group=NAME,name=name,variant=arm,
                    notes='Clean record-window comparison; all arms share actual-length normalized loss; fresh initialization; fixed 320k; final last only.')
                cfg['runtime'].update(device='cuda',cuda_device=0)
                path=GEN/(name.replace('/','__')+'.yaml');text=yaml.safe_dump(cfg,sort_keys=False)
                if path.exists():assert path.read_text()==text,f'Immutable config changed: {path}'
                else:path.write_text(text)
                load_config([path],[])
                jobs.append(dict(name=name,config=str(path),depth=depth,seed=seed,arm=arm))
    manifest=dict(jobs=jobs,policy=length_policy.build_policy(),source=source_identity(),
        graph_family='layered',graph_edges='floor(1.5N)',steps=320000,context=4096)
    scheduler.write_json(GEN/'manifest.json',manifest)
    return manifest

def weight_hash(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def prepare(manifest):
    cfg=load_config([manifest['jobs'][0]['config']],[]);tok=Tokenizer.from_config(cfg)
    roots=list(dict.fromkeys(list(cfg.train.scale_roots)+[s.split('=',1)[1]for s in cfg.evaluation.extra_sets]))
    reports=[];inventory={}
    for root in roots:
        path=Path(root);store=ContextStore(path);rows=flat_rows(store,cfg,tok,verify=True);checks=0
        for archive in path.rglob('*.npz'):assert is_complete(archive),archive
        for split in ['train','val','test']:
            ids=store.indices(split)
            for index in ids[np.linspace(0,len(ids)-1,min(8,len(ids)),dtype=int)]:
                index=int(index);nodes=store.node_vocab_ids(index,tok)
                adj={nodes[u]:[nodes[v]for v in vs]for u,vs in enumerate(store.adj(index))}
                src=nodes[int(store.arrays['s_topo'][index])];dst=nodes[int(store.arrays['t_topo'][index])]
                for variant in range(rows.variants(index)):
                    row=rows.row(index,variant);trace=row.tokens[row.prompt_len:row.answer_index]
                    if tok.id('<STOP>')in trace:
                        assert trace[-1]==tok.id('<STOP>')and trace[-2]==tok.id(';')
                        result=validate_trace(trace[:-1],row.label,'dfs_state',adj,src,dst,tok)
                        assert result.get('error')=='missing_or_inconsistent_terminal',result
                    else:assert validate_trace(trace,row.label,'dfs_state',adj,src,dst,tok)['valid']
                    assert len(row.tokens)<=4096;checks+=1
        reports.append(dict(root=root,cache=rows.source_path,sha256=file_sha256(rows.source_path),checked=checks))
        for f in path.rglob('*'):
            if f.is_file():s=f.stat();inventory[str(f)]=[s.st_size,s.st_mtime_ns]
        print('AUDITED',root,checks,flush=True)
        del rows,store
    initial={}
    for job in manifest['jobs']:
        c=load_config([job['config']],[]);torch.manual_seed(c.train.model_seed)
        model=build_model(c.model,len(tok),pad_id=tok.pad);initial[job['name']]=weight_hash(model);del model
    for depth in [6,9]:
        for seed in [0,1,2]:assert len({initial[f'L{depth}/s{seed}/{arm}']for arm in ARMS})==1
        assert len({initial[f'L{depth}/s{seed}/full']for seed in [0,1,2]})==3
    receipt=dict(source=source_identity(),policy=length_policy.build_policy(),datasets=reports,
        inventory=inventory,initialization=initial,paired_initialization_verified=True)
    scheduler.write_json(GEN/'prepared.json',receipt)
    return receipt

def ready(jobs,output,env):
    receipt=json.loads((GEN/'prepared.json').read_text());assert receipt['source']==source_identity()
    assert (GEN/'gpu_verified.json').exists(),'Run --verify-gpu before launching.'
    for name,signature in receipt['inventory'].items():
        s=Path(name).stat();assert [s.st_size,s.st_mtime_ns]==signature,name
    scheduler.write_json(output/'prepared.json',receipt)

def install_objective():
    import cot_navigation.training.trainer as training
    original=training._arch_snapshot
    training._arch_snapshot=lambda cfg,tok,data_hash:{**original(cfg,tok,data_hash),
        'clean_swa_study':1,'trace_weight_policy':length_policy.build_policy()}
    training.combine_losses=length_policy.combine_length_losses

class SwaTrainer(FineNodeTrainer):
    def _render_task(self,index,*,path_index=0):
        row=super()._render_task(index,path_index=path_index)
        if len(self.render_cache)>2048:self.render_cache.pop(next(iter(self.render_cache)))
        return row

    def _update(self,collect_metrics=True):
        if not hasattr(self,'_observed_length_loss'):
            fn=self.loss_fn
            def observed(*args,**kwargs):
                parts=fn(*args,**kwargs)
                if self._collect_length:
                    take=parts['has_answer']
                    self._length_chunks.append(torch.stack([parts['weighted_answer'].detach().sum(),
                        parts['weighted_trace'].detach().sum(),torch.where(take,parts['trace_weights'],0.).sum().detach(),
                        torch.where(take,parts['trace_lengths'],0.).sum().detach(),take.sum()]))
                return parts
            self.loss_fn=observed;self._observed_length_loss=True
        self._collect_length=collect_metrics;self._length_chunks=[]
        result=super()._update(collect_metrics)
        if collect_metrics and self._length_chunks:
            a,t,w,length,n=torch.stack(self._length_chunks).sum(0).cpu().tolist();n=max(1,n)
            result.update(weighted_answer_loss=a/n,weighted_trace_loss=t/n,task_loss=(a+t)/n,
                **{'trace_weight/mean':w/n,'trace_length/mean':length/n})
            result.pop('trace_loss_lambda',None)
        self._length_chunks=[]
        return result

    def _free(self,rows,*,record_name=None):
        report=NativeTrainer._free(self,rows,record_name=record_name)
        if not self.persist:return report
        name=record_name or f'val_{self.completed:08d}'
        records=[json.loads(x)for x in(self.out/'eval'/f'predictions_{name}.jsonl').read_text().splitlines()]
        counts=Counter();errors=Counter()
        for row in records:
            prompt=row['prompt_tokens'];q=prompt.index(self.tok.id('<QUERY>'));adj={self.tok.node_id(i):[]for i in range(256)}
            assert q%3==0 and len(prompt)==q+4
            for i in range(0,q,3):adj[prompt[i+1]].append(prompt[i+2])
            result=validate_trace(row['trace_tokens'],row['label'],'dfs_state',adj,prompt[q+1],prompt[q+2],self.tok)
            counts['n']+=1;counts['positive']+=row['label']==1;counts['valid']+=result['valid']
            counts['posvalid']+=result['valid']and row['label']==1;counts['joint']+=result['valid']and row['correct']
            if not result['valid']:errors[result['error']]+=1
        report['semantic']=dict(n=counts['n'],n_positive=counts['positive'],valid_rate=counts['valid']/counts['n'],
            positive_valid_rate=counts['posvalid']/max(1,counts['positive']),joint_rate=counts['joint']/counts['n'],errors=dict(errors))
        return report

def verify_gpu(manifest):
    from cot_navigation.data.dataset import collate_examples
    cfg=load_config([manifest['jobs'][0]['config']],[]);tok=Tokenizer.from_config(cfg)
    store=ContextStore(cfg.data.root);rows=flat_rows(store,cfg,tok)
    flat=int(np.argmax(np.diff(rows.arrays['offsets'])));query=int(np.searchsorted(rows.arrays['query_offsets'],flat,side='right')-1)
    variant=flat-int(rows.arrays['query_offsets'][query]);row=rows.row(query,variant)
    arrays=collate_examples([row,row],tok.pad,128)
    batch={k:torch.from_numpy(arrays[k]).cuda()for k in ['tokens','answer_mask','trace_mask','walk_mask']}
    checks=[]
    for depth in [6,9]:
        for arm in ARMS:
            job=next(j for j in manifest['jobs']if j['depth']==depth and j['arm']==arm and j['seed']==0)
            c=load_config([job['config']],[]);torch.manual_seed(0);model=build_model(c.model,len(tok),pad_id=tok.pad).cuda().train()
            with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(batch['tokens'])
            loss=length_policy.combine_length_losses(logits,batch['tokens'],batch['answer_mask'],batch['trace_mask'],batch['walk_mask'],2,0)
            loss['total_loss'].backward();assert torch.isfinite(loss['total_loss'])
            assert all(torch.isfinite(p.grad).all()for p in model.parameters()if p.grad is not None)
            checks.append(dict(depth=depth,arm=arm,length=int(batch['tokens'].shape[1]),loss=float(loss['total_loss'].detach())))
            print('GPU_PASS',checks[-1],flush=True);del model,loss,logits;torch.cuda.empty_cache()
    scheduler.write_json(GEN/'gpu_verified.json',dict(source=source_identity(),checks=checks))

@dataclass
class Job(scheduler.Job):
    def command(self,checkpoint=None,parent=None):
        assert parent is None
        args=[sys.executable,str(HERE/'run.py'),'--worker',self.name]
        if checkpoint:args+=['--checkpoint',str(checkpoint)]
        return args

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--verify-gpu',action='store_true')
    p.add_argument('--worker');p.add_argument('--checkpoint');p.add_argument('--run-name',default=NAME)
    p.add_argument('--gpus',type=int,nargs='+',default=[0,1,2,3])
    for flag in ['execute','wait','background','resume']:p.add_argument('--'+flag,action='store_true')
    args=p.parse_args();assert args.run_name==NAME;torch.set_num_threads(2)
    if args.worker:
        manifest=json.loads((GEN/'manifest.json').read_text());job=next(j for j in manifest['jobs']if j['name']==args.worker)
        assert manifest['source']==source_identity();cfg=load_config([job['config']],[]);install_objective()
        with scheduler.file_lock(Path(cfg.output_dir)/'.worker.lock',blocking=False):
            trainer=SwaTrainer(cfg,resume=args.checkpoint)
            if not args.checkpoint:
                actual=weight_hash(trainer.raw)
                expected=json.loads((GEN/'prepared.json').read_text())['initialization'][job['name']]
                assert actual==expected,'Worker initialization differs from the paired CPU audit'
                scheduler.write_json(trainer.out/'initialization.json',dict(weights_sha256=actual))
            trainer.fit();scheduler.write_json(trainer.out/'COMPLETE.json',dict(step=trainer.completed))
        return
    manifest=configure()
    if args.prepare:prepare(manifest);return
    if args.verify_gpu:verify_gpu(manifest);return
    jobs=[Job(j['name'],Path(j['config']),[],OUT/j['name'],320000)for j in manifest['jobs']]
    args.dry_run=not args.execute
    scheduler.execute(args,OUT,jobs,sys.argv[1:],script_path=HERE/'run.py',data_preparer=ready)

if __name__=='__main__':main()

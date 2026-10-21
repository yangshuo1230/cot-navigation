"""Single-world 10x10 fixed-DAG: <=3-hop versus all-hop QUERY training.

No historical modules/data are edited. Both arms share graph, WALK corpus,
Stage-A initialization, tokenizer, optimizer and query quotas. Only the allowed
forward-query span changes. A signed gap is retained separately from distance.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1] if Path(__file__).parent.name=='scripts' else Path.cwd()
sys.path.insert(0,str(ROOT/'src'))
PROTOCOLS=['direct','path','dfs_state','bfs_state','dfs_nodes','bfs_nodes']
DATA=Path('data/fixed_short3_v1/n100_m150_s0')
SOURCE=Path('data/scale_v1/fixed_l10w10_n100_m150_s0/main-seed0')


def write(path,obj):
    from cot_navigation.data.shards import atomic_json
    atomic_json(path,obj)


def prepare_world():
    import numpy as np
    from cot_navigation.data.dataset import ParametricWorld
    from cot_navigation.data.shards import atomic_savez,mark_complete,file_sha256
    dest=ROOT/DATA/'main-seed0';source=ParametricWorld(ROOT/SOURCE)
    identity={'version':1,'source_fingerprint':source.fingerprint,'reverse_split_seed':20261019,
              'reverse_train_max_abs_gap':3,'forward_splits':'unchanged'}
    if (dest/'contract.json').exists():
        assert json.loads((dest/'contract.json').read_text())['identity']==identity
        return dest
    if dest.exists() and any(dest.iterdir()):raise ValueError('Incomplete dataset; inspect before retry: '+str(dest))
    dest.mkdir(parents=True,exist_ok=True)
    for name in ['graph.npz','walks.npz']:shutil.copy2(ROOT/SOURCE/name,dest/name)
    a={k:v.copy() for k,v in source.queries.items()};count=len(a['label'])
    rows=[];rng=np.random.default_rng(20261019)
    # Reverse negatives have explicit abs-span, never a fictitious path length.
    for gap in range(1,10):
        for low in range(10-gap):
            pairs=[(u,v) for u in range((low+gap)*10,(low+gap+1)*10) for v in range(low*10,(low+1)*10)]
            order=rng.permutation(len(pairs))
            for j,k in enumerate(order):
                split=(0 if j<50 else 1 if j<70 else 2) if gap<=3 else 2
                u,v=pairs[int(k)];rows.append((split,gap,0,u,v,-1))
    fields=['split','distance','label','s_topo','t_topo','witness_seen']
    table=np.asarray(rows,dtype=np.int16)
    for j,k in enumerate(fields):a[k]=np.concatenate([a[k],table[:,j].astype(a[k].dtype)])
    a['signed_gap']=(a['t_topo']//10-a['s_topo']//10).astype(np.int16)
    atomic_savez(dest/'queries.npz',**a)
    paths={k:v.copy() for k,v in source.paths.items()}
    paths['query_offsets']=np.r_[paths['query_offsets'],np.full(len(rows),paths['query_offsets'][-1],dtype=paths['query_offsets'].dtype)]
    atomic_savez(dest/'paths.npz',**paths)
    for split in ['train','val','test']:shutil.copy2(ROOT/SOURCE/f'balanced_{split}.npy',dest/f'balanced_{split}.npy')
    info=copy.deepcopy(source.info);info.update(n_queries=len(a['label']),source_world=str(SOURCE),reverse_queries=len(rows))
    write(dest/'world.json',info);mark_complete(dest/'world.json',file_sha256(dest/'world.json'))
    write(ROOT/DATA/'manifest.json',{'kind':'parametric_graph','worlds':[info],'derivation':identity})
    # Independent reachability validation for EVERY query; no model evaluation.
    adj=source.adj();reach=[]
    for u in range(100):
        seen={u};todo=[u]
        for v in todo:
            for w in adj[v]:
                if w not in seen:seen.add(w);todo.append(w)
        reach.append(seen)
    assert all(int(v in reach[int(u)])==int(y) for u,v,y in zip(a['s_topo'],a['t_topo'],a['label']))
    assert np.all(a['signed_gap'][:count]>0) and np.all(a['signed_gap'][count:]<0)
    assert not np.diff(paths['query_offsets'][count:]).any()
    assert len(set(zip(a['s_topo'].tolist(),a['t_topo'].tolist())))==len(a['label'])
    audit={'identity':identity,'queries':len(a['label']),'forward_queries':count,'reverse_queries':len(rows),
           'graph_sha256':file_sha256(dest/'graph.npz'),'walks_sha256':file_sha256(dest/'walks.npz'),
           'buckets':{f'{split}/{direction}/{gap}':int(((a['split']==split)&(np.sign(a['signed_gap'])==direction)&(a['distance']==gap)).sum())
                      for split in [0,1,2] for direction in [-1,1] for gap in range(1,10)}}
    assert audit['graph_sha256']==file_sha256(ROOT/SOURCE/'graph.npz')
    assert audit['walks_sha256']==file_sha256(ROOT/SOURCE/'walks.npz')
    write(dest/'contract.json',audit)
    return dest


class QuotaSampler:
    """Exact 64-query quotas: 32 YES, 24 forward NO, 8 reverse NO."""
    def __init__(self,arrays,ids,batch,seed):
        import numpy as np
        from cot_navigation.training.sampler import ScaleMixtureSampler
        hop=arrays['distance'][ids];y=arrays['label'][ids];gap=arrays['signed_gap'][ids]
        self.groups=[]
        for one in [True,False]:
            h=hop==1 if one else hop>=2
            for label,direction in [(1,1),(0,1),(0,-1)]:
                found=np.flatnonzero(h&(y==label)&(np.sign(gap)==direction))
                if not len(found):raise ValueError('Empty query stratum')
                self.groups.append(found)
        self.delegate=ScaleMixtureSampler([len(g) for g in self.groups],batch,seed,[8,6,2,24,18,6])
        self.last=[]
    def next(self):
        self.last=[int(self.groups[g][i]) for g,i in self.delegate.next()]
        return [(0,i) for i in self.last]
    def state_dict(self):return {'kind':'short_hop_quotas_v1','groups':[g.tolist() for g in self.groups],'delegate':self.delegate.state_dict()}
    def load_state_dict(self,state):
        assert state['kind']=='short_hop_quotas_v1' and state['groups']==[g.tolist() for g in self.groups]
        self.delegate.load_state_dict(state['delegate'])


def trainer_type():
    import numpy as np
    import torch
    from cot_navigation.training.trainer import Trainer
    from cot_navigation.evaluation.sampling import stratified_ids
    class ShortHopTrainer(Trainer):
        def _load_data(self):
            super()._load_data()
            if self.cfg.train.stage=='A':return
            maximum=3 if self.cfg.tracking.variant=='short3' else 9
            for m in self.scale_members:
                a=m.world.queries;ids=m.train_ids;g=a['signed_gap'][ids]
                m.train_ids=ids[((g>0)&(g<=maximum))|((g<0)&(-g<=3))]
            self.train_ids=self.scale_members[0].train_ids
            a=self.world.queries;self.val_ids=self.val_ids[(a['signed_gap'][self.val_ids]>0)&(a['distance'][self.val_ids]<=3)]
        def _init_samplers(self):
            super()._init_samplers()
            self.snapshot.update(short_hop_suite=1,forward_train_max_hops=3 if self.cfg.tracking.variant=='short3' else 9,
                                 reverse_negative_query_fraction=.125,validation='forward_gap_2_3')
            if self.cfg.train.stage!='A':
                m=self.scale_members[0]
                self.query_sampler=QuotaSampler(m.world.queries,m.train_ids,self.cfg.train.effective_query_batch_size,self.cfg.train.exposure_seed+3)
        def _selected_ids(self,split,limit,query_view=None):
            a=self.world.queries;ids=self.world.indices(split,query_view or 'full')
            ids=ids[a['signed_gap'][ids]>0]
            if split=='val':ids=ids[a['distance'][ids]<=3]
            return stratified_ids(ids,np.zeros(len(a['label']),dtype=np.int8),a['distance'],a['label'],limit,self.cfg.evaluation.seed)
        def _update(self,collect_metrics=True):
            out=super()._update(collect_metrics)
            if collect_metrics and self.cfg.train.stage!='A':
                ids=self.scale_members[0].train_ids[self.query_sampler.last];g=self.world.queries['signed_gap'][ids]
                out['query_direction/reverse_fraction']=float((g<0).mean())
                out['query_direction/max_forward_gap']=int(g[g>0].max())
            return out
        def _evaluate_final_checkpoints(self):
            super()._evaluate_final_checkpoints()
            # Reverse negatives are separate diagnostics, not checkpoint selection.
            a=self.world.queries;ids=self.world.indices('test','full')
            latest=torch.load(self.out/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
            try:
                for tag in self.cfg.evaluation.final_checkpoints:
                    blob=torch.load(self.out/'checkpoints'/('latest.pt' if tag=='last' else 'best.pt'),map_location='cpu',weights_only=False)
                    self.raw.load_state_dict(blob['model'])
                    for name,mask in [('reverse_short',(a['signed_gap'][ids]<0)&(a['distance'][ids]<=3)),
                                      ('reverse_long',(a['signed_gap'][ids]<0)&(a['distance'][ids]>=4))]:
                        chosen=stratified_ids(ids[mask],np.zeros(len(a['label']),dtype=np.int8),a['distance'],a['label'],128,self.cfg.evaluation.seed)
                        report=self._free((self._render_task(int(i)) for i in chosen),record_name=f'test_{tag}-{name}')
                        write(self.out/'eval'/f'free_test_{tag}-{name}.json',report)
                        self.tracker.log(self.completed,{f'test/{tag}/{name}/accuracy':report['overall']['answer_accuracy']})
            finally:self.raw.load_state_dict(latest['model'])
    return ShortHopTrainer


def build(output,args):
    import yaml
    configs=output/'configs';configs.mkdir(parents=True,exist_ok=True)
    old=ROOT/'outputs/scale_v1/launches/scale_v1_s0/fixed/l10/stage_b'
    jobs=[]
    for arm in ['knowledge','short3','all_hops']:
        for protocol in (['direct'] if arm=='knowledge' else args.protocols):
            name=arm if arm=='knowledge' else arm+'/'+protocol
            c=yaml.safe_load((old/protocol/'resolved_config.yaml').read_text())
            c['output_dir']=str(output/name);c['data']['root']=str(DATA)
            t=c['train'];t.update(stage='A' if arm=='knowledge' else 'B',max_steps=1000 if arm=='knowledge' else args.steps,
                scale_roots=[str(DATA)],scale_weights=[1.0],scale_stage_ends=[],scale_stage_weights=[],scale_walk_weights=[],
                world_roots=[str(DATA/'main-seed0')],free_gen_samples=128,eval_samples=128,eval_interval=5000,save_interval=5000,
                walk_batch_size=64 if arm=='knowledge' else 16)
            c['tracking'].update(mode=args.tracking,name=f'{args.run_name}/{name}',group=args.run_name,variant=arm,
                tags=['fixed_short3','n100','matched_single_world',arm],notes='Forward query restriction only; reverse negatives 12.5%; CoT/WALK depth not restricted')
            c['evaluation'].update(extra_sets=[],final_checkpoints=[] if arm=='knowledge' else ['best','last'],test_samples=256,trace_diagnostics=False)
            if arm=='knowledge':c['protocol'].update(name='direct',negative_random_path=False,random_search_order=False,trajectory_variants=0)
            f=configs/(name.replace('/','_')+'.yaml');f.write_text(yaml.safe_dump(c,sort_keys=False))
            jobs.append({'name':name,'config':str(f),'steps':t['max_steps']})
    # Run each protocol's arms adjacent to make the first comparison available early.
    jobs=[jobs[0]]+[next(j for j in jobs if j['name']==arm+'/'+p) for p in args.protocols for arm in ['short3','all_hops']]
    path=output/'launch_plan.json'
    if path.exists():assert json.loads(path.read_text())==jobs,'Run plan changed'
    else:write(path,jobs)
    return jobs


def prepare(output,args,jobs):
    import numpy as np
    from cot_navigation.config import load_config
    from cot_navigation.data.dataset import ParametricWorld
    from cot_navigation.protocols.tokenizer import Tokenizer
    from cot_navigation.data.scale_prepared import prepare_variants
    from cot_navigation.data.prepared import _write_rows
    source=ParametricWorld(ROOT/DATA/'main-seed0');receipts=[]
    for job in jobs:
        cfg=load_config([job['config']]);tok=Tokenizer.from_config(cfg)
        if job['name']=='knowledge':_write_rows(source,'walk',tok,0);continue
        prepare_variants(source,cfg,tok,cfg.protocol.name)
        ids=source.indices('train','full');g=source.queries['signed_gap'][ids]
        maximum=3 if cfg.tracking.variant=='short3' else 9
        ids=ids[((g>0)&(g<=maximum))|((g<0)&(-g<=3))]
        sampler=QuotaSampler(source.queries,ids,64,3)
        for _ in range(100):
            selected=ids[[i for _,i in sampler.next()]];gap=source.queries['signed_gap'][selected]
            assert sum(gap<0)==8 and sum(source.queries['label'][selected]==1)==32
            assert sum(source.queries['distance'][selected]==1)==16
            assert gap[gap>0].max()<=maximum and (-gap[gap<0]).max()<=3
        state=sampler.state_dict();expected=sampler.next();sampler.load_state_dict(state);assert sampler.next()==expected
        receipts.append({'job':job['name'],'train_queries':len(ids),'forward_train_queries':int((source.queries['signed_gap'][ids]>0).sum()),
                         'reverse_train_queries':int((source.queries['signed_gap'][ids]<0).sum()),'forward_max_gap':maximum,
                         'reverse_fraction':.125,'source_fingerprint':source.fingerprint,'vocab_hash':tok.hash()})
    write(output/'data_contract.json',receipts)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name',default='short3_n100_s0')
    parser.add_argument('--steps',type=int,default=125000)
    parser.add_argument('--protocols',nargs='+',choices=PROTOCOLS,default=PROTOCOLS)
    parser.add_argument('--tracking',choices=['online','offline','disabled'],default='online')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--run-one')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();os.chdir(ROOT)
    if not args.run_name.replace('_','').replace('-','').isalnum():parser.error('plain run name required')
    output=ROOT/'outputs/fixed_short3_v1/launches'/args.run_name;output.mkdir(parents=True,exist_ok=True)
    if args.run_one:
        from cot_navigation.config import load_config
        cfg=load_config([output/'configs'/(args.run_one.replace('/','_')+'.yaml')])
        checkpoint=Path(cfg.output_dir)/'checkpoints/latest.pt'
        trainer=trainer_type()(cfg,resume=str(checkpoint) if args.resume and checkpoint.exists() else None,
            init_from=None if args.run_one=='knowledge' or (args.resume and checkpoint.exists()) else str(output/'knowledge/checkpoints/latest.pt'))
        try:trainer.fit()
        finally:trainer.tracker.finish()
        return
    prepare_world();jobs=build(output,args);prepare(output,args,jobs)
    if args.prepare_only:print('PREPARED',output,flush=True);return
    import fcntl
    with (output/'.launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for job in jobs:
            import torch
            ckpt=output/job['name']/'checkpoints/latest.pt'
            if ckpt.exists() and torch.load(ckpt,map_location='cpu',weights_only=False)['step']>=job['steps']:
                # Completion includes final test, not just the last optimizer update.
                if (output/job['name']/'COMPLETE.json').exists():continue
            command=[sys.executable,str(Path(__file__).resolve()),'--run-name',args.run_name,'--run-one',job['name'],'--resume']
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','TORCHINDUCTOR_COMPILE_THREADS':'2'}
            print('START',job['name'],flush=True)
            with (output/(job['name'].replace('/','_')+'.log')).open('a') as log:
                result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(f"Job {job['name']} exited {result.returncode}")
            write(output/job['name']/'COMPLETE.json',{'steps':job['steps']})
            print('DONE',job['name'],flush=True)


if __name__=='__main__':main()

"""Fixed-DAG query-block holdout, with unchanged full-graph WALK knowledge.

Layers 0..4 and 5..9 form the blocks. Both arms use forward spans 1..4;
within_only excludes every cross-block query, including reverse negatives.
The matched cross_allowed arm has the same unique query counts per stratum.
No source arrays, existing launchers, or running experiments are modified.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1] if Path(__file__).parent.name=='scripts' else Path.cwd()
sys.path.insert(0,str(ROOT/'src'))
PREVIOUS=Path('outputs/fixed_short3_v1/launches/short3_n100_s0')
ARMS=['within_only','cross_allowed']
PROTOCOLS=['direct','path','dfs_state','bfs_state','dfs_nodes','bfs_nodes']


def task_name(arm,protocol):
    # Preserve the already-running Direct paths and W&B identities.
    return arm if protocol=='direct' else arm+'/'+protocol

# (absolute layer gap, label, direction, number in each 64-query batch)
STRATA=[(1,1,1,8),(1,0,1,6),(1,0,-1,2),
        (2,1,1,8),(2,0,1,5),(2,0,-1,3),
        (3,1,1,8),(3,0,1,5),(3,0,-1,3),
        (4,1,1,8),(4,0,1,8)]


def write(path,obj):
    from cot_navigation.data.shards import atomic_json
    atomic_json(path,obj)


def make_policy(world):
    import numpy as np
    a=world.queries
    cross=(a['s_topo']//50)!=(a['t_topo']//50)
    walk=np.load(world.root/'walks.npz')
    seen=set();edges=set();crosswalks=0
    for lo,hi in zip(walk['offset'][:-1],walk['offset'][1:]):
        ns=walk['nodes'][lo:hi].tolist()
        seen.update((u,v) for i,u in enumerate(ns) for v in ns[i+1:])
        edges.update(zip(ns,ns[1:]))
        crosswalks+=int(any(u//50!=v//50 for u,v in zip(ns,ns[1:])))
    witnessed=np.array([(int(u),int(v)) in seen for u,v in zip(a['s_topo'],a['t_topo'])])
    # These pairs are reserved from BOTH arms, even if the source marked them train.
    novel=np.flatnonzero(cross&(a['signed_gap']>0)&(a['distance']<=4)&(a['label']==1)&~witnessed)
    reserved=set(novel.tolist());original=world.indices('train','full')
    pool=np.array([q for q in original if int(q) not in reserved],dtype=np.int64)
    chosen={arm:[] for arm in ARMS};counts=[]
    rng=np.random.default_rng(20261019)
    for gap,label,direction,quota in STRATA:
        mask=(a['distance'][pool]==gap)&(a['label'][pool]==label)&(np.sign(a['signed_gap'][pool])==direction)
        candidates=pool[mask];within=candidates[~cross[candidates]]
        assert len(within)>0
        allowed=rng.choice(candidates,size=len(within),replace=False)
        chosen['within_only'].extend(within.tolist());chosen['cross_allowed'].extend(allowed.tolist())
        counts.append(dict(gap=gap,label=label,direction=direction,quota=quota,
                           unique_queries_each_arm=len(within),allowed_cross_queries=int(cross[allowed].sum())))
    train={k:sorted(v) for k,v in chosen.items()}
    val=world.indices('val','full')
    val=val[(a['signed_gap'][val]>0)&(a['distance'][val]<=4)&~cross[val]]
    test=world.indices('test','full');forward=a['signed_gap']>0;short=(a['distance']>=2)&(a['distance']<=4)
    views={'within_short':test[forward[test]&short[test]&~cross[test]].tolist(),
           'cross_short':test[forward[test]&short[test]&cross[test]].tolist(),
           'cross_long':test[forward[test]&(a['distance'][test]>=5)&cross[test]].tolist(),
           'cross_unwitnessed_yes':novel.tolist(),
           'reverse_cross':test[(a['signed_gap'][test]<0)&cross[test]].tolist()}
    # Boundary-layer nodes can have a new QUERY role, even at the same span.
    # Report the subset whose source/target roles have positive training examples
    # in BOTH arms, so a new endpoint role does not explain every failure.
    role_sources=[];role_targets=[]
    for arm in ARMS:
        positives=[q for q in train[arm] if a['label'][q]==1]
        role_sources.append(set(a['s_topo'][positives].tolist()))
        role_targets.append(set(a['t_topo'][positives].tolist()))
    src_roles=set.intersection(*role_sources);dst_roles=set.intersection(*role_targets)
    role_covered=lambda q:int(a['s_topo'][q]) in src_roles and int(a['t_topo'][q]) in dst_roles
    views['cross_short_role_covered']=[q for q in views['cross_short'] if role_covered(q)]
    views['cross_unwitnessed_role_covered_yes']=[q for q in novel.tolist() if role_covered(q)]
    views['test']=sorted(views['within_short']+views['cross_short'])
    for arm,ids in train.items():
        assert len(ids)==len(set(ids))
        assert not set(ids)&set(val.tolist())
        assert not any(set(ids)&set(v) for v in views.values())
        if arm=='within_only':assert not cross[ids].any()
        else:assert cross[ids].any()
    actual={(u,v) for u,vs in enumerate(world.adj()) for v in vs}
    assert world.n_nodes==100 and all(v//10==u//10+1 for u,v in actual)
    assert edges==actual and crosswalks>0
    boundary={e for e in actual if e[0]//50!=e[1]//50}
    assert boundary and boundary<=edges
    return dict(version=1,blocks=[[0,4],[5,9]],train_ids=train,val_ids=val.tolist(),test_views=views,
        strata=counts,source_fingerprint=world.fingerprint,
        knowledge=dict(walks=world.n_walks(),crossing_walks=crosswalks,all_edges=len(actual),
            crossing_edges=len(boundary),all_edges_covered=True,max_walk_edges=int(np.diff(walk['offset']).max()-1),
            supplements=int(walk['is_supplement'].sum())),
        test_witness_counts={name:dict(n=len(ids),yes=int(a['label'][ids].sum()),
            yes_walk_seen=int((witnessed[ids]&(a['label'][ids]==1)).sum())) for name,ids in views.items()})


class BlockSampler:
    def __init__(self,arrays,ids,seed):
        import numpy as np
        from cot_navigation.training.sampler import ScaleMixtureSampler
        self.groups=[]
        for gap,label,direction,quota in STRATA:
            good=(arrays['distance'][ids]==gap)&(arrays['label'][ids]==label)&(np.sign(arrays['signed_gap'][ids])==direction)
            self.groups.append(np.flatnonzero(good))
        self.delegate=ScaleMixtureSampler([len(g) for g in self.groups],64,seed,[s[3] for s in STRATA])
        self.last=[]
    def next(self):
        self.last=[int(self.groups[g][i]) for g,i in self.delegate.next()]
        return [(0,i) for i in self.last]
    def state_dict(self):return dict(kind='block_query_v1',groups=[g.tolist() for g in self.groups],delegate=self.delegate.state_dict())
    def load_state_dict(self,state):
        assert state['kind']=='block_query_v1' and state['groups']==[g.tolist() for g in self.groups]
        self.delegate.load_state_dict(state['delegate'])


def trainer_type(policy):
    import numpy as np
    from cot_navigation.training.trainer import Trainer
    from cot_navigation.evaluation.sampling import stratified_ids
    digest=hashlib.sha256(json.dumps(policy,sort_keys=True).encode()).hexdigest()
    class BlockTrainer(Trainer):
        def _load_data(self):
            super()._load_data()
            assert self.cfg.train.stage=='B' and len(self.scale_members)==1
            assert self.world.fingerprint==policy['source_fingerprint']
            self.scale_members[0].train_ids=np.array(policy['train_ids'][self.cfg.tracking.variant],dtype=np.int64)
            self.train_ids=self.scale_members[0].train_ids
            self.val_ids=np.array(policy['val_ids'],dtype=np.int64)
        def _init_samplers(self):
            super()._init_samplers()
            self.snapshot.update(query_block_suite=1,query_policy_sha256=digest,query_block_arm=self.cfg.tracking.variant)
            self.query_sampler=BlockSampler(self.world.queries,self.train_ids,self.cfg.train.exposure_seed+3)
        def _load_resume(self,path):
            import torch
            arch=torch.load(path,map_location='cpu',weights_only=False)['arch']
            assert arch['query_policy_sha256']==digest and arch['query_block_arm']==self.cfg.tracking.variant
            super()._load_resume(path)
        def _selected_ids(self,split,limit,query_view=None):
            a=self.world.queries
            ids=np.array(policy['val_ids'] if split=='val' else policy['test_views'][query_view or 'test'],dtype=np.int64)
            return stratified_ids(ids,np.zeros(len(a['label']),dtype=np.int8),a['distance'],a['label'],limit,self.cfg.evaluation.seed)
        def _update(self,collect_metrics=True):
            out=super()._update(collect_metrics)
            if collect_metrics:
                ids=self.train_ids[self.query_sampler.last];a=self.world.queries
                out['query_blocks/cross_fraction']=float(((a['s_topo'][ids]//50)!=(a['t_topo'][ids]//50)).mean())
                out['query_blocks/reverse_fraction']=float((a['signed_gap'][ids]<0).mean())
            return out
        def _evaluate_final_checkpoints(self):
            import torch
            super()._evaluate_final_checkpoints()
            try:
                for tag in self.cfg.evaluation.final_checkpoints:
                    blob=torch.load(self.out/'checkpoints'/('latest.pt' if tag=='last' else 'best.pt'),map_location='cpu',weights_only=False)
                    self.raw.load_state_dict(blob['model'])
                    for view in policy['test_views']:
                        if view=='test':continue
                        self.evaluate_split('test',query_view=view,checkpoint_tag=f'{tag}-{view}',checkpoint_step=int(blob['step']))
            finally:
                blob=torch.load(self.out/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
                self.raw.load_state_dict(blob['model'])
    return BlockTrainer


def prepare(output,protocols=PROTOCOLS):
    import yaml
    import numpy as np
    from cot_navigation.config import load_config
    from cot_navigation.data.dataset import ParametricWorld
    from cot_navigation.data.scale_prepared import ScalePreparedRows
    from cot_navigation.protocols.tokenizer import Tokenizer
    source=yaml.safe_load((ROOT/PREVIOUS/'configs/short3_direct.yaml').read_text())
    world=ParametricWorld(ROOT/source['data']['root']/'main-seed0')
    policy=make_policy(world)
    path=output/'query_policy.json'
    if path.exists():assert json.loads(path.read_text())==policy
    else:write(path,policy)
    (output/'configs').mkdir(exist_ok=True)
    # Sampler checks are protocol independent; never draw the training RNG here.
    for arm in ARMS:
        ids=np.array(policy['train_ids'][arm],dtype=np.int64);sampler=BlockSampler(world.queries,ids,3)
        for _ in range(100):
            selected=ids[[i for _,i in sampler.next()]];a=world.queries
            assert sum(a['label'][selected])==32 and sum(a['signed_gap'][selected]<0)==8
            assert np.array_equal(np.bincount(a['distance'][selected],minlength=5)[1:5],[16]*4)
            if arm=='within_only':assert np.all(a['s_topo'][selected]//50==a['t_topo'][selected]//50)
        state=sampler.state_dict();expected=sampler.next();sampler.load_state_dict(state);assert sampler.next()==expected
    for protocol in protocols:
        template=yaml.safe_load((ROOT/PREVIOUS/'configs'/f'short3_{protocol}.yaml').read_text())
        exposures={}
        for arm in ARMS:
            name=task_name(arm,protocol)
            c=copy.deepcopy(template);c['output_dir']=str(output/name)
            c['tracking'].update(name=f'block_n100_s0/{name}',group='block_n100_s0',variant=arm,
                tags=['query_block_holdout','n100',protocol,arm],notes='Full WALK; two 5-layer blocks; matched per-gap/label/direction counts; validation within blocks only')
            c['evaluation'].update(test_samples=10000)
            cfgpath=output/'configs'/(name.replace('/','_')+'.yaml')
            if cfgpath.exists():assert yaml.safe_load(cfgpath.read_text())==c
            else:cfgpath.write_text(yaml.safe_dump(c,sort_keys=False))
            cfg=load_config([cfgpath]);tok=Tokenizer.from_config(cfg)
            cache=ScalePreparedRows(world,cfg,tok).arrays
            ids=np.array(policy['train_ids'][arm],dtype=np.int64)
            block=np.full(len(tok),-1,dtype=np.int8)
            block[world.node_vocab_ids(tok)]=np.arange(100)//50
            novel_by_root={};novel_shown=set();node_tokens=world.node_vocab_ids(tok)
            for query in policy['test_views']['cross_unwitnessed_yes']:
                novel_by_root.setdefault(int(world.queries['s_topo'][query]),[]).append(
                    (query,node_tokens[int(world.queries['t_topo'][query])]))
            # A within-block QUERY may have a legal full-graph search that visits
            # the other block. Quantify that separate source of training exposure.
            weighted={0:0.,1:0.};mass={0:0.,1:0.}
            for gap,label,direction,quota in STRATA:
                mask=(world.queries['distance'][ids]==gap)&(world.queries['label'][ids]==label)&(np.sign(world.queries['signed_gap'][ids])==direction)
                group=ids[mask]
                for q in group:
                    lo,hi=cache['query_offsets'][q:q+2]
                    weight=quota/64/len(group)/(hi-lo)
                    for j in range(lo,hi):
                        prompt,answer,_,_=cache['fields'][j];offset=cache['offsets'][j]
                        body=cache['tokens'][offset+prompt:offset+answer]
                        visited_blocks=block[body]
                        crosses=bool(np.any(visited_blocks==0) and np.any(visited_blocks==1))
                        weighted[label]+=weight*crosses;mass[label]+=weight
                        # Any valid search visit is reachable from its query root.
                        # This is a lower bound on information disclosed by CoT:
                        # other traces can also expose intermediate path segments.
                        for reserved,target_token in novel_by_root.get(int(world.queries['s_topo'][q]),[]):
                            if target_token in body:novel_shown.add(reserved)
            exposures[arm]={'yes_trace_spans_both_blocks':weighted[1]/mass[1],
                            'no_trace_spans_both_blocks':weighted[0]/mass[0],
                            'overall_trace_spans_both_blocks':sum(weighted.values()),
                            'reserved_yes_count':len(policy['test_views']['cross_unwitnessed_yes']),
                            'reserved_yes_shown_as_root_to_visited':sorted(novel_shown),
                            'note':'Root-to-visited exposure is a lower bound; QUERY and WALK holdout alone does not exclude CoT information exposure.'}
        write(output/f'preparation_{protocol}.json',dict(status='PASS',protocol=protocol,
            query_policy_unchanged=True,prepared_cache_exists=True,trace_block_exposure=exposures))
    write(output/'preparation_checks.json',dict(status='PASS',no_query_overlap=True,full_walk_edge_coverage=True,
        exact_batch_quotas=True,sampler_resume=True,unique_train_queries={k:len(v) for k,v in policy['train_ids'].items()},
        test_witness_counts=policy['test_witness_counts'],knowledge=policy['knowledge']))
    return policy


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-one',choices=[task_name(a,p) for p in PROTOCOLS for a in ARMS])
    parser.add_argument('--protocols',nargs='+',choices=PROTOCOLS,default=PROTOCOLS)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();os.chdir(ROOT)
    output=ROOT/'outputs/block_query_v1/launches/block_n100_s0';output.mkdir(parents=True,exist_ok=True)
    if args.run_one:
        from cot_navigation.config import load_config
        policy=json.loads((output/'query_policy.json').read_text())
        cfg=load_config([output/'configs'/(args.run_one.replace('/','_')+'.yaml')])
        latest=Path(cfg.output_dir)/'checkpoints/latest.pt'
        trainer=trainer_type(policy)(cfg,resume=str(latest) if latest.exists() else None,
            init_from=None if latest.exists() else str(ROOT/PREVIOUS/'knowledge/checkpoints/latest.pt'))
        try:trainer.fit()
        finally:trainer.tracker.finish()
        return
    import fcntl
    with (output/'.launch.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepare(output,args.protocols);print('PREPARED',output,flush=True)
        if args.prepare_only:return
        # Retain the existing queue's lock while using GPU 0; no preemption.
        with (ROOT/PREVIOUS/'.launch.lock').open('a') as gpu_queue:
            print('WAITING for existing GPU 0 queue',flush=True)
            fcntl.flock(gpu_queue,fcntl.LOCK_EX)
            print('ACQUIRED GPU 0 queue',flush=True)
            for name in [task_name(a,p) for p in args.protocols for a in ARMS]:
                if (output/name/'COMPLETE.json').exists():continue
                env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','TORCHINDUCTOR_COMPILE_THREADS':'2'}
                print('START',name,flush=True)
                with (output/(name.replace('/','_')+'.log')).open('a') as log:
                    subprocess.run([sys.executable,str(Path(__file__).resolve()),'--run-one',name],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                write(output/name/'COMPLETE.json',dict(steps=125000))
                print('DONE',name,flush=True)


if __name__=='__main__':main()

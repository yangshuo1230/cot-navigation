"""15 runs, 30 GPU hours, no model changes or CoT."""
from pathlib import Path
GROUP='direct_multihop_20261021_v1'
BASE=Path(str(next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'pyproject.toml').is_file() and (_p / 'studies').is_dir())) + '/studies/direct_learning/runs/direct_learning_20261020_v1')


def matrix():
    parents=[('target_best','curr_target_gate_s1','best',1),
             ('mix_source_last','curr_mix_source_gate_s0','latest',0),
             ('timed_source_last','curr_source_timed_s0','latest',0)]
    jobs=[]
    def add(name,parent,seed,scheme,lr):
        weights=[[1/3,1/3,1/3]] if scheme=='joint' else [[.25,.75,0.],[.25,.25,.5]]
        jobs.append({'name':name,'seed':seed,'parent_checkpoint':str(parent) if parent else None,
                     'scheme':scheme,'lr':lr,'budget_seconds':7200,
                     'stages':[{'hops':[1,2,3],'weights':w} for w in weights],
                     'group':GROUP,'optimizer_initialization':'fresh AdamW for all parents and scratch',
                     'batch_size':64,'pure_direct':True})
    for short,run,tag,seed in parents:
        path=BASE/run/'checkpoints'/(tag+'.pt')
        for scheme in ('joint','timed','gated'):
            add(short+'_'+scheme,path,seed,scheme,1e-4)
        add(short+'_joint_lr3e4',path,seed,'joint',3e-4)
    add('scratch_joint',None,0,'joint',1e-4)
    add('scratch_timed',None,0,'timed',1e-4)
    add('scratch_joint_lr3e4',None,0,'joint',3e-4)
    assert len(jobs)==15 and sum(j['budget_seconds'] for j in jobs)==30*3600
    return jobs

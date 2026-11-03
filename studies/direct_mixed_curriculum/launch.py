"""Three independent jobs on explicitly idle GPUs; never use GPU0."""
from pathlib import Path
import fcntl
import json
import os
import subprocess
import sys
import time
from data import prepare, atomic_json

ROOT=Path(str(next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'pyproject.toml').is_file() and (_p / 'studies').is_dir())))
STUDY=ROOT/'studies/direct_mixed_curriculum'
RUNS=STUDY/'runs/direct_mixed_curriculum_20261021_v1'

def stage(kind='full',vocab=16,edges=6,shuffle=True,mixture=None):
    return dict(kind=kind,vocab=vocab,edges=edges,shuffle=shuffle,mixture=mixture)

def main():
    os.chdir(ROOT);RUNS.mkdir(parents=True,exist_ok=True)
    lock=open(RUNS/'launcher.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tail=[stage(vocab=64),stage(vocab=256)]+[stage(vocab=256,edges=m) for m in (8,16,24)]
    configs=[('target_gated',1,'gated',[stage('fixed_target'),stage(shuffle=False),stage()]+tail,1),
             ('source_timed',0,'timed',[stage('fixed_source'),stage(shuffle=False),stage()]+tail,6),
             ('mix_source_gated',0,'gated',[stage('mix_source',mixture=p) for p in (.8,.5,.2)]+[stage()]+tail,7)]
    specs=[]
    for name,seed,schedule,stages,gpu in configs:
        out=RUNS/name;out.mkdir(exist_ok=True);(out/'eval').mkdir(exist_ok=True)
        spec=dict(name=name,seed=seed,schedule=schedule,stages=stages,lr=3e-4,packed=False,
                  budget_seconds=14400,group='direct_mixed_curriculum_20261021_v1',
                  data_dir=str(STUDY/'data_v1'),gpu=gpu,parent_checkpoint=None,
                  mixed_hops_from_start=[1,2,3])
        dest=out/'experiment.json'
        if dest.exists(): assert json.loads(dest.read_text())==spec
        else: atomic_json(dest,spec)
        specs.append((spec,dest))
    atomic_json(RUNS/'plan.json',{'jobs':[s for s,_ in specs],'allocated_gpu_hours':12})
    print('Preparing and auditing paired data before GPU training',flush=True)
    prepare(STUDY/'data_v1')
    print('DATA_READY',flush=True)
    running=[]
    for spec,path in specs:
        out=path.parent
        if (out/'completed.json').exists(): continue
        query=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        usage={int(x.split(',')[0]):[int(v.strip()) for v in x.split(',')[1:]] for x in query.splitlines()}
        gpu=spec['gpu'];memory,util=usage[gpu]
        if memory>500 or util>10: raise RuntimeError(f'GPU {gpu} is busy; refusing to share it')
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
        env['PYTHONPATH']=str(ROOT/'src')+os.pathsep+env.get('PYTHONPATH','')
        log=open(out/'train.log','a')
        process=subprocess.Popen([sys.executable,'-u',str(STUDY/'train.py'),'--spec',str(path)],env=env,stdout=log,stderr=subprocess.STDOUT)
        running.append((spec['name'],process,log))
        atomic_json(out/'process.json',{'pid':process.pid,'gpu':gpu,'started_unix':time.time()})
        print('LAUNCHED',spec['name'],'GPU',gpu,'PID',process.pid,flush=True)
    failures=[]
    for name,process,log in running:
        code=process.wait();log.close()
        if code: failures.append({'name':name,'exit_code':code})
        print('EXIT',name,code,flush=True)
    atomic_json(RUNS/'finished.json',{'failures':failures,'finished_unix':time.time()})
    if failures: raise RuntimeError(failures)

if __name__=='__main__': main()

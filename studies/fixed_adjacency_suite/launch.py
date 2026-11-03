"""Six finite workers plus two reserved event-driven workers. Default prints plan."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent


def idle_gpus():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
                                  '--format=csv,noheader,nounits'],text=True)
    return {int(parts[0]) for line in rows.splitlines()
            if (parts:=line.split(',')) and int(parts[1])<256 and int(parts[2])<5}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    repo=args.repo_root.resolve()
    manifest=json.loads((HERE/'manifest.json').read_text())
    jobs=manifest['jobs']
    if not args.execute:
        print(json.dumps(dict(fixed_tasks=36,knowledge_dependencies=6,grokking_runs=2,
             fixed_gpu_pool=list(range(6)),grokking_reserved=[6,7],
             fixed_steps=125000,grokking_step_cap=None,training_started=False),indent=2))
        return
    output=repo/'outputs/exposure_grokking_adj_v3'
    output.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(output/'.scheduler.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    active={};completed=set();failed=[];files=[]
    for job in jobs:
        if (output/job['id']/'COMPLETE.json').is_file():completed.add(job['id'])
    try:
        while len(completed)<len(jobs):
            for gpu,(job,proc) in list(active.items()):
                code=proc.poll()
                if code is None:continue
                del active[gpu]
                if code==0 and (output/job['id']/'COMPLETE.json').exists():
                    completed.add(job['id'])
                else:failed.append(dict(id=job['id'],returncode=code))
            status=dict(completed=sorted(completed),failed=failed,
                active={str(g):dict(id=j['id'],pid=proc.pid) for g,(j,proc) in active.items()},
                note='GPU6/7 have no step cap; finite queue never uses their slots')
            temp=output/'status.writing'
            temp.write_text(json.dumps(status,indent=2));temp.replace(output/'status.json')
            if failed:
                raise RuntimeError('A job failed. No new jobs dispatched; running processes remain untouched.')
            busy={job['id'] for job,proc in active.values()}
            free=idle_gpus()-set(active)
            for gpu in sorted(free):
                candidate=next((j for j in jobs if j['id'] not in completed|busy
                    and gpu in j['gpu_pool'] and set(j['dependencies'])<=completed),None)
                if candidate is None:continue
                log=output/(candidate['id'].replace('/','__')+'.log')
                stream=log.open('a');files.append(stream)
                command=[sys.executable,str(HERE/'train_suite.py'),'--repo-root',str(repo),
                         '--gpu',str(gpu),'--job',candidate['id'],'--execute']
                if (output/candidate['id']/'latest.pt').exists():command.append('--resume')
                elif (output/candidate['id']).exists():
                    raise RuntimeError('Existing incomplete directory without checkpoint; inspect before retry')
                env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
                     'TORCHINDUCTOR_COMPILE_THREADS':'2'}
                proc=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,cwd=repo,env=env)
                active[gpu]=(candidate,proc);busy.add(candidate['id'])
                print(json.dumps(dict(start=candidate['id'],gpu=gpu,pid=proc.pid)),flush=True)
            if len(completed)<len(jobs):time.sleep(20)
    finally:
        for stream in files:stream.close()
        lock.close()


if __name__=='__main__':main()

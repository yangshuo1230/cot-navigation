"""Detached 39-job queue over verified-idle GPUs; fails closed on worker error."""
from pathlib import Path
import os,sys,json,subprocess,time,fcntl
ROOT=Path(str(next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'pyproject.toml').is_file() and (_p / 'studies').is_dir())));HERE=ROOT/'studies/incontext_prefix_suite';OUT=ROOT/'outputs/incontext_prefix_v1'
def main():
    OUT.mkdir(parents=True,exist_ok=True);lock=(OUT/'.scheduler.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=json.loads((HERE/'manifest.json').read_text())['jobs'];active={};done=set();failed=[];files=[]
    for j in jobs:
        if (OUT/j['id']/'COMPLETE.json').exists():done.add(j['id'])
    while len(done)<len(jobs):
        for g,(j,p) in list(active.items()):
            code=p.poll()
            if code is None:continue
            del active[g]
            if code==0 and (OUT/j['id']/'COMPLETE.json').exists():done.add(j['id'])
            else:failed.append(dict(id=j['id'],pid=p.pid,returncode=code))
        status=dict(completed=sorted(done),failed=failed,active={str(g):dict(id=j['id'],pid=p.pid) for g,(j,p) in active.items()},total=len(jobs))
        temp=OUT/'status.writing';temp.write_text(json.dumps(status,indent=2));temp.replace(OUT/'status.json')
        if failed:raise RuntimeError('Dispatch stopped after worker failure; other workers remain untouched')
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        free={int(p[0]) for line in raw.splitlines() if(p:=line.split(',')) and int(p[1])<256 and int(p[2])<5}-set(active)
        busy={j['id'] for j,p in active.values()}
        for gpu in sorted(free):
            job=next((j for j in jobs if j['id'] not in done|busy),None)
            if job is None:break
            stream=(OUT/(job['id'].replace('/','__')+'.log')).open('a');files.append(stream)
            cmd=[sys.executable,str(HERE/'run.py'),'--job',job['id'],'--gpu',str(gpu)]
            if (OUT/job['id']/'checkpoints/latest.pt').exists():cmd.append('--resume')
            env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','TORCHINDUCTOR_COMPILE_THREADS':'2'}
            proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,env=env)
            active[gpu]=(job,proc);busy.add(job['id']);print(json.dumps(dict(start=job['id'],gpu=gpu,pid=proc.pid)),flush=True)
        if len(done)<len(jobs):time.sleep(15)
    (OUT/'COMPLETE.json').write_text(json.dumps(dict(jobs=len(done))))
if __name__=='__main__':main()

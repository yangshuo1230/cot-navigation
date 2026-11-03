"""Queue twelve finite jobs exclusively on GPUs 0-5."""
from pathlib import Path
import subprocess,json,os,sys,time,fcntl
ROOT=Path(str(next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'pyproject.toml').is_file() and (_p / 'studies').is_dir())))
HERE=ROOT/'studies/stepwise_reference'
OUT=ROOT/'outputs/stepwise_reference'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'.scheduler.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=json.loads((HERE/'manifest.json').read_text())['jobs'];active={};completed=set();failed=[];streams=[]
    for j in jobs:
        if (OUT/j['id']/'COMPLETE.json').exists():completed.add(j['id'])
    while len(completed)<len(jobs):
        for g,(j,p) in list(active.items()):
            code=p.poll()
            if code is None:continue
            del active[g]
            if code==0 and (OUT/j['id']/'COMPLETE.json').exists():completed.add(j['id'])
            else:failed.append(dict(id=j['id'],returncode=code))
        status=dict(completed=sorted(completed),failed=failed,active={str(g):dict(id=j['id'],pid=p.pid) for g,(j,p) in active.items()},reserved_gpus=[6,7])
        temp=OUT/'status.writing';temp.write_text(json.dumps(status,indent=2));temp.replace(OUT/'status.json')
        if failed:raise RuntimeError('Worker failed; dispatch stopped, other running jobs untouched')
        text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        idle={int(p[0]) for l in text.splitlines() if (p:=l.split(',')) and int(p[1])<256 and int(p[2])<5}
        busy={j['id'] for j,p in active.values()}
        for gpu in sorted(idle-set(active)-{6,7}):
            j=next((j for j in jobs if j['id'] not in completed|busy),None)
            if not j:break
            f=(OUT/(j['id'].replace('/','__')+'.log')).open('a');streams.append(f)
            cmd=[sys.executable,str(HERE/'train.py'),'--job',j['id'],'--gpu',str(gpu),'--repo-root',str(ROOT)]
            env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','TORCHINDUCTOR_COMPILE_THREADS':'2'}
            p=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,env=env)
            active[gpu]=(j,p);busy.add(j['id']);print(json.dumps(dict(start=j['id'],gpu=gpu,pid=p.pid)),flush=True)
        if len(completed)<len(jobs):time.sleep(15)
    (OUT/'status.json').write_text(json.dumps(dict(completed=sorted(completed),failed=failed,active={}),indent=2))
if __name__=='__main__':main()

"""Run four frozen-inference workers on idle GPUs, preserving GPU0."""
from pathlib import Path
import fcntl,json,os,subprocess,sys,time
ROOT=Path(str(next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'pyproject.toml').is_file() and (_p / 'studies').is_dir())))
STUDY=ROOT/'studies/terminal_action_diagnostic'
STUDY.mkdir(exist_ok=True)
lock=open(STUDY/'launcher.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
processes=[]
for worker,gpu in enumerate((1,4,5,7)):
    raw=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    mem,util=map(int,raw.strip().split(','))
    if mem>500 or util>10:raise RuntimeError(f'GPU{gpu} busy: {raw}')
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
    log=open(STUDY/f'worker{worker}.log','a')
    p=subprocess.Popen([sys.executable,'-u',str(STUDY/'diagnose.py'),'--worker',str(worker),'--workers','4','--batch','4'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    processes.append((worker,gpu,p,log));print('START',worker,gpu,p.pid,flush=True)
(STUDY/'processes.json').write_text(json.dumps([{'worker':w,'gpu':g,'pid':p.pid} for w,g,p,l in processes],indent=2))
codes=[]
for w,g,p,log in processes:
    code=p.wait();log.close();codes.append({'worker':w,'gpu':g,'exit_code':code});print('EXIT',w,code,flush=True)
(STUDY/'finished.json').write_text(json.dumps({'at_unix':time.time(),'workers':codes},indent=2))
if any(x['exit_code'] for x in codes):sys.exit(1)

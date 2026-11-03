from pathlib import Path
import os,subprocess,json
root=Path(str(next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'pyproject.toml').is_file() and (_p / 'studies').is_dir())));out=root/'studies/scale_trace_audit'
procs=[]
for worker,gpu in enumerate(range(1,8)):
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),WANDB_MODE='disabled',OMP_NUM_THREADS='2',PYTHONPATH=str(root/'src'))
 log=(out/f'worker_{worker}.log').open('a')
 p=subprocess.Popen([str(root/'.venv/bin/python'),str(out/'audit.py'),'--worker',str(worker)],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
 log.close();procs.append(dict(worker=worker,gpu=gpu,pid=p.pid))
(out/'launch.json').write_text(json.dumps(procs,indent=2));print(json.dumps(procs))

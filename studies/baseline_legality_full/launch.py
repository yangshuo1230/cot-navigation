"""Run eight bounded frozen-inference workers; never train/save checkpoints."""
from pathlib import Path
import subprocess,json,os,sys
F=Path(__file__).parent;ROOT=F.parents[1]
tasks=[(0,'dfs_state',['main']),(1,'bfs_state',['main']),
       (2,'dfs_state',['larger']),(3,'bfs_state',['larger']),
       (4,'dfs_state',['deep','wide']),(5,'bfs_state',['deep','wide']),
       (6,'dfs_state',['n16','n24','n32','n36','n64']),(7,'bfs_state',['n16','n24','n32','n36','n64'])]
processes=[]
for gpu,protocol,cases in tasks:
 occupied=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
 assert not occupied,f'GPU {gpu} is occupied: {occupied}'
 with (F/f'gpu{gpu}.log').open('a') as log:
  proc=subprocess.Popen([sys.executable,'-u',str(F/'worker.py'),'--gpu',str(gpu),'--protocol',protocol,'--cases',*cases],cwd=ROOT,
    env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','TORCHINDUCTOR_COMPILE_THREADS':'2'},stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
 processes.append(dict(gpu=gpu,pid=proc.pid,protocol=protocol,cases=cases))
(F/'processes.json').write_text(json.dumps(processes,indent=2)+'\n');print(json.dumps(processes,indent=2))

"""Redistribute remaining frozen BFS cases after committed batch boundaries."""
from pathlib import Path
import json,os,signal,time,subprocess,sys
F=Path(__file__).parent;ROOT=F.parents[1]
original=next(x['pid'] for x in json.loads((F/'processes.json').read_text()) if x['gpu']==7)
while not (F/'baseline/bfs_state/wide/COMPLETE.json').exists():time.sleep(2)
workers=[]
for gpu,batch in [(0,0),(2,1),(4,2),(6,3)]:
 while subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():time.sleep(2)
 with (F/f'finish_gpu{gpu}.log').open('a') as log:
  proc=subprocess.Popen([sys.executable,'-u',str(F/'worker.py'),'--gpu',str(gpu),'--protocol','bfs_state','--cases','n36','n64','--batch-indices',str(batch)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL,env={**os.environ,'OMP_NUM_THREADS':'2','TORCHINDUCTOR_COMPILE_THREADS':'2'})
 workers.append(dict(gpu=gpu,pid=proc.pid,batch=batch))
(F/'finish_processes.json').write_text(json.dumps(workers,indent=2)+'\n');print(workers,flush=True)

# Stop only our original evaluator after all N32 results are committed; its
# pending N36/N64 cases are now owned by the four workers above.
while not (F/'baseline/bfs_state/n32/COMPLETE.json').exists():time.sleep(.5)
try:
 cmd=Path(f'/proc/{original}/cmdline').read_bytes()
 if b'worker.py' in cmd and b'bfs_state' in cmd:
  os.kill(original,signal.SIGTERM);print('REDISTRIBUTED after N32',original,flush=True)
except FileNotFoundError:pass

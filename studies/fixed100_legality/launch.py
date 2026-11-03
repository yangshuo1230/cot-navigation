"""Use currently idle GPUs only; never share a card with ongoing training."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT=Path.cwd();OUT=ROOT/'studies/fixed100_legality'
sys.path.insert(0,str(ROOT/'src'))
from cot_navigation.runtime.launch import idle_gpus

jobs=[(s,p) for p in ('dfs_state','bfs_state','path','dfs_nodes','bfs_nodes') for s in ('simple','hard')]
pending=list(jobs);running={};completed=[]
while pending or running:
    for gpu,item in list(running.items()):
        code=item['process'].poll()
        if code is None:continue
        item['log'].close();running.pop(gpu)
        if code:
            status=dict(phase='failed',job=item['name'],exit_code=code,completed=completed)
            (OUT/'status.json').write_text(json.dumps(status,indent=2));raise SystemExit(code)
        completed.append(item['name']);print('FINISHED',item['name'],flush=True)
    free=[x['index'] for x in idle_gpus() if x['idle'] and x['index'] in (0,1,4,5) and x['index'] not in running]
    while free and pending:
        gpu=free.pop(0);setting,protocol=pending.pop(0);name=f'{setting}_{protocol}'
        log=(OUT/f'{name}.log').open('w')
        process=subprocess.Popen([sys.executable,'-u',str(OUT/'audit.py'),setting,protocol],
            cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'},
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        running[gpu]=dict(process=process,name=name,log=log);print('START',name,'GPU',gpu,'PID',process.pid,flush=True)
    status=dict(phase='running' if pending or running else 'complete',completed=completed,
        pending=[f'{s}_{p}' for s,p in pending],running={str(gpu):dict(job=x['name'],pid=x['process'].pid) for gpu,x in running.items()})
    (OUT/'status.json').write_text(json.dumps(status,indent=2))
    if pending or running:time.sleep(10)

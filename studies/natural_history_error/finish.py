"""Finish stream extraction/CPU checks after this study's full-budget jobs.

Only uses the GPUs already assigned in full_workers.json, only after those
workers finish and the GPU is idle. Never signals or preempts another job.
"""
import json
from pathlib import Path
import subprocess
import sys
import time

root=Path.cwd(); folder=root/'studies/natural_history_error'
workers=json.loads((folder/'full_workers.json').read_text())
pending={w['model']:w for w in workers}; running={}; started=time.monotonic()
while pending or running:
    if time.monotonic()-started>1800: raise TimeoutError('Study exceeded 30 minutes')
    for model,w in list(pending.items()):
        result=folder/'results/models'/f'{model}.json'
        if not result.exists() or not json.loads(result.read_text()).get('complete'): continue
        pidpath=Path(f"/proc/{w['pid']}/cmdline")
        if pidpath.exists() and pidpath.read_bytes(): continue
        raw=subprocess.check_output(['nvidia-smi',f"--id={w['gpu']}",
            '--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        memory,util=map(int,raw.strip().split(','))
        if memory>500 or util>10: continue
        with (folder/f'full_stream_{model}.log').open('w') as f:
            running[model]=subprocess.Popen([sys.executable,'-u',str(folder/'stream.py'),
                '--gpu',str(w['gpu']),'--models',model],cwd=root,stdout=f,stderr=subprocess.STDOUT)
        print('STREAM START',model,flush=True);del pending[model]
    for model,p in list(running.items()):
        if p.poll() is not None:
            assert p.returncode==0,(model,p.returncode)
            print('STREAM DONE',model,flush=True);del running[model]
    time.sleep(5)
for script in ('verify.py','verify_stream.py'):
    subprocess.run([sys.executable,'-u',str(folder/script)],cwd=root,check=True)
print('ALL COMPLETE AND VERIFIED',flush=True)

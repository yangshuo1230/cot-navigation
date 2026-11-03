"""User-authorized eight-GPU completion of frozen state-model evaluations."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/mnt/ningxuefei/yangshuo/cot-navigation-unified')
sys.path.insert(0,str(ROOT/'studies/depth_generalization'))
from run import OUT, summarize

def main():
    running=[]
    for gpu,(depth,seed) in enumerate((d,s)for d in [3,6,9,12]for s in [0,1]):
        log=(OUT/f'state_L{depth}_s{seed}.log').open('a')
        proc=subprocess.Popen([sys.executable,str(ROOT/'studies/depth_generalization/run.py'),
            '--worker','dfs_state','--depth',str(depth),'--seed',str(seed)],cwd=ROOT,
            env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONUNBUFFERED':'1'},
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        running.append(dict(gpu=gpu,depth=depth,seed=seed,proc=proc,log=log))
    while True:
        status=[dict(gpu=r['gpu'],depth=r['depth'],seed=r['seed'],pid=r['proc'].pid,exit_code=r['proc'].poll())for r in running]
        (OUT/'parallel_status.json').write_text(json.dumps(status,indent=2))
        if all(r['exit_code']is not None for r in status):break
        time.sleep(10)
    for r in running:r['log'].close()
    assert all(r['exit_code']==0 for r in status),status
    summarize();summary=json.loads((OUT/'summary.json').read_text())
    assert summary['complete']==summary['expected']==208
    (OUT/'COMPLETE.json').write_text(json.dumps(dict(cells=208)))

if __name__=='__main__':main()

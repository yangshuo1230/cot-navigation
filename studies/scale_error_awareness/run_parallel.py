"""Run disjoint remaining probe jobs concurrently on physical GPU0 only.

No model/data/seed changes. Stagger initialization until settings/selection
writes finish, then multiplex stdout into one log. Resume completed results.
"""
from pathlib import Path
import json
import os
import subprocess
import threading

ROOT=Path(__file__).resolve().parents[2]
STUDY=Path(__file__).resolve().parent


def main():
    selected=json.loads((STUDY/'results/selection.json').read_text())['models']
    remaining=[e for e in selected if not (STUDY/'results/models'/f'{e["id"]}.json').exists()]
    def cost(e):
        return (8 if e['protocol'].endswith('_state') else 1)*(e['n_nodes']/36)**.5
    groups=[[] for _ in range(3)];loads=[0.]*3
    for entry in sorted(remaining,key=cost,reverse=True):
        worker=min(range(3),key=lambda i:loads[i])
        groups[worker].append(entry['id']);loads[worker]+=cost(entry)
    plan=dict(physical_gpu=0,workers=groups,completed_before=len(selected)-len(remaining))
    (STUDY/'parallel_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print('PARALLEL PLAN',json.dumps(plan),flush=True)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','WANDB_MODE':'disabled'}
    processes=[];readers=[]
    for worker,names in enumerate(groups):
        if not names:
            continue
        process=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','-m',
            'cot_navigation.analysis.scale_error_awareness','--resume','--models',*names],
            cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        ready=threading.Event()
        def read(proc=process,number=worker,started=ready):
            for line in proc.stdout:
                print(f'[worker {number}] '+line,end='',flush=True)
                # MODEL/SKIP follows settings and selection writes; launching
                # the next process now avoids shared .json.tmp name races.
                if line.startswith(('MODEL','SKIP','COMPLETE')):
                    started.set()
            started.set()
        reader=threading.Thread(target=read);reader.start()
        processes.append(process);readers.append(reader)
        if not ready.wait(120):
            raise RuntimeError('Worker initialization stalled; inspect log')
    codes=[p.wait() for p in processes]
    for reader in readers:
        reader.join()
    if any(codes):
        raise RuntimeError(f'Worker exit codes {codes}; resume missing jobs')
    print('PARALLEL COMPLETE',flush=True)


if __name__=='__main__':
    main()

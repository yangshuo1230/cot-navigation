"""Eight independent experiments, one per physical GPU, preserving configs.

Adopts the live GPU-0 worker without interrupting it. Holds both legacy queue
locks so their sequential launchers cannot race this dispatcher. The stage-A
checkpoint and all completed runs are reused unchanged. No smoke runs.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
SHORT=ROOT/'outputs/fixed_short3_v1/launches/short3_n100_s0'
BLOCK=ROOT/'outputs/block_query_v1/launches/block_n100_s0'
OUT=ROOT/'outputs/parallel_controls_v1'


def write(path,obj):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(obj,indent=2)+'\n');temporary.replace(path)


def job(suite,name):
    root=SHORT if suite=='short' else BLOCK
    return dict(id=f'{suite}/{name}',suite=suite,name=name,output=str(root/name))


def identity(pid):
    try:
        # Everything after the process name: state is field 3, start time 22.
        fields=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
        return None if fields[0]=='Z' else fields[19]
    except FileNotFoundError:return None


def completed(item):
    root=Path(item['output'])
    if (root/'COMPLETE.json').exists():return True
    metrics=root/'metrics.jsonl'
    if not metrics.exists():return False
    lines=metrics.read_text().splitlines()
    if not lines or json.loads(lines[-1]).get('step',0)<125000:return False
    views=['','-reverse_short','-reverse_long'] if item['suite']=='short' else [
        '', '-within_short','-cross_short','-cross_long','-cross_unwitnessed_yes',
        '-reverse_cross','-cross_short_role_covered','-cross_unwitnessed_role_covered_yes']
    return all((root/'eval'/f'free_test_{tag}{view}.json').exists() for tag in ['best','last'] for view in views)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adopt-pid',type=int)
    args=parser.parse_args();os.chdir(ROOT);OUT.mkdir(parents=True,exist_ok=True)
    # Retain descriptors until every worker is finished.
    locks=[]
    for path in [OUT/'.launch.lock',SHORT/'.launch.lock',BLOCK/'.launch.lock']:
        handle=path.open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(handle)
    jobs=[job('short','short3/dfs_state'),job('block','within_only'),job('block','cross_allowed'),
          *[job('block',f'{arm}/{protocol}') for protocol in ['path','dfs_state','bfs_state','dfs_nodes','bfs_nodes'] for arm in ['within_only','cross_allowed']],
          job('short','all_hops/dfs_state'),job('short','short3/bfs_state'),job('short','all_hops/bfs_state'),
          job('short','short3/dfs_nodes'),job('short','all_hops/dfs_nodes'),
          job('short','short3/bfs_nodes'),job('short','all_hops/bfs_nodes')]
    pending=[j for j in jobs if not completed(j)];running={};finished=[];failed=[]
    if (OUT/'status.json').exists():
        previous=json.loads((OUT/'status.json').read_text())
        for gpu,entry in previous.get('running',{}).items():
            if identity(entry['pid'])!=entry['start']:continue
            item=next(j for j in jobs if j['id']==entry['job']['id'])
            cmd=Path(f"/proc/{entry['pid']}/cmdline").read_bytes().split(b'\0')
            assert b'--run-one' in cmd and item['name'].encode() in cmd
            if item in pending:pending.remove(item)
            running[int(gpu)]={**entry,'job':item,'process':None,'adopted':True}
            print('RECOVER',gpu,entry['pid'],item['id'],flush=True)
    if args.adopt_pid and 0 not in running:
        ident=identity(args.adopt_pid)
        assert ident is not None,'Adopted worker is not alive'
        cmd=Path(f'/proc/{args.adopt_pid}/cmdline').read_bytes().split(b'\0')
        assert b'--run-one' in cmd and b'short3/dfs_state' in cmd
        item=next(j for j in pending if j['id']=='short/short3/dfs_state');pending.remove(item)
        running[0]=dict(job=item,pid=args.adopt_pid,start=ident,process=None,adopted=True)
        print('ADOPT GPU 0',args.adopt_pid,item['id'],flush=True)
    for gpu in range(8):
        if gpu in running:continue
        pids=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        assert not pids,f'GPU {gpu} has an unmanaged process: {pids}'
    def status():
        write(OUT/'status.json',dict(updated_unix=time.time(),running={str(g):{k:v for k,v in entry.items() if k!='process'} for g,entry in running.items()},pending=pending,finished=finished,failed=failed))
    while pending or running:
        for gpu,entry in list(running.items()):
            proc=entry['process'];alive=(proc.poll() is None) if proc else identity(entry['pid'])==entry['start']
            if alive:continue
            item=entry['job'];success=(proc is None or proc.returncode==0) and completed(item)
            if success:
                write(Path(item['output'])/'COMPLETE.json',dict(steps=125000,dispatcher='parallel_controls_v1'))
                finished.append(item);print('DONE',gpu,item['id'],flush=True)
            else:
                failed.append(dict(job=item,exit_code=None if proc is None else proc.returncode))
                print('FAILED',gpu,item['id'],flush=True)
            del running[gpu]
        for gpu in range(8):
            if gpu in running or not pending:continue
            item=pending.pop(0)
            if completed(item):finished.append(item);continue
            if item['suite']=='short':
                command=[sys.executable,'-u',str(ROOT/'scripts/short_hop_control.py'),'--run-name','short3_n100_s0','--run-one',item['name'],'--resume']
            else:command=[sys.executable,'-u',str(ROOT/'scripts/block_query_control.py'),'--run-one',item['name']]
            env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','TORCHINDUCTOR_COMPILE_THREADS':'2','PYTHONPATH':str(ROOT/'src')}
            with (OUT/(item['id'].replace('/','_')+'.log')).open('a') as log:
                proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            running[gpu]=dict(job=item,pid=proc.pid,start=identity(proc.pid),process=proc,adopted=False)
            print('START',gpu,proc.pid,item['id'],flush=True)
        status()
        if pending or running:time.sleep(10)
    print('FINISHED',len(finished),'FAILED',len(failed),flush=True)
    if failed:raise SystemExit(1)


if __name__=='__main__':main()

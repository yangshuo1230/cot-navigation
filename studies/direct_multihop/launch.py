"""Bounded background queue, one process per free physical GPU 1-7.

The only mutations outside this study are new results under its own directory.
Existing jobs are never terminated. GPU0 is explicitly forbidden. A launcher
lock prevents duplicate queues, and a persistent ledger charges failed runs
against the same per-job wall allocation before a checkpoint resume.
"""
from pathlib import Path
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

from data import atomic_json, digest, prepare
from plan import GROUP, matrix

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def utc():
    return datetime.now(timezone.utc).isoformat()


def plan(out, data_dir):
    out.mkdir(parents=True,exist_ok=True)
    manifest = prepare(data_dir)
    from train import cpu_preflight
    receipt = cpu_preflight()
    code_hash = digest({p.name:__import__('hashlib').sha256(p.read_bytes()).hexdigest()
                        for p in HERE.glob('*.py')})
    jobs = []
    for spec in matrix():
        parent = spec.get('parent_checkpoint')
        if parent:
            parent_path = Path(parent)
            assert parent_path.is_file(), parent
            spec['parent_sha256'] = __import__('hashlib').sha256(parent_path.read_bytes()).hexdigest()
            wb = parent_path.parent.parent/'wandb_status.json'
            spec['parent_run_id'] = json.loads(wb.read_text())['run_id'] if wb.exists() else None
        else:
            spec['parent_sha256'] = spec['parent_run_id'] = None
        helper = ROOT/'studies/direct_learning/train.py'
        spec['core_helper_sha256'] = __import__('hashlib').sha256(helper.read_bytes()).hexdigest()
        job = {**spec,'group':GROUP,'data_dir':str(data_dir.resolve()),
               'data_manifest_sha256':digest(manifest),'code_sha256':code_hash,
               'batch_size':64,'protocol':'direct','vocab_size':270,
               'minimum_stage_steps':20000,'gate_consecutive_evaluations':3,
               'selection_policy':'validation only; final fixed-budget checkpoint is primary'}
        folder=out/job['name'];folder.mkdir(exist_ok=True)
        target=folder/'experiment.json'
        if target.exists():
            assert json.loads(target.read_text()) == job, 'Existing experiment differs; use a new study directory.'
        else:
            atomic_json(target,job)
        jobs.append({'name':job['name'],'spec':str(target),'budget_seconds':job['budget_seconds']})
    atomic_json(out/'preflight.json',{'at':utc(),'cpu_loss_equivalence':receipt,
                                     'no_optimizer_updates':True,'data_manifest':str(data_dir/'manifest.json'),
                                     'source_sha256':code_hash})
    atomic_json(out/'launch_plan.json',{'group':GROUP,'gpu_hours':30,'jobs':jobs,
                                       'gpus':[1,2,3,4,5,6,7],
                                       'seed_meaning':'parent initialization and finetuning sampler; shared original data'})
    return jobs


def idle_gpus():
    response=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
                                      '--format=csv,noheader,nounits'],text=True)
    result=[]
    for line in response.splitlines():
        gpu,memory,util=map(int,line.split(','))
        if gpu in range(1,8) and memory < 200 and util < 5:
            result.append(gpu)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--out',type=Path,default=HERE/'runs'/GROUP)
    args=parser.parse_args()
    os.chdir(ROOT)
    args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=True)
    lock=(args.out/'.launch.lock').open('w')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('A launcher already owns this exact queue.')
    jobs=plan(args.out,HERE/'data_v1')
    if args.prepare_only:
        print('PREPARED',len(jobs),'jobs, 30 GPU hours; CPU checks only.',flush=True)
        return
    ledger_file=args.out/'ledger.json'
    ledger=json.loads(ledger_file.read_text()) if ledger_file.exists() else {
        'started_at':utc(),'jobs':{j['name']:{'charged_seconds':0.,'attempts':0,'status':'pending'} for j in jobs}}
    processes={}
    def persist():
        ledger['updated_at']=utc()
        atomic_json(ledger_file,ledger)
        atomic_json(args.out/'status.json',{
            'updated_at':ledger['updated_at'],
            'completed':[k for k,v in ledger['jobs'].items() if v['status']=='completed'],
            'running':{item['name']:{'gpu':g,'pid':item['process'].pid} for g,item in processes.items()},
            'pending':[k for k,v in ledger['jobs'].items() if v['status']=='pending'],
            'failed':[k for k,v in ledger['jobs'].items() if v['status']=='failed'],
            'charged_gpu_hours':sum(v['charged_seconds'] for v in ledger['jobs'].values())/3600,
            'allocated_gpu_hours':30})
    # If a launcher was lost, do not overlap still-live child processes.
    for name,item in ledger['jobs'].items():
        if item['status']=='running':
            pid=item.get('pid')
            if pid and Path(f'/proc/{pid}/cmdline').exists():
                cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
                if str(args.out/name/'experiment.json') in cmd:
                    raise RuntimeError(f'Orphan child {pid} is still running for {name}; will not duplicate it.')
            # Charge the interrupted attempt through the last durable heartbeat.
            item['charged_seconds']+=item.pop('live_seconds',0.)
            item['status']='pending'
    print('QUEUE START',utc(),'15 jobs / 30 GPU hours / GPU1-7',flush=True)
    while True:
        for gpu,item in list(processes.items()):
            proc=item['process'];meta=ledger['jobs'][item['name']]
            live=time.monotonic()-item['started']
            meta['live_seconds']=live
            if proc.poll() is None:
                if live > item['remaining']+600 and not item.get('signaled'):
                    proc.terminate();item['signaled']=True
                if live > item['remaining']+900:
                    proc.kill()
                continue
            meta['charged_seconds']+=live;meta.pop('live_seconds',None)
            meta['exit_code']=proc.returncode;meta['finished_at']=utc();item['log'].close()
            folder=args.out/item['name']
            if proc.returncode==0 and (folder/'completed.json').exists():
                meta['status']='completed'
            elif meta['attempts'] < 2 and meta['charged_seconds'] < item['budget']-60:
                meta['status']='pending'
            else:
                meta['status']='failed'
            print('FINISH',item['name'],meta['status'],'seconds',round(meta['charged_seconds'],1),flush=True)
            del processes[gpu]
        pending=[j for j in jobs if ledger['jobs'][j['name']]['status']=='pending']
        if not pending and not processes:
            persist();break
        for gpu in idle_gpus():
            if gpu in processes or not pending:
                continue
            j=pending.pop(0);meta=ledger['jobs'][j['name']]
            remaining=j['budget_seconds']-meta['charged_seconds']
            if remaining<=60:
                meta['status']='failed';meta['error']='Allocation exhausted before a complete result receipt.';continue
            folder=args.out/j['name']
            log=(folder/'train.log').open('a',buffering=1)
            env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'2',
                 'OPENBLAS_NUM_THREADS':'2','TORCHINDUCTOR_COMPILE_THREADS':'1',
                 'TORCHINDUCTOR_CACHE_DIR':str(HERE/'compile_cache'/str(gpu)),
                 'PYTHONUNBUFFERED':'1','WANDB_SILENT':'true'}
            cmd=[str(ROOT/'.venv/bin/python'),str(HERE/'train.py'),'--spec',j['spec'],
                 '--budget-seconds',str(remaining)]
            proc=subprocess.Popen(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            meta.update(status='running',pid=proc.pid,gpu=gpu,started_at=utc(),attempts=meta['attempts']+1)
            processes[gpu]={'name':j['name'],'process':proc,'started':time.monotonic(),
                            'remaining':remaining,'budget':j['budget_seconds'],'log':log}
            print('START',j['name'],'GPU',gpu,'PID',proc.pid,flush=True)
        persist();time.sleep(10)
    atomic_json(args.out/'finished.json',{'at':utc(),'completed':sum(v['status']=='completed' for v in ledger['jobs'].values()),
                                         'failed':sum(v['status']=='failed' for v in ledger['jobs'].values()),
                                         'charged_gpu_hours':sum(v['charged_seconds'] for v in ledger['jobs'].values())/3600})
    print('QUEUE FINISHED',utc(),flush=True)


if __name__=='__main__':main()

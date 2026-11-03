"""Run the validated cohort's studies sequentially on GPU0, resumably."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
MANIFEST='studies/correction_behavior/cohorts/all_positions_tokenmatched_20261103.json'


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(path)


def process_running(pid):
    try:
        os.kill(pid,0)
        status=Path(f'/proc/{pid}/stat')
        return not (status.exists() and status.read_text().split(') ',1)[1].startswith('Z'))
    except ProcessLookupError:return False


def adopt_history(pid,out,manifest):
    """Keep the existing probe worker alive when replacing its coordinator."""
    if process_running(pid):
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\0',' ')
        assert 'correction_mechanisms/run.py' in cmd and '--phase history' in cmd
    while process_running(pid):time.sleep(5)
    # An orphan's exit status cannot be waited on. Validate every required
    # atomic result instead, and never mark a failed/incomplete worker done.
    for entry in manifest['models']:
        for alignment in ['error_token_0_5','record_0_2']:
            path=out/'history'/(entry['id']+'__'+alignment+'.json')
            data=json.loads(path.read_text())
            assert data['complete'] and data['model']['id']==entry['id']
            assert data['model']['run']==entry['run'] and data['checkpoint']['step']==entry['checkpoint_step']
            assert data['checkpoint']['sha256']==data['model']['checkpoint_sha256']


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',default=MANIFEST);p.add_argument('--launch',action='store_true')
    p.add_argument('--skip-natural',action='store_true');p.add_argument('--natural-after-probes',action='store_true')
    p.add_argument('--defer-forced',action='store_true');p.add_argument('--adopt-history-pid',type=int);args=p.parse_args()
    os.chdir(ROOT);manifest=json.loads(Path(args.manifest).read_text())
    out=ROOT/'outputs/analysis/correction_mechanisms'/manifest['cohort'];out.mkdir(parents=True,exist_ok=True)
    choices=out/'execution_choices.json';options=json.loads(choices.read_text()) if choices.exists() else {}
    if args.skip_natural:
        options.update(skip_natural=True,reason='User cancelled further natural behavior analysis',updated=time.time());write(choices,options)
        natural_status=ROOT/manifest['output']/'status.json'
        prior=json.loads(natural_status.read_text()) if natural_status.exists() else {}
        write(natural_status,dict(prior,phase='stopped_by_user',complete=False,reason=options['reason']))
    if args.natural_after_probes:
        options.update(skip_natural=False,natural_after_probes=True,reason='Finish probes, then resume natural analysis and draw figures',updated=time.time())
        natural_status=ROOT/manifest['output']/'status.json'
        prior=json.loads(natural_status.read_text()) if natural_status.exists() else {}
        write(natural_status,dict(prior,phase='queued_after_probes',complete=False,reason=options['reason']))
    if args.defer_forced:options['skip_forced']=True
    if args.adopt_history_pid:options.setdefault('adopt_running',{})['natural_history_probes']=args.adopt_history_pid
    write(choices,options)
    if args.launch:
        if (out/'process.json').exists():
            pid=json.loads((out/'process.json').read_text())['pid']
            if process_running(pid):raise RuntimeError(f'Pipeline already active: {pid}')
        with (out/'pipeline.log').open('a') as log:
            child=subprocess.Popen([sys.executable,str(Path(__file__)), '--manifest',args.manifest],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','WANDB_MODE':'disabled'})
        write(out/'process.json',dict(pid=child.pid,started=time.time(),log=str(out/'pipeline.log')))
        print(child.pid);return
    behavior=manifest['output']
    stages=[('natural_behavior',['studies/correction_behavior/run.py','--manifest',args.manifest]),
        ('natural_probe_corpus',['studies/correction_mechanisms/run.py','--manifest',args.manifest,'--phase','prepare']),
        ('state_probes_and_lenses',['studies/correction_mechanisms/run.py','--manifest',args.manifest,'--phase','state']),
        ('natural_history_probes',['studies/correction_mechanisms/run.py','--manifest',args.manifest,'--phase','history']),
        ('forced_mistake',['studies/correction_behavior/oracle_mistake.py','--output',behavior]+(['--allow-partial-reference'] if options.get('skip_natural') else [])),
        ('figures',['studies/correction_mechanisms/report.py','--manifest',args.manifest])]
    if options.get('natural_after_probes'):
        natural=stages.pop(0);stages.insert(3,natural)
    status=json.loads((out/'pipeline_status.json').read_text()) if (out/'pipeline_status.json').exists() else dict(completed=[],total=len(stages))
    status['deferred']=['forced_mistake'] if options.get('skip_forced') else []
    status['skipped']=[s for s in status.get('skipped',[]) if s!='natural_behavior' or options.get('skip_natural')]
    for name,command in stages:
        options=json.loads(choices.read_text())
        if (name=='natural_behavior' and options.get('skip_natural')) or (name=='forced_mistake' and options.get('skip_forced')):
            status.setdefault('skipped',[])
            if name not in status['skipped']:status['skipped'].append(name)
            write(out/'pipeline_status.json',status);continue
        if name in status['completed']:continue
        status.update(phase=name,state='running',updated=time.time());write(out/'pipeline_status.json',status)
        print('PHASE',name,flush=True)
        try:
            adopted=options.get('adopt_running',{}).get(name)
            if adopted:
                assert name=='natural_history_probes'
                print('ADOPT HISTORY',adopted,flush=True)
                try:adopt_history(adopted,out,manifest)
                finally:
                    options=json.loads(choices.read_text());options.get('adopt_running',{}).pop(name,None);write(choices,options)
            else:subprocess.run([sys.executable,*command],cwd=ROOT,check=True)
        except BaseException as exc:
            status.update(state='failed',error=repr(exc),updated=time.time());write(out/'pipeline_status.json',status);raise
        status['completed'].append(name);status['updated']=time.time();write(out/'pipeline_status.json',status)
    status.update(state='complete',updated=time.time());write(out/'pipeline_status.json',status)


if __name__=='__main__':main()

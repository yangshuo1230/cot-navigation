"""Reuse the common idle-GPU launcher; default prints a read-only plan."""
import argparse,json,shlex,sys
from pathlib import Path
from cot_navigation.runtime.workspace import ROOT
from cot_navigation.runtime.launch import plan_launch
from cot_navigation.experiments.artifacts import atomic_json
P=Path(__file__).resolve().parent


def remaining(arm,seed):
    file=P/'results'/f'{arm}_s{seed}.json'
    blob=json.loads(file.read_text()) if file.exists() else {}
    count=0
    for task in ['main','n64','hard','deep','n32']:
        case=blob.get('cases',{}).get(task,{})
        if arm=='correction':
            count+=0 if all(case.get(c,{}).get('finished') for c in ['native','no_mistake']) else 2
        else:count+=int(not case.get('native',{}).get('finished'))
    return count


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus',nargs='+',type=int,default=list(range(8)))
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    bins=[dict(gpu=g,load=0,jobs=[]) for g in a.gpus]
    jobs=sorted([(remaining(arm,s),arm,s) for arm in ['correction','matched_clean','original_clean']
                 for s in range(4)],reverse=True)
    for work,arm,seed in jobs:
        if not work:continue
        target=min(bins,key=lambda b:b['load']);target['load']+=work;target['jobs'].append(f'{arm}:{seed}')
    bins=[b for b in bins if b['jobs']]
    commands=[shlex.join([sys.executable,str(P/'worker.py'),*b['jobs']]) for b in bins]
    report=plan_launch(commands,[b['gpu'] for b in bins],str(ROOT/'outputs/analysis/correction_breakdown'),dry_run=not a.execute)
    report['job_bins']=bins
    if a.execute:atomic_json(P/'launch_record.json',report)
    print(json.dumps(report,indent=2))
    return report['exit_code']


if __name__=='__main__':raise SystemExit(main())

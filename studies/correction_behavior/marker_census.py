"""Recover raw marker counts using the exact frozen generation workflow.

Separate compact census; original behavior results remain immutable. Every
replayed trajectory is compared with its original hash, not assumed identical.
"""
import argparse
from collections import Counter,defaultdict
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

P=Path(__file__).resolve().parent


def report(out):
    groups=defaultdict(Counter);done=0
    for file in (out/'results').glob('*.json.gz'):
        with gzip.open(file,'rt') as f:data=json.load(f)
        if not data['complete']:continue
        done+=1;m=data['identity']['model'];c=groups[data['identity']['task'],m['group']]
        for row in data['rows']:
            for k,v in row['markers'].items():c[k]+=v
            c['trajectories']+=1;c['original_hash_match']+=row['original_hash_match']
    result=dict(complete_tasks=done,total_tasks=60,complete=done==60,groups={})
    for (task,group),c in groups.items():result['groups'].setdefault(task,{})[group]=dict(c)
    (out/'marker_counts.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('MARKER CENSUS',done,60,flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--report',action='store_true');args=ap.parse_args()
    if (args.output/'DEPRECATED.json').exists():raise ValueError('Withdrawn cohort: marker replay disabled')
    out=args.output/'marker_census'
    if args.report:report(out);return
    spec=importlib.util.spec_from_file_location('frozen_behavior_census',P/'run.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    original=args.output.resolve();plan=json.loads((original/'plan.json').read_text());out.mkdir(exist_ok=True,parents=True)
    assert all(base.sha_file(base.resolve(name))==digest for name,digest in plan['source_sha256'].items())
    census_plan=dict(plan,marker_census_sha256=base.sha_file(Path(__file__)))
    if (out/'plan.json').exists():assert json.loads((out/'plan.json').read_text())==census_plan
    else:base.atomic_json(out/'plan.json',census_plan)
    pending={};hashes={};real_replay=base.replay;real_write=base.write_cache;real_subprocess=base.subprocess.run
    def replay(trace,pred,adj,src,dst,tok,**kwargs):
        result=real_replay(trace,pred,adj,src,dst,tok,**kwargs)
        detail=base.audit_continuation(trace,adj,src,dst,tok);rr=detail['records'];success=0;true=0;false=0
        for i,r in enumerate(rr):
            true+=r['true_cancel'];false+=r['false_cancel']
            if r['true_cancel'] and i+1<len(rr):success+=not rr[i+1]['bad'] and rr[i+1]['action_legal'] and not rr[i+1]['cancel']
        whole=bool(result['trace_valid'] and pred==1)
        raw=list(trace).count(tok.id('<MISTAKE>'))
        key=hashlib.sha256(base.np.asarray(trace,base.np.int32).tobytes()).hexdigest()
        pending.setdefault(key,[]).append(dict(raw=raw,auditable=true+false,true=true,false=false,unclassified=raw-true-false,
            immediate_repair=success,whole_success_marker_events=success if whole else 0,
            whole_success_trajectories=int(whole and success>0)))
        return result
    def write(path,data):
        name=path.name
        if name not in hashes:
            with gzip.open(original/'results'/name,'rt') as f:hashes[name]={r['query']:r['trace_sha256'] for r in json.load(f)['rows']}
        compact=[]
        for row in data['rows']:
            if 'markers' in row:compact.append(row);continue
            digest=row['trace_sha256']
            compact.append(dict(query=row['query'],trace_sha256=digest,original_hash_match=digest==hashes[name][row['query']],markers=pending[digest].pop(0)))
            if not pending[digest]:del pending[digest]
        data['rows']=compact;real_write(path,data)
    def subprocess_run(argv,**kwargs):
        if len(argv)>1 and argv[1]==str(P/'report.py'):
            report(out)
            return base.subprocess.CompletedProcess(argv,0)
        return real_subprocess(argv,**kwargs)
    base.replay=replay;base.write_cache=write;base.subprocess.run=subprocess_run
    try:base.run(plan['manifest'],out,plan)
    except BaseException as exc:base.atomic_json(out/'failure.json',dict(error=repr(exc)));raise


if __name__=='__main__':main()

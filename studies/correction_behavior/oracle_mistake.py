"""Paired natural vs oracle-MISTAKE generations, without supplying repairs."""
import argparse
from collections import Counter,defaultdict,deque
import gzip
import importlib.util
import json
from pathlib import Path

P=Path(__file__).resolve().parent


def report(out):
    plan=json.loads((out/'plan.json').read_text());total=len(plan['models'])*len(plan['manifest']['tasks'])
    groups=defaultdict(Counter);done=0
    for file in (out/'results').glob('*.json.gz'):
        with gzip.open(file,'rt') as f:b=json.load(f)
        if not b['complete']:continue
        done+=1;m=b['identity']['model'];c=groups[b['identity']['task'],m['group']]
        for r in b['rows']:
            n=r['oracle_pair'];c['n']+=1;c['native_correct']+=n['correct'];c['forced_correct']+=r['correct']
            c['native_valid']+=n['valid'];c['forced_valid']+=r['trace_valid']
            c['helped']+=r['correct'] and not n['correct'];c['harmed']+=n['correct'] and not r['correct']
            c['native_cut']+=n['interrupted'];c['forced_cut']+=r['interrupted'];c['forced_actions']+=n['forced']
            c['triggered_queries']+=n['forced']>0;c['unsupported']+=n['unsupported']
            c['native_original_hash_match']+=n['original_hash_match'] is True
            c['reference_queries']+=n['original_hash_match'] is not None;c['forced_repair_records']+=r['repairs']
            c['no_trigger_trace_drift']+=n['forced']==0 and n['hash']!=r['trace_sha256']
    data=dict(completed_tasks=done,total_tasks=total,complete=done==total,groups={})
    text=['# Oracle MISTAKE配对实验','','完整错误记录的动作被外部替换为MISTAKE分号，模型自由重写；不提供正确字段/动作。',
          '原生4K上下文不变，动作回退带来额外计算。未闭合记录、提前回答不强行修复。','',f'完成：{done}/{total}任务。','']
    for (task,group),c in groups.items():data['groups'].setdefault(task,{})[group]=dict(c)
    for task,gg in data['groups'].items():
        text+=['## '+task,'','|组|查询|自然正确|干预正确|改善/损害|自然合法|干预合法|触发查询|强制次数|','|---|---:|---:|---:|---|---:|---:|---:|---:|']
        for group,c in gg.items():text.append(f"|{group}|{c['n']}|{c['native_correct']}|{c['forced_correct']}|{c['helped']}/{c['harmed']}|{c['native_valid']}|{c['forced_valid']}|{c['triggered_queries']}|{c['forced_actions']}|")
        text+=['']
    for name,value in [('comparison.json',json.dumps(data,ensure_ascii=False,indent=2)),('REPORT.md','\n'.join(text))]:
        temp=out/(name+'.tmp');temp.write_text(value,encoding='utf-8');temp.replace(out/name)
    print('ORACLE REPORT',done,total,flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--report',action='store_true');ap.add_argument('--capture-native',action='store_true')
    ap.add_argument('--allow-partial-reference',action='store_true');args=ap.parse_args()
    original=args.output.resolve()
    if (original/'DEPRECATED.json').exists():raise ValueError('Withdrawn cohort: oracle replay disabled')
    out=original/'oracle_mistake';out.mkdir(exist_ok=True,parents=True)
    if args.report:report(out);return
    spec=importlib.util.spec_from_file_location('behavior_oracle_base',P/'run.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    from cot_navigation.analysis.oracle_mistake import MistakeOracle
    plan=json.loads((original/'plan.json').read_text())
    reference_root=base.resolve(plan['manifest'].get('reference_results_dir',str(original/'results')))
    if args.capture_native:
        native_out=original/'native_supplement';native_out.mkdir(exist_ok=True,parents=True)
        native_plan=json.loads(json.dumps(plan));native_plan['manifest']['output']=str(native_out)
        base.atomic_json(native_out/'plan.json',native_plan)
    plan['manifest']['cohort']+='__oracle_mistake';plan['manifest']['output']=str(out)
    # Probe corpus is the unmodified baseline, captured by the natural run.
    # Do not persist a second, intervention-contaminated token corpus.
    if 'capture_probe_test' in plan['manifest']:plan['manifest']['capture_probe_test']=False
    if args.allow_partial_reference:plan['manifest']['reference_policy']='available_only; contemporaneous paired natural control always required'
    # Explicitly pin the new optional decode hook instead of claiming identical code.
    for name in list(plan['source_sha256']):plan['source_sha256'][name]=base.sha_file(base.resolve(name))
    for file in [Path(__file__),base.ROOT/'src/cot_navigation/analysis/oracle_mistake.py']:
        plan['source_sha256'][str(file.relative_to(base.ROOT))]=base.sha_file(file)
    if (out/'plan.json').exists():assert json.loads((out/'plan.json').read_text())==plan
    else:base.atomic_json(out/'plan.json',plan)
    actual_generate=base.generate;actual_replay=base.replay;actual_write=base.write_cache;actual_subprocess=base.subprocess.run
    waiting=deque();pending=defaultdict(deque);original_hashes={};sham_checks=set();native_rows={}
    def generate(*a,**kw):
        model,prompts,protocol,tok=a[:4]
        native=actual_generate(*a,**kw)
        # Same model state and cache implementation, with an identity callback.
        shape=(id(model),len(prompts[0]))
        if shape not in sham_checks:
            sham=actual_generate(*a,**kw,action_override=lambda i,t:None)
            assert all(x['generated']==y['generated'] for x,y in zip(native,sham))
            sham_checks.add(shape)
        oracles=[]
        for prompt in prompts:
            assert prompt[-4]==tok.id('<QUERY>') and prompt[-1]==tok.id('|')
            adj=defaultdict(list)
            for k in range(0,len(prompt)-4,3):
                assert prompt[k]==tok.id('<EDGE>');adj[prompt[k+1]].append(prompt[k+2])
            oracles.append(MistakeOracle(dict(adj),prompt[-3],prompt[-2],tok))
        forced=actual_generate(*a,**kw,action_override=lambda i,t:oracles[i](t))
        waiting.extend(zip(native,oracles));return forced
    def replay(trace,pred,adj,src,dst,tok,**kw):
        result=actual_replay(trace,pred,adj,src,dst,tok,**kw)
        native,oracle=waiting.popleft();n=actual_replay(native['trace_tokens'],native['pred'],adj,src,dst,tok,**kw)
        digest=lambda ts:base.hashlib.sha256(base.np.asarray(ts,base.np.int32).tobytes()).hexdigest()
        extra=None
        if args.capture_native:
            extra=dict(correct=bool(native['pred']==1 and not native['format_fail']),pred=native['pred'],interrupted=native['cot_interrupted'],
                forced_stop=native['forced_stop'],trace_valid=n['trace_valid'],first=n['first'],true_marks=n['true_marks'],false_marks=n['false_marks'],
                repairs=n['repairs'],raw_mistake_tokens=n['raw_mistake_tokens'],trace_tokens=len(native['trace_tokens']),
                trace_sha256=digest(native['trace_tokens']),dependencies=base.dependencies(native['trace_tokens'],adj,src,dst,tok),
                **base.audit_continuation(native['trace_tokens'],adj,src,dst,tok))
        pending[digest(trace)].append(dict(native_detail=extra,correct=bool(native['pred']==1 and not native['format_fail']),valid=n['trace_valid'],
            interrupted=native['cot_interrupted'],hash=digest(native['trace_tokens']),forced=oracle.counts['forced'],
            bad_fields=oracle.counts['bad_fields'],bad_action=oracle.counts['bad_action'],unsupported=oracle.counts['unsupported']))
        return result
    def write(path,data):
        if args.capture_native and path.name not in native_rows:
            native_path=original/'native_supplement/results'/path.name
            native_rows[path.name]=json.load(gzip.open(native_path,'rt'))['rows'] if native_path.exists() else []
        if path.name not in original_hashes:
            reference_path=reference_root/path.name
            if reference_path.exists():
                with gzip.open(reference_path,'rt') as f:original_hashes[path.name]={r['query']:r['trace_sha256'] for r in json.load(f)['rows']}
            else:
                if not args.allow_partial_reference:raise FileNotFoundError(reference_path)
                original_hashes[path.name]={}
        for r in data['rows']:
            if 'oracle_pair' not in r:
                pair=pending[r['trace_sha256']].popleft();reference_hash=original_hashes[path.name].get(r['query'])
                if reference_hash is None and not args.allow_partial_reference:raise KeyError(r['query'])
                pair['original_hash_match']=pair['hash']==reference_hash if reference_hash is not None else None
                extra=pair.pop('native_detail')
                if extra is not None:
                    extra.update(query=r['query'],distance=r['distance'],prompt_tokens=r['prompt_tokens'],original_hash_match=pair['original_hash_match'])
                    native_rows[path.name].append(extra)
                r['oracle_pair']=pair
        if args.capture_native and path.name in native_rows:
            assert [r['query'] for r in native_rows[path.name]]==[r['query'] for r in data['rows']]
            actual_write(original/'native_supplement/results'/path.name,dict(identity=data['identity'],complete=data['complete'],rows=native_rows[path.name]))
        actual_write(path,data)
    def subprocess_run(argv,**kw):
        if len(argv)>1 and argv[1]==str(P/'report.py'):report(out);return base.subprocess.CompletedProcess(argv,0)
        return actual_subprocess(argv,**kw)
    base.generate=generate;base.replay=replay;base.write_cache=write;base.subprocess.run=subprocess_run
    try:base.run(plan['manifest'],out,plan)
    except BaseException as exc:
        base.atomic_json(out/'failure.json',dict(error=repr(exc)));raise


if __name__=='__main__':main()

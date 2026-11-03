"""Only missing analyses on existing valid data; mark raw-count coverage explicitly."""
import argparse
from collections import Counter,defaultdict
import gzip
import json
from pathlib import Path

P=Path(__file__).resolve().parent;ROOT=P.parents[1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();out=a.output
    plan=json.loads((out/'plan.json').read_text());groups=defaultdict(Counter);types=defaultdict(lambda:defaultdict(lambda:[0,0]));breakdown={};native_done=0;structure_tasks=0
    for model in plan['models']:
        arm=model['training_arm'];seed=model['seed'];bpath=ROOT/'studies/correction_reproduction/breakdown/results'/f'{arm}_s{seed}.json'
        breakdown[arm,seed]=json.loads(bpath.read_text())
        for task in plan['manifest']['tasks']:
            name=f"{model['id']}__{task}.json.gz";reference=ROOT/plan['manifest']['reference_results_dir']/name
            with gzip.open(reference,'rt') as f:ref=json.load(f)
            fresh=out/'native_supplement/results'/name;data=ref
            if fresh.exists():
                with gzip.open(fresh,'rt') as f:replayed=json.load(f)
                if replayed['complete']:data=replayed;native_done+=1
            oldhash={r['query']:r['trace_sha256'] for r in ref['rows']}
            original={r['query_id']:r for r in breakdown[arm,seed]['cases'][task]['native']['rows']}
            key=task,model['group'];c=groups[key]
            structure_tasks+=all('dependencies' in t for t in data['rows'])
            for t in data['rows']:
                rr=t['records'];c['queries']+=1;c['native_hash_match']+=t['trace_sha256']==oldhash[t['query']]
                true=sum(r['true_cancel'] for r in rr);false=sum(r['false_cancel'] for r in rr)
                c['true']+=true;c['false']+=false;c['errors']+=sum(r['bad'] for r in rr)
                if t['first']:
                    first=t['first'];i=first['record']-1;marked=i<len(rr) and rr[i]['true_cancel'];c['first_errors']+=1;c['first_marked']+=marked
                    types[key]['first:'+first['kind']][0]+=1;types[key]['first:'+first['kind']][1]+=marked
                for r in rr:
                    if not r['bad']:continue
                    for typ in r['errors']:
                        types[key]['all:'+typ][0]+=1;types[key]['all:'+typ][1]+=r['true_cancel']
                    if len(r['errors'])==1:
                        typ=next(iter(r['errors']));types[key]['single:'+typ][0]+=1;types[key]['single:'+typ][1]+=r['true_cancel']
                success=sum(r['true_cancel'] and i+1<len(rr) and not rr[i+1]['bad'] and rr[i+1]['action_legal'] and not rr[i+1]['cancel'] for i,r in enumerate(rr))
                c['immediate_repair']+=success;c['whole_repaired']+=bool(success and t['trace_valid'] and t['correct'])
                if t['correct'] and not t['trace_valid']:
                    c['correct_invalid']+=1
                    if any(r['at_goal'] for r in rr):c['legal_execution_reached']+=1
                    elif t['stopped_on_illegal_action']:c['illegal_before_goal']+=1
                    else:c['unproven_early_answer']+=1
                previous=original[t['query']]
                if 'raw_mistake_tokens' in t:
                    c['raw_known_queries']+=1;c['raw_mistakes_known']+=t['raw_mistake_tokens']
                elif previous['trace_hash']==t['trace_sha256']:
                    c['raw_known_queries']+=1;c['raw_mistakes_known']+=previous['raw_mistake_tokens']
    result=dict(cohort=plan['manifest']['cohort'],native_replay_complete=native_done,total_tasks=40,structural_complete=structure_tasks,
        note='原始MISTAKE只累加哈希相同的旧计数或新原生重放计数；未覆盖查询不假定为0。所有计数不含废弃批次。',groups={})
    text=['# 旧版8模型：缺项分析进度','','不使用已废弃批次；复用旧自然结果，原始计数仅在轨迹哈希匹配时复用。',
        f'节点级结构覆盖：{structure_tasks}/40任务；配对原生补录完成：{native_done}/40任务。','']
    for task in plan['manifest']['tasks']:
        text+=['## '+task,'','|组|查询|原始标记计数覆盖|可核验标对|首错未标/首错|全部错误未标/错误|本条修复|修复后整条成功|','|---|---:|---:|---:|---|---|---:|---:|']
        for (tt,group),c in groups.items():
            if tt!=task:continue
            result['groups'].setdefault(task,{})[group]=dict(c,detection_types=dict(types[tt,group]))
            text.append(f"|{group}|{c['queries']}|{c['raw_known_queries']}/{c['queries']}|{c['true']}|{c['first_errors']-c['first_marked']}/{c['first_errors']}|{c['errors']-c['true']}/{c['errors']}|{c['immediate_repair']}|{c['whole_repaired']}|")
        text+=['']
    for name,value in [('supplement_summary.json',json.dumps(result,ensure_ascii=False,indent=2)),('SUPPLEMENT_REPORT.md','\n'.join(text))]:
        temp=out/(name+'.tmp');temp.write_text(value,encoding='utf-8');temp.replace(out/name)
    print('SUPPLEMENT REPORT native',native_done,'structure',structure_tasks,'/40')


if __name__=='__main__':main()

"""Separate annotation-only success from unverified success after bad execution.

Uses the saved strict reference machine; never extrapolates after censoring.
This is behavioral classification, not a counterfactual effect on the answer.
"""
import argparse
from collections import Counter,defaultdict
import gzip
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    plan=json.loads((args.output/'plan.json').read_text());groups=defaultdict(Counter);types=defaultdict(Counter)
    for model in plan['models']:
        for task in plan['manifest']['tasks']:
            file=args.output/'results'/f"{model['id']}__{task}.json.gz"
            with gzip.open(file,'rt') as f:data=json.load(f)
            assert data['complete']
            key=(task,model['group']);counts=groups[key]
            for trajectory in data['rows']:
                counts['queries']+=1;counts['correct']+=trajectory['correct']
                if not trajectory['correct'] or trajectory['trace_valid']:continue
                counts['correct_invalid']+=1
                rr=trajectory['records']
                reached=any(r['at_goal'] for r in rr)
                found=any(r['at_goal'] and r['action_legal'] and not r['cancel'] for r in rr)
                if reached:
                    counts['reached_on_legal_execution']+=1
                    counts['legal_found']+=found
                elif trajectory['stopped_on_illegal_action']:
                    counts['illegal_before_proven_goal']+=1
                    counts['illegal_push']+=rr[-1]['illegal_push']
                    counts['illegal_push_from_candidates']+=rr[-1]['illegal_push_from_current_candidates']
                    for kind in rr[-1]['errors']:
                        if kind.startswith(('action_','format_')):types[key][kind]+=1
                else:counts['unproven_early_end']+=1
                counts['interrupted']+=trajectory['interrupted']
    for counts in groups.values():
        for name in ['correct_invalid','reached_on_legal_execution','legal_found','illegal_before_proven_goal','unproven_early_end','illegal_push','illegal_push_from_candidates','interrupted']:counts.setdefault(name,0)
        assert counts['correct_invalid']==counts['reached_on_legal_execution']+counts['illegal_before_proven_goal']+counts['unproven_early_end']
    result={task:{group:dict(counts,execution_failure_types=dict(types[task,group])) for (t,group),counts in groups.items() if t==task} for task in plan['manifest']['tasks']}
    (args.output/'invalid_correct_breakdown.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    text=['# 答案正确但CoT不合法的细分','','自然生成行为分类，不是纠正前缀的因果实验。','本cohort只有可达查询，不能排除YES偏置或独立答案通道。','']
    for task,models in result.items():
        text+=['## '+task,'','|模型|正确但CoT非法|合法执行已到目标|到目标前执行非法|未证实目标且提前结束|','|---|---:|---:|---:|---:|']
        for name,c in models.items():text.append(f"|{name}|{c['correct_invalid']}|{c['reached_on_legal_execution']}|{c['illegal_before_proven_goal']}|{c['unproven_early_end']}|")
        text+=['']
    text+=['合法执行到达目标不代表错误字段未影响搜索顺序或代价；执行非法后仍答对不证明错误对最终答案有因果影响，也不直接叫猜中。', '']
    (args.output/'INVALID_CORRECT_BREAKDOWN.md').write_text('\n'.join(text),encoding='utf-8')
    print(json.dumps(result['hard']['l9_p60']))


if __name__=='__main__':main()

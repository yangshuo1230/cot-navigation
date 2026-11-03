"""Summarize paired oracle/free inference and negative execution certificates."""
from pathlib import Path
from collections import Counter
import csv,gzip,json
ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results'

def ratio(n,d):return n/d if d else None
def aggregate(rows,meta):
    if meta['model']['protocol'] in ('dfs_nodes','bfs_nodes'):
        for r in rows:
            r.setdefault('execution_valid',r['complete_valid'])
    labels={y:[r for r in rows if r['label']==y] for y in (0,1)}
    oracle=[r for r in rows if not r['oracle_truncated']]
    matched={y:[r for r in oracle if r['label']==y] for y in (0,1)}
    neg=[r for r in labels[0] if r['free_correct']]
    yes_missed=[r for r in rows if r['label']==1 and r['free_pred']==0]
    supported=bool(rows[0]['negative_certificate_supported'])
    result={'model':meta['model']['id'],'case':meta['case'],'protocol':meta['model']['protocol'],
        'checkpoint':meta['model']['tag'],'n':len(rows),'oracle_untruncated_n':len(oracle),
        'free_accuracy':ratio(sum(r['free_correct'] for r in rows),len(rows)),
        'free_no_fraction':ratio(sum(r['free_pred']==0 for r in rows),len(rows)),
        'oracle_complete_only_accuracy':ratio(sum(r['oracle_correct'] for r in oracle),len(oracle)),
        'oracle_complete_format_fail_fraction':ratio(sum(r['oracle']['pred']<0 for r in oracle),len(oracle)),
        'oracle_complete_label_only_accuracy':ratio(sum(r['oracle']['label_only_pred']==r['label'] for r in oracle),len(oracle)),
        'oracle_truncated_n':len(rows)-len(oracle),'negative_correct_n':len(neg),
        'negative_certificate_supported':supported,
        'negative_correct_incomplete_fraction':ratio(sum(not r['complete_valid'] for r in neg),len(neg)) if supported else None,
        'negative_execution_audited_n':sum('execution_valid' in r for r in neg),
        'negative_correct_invalid_execution_fraction':ratio(sum(not r['execution_valid'] for r in neg if 'execution_valid' in r),sum('execution_valid' in r for r in neg)) if supported else None,
        'negative_correct_annotation_only_fraction':ratio(sum(r['execution_valid'] and not r['complete_valid'] for r in neg if 'execution_valid' in r),sum('execution_valid' in r for r in neg)) if supported else None,
        'negative_correct_stop_fraction':ratio(sum(r['stop'] for r in neg),len(neg)),
        'negative_correct_legal_prefix_incomplete_fraction':ratio(sum(r['prefix_valid'] and not r['complete_valid'] for r in neg),len(neg)) if supported else None,
        'negative_correct_missing_reachable_mentions_fraction':ratio(sum(not r['all_reachable_nodes_mentioned'] for r in neg),len(neg)) if supported else None,
        'positive_pred_no_n':len(yes_missed),
        'positive_pred_no_target_current_n':sum(r['target_current'] for r in yes_missed),
        'positive_pred_no_has_found_n':sum(r['has_found'] for r in yes_missed),
        'positive_pred_no_stop_n':sum(r['stop'] for r in yes_missed)}
    for y,name in ((0,'no'),(1,'yes')):
        a=labels[y];b=matched[y]
        result[f'free_{name}_recall']=ratio(sum(r['free_correct'] for r in a),len(a))
        result[f'matched_free_{name}_recall']=ratio(sum(r['free_correct'] for r in b),len(b))
        result[f'oracle_complete_{name}_recall']=ratio(sum(r['oracle_correct'] for r in b),len(b))
        result[f'oracle_label_only_{name}_recall']=ratio(sum(r['oracle']['label_only_pred']==y for r in b),len(b))
        result[f'oracle_complete_{name}_n']=len(b)
        errors=[r for r in b if not r['free_correct']]
        result[f'oracle_rescue_{name}_fraction']=ratio(sum(r['oracle_correct'] for r in errors),len(errors))
    result['negative_correct_first_errors']=dict(Counter(r['first_error']['kind'] for r in neg if r['first_error']))
    result['negative_correct_action_first_errors']=dict(Counter(r['action_replay']['first_error'] for r in neg
        if r.get('action_replay',{}).get('first_error')))
    result['negative_correct_error_queries']=dict(Counter(k for r in neg for k in set(r['errors'])))
    return result

def main():
    stats=[];examples=[]
    for p in sorted(RESULTS.glob('*/*.jsonl.gz')):
        meta=json.loads(p.with_suffix('').with_suffix('.json').read_text())
        with gzip.open(p,'rt') as f:rows=[json.loads(x) for x in f]
        stats.append(aggregate(rows,meta))
        chosen=[r for r in rows if r['label']==1 and r['free_pred']==0 and r['oracle_correct'] and not r['oracle_truncated']]
        chosen += [r for r in rows if r['label']==0 and r['free_correct'] and r['complete_valid'] is False]
        for r in chosen[:3]:examples.append({'model':meta['model']['id'],'case':meta['case'],**r})
    (ROOT/'summary.json').write_text(json.dumps(stats,indent=2))
    (ROOT/'examples.json').write_text(json.dumps(examples,indent=2))
    if not stats:return
    keys=[k for k in stats[0] if not k.endswith('errors') and k!='negative_correct_error_queries']
    with (ROOT/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,keys,extrasaction='ignore');writer.writeheader();writer.writerows(stats)
    def pct(x):return 'N/A' if x is None else f'{x*100:.1f}%'
    lines=['# NO偏向：正确参考轨迹与负例完整性诊断','','冻结推理，不训练、不创建checkpoint或W&B run。查询在TEST内按层差×标签分层，最多每格4条；同图设置各协议共享查询。老规模课程采用验证best，新loss/SWA五组采用最终checkpoint。','','## 正确参考轨迹下的答案读取','','完整未截断轨迹不包含答案或EOS，但状态协议包含FOUND/EXHAUST，结果只能说明给出正确计算及终止线索后能否读取答案，不代表独立完成推理。截断轨迹从以下oracle对比中排除；自由生成列也使用相同未截断子集。Direct无CoT，oracle只是原始prompt，不能称为有正确轨迹干预。','','|模型|条件|配对完整oracle正/负n|自由YES/NO召回|正确轨迹YES/NO召回|','|---|---|---|---|---|']
    for s in stats:
        lines.append(f"|{s['model']}|{s['case']}|{s['oracle_complete_yes_n']}/{s['oracle_complete_no_n']}|{pct(s['matched_free_yes_recall'])}/{pct(s['matched_free_no_recall'])}|{pct(s['oracle_complete_yes_recall'])}/{pct(s['oracle_complete_no_recall'])}|")
    lines+=['','## 负例答对时，过程是否完整','','分母是各条件中真实负例且自由生成答对的查询数。动作完整性要求忽略打印注释后仍能合法执行并穷尽搜索；完整状态还要求frontier、visited和候选正确。STOP按未完成处理，不自动解释为算法错误。Path只提供路径、不承诺穷尽；Direct没有过程，两者搜索完整性为N/A。','','|模型|条件|答对负例n|完整状态不合法|实际动作不完整/非法|仅状态注释错|其中STOP|','|---|---|---:|---:|---:|---:|---:|']
    for s in stats:
        lines.append(f"|{s['model']}|{s['case']}|{s['negative_correct_n']}|{pct(s['negative_correct_incomplete_fraction'])}|{pct(s['negative_correct_invalid_execution_fraction'])}|{pct(s['negative_correct_annotation_only_fraction'])}|{pct(s['negative_correct_stop_fraction'])}|")
    lines+=['','结果限于每格最多4条的单checkpoint诊断面板，不能替代完整测试准确率。状态首错之后的局部错误只是打印内容检查，不是有效执行历史；节点被提及也不证明曾合法扩展。可支持定位错误发生在过程还是最终答案，但不能单独证明NO偏向的唯一因果机制。','','[逐条件统计](summary.json)；[CSV](summary.csv)；[少量配对例子](examples.json)。']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('conditions',len(stats),'queries',sum(s['n'] for s in stats))

if __name__=='__main__':main()

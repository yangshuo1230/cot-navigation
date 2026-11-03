from pathlib import Path
from collections import Counter
import csv,gzip,json
ROOT=Path(__file__).resolve().parent

def mean(rows,key):return sum(bool(r[key]) for r in rows)/len(rows) if rows else None
def pct(x):return 'N/A' if x is None else f'{100*x:.2f}%'

def main():
    stats=[];examples=[]
    for p in sorted((ROOT/'results').glob('*/*.jsonl.gz')):
        meta=json.loads(p.with_suffix('').with_suffix('.json').read_text())
        with gzip.open(p,'rt') as f:rows=[json.loads(x) for x in f]
        for label in (None,1,0):
            part=rows if label is None else [r for r in rows if r['label']==label]
            action_right=[r for r in part if r['action_correct']]
            stat=dict(model=meta['model']['id'],case=meta['case'],label='all' if label is None else label,
                n=len(part),excluded_truncated=meta['excluded_truncated'],
                action_accuracy=mean(part,'action_correct'),exact_triplet_accuracy=mean(part,'exact_action_separator_answer'),
                answer_accuracy_32tokens=mean(part,'answer_correct'),
                answer_accuracy_given_right_action=mean(action_right,'answer_correct'),
                right_action_n=len(action_right),given_gold_terminal_answer_accuracy=mean(part,'gold_terminal_control_correct'),
                previous_free_answer_accuracy=mean(part,'prior_free_correct'),
                first_token_counts=dict(Counter(r['predicted_first_token'] for r in part)),
                format_fail_fraction=mean(part,'format_fail'))
            stats.append(stat)
        errors=[r for r in rows if not r['exact_action_separator_answer']]
        for r in errors[:4]:examples.append({'model':meta['model']['id'],'case':meta['case'],**r})
    (ROOT/'summary.json').write_text(json.dumps(stats,indent=2))
    (ROOT/'examples.json').write_text(json.dumps(examples,indent=2))
    if not stats:return
    with (ROOT/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,[k for k in stats[0] if k!='first_token_counts'],extrasaction='ignore');writer.writeheader();writer.writerows(stats)
    lines=['# 隐去终止动作的正确CoT诊断','','输入为正确完整轨迹直到最后状态的第三个竖线，去掉最后FOUND/EXHAUST及其分号，并且不输入YES/NO/EOS。模型全词表greedy续写，不强制动作、分号或答案。每例最多续写32 token；主要联合指标为前三个token恰好等于“正确动作 ; 正确答案”。完整参考过长的查询沿用上一轮排除，报告保留样本数。','','九个完整状态checkpoint：固定图DFS/BFS、新图DFS/BFS验证best，及新图DFS loss/SWA五组最终checkpoint；同一批TEST查询。','','|模型|条件|标签|n|首动作正确|动作+分号+答案全对|32token内答案对|动作正确后答案对|直接给正确终止动作后答案对|','|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in stats:
        lines.append('|'+ '|'.join([r['model'],r['case'],str(r['label']),str(r['n'])]+[pct(r[k]) for k in ('action_accuracy','exact_triplet_accuracy','answer_accuracy_32tokens','answer_accuracy_given_right_action','given_gold_terminal_answer_accuracy')])+'|')
    lines+=['','动作错后的32token内答案仅是有上限的续写结果，不能说明更长续写一定不会答对；三token联合指标没有这个歧义。前缀包含完整正确最终状态，是搜索终点处的局部判断诊断，不检验模型是否会独立到达这个状态。PATH、节点版和Direct没有训练过这些动作token，不纳入此项。','','[统计JSON](summary.json)；[CSV](summary.csv)；[错误例子](examples.json)。']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    allrows=[r for r in stats if r['label']=='all']
    print('conditions',len(allrows),'queries',sum(r['n'] for r in allrows))

if __name__=='__main__':main()

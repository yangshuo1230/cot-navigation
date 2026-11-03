"""Create complete/partial semantic-audit tables from existing JSON only."""
from pathlib import Path
import json
import argparse

HERE=Path(__file__).resolve().parent
P=['path','dfs_state','bfs_state','dfs_nodes','bfs_nodes']
S=['l4','l6','l8','l10','curriculum']
def pct(x):return f'{100*x:.2f}%'
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',
                      *['| '+' | '.join(map(str,row))+' |' for row in rows]])
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args()
    rows=[json.loads(p.read_text()) for p in (HERE/'results').rglob('*.json') if p.name!='complete.json']
    indexed={(r['job'],r['case']):r for r in rows}
    if not args.partial:
        assert len(rows)==84,len(rows)
        assert len(list((HERE/'results').rglob('complete.json')))==51
    main_finished=sum(r['case']=='main' for r in rows)
    text=['# 最新scale实验：CoT轨迹语义合法性检查',
          f'主分布完成 {main_finished}/51；主分布及扩展条件完成 {len(rows)}/84；所有条件完成的模型 {len(list((HERE/"results").rglob("complete.json")))}/51。'
          '主分布使用与原评估相同的全部查询（最多256）和batch16；N100静态、课程及固定迁移探针另测deep/wide/larger，每条件64条分层子集。',
          '冻结latest checkpoint、BF16、原4096上下文及原STOP策略。原模型文件检查前后SHA256一致。'
          '不创建W&B或训练checkpoint。语义验证器接受任意合法邻居优先级；已通过7680项4节点DAG穷举断言、STOP检查，以及49152个候选节点序列与全部参考邻居优先级的穷举等价检查。',
          '完整轨迹合法要求所有状态/动作正确并合法终止；STOP后的未完成搜索不算完整合法。'
          '前缀合法允许未完成但已输出部分必须全部正确，空STOP单列。动作骨架忽略显示状态，仍要求动作可执行且完整终止。'
          'Path负例随机walk只检查起点、真实边及sink终点，不构成NO证明，所以合法路径和正确答案必须分别报告。'
          'non_node_token表示序列包含非本图节点或语法token，并不只表示特殊token。']
    for setting,label in [('context','新图'),('fixed','固定图')]:
        result=[]
        for scale in S:
            row=[scale]
            for p in P:
                key=f'{setting}/{scale}/'+('stage_b/' if setting=='fixed' else '')+p
                r=indexed.get((key,'main'))
                row.append('待完成' if r is None else pct(r['answer_accuracy'])+' / '+pct(r['complete_trace_accuracy']))
            result.append(row)
        text += [f'## {label}：答案准确率 / 完整轨迹合法率',table(['训练',*P],result)]
    detail=[]
    for r in sorted(rows,key=lambda r:(r['job'],r['case'])):
        c=r['counts'];n=r['n'];interrupted=c['interrupted']
        detail.append([r['job'],r['case'],n,pct(r['answer_accuracy']),pct(r['complete_trace_accuracy']),pct(r['joint_accuracy']),
                       pct(r['skeleton_accuracy']),pct(r['prefix_valid_rate']),pct(interrupted/n),
                       '—' if not interrupted else pct(c['interrupted_prefix_valid']/interrupted),
                       str(c['old_prediction_matches'])+'/'+str(c['old_prediction_compared'])])
    text += ['## 全部分项',table(['模型','条件','n','答案','完整合法','答案且完整合法','动作骨架','前缀合法','STOP','STOP中前缀合法','原答案一致'],detail)]
    errors=[[r['job'],r['case'],json.dumps(r['errors'],ensure_ascii=False)] for r in rows]
    text += ['## 首错类别',table(['模型','条件','首次导致不合法的原因'],errors),
             '每个错误类别计该轨迹的首个失败原因；这不表示后续动作、状态一定正确。'
             '完整合法率低于答案准确率时，不能认为都是无关状态注释错误，还要看动作骨架。'
             'OOD为64例诊断子集，其答案不与此前256例总体直接作差；BF16重组batch也可能影响临界生成。']
    (HERE/'REPORT.md').write_text('\n\n'.join(text)+'\n',encoding='utf-8')
    (HERE/'summary.json').write_text(json.dumps([{k:v for k,v in r.items() if k!='examples'} for r in rows],ensure_ascii=False,indent=2))
    print('Report saved:',len(rows),'conditions')
if __name__=='__main__':main()

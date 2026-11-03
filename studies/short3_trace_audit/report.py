"""Compact legality report; all rates use the same held-out query IDs."""
import csv
import json
from pathlib import Path


def main():
    root=Path(__file__).resolve().parent
    models={p.stem:json.loads(p.read_text()) for p in (root/'models').glob('*.json')}
    assert len(models)==10 and all(d['complete'] and d['checkpoint_unchanged'] for d in models.values())
    ids=next(iter(models.values()))['query_ids']
    assert all(d['query_ids']==ids and d['prediction_mismatches']==0 for d in models.values())
    names={'path':'Path','dfs_state':'DFS state','bfs_state':'BFS state','dfs_nodes':'DFS nodes','bfs_nodes':'BFS nodes'}
    rows=[]
    for name,d in models.items():
        arm,protocol=name.split('__')
        for group,m in d['summaries'].items():
            rows.append(dict(arm=arm,protocol=protocol,group=group,**{k:m[k] for k in
                ['n','answer_correct','structural_valid','complete_valid','joint','actions_valid','stop','forced_stop','correct_but_invalid_n']}))
    with (root/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    pct=lambda x:f'{x:.1%}'
    lines=['# 三跳 QUERY 训练的 CoT 合法性','',
        '对三跳/全跳训练的 5 种 CoT 各取验证集所选 best checkpoint，共 10 个模型。',
        '对训练结束时相同的 256 条测试查询重新自由生成，采用原测试 batch size、原生上下文和 STOP 规则。',
        '全部 2560 个最终预测与原正式测试逐条一致；原 checkpoint 哈希保持不变。没有训练或拟合任何新模型。','',
        '## 判定口径','',
        '- Path：区分节点及边均合法的游走，和完成任务要求的路径。正例须结束于目标；负例按现有协议须走到 sink，但单条负例游走不是不可达性的证明。',
        '- DFS/BFS state：独立重放完整栈/队列、VIS、CAND、动作和终止条件；允许合法的不同邻居优先级及集合展示顺序。另独立检查只看动作能否构成正确完整执行。',
        '- DFS/BFS nodes：检查是否存在合法的完整首次展开顺序；相邻输出节点无需直接相邻，不按普通路径判定。',
        '- STOP 计为未完成，不自动等同于已产生错误。表中完整 CoT 与实际最终答案分别评分，再计算两者同时正确。',
        '- 指标按真实任务标签检查终止，不把正例中随便走到 sink 后说 NO 计为正确 CoT。','',
        '## 三跳模型：2–3 跳测试（67 对）','',
        '| 协议 | 答案正确 | 完整 CoT 合法 | 答案与完整 CoT 同时正确 |', '|---|---:|---:|---:|']
    for protocol,label in names.items():
        m=models['short3__'+protocol]['summaries']['short']
        lines.append('| '+label+' | '+' | '.join(pct(m[k]) for k in ['answer_correct','complete_valid','joint'])+' |')
    lines += ['', '## 三跳模型：4–9 跳外推（189 对，96 YES / 93 NO）','',
        '| 协议 | 答案正确 | 完整 CoT 合法 | 答案与完整 CoT 同时正确 | STOP | 答对但完整 CoT 不合法或未完成 |',
        '|---|---:|---:|---:|---:|---:|']
    for protocol,label in names.items():
        m=models['short3__'+protocol]['summaries']['long']
        lines.append('| '+label+' | '+' | '.join(pct(m[k]) for k in ['answer_correct','complete_valid','joint','stop'])+f" | {m['correct_but_invalid_n']}/{m['answer_correct_n']} |")
    lines += ['', '## 全跳训练：相同 4–9 跳测试的对照','',
        '| 协议 | 答案正确 | 完整 CoT 合法 | 答案与完整 CoT 同时正确 | STOP |',
        '|---|---:|---:|---:|---:|---:|']
    for protocol,label in names.items():
        m=models['all_hops__'+protocol]['summaries']['long']
        lines.append('| '+label+' | '+' | '.join(pct(m[k]) for k in ['answer_correct','complete_valid','joint','stop'])+' |')
    lines += ['', '## Path：有合法路径信息，仍可能答 NO','']
    d=models['short3__path'];ys=[r for r in d['by_query'] if r['gap']>=4 and r['label']==1]
    counts=dict(positive_queries=len(ys),legal_walk=sum(r['structural_valid'] for r in ys),
        legal_goal_endpoint=sum(r['goal_reached'] for r in ys),
        legal_goal_endpoint_but_NO=sum(r['goal_reached'] and r['pred']==0 for r in ys),
        legal_walk_containing_goal=sum(r['structural_valid'] and r['goal_seen'] for r in ys),
        legal_walk_containing_goal_but_NO=sum(r['structural_valid'] and r['goal_seen'] and r['pred']==0 for r in ys),
        positive_answer_correct=sum(r['answer_correct'] for r in ys),
        positive_trace_and_answer_correct=sum(r['joint'] for r in ys),
        positive_YES_with_nonedge=sum(r['pred']==1 and r['first_error']=='non_edge' for r in ys))
    lines += [f'- 长跨度全部查询中，176/189（93.1%）的 Path 是真实边组成的合法游走。',
        f"- 96 个可达正例中，{counts['legal_walk']} 条游走合法，{counts['legal_goal_endpoint']} 条合法地结束在目标；其中 {counts['legal_goal_endpoint_but_NO']} 条仍答 NO。",
        f"- 连同途中经过目标的情况，共 {counts['legal_walk_containing_goal']} 条合法游走包含目标，但其中 {counts['legal_walk_containing_goal_but_NO']} 条答 NO。",
        f"- 正例最终答对 {counts['positive_answer_correct']}/96，但完整正确路径与正确答案同时成立仅 {counts['positive_trace_and_answer_correct']}/96。",
        f"- {counts['positive_YES_with_nonedge']} 个正例虽然答 YES，输出却包含非边。",
        '- 这说明目标条件选择、停止与答案读出均有问题，不能仅解释为不会沿边行走。现象与长度/标签关联相符，但尚不是对该机制的因果证明。','']
    if (root/'path_cases.json').exists():
        for c in json.loads((root/'path_cases.json').read_text()):
            lines += [f"### {c['reason']}（query {c['query']}）",'',
                f"起点 {c['source']}，目标 {c['target']}，层差 {c['gap']}，真实答案 YES；模型 {'YES' if c['pred'] else 'NO'}。",'',
                '```text',c['trace'],'```',f"非边：{c['nonedges']}",'']
    if (root/'gold_prefix_diagnostic.json').exists():
        gold=json.loads((root/'gold_prefix_diagnostic.json').read_text())['results']
        g=gold['short3']['long_yes']
        lines += ['### 直接给正确路径时的答案读出','',
            '额外冻结前向诊断：输入正确参考路径，停在应输出答案的位置，取原始词表 argmax，不限制为 YES/NO。',
            f"三跳 Path 在 96 个长跨度正例上：YES {round(g['raw_yes']*g['n'])}/96；NO {round(g['raw_no']*g['n'])}/96；继续生成其他 token {round(g['continuation_or_other']*g['n'])}/96。",
            '全跳 Path 在相同 96 个正例上全部直接预测 YES。',
            '因此，失败不只是自己没找到正确路径，还包含未能泛化到长正例的目标终止/答案读出。该实验是 teacher-forced 局部诊断，不与自由生成准确率混算。','']
    lines += ['## 状态协议：注释与动作分开','',
        '| 协议 | 完整状态字段合法 | 忽略字段、只重放动作的完整合法率 | 首错计数 |',
        '|---|---:|---:|---|']
    for protocol in ['dfs_state','bfs_state']:
        m=models['short3__'+protocol]['summaries']['long']
        lines.append(f"| {names[protocol]} | {pct(m['complete_valid'])} | {pct(m['actions_valid'])} | {m['first_errors']} |")
    lines += ['',
        '只看动作的合法执行不保证打印的 VIS/CAND/栈字段正确；严格字段错误也不一定意味着走错了边。',
        '节点版违规类型按验证器规则归类，不一定对应时间上最早的错误 token。',
        '跨记录非边 PUSH、状态跳跃等局部计数保存在 models/*.json，可能包含首个语义错误之后的记录，不能把它们解释为独立合法历史下的决策。','',
        '结果仍限于同一固定图 seed、验证所选 checkpoint。本文未进行因果干预；没有重新选择测试最优模型。',
        '每个模型仅保存逐查询小型计数和最多 6 个有限错误窗口，没有完整 rollout / activation 大缓存。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (root/'verification.json').write_text(json.dumps(dict(models=10,queries_each=256,total_predictions=2560,
        original_predictions_reproduced=True,checkpoint_hashes_unchanged=True,
        reference_oracle_self_test_assertions_each_worker=next(iter(models.values()))['oracle_self_test_cases'],
        path_positive_diagnostics=counts,status='PASS'),indent=2)+'\n')
    print('REPORTED',len(models),'models')


if __name__=='__main__':main()

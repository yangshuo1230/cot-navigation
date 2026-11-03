"""Final-50k initialization means, explicit denominators and strict validity."""
import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
data = json.loads((HERE/'collected.json').read_text())
summary = {}
panels = ['short_test','long_AA_test','long_TA_test','long_AT_test','long_TT_test']
labels = dict(zip(panels,['短程','长程 AA','长程 TA','长程 AT','长程 TT']))
names = {'direct':'Direct','path':'PATH','dfs_state':'DFS State','dfs_nodes':'DFS Nodes'}
lines = ['# 2026-10-27：简单图剩余 DFS 与困难图训练结果', '',
         '已核对 COMPLETE、最终测试及末条训练记录：所有汇报均来自最后50,000步checkpoint；不是按测试成绩挑选种子或checkpoint。每种协议4个初始化，同一图seed0。简单图12/12完成，困难图16/16完成；调度器无失败、无活跃任务，8卡空闲。', '',
         '答案BA=(正例正确率+负例正确率)/2。下表先在每个初始化内计算，再对4个初始化等权平均；轨迹合法率的分母是正负例合计。节点留出是长程查询的端点角色留出，节点在短程/邻接知识/轨迹中仍可出现。', '']
for suite, group in data['suites'].items():
    summary[suite] = {}
    protocols = list(dict.fromkeys(r['protocol'] for r in group['runs']))
    for protocol in protocols:
        runs = [r for r in group['runs'] if r['protocol'] == protocol]
        assert len(runs) == 4 and all(r['COMPLETE']['step'] == r['final']['step'] == r['last_training']['optimizer_step'] == 50000 for r in runs)
        output = {'panels': {}, 'last_loss': {}, 'training_to_50k_hours': [r['last_training']['elapsed']/3600 for r in runs]}
        for key in ('loss/answer','loss/trace','loss/knowledge'):
            output['last_loss'][key] = [r['last_training'][key] for r in runs]
        for panel in panels:
            cells = [r['final']['panels'][panel]['metrics']['all'] for r in runs]
            assert len({(c['n'], c['n1'], c['n0']) for c in cells}) == 1
            entry = {'n':cells[0]['n'], 'positive_n':cells[0]['n1'], 'negative_n':cells[0]['n0']}
            for key in ('yes_recall','no_recall','ba','trace_valid_rate','positive_trace_valid_rate','negative_trace_valid_rate'):
                values = [c.get(key) for c in cells]
                if all(v is not None for v in values):
                    entry[key] = dict(mean=mean(values), min=min(values), max=max(values), by_seed=values)
            entry['truncated_per_seed'] = [c.get('truncated',0) for c in cells]
            entry['errors_per_seed'] = [r['final']['panels'][panel]['errors'] for r in runs]
            output['panels'][panel] = entry
        summary[suite][protocol] = output
    lines += ['## '+('简单图：10层×8节点、160条相邻层边' if suite=='simple' else '困难图：80节点、160条多尺度前向边'), '',
              '|测试组|'+ '|'.join(names[p]+' BA' for p in protocols)+'|正/负例数（每种子）|',
              '|---|'+ '|'.join('---:' for p in protocols)+'|---:|']
    for panel in panels:
        scores = '|'.join(f"{100*summary[suite][p]['panels'][panel]['ba']['mean']:.2f}%" for p in protocols)
        first = summary[suite][protocols[0]]['panels'][panel]
        lines.append(f"|{labels[panel]}|{scores}|{first['positive_n']}/{first['negative_n']}|")
    lines += ['', '|协议/组别|正例正确率|负例正确率|整段合法率|合法率：正例|合法率：负例|', '|---|---:|---:|---:|---:|---:|']
    for protocol in protocols:
        if protocol=='direct':
            selected = ['long_TT_test']
        elif suite=='simple' and protocol=='dfs_state':
            selected = panels
        else:
            selected = ['long_TT_test']
        for panel in selected:
            v = summary[suite][protocol]['panels'][panel]
            fields = ['yes_recall','no_recall','trace_valid_rate','positive_trace_valid_rate','negative_trace_valid_rate']
            cells = [f"{100*v[k]['mean']:.2f}%" if k in v else '—' for k in fields]
            lines.append(f"|{names[protocol]} / {labels[panel]}|"+'|'.join(cells)+'|')
    lines.append('')
lines += ['## 解读与边界', '',
    '简单图DFS State：TT最终答案BA为90.28%，整段状态合法率65.48%；目标端点留出的AT合法率也只有64.73%。所以最终答案正确不能等同于每一步正确执行DFS。', '',
    '困难图：TT的PATH答案BA95.57%、DFS State95.12%、DFS Nodes91.01%、Direct85.87%。PATH与State差距仅0.45个百分点，只有4个初始化且同一张图，不能据此确定稳定排名。State在TT的整段合法率76.47%，Nodes70.29%。', '',
    '困难图PATH合法性的含义：正例为合法且到达目标的路径；负例只要求合法游走，不是完整不可达证明。DFS State则验证每条stack/visited/candidates/action与完整终止，DFS Nodes重建隐式POP与终止，因此不同协议的合法率不能作为等价难度指标直接排名。', '',
    '困难图与简单图是不同图实例，查询及类别分母不同。困难图正例按真实最短跳数划分；负例按前向层带间距分桶。困难图的结构更复杂，不代表每个查询或测试集一定更难，不能仅凭跨图准确率比较判断图复杂性的影响。', '',
    '末条训练answer loss通常接近0，而测试仍有差距；这不能支持“只因为训练步数不足”。DFS轨迹loss按每例平均token计算，较低loss也不保证整段每一步都对。每个seed具体loss、截断计数、错误类别和验证曲线保存在JSON。', '',
    '训练到50k耗时包含周期验证、不含最后完整测试，也不含排队；详见summary。未重新训练、未新增checkpoint。']
(HERE/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2), encoding='utf-8')
(HERE/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
for suite in summary:
    for p,v in summary[suite].items():
        t = v['panels']['long_TT_test']
        print(suite,p,'TT BA range',t['ba'], 'TT positive valid',t.get('positive_trace_valid_rate'), 'TT negative valid',t.get('negative_trace_valid_rate'))

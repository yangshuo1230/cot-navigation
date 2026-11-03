"""Summaries separate complete state validity from sampled action execution."""
from pathlib import Path
import csv,json,gzip
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0,str(Path.cwd()/'src'))
from cot_navigation.config import load_config
from cot_navigation.data.dataset import ParametricWorld
from cot_navigation.protocols.tokenizer import Tokenizer

OUT=Path.cwd()/'studies/fixed100_legality'
PROTOCOLS=('path','dfs_nodes','bfs_nodes','dfs_state','bfs_state')
results={}
table=[]
for setting in ('simple','hard'):
    for protocol in PROTOCOLS:
        r=json.loads((OUT/f'{setting}_{protocol}.json').read_text());results[f'{setting}_{protocol}']=r
        full=r['summaries']['full_test_semantic']
        cfg=load_config([Path(r['run'])/'resolved_config.yaml']);tok=Tokenizer.from_config(cfg)
        world=ParametricWorld(Path(cfg.data.root)/'main-seed0');nodes=world.node_vocab_ids(tok)
        adjacency={nodes[u]:set(nodes[v] for v in vs) for u,vs in enumerate(world.adj())}
        detail_counts=Counter()
        order_counts=Counter()
        with gzip.open(OUT/f'{setting}_{protocol}_full_test_semantic.jsonl.gz','rt') as f:
            records=[json.loads(x) for x in f]
        def printed_valid(rec):
            a=rec['audit'];error=a['first_error']
            if a.get('prefix_valid_records',0)<1:return False
            if 'local_path_prefix_legal' in a:return bool(a['local_path_prefix_legal'])
            return error is None or error['kind'] in ('legal_prefix_stopped','missing_terminal','terminal_answer_mismatch')
        full['all']['printed_steps_semantic_valid']=sum(printed_valid(rec) for rec in records)
        full['all']['printed_steps_semantic_valid_rate']=full['all']['printed_steps_semantic_valid']/len(records)
        for rec in records:
            for key,value in rec['audit'].get('protocol_errors',{}).items():
                if value:order_counts[key+'_queries']+=1
            error=rec['audit']['first_error']
            if error and error['kind']=='candidate_mismatch':
                detail=error.get('detail',{})
                detail_counts['at_goal' if detail.get('current')==nodes[rec['target']] else 'before_goal']+=1
                if detail.get('missing'):detail_counts['missing_candidates']+=1
                if detail.get('extra'):detail_counts['extra_candidates']+=1
                extra=set(detail.get('extra',[]));current=detail.get('current')
                if extra-set(adjacency.get(current,())):detail_counts['extra_nonedge_candidates']+=1
                if extra&set(detail.get('visited',[])):detail_counts['extra_visited_candidates']+=1
                action=detail.get('shown_action',[])
                if len(action)==2 and action[0]==tok.id('<PUSH>'):
                    if action[1] not in adjacency.get(current,set()):detail_counts['nonedge_push_at_first_candidate_error']+=1
                    if action[1] in detail.get('visited',[]):detail_counts['revisited_push_at_first_candidate_error']+=1
        r['candidate_first_error_details']=dict(detail_counts)
        r['protocol_order_error_queries_in_checked_prefixes']=dict(order_counts)
        row=dict(setting=setting,protocol=protocol,checkpoint_step=r['checkpoint_step'],
            **{k:full['all'][k] for k in ('n','semantic_valid','protocol_valid','semantic_valid_rate','protocol_valid_rate','early_censored','interrupted')},
            positive_n=full['positive']['n'],positive_valid=full['positive']['protocol_valid'],
            negative_n=full['negative']['n'],negative_valid=full['negative']['protocol_valid'])
        row['printed_steps_valid']=full['all']['printed_steps_semantic_valid']
        row['printed_steps_valid_rate']=full['all']['printed_steps_semantic_valid_rate']
        table.append(row)
with (OUT/'legality_summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
write=lambda path,obj:path.write_text(json.dumps(obj,ensure_ascii=False,indent=2))
write(OUT/'summary.json',results)

fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
for ax,setting in zip(axes,('simple','hard')):
    x=np.arange(5)
    semantic=[100*results[f'{setting}_{p}']['summaries']['full_test_semantic']['all']['printed_steps_semantic_valid_rate'] for p in PROTOCOLS]
    strict=[100*results[f'{setting}_{p}']['summaries']['full_test_semantic']['all']['protocol_valid_rate'] for p in PROTOCOLS]
    ax.bar(x-.17,semantic,.34,label='All printed steps valid',color='#4c83b8')
    bars=ax.bar(x+.17,strict,.34,label='Complete protocol + answer',color='#d98c36')
    for rect,value in zip(bars,strict):ax.text(rect.get_x()+rect.get_width()/2,value+1,f'{value:.1f}',ha='center',fontsize=9)
    ax.set(title=f'{setting.title()} fixed DAG, N100',ylabel='Queries (%)',
        xticks=x,xticklabels=[p.replace('_','\n') for p in PROTOCOLS],ylim=(0,110))
    ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
fig.savefig(OUT/'complete_legality.png',dpi=180);plt.close(fig)

paragraphs=['# N100 Fixed DAG：逐步轨迹合法性审计','',
    '使用简单图 all_hops 与困难图的验证集 best checkpoint。所有模型冻结，不进行训练。',
    '主表覆盖完整正向测试池，旧图 1,091 对、新图 555 对；同图各协议使用相同 query ID。',
    '主表为首次确定语义错误即停止的严格审计。已证明非法的前缀不会因为后续输出恢复整条合法；',
    '提前停止行不猜测其最终答案，不将“未生成答案”记成 NO。合法率针对本次 audited greedy 轨迹。','',
    '## 全测试集结果','',
    '| 图 | 协议 | best step | 所有已输出步骤语义合法 | 完整轨迹及答案合法 | 比例 | 正例完整合法 | 负例完整合法 |',
    '|---|---|---:|---:|---:|---:|---:|---:|']
for r in table:
    paragraphs.append(f"| {r['setting']} | {r['protocol']} | {r['checkpoint_step']} | {r['printed_steps_valid']}/{r['n']} | {r['protocol_valid']}/{r['n']} | {100*r['protocol_valid_rate']:.1f}% | {r['positive_valid']}/{r['positive_n']} | {r['negative_valid']}/{r['negative_n']} |")
paragraphs+=['','## 核心发现','',
    '不能把 Path 的高合法性推广到所有 CoT。困难图 Path 的全部已输出路径均逐边合法，完整协议合法率为 551/555；',
    'DFS State 为 492/555，BFS State 为 526/555；DFS Node 为 500/555，BFS Node 为 396/555。',
    '困难 BFS Node 的首错包括 105 条不符合 BFS 顺序、39 条重复节点及 2 条到目标后仍继续输出；另外 13 条未完成。',
    '困难 DFS State 的首错包括 52 条候选集合错误、8 条动作错误、3 条栈状态错误；候选首错中 51 条发生在到达目标之前。',
    '在候选首错位置还可直接确认 6 条非边 PUSH 和 7 条重复访问 PUSH，这两类计数可能重叠。',
    '因此并非只有无关紧要的终点注释问题。','',
    '简单 DFS/BFS State 分别有 18/49 条记录始终合法但预算中止的轨迹；它们降低完整搜索率，不能算作某一步的语义错误。','',
    '![完整合法率](complete_legality.png)','',
    '## 判定标准','',
    '* Path：从正确起点出发，每步为真实边，无重复或非节点 token；正例实际到达目标；负例随机路径到汇节点；最终答案与任务标签一致。负例路径不是不可达证明。',
    '* DFS/BFS Node：验证是否存在与整个扩展序列一致的合法 DFS/BFS 遍历；不错误地要求回溯后或 BFS 相邻扩展节点之间也有边。',
    '* DFS/BFS State：从真实起点状态逐条重放。检查 frontier 顺序、visited 集合、候选集合、动作及更新、终止条件和答案。',
    '* 语义合法允许不同合法邻居顺序；严格协议另要求 visited 为发现顺序、PUSH 选择打印候选列表的第一项。','',
    '所有已输出步骤合法与完整搜索合法分开报告。STOP 后每条已打印记录可能都合法，但搜索未完成；STOP 是旧训练中允许的预算接口，不能把它说成某一步的图语义错误。',
    '空轨迹不计入“所有已输出步骤合法”的成功项。Path 的正确边序列未到目标、Node 的合法扩展尚未穷尽，也会出现前者高于后者。','',
    '## 首次失败类型','']
for setting in ('simple','hard'):
    for p in ('dfs_state','bfs_state'):
        r=results[f'{setting}_{p}'];errors=r['summaries']['full_test_semantic']['all']['first_errors']
        paragraphs.append(f"* {setting} {p}: "+', '.join(f'{k}={v}' for k,v in sorted(errors.items(),key=lambda z:-z[1])))
        paragraphs.append('  候选首错分解：'+json.dumps(r['candidate_first_error_details'],ensure_ascii=False))
        paragraphs.append('  已检查前缀内的协议排序错误查询数：'+json.dumps(r['protocol_order_error_queries_in_checked_prefixes'],ensure_ascii=False))
paragraphs+=['','## 动作重放诊断面板','',
    '在完整测试池中按 rank-gap × label 每桶固定抽取最多 4 条。该面板从一开始就另行运行，',
    '忽略状态注释错误继续生成，直到动作本身非法或完整结束。此处分母不同，不能与全测试主表直接相减。','',
    '| 图 | 协议 | 面板数 | 完整状态严格合法 | 忽略注释后的动作执行合法 |',
    '|---|---|---:|---:|---:|']
for setting in ('simple','hard'):
    for p in ('dfs_state','bfs_state'):
        s=results[f'{setting}_{p}']['summaries']['action_panel'];n=s['all']['n']
        paragraphs.append(f"| {setting} | {p} | {n} | {s['full_state_on_same_panel']['protocol_valid']}/{n} | {s['all']['protocol_valid']}/{n} |")
paragraphs+=['','## 实际失败记录','']
for setting in ('simple','hard'):
    for p in ('dfs_state','bfs_state'):
        r=results[f'{setting}_{p}'];examples=[e for e in r['examples'] if e['mode']=='full_test_semantic']
        if not examples:continue
        e=examples[0]
        paragraphs += [f"### {setting} {p}，query={e['query_id']}，{e['source']} → {e['target']}",'',
            '首错：`'+json.dumps(e['first_error'],ensure_ascii=False)+'`','',
            '```text','前一条：'+e['previous_record'],'出错条：'+e['bad_record'],'```','']
if (OUT/'nonedge_push_example.json').exists():
    ex=json.loads((OUT/'nonedge_push_example.json').read_text())
    paragraphs+=['## 已确认的非边 PUSH 实例','',
        f"困难 DFS State，query={ex['query_id']}，查询 {ex['source']} → {ex['target']}，第 {ex['first_error_record_1based']} 条记录：",'',
        '```text',f"当前节点：{ex['current']}",
        '真实出边：'+', '.join(ex['actual_successors']),
        '真实未访问后继：'+(', '.join(ex['expected_unvisited']) or '空'),
        '模型候选：'+', '.join(ex['shown_candidates']),
        '模型动作：PUSH '+ex['push'],'```','',
        f"图中不存在 {ex['current']} → {ex['push']}。本例在非法动作后停止，不推断它后续的最终答案。",'']
paragraphs+=['## 验证、效率及解释边界','',
    '检查器通过 1,536 个全部 4 节点 DAG 的参考查询/搜索案例，并测试错误答案、候选缺失与非边 PUSH。',
    '另与独立 Node 扩展序列判定器进行了 24,576 个小图语言案例的等价性检查；不把合法回溯或 BFS 兄弟跳转误判为路径非边。',
    '与原生 BF16 greedy 的前缀对照覆盖四个 state checkpoint，共八个查询。',
    '仅在审计进程中将 KV 访问裁到可见前缀，去掉原实现中被遮罩的未来缓存槽；权重、上下文上限和训练代码不变。',
    '优化后的实际生成前缀与上述原生对照一致。本次批量为 32，原生训练末评测常为 16；浮点内核与分批可能影响少数近似并列 token。',
    '这些测试是执行合法性审计，不是与某一参考字符串比较，也不是因果证明模型真的用 CoT 计算答案。',
    '简单图与困难图的训练数据、WORLD 前缀和 checkpoint 步数不同；困难 BFS State 可能仍在训练，使用本次读入时的 best 权重。',
    '所有 checkpoint 从一次完整文件读取中冻结，SHA256、步数及逐查询首错都已保存。',
    '提前停止导致后续错误分布不可知；不报告被截断样本的全轨迹逐记录错误率，以避免产生虚假的高“每步正确率”。']
(OUT/'REPORT.md').write_text('\n'.join(paragraphs)+'\n')
print('Report, CSV, summary and key figure saved.')

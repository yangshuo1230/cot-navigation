"""Build one report and one key figure from compact, reproducible audit data."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

HERE = Path(__file__).resolve().parent
doc = json.loads((HERE/'results.json').read_text())
assert doc['version']==2 and len(doc['seeds'])==4
control = json.loads((HERE/'reference_control.json').read_text())
histories = json.loads((HERE/'history_diversity.json').read_text())
official = json.loads((HERE.parent/'navigation_results_check/summary.json').read_text())['simple']
world = json.loads((HERE.parent/'fixed_navigation_v5/data/n80_g0.json').read_text())
panels = ['short_test','long_AA_test','long_TA_test','long_AT_test','long_TT_test']
labels = ['短程','长程 AA','长程 TA','长程 AT','长程 TT']
seeds = doc['seeds']
first = [dict(seed=s['seed'],**f) for s in seeds for f in s['first_failures']]
candidate = [f for f in first if f['kind']=='candidate_error']
flags = Counter(k for f in candidate for k,v in f['flags'].items() if v)
actions = Counter(f['expected_action']+'->'+f['shown_action'] for f in first if f['kind']=='illegal_action')
phase = {}
for p in ['root','fresh_pushed','revisit']:
    phase[p] = {k:sum(x['candidate_phase'][p][k] for s in seeds for x in s['panels'].values()) for k in ['n','bad']}
    phase[p]['nonempty'] = {k:sum(x['candidate_phase_nonempty'][p][k] for s in seeds for x in s['panels'].values()) for k in ['n','bad']}
by_panel = {}
for panel in panels:
    fs = [f for f in first if f['panel']==panel]
    cs = [f for f in fs if f['kind']=='candidate_error']
    by_panel[panel] = dict(first_errors=dict(Counter(f['kind'] for f in fs)),
                          candidate_flags=dict(Counter(k for f in cs for k,v in f['flags'].items() if v)),
                          candidate_at_goal=sum(f['at_goal'] for f in cs),
                          illegal_action_pairs=dict(Counter(f['expected_action']+'->'+f['shown_action'] for f in fs if f['kind']=='illegal_action')))
reach = []
for source in range(world['n_nodes']):
    seen = {source}
    todo = [source]
    for u in todo:
        for v in world['adj'][u]:
            if v not in seen:
                seen.add(v)
                todo.append(v)
    reach.append(seen)
missing_useful = sum(any(f['target'] in reach[v] for v in set(f['expected'])-set(f['shown'])) for f in candidate)
patterns = {k:sum(bool(f[k]) for f in candidate) for k in ['at_goal','stale_previous_list','full_unfiltered_adjacency','missing_target','missing_all_eligible']}
patterns['empty_instead_nonempty']=sum(bool(f['expected']) and not f['shown'] for f in candidate)
patterns['empty_instead_nonempty_not_at_goal']=sum(bool(f['expected']) and not f['shown'] and not f['at_goal'] for f in candidate)
patterns['partial_omission_without_extra_or_duplicate']=sum(f['flags']['missing'] and bool(f['shown']) and not any(v for k,v in f['flags'].items() if k!='missing') for f in candidate)
summary = dict(n_generated=4*sum(official['dfs_state']['panels'][p]['n'] for p in panels),
               first_error_counts=dict(Counter(f['kind'] for f in first)),candidate_n=len(candidate),
               candidate_flags=dict(flags),candidate_patterns=patterns,
               missing_goal_reachable_successor=missing_useful,
               candidate_answer_correct=sum(f['answer_correct'] for f in candidate),
               candidate_phase=phase,action_errors=dict(actions),by_panel=by_panel,
               training=seeds[0]['train'],training_uniform_control=control,training_histories=histories,
               greedy_diversity=[{k:v for k,v in s['greedy_root_diversity'].items() if k!='per_source'} for s in seeds],
               greedy_negative_diversity=[s['greedy_negative_root_diversity'] for s in seeds],
               original_count_differences=[dict(seed=s['seed'],panels={p:x['original_count_differences'] for p,x in s['panels'].items() if x['original_count_differences']}) for s in seeds])
(HERE/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')


def pct(x):
    return f'{100*x:.2f}%'


def node(v):
    return f'V{world["node_names"][v]}' if v>=0 else f'token_id={-v-1}'


def names(values):
    return '['+', '.join(map(node,values))+']'


lines = ['# 简单图 v6 Fixed DFS：训练结果、顺序多样性与候选错误', '',
    '分析四个初始化 seed0–3 的固定 final50k checkpoint；同一张10层×8节点、160边图。不是四张独立图，也没有用测试挑选 checkpoint。', '',
    '## 训练成绩（原始最终测试）', '',
    'A 为长程训练端点，T 为长程测试留出端点；T 仍可出现在短程查询、邻接知识和搜索内部。', '',
    '|面板|Direct BA|DFS State BA|DFS Nodes BA|State 整段合法率|State 正例合法率|State 负例合法率|',
    '|---|---:|---:|---:|---:|---:|---:|']
for p,label in zip(panels,labels):
    s = official['dfs_state']['panels'][p]
    vals = [official[k]['panels'][p]['ba']['mean'] for k in ['direct','dfs_state','dfs_nodes']]
    vals += [s[k]['mean'] for k in ['trace_valid_rate','positive_trace_valid_rate','negative_trace_valid_rate']]
    lines.append('|'+label+'|'+'|'.join(map(pct,vals))+'|')
lines += ['', '## 训练随机化与模型行为', '',
    f"实际曝光 {seeds[0]['train']['queries']} 个训练查询，每个8变体，共 {seeds[0]['train']['actual_exposed_variants']} 条；四个初始化均完整见过这批变体。每次离线生成先独立打乱各节点的邻接顺序，再执行确定的DFS。训练时随机选冻结变体，不是每次产生全新搜索。",
    f"每查询平均 {seeds[0]['train']['mean_unique_node_orders_per_query']:.2f}/8 种不同节点展开序列；{pct(seeds[0]['train']['fraction_queries_multiple_node_orders'])} 的查询有多种序列。部分查询没有分支或很快到目标，不能要求8种全不相同。",
    f"起点有多分支的 {seeds[0]['train']['root_branch_queries']} 个查询中，{pct(seeds[0]['train']['root_branch_multiple_first_fraction'])} 在8变体里选过不同首分支；归一化首分支熵 {control['observed_train_normalized_entropy']:.4f}，均匀独立采样8次的有限样本期望为 {control['expected_iid_uniform_eight_draw_entropy']:.4f}。这支持本轮局部顺序随机化有效，但不代表穷尽所有合法搜索状态。", '',
    f"不仅是顺序换位：排除前两个状态及到目标的状态，实际训练库中有 {histories['distinct_ordered_stacks']} 种有序栈，其中 {histories['stacks_multiple_visited']} 种（{pct(histories['stacks_multiple_visited']/histories['distinct_ordered_stacks'])}）对应多个不同visited集合，{histories['stacks_multiple_eligible_sets']} 种（{pct(histories['stacks_multiple_eligible_sets']/histories['distinct_ordered_stacks'])}）对应多个不同合法候选集合。共 {histories['distinct_stack_visited_states']} 种不同的栈＋visited状态，说明训练确实包含相同栈但需要不同历史处理的例子；尚不能因此证明模型使用了这些区别。", '',
    '测试使用原生贪心解码，同一输入并非随机采样。以下只衡量换目标/换查询后，同一起点是否始终固定排序；每起点至少4条有效根候选记录，按起点等权。', '',
    '|初始化|符合支持条件的起点数|使用过多个首分支的起点比例|首分支归一化熵|最常用首分支占比|',
    '|---|---:|---:|---:|---:|']
for s in seeds:
    d=s['greedy_root_diversity']
    lines.append(f"|{s['seed']}|{d['sources_with_at_least4_queries']}|{pct(d['fraction_sources_multiple_first'])}|{d['mean_normalized_entropy']:.3f}|{pct(d['mean_dominant_choice_fraction'])}|")
lines += ['', '模型没有普遍坍缩成每个起点唯一首分支，但也没有理由要求贪心输出均匀随机。跨目标多样性不证明模型在同一查询上具有采样熵，也不排除目标条件模板记忆。', '',
    '|面板|模型整段状态体匹配训练库|随机合法参考匹配训练库|模型节点展开序列匹配训练库|',
    '|---|---:|---:|---:|']
for p,label in zip(panels,labels):
    n = official['dfs_state']['panels'][p]['n']
    model_body=np.mean([s['panels'][p]['matches'].get('body_exact',0)/n for s in seeds])
    model_order=np.mean([s['panels'][p]['matches'].get('push_order_exact',0)/n for s in seeds])
    ref=control['panels'][p]
    lines.append(f"|{label}|{pct(model_body)}|{pct(ref['body_exact']/ref['n'])}|{pct(model_order)}|")
lines += ['', '匹配时去掉QUERY文本与最终答案，只比完整搜索体或节点展开顺序；训练库仅含实际曝光变体。随机合法参考也经常匹配，不能把匹配自动解释为抄袭。反过来，低整段匹配也不能排除局部记忆。参考与模型的长度/偏好没有配平，该对照是描述性的。', '',
    '## 候选错误：首个错误之前的完整状态重放', '',
    f"重新自由生成 {summary['n_generated']} 条；每条最多归因一个首错，不统计错误之后的连锁后果。独立重放验证stack、visited有序历史、候选全集、首候选PUSH和终止。共 {len(first)} 条有语义首错，其中候选字段 {len(candidate)} 条（占失败 {pct(len(candidate)/len(first))}，占全部查询 {pct(len(candidate)/summary['n_generated'])}）。候选顺序任意合法都接受，重复不接受。", '',
    '|候选首错类别（可重叠）|条数|占候选首错|',
    '|---|---:|---:|']
flag_names={'missing':'漏掉合法未访问后继','extra_non_neighbor':'生成非直接后继','extra_visited_neighbor':'重新加入已访问的真实后继','duplicate':'重复候选','outside_graph':'本图外节点／非节点协议符混入','unclosed':'候选字段未闭合'}
for k,label in flag_names.items():
    lines.append(f'|{label}|{flags[k]}|{pct(flags[k]/len(candidate))}|')
lines += ['',
    f"其中 {patterns['at_goal']} 条发生在已经到达目标的状态。本训练协议在FOUND记录中也要求打印完整候选集，因此这部分不能简单当成实际未能寻路；它是严格状态协议失败。",
    f"有 {patterns['empty_instead_nonempty']} 条把非空合法集合直接输出为空，其中 {patterns['empty_instead_nonempty_not_at_goal']} 条当前尚未到达目标；另有 {patterns['partial_omission_without_extra_or_duplicate']} 条是只漏部分候选、没有多余/重复。'过早清空'是独立于visited误加的重要行为模式，但输出形态本身不能证明内部是过滤、复制指针还是集合检索失败。",
    f"{patterns['missing_target']} 条漏掉的候选就是目标本身；{missing_useful} 条漏掉至少一个能通向目标的合法后继（包含目标自身），因此部分漏项具有影响寻路的条件。这个计数不是证明最终失败由该漏项导致。",
    f"回访时原封不动重复该节点上一候选列表的失败有 {patterns['stale_previous_list']} 条；直接输出未过滤的完整邻接集合有 {patterns['full_unfiltered_adjacency']} 条。两项可重叠，仅为行为签名，不能自动定位内部回路。",
    f"候选首错发生后，最终答案仍正确 {summary['candidate_answer_correct']}/{len(candidate)} 条。", '',
    '|阶段|首错候选数 / 候选生成机会|非空候选下：错误 / 机会|',
    '|---|---:|---:|']
for p,label in [('root','查询根状态'),('fresh_pushed','新节点首次展开，不含根'),('revisit','回退后再次展开')]:
    z=phase[p]; nz=z['nonempty']
    lines.append(f"|{label}|{z['bad']}/{z['n']} = {pct(z['bad']/z['n'])}|{nz['bad']}/{nz['n']} = {pct(nz['bad']/nz['n'])}|")
lines += ['', '机会只来自语义正确前缀。不同阶段的状态难度和生存选择不同，不能将这些比例直接解释为阶段的因果作用。新节点也可能已有后继被其他分支访问，首次展开不等于不需要visited过滤。', '',
    '## 其他首错与长程目标泛化', '',
    '|首错类别|条数|', '|---|---:|---:|']
for k,v in Counter(summary['first_error_counts']).most_common():
    lines.append(f'|{k}|{v}|')
lines += ['', '动作错误细分（此前候选已验证正确）：'+json.dumps(dict(actions),ensure_ascii=False)+'。', '',
    '不能把整段不合法全归给候选：特别应将到目标后未FOUND、未到目标却FOUND、stack/visited打印错误单独区分。逐面板动作细分在summary.json中。', '',
    '## 可复现的实际候选错误', '']
for key in ['missing','extra_visited_neighbor','extra_non_neighbor','duplicate']:
    eligible=[f for f in candidate if f['flags'].get(key) and not f['at_goal']]
    if not eligible:
        continue
    # Prefer a single, small, unambiguous error over an enormous state dump.
    f=min(eligible,key=lambda f:(sum(f['flags'].values())>1,len(f['visited']),f['record']))
    lines += [f"### {flag_names[key]}：seed{f['seed']} / {f['panel']} / query{f['query']}", '',
        f"查询 {node(world['records'][f['query']]['s'])} → {node(f['target'])}；第{f['record']}个状态，当前 {node(f['current'])}，阶段 {f['phase']}。",
        f"visited：`{names(f['visited'])}`。",
        f"应有候选：`{names(f['expected'])}`；实际：`{names(f['shown'])}`。", '']
lines += ['## 验证与边界', '',
    '全部2408条生成的整段有效/无效判断与独立原始replay_state一致；额外200条冻结参考轨迹和人工DAG的漏项、重复、非边、首次展开已有visited后继等案例通过。',
    '分析使用GPU0、BF16、compile，解码batch64（原最终测试State为16）；记录了与原final.json的逐计数差异。不同批形状可能通过BF16数值路径改变接近决策边界的贪心分支；本次有1条到达生成上限，已计为失败。因此训练成绩采用原始最终测试，候选细分采用本次实际生成，不宣称逐token复现。',
    '本轮限制的是长程端点角色及query曝光。先前曝光审计显示全部测试正例的发现树关系都被训练搜索语义覆盖，因此不是严格未见witness实验。',
    '未训练新模型、未保存activation或完整轨迹、未新增checkpoint和W&B run；只保存紧凑首错状态、聚合JSON和一张关键图。']
(HERE/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

font_path=Path('C:/Windows/Fonts/msyh.ttc')
if font_path.exists():
    plt.rcParams['font.family']=FontProperties(fname=str(font_path)).get_name()
plt.rcParams['axes.unicode_minus']=False
fig,ax=plt.subplots(2,2,figsize=(13,9),layout='constrained')
x=np.arange(5)
for k,style in [('direct',dict(marker='o')),('dfs_state',dict(marker='s')),('dfs_nodes',dict(marker='^'))]:
    ax[0,0].plot(x,[100*official[k]['panels'][p]['ba']['mean'] for p in panels],label=k,**style)
ax[0,0].plot(x,[100*official['dfs_state']['panels'][p]['trace_valid_rate']['mean'] for p in panels],':s',label='State 整段合法')
ax[0,0].set(xticks=x,xticklabels=labels,ylim=(40,102),ylabel='%',title='原始最终测试：四初始化平均')
ax[0,0].legend(fontsize=8)
shown=['missing','extra_non_neighbor','extra_visited_neighbor','duplicate']
ax[0,1].barh([flag_names[k] for k in shown],[100*flags[k]/len(candidate) for k in shown],color=['#df8c45','#9467bd','#4d9f91','#6686bf'])
ax[0,1].invert_yaxis()
ax[0,1].set(xlabel='占候选首错比例（%，类别可重叠）',title=f'本次候选首错 {len(candidate)} 条')
for i,k in enumerate(shown):
    ax[0,1].text(100*flags[k]/len(candidate)+1,i,str(flags[k]),va='center')
err=Counter(summary['first_error_counts'])
error_labels={'candidate_error':'候选错误','illegal_action':'动作错误','visited_mismatch':'visited 错误','stack_mismatch':'stack 错误','bad_action':'动作格式错误','push_not_first_candidate':'PUSH 与首候选不符','missing_terminal':'无终止','missing_record_end':'记录未闭合'}
items=err.most_common(6)
ax[1,0].barh([error_labels.get(k,k) for k,v in items],[v for k,v in items],color='#638fa8')
ax[1,0].invert_yaxis()
ax[1,0].set(xlabel='查询数，每条只计首错',title='不能把所有 DFS 不合法归因于候选')
xs=np.arange(4)
ax[1,1].bar(xs,[100*s['greedy_root_diversity']['fraction_sources_multiple_first'] for s in seeds],label='起点使用多个首分支',color='#4d9f91')
ax[1,1].set(xticks=xs,xticklabels=['seed'+str(s['seed']) for s in seeds],ylim=(0,105),ylabel='起点比例（%）',title='换目标查询后：贪心搜索未统一固定首分支')
ax[1,1].text(.03,.04,'每起点至少4条有效根候选记录\n跨查询多样性 ≠ 同一查询随机采样',transform=ax[1,1].transAxes,fontsize=9)
fig.suptitle('N80 简单 Fixed DAG · DFS State · 固定 final50k',fontsize=15)
fig.savefig(HERE/'audit.png',dpi=170)
fig.savefig(HERE/'audit.pdf')
print(json.dumps({k:v for k,v in summary.items() if k not in ['by_panel','training_uniform_control','training_histories','original_count_differences']},ensure_ascii=False,indent=2))

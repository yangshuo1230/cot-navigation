"""Summarize original splits, endpoint roles and the later probe split."""
import json,gzip
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parent
doc=json.loads((ROOT/'results.json').read_text())
rows=[]
for s in doc['seeds']:
    with gzip.open(ROOT/f"seed{s['seed']}_queries.jsonl.gz",'rt',encoding='utf-8') as f:rows.extend(json.loads(line) for line in f)
worldpath=ROOT/'world.json'
if not worldpath.exists():worldpath=ROOT.parent/'fixed_navigation_v5/data/n80_g0.json'
world=json.loads(worldpath.read_text())

GOAL={'continued_after_goal','wrong_answer_after_goal','missing_answer_after_goal'}
EARLY={'premature_yes','premature_no','premature_eos','missing_answer_before_completion'}
CAT=['valid','non_edge_jump','skipped_live_frontier','revisited_node','goal_stop','early_stop','other']
NAMES=['完整协议合法','非边跳转','跳过未探索分支','重复访问','目标停止错误','过早结束','其他']
COLORS=['#4f9b79','#3e7fb0','#5bb5b0','#e5ab42','#d75f59','#9876ac','#aab0b6']

def category(error):
    return 'goal_stop' if error in GOAL else 'early_stop' if error in EARLY else error if error in CAT else 'other'


def aggregate(rr):
    if not rr:return {'n':0}
    errors=Counter(r['error'] for r in rr);groups=Counter(category(r['error']) for r in rr)
    opp=Counter()
    for r in rr:opp.update(r['opportunities'])
    yes=[r for r in rr if r['label']];no=[r for r in rr if not r['label']]
    gr=opp['goal_reached'];ex=opp['search_exhausted']
    goal_rows=[r for r in rr if r['error']=='continued_after_goal']
    return dict(n=len(rr),n_unique_queries=len({r['query'] for r in rr}),errors=dict(errors),groups=dict(groups),
                protocol_valid_rate=sum(r['protocol_valid'] for r in rr)/len(rr),
                semantic_valid_rate=sum(r['semantic_valid'] for r in rr)/len(rr),
                answer_accuracy=sum(r['answer_correct'] for r in rr)/len(rr),
                answer_BA=.5*(np.mean([r['answer_correct'] for r in yes])+np.mean([r['answer_correct'] for r in no])) if yes and no else None,
                positive_valid_rate=np.mean([r['protocol_valid'] for r in yes]) if yes else None,
                negative_valid_rate=np.mean([r['protocol_valid'] for r in no]) if no else None,
                answer_correct_but_protocol_invalid=sum(r['answer_correct'] and not r['protocol_valid'] for r in rr),
                opportunities=dict(opp),goal_fail_n=gr-opp['goal_stopped_correctly'],
                goal_fail_rate=(gr-opp['goal_stopped_correctly'])/gr if gr else None,
                exhaust_fail_n=ex-opp['exhaust_stopped_correctly'],
                exhaust_fail_rate=(ex-opp['exhaust_stopped_correctly'])/ex if ex else None,
                goal_continuations_following_DFS=sum(r['detail'].get('follows_DFS_if_goal_ignored',False) for r in goal_rows),
                goal_continuations_n=len(goal_rows),
                goal_continuations_answer_correct=sum(r['answer_correct'] for r in goal_rows),
                repeat_subtypes=dict(Counter(r['detail'].get('repeat_subtype') for r in rr if r['error']=='revisited_node')),
                revisited_true_neighbor_count=sum(r['error']=='revisited_node' and r['detail'].get('edge_from_decision_top',False) for r in rr),
                jump_subtypes=dict(Counter(r['detail'].get('jump_subtype') for r in rr if r['error']=='non_edge_jump')),
                non_edge_jumps_directly_to_target=sum(r['error']=='non_edge_jump' and r['detail'].get('actual_is_target',False) for r in rr),
                skipped_frontier_directly_to_target=sum(r['error']=='skipped_live_frontier' and r['detail'].get('actual_is_target',False) for r in rr),
                truncated_n=sum(r['truncated'] for r in rr))


panels=list(doc['seeds'][0]['panels'])
summary={'original_panels':{},'endpoint_roles':{},'probe_split':{},'main_split':{}}
for panel in panels:summary['original_panels'][panel]=aggregate([r for r in rows if r['panel']==panel])
for role in ['short','AA','VA','AV','VV','TA','AT','TT']:summary['endpoint_roles'][role]=aggregate([r for r in rows if r['role']==role])
for split in ['train','val','test']:
    summary['probe_split'][split]=aggregate([r for r in rows if r['probe_split']==split])
    summary['main_split'][split]=aggregate([r for r in rows if r['main_split']==split])
summary['overall']=aggregate(rows)
(ROOT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')

font=Path('C:/Windows/Fonts/msyh.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.size':10,'axes.unicode_minus':False,'pdf.fonttype':42})
chosen=['short_train','long_train','short_test','long_AA_test','long_TA_test','long_AT_test','long_TT_test']
labels=['短程训练','长程 A→A 训练','短程测试','长程 A→A 测试','长程 T→A 测试','长程 A→T 测试','长程 T→T 测试']
fig,axes=plt.subplots(1,2,figsize=(14,7.3),gridspec_kw={'width_ratios':[1.65,1]},sharey=True)
fig.subplots_adjust(left=.19,right=.97,top=.76,bottom=.27,wspace=.14)
yy=np.arange(len(chosen));left=np.zeros(len(chosen))
for cat,label,color in zip(CAT,NAMES,COLORS):
    values=np.array([100*summary['original_panels'][p]['groups'].get(cat,0)/summary['original_panels'][p]['n'] for p in chosen])
    axes[0].barh(yy,values,left=left,color=color,label=label,height=.68)
    if cat=='valid':
        for y,v in zip(yy,values):axes[0].text(v/2,y,f'{v:.1f}%',va='center',ha='center',color='white',fontsize=10,weight='bold')
    left+=values
axes[0].set_yticks(yy,[f"{l}  n={summary['original_panels'][p]['n_unique_queries']}" for p,l in zip(chosen,labels)])
axes[0].invert_yaxis();axes[0].set_xlim(0,100);axes[0].set_xticks([0,25,50,75,100])
axes[0].set_xlabel('查询占比（%）；首错互斥分类')
axes[0].set_title('整条协议成功率与首次违规构成',pad=14,fontsize=13)
for shift,key,color,label in [(-.17,'goal','#d75f59','已到目标后未正确YES停止'),(.17,'exhaust','#3e7fb0','已穷尽后未正确NO停止')]:
    rates=[]
    for p in chosen:
        per_seed=[]
        for seed in range(len(doc['seeds'])):
            a=aggregate([r for r in rows if r['panel']==p and r['seed']==seed])
            if a[key+'_fail_rate'] is not None:per_seed.append(a[key+'_fail_rate'])
        rates.append(100*np.mean(per_seed) if per_seed else np.nan)
    axes[1].barh(yy+shift,rates,height=.29,color=color,label=label)
    for y,rate in zip(yy+shift,rates):
        if not np.isnan(rate):axes[1].text(rate+.35,y,f'{rate:.1f}%',va='center',fontsize=9)
axes[1].set_xlim(0,max(30,max(rates,default=0)+4));axes[1].set_xlabel('终止条件满足后的条件出错率（%）')
axes[1].set_title('区分目标检测与搜索穷尽后的终止',pad=14,fontsize=13)
for ax in axes:
    ax.spines[['top','right']].set_visible(False);ax.set_axisbelow(True);ax.grid(axis='x',alpha=.13)
handles,legend=axes[0].get_legend_handles_labels()
fig.legend(handles,legend,loc='upper center',bbox_to_anchor=(.56,.855),ncol=7,frameon=False,fontsize=9)
axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.15),frameon=False,fontsize=8)
fig.text(.045,.95,'DFS nodes 首错分析：查询迁移与目标停止',fontsize=20,weight='bold',color='#17324d')
fig.text(.045,.895,'v6 · N80 单图 · 4个 final50k · 全部2712查询×4 · 原生自由生成，首次违规之后不再归因',fontsize=11,color='#536575')
fig.text(.045,.13,'左：每条query只计首次违规，完整协议合法还要求最终答案正确。右：仅统计此前始终合法且已满足终止条件的机会。\n'
         '训练与测试分组来自导航模型原始划分；A为长程训练端点，T只留出长程端点角色，并非从未出现的节点。',fontsize=10,color='#536575',linespacing=1.8)
fig.text(.045,.035,'行为类型不是内部机制的因果证明。右图为4初始化条件错误率均值；同一查询跨seed复用，不作独立样本显著性解释。\n'
         '隐式合法回溯被接受；查询数n为每seed支持。新生成与历史BF16汇总有少量差异，完整记录见报告。',fontsize=9,color='#697783',linespacing=1.8)
for ext in ['png','pdf']:fig.savefig(ROOT/f'first_error_by_split.{ext}',dpi=200,facecolor='white')
plt.close(fig)

def pct(v):return 'NA' if v is None else f'{v:.1%}'
lines=['# v6 DFS nodes：不同查询划分的首错分析','',
       f"冻结4个N80 graph0 final50k checkpoint，全部{len(rows)}条原生自由生成。推理GPU1，无训练、权重/activation保存或W&B。", 
       '按照生成事件的时间顺序检查，首次违规之后停止归因；包含最终答案/EOS事件。合法隐式回退之后才检查下一节点是否为当前真实未访问后继，绝不将所有相邻输出非边都判错。',
       '完整协议成功要求节点展开合法、完整终止并回答正确；另记录仅节点轨迹语义合法性。此处“首错因”是行为分类，不能直接等同于内部机制原因。',
       '已有replay_nodes会先全局检查非节点token；本次流式判定避免未来语法错误遮蔽更早的搜索错误。349184个小DAG测试及全部10848条实际轨迹的最终语义合法性与原验证器一致。','',
       '## 主要发现','',
       '- 导航训练query完整协议合法率：短程约99.0%、长AA约99.3%；未见AT/TT约61%。当前差距主要体现在查询泛化，并非训练query普遍仍未拟合。',
       '- 合法到达目标之后，AA/TA未正确YES停止的条件率约1.2%/2.1%，AT/TT约24.2%/23.0%；满足穷尽条件后的NO停止接近正确。V目标也有类似差别：AV约21.5%、VA约1.3%。这些均为观察性分组，未控制图节点属性等因素。',
       '- 短程测试也有36/213的目标停止失败，而短程训练为0/1980，因此问题不只发生在长程留出目标角色；未见query组合也出现迁移缺口。']
for panel in ['long_AT_test','long_TT_test']:
    a=summary['original_panels'][panel];bad=a['n']-a['errors'].get('valid',0)
    search=sum(a['errors'].get(k,0) for k in ['non_edge_jump','skipped_live_frontier','revisited_node'])
    lines.append(f"- {panel}：到目标后继续的{a['goal_continuations_n']}次中，{a['goal_continuations_following_DFS']}次下一节点符合忽略目标的合法DFS；搜索展开类首错共{search}/{bad}，占非法轨迹{search/bad:.1%}。目标停止问题不是全部失败的解释。")
lines+=['- 高visited/过滤探针成绩与这些行为错误可以同时成立；平均排序可读性不是原生选择正确率，也没有在本次首错位置检验模型是否使用该方向。','',
       '## 导航模型原始查询划分','',
       '|原始面板|每seed查询数|完整协议合法率|答案BA|非边跳转|跳过未探索分支|重复访问|目标停止错误|过早结束|其他|',
       '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for p in panels:
    a=summary['original_panels'][p]
    lines.append(f"|{p}|{a['n_unique_queries']}|{pct(a['protocol_valid_rate'])}|{pct(a['answer_BA'])}|"+'|'.join(pct(a['groups'].get(k,0)/a['n']) for k in CAT[1:])+'|')
lines+=['','## 已合法到达终止条件的机会','',
        '分母是终止条件出现前始终合法的机会数，不是全部正例/负例；计数为4seed合计，百分比为合计计数之比。图中为逐seed条件比例均值，可能略有不同。',
        '', '|面板|目标后错误/机会|穷尽后错误/机会|继续节点时仍符合忽略目标的DFS|',
        '|---|---|---|---|']
for p in panels:
    a=summary['original_panels'][p];o=a['opportunities']
    lines.append(f"|{p}|{a['goal_fail_n']}/{o.get('goal_reached',0)} ({pct(a['goal_fail_rate'])})|{a['exhaust_fail_n']}/{o.get('search_exhausted',0)} ({pct(a['exhaust_fail_rate'])})|{a['goal_continuations_following_DFS']}/{a['goal_continuations_n']}|")
lines+=['','## 非边、回溯与重复访问的进一步分解','',
        '- non_edge_jump：按应有隐式回退后的栈顶判断，下一未访问节点不是真实后继，且不是当前栈祖先的未访问后继。',
        '- skipped_live_frontier：当前栈顶仍有合法候选，却输出严格祖先的某个未访问后继。这是跳过未探索分支的行为，不证明模型内部确实弹出了栈。',
        '- revisited_node：nodes协议输出已发现节点。区分重输当前点、重输栈祖先、重访已关闭分支，以及把隐式返回栈顶显式打印出来；不能全部叫visited编码失败。',
        '- reachable_descendant_skip：非边跳转的落点在应有栈顶的可达后代中，至少缺少一个中间节点；这是多跳跳转签名，不等于证明模型把可达性当成直接边。','',
        '|测试面板|非边跳转：跳可达后代/总数|非边直接跳目标|跳过分支直接跳目标|重复节点是真实后继|重复访问细分|',
        '|---|---|---:|---:|---|---|']
for p in ['short_test','long_AA_test','long_TA_test','long_AT_test','long_TT_test']:
    a=summary['original_panels'][p];e=a['errors']
    lines.append(f"|{p}|{a['jump_subtypes'].get('reachable_descendant_skip',0)}/{e.get('non_edge_jump',0)}|{a['non_edge_jumps_directly_to_target']}|{a['skipped_frontier_directly_to_target']}|{a['revisited_true_neighbor_count']}/{e.get('revisited_node',0)}|{a['repeat_subtypes']}|")
lines+=['','## 正例/负例与层差','',
        '以下“gap”为层差；正例已由独立BFS核验等于最短路，负例没有有限最短路。计数合并4seed，主导错误取首次违规最多的一类。',
        '', '|测试面板|标签|gap|n（含4seed）|完整协议合法率|最多的首错及次数|','|---|---|---:|---:|---:|---|']
for p in ['short_test','long_AA_test','long_TA_test','long_AT_test','long_TT_test']:
    for label in [1,0]:
        for gap in sorted({r['gap'] for r in rows if r['panel']==p}):
            rr=[r for r in rows if r['panel']==p and r['label']==label and r['gap']==gap]
            if not rr:continue
            a=aggregate(rr);c=Counter(r['error'] for r in rr if r['error']!='valid');most=c.most_common(1)
            lines.append(f"|{p}|{label}|{gap}|{len(rr)}|{pct(a['protocol_valid_rate'])}|{most or '无'}|")
lines+=['','## 原模型划分与后续probe重划分不能混用','',
        '|划分来源|split|每seed查询数|完整协议合法率|答案BA|','|---|---|---:|---:|---:|']
for group in ['main_split','probe_split']:
    for split,a in summary[group].items():lines.append(f"|{group}|{split}|{a['n_unique_queries']}|{pct(a['protocol_valid_rate'])}|{pct(a['answer_BA'])}|")
lines+=['','## 目标端点角色的完整性与错误位置','',
        '|原测试面板|正例协议合法率|负例协议合法率|答案对但协议错（4seed计数）|首错时已访问节点占全图 Q25/中位/Q75|',
        '|---|---:|---:|---:|---|']
for p in ['short_test','long_AA_test','long_TA_test','long_AT_test','long_TT_test']:
    a=summary['original_panels'][p];rr=[r for r in rows if r['panel']==p and not r['protocol_valid']]
    quant=np.quantile([r['visited_fraction'] for r in rr],[.25,.5,.75]).tolist() if rr else []
    lines.append(f"|{p}|{pct(a['positive_valid_rate'])}|{pct(a['negative_valid_rate'])}|{a['answer_correct_but_protocol_invalid']}|{'/'.join(pct(v) for v in quant)}|")
lines+=['','## 可复核实例','']
examples=json.loads((ROOT/'examples.json').read_text())
for p,error in [('long_AT_test','continued_after_goal'),('long_TT_test','non_edge_jump'),('long_AT_test','skipped_live_frontier'),('long_TT_test','revisited_node')]:
    e=examples.get(f'{p}/{error}')
    if not e:continue
    lines += [f"### {p}，query{e['query']}，seed{e['seed']}：{error}",'',
              f"查询：{e['source']} → {e['target']}；label={e['label']}；首错token={e['actual']}。",'',
              '首错前栈：`'+' '.join(e['stack'])+'`','',
              '执行必要回退后的栈：`'+' '.join(e['decision_stack'])+'`','',
              '当时合法候选：`'+' '.join(e['candidates'])+'`','',
              '局部输出：`'+' '.join(e['generated_tail'])+'`','',f"进一步分类：{e['detail']}",'']
lines+=['## 解释边界','',
        '- 条件停止错误和搜索展开错误都发生在已验证合法的历史之后，但本实验未在首错位置测activation或做干预，不能据此定位某层FFN、attention或visited子空间失效。',
        '- 查询角色、长度、分支和可达性分布不同；条件率仍是观察性比较，不是只改变目标角色的受控因果实验。',
        '- 非边跳转、跳过活跃分支、重复输出可能来自局部记忆、栈/visited状态使用、协议序列化或解码等多种机制，不强行单因解释。',
        '- 首错后不继续猜测内部状态或累计次生错误。零次某类首错不意味着后续永不发生。',
        '- 新BF16生成与历史final.json有少量差异，逐seed逐面板记录于results.json；所有结论用本次实际轨迹。',
        '- 只有一张N80图，4初始化重复同一组query；不以四倍query数当独立图样本。']
(ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Summarized',len(rows),'queries;',len(doc['seeds']),'seeds')

"""Actual hard-DAG report with explicit training family and native budgets."""
import json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;O=P/'hard_retest'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
models=['hard_clean','simple_clean','simple_error']
names={'hard_clean':'困难DAG训练·干净λ1','simple_clean':'简单图训练·干净λ16','simple_error':'简单图训练·纠错30% λ16'}
nav={m:read(O/(m+'_navigation.json')) for m in models}
assert all(v['finished'] for v in nav.values())
ids=[r['query_id'] for r in nav['hard_clean']['rows']]
assert all([r['query_id'] for r in nav[m]['rows']]==ids for m in models)
assert len(ids)==len(set(ids))==256
anchor=read(O/'hard_clean_attention.json');g6=read(O/'hard_clean_graph6.json')
carriers=read(O/'carrier_audit.json');assert carriers['finished']
old=read(P/'harder/hard_dag.json')
att={'hard_clean':anchor['attention'],'simple_clean':old['attention']['clean'],'simple_error':old['attention']['error']}
values={'hard_clean':anchor['value'],'simple_clean':old['value']['clean'],'simple_error':old['value']['error']}
def pct(v):return f'{v*100:.2f}%'
def rate(v,k):return v['counts'][k]/v['counts']['n']
def val(v,condition):return v['means'][1][v['conditions'].index(condition)][0]
lines=['# 真正的困难DAG重测：完整自由导航与训练域对照','',
'2026-10-24；全部完成。只用GPU0，无训练、checkpoint或W&B run。','',
'## 实际测试数据与模型身份','',
'本次所有完整导航均使用 `data/hard_dag_v1/context_r10b10_n100_m200_train50000_s0` 的同一批256个测试查询，128可达/128不可达。图家族为非严格分层 `multiscale_dag_v1`，不是先前的深窄严格分层图。每个输入仅边列表和query，没有提供正确CoT、visited或候选。每条label均对原始图重新做可达性搜索校验。','',
'三个checkpoint均为last200k。没有困难DAG训练的纠错checkpoint，所以困难DAG干净模型作为训练域参考，纠错模型仍然是简单图训练后迁移；不能把它们当作同训练设置下的纠错消融。','',
'|模型|训练家族|轨迹λ|原生上下文上限|','|---|---|---:|---:|']
for m in models:
    meta=nav[m]['metadata'];lines.append(f'|{names[m]}|{meta["training_graph_family"]}|{meta["lambda_trace"]}|{meta["max_seq_len"]}|')
lines+=['','## 完整自由生成导航','',
'原生上下文预算、batch16、BF16、贪心生成。每模型全部256条均已完成，以下没有混入教师强制、局部首token或best成绩。STOP可在预算不足时中断，模型再预测最终答案；这类情况单列。','',
'|模型|答对数|最终准确率|可达正例准确率|不可达负例准确率|完整有效合法轨迹|中断/强制STOP|',
'|---|---:|---:|---:|---:|---:|---:|']
for m in models:
    d=nav[m];c=d['counts']
    lines.append(f'|{names[m]}|{c["correct"]}/256|{pct(c["correct"]/256)}|{pct(c["yes_correct"]/c["yes_n"])}|{pct(c["no_correct"]/c["no_n"])}|{c["effective_complete"]}/256|{c["interrupted"]}/{c["forced_stop"]}|')
hard=nav['hard_clean']['counts']
lines+=['',f'困难DAG训练baseline与其历史逐查询预测一致：{hard["matches_saved"]}/256。历史last为199/256=77.734375%，best为75%；本次使用last。',
'',
'“完整有效合法轨迹”按原图独立重放状态和动作，允许候选顺序改变、允许MISTAKE取消错误记录；动作非法后不再把后续字段计作独立错误。该指标要求正确终止，因此与答案正确相关是定义的一部分，不能作为合法率导致准确率的独立因果证明。','',
'## 配对查询与预算诊断','',
'|模型|候选错误查询（可信重放前缀）|出现MISTAKE的查询（未截断计数）|trace长度中位数/最大值|未中断样本准确率|',
'|---|---:|---:|---:|---:|']
for m in models:
    d=nav[m];c=d['counts'];rows=d['rows'];good=[r for r in rows if not r['interrupted']]
    lengths=[r['trace_length'] for r in rows]
    lines.append(f'|{names[m]}|{c["candidate_bad_queries"]}|{c["marker_queries"]}|{np.median(lengths):.0f}/{max(lengths)}|{pct(sum(r["correct"] for r in good)/len(good)) if good else "无样本"}|')
hardrows=nav['hard_clean']['rows']
over4k=[r for r in hardrows if r['trace_length']+604>4096]
lines+=['',f'困难DAG baseline的256条中，仅{len(over4k)}条保留轨迹加604-token输入超过4096；最终答对但轨迹不完整有效合法的有{sum(r["correct"] and not r["effective_complete"] for r in hardrows)}条。这是原生32K的观察统计，不是4K重跑结果。',
        '预算中断可能发生在错误重复输出之后，不能把所有中断都当作图的合法DFS本身需要更长上下文。']
paired={}
for reference,target in [('simple_clean','simple_error'),('hard_clean','simple_clean'),('hard_clean','simple_error')]:
    a=np.array([r['correct'] for r in nav[reference]['rows']]);b=np.array([r['correct'] for r in nav[target]['rows']])
    gains=int((~a&b).sum());losses=int((a&~b).sum());diff=b.astype(float)-a.astype(float)
    rng=np.random.default_rng(20261024);boot=np.array([diff[rng.integers(0,256,256)].mean() for _ in range(2000)])
    n=gains+losses;prob=min(1.,2*sum(math.comb(n,i) for i in range(min(gains,losses)+1))/(2**n)) if n else 1.
    paired[target+'_vs_'+reference]=dict(gains=gains,losses=losses,delta=float(diff.mean()),
        query_bootstrap_ci=np.quantile(boot,[.025,.975]).tolist(),mcnemar_exact_p=prob,
        gained_query_ids=[r['query_id'] for i,r in enumerate(nav[target]['rows']) if not a[i] and b[i]],
        gains_with_marker=int(sum(not a[i] and b[i] and r['raw_mistake_tokens']>0 for i,r in enumerate(nav[target]['rows']))),
        gains_with_complete_legal_trace=int(sum(not a[i] and b[i] and r['effective_complete'] for i,r in enumerate(nav[target]['rows']))))
lines+=['','配对结果（目标模型相对参考模型）：','']
for k,d in paired.items():
    target,reference=k.split('_vs_');lo,hi=d['query_bootstrap_ci']
    lines.append(f'- {names[target]} 相对 {names[reference]}：修正{d["gains"]}条，引入{d["losses"]}条新错，净差{d["delta"]*100:+.2f}pp，查询bootstrap区间[{lo*100:.2f},{hi*100:.2f}]pp，精确McNemar p={d["mcnemar_exact_p"]:.4f}。')
gain=paired['simple_error_vs_simple_clean']
lines+=['',f'值得保留：简单图纠错模型相对简单图干净模型的{gain["gains"]}条新答对中，仅{gain["gains_with_marker"]}条输出过MISTAKE，仅{gain["gains_with_complete_legal_trace"]}条完成有效合法搜索。不能把最终答案收益全部解释为显式纠错或整条轨迹恢复合法；未完成合法搜索也不等同于随机猜测，仍可能有部分有效信息。']
lines+=['','该区间反映这256个查询的差异，不覆盖训练seed波动。困难图训练模型和简单图模型的训练λ、图家族和上下文不同，跨训练域比较不能归因为纠错监督。未中断样本也是不同选择子集，不能单独消除预算混淆。','',
'## 真正困难图baseline的attention','',
'为与两简单图模型的已有局部分析严格配对，选用相同897个正确轨迹状态，均位于前缀≤4096范围；所有模型原始上下文配置保留。困难模型32K后续状态未纳入此局部对照，但其完整导航没有4K截断。','',
'以下取非空候选首token预测位置，先按查询平均。对图内key的低直接attention不排除信息已被汇集至其他位置。','',
'|模型|L2 visited质量|L4 visited质量|L5合法出边终点质量|L6图attention总质量|',
'|---|---:|---:|---:|---:|']
for m in models:
    a=att[m];lines.append(f'|{names[m]}|{pct(a[1][2][5])}|{pct(a[3][2][5])}|{pct(a[4][2][9])}|{pct(a[5][2][0])}|')
lines+=['','## 关键修正：相关边信息位于后继边界标记','',
'额外检查了当前节点每条出边在输入中的位置偏移。困难图baseline L6并非不读取相关边，而是集中读取该边之后的EDGE标记；最后一条边之后则为QUERY标记。这些位置能看到之前完整的边三元组。它们不在原始相关边三元组内部，旧的按原始节点/三元组统计会漏掉这种读取。','',
'以下508个非空候选状态、213个查询，与前述局部对照同一状态池。表内是当前节点全部出边对应位置的质量，包含已visited邻居，不能直接等同上表“合法出边”的未visited子集。原始终点与后继标记处于不同相对距离，不用raw attention大小单独判定因果。','',
'|模型/主要层|原始出边终点attention|出边后继边界attention|','|---|---:|---:|']
for m,li in [('hard_clean',6),('simple_clean',5),('simple_error',5)]:
    a=carriers['results'][m]['attention_mass'][li-1]
    lines.append(f'|{names[m]}/L{li}|{pct(a[3])}|{pct(a[4])}|')
lines+=['',
'再仅在候选预测行、所有层分别屏蔽相关原始三元组/对应后继边界。控制组屏蔽数量相同、输入位置接近的无关边后继标记，保持其他历史表示可用。','',
'|模型|identity|遮挡原始相关边|遮挡相关后继边界|遮挡匹配无关后继边界|','|---|---:|---:|---:|---:|']
for m in models:
    v=carriers['results'][m]['legal_next_token']
    lines.append('|'+names[m]+'|'+'|'.join(pct(x) for x in v)+'|')
lines+=['',
'困难图baseline后继边界屏蔽的大幅效应，与L6该位置V去上下文的大幅效应相互支持：baseline实际使用了汇集到其他图内位置的相关边信息。旧分析若据“原始相关边终点attention很低”推断baseline较少读取相关边，需要修正。两模型的差别至少包含表征承载位置和使用层不同，不能直接归结为图结构attention数量多少。尚未逐维解码该V具体编码的邻接内容。']
lines+=['','## 局部因果干预（不是最终答案准确率）','',
'屏蔽visited仅发生在候选预测行；V替换保留原始Q/K和softmax权重，使用相同token在此前各层只关注自己的无上下文V。每层用同计算路径identity作对照。','',
'|模型/visited层|同层identity|屏蔽visited全部|只屏蔽当前节点|只屏蔽其他visited节点|',
'|---|---:|---:|---:|---:|']
for m,li in [('hard_clean',4),('simple_clean',4),('simple_error',2)]:
    v=values[m];conds=[f'identity_L{li}',f'visited_all_L{li}',f'visited_current_L{li}',f'visited_other_L{li}']
    lines.append('|'+names[m]+f'/L{li}|'+'|'.join(pct(val(v,k)) for k in conds)+'|')
lines+=['','|模型/图内V层|identity|全部图V无上下文|仅EDGE V无上下文|仅节点V无上下文|',
'|---|---:|---:|---:|---:|']
for m,li,v in [(m,5,values[m]) for m in models]+[('hard_clean',6,g6['value'])]:
    conds=['identity','all_graph_contextfree_V','marker_contextfree_V','endpoint_contextfree_V']
    lines.append('|'+names[m]+f'/L{li}|'+'|'.join(pct(val(v,k)) for k in conds)+'|')
lines+=['',
'困难baseline的visited依赖以及图内V对照应以这张表为准，不应拿简单图λ16 baseline代替。保留Q/K的V替换不能测出所有寻址功能；大attention不一定意味着该位置V中上下文内容对这一步必需。','',
'## 文件与验证','',
'- 三个 *_navigation.json：逐查询答案、紧凑轨迹重放指标、checkpoint路径/step/SHA256/原生预算、少量轨迹片段。','- hard_clean_attention.json：困难图训练baseline的同状态attention和L2/L4/L5对照。','- hard_clean_graph6.json：追加的第6层V干预。','- summary.json：模型总计、配对差异和查询级区间。','- key_results.png：唯一关键结果图。','',
'所有图label重新校验；相同query ID、顺序、prompt、batch16；新结果与历史baseline逐条核对。未生成新图、未训练、未改原checkpoint。原生预算不同已明确列出；没有困难图训练的纠错模型，不能声称完成了该设置下的匹配训练对照。']
(O/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(O/'summary.json').write_text(json.dumps(dict(models={m:dict(counts=nav[m]['counts'],metadata=nav[m]['metadata']) for m in models},paired=paired),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
labels=['Hard-trained clean (lambda=1)','Simple-trained clean (lambda=16)','Simple-trained correction (lambda=16)']
colors=['#2ca02c','#1f77b4','#ff7f0e'];x=np.arange(3)
for j,(metric,label) in enumerate([('overall','Overall'),('yes','Reachable'),('no','Unreachable')]):
    ys=[]
    for m in models:
        c=nav[m]['counts'];ys.append((c['correct']/256 if metric=='overall' else c[metric+'_correct']/c[metric+'_n'])*100)
    axes[0].bar(x+(j-1)*.23,ys,width=.23,label=label)
axes[0].set_xticks(x,['Hard-trained\nclean','Simple-trained\nclean','Simple-trained\ncorrection'])
axes[0].set_title('Full free navigation: actual hard DAG');axes[0].set_ylabel('Final answer accuracy (%)');axes[0].legend(fontsize=8)
for m,label,color in zip(models,labels,colors):
    axes[1].plot(np.arange(1,7),[r[2][5]*100 for r in att[m]],'o-',label=label,color=color)
axes[1].set_title('Visited attention at candidate prediction');axes[1].set_ylabel('Attention mass (%)')
for j,(condition,label) in enumerate([('identity','Identity'),('raw_relevant_edges','Block raw related edges'),('next_carriers','Block related successor markers'),('matched_carriers','Block matched other markers')]):
    ci=carriers['conditions'].index(condition)
    axes[2].bar(x+(j-1.5)*.18,[carriers['results'][m]['legal_next_token'][ci]*100 for m in models],width=.18,label=label)
axes[2].set_xticks(x,['Hard-trained\nclean','Simple-trained\nclean','Simple-trained\ncorrection'])
axes[2].set_title('Raw edges vs. contextual successor tokens');axes[2].set_ylabel('Legal next token, nonempty states (%)');axes[2].legend(fontsize=6)
axes[1].set_xlabel('Layer');axes[1].set_xticks(range(1,7));axes[1].legend(fontsize=7)
for ax in axes:ax.set_ylim(0,105);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.suptitle('N100 / M200 multiscale DAG: same 256 queries; native context budgets',fontsize=12)
fig.savefig(O/'key_results.png',dpi=180);plt.close(fig)
print('Wrote hard-DAG retest report and key figure.')

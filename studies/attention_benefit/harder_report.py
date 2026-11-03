"""Render one key figure and a compact report from completed aggregates."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;O=P/'harder'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
cases=['main','deep','wide','larger','hard_dag']
names={'main':'普通N100','deep':'深窄N100','wide':'宽浅N100','larger':'N144','hard_dag':'非严格分层DAG N100'}
data={c:read(O/(c+'.json')) for c in cases if c!='main'}
assert all(d['finished'] for d in data.values())
main=read(P/'main.json');main['value']=read(P/'graph_values_visited.json');data['main']=main
main_roles=read(P/'graph_value_roles.json')
fields=read(O/'deep_fields_batch16.json');assert fields['finished']
def attention(c,arm,layer,feature):return data[c]['attention'][arm][layer-1][2][feature]
def value(c,arm,condition,group=1):
    d=data[c]['value'][arm]
    if c=='main' and arm=='clean' and condition in ('marker_contextfree_V','endpoint_contextfree_V'):d=main_roles
    return d['means'][group][d['conditions'].index(condition)][0]*100
def pct(x):return f'{x:.2f}%'
lines=['# 困难图上的 attention 与局部因果对照','',
'2026-10-24；所有实验完成，只用GPU0，无训练、checkpoint、W&B run。','',
'## 范围与口径','',
'沿用同一对简单图课程训练的DFS模型：clean λ16和自然错误30% λ16，均last200k、6层128维、4K上下文。这里测分布迁移，不是在困难DAG上重新训练的两模型；现有hard_dag训练批次没有匹配的纠错模型。',
'',
'每类256个相同测试查询、每查询最多4个相同正确轨迹状态。另用不重叠的256个验证查询，分别按模型/层/头/角色/精确相对距离估计attention均值。下面attention均取非空候选首token的预测位置；干预也只看该位置的下一token合法率，允许任意合法候选顺序。它不是完整候选列表或最终答案准确率。query macro平均防止长轨迹主导。','',
'普通/深窄/宽浅/更大图均为严格分层DAG，M分别150/150/150/216；深窄20×5、宽浅5×20、更大12×12。困难DAG是multiscale_dag_v1、N100/M200，允许非相邻秩带间的边，改变了图家族和密度。','',
'## 最终导航与完整候选字段','',
'最终导航来自原先保存的完整batch16自由生成测试，本次没有重跑完整导航；困难DAG没有这对模型的完整导航结果。完整候选来自独立字段补全实验，统计所有选定状态（含空候选），按字段计数，不是query macro。','',
'|图|baseline最终导航|纠错模型最终导航|baseline完整候选正确率|纠错模型完整候选正确率|',
'|---|---:|---:|---:|---:|']
for c in cases:
    if c=='hard_dag':lines.append(f'|{names[c]}|未测|未测|未测|未测|');continue
    if c=='main':nav=[239/256*100,246/256*100]
    else:
        n=data[c]['navigation'];nav=[n[a+'_correct']/n['n']*100 for a in ('clean','error')]
    fs=[]
    for arm in ('clean','error'):
        f=read(P.parent/'accuracy_source'/f'prefix_dfs_{arm}_{c}.json')
        fs.append(f['correct']/f['n']*100)
    lines.append(f'|{names[c]}|{pct(nav[0])}|{pct(nav[1])}|{pct(fs[0])}|{pct(fs[1])}|')
lines+=['','深窄与N144上，纠错模型最终导航更高，但完整候选字段更低。这两个指标不能互相替代。不同图集合也不能作为严格配对的图形难度排序。','',
'## attention通路是否改变','',
'|图|baseline L4 visited质量|纠错 L2 visited质量|baseline L5合法出边终点质量|纠错 L5合法出边终点质量|纠错 L5距离校正终点密度差|',
'|---|---:|---:|---:|---:|---:|']
for c in cases:
    row=[attention(c,'clean',4,5)*100,attention(c,'error',2,5)*100,attention(c,'clean',5,9)*100,attention(c,'error',5,9)*100]
    lines.append('|'+names[c]+'|'+'|'.join(pct(v) for v in row)+f'|{attention(c,"error",5,18):.4f}|')
lines+=['','“合法出边终点”只指输入中当前节点的未访问出邻居位置，不覆盖已传播到其他图token/CoT位置的同一信息；baseline该指标小不表示没有读图。密度差是逐key残差对比，不是概率或百分比，距离校正不等于完全排除位置与内容的混淆。','',
'## 图内V：保留Q、K和attention权重','',
'只在L5候选预测行，把相同图内token的V换成前4层始终只能关注自身的donor V；这去掉周围图上下文，保留token本身。没有编辑其他query行的输出。以下均为非空候选、query macro下一token合法率。','',
'|图/模型|identity|全部图内V无上下文|仅EDGE V无上下文|仅节点V无上下文|',
'|---|---:|---:|---:|---:|']
for c in cases:
    for arm in ('clean','error'):
        conds=['identity','all_graph_contextfree_V','marker_contextfree_V','endpoint_contextfree_V']
        v=data[c]['value'][arm]
        vals=[pct(value(c,arm,k)) if k in v['conditions'] or (c=='main' and arm=='clean' and k in main_roles['conditions']) else '未测' for k in conds]
        lines.append('|'+names[c]+'/'+('baseline' if arm=='clean' else '纠错')+'|'+'|'.join(vals)+'|')
lines+=['','## visited：分别干预各模型使用的层','',
'仅屏蔽候选预测行对当前记录visited节点key的读取，并重归一化其余权重。baseline测L4，纠错模型测L2；同层identity为参照。','',
'|图/模型/层|同层identity|屏蔽整个visited|只屏蔽当前节点|只屏蔽其他visited节点|',
'|---|---:|---:|---:|---:|']
for c in cases:
    for arm,layer in [('clean',4),('error',2)]:
        conds=[f'identity_L{layer}',f'visited_all_L{layer}',f'visited_current_L{layer}',f'visited_other_L{layer}']
        v=data[c]['value'][arm]
        vals=[pct(value(c,arm,k)) if k in v['conditions'] else '未测' for k in conds]
        lines.append('|'+names[c]+'/'+('baseline' if arm=='clean' else '纠错')+f'/L{layer}|'+'|'.join(vals)+'|')
lines+=['','## 出乎意料的边界：深窄图候选重复','',
'按attention实验相同状态，用batch16再次完成候选列表，贪心生成直到模型自己输出分隔符；上限64个候选token，不能主动强制补分隔符。结果与之前batch32的关键计数一致。','',
'|深窄N100，854个字段|baseline|纠错模型|','|---|---:|---:|']
for key,label in [('correct','完整列表正确'),('first_legal','首token合法（含空字段闭合）'),('duplicate_fields','生成重复候选的字段'),('wrong_after_legal_first','首token合法但列表错误'),('duplicate_after_legal_first','其中出现重复'),('unclosed','未正常闭合')]:
    lines.append(f'|{label}|{fields["clean"]["counts"][key]}|{fields["error"]["counts"][key]}|')
lines+=['','纠错模型一个真实例子：正确候选 `{V219,V204,V112}`，生成 `V112 V204 V219 V204 |`。另一个正确候选 `{V22,V225,V180}`，生成 `V225 V22 V180 V225 V22 V180 |`。首节点和全部合法邻居都能找出，但随后又重复。','',
'这提示“读出相关邻居”与“维护本字段已经输出了哪些候选、何时结束”是不同环节。它是错误模式证据，还不是后者隐藏状态编码失效的完整因果证明。下一步若专门研究它，应在第二/第三个候选和结束符预测处检查已输出集合及抑制重复的计算。','',
'## 上下文覆盖与数值检查','',
'|图|测试查询|采样状态|参考已截断查询|参考内可用状态/枚举状态|最大采样前缀|','|---|---:|---:|---:|---:|---:|']
for c in cases[1:]:
    v=data[c]['coverage']
    lines.append(f'|{names[c]}|{v["queries"]}|{v["sampled_states"]}|{v["reference_truncated_queries"]}|{v["inside_context_states"]}/{v["all_candidate_states"]}|{v["max_selected_prefix"]}|')
lines+=['',
'严格分层图复用既有4K参考，因此“枚举状态”不含参考截断后的未知状态；深图43个查询、N144的54个查询参考已截断。困难DAG复用32K完整参考缓存，但只在前缀≤4096的非FOUND状态中抽最多4个状态；没有静默把长状态缩短。这里不能声称覆盖困难DAG全部搜索过程。',
'',
'每组都记录SDPA/math identity的BF16 argmax差异；干预和其identity使用相同计算路径。候选可以有多个合法首节点，因此argmax改变不总是合法率改变。图内同token donor V的最大差均为0，验证其未含周围图上下文。L2/L4/L5分别设identity。',
'',
'所有比较均为单次训练seed，沿用历史词表/初始化和难例重采样差异。局部next-token效应不能折算为最终答案损失，强attention不能证明CoT整体可靠。原始每图JSON保留查询级配对bootstrap区间、覆盖率、少量校验信息；没有全attention矩阵或activation dump。']
(O/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

fig,axes=plt.subplots(1,3,figsize=(16,4.4),layout='constrained')
x=np.arange(len(cases));labels=['Main\nN100','Deep\nN100','Wide\nN100','Larger\nN144','Hard DAG\nN100']
axes[0].plot(x,[attention(c,'clean',4,5)*100 for c in cases],'o-',label='Baseline L4')
axes[0].plot(x,[attention(c,'error',2,5)*100 for c in cases],'o-',label='Error-trained L2')
axes[0].set_title('Visited attention at candidate prediction');axes[0].set_ylabel('Attention mass (%)');axes[0].legend(fontsize=8)
for arm,label in [('clean','Baseline'),('error','Error-trained')]:
    axes[1].plot(x,[attention(c,arm,5,9)*100 for c in cases],'o-',label=label)
axes[1].set_title('L5 attention to eligible edge destinations');axes[1].set_ylabel('Attention mass (%)');axes[1].legend(fontsize=8)
for j,(cond,label) in enumerate([('identity_L2','Identity'),('all_graph_contextfree_V','Context-free graph V (L5)'),('visited_all_L2','Block visited (L2)')]):
    axes[2].bar(x+(j-1)*.23,[value(c,'error',cond) for c in cases],width=.23,label=label)
axes[2].set_title('Error-trained: local causal interventions');axes[2].set_ylabel('Legal next token, nonempty states (%)');axes[2].legend(fontsize=7)
for ax in axes:
    ax.set_xticks(x,labels);ax.set_ylim(0,105);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.suptitle('Same frozen simple-graph checkpoints evaluated on harder graphs',fontsize=13)
fig.savefig(O/'key_results.png',dpi=180);plt.close(fig)
print('Wrote',O/'REPORT.md',O/'key_results.png')

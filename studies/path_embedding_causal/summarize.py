"""Two key figures and complete tables; no selection of rank by test scores."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
DATA={s:json.loads((HERE/f'{s}.json').read_text()) for s in ('simple','hard')}
for d in DATA.values():assert d['analysis_version']==2 and d['checks']['embedding_and_tied_head_unchanged']
RANKS=DATA['simple']['ranks'];ROLES=('ancestors','descendants')

def curve(setting,role,method,rank,metric='counterfactual_answer_success',group='baseline_both_correct'):
    # Average control seeds inside each fold before averaging node splits.
    values=[]
    for entry in DATA[setting]['causal'][role]:
        rows=[r['metrics'][group] for r in entry['curves'] if r['method']==method and r['rank']==rank and r['metrics'][group]['n']]
        if rows:values.append(np.mean([r[metric] for r in rows]))
    return np.array(values)

def branch_curve(setting,method,rank,metric='donor_useful_argmax'):
    result=[]
    for entry in DATA[setting]['branching']:
        rows=[r['metrics'][metric] for r in entry['curves'] if r['method']==method and r['rank']==rank]
        if rows:result.append(np.mean(rows))
    return np.array(result)

def pct(value):return f'{100*value:.1f}%'
def avg(values):return float(np.mean(values))

fig,axes=plt.subplots(1,2,figsize=(10,4.4),layout='constrained',sharey=True)
for ax,(setting,d) in zip(axes,DATA.items()):
    roles=('ancestors','descendants','successors');x=np.arange(3)
    a=[];sd=[];controls=[];csd=[];structural=[];structural_sd=[]
    for role in roles:
        entries=d['probes'][role];real=[v['actual']['auc'] for v in entries]
        random=[np.mean([r['auc'] for r in v['random']]) for v in entries]
        a.append(np.mean(real));sd.append(np.std(real));controls.append(np.mean(random));csd.append(np.std(random))
        structure=[v['graph_structure']['auc'] for v in d['structural_controls'][role]]
        structural.append(np.mean(structure));structural_sd.append(np.std(structure))
    for offset,values,error,color,label in [(-.25,a,sd,'#2675b9','Trained PATH embedding'),(0,controls,csd,'#888888','Fixed random node vectors'),
                                          (.25,structural,structural_sd,'#31915c','Oracle rank / depth / degree / band')]:
        bars=ax.bar(x+offset,values,.24,yerr=error,capsize=3,color=color,label=label)
        ax.bar_label(bars,fmt='%.3f',padding=7,fontsize=8)
    ax.axhline(.5,color='black',ls=':',lw=1)
    ax.set_xticks(x,['All ancestors','Reachable descendants','Direct successors'],fontsize=9)
    ax.set_title(setting.capitalize()+' fixed DAG / PATH N100');ax.set_ylim(0,1.10)
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    ax.legend(fontsize=8,loc='lower right')
axes[0].set_ylabel('Fixed-candidate held-out-node AUROC')
fig.suptitle('Graph relations in PATH node embeddings\nPure embedding input; 60/20/20 node splits; bars = mean +/- node-split SD (3 splits)',fontsize=12)
fig.savefig(HERE/'embedding_relations.png',dpi=200);fig.savefig(HERE/'embedding_relations.pdf');plt.close(fig)

fig,axes=plt.subplots(2,3,figsize=(14,7.5),layout='constrained',sharex=True,sharey=True)
METHODS=[('semantic_swap','Natural donor projection','#d87929','-'),('targeted_semantic','Target one fact inside subspace','#2675b9','-'),
    ('targeted_random_equal_norm','Targeted-edit norm: random','#888888','--'),('targeted_PCA_equal_norm','Targeted-edit norm: PCA','#9467bd','--'),
    ('targeted_shuffled_equal_norm','Targeted-edit norm: shuffled','#be6671',':')]
for row,setting in enumerate(DATA):
    for col,role in enumerate(ROLES):
        ax=axes[row,col]
        for method,label,color,style in METHODS:
            values=[avg(curve(setting,role,method,rank))*100 for rank in RANKS]
            ax.plot(range(len(RANKS)),values,marker='o',ms=3.5,lw=1.6,color=color,ls=style,label=label)
        full=avg(curve(setting,role,'full_swap',0))*100;ax.axhline(full,color='#31915c',ls='--',lw=1,label='Full input-vector swap')
        ax.set_title(setting.capitalize()+' / '+('target ancestors' if role=='ancestors' else 'source descendants')+'\nFinal counterfactual answer',fontsize=10)
        if col==0:ax.set_ylabel('Counterfactual answer success (%)')
    ax=axes[row,2]
    for method,label,color,style in [('semantic_swap','Natural ancestor projection','#2675b9','-'),('random_equal_norm','Equal-norm random','#888888','--'),
                                    ('PCA_equal_norm','Equal-norm PCA','#9467bd','--'),('shuffled_equal_norm','Equal-norm shuffled','#be6671',':')]:
        ax.plot(range(len(RANKS)),[100*avg(branch_curve(setting,method,r)) for r in RANKS],marker='o',ms=3.5,lw=1.6,color=color,ls=style,label=label)
    ax.axhline(100*avg(branch_curve(setting,'no_op',0)),color='gray',ls=':',lw=1,label='No edit')
    ax.axhline(100*avg(branch_curve(setting,'full_swap',0)),color='#31915c',ls='--',lw=1,label='Full goal-vector swap')
    ax.set_title(setting.capitalize()+' / target ancestors\nNext node steered toward donor goal',fontsize=10)
    ax.set_ylabel('Argmax is a donor-useful successor (%)')
for row in range(2):
    for col in range(3):
        ax=axes[row,col];ax.set_xticks(range(len(RANKS)),RANKS);ax.set_ylim(-3,105);ax.grid(alpha=.15)
        ax.spines[['top','right']].set_visible(False)
        if row==1:ax.set_xlabel('Shared subspace rank')
        if row==0 and col in (0,2):ax.legend(fontsize=7,loc='upper left')
fig.suptitle('PATH: sparse input-embedding causal edits with held-out focal nodes\nFinal-answer panels: both original queries test; branching panel: recipient query test, donor goal probe-test',fontsize=12)
fig.savefig(HERE/'embedding_causal.png',dpi=200);fig.savefig(HERE/'embedding_causal.pdf');plt.close(fig)

lines=['# PATH embedding 的祖先/后代读出与低维因果编辑','',
    '本次完成旧 N100 简单图、困难图 PATH 验证最佳 checkpoint 的分析。节点 embedding 能泛化读出祖先/可达后代相关信号；但困难图的祖先AUC没有超过已测图结构oracle基线。8维定向编辑的行为作用较弱且偏向YES→NO，未实现稳定、对称、特异的关系修改。不能把探针变了等同于原模型知识被可靠编辑。','',
    '## 设置与严格留出','',
    '- 纯节点embedding探针：100个节点按60/20/20分组，seed17/31/47；祖先、可达后代、直接后继各自拟合128→100仿射ridge，val MSE选正则、val集合F1选阈值。主模型见过所有节点，探针test节点未参与拟合/验证。',
    '- 主指标固定一个候选节点，跨不同test焦点节点计算AUROC，再对有正负支持的候选平均。仅方向正确候选（祖先更早、后代/后继更晚）、排除自己；不靠给常见祖先统一高分获益。随机节点向量三seed匹配训练embedding坐标均值/标准差并走相同流程。',
    '- 因果方向构造只用60个train节点中的30个；独立审计线性解码器使用另30个，二者节点不相交。val20选各自正则与阈值，干预原节点及供体节点均来自test20；固定的另一个查询端点不要求节点留出。另有不使用该线性方向的5近邻审计。',
    '- 主因果配对两条查询都属于原query-test；目标祖先臂固定起点、换目标，起点后代臂固定目标、换起点，标签相反。先按图标签构造配对，再单列自然原/供体都答对的共同子集；各方法同一子集，不按干预结果筛选。',
    '- 每折最多128个有向配对，每个查询最多在3个无向配对中出现；同一无向配对两个方向都测。跨折可能重复，百分比不是独立图重复或显著性检验。',
    '- 只在QUERY的一个输入位置替换embedding向量，后续每步模型自由生成；全局embedding、绑定输出头、其他参数均未改变。完整向量替换等价于自然更换输入节点，prefill logits核验误差<1e-4。',
    '- 冻结GPU2 FP32前向，闭式probe拟合GPU0。无图约束、无重试、无oracle修正生成；最多N+3个新token，非法路径仍继续到答案以分开统计答案与轨迹。没有训练主模型或保存activation/probe checkpoint。','',
    '## embedding读出','',
    '|图|关系|节点留出AUC均值±SD|随机AUC均值|集合F1|完整集合准确率|','|---|---|---:|---:|---:|---:|']
for setting,d in DATA.items():
    for role in ('ancestors','descendants','successors'):
        entries=d['probes'][role];actual=np.array([e['actual']['auc'] for e in entries]);control=[v['auc'] for e in entries for v in e['random']]
        lines.append(f'|{setting}|{role}|{actual.mean():.3f} ± {actual.std():.3f}|{np.mean(control):.3f}|'+f"{np.mean([e['actual']['f1'] for e in entries]):.3f}|{pct(np.mean([e['actual']['exact_set'] for e in entries]))}|")
lines+=['','AUC表示关系排序，不代表完整集合已被准确恢复；F1与exact-set使用全候选集合（排除自己）及val选阈值，和方向限制AUC分母不同。误差是三个节点划分SD，不是置信区间。','',
    '## 图结构oracle基线：不能只和随机向量比','',
    '|图/关系|embedding AUC|拓扑rank单标量|生成器rank-band onehot|rank+最长深度/高度+入出度+band|','|---|---:|---:|---:|---:|']
for setting,d in DATA.items():
    for role in ('ancestors','descendants','successors'):
        real=np.mean([v['actual']['auc'] for v in d['probes'][role]])
        values=[np.mean([v[name]['auc'] for v in d['structural_controls'][role]]) for name in ('rank_only','rank_band_onehot','graph_structure')]
        lines.append(f'|{setting}/{role}|{real:.3f}|'+'|'.join(f'{v:.3f}' for v in values)+'|')
lines+=['','结构特征来自真实图，只是诊断对照，没有提供给主模型；困难图的rank-band不是固有DAG层。相同60/20/20节点划分、同样ridge和验证选择。困难图祖先0.935未超过结构基线0.941，不能把高祖先AUC单独解释为显式祖先集合存储。简单图祖先/后代及困难图后代高于这组已测结构基线，也不代表已排除所有shortcut。','',
    '## 两种编辑：不是只测自然供体投影','',
    '自然移植：由构造probe权重SVD得U_r，P=U_r U_r^T，e_edit=e+P(e_donor-e)。补空间移植e+(I-P)(e_donor-e)作比较，它不是少数维编辑。',
    '单事实定向编辑：在同一个共享rank-r子空间内，求最小L2改动，将构造probe对特定关系的分数置为val阈值±0.5（按供体真值）。若w_j是该关系候选的线性方向，则delta=(目标分数−当前分数)P w_j/||P w_j||²。为防数值爆炸，编辑L2范数最多为原embedding范数；记录截断后构造probe是否达到目标。',
    '这一步明确使用供体关系标签定义编辑目标，是oracle因果诊断，不是未知答案任务的求解算法。构造probe目标成功只是操作检查，仍须看独立审计与模型行为。','',
    '三seed随机方向、三seed训练节点内打乱关系标签方向、无标签PCA方向均逐例匹配相应编辑L2范数。定向编辑对照匹配的是定向delta，不能与自然投影的小范数对照混用。','',
    '## rank8：最终答案是否按供体关系改变','',
    '|图/臂|原/供体自然均答对配对数，按fold列出|自然投影|定向编辑|定向等范数随机|定向等范数PCA|定向等范数打乱|完整替换|',
    '|---|---|---:|---:|---:|---:|---:|---:|']
for setting,d in DATA.items():
    for role in ROLES:
        ns=[next(r for r in e['curves'] if r['method']=='no_op')['metrics']['baseline_both_correct']['n'] for e in d['causal'][role]]
        values=[avg(curve(setting,role,m,r)) for m,r in [('semantic_swap',8),('targeted_semantic',8),('targeted_random_equal_norm',8),('targeted_PCA_equal_norm',8),('targeted_shuffled_equal_norm',8),('full_swap',0)]]
        lines.append(f'|{setting}/{role}|'+','.join(map(str,ns))+'|'+'|'.join(map(pct,values))+'|')
lines+=['','主表先在每折内部平均控制seed，再对三折等权平均，不按样本量加权；原始JSON保留每折结果及全部配对结果。','',
    '## rank8：定向编辑是否只改变探针','',
    '|图/臂|构造probe达成指定关系|独立审计达成供体关系|最近embedding仍为原节点|相对编辑L2范数|正→负成功|负→正成功|','|---|---:|---:|---:|---:|---:|---:|']
for setting in DATA:
    for role in ROLES:
        keys=('constructor_query_membership_counterfactual','audit_query_membership_counterfactual','nearest_embedding_identity_retained','relative_edit_norm')
        values=[avg(curve(setting,role,'targeted_semantic',8,k)) for k in keys]
        values+=[avg(curve(setting,role,'targeted_semantic',8,group=group)) for group in ('positive_to_negative','negative_to_positive')]
        lines.append(f'|{setting}/{role}|'+'|'.join(map(pct,values))+'|')
lines+=['','最近embedding身份用全部原节点embedding的余弦最近邻衡量，只是身份保留代理，不证明其他身份特征、后继信息或输出协议完全未变。起点后代编辑的实际第一生成节点是否仍为原起点、是否采用供体起点，及路径合法率、原/供体完整witness率均保留在JSON。','',
    '## rank16：较宽但仍有限的子空间','',
    '|图/臂|自然投影答案改变|定向答案改变|定向随机|定向PCA|定向打乱|','|---|---:|---:|---:|---:|---:|']
for setting in DATA:
    for role in ROLES:
        values=[avg(curve(setting,role,m,16)) for m in ('semantic_swap','targeted_semantic','targeted_random_equal_norm','targeted_PCA_equal_norm','targeted_shuffled_equal_norm')]
        lines.append(f'|{setting}/{role}|'+'|'.join(map(pct,values))+'|')
lines+=['','rank8是预设主切片，rank1/2/4/8/16全曲线同时保留，没有按test选最优rank。维数少不等于编辑范数小，表中列出相对L2；定向编辑可能改变其他结构或产生分布外状态。','',
    '## 祖先方向能否改变下一步寻路','',
    '次级面板：接收者是原query-test可达查询，接收与供体目标都是probe-test节点，两个目标在相同起点的有用后继集合互不相交。供体目标从图真值选择，反事实查询可能属于主模型query-train/val，这不是严格未见查询泛化测试。prefix仅含原QUERY和已复制的起点，不含任何沿边移动。', '',
    '|图|每折配对数|无编辑选供体有用后继|rank8投影|rank16投影|rank16随机|rank16PCA|rank16打乱|完整目标替换|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
for setting,d in DATA.items():
    ns=[e['n'] for e in d['branching']]
    values=[avg(branch_curve(setting,m,r)) for m,r in [('no_op',0),('semantic_swap',8),('semantic_swap',16),('random_equal_norm',16),('PCA_equal_norm',16),('shuffled_equal_norm',16),('full_swap',0)]]
    lines.append(f'|{setting}|'+','.join(map(str,ns))+'|'+'|'.join(map(pct,values))+'|')
lines+=['','上述表是所有分支配对的完整词表argmax，原模型选错原分支时，无编辑也可能落入供体有用集合。因此必须与无编辑比较，而不能把全部供体命中都算编辑成功。不同fold面板量不相同，三折等权的百分比不等于直接汇总计数。原/供体有用概率、条件概率、原分支选择率与真实边合法率也保存在JSON。','',
    '## 验证与解释限制','',
    '图闭包逐起点与独立BFS核验；no-op答案与本次FP32自然生成一致，完整输入向量替换与自然供体输入的prefill logits一致；全部结束后原embedding与绑定输出头逐位未变。原始BF16生成与本次FP32自然答案一致率见JSON。','']
for setting,d in DATA.items():lines.append(f'- {setting}: {json.dumps(d["checks"],ensure_ascii=False)}')
lines+=['','有可解码的关系不证明原模型使用同一解码器方向；构造probe被强制翻转不证明语义被真实编辑。小子空间结果弱也不能推出祖先/后代完全不被使用，可能是子空间不完备、非线性编码或其他信息冗余。',
    '仅两张固定图各一个checkpoint，probe划分seed不是图或训练重复。不能声称已经发现可通用编辑的图记忆模块。','',
    '文件：run.py复现主实验，structural_controls.py补图结构诊断对照，summarize.py作图；simple.json/hard.json为最终汇总和少量实例。只有两个关键PNG/PDF，没有中间activation、探针权重或新checkpoint。','']
(HERE/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
print(HERE/'REPORT.md')

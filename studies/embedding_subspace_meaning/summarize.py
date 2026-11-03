"""One explanatory figure, both node-heldout protocols, compact audit report."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
HERE=Path(__file__).resolve().parent
DATA={s:json.loads((HERE/f'{s}.json').read_text()) for s in ('simple','hard')}
assert all(d.get('complete') for d in DATA.values())
ORIGINAL='original_space_nodeheld_probes';STRICT='strict_nodeheld_pipeline'
ATTRS=DATA['simple']['attribute_names']
TITLES=['QUERY source count','Source-role YES rate','QUERY target count','Target-role YES rate',
    'QUERY total count','WALK start count','WALK end count','WALK node count','In-degree','Out-degree',
    'Depth from roots','Height to sinks','Topological rank','Generator rank band','Ancestor count','Descendant count']

def entries(setting,role,policy):
    return [q for q in DATA[setting]['roles'][role][policy] if 'probes' in q]

def statistic(setting,role,policy,feature,attribute):
    result=[]
    for entry in entries(setting,role,policy):
        if feature=='random':
            result.extend(entry['probes'][f'random_subspace_{seed}']['metrics'][attribute]['r2'] for seed in (201,303,407))
        elif feature=='random_identity':
            result.extend(entry['probes'][f'random_node_identity_{seed}']['metrics'][attribute]['r2'] for seed in (201,303,407))
        else:result.append(entry['probes'][feature]['metrics'][attribute]['r2'])
    return np.array([v for v in result if v is not None],float)

def fmt(values):return f'{np.mean(values):.3f} ± {np.std(values):.3f}' if len(values) else 'NA'

fig,axes=plt.subplots(2,2,figsize=(13.5,11),layout='constrained')
for row,setting in enumerate(('simple','hard')):
    for col,role in enumerate(('source','target')):
        ax=axes[row,col];features=('learned','random','PCA8','full_embedding','complement')
        matrix=np.array([[statistic(setting,role,ORIGINAL,f,a).mean() for f in features] for a in ATTRS])
        im=ax.imshow(matrix,cmap='RdBu',norm=TwoSlopeNorm(vmin=-1,vcenter=0,vmax=1),aspect='auto')
        for i in range(len(ATTRS)):
            for j in range(len(features)):
                value=matrix[i,j];ax.text(j,i,f'{value:.2f}',ha='center',va='center',fontsize=8,color='white' if abs(value)>.65 else '#222222')
        ax.set_yticks(range(len(ATTRS)),TITLES,fontsize=8)
        ax.set_xticks(range(5),['Causal 8D','Random 8D','PCA 8D','Full 128D','Rest 120D'],rotation=20,ha='right',fontsize=9)
        ax.set_title(setting.capitalize()+' / '+role+' embedding causal subspace',fontsize=11)
fig.colorbar(im,ax=axes,fraction=.02,pad=.015,label='Held-out-node prediction R² (mean across splits)')
fig.suptitle('What can be decoded from the important embedding subspace?\nOriginal learned subspace fixed; explanatory probes use 60/20/20 disjoint nodes, 3 splits; negative R² retained',fontsize=12)
fig.savefig(HERE/'embedding_subspace_attributes.png',dpi=190)
fig.savefig(HERE/'embedding_subspace_attributes.pdf');plt.close(fig)

summary={}
for setting in DATA:
    summary[setting]={}
    for role in ('source','target'):
        item={}
        for policy in (ORIGINAL,STRICT):
            item[policy]={}
            for name in ATTRS:
                item[policy][name]={}
                for feature in ('learned','random','random_identity','PCA8','full_embedding','complement','position','position_plus_learned'):
                    values=statistic(setting,role,policy,feature,name)
                    item[policy][name][feature]=dict(mean=float(values.mean()),sd=float(values.std()))
        summary[setting][role]=item
(HERE/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))

lines=['# 重要 embedding 子空间的属性解释：曝光频率、度数与结构','',
    '结论：原8维因果子空间不是一个已被证实的“出现频率空间”或“入出度空间”。简单图起点空间有起点曝光频率及到汇点高度的部分线性信号；两图目标空间更明显地含深度、祖先数量、终点角色正例比例等相关信号，入出度通常很弱。这些标量属性只能解释少部分子空间坐标方差，不能为整个重要空间命名。','',
    '## 对象与节点留出','',
    '- 旧N100简单/困难固定图Direct checkpoint，分析起点/目标embedding上此前学习的rank8子空间，保持两角色维数相同。不是简单图L1 FFN后的1维空间，也不是最新N160暴露限制模型。',
    '- 协议A（主图）：复现原rank8因果子空间并固定。解释probe严格按节点60训练/20验证/20测试，seed17/31/47；每个节点仅一条embedding记录。这个分支的probe未见测试节点，但原U曾在因果训练中见过这些节点。',
    '- 协议B（更严格）：先排除probe测试节点，因果子空间也只用train节点配对学习、val节点选择checkpoint；之后解释probe用同一node fold。整个新分析管线未使用test节点属性或配对训练。原始主模型始终见过全部节点，这不是新图/新节点语言模型泛化。',
    '- 正交子空间坐标eU、补空间120维、完整embedding128维、训练节点拟合PCA8、3个随机8维投影、3个随机节点向量对照。线性ridge均只用train节点标准化，val节点选正则，test节点评分。',
    '- 主要等维对照是因果8维/随机8维/PCA8。Full128与Rest120自由度不同；逐坐标标准化、正则和有限样本都会影响R²，不能把表中高低直接当作表示的内在信息量或严格可解码性排序。',
    '- R²为1−test SSE/test SST；不是集合准确率，也不是属性被完整存储的比例。0.5–0.6仅表示部分预测能力；负数保留，表示差于测试均值常数基线。三fold标准差不是置信区间，均未把重复query作为新节点样本。','',
    '## 真实累计曝光审计','',
    '|图|QUERY次数|任务阶段WALK条数|共享知识阶段WALK条数|checkpoint|','|---|---:|---:|---:|---:|']
for setting,d in DATA.items():
    a=d['exposure_audit'];lines.append(f"|{setting}|{a['query_actual_count']}|{a['task_walk_actual_count']}|{a['knowledge_walk_actual_count']}|{a['task_checkpoint_step']}|")
lines+=['','频率由保存的IndexSampler整轮次数和当前排列已消费前缀恢复，并减去buffer尚未消费项。simple的历史short-hop包装保存的是局部train_ids索引，按历史signed-gap过滤与六个标签/方向stratum重建；hard的FixedQuerySampler也逐组核验。总数分别严格等于检查点queries_seen/walks_seen，知识父检查点SHA256与init_from一致。不是缓存潜在次数，也不是按均匀采样估计。',
    'QUERY起点/终点频率统计完整实际query流，simple含反向负例；WALK起点/终点/总节点出现次数包含共同Stage A与Stage B。角色YES比例用(yes+1)/(count+2)平滑，无该角色query节点为0.5。这里只量化可见出现次数，不等同于梯度影响或绑定输出头的全部训练信号。','',
    '## 原重要子空间：节点留出属性预测','',
    '|属性|简单起点空间 R²|简单目标空间 R²|困难起点空间 R²|困难目标空间 R²|','|---|---|---|---|']
for name in ATTRS:
    lines.append('|'+name+'|'+'|'.join(fmt(statistic(s,r,ORIGINAL,'learned',name)) for s in ('simple','hard') for r in ('source','target'))+'|')
lines+=['','这里“作为起点/终点的频率”和“该角色中YES的比例”是两个不同属性。后者本身与图可达结构及训练query选择相关，不能把它的可预测性称为频率计数机制。','',
    '## 控制层级/深度后，是否提供额外频率信息','',
    '位置基线输入只有拓扑序、生成rank band、真实depth和height。它是oracle shortcut诊断，不是主模型输入。只对频率/度数等属性比较增量；用depth预测depth的近1分数没有解释意义。','',
    '|图/空间|属性|位置基线 R²|位置+因果坐标 R²|','|---|---|---:|---:|']
for setting in DATA:
    for role in ('source','target'):
        for name in ('query_source_frequency','query_target_frequency','in_degree','out_degree'):
            a=statistic(setting,role,ORIGINAL,'position',name).mean();b=statistic(setting,role,ORIGINAL,'position_plus_learned',name).mean()
            lines.append(f'|{setting}/{role}|{name}|{a:.3f}|{b:.3f}|')
lines+=['','## 同时留出子空间学习与probe节点','',
    '|属性|简单起点 R²|简单目标 R²|困难起点 R²|困难目标 R²|','|---|---|---|---|']
for name in ('query_source_frequency','query_target_frequency','query_source_yes_rate','query_target_yes_rate','in_degree','out_degree','depth','height','ancestor_count','descendant_count'):
    lines.append('|'+name+'|'+'|'.join(fmt(statistic(s,r,STRICT,'learned',name)) for s in ('simple','hard') for r in ('source','target'))+'|')
lines+=['','两个协议数值不完全相同，因为严格分支重新学习U，不能把它当作原U的又一次独立检验。其用途是排除只在“子空间学习见过的节点”上解释。','',
    '## 反向解释：属性能重建多少子空间坐标','',
    '再用节点属性线性预测8维坐标z=eU。train拟合、val MSE选正则、test评分，采用整体欧氏R²，保持子空间坐标旋转不变。频率组含QUERY/WALK次数及角色YES比例；局部结构组含入出度、depth、height、拓扑序、band；全结构再加祖先/后代数量。','',
    '|图/空间/协议|频率组 R²|局部结构 R²|全部结构 R²|全部属性 R²|','|---|---|---|---|---|']
for setting in DATA:
    for role in ('source','target'):
        for policy in (ORIGINAL,STRICT):
            es=entries(setting,role,policy)
            stats=[fmt(np.array([q['coordinate_reconstruction'][g]['heldout_r2'] for q in es])) for g in ('frequency','local_structure','all_structure','combined')]
            lines.append(f'|{setting}/{role}/{"fixed U" if policy==ORIGINAL else "strict U"}|'+'|'.join(stats)+'|')
lines+=['','## 属性重建能否复现干预（严格管线）','',
    '把属性预测的供体−原节点坐标差移植进U，其余embedding分量保留；每例编辑长度上限为自然U移植长度，记录截幅比例。它是oracle属性替代诊断，不是原模型自主求解，也不是可加和的因果中介比例。低坐标R²与高干预率可同时出现；后者取决于下游敏感方向和阈值。','',
    '|图/空间|完整U移植成功|频率重建|局部结构重建|全结构重建|全部属性重建|','|---|---|---|---|---|---|']
for setting in DATA:
    for role in ('source','target'):
        es=entries(setting,role,STRICT)
        scores=[np.array([q['native_subspace']['success'] for q in es])]+[np.array([q['coordinate_reconstruction'][g]['intervention']['success'] for q in es]) for g in ('frequency','local_structure','all_structure','combined')]
        lines.append(f'|{setting}/{role}|'+'|'.join(f'{v.mean()*100:.1f}% ± {v.std()*100:.1f}pp' for v in scores)+'|')
lines+=['','## 匹配图属性/频率后是否还有干预效果','',
    '对严格管线同一test面板，按图/属性选择子集：入出度完全相同、depth与band完全相同、起点/终点曝光均落入同一train-defined四分位桶，以及上述交集。频率桶匹配不是频率数值完全相同。','',
    '|图/空间|seed|test配对数|同入出度 n/成功|同depth/band n/成功|同频率桶 n/成功|全部匹配 n/成功|','|---|---:|---:|---|---|---|---|']
for setting in DATA:
    for role in ('source','target'):
        for q in entries(setting,role,STRICT):
            parts=[]
            for key in ('same_in_out_degree','same_depth_and_band','same_source_target_frequency_bins','all_matched'):
                m=q['matched_pairs'][key];parts.append(f"{m['n']} / {m['success']*100:.1f}%" if m['n'] else '0 / NA')
            lines.append(f"|{setting}/{role}|{q['seed']}|{q['native_subspace']['n']}|"+'|'.join(parts)+'|')
lines+=['','很多子集很小，严格全部匹配常为空，不能据此完全排除联合shortcut；不按效果扩大或更换匹配条件。','',
    '## 可复现与边界','',
    '原rank8 U按之前site/seed重新学习，测试干预成功率与原记录差异<2pp；原权重、绑定头未修改。未保存U/探针权重或activation。verification.json验证sampler累计计数/缓冲扣除、node fold不交叉、ties rank，以及坐标重建在正交旋转后预测等价（最大误差<1e-4）。',
    '图形原U结果、严格重新学习结果、Pearson/Spearman、完整R²/MAE、正则、节点划分、静态节点属性、曝光审计和匹配支持量均保留在JSON。每图只有一个原checkpoint，3次node split不是3个训练模型重复。',
    '不能由R²低推出信息不存在，也不能由R²高推出模型使用该属性。当前只测试所列标量的线性解释；非线性统计、图社区或关系集合等仍可能重要。频率和结构在固定图中高度相关，现有观察与诊断不足以证明训练频率因果塑造了U。','',
    '复现：repo根目录执行run.py、verify.py、summarize.py；所有拟合和子空间训练GPU0，主模型冻结。','']
(HERE/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(HERE/'REPORT.md')

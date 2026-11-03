"""Summarize paired graph/init results and produce one exposure comparison figure."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
ARMS=('R0','R2','R3');TASKS=('ancestors','successors','descendants')
COLORS={'R0':'#2878b5','R2':'#da8b28','R3':'#31915c'}
TITLES=dict(ancestors='all ancestors',successors='direct successors',descendants='reachable descendants')
data={}
for graph in range(3):
    for seed in range(2):
        for arm in ARMS:
            data[graph,seed,arm]=json.loads((HERE/f'g{graph}_s{seed}_{arm}.json').read_text())

def one_model(result,role,task,panel='base_final',metric='auc',random=False):
    entries=result['roles'][role][task]
    curves=[]
    for entry in entries:
        values=entry['panels'][panel]
        if random:
            curves.extend([[values['random'][str(rs)][l][metric] for l in range(7)] for rs in (201,303,407)])
        else:curves.append([values['actual'][l][metric] for l in range(7)])
    return np.nanmean(np.array(curves,dtype=float),axis=0)

def graphs(arm,role,task,panel='base_final',metric='auc',random=False):
    return np.array([np.mean([one_model(data[g,s,arm],role,task,panel,metric,random) for s in range(2)],axis=0) for g in range(3)])

fig,axes=plt.subplots(2,3,figsize=(12.5,6.6),layout='constrained',sharex=True,sharey=True)
for row,role in enumerate(('source','target')):
    for col,task in enumerate(TASKS):
        ax=axes[row,col]
        random=[]
        for arm in ARMS:
            values=graphs(arm,role,task);mean=values.mean(0);sd=values.std(0)
            ax.plot(range(7),mean,'o-',ms=4,color=COLORS[arm],label=arm)
            ax.fill_between(range(7),mean-sd,mean+sd,color=COLORS[arm],alpha=.13)
            random.append(graphs(arm,role,task,random=True))
        ax.plot(range(7),np.mean(random,axis=(0,1)),':',color='gray',label='Random node identity')
        ax.axhline(.5,ls='--',lw=.8,color='black',alpha=.45)
        ax.set_title(role+' state / '+TITLES[task],fontsize=11)
        ax.set_xticks(range(7),['E']+[f'L{i}' for i in range(1,7)]);ax.set_ylim(.35,1.025)
        ax.grid(alpha=.16);ax.spines[['top','right']].set_visible(False)
        if col==0:ax.set_ylabel('Held-out-node AUROC')
        if row==1:ax.set_xlabel('Residual after transformer block')
axes[0,0].legend(fontsize=8,loc='lower right')
fig.suptitle('Direct N160: exposure-controlled training, final step 125k\nQuery + focal-node holdout; shading = SD across 3 graphs (2 initializations averaged)',fontsize=12)
fig.savefig(HERE/'direct_exposure.png',dpi=200)
fig.savefig(HERE/'direct_exposure.pdf');plt.close(fig)

summary={'arms':{},'models':[]}
for arm in ARMS:
    entry={}
    for role in ('source','target'):
        entry[role]={}
        for task in ('predecessors','successors','ancestors','descendants'):
            entry[role][task]={}
            for panel in ('base_final','core_final'):
                real=graphs(arm,role,task,panel);random=graphs(arm,role,task,panel,random=True)
                entry[role][task][panel]=dict(actual_mean=real.mean(0).tolist(),actual_graph_sd=real.std(0).tolist(),
                    random_mean=random.mean(0).tolist(),graph_means=real.tolist())
    ba=np.array([[data[g,s,arm]['manifest']['official_final']['balanced_accuracy'] for s in range(2)] for g in range(3)])
    entry['official_core_BA']=dict(mean=float(ba.mean()),graph_sd=float(ba.mean(1).std()),per_graph=ba.mean(1).tolist())
    summary['arms'][arm]=entry
for key,result in data.items():
    m=result['manifest'];assert m['CPU_GPU_max_logit_error']<1e-3
    summary['models'].append(dict(graph=key[0],init=key[1],arm=key[2],job=m['job'],
        checkpoint_sha256=m['checkpoint_sha256'],official_core_BA=m['official_final']['balanced_accuracy'],
        FP32_agreement=m['FP32_vs_official_prediction_agreement']))
(HERE/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))

lines=['# Direct：已有展示图与限制暴露模型结构探针','',
    '结果：旧 Direct 模型的逐层图已生成；新 adj_v3 全部 18 个 Direct 的同口径结构探针已完成。新模型没有复现旧困难图起点后代在 L1 大幅增强的模式：R2/R3 的后继与后代读出主要在较深层增强，而目标祖先读出往往随层数下降。原生 core 答案 BA 仍约 60%，不能由结构信号存在推断模型会使用它求解。','',
    '## 实验范围与公平口径','',
    '- 旧图：N100 简单图/困难图的原验证最佳 Direct，复用既有探针 JSON，未重新挑选 checkpoint。图 `direct_previous.png/.pdf`。',
    '- 新图：adj_v3 的 3 张 N160/M320 固定图 × 2 初始化 × R0/R2/R3，全部 18 个 Direct 使用预定 125000 步最终 checkpoint。未按测试性能筛选。图 `direct_exposure.png/.pdf`。',
    '- R0：完整局部邻接表；R2：另加最长 4 跳片段、不披露 core 测试完整 witness；R3：包含 core 正例完整 witness，是暴露上界。三组查询主任务相同，R3 的 witness 已在知识语料暴露，不能叫完全未见关系。',
    '- 每个原始 QUERY prompt 只含 QUERY、起点、目标、分隔符，不含图或答案；提取起点/目标的 E、L1–L6。模型冻结 FP32 GPU1，探针拟合 GPU0。',
    '- 四种标签均排除自己：直接前驱、直接后继、所有祖先、所有可达后代。祖先/后代闭包用每个起点独立 BFS 核验。',
    '- 128→160 仿射 ridge，按原 base train/monitor/final 查询分区；额外按焦点节点 60/20/20 留出，三个划分种子 17/31/47。原模型仍见过全部节点；这是探针未见节点，不是主模型未见节点。',
    '- 三组固定高斯随机节点向量（201/303/407），匹配各层训练均值/标准差，走完全相同拟合与验证选正则流程。不是每次查询重新抽噪声。',
    '- 主指标：固定一个候选节点，在不同留出焦点节点之间计算 AUROC；先按焦点聚合重复查询，仅保留方向正确候选，排除查询两个端点。恒定候选频率基线精确为 0.5。AUROC 不是集合准确率；不能按不同关系 AUROC 给信息量排序。',
    '- 同一探针另测 core_final 的留出焦点子集，不拿 core_final 选择正则。主图用 base_final，以保留更充分的节点支持；core 测试支持量更小，表格单列。',
    '- 图中先平均同图两个初始化和三个节点划分，再在三张图之间计算均值/标准差。阴影不是置信区间，也未把查询数当独立训练重复。','',
    '## 主结果：base_final','',
    '|位置/标签|暴露组|E AUROC|L1 AUROC|L6 AUROC|随机对照 E/L1/L6|',
    '|---|---|---:|---:|---:|---|']
for role,task in [('target','ancestors'),('source','successors'),('source','descendants')]:
    for arm in ARMS:
        q=summary['arms'][arm][role][task]['base_final'];a=q['actual_mean'];sd=q['actual_graph_sd'];r=q['random_mean']
        lines.append(f"|{role}/{task}|{arm}|{a[0]:.3f} ± {sd[0]:.3f}|{a[1]:.3f} ± {sd[1]:.3f}|{a[6]:.3f} ± {sd[6]:.3f}|"+' / '.join(f'{r[i]:.3f}' for i in (0,1,6))+'|')
lines+=['','R2、R3 的起点直接后继 L6 AUC 均约 0.85，R0 约 0.66；可达后代 L6 分别约 0.81、0.78、0.71。变化不是暴露越多每个指标越好：R2 的 embedding 祖先与后代读出可高于 R3。后继集合在 L1 的精确恢复仍很弱，详见下表。','',
    '## Core 泛化行为与结构读出','',
    '|组别|原生 Direct core BA，图均值±SD|目标祖先 E/L1/L6|起点后代 E/L1/L6|','|---|---:|---|---|']
for arm in ARMS:
    entry=summary['arms'][arm];ba=entry['official_core_BA']
    curves=[entry[r][t]['core_final']['actual_mean'] for r,t in [('target','ancestors'),('source','descendants')]]
    lines.append(f"|{arm}|{ba['mean']:.3f} ± {ba['graph_sd']:.3f}|"+'|'.join(' / '.join(f'{v[i]:.3f}' for i in (0,1,6)) for v in curves)+'|')
lines+=['','探针的 core 指标仅来自相应节点划分的子集，原生 Direct BA 来自完整 256 条 core 查询；二者分母不同，不能把探针 AUROC 当成模型解题准确率。','',
    '## 支持量、集合指标与验证','']
for panel in ('base_final','core_final'):
    for role in ('source','target'):
        records=[v['panels'][panel]['actual'][0] for d in data.values() for v in d['roles'][role]['ancestors']]
        lines.append(f"- {panel}/{role}：每个节点划分评估 {min(v['focal_nodes'] for v in records)}–{max(v['focal_nodes'] for v in records)} 个焦点节点、{min(v['queries'] for v in records)}–{max(v['queries'] for v in records)} 条查询；祖先 AUC 的有效候选 {min(v['n_candidates'] for v in records)}–{max(v['n_candidates'] for v in records)}。")
lines+=['','集合 F1、precision、recall、exact-set，以及 direction-valid recall@true-k/随机基线逐模型保留在 JSON；集合阈值只用 base monitor 选择。这些集合指标补充 AUROC，不能据 AUC 高宣称完整集合被精确恢复。','',
    '|位置/标签|组别|L1 集合 F1|L1 完整集合准确率|','|---|---|---:|---:|']
for role,task in [('target','ancestors'),('source','successors'),('source','descendants')]:
    for arm in ARMS:
        f=graphs(arm,role,task,metric='f1')[:,1].mean();e=graphs(arm,role,task,metric='exact_set')[:,1].mean()
        lines.append(f'|{role}/{task}|{arm}|{f:.3f}|{e:.3f}|')
agreements=[r['manifest']['FP32_vs_official_prediction_agreement'] for r in data.values()]
lines+=['',f'FP32 前向与正式保存的 BF16 core 预测一致率范围 {min(agreements):.4f}–{max(agreements):.4f}；行为表引用正式结果，不以 FP32 重放替代。每个 checkpoint 和图均校验 SHA256，CPU/GPU 前向差异小于 1e-3。`verification.json` 检查多标签批量 ridge 与旧单标签实现一致、oracle 指标为 1、恒定候选频率 AUC 为 0.5。','',
    '## 解释边界与文件','',
    '旧模型与新模型的节点数、图生成方式、数据划分、WALK/邻接表接口、训练过程均有区别；两套图并列展示，不当作只操纵暴露率的因果对照。新模型组内同图、同初始化的 R0/R2/R3 更适合暴露比较。',
    '读出改善不等于新增信息，也不证明模型用这个子空间作答；后续仍需选择性因果干预。特别是可读出的祖先/后代标签来自诊断 oracle，而不是主模型额外获得这些标签。','',
    '复现：在 repo 根目录执行 `.venv/bin/python studies/direct_exposure_probes/verify.py`、`run.py`、`summarize.py`。旧图由 `plot_previous.py` 从已有结果生成。所有拟合只在内存，不保存 activation、探针权重或新 checkpoint。','']
(HERE/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({arm:summary['arms'][arm]['official_core_BA'] for arm in ARMS},indent=2))

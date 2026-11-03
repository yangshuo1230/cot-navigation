"""Summarize training fit versus four diagnostic generalization panels."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
AUDIT=json.loads((HERE/'audit.json').read_text());PANELS=json.loads((HERE/'panels.json').read_text())
assert len(PANELS['jobs'])==36
assert all(j['job']['_code_sha256']==AUDIT['current_train_suite_sha256'] for j in AUDIT['jobs'])
ARMS=('R0','R2','R3');NAMES=('full_train','short_nonBB','short_BB','long_nonBB','long_BB_core')
summary={}
for protocol in ('direct','path'):
    summary[protocol]={}
    for arm in ARMS:
        jobs=[j for j in PANELS['jobs'] if j['protocol']==protocol and j['arm']==arm]
        entry={}
        for name in NAMES:
            metrics=('balanced_accuracy','yes_recall','no_recall')+(('positive_legal_goal_rate','positive_no_trace_rate') if protocol=='path' else ())
            entry[name]={key:float(np.mean([j['panels'][name][key] for j in jobs])) for key in metrics}
        for gap in ('1','2'):
            rows=[j['panels']['short_nonBB']['by_gap'][gap] for j in jobs]
            entry['short_nonBB_gap'+gap]={key:float(np.mean([r['correct'+label]/r['n'+label] for r in rows])) for key,label in [('positive','1'),('negative','0')]}
        summary[protocol][arm]=entry
(HERE/'summary.json').write_text(json.dumps(summary,indent=2))

fig,axes=plt.subplots(2,3,figsize=(13,7.8),layout='constrained',sharey=True)
x=np.arange(5)
for row,protocol in enumerate(('direct','path')):
    for col,arm in enumerate(ARMS):
        ax=axes[row,col];d=summary[protocol][arm]
        width=.23 if protocol=='path' else .34
        methods=[('yes_recall','Positive answer','#2675b9',-width/2),('no_recall','Negative answer','#d87929',width/2)]
        if protocol=='path':methods=[('yes_recall','Positive answer','#2675b9',-width),('no_recall','Negative answer','#d87929',0),('positive_legal_goal_rate','Positive valid goal path','#31915c',width)]
        for key,label,color,offset in methods:
            ax.bar(x+offset,[100*d[name][key] for name in NAMES],width,color=color,label=label)
        ax.set_title(protocol.upper()+' / '+arm);ax.set_xticks(x,['Train','Short\nnon-BB','Short\nBB','Long\nnon-BB','Core\nlong BB'],fontsize=8)
        ax.set_ylim(0,108);ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
        if col==0:ax.set_ylabel('Class accuracy / valid path (%)');ax.legend(fontsize=7,loc='lower left')
fig.suptitle('adj_v3 diagnosis: training fit is high; held-out QUERY generalization is poor\nFinal step125k; means across 3 graphs x 2 initializations; native BF16 frozen evaluation',fontsize=12)
fig.savefig(HERE/'training_diagnosis.png',dpi=190);fig.savefig(HERE/'training_diagnosis.pdf');plt.close(fig)

lines=['# R0/R2/R3 训练设置诊断','',
    '结论：不是连训练数据都未拟合，而是训练样本近乎完全拟合、未见短程QUERY也泛化很差。更重要的是，这批PATH的负例没有轨迹、直接NO，正例才生成起点，首token已承担分类；加上QUERY仅1–2层间距、R3长witness仅作为无目标条件WALK，当前结果不能用于干净判断模型是否具有长程规划能力。','',
    '## 核查范围','',
    '- adj_v3全部36个最终step125000 checkpoint：3图×2初始化×Direct/PATH×R0/R2/R3；训练源码SHA256与每个resolved_job记录一致。',
    '- 新补充冻结BF16原生评估，GPU1当时空闲；未训练、未调整权重、未创建W&B run。原core结果直接读取正式final.json，不按测试选checkpoint。',
    '- Train为完整base训练池，不再只看固定128条面板；Short non-BB为原base.final，即同样1–2间距、普通query留出；Short BB为1–2间距的全部B×B查询；Long non-BB为非B×B的6–9间距查询，标签和gap桶数与core匹配；Core为原256条长距离B×B最终面板。',
    '- 图是本批10层×16节点、相邻层间有向边的DAG；正例的gap等于最短跳数。负例没有有限最短路。改变端点块和长度仍可能伴随其他结构属性变化，四格结果是诊断而非全部混杂已消除的因果估计。',
    '- 下表均先每模型统计，再平均6个模型，不把查询当独立训练重复。','',
    '## PATH：训练拟合与泛化','',
    '|指标|R0|R2|R3|','|---|---:|---:|---:|']
for label,panel,key in [('完整训练集BA','full_train','balanced_accuracy'),('普通短程留出BA','short_nonBB','balanced_accuracy'),
    ('短程BB留出BA','short_BB','balanced_accuracy'),('长程non-BB BA','long_nonBB','balanced_accuracy'),
    ('core BA','long_BB_core','balanced_accuracy'),('core正例召回','long_BB_core','yes_recall'),
    ('core负例召回','long_BB_core','no_recall'),('core正例合法到达目标','long_BB_core','positive_legal_goal_rate'),
    ('core正例未生成任何轨迹','long_BB_core','positive_no_trace_rate')]:
    lines.append('|'+label+'|'+'|'.join(f"{100*summary['path'][arm][panel][key]:.2f}%" for arm in ARMS)+'|')
lines+=['','## 连一跳QUERY接口也没有稳定泛化','',
    '|模型/组别|未见一跳正例|未见一跳负例|未见两跳正例|未见两跳负例|','|---|---:|---:|---:|---:|']
for protocol in ('direct','path'):
    for arm in ARMS:
        d=summary[protocol][arm];values=[d['short_nonBB_gap'+gap][label] for gap,label in [('1','positive'),('1','negative'),('2','positive'),('2','negative')]]
        lines.append(f'|{protocol}/{arm}|'+'|'.join(f'{100*v:.1f}%' for v in values)+'|')
lines+=['','PATH三组的完整邻接表exact-set恢复在125k均为100%，但未见一跳QUERY正例只有26.6%/35.0%/34.6%。完整邻接表知识可被ADJ提示访问，不代表已经能通过QUERY格式稳定调用。Direct也存在同样的短程正例泛化问题，所以不能只归因于PATH长轨迹生成。','',
    '## 已确认的训练设计问题','',
    '1. 负例PATH空body。Samples.query调用render_task时没有生成负例随机路径，graph_reference注释也明确规定PATH negatives have empty body。保存的训练预测确认负例直接输出NO；正例第一个token为起点。因此第一步必须先完成正负分类，无法让两类样本都先执行路径计算再判断。这与旧N100的负例随机游走格式不同。',
    '2. QUERY监督仅base gap<=2，core gap>=6，缺乏长目标条件PATH监督。中间3–5跳也不参与QUERY训练；R2/R3的WALK不是相同接口。当前core成绩混合了查询泛化、知识访问、长度外推和路径控制。',
    '3. R3只是信息暴露上界，不是同协议任务学习阳性对照。R3完整witness进入<WALK>节点序列，没有前置QUERY起点/目标，因此未监督模型在给定远目标时选择该路径和在目标处停止。R3失败不能直接证明完整路径暴露无效或模型不具备规划。',
    '4. 训练正式monitor使用core_monitor；没有把普通短程留出能力作为训练质量门槛。此前只报告core低分，遗漏了普通短程泛化同样差这个诊断。','',
    '## 是否只是训练不够','',
    '每个任务125k×64≈800万条QUERY采样，对约两千个训练查询重复很多次；完整训练集答案几乎100%，PATH R0训练损失接近0。没有普通意义上的训练集欠拟合证据。R2/R3总loss主要来自knowledge_loss；WALK在相同前缀下可有多个后继，不能把这个非零损失直接当作任务没收敛。',
    '这不能排除后期grokking或优化/正则影响，但目前不应把单纯加步数作为首要修复。','',
    '## 建议先做的小规模诊断修复，尚未启动','',
    '- 知识阶段继续保持无查询。任务阶段让正负例都先输出起点与合法轨迹，避免是否生成CoT直接泄露类别；负例轨迹须受暴露审计，先限制长度<=2或与新增任务长度匹配。随机负路径仍不是不可达证明。',
    '- 保留完整局部邻接知识，增加任务阶段局部QUERY访问桥，并把未见一跳正/负例能力设为门槛；邻接表背得对不应替代此门槛。若增加局部BB查询，要明确改变了原整块留出定义，不能仍声称所有BB组合未见。',
    '- 在不包含core完整witness的条件下，增加3–5跳目标条件QUERY/PATH课程。因core最短6–9跳，单个<=5跳轨迹不会直接包含其完整witness；仍需审计完整数据流、重新计算宏关系拼接距离。该版本先用于训练可行性诊断，不直接声称强组合泛化。',
    '- 单独分叉R3-Q阳性对照：给已暴露长witness加相同QUERY和目标条件监督，检查这些已暴露长任务能否同时学会合法路径与正确停止。明确它不是未见查询测试，保留原R3-WALK结果。',
    '- 首轮先一张图一个初始化做配对诊断，分别控制负例格式与课程，而不是立即全36任务重训；通过局部任务与阳性对照后再扩大正式暴露实验。正式规划区分仍需高宏拼接数、多个未见混合分叉，不能用修复后的容易组合样本替代。','',
    '文件：audit.json为训练日志/知识/正式core审计，panels.json为36模型冻结诊断，summary.json为聚合；training_diagnosis.png/.pdf是唯一关键图。本次没有修改任何训练代码或启动新训练。','']
(HERE/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
print(HERE/'REPORT.md')

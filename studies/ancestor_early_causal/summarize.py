"""One key figure plus auditable tables; ranks are reported without test selection."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
DATA={s:json.loads((HERE/f'{s}.json').read_text()) for s in ('simple','hard')}
RANKS=DATA['simple']['ranks']

def records(entry,method,rank=0):
    return [r['metrics'] for r in entry['curves'] if r['method']==method and r['rank']==rank]

def mean(entry,method,rank,metric):
    return float(np.mean([r[metric] for r in records(entry,method,rank)]))

def primary(setting,stage):return DATA[setting]['stages'][stage][0]

fig,axes=plt.subplots(2,4,figsize=(15.5,7.1),layout='constrained',sharex=True,sharey=True)
for row,setting in enumerate(('simple','hard')):
    for col,(stage,erasure) in enumerate([('embedding',False),('L1_attention',False),('embedding',True),('L1_attention',True)]):
        ax=axes[row,col];entry=primary(setting,stage)
        names=[('ancestor_erase','Ancestor','#2675b9'),('random_erase','Equal-norm random','#777777'),('shuffled_erase','Shuffled-label','#9467bd')] if erasure else [
            ('ancestor_swap','Ancestor','#2675b9'),('complement_swap','Complement','#d87929'),('random_swap','Equal-norm random','#777777'),('shuffled_swap','Shuffled-label','#9467bd')]
        metric='edited_original_accuracy' if erasure else 'counterfactual_success'
        for method,label,color in names:
            values=[[m[metric]*100 for m in records(entry,method,r)] for r in RANKS]
            avg=np.array([np.mean(v) for v in values]);sd=np.array([np.std(v) for v in values])
            ax.plot(range(len(RANKS)),avg,'o-',ms=3.5,lw=1.7,color=color,label=label)
            if 'random' in method or 'shuffled' in method:ax.fill_between(range(len(RANKS)),avg-sd,avg+sd,color=color,alpha=.12)
        control=100 if erasure else 100*mean(entry,'full_swap',0,'counterfactual_success')
        ax.axhline(control,color='#31915c',ls='--',lw=1,label='No-op' if erasure else 'Full swap')
        stage_title='Embedding' if stage=='embedding' else 'L1 after attention'
        ax.set_title(f'{setting.capitalize()} / {stage_title}\n'+('Ancestor erasure' if erasure else 'Donor subspace swap'),fontsize=10)
        ax.set_xticks(range(len(RANKS)),RANKS);ax.set_ylim(-4,106);ax.grid(alpha=.15)
        ax.spines[['top','right']].set_visible(False)
        if row==1:ax.set_xlabel('Subspace rank')
        if col==0:ax.set_ylabel('Counterfactual answer success (%)')
        if col==2:ax.set_ylabel('Original answer retained (%)')
        if row==0 and col in (0,2):ax.legend(fontsize=7,loc='center left')
fig.suptitle('Direct N100: causal edits to target embedding vs L1 post-attention residual\nSame-source donors, opposite labels; original-correct test pairs; query-only probe split (node-heldout controls in report)',fontsize=12)
fig.savefig(HERE/'ancestor_early_causal.png',dpi=190);fig.savefig(HERE/'ancestor_early_causal.pdf');plt.close(fig)

lines=['# 祖先方向：embedding 与 L1 attention 后的因果干预','',
    '结论：在目标 embedding 中移植祖先探针子空间，可产生超过等范数随机方向的定向答案变化；但同时改变节点身份等信息，尚不能单独归因为祖先语义。L1 attention 后 rank32 的祖先移植则在两张图上都只改变约0.8%的答案，即使独立审计解码器明显转向供体；补空间移植几乎完全改变答案。留出节点分支中的强效应与打乱标签对照接近，不能声称特异祖先因果中介。','',
    '## 设置与干预位置','',
    '- 仅旧 N100 简单/困难固定图 Direct 验证最佳 checkpoint；未分析最新 N160 限制暴露模型。checkpoint 哈希沿用并核验。',
    '- 输入为原 QUERY prompt，只修改目标位置一次；起点不变。embedding 通过输入位置 hook 替换，权重和绑定输出头逐位确认不变。',
    '- L1 attention 后表示 attention 残差相加后、norm2/FFN之前。通过修改attention增量实现目标残差替换，FFN输入和skip残差同时获得新状态；不是只改LayerNorm输入。后续所有计算重新执行。',
    '- 使用旧 crossed-square 测试面板，供体为同起点换目标的自然查询 b=(s,t\u2032)，答案相反。固定原四问均正确子集：简单238对、困难119对；不按干预结果筛选。配对共享节点与查询，不是独立模型重复。',
    '- 祖先标签为严格祖先集合，排除自身。构造方向的probe与独立审计probe用原query-train的不相交两半拟合，各自只用原query-val选正则。query-only分支二者仍共享节点身份，不能把这种独立称为节点独立。',
    '- 另做target节点60/20/20划分（seed17/31/47），与原query split相交；构造probe和审计probe均未训练过干预/供体目标节点；二者都必须为test-node，且沿用同一原面板。身份审计单独使用全query-train拟合，以使节点身份保留可以测量。',
    '- 扫描rank1/2/4/8/16/32/64，rank32为与旧实验对齐的展示切片，rank64为较宽子空间补充；没有根据test选最优rank。', '',
    '## 干预公式','',
    '设原状态h、同起点供体状态hd、训练均值mu、祖先解码器左奇异向量投影P：',
    '- 祖先移植：h + P(hd-h)。',
    '- 补空间移植：h + (I-P)(hd-h)。',
    '- 祖先擦除：h - P(h-mu)，将该子空间设置为训练均值。',
    '- 全替换hd和no-op为正/负控制。三seed随机方向、三seed打乱目标→祖先标签的方向，逐样本匹配祖先移植或擦除的编辑L2范数；未强制保留编辑后状态总范数，JSON中保留其变化。', '',
    '## 原查询划分：答案定向改变','',
    '|图/位置|rank|祖先移植|补空间移植|随机方向均值|打乱标签均值|全替换|',
    '|---|---:|---:|---:|---:|---:|---:|']
for setting in DATA:
    for stage in ('embedding','L1_attention'):
        entry=primary(setting,stage)
        for rank in (32,64):
            values=[mean(entry,method,r,'counterfactual_success') for method,r in [('ancestor_swap',rank),('complement_swap',rank),('random_swap',rank),('shuffled_swap',rank),('full_swap',0)]]
            lines.append(f'|{setting}/{stage}|{rank}|'+'|'.join(f'{v*100:.1f}%' for v in values)+'|')
lines+=['','## 原查询划分：擦除后的原答案保留率','',
    '|图/位置|rank|祖先擦除|等范数随机擦除|打乱标签擦除|','|---|---:|---:|---:|---:|']
for setting in DATA:
    for stage in ('embedding','L1_attention'):
        entry=primary(setting,stage)
        for rank in (32,64):
            values=[mean(entry,method,rank,'edited_original_accuracy') for method in ('ancestor_erase','random_erase','shuffled_erase')]
            lines.append(f'|{setting}/{stage}|{rank}|'+'|'.join(f'{v*100:.1f}%' for v in values)+'|')
lines+=['','## rank32 的独立审计与身份混杂','',
    '|图/位置|祖先分数向供体进展|供体祖先集合F1|起点祖先归属按反事实分类正确|原目标身份保留|模型答案改变|',
    '|---|---:|---:|---:|---:|---:|']
for setting in DATA:
    for stage in ('embedding','L1_attention'):
        entry=primary(setting,stage);m=records(entry,'ancestor_swap',32)[0]
        keys=('independent_ancestor_score_progress','independent_ancestor_donor_F1','independent_source_membership_counterfactual_accuracy','target_identity_retained','counterfactual_success')
        lines.append(f'|{setting}/{stage}|'+'|'.join(f'{m[k]*100:.1f}%' for k in keys)+'|')
lines+=['','分数进展是沿“审计probe供体分数−原分数”的投影系数；1为完成自然供体差值，允许超过1或为负，不是百分比恢复率或解释方差。集合F1使用val选阈值，排除原目标、新目标、起点；起点祖先归属另单独审计。二者不使用相同候选分母。', '',
    '在L1 attention后，rank32擦除将审计解码器的起点祖先归属判断推离原标签，但两模型仍全部保留原答案。它说明此解码器可见方向并非可靠的单一必要中介，而非证明全部祖先信息被清除。','',
    '## 留出目标节点分支：支持量与打乱控制','',
    '|图/位置|节点seed|配对数|祖先移植rank32|随机均值|打乱标签均值|有效W rank|祖先/打乱子空间重叠|',
    '|---|---:|---:|---:|---:|---:|---:|---:|']
for setting,d in DATA.items():
    for stage,entries in d['stages'].items():
        for entry in entries[1:]:
            if not entry['n']:
                lines.append(f"|{setting}/{stage}|{entry['node_seed']}|0|—|—|—|—|—|");continue
            values=[mean(entry,method,32,'counterfactual_success') for method in ('ancestor_swap','random_swap','shuffled_swap')]
            overlap=np.mean(entry['control_subspace_overlap']['32']['shuffled'])
            lines.append(f"|{setting}/{stage}|{entry['node_seed']}|{entry['n']}|"+'|'.join(f'{v*100:.1f}%' for v in values)+f"|{entry['effective_rank_1e5']}|{overlap:.3f}|")
lines+=['','这里配对数仅6–23对，且部分seed没有合格配对，不能做稳定跨划分推断。特别是困难图L1 attention后，留出节点的祖先移植成功率很高，但打乱标签方向可复现相同结果，故不具有祖先语义特异性。',
    '重叠定义为 ||U_ancestor^T U_shuffled||_F² / rank；1表示两个投影子空间相同，随机同rank子空间期望约为rank/128。标签、节点身份及样本行空间可导致不同语义probe共享方向。rank超过有效解码器rank时还会包含近零奇异值方向，应结合JSON中的奇异值谱解释，不能把所有选中方向称作祖先语义。','',
    '## 验证、实例与解释边界','',
    '观察hook/no-op不改变模型结果；embedding完整替换与自然更换目标的查询logits对齐；attention位置核验实际FFN输入与所请求的残差一致；全局embedding与绑定输出头未修改。每个位置的独立审计还检查直接后继、可达后代、深度读出进展与源节点身份副本，详细数值见JSON。','']
for setting,d in DATA.items():
    lines.append(f"- {setting}：{json.dumps(d['checks'],ensure_ascii=False)}")
lines+=['','不能把embedding干预有效表述为“证明祖先集合被用于答题”：目标身份已明显被破坏，其他关系特征也可能同时改变；对照支持的是相关方向的部分因果影响。不能把attention后干预阴性表述为“祖先未被使用”：可能有冗余、非线性读法、未清除方向，且L1分隔符已经从原embedding读过信息。', '',
    '未停止正在进行的训练；冻结前向CPU FP32/2线程，短时ridge拟合GPU0。仅保存simple.json、hard.json、关键PNG/PDF、报告和复用脚本；没有activation、probe权重、模型checkpoint或W&B run。','']
(HERE/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
print(HERE/'REPORT.md')

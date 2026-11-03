"""One final figure and report for source-descendant causal edits."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
DATA={s:json.loads((HERE/f'{s}.json').read_text()) for s in ('simple','hard')}
RANKS=DATA['simple']['ranks']

def values(entry,method,rank=0):
    return [r['metrics'] for r in entry['curves'] if r['method']==method and r['rank']==rank]

def mean(entry,method,rank,metric):return float(np.mean([r[metric] for r in values(entry,method,rank)]))
def primary(setting,stage):return DATA[setting]['stages'][stage][0]

fig,axes=plt.subplots(2,4,figsize=(15.5,7.1),layout='constrained',sharex=True,sharey=True)
for row,setting in enumerate(('simple','hard')):
    for col,(stage,erase) in enumerate([('embedding',False),('L1_attention',False),('embedding',True),('L1_attention',True)]):
        ax=axes[row,col];entry=primary(setting,stage)
        methods=[('descendant_erase','Descendant','#2675b9'),('random_erase','Equal-norm random','#777777'),('shuffled_erase','Shuffled-label','#9467bd')] if erase else [
            ('descendant_swap','Descendant','#2675b9'),('complement_swap','Complement','#d87929'),('random_swap','Equal-norm random','#777777'),('shuffled_swap','Shuffled-label','#9467bd')]
        metric='edited_original_accuracy' if erase else 'counterfactual_success'
        for method,label,color in methods:
            curves=[[m[metric]*100 for m in values(entry,method,r)] for r in RANKS]
            avg=np.array([np.mean(v) for v in curves]);sd=np.array([np.std(v) for v in curves])
            ax.plot(range(len(RANKS)),avg,'o-',ms=3.5,lw=1.7,color=color,label=label)
            if len(curves[0])>1:ax.fill_between(range(len(RANKS)),avg-sd,avg+sd,color=color,alpha=.12)
        reference=100 if erase else mean(entry,'full_swap',0,'counterfactual_success')*100
        ax.axhline(reference,color='#31915c',ls='--',lw=1,label='No-op' if erase else 'Full source swap')
        stage_title='Embedding' if stage=='embedding' else 'L1 after attention'
        ax.set_title(f'{setting.capitalize()} / {stage_title}\n'+('Descendant erasure' if erase else 'Donor subspace swap'),fontsize=10)
        ax.set_xticks(range(len(RANKS)),RANKS);ax.set_ylim(-4,106);ax.grid(alpha=.15)
        ax.spines[['top','right']].set_visible(False)
        if row==1:ax.set_xlabel('Subspace rank')
        if col==0:ax.set_ylabel('Counterfactual answer success (%)')
        if col==2:ax.set_ylabel('Original answer retained (%)')
        if row==0 and col in (0,2):ax.legend(fontsize=7,loc='center left')
fig.suptitle('Direct N100: reachable-descendant edits at source embedding vs L1 post-attention\nFixed target, changed-source donor; original-correct test pairs; query-only probe split (node holdout in report)',fontsize=12)
fig.savefig(HERE/'descendant_early_causal.png',dpi=190);fig.savefig(HERE/'descendant_early_causal.pdf');plt.close(fig)

lines=['# 可达后代方向：起点 embedding 与 L1 attention 后的因果干预','',
    '结果：可达后代探针方向在embedding上的移植可影响答案，但同时改变节点身份；L1 attention后rank32移植在简单/困难图上均只改变0.8%的答案，尽管独立后代读出已发生变化。困难图rank64的效果还低于同范数随机对照，未证明后代子空间是主要、特异的因果中介。简单图L1 attention后完整起点替换的正对照仅8%，该位置阴性不能解释为起点后代不重要。','',
    '额外核验：困难图embedding rank32擦除使固定阈值集合F1从0.994降至0.016，但原集合查询内节点排序AUC仍为0.960；因此不能声称已清除后代信息。擦除后模型原答案保留99.2%的阴性结果也不能否定后代关系的使用。','',
    '## 设置','',
    '- 用户的“可达后继”按严格可达后代集合 D(s)={v:v≠s 且 s可达v}实现，不是一跳直接后继。',
    '- 旧N100简单/困难固定图Direct验证最佳checkpoint，与上一轮祖先实验一致；不是最新N160限制暴露模型。',
    '- 改起点位置(-3)，目标保持不变。供体c=(s\u2032,t)，原查询a=(s,t)，答案相反；沿用原crossed-square test面板、四问原本均答对的固定子集，简单238对、困难119对。a为原query-test，c可来自原任意query split，需与上一轮同起点换目标的供体区别。',
    '- 位置为输入embedding、L1 attention残差相加后且FFN前。只替换这一输入位置的状态，不改全局embedding或绑定输出头。attention后通过修改attention增量来同时替换FFN输入及skip残差，不只修改norm2输入。',
    '- 构造方向与独立审计probe分别用query-train不相交两半，val选正则/集合阈值；query-only分支仍共享节点身份。另做source节点60/20/20留出，两个起点均须属于test-node，3个划分seed17/31/47。',
    '- 子空间投影P来自后代解码器权重的左奇异向量。移植 h+P(hd-h)，补空间 h+(I-P)(hd-h)，擦除 h-P(h-mu)。rank1/2/4/8/16/32/64完整报告，不按test选rank。',
    '- 三个固定随机子空间seed与三个打乱“起点→后代标签”子空间seed，每例匹配相应后代干预的编辑L2范数；不约束编辑后总状态范数。身份审计单独用全query-train拟合以保证身份可以测量。',
    '- 冻结推理CPU FP32/2线程，拟合GPU0。不训练主模型、不保存activation或probe/checkpoint。','',
    '## Query-only：答案按供体关系改变','',
    '|图/位置|rank|后代移植|补空间移植|随机均值|打乱标签均值|全起点替换|','|---|---:|---:|---:|---:|---:|---:|']
for setting in DATA:
    for stage in ('embedding','L1_attention'):
        entry=primary(setting,stage)
        for rank in (32,64):
            scores=[mean(entry,method,r,'counterfactual_success') for method,r in [('descendant_swap',rank),('complement_swap',rank),('random_swap',rank),('shuffled_swap',rank),('full_swap',0)]]
            lines.append(f'|{setting}/{stage}|{rank}|'+'|'.join(f'{v*100:.1f}%' for v in scores)+'|')
lines+=['','简单图attention后全替换只有8%，而embedding全替换100%；已排除把该位置的低子空间效果解释成后代信息不重要。此前起点信息可能已经被L1 attention复制到目标/分隔符；单位置替换不能撤回这些副本。困难图attention后全替换99.2%，在此位置的子空间/补空间比较更有解释力。','',
    '## 擦除：原答案保留率','',
    '|图/位置|rank|后代擦除|随机擦除均值|打乱标签擦除均值|','|---|---:|---:|---:|---:|']
for setting in DATA:
    for stage in ('embedding','L1_attention'):
        entry=primary(setting,stage)
        for rank in (32,64):
            scores=[mean(entry,method,rank,'edited_original_accuracy') for method in ('descendant_erase','random_erase','shuffled_erase')]
            lines.append(f'|{setting}/{stage}|{rank}|'+'|'.join(f'{v*100:.1f}%' for v in scores)+'|')
lines+=['','## rank32 移植的独立审计','',
    '|图/位置|供体后代集合F1|目标后代归属符合供体关系|原起点身份保留|模型答案改变|','|---|---:|---:|---:|---:|']
for setting in DATA:
    for stage in ('embedding','L1_attention'):
        metrics=values(primary(setting,stage),'descendant_swap',32)[0]
        keys=('independent_descendant_donor_F1','independent_target_membership_counterfactual_accuracy','source_identity_retained','counterfactual_success')
        lines.append(f'|{setting}/{stage}|'+'|'.join(f'{metrics[k]*100:.1f}%' for k in keys)+'|')
lines+=['','集合F1排除原起点、供体起点、目标；目标是否属于后代集合另单独评估，二者分母不同。审计probe不参与方向构造，但query-only版本仍见过相同节点身份。原生模型最终argmax非YES/NO计失败，不把强制二选一冒充合法输出。','',
    '## 擦除是否真的改变了独立读出','',
    '|图/位置|rank|原后代集合F1 前→后|原集合节点排序AUC 前→后|目标后代归属原关系准确率 前→后|模型原答案保留|','|---|---:|---|---|---|---:|']
for setting in DATA:
    for stage in ('embedding','L1_attention'):
        for rank in (32,64):
            m=values(primary(setting,stage),'descendant_erase',rank)[0]
            lines.append(f"|{setting}/{stage}|{rank}|{m['independent_descendant_original_F1_before']:.3f} → {m['independent_descendant_original_F1_after']:.3f}|{m['original_set_within_query_AUC_before']:.3f} → {m['original_set_within_query_AUC_after']:.3f}|{m['independent_target_membership_original_accuracy']*100:.1f}% → {m['independent_target_membership_edited_original_accuracy']*100:.1f}%|{m['edited_original_accuracy']*100:.1f}%|")
lines+=['','此处AUC是每个查询内部比较原后代节点与非后代节点的排序，再对查询平均，仅用于同一干预前后的阈值无关审计；不同于此前固定候选、跨留出焦点节点的主probe AUROC，且可能受节点频率/拓扑顺序影响。F1大幅下降而排序仍高时，不能称后代信息被清除。即使这一个审计probe排序也下降，仍可能有其他方向、非线性读法或冗余副本。','',
    '## 留出起点身份的敏感性对照','',
    '|图/位置|node seed|配对数|后代移植rank32|随机均值|打乱标签均值|有效W rank|与打乱子空间重叠|','|---|---:|---:|---:|---:|---:|---:|---:|']
for setting,d in DATA.items():
    for stage,entries in d['stages'].items():
        for entry in entries[1:]:
            if not entry['n']:
                lines.append(f"|{setting}/{stage}|{entry['node_seed']}|0|—|—|—|—|—|");continue
            scores=[mean(entry,method,32,'counterfactual_success') for method in ('descendant_swap','random_swap','shuffled_swap')]
            overlap=np.mean(entry['control_subspace_overlap']['32']['shuffled'])
            lines.append(f"|{setting}/{stage}|{entry['node_seed']}|{entry['n']}|"+'|'.join(f'{v*100:.1f}%' for v in scores)+f"|{entry['effective_rank_1e5']}|{overlap:.3f}|")
lines+=['','这些子集只有3–6对，部分划分没有配对；只能作敏感性记录，不能给出稳定独立节点因果结论。重叠为||U_desc^T U_shuffled||_F²/rank；随机基线约rank/128。不能把高rank或与打乱标签高度重合的方向当作纯后代语义。','',
    '## 实例：审计目标归属改变，模型答案不变','']
for setting,d in DATA.items():
    for example in d['examples']:
        if example['stage']!='L1_attention':continue
        lines.append(f"- {setting}，{example['source']}→{example['target']}，供体起点{example['donor_source']}：原关系{example['original_label']}，供体关系{example['desired_label']}；审计目标后代分数{example['target_descendant_score_before']:.3f}→{example['target_descendant_score_after']:.3f}，阈值{example['threshold']:.3f}，模型仍答{example['edited_answer']}。分数不是概率。")
lines+=['','## 验证与边界','']
for setting,d in DATA.items():
    checks=d['checks'];assert checks['embedding_and_tied_head_unchanged']
    assert checks['observer_margin_error']==0
    assert checks['embedding']['full_swap_vs_natural_donor_margin_error']<1e-4
    for stage in ('embedding','L1_attention'):
        assert checks[stage]['noop_margin_error']<1e-4 and checks[stage]['patched_residual_error']<1e-4
    donor_fraction=values(primary(setting,'embedding'),'no_op',0)[0]['donor_query_test_fraction']
    lines.append(f'- {setting}：observer/no-op检查通过；embedding整替换与自然供体margin误差为{checks["embedding"]["full_swap_vs_natural_donor_margin_error"]:.2g}；attention实际FFN输入误差{checks["L1_attention"]["patched_residual_error"]:.2g}；主面板供体原query-test比例{donor_fraction*100:.1f}%。')
lines+=['','两图各一个固定图/模型，不能做总体稳健因果结论。子空间移植同时影响节点身份、直接后继/祖先/深度等读出，后者全体配对的聚合分数进展保存于JSON，不以接近零供体差值逐对相除。读出变化与答案变化之间的分离不能证明模型完全不用后代关系，也不能从embedding移植有效证明它只用后代关系。', '',
    '复现：repo根目录运行 `.venv/bin/python studies/descendant_early_causal/run.py`，再运行 `summarize.py`。共享前向/拟合原语来自 `ancestor_early_causal/run.py`，新增显式position参数默认仍为目标位置，保留旧实验行为。','']
(HERE/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
print(HERE/'REPORT.md')

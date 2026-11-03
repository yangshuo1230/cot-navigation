"""Report validation-selected causal subspaces, separate output-writing control."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
HERE=Path(__file__).resolve().parent
DATA={s:json.loads((HERE/f'{s}.json').read_text()) for s in ('simple','hard')}
assert all(d.get('complete') for d in DATA.values()),'Wait for both complete model runs'
RANKS=DATA['simple']['ranks']
SITES=[('source','E'),('source','1A'),('source','1F'),('target','E'),('target','1A'),('target','1F'),('target','2F')]
fig,axes=plt.subplots(2,2,figsize=(11.5,8),layout='constrained')
for column,(setting,d) in enumerate(DATA.items()):
    selected=d['candidates'][d['selected_candidate_index']]
    grid=np.zeros((len(SITES),len(RANKS)))
    for c in d['candidates']:grid[SITES.index(tuple(c['site'])),RANKS.index(c['rank'])]=c['test']['swap']['success']
    ax=axes[0,column];im=ax.imshow(grid,cmap='Blues',vmin=0,vmax=1,aspect='auto')
    for i in range(len(SITES)):
        for j in range(len(RANKS)):
            ax.text(j,i,f'{100*grid[i,j]:.1f}',ha='center',va='center',fontsize=9,color='white' if grid[i,j]>.6 else 'black')
    si=SITES.index(tuple(d['selected_site']));ri=RANKS.index(d['selected_rank'])
    ax.add_patch(Rectangle((ri-.5,si-.5),1,1,fill=False,edgecolor='#e17c21',lw=2.5))
    ax.set_xticks(range(len(RANKS)),RANKS);ax.set_yticks(range(len(SITES)),[r+' '+s for r,s in SITES])
    ax.set_xlabel('Rank');ax.set_title(setting.capitalize()+': held-out query counterfactual success (%)',fontsize=11)
    fig.colorbar(im,ax=ax,fraction=.04,pad=.02)
    ax=axes[1,column];t=selected['test'];random=[v['success'] for v in t['random_equal_norm']]
    numbers=[t['swap']['success'],np.mean(random),t['complement']['success'],t['full']['success'],d['fresh_initialization']['test']['swap']['success']]
    bars=ax.bar(range(5),np.array(numbers)*100,color=['#2675b9','#999999','#d87929','#31915c','#78a9d0'])
    ax.errorbar(1,np.mean(random)*100,yerr=np.std(random)*100,fmt='none',ecolor='black',capsize=3)
    ax.bar_label(bars,fmt='%.1f',fontsize=9,padding=3)
    ax.set_xticks(range(5),['Learned','Random\nequal norm','Complement','Full swap','Fresh init'])
    ax.set_ylim(0,110);ax.set_ylabel('Test counterfactual success (%)')
    ax.set_title(f"Validation-selected: {' '.join(d['selected_site'])}, rank {d['selected_rank']}",fontsize=11)
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
fig.suptitle('Direct N100: learned shared low-dimensional causal subspaces\nNatural-donor coefficients, no amplification; original query train/val/test disjoint; orange box selected on validation',fontsize=12)
fig.savefig(HERE/'causal_subspace_search.png',dpi=190);fig.savefig(HERE/'causal_subspace_search.pdf');plt.close(fig)

lines=['# Direct：共享低维因果子空间搜索','',
    '结果：已找到可跨测试查询定向控制答案的共享低维子空间。验证选中简单图target L1 FFN后rank1（test96.9%，等范数随机0.3%），困难图target embedding rank8（test91.2%，随机5.1%）。简单图节点留出重训仍95.0%–98.3%；困难图降至62.8%–85.7%，稳健性较弱。','',
    '本实验直接优化低维状态移植的答案效果，不使用祖先/后代标签定义方向。跨查询共享同一个正交投影；不按测试效果挑选位置/维数。完整测试网格为描述性结果，主结论只使用验证集选定配置及独立初始化/节点留出检查。','',
    '## 方法','',
    '- 旧N100简单/困难Fixed DAG Direct验证最佳checkpoint，权重全部冻结。只训练128×r的参数矩阵，每次QR得到正交基U，r=1/2/4/8/16。不是训练或微调主模型。',
    '- 起点干预用同目标、换起点供体；目标干预用同起点、换目标供体；真实关系相反。所有训练配对两条查询都来自原train，验证配对两条都来自val，测试配对两条都来自test；查询集合严格不交叉。',
    '- 图定义候选配对、固定每个接收查询最多两次；取自然原/供体均正确且输出合法的面板，正反方向成对加入，期望YES/NO各50%。统计不是独立模型重复。',
    '- 仅替换一个位置：h_edit=h+UUᵀ(h_donor-h)。供体系数来自自然状态，不学习任意offset、每查询方向或放大倍率。所有查询共享同一个U。',
    '- 扫描起点E/1A/1F、目标E/1A/1F/2F。A是attention残差相加后、FFN之前，F是FFN残差相加后。后续计算重新执行。',
    '- GPU0 FP32优化，只用训练配对的供体答案作完整词表交叉熵；Adam lr0.02、batch64、300步。100/200/300步仅按val原词表argmax成功率选checkpoint。',
    '- 预定选择：优先最小rank且val成功≥90%、较等范数随机高≥20pp；并列比较val随机对照差值。无满足者则选择val对随机差值最大者、并列偏好低rank。test不参与优化/选择。',
    '- 三随机seed201/303/407，每对编辑L2范数与学习子空间严格匹配。另测补空间、全状态替换和no-op。反事实成功要求原生完整词表argmax等于供体真实YES/NO，非法token计失败。',
    '- 选定site/rank额外用全新初始化重训，及3次焦点节点60/20/20留出重训。主模型仍训练过全部节点，留出只作用于新学习子空间。','',
    '## 按验证选定的主结果','',
    '|图|位置|rank|val成功|test成功|test YES/NO|随机均值±SD|补空间|全替换|独立初始化|编辑范数/全供体差值|',
    '|---|---|---:|---:|---:|---|---|---:|---:|---:|---:|']
for setting,d in DATA.items():
    selected=d['candidates'][d['selected_candidate_index']];t=selected['test'];r=np.array([v['success'] for v in t['random_equal_norm']]);m=t['swap']
    lines.append(f"|{setting}|{' '.join(d['selected_site'])}|{d['selected_rank']}|{selected['validation']['swap']['success']*100:.1f}%|{m['success']*100:.1f}%|{m['yes_success']*100:.1f}% / {m['no_success']*100:.1f}%|{r.mean()*100:.1f}% ± {r.std()*100:.1f}pp|{t['complement']['success']*100:.1f}%|{t['full']['success']*100:.1f}%|{d['fresh_initialization']['test']['swap']['success']*100:.1f}%|{m['edit_to_full_difference_ratio']:.3f}|")
lines+=['','低维不等于微小扰动：所选简单/困难配置的编辑长度分别约为完整供体差值的70%/49%。本实验没有任意放大编辑幅度，但不能把它说成极小范数变化。','',
    '## 全部候选测试网格','',
    '|图/位置|rank1|rank2|rank4|rank8|rank16|','|---|---:|---:|---:|---:|---:|']
for setting,d in DATA.items():
    for site in SITES:
        row=[next(c for c in d['candidates'] if tuple(c['site'])==site and c['rank']==r)['test']['swap']['success'] for r in RANKS]
        lines.append(f"|{setting}/{' '.join(site)}|"+'|'.join(f'{v*100:.1f}%' for v in row)+'|')
lines+=['','网格中的test峰值不作为主配置；不同角色的配对面板不同，也不能把数值差完全归因于干预位置。','',
    '## 选定位置/维数的焦点节点留出','',
    '|图|seed|train/val/test配对数|test节点数|test学习子空间|随机均值|全替换|','|---|---:|---|---:|---:|---:|---:|']
for setting,d in DATA.items():
    for item in d['heldout_nodes']:
        counts='/'.join(str(v['n']) for v in item['panels'])
        if 'skipped' in item:lines.append(f"|{setting}|{item['node_seed']}|{counts}|—|支持量不足|—|—|");continue
        t=item['test'];random=np.mean([r['success'] for r in t['random_equal_norm']])
        lines.append(f"|{setting}|{item['node_seed']}|{counts}|{item['panels'][2]['focal_nodes']}|{t['swap']['success']*100:.1f}%|{random*100:.1f}%|{t['full']['success']*100:.1f}%|")
lines+=['','## 最后层分隔符：答案写出正对照','',
    '预先用原YES/NO输出权重差、final LayerNorm gamma及去均值构造一个1维方向。只在最终分隔符状态移植其供体分量，不用它竞争上游子空间搜索。','',
    '|图|1维输出方向反事实成功|全分隔符替换|','|---|---:|---:|']
for setting,d in DATA.items():
    t=d['last_separator_control'];lines.append(f"|{setting}|{100*t['swap']['success']:.1f}%|{100*t['full']['success']:.1f}%|")
lines+=['','该对照能改变YES/NO输出并不说明图知识或推理过程只有1维；它只是答案写出通道。上游低维干预有效也可能控制已形成的联合答案/身份消息，而非隔离某个图关系集合。','',
    '## 支持量、验证与范围','']
for setting,d in DATA.items():
    for role,panels in d['panels'].items():
        lines.append(f"- {setting}/{role}：train/val/test配对数 {'/'.join(str(v['n']) for v in panels)}；test涉及{panels[2]['unique_queries']}条不同查询、{panels[2]['focal_nodes']}个焦点节点。")
    assert d['checks']['embedding_and_tied_head_unchanged']
    for c in d['candidates']:
        assert c['test']['noop']['success']<.02
        assert c['test']['swap']['edit_to_full_difference_ratio']<=1.0001
        if c['site'][1]=='E':assert c['test']['full']['success']>.98
lines+=['','模型参数梯度始终为None，训练后全局embedding与绑定输出头逐位未变；正交误差<1e-5；自然embedding全替换与no-op控制通过。旧loader的inference tensor被正常tensor副本重建，以允许输入梯度经过冻结层，未改变数值权重。', '',
    '一次初始化或某个负结果不是存在性定理/不可能性证明；未达到高成功不代表不存在其他低维子空间。高成功则仅支持本checkpoint、同图查询分布上的共享低维控制通道。需要结合留出节点、随机对照、独立初始化和早晚位置解释；不声称纯祖先/后代语义、跨图算法或全部任务信息的内在维度。', '',
    '仅保存simple.json、hard.json、关键PNG/PDF、报告与脚本；不保存优化器、basis权重、activation或主模型checkpoint，不创建W&B run。复现：repo根目录运行run.py及summarize.py。','']
(HERE/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
print(HERE/'REPORT.md')

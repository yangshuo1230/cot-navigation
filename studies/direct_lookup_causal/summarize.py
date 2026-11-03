"""Two key figures and a self-contained, qualified experiment report."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path.cwd()/'studies/direct_lookup_causal'

def pct(x):return 'NA' if x is None else f'{100*x:.1f}%'
def ms(values):return f'{np.mean(values):.3f} ± {np.std(values):.3f}'
def percent_ms(values):return f'{100*np.mean(values):.1f}% ± {100*np.std(values):.1f}pp'
def get(r,method,l,rank=None,group='all_four_correct',split='test'):
    rows=r['causal']['curves'] if 'causal' in r else r['curves']
    z=next(z for z in rows if z['split']==split and z['method']==method and z['layer']==l and (rank is None or z.get('rank')==rank))
    return z['result'][group]['counterfactual_accuracy']
def follow_stat(f,method,l,group='all_four_correct'):
    return next(z['result'][group] for z in f['curves'] if z['split']=='test' and z['method']==method and z['layer']==l)
def ancestor(r,role,l):
    return [v['features']['target_and_separator'][role*7+l]['earlier_topological_nodes']['fixed_candidate_across_targets']['auc'] for v in r['readouts']['ancestor_target_heldout']]

def main():
    results={s:json.loads((HERE/f'{s}.json').read_text()) for s in ('simple','hard')}
    follow={s:json.loads((HERE/f'{s}_followup.json').read_text()) for s in results}
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,(s,r) in zip(axes,results.items()):
        f=follow[s]
        for method,label,color in [('same_source_other_target_residual','b: same source, new target','tab:blue'),('same_target_other_source','c: new source, same target','tab:orange'),('full_target','d: new source AND target','tab:green')]:
            data=f if method.startswith('same_source') else r
            ax.plot(range(7),[get(data,method,l) for l in range(7)],'o-',color=color,label=label)
        ax.set(title=s,xticks=range(7),xticklabels=['E']+[f'L{i}' for i in range(1,7)],ylim=(-.04,1.05),ylabel='Counterfactual answer success',xlabel='Replaced target residual after layer')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(HERE/'causal_target_states.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained');ranks=[1,2,4,8,16,32,64]
    for col,(s,r) in enumerate(results.items()):
        for method,label,color in [('ancestor_subspace','Ancestor readout directions','tab:blue'),('ancestor_complement','Remaining directions','tab:orange'),('random_equal_norm','Equal-norm random directions','tab:green')]:
            axes[0,col].plot(range(len(ranks)),[get(r,method,1,k) for k in ranks],'o-',label=label,color=color)
            values=[next(z for z in r['causal']['curves'] if z['split']=='test' and z['method']==method and z['layer']==1 and z['rank']==k)['result']['ancestor_reader_desired_score_shift'] for k in ranks]
            axes[1,col].plot(range(len(ranks)),values,'o-',label=label,color=color)
        axes[0,col].set(title=s+': target L1 intervention',ylabel='Counterfactual answer success',ylim=(-.04,1.05))
        axes[1,col].set(ylabel='Desired ancestor score shift (NOT probability)',xlabel='Subspace rank')
        for row in (0,1):
            axes[row,col].set(xticks=range(len(ranks)),xticklabels=ranks);axes[row,col].grid(alpha=.2)
        axes[0,col].legend(fontsize=8)
    fig.savefig(HERE/'ancestor_subspace.png',dpi=180);plt.close(fig)
    lines=['# Direct：固定图关系查找假设的读出与因果实验','',
        '结论：尚未证明“模型存储完整后继集合、祖先集合，再做一次集合组合”这一强假设。目标相关结构信息确实可读出，部分方向也能影响答案；但把读出定义的祖先方向当成实际因果中介，出现了明确反例。简单图目标位置很早已编码源目标联合信息；困难图第 1 层更接近独立的节点编码，后续层才形成联合答案。',
        '', '## 模型与范围','',
        '原 N100 固定图各一张：简单分层 DAG M150；困难多尺度 DAG M200。原验证最佳 Direct checkpoint，6 层、1 attention head、hidden 128、FFN 512。沿用原 forward 查询 train/val/test，不包含 reverse。固定图本身曾被原模型训练见过，以下是机制实验，不是新图泛化。冻结 FP32 native SDPA 推理 GPU1，线性拟合 GPU0。没有重新训练原模型、生成训练 checkpoint 或上传 W&B。',
        '', '| 图 | best step | 原测试集准确率 | FP32 与原 BF16 答案一致率 |','|---|---:|---:|---:|']
    for s,r in results.items():lines.append(f"| {s} | {r['manifest']['checkpoint_step']} | {pct(r['native_test']['accuracy'])} ({r['native_test']['n']}) | {pct(r['native_test']['agreement_with_prior_bf16'])} |")
    lines+=['','## 1. 多跳祖先相关信息能否读出','',
        '目标标签为严格祖先 A(t)。仿射 ridge 用训练查询拟合，val 选正则；额外同时留出目标身份与 query，三个 seed。固定候选 u，只比较拓扑序早于目标的不同目标 t，检查 u→t 为真时得分是否更高；按目标先聚合，避免重复 query 加权。排除当前 source 与 target 位置，固定的祖先频率在此指标恰为 AUROC .5。不是准确率，也不代表完整集合恢复。',
        '', '| 图 | 目标 E | 目标 L1 | 目标 L4 | 分隔符 L1 | 分隔符 L4 | 随机目标身份 |','|---|---|---|---|---|---|---|']
    for s,r in results.items():
        rnd=[v['features']['random_target_identity'][0]['earlier_topological_nodes']['fixed_candidate_across_targets']['auc'] for v in r['readouts']['ancestor_target_heldout']]
        vals=[ms(ancestor(r,0,l)) for l in (0,1,4)]+[ms(ancestor(r,1,l)) for l in (1,4)]+[ms(rnd)]
        lines.append('| '+s+' | '+' | '.join(vals)+' |')
    lines+=['','源节点后继集合另用 probe 未见的 source 身份测试（原模型仍见过）。以下为 top-出度 recall，三个划分的均值：','',
        '| 图 | source E | source L1 | source L6 |','|---|---:|---:|---:|']
    for s,r in results.items():
        v=r['readouts']['successor_source_heldout'];lines.append('| '+s+' | '+' | '.join(pct(np.mean([z['layers'][l]['recall_at_true_k'] for z in v])) for l in (0,1,6))+' |')
    lines+=['','目标 embedding 已有祖先相关信号，但尚不能排除深度、秩等结构统计。所有 probe 都有额外图标签监督；普通 query 划分下的高分会受到目标身份查表影响，不能当模型能力或因果使用证据。',
        '', '## 2. 交叉查询的因果组合设计','',
        '构造四个自然查询：a=(sa,ta), b=(sa,tb), c=(sb,ta), d=(sb,tb)，满足 R(a)=R(d) ≠ R(b)=R(c)。a 为接收查询，b 为期望的新目标查询，d 为跨起点供体，其答案与 a 相同。若移植 d 的目标状态后模型输出 b 的正确答案，就无法用复制 d 的答案解释。对照 c 则只改起点、保留目标。',
        '', 'a、b 都必须来自同一原 val/test 分区；c、d 可来自任一原分区。额外标出四个查询都来自 heldout 的严格小面板。选例只看图标签，不看模型预测。以四个查询原本全答对的样本作为主要机制面板，同时保留未筛选总面板结果。每个接收 query 最多出现两次，配对之间不独立。',
        '', '| 图 | 测试配对总数 | 独立接收 query | 四问均答对 | 同深度同 rank band 且全答对 | 四问均 test 且全答对 |','|---|---:|---:|---:|---:|---:|']
    for s,r in results.items():
        p=r['causal']['panels']['test'];curve=next(z for z in r['causal']['curves'] if z['split']=='test' and z['method']=='full_target' and z['layer']==0)['result']
        lines.append(f"| {s} | {p['n']} | {p['unique_recipient_queries']} | {p['all_four_correct']} | {curve['matched_depth_band']['n']} | {curve['all_four_heldout']['n']} |")
    lines+=['','困难图没有满足目标同实际深度且同 rank band 的交叉配对，不能声称这一项结构捷径已被排除。严格 all-four-test 面板也较小，主要面板只要求接收与反事实查询 test。',
        '', '## 3. 目标状态是目标自己的集合，还是查询联合信息','',
        '把 a 的目标位置、第 1 层后的整个 residual 替换为 b/c/d 的同位置状态。下表是输出变成 b 的正确答案的比例；分母仅四问原本全答对。',
        '', '| 供体 | 如果状态只表示目标自身 | 如果状态已表示供体查询答案 | 简单图实际 | 困难图实际 |','|---|---|---|---:|---:|']
    for method,label,goal,joint in [('same_source_other_target_residual','b：同起点，换目标','应改变','应改变'),('same_target_other_source','c：换起点，同目标','应不变','应改变'),('full_target','d：起点目标都换','应改变','应不变')]:
        values=[get(follow[s] if method.startswith('same_source') else results[s],method,1) for s in results]
        lines.append('| '+label+' | '+goal+' | '+joint+' | '+' | '.join(pct(v) for v in values)+' |')
    lines+=['',
        '简单图接近“联合信息”模式：目标相同、只换上下文起点的状态，足以改变答案。困难图第 1 层接近“目标自身”模式：跨起点的目标状态仍可组合到原查询。但这也可以是目标身份/其他图编码，并不单独证明祖先集合。后续层的转移曲线见 causal_target_states.png。',
        '', '查询答案仿射探针只用原 query-train 标签，不使用全图祖先标签。XOR 四问全部正确比率进一步排除只利用单端点偏好：任何独立 source-only、target-only 或 additive source+target score 都不能同时答对交叉四问。','',
        '| 图 | target L1 BA / XOR 四问全对 | target L4 BA / XOR 四问全对 | separator L2 BA / XOR 四问全对 | separator L6 BA / XOR 四问全对 |','|---|---|---|---|---|']
    for s,f in follow.items():
        vals=[]
        for role,l in [('target',1),('target',4),('separator',2),('separator',6)]:
            z=next(z for z in f['joint_answer_probes']['residual'] if z['role']==role and z['layer']==l)
            vals.append(pct(z['test']['balanced_accuracy'])+' / '+pct(z['square_all_four_accuracy_native_correct']))
        lines.append('| '+s+' | '+' | '.join(vals)+' |')
    lines+=['','简单图的 target L1 已能读出联合答案，困难图更晚形成联合信息。这里读出能力仍不等同于因果中介，但与上面的状态转移证据相互对应。',
        '', '## 4. 单独的 attention 信息边','',
        '只替换 target→separator 一条 attention 边的 V 消息；保持接收者 Q、K、其他边和目标 residual 不变。这样可确认一条信息通路，但 V-only 不能捕捉通过 key 改变注意力分配的作用，也不等价于整个目标状态替换。','']
    for s,f in follow.items():
        attention=f['attention']['receivers'];target=np.array(attention['target']['raw_mass']);bar=np.array(attention['answer_separator']['raw_mass'])
        lines.append(f"- {s}：目标 L1 关注 source {pct(target[0,1])}、自身 {pct(target[0,2])}；separator L2 关注目标 {pct(bar[1,2])}。separator 各层目标 raw mass = {', '.join(pct(v) for v in bar[:,2])}。距离校正版保留在 followup.json；固定长度 prompt 中角色与位置仍部分混淆。")
        for l in ([2] if s=='simple' else [2,5]):
            vals=[get(f,method,l) for method in ('same_source_other_target_edge','same_target_other_source_edge','other_source_other_target_edge')]
            lines.append(f"  target→separator V 边 L{l}：供体 b/c/d 的反事实成功率分别为 {' / '.join(pct(v) for v in vals)}。这些是层扫描，不把事后峰值作为独立确认实验。")
    lines+=['','## 5. 祖先读出方向是不是模型实际使用的中介','',
        '在目标 activation 上拟合全祖先线性解码器 W，SVD 得到 rank 1/2/4/8/16/32/64 的方向。祖先方向移植 h←h+P(h_d−h)，其余方向移植 h←h+(I−P)(h_d−h)。随机方向控制逐样本匹配祖先编辑的 L2 范数；打乱目标→祖先对应关系的控制也补做三个 seed 并匹配范数。身份解码器检查是否同时破坏目标身份，但身份读出不能保证其他身份信息未变。',
        '', '先看内部第 1 层的 rank32 对照（这是描述性固定层/rank切片，不是验证选出的主配置）：','',
        '| 图 | 替换祖先方向：最终答案改变 | 同时的祖先成员分数变化 | 替换其余方向：答案改变 | 同时的祖先成员分数变化 |','|---|---:|---:|---:|---:|']
    for s,r in results.items():
        vals=[]
        for method in ('ancestor_subspace','ancestor_complement'):
            z=next(z for z in r['causal']['curves'] if z['split']=='test' and z['layer']==1 and z['rank']==32 and z['method']==method)['result']
            vals.extend([pct(z['all_four_correct']['counterfactual_accuracy']),f"{z['ancestor_reader_desired_score_shift']:+.3f}"])
        lines.append('| '+s+' | '+' | '.join(vals)+' |')
    lines+=['','分数变化是朝新标签方向的平均 ridge score 变化，不是概率。尤其困难图中，可以强烈改变 probe 的祖先成员读出，但几乎不改变模型答案；其余方向反而能改变答案。因此不能把这个 probe 定义出的祖先方向直接当作模型在用的祖先集合中介。',
        '', '主配置按 val 上“祖先移植成功率减去随机/打乱控制均值”选择层与秩，固定后汇报 test。两图都选到 embedding 层，而不是内部层。额外控制使用三个新随机种子，均不重新选参数。','',
        '| 图 | val 选层/rank | 祖先移植成功 | 等范数随机 mean±sd | 等范数打乱 mean±sd | 移植后原目标身份保留 |','|---|---|---:|---|---|---:|']
    for s,r in results.items():
        sel=r['causal']['validation_selected'];z=next(z for z in sel['comparisons'] if z['method']=='ancestor_subspace')['result'];f=follow[s]
        rand=[z['result']['all_four_correct']['counterfactual_accuracy'] for z in f['replicated_norm_matched_controls'] if z['method']=='random']
        shuffle=[z['result']['all_four_correct']['counterfactual_accuracy'] for z in f['replicated_norm_matched_controls'] if z['method']=='shuffled_ancestors']
        lines.append(f"| {s} | E / {sel['rank']} | {pct(z['all_four_correct']['counterfactual_accuracy'])} | {percent_ms(rand)} | {percent_ms(shuffle)} | {pct(z['identity']['recipient_retained'])} |")
    lines+=['','这些方向影响答案的能力超过若干控制，但祖先方向也明显改变目标身份，无法证明是“保留身份、只改祖先集合”。身份保留率按整个 test 配对面板统计，成功率按四问均答对的子面板统计。身份保留子集的各方法分母不同，属于干预后的条件筛选，不用它作方法间公平效果比较。等范数控制三 seed 的标准差也不是置信区间。',
        '', '另外，在 val 选出的 embedding 层构造完整 W 的近似数值零空间，保留全部 100 个线性祖先读出分数，仅移植零空间部分：','',
        '| 图 | decoder rank / null dim | 祖先读出最大变化 | 答案反事实改变 |','|---|---|---:|---:|']
    for s,f in follow.items():
        z=f['exact_ancestor_readout_nullspace'];lines.append(f"| {s} | {z['decoder_matrix_rank']} / {z['null_dimension']} | {z['max_all_ancestor_score_change']:.2e} | {pct(z['result']['all_four_correct']['counterfactual_accuracy'])} |")
    lines+=['','这说明该线性祖先读出并未覆盖全部影响答案的信息。零空间编辑仍可能改变其他图信息、身份或离开自然 activation 分布，所以不把它解释为对所有形式的祖先表示的证伪。',
        '', '## 验证、可复现性与限制','',
        '- 全部 100 个源点的反向 DP 可达性标签均与独立 BFS 一致，查询标签与图一致。',
        '- embedding 整个目标交换，输出 margin 与真正修改输入目标的自然查询一致（<1e-4）；最后一层只换目标不能改变 separator 输出；同 V 的 attention 边交换必须 no-op。',
        '- FP32 原生完整词表 argmax 的 YES/NO 才计入成功，非法答案 token 计失败；同时保存 YES−NO margin。观察 attention hook 不改变原生 logits。',
        '- 原生 test 答案与既有 BF16 自由生成完全一致。模型没有 optimizer，模型参数被冻结，实验文件不包含 activations 或 checkpoint。',
        '- 每类图只分析一张固定图、一个原最佳模型；不把机制差异单独归因于图复杂度，因为初始化和训练轨迹也不同。',
        '- graph-aware 祖先探针有额外 oracle 标签，作用是定位候选方向，不是独立无泄漏任务求解器。强语义证明仍需要更具概念选择性的中介干预，而非一般目标身份替换。',
        '', '执行：先 `study.py`，后 `followup.py`，最后 `summarize.py`，均从远端 repo root 使用 `.venv/bin/python studies/direct_lookup_causal/文件名`。通用 patch 支持任意 sender/receiver 位置、任意层的 residual 或单条 attention V 边；本报告画图按当前 6 层模型。',
        '', '## 保守解释','',
        '固定图 Direct 可以利用训练后节点表示与少量网络层完成源目标关系匹配；不需要生成显式搜索轨迹。困难图有目标独立节点表示及多跳祖先相关信号，简单图很早把源目标联合关系写入目标位置。但这批因果实验没有证明模型显式存储完整后继/祖先集合，并以其交集作为主要答案计算；现有 probe 定义的祖先子空间尤其不能承担这个结论。',
        '', '数据：simple.json、hard.json（读出/主干预全部曲线）；simple_followup.json、hard_followup.json（联合答案、信息边、零空间和复现实验）；关键图只有 causal_target_states.png、ancestor_subspace.png。']
    examples=HERE/'counterexamples.json'
    if examples.exists():
        saved=json.loads(examples.read_text());lines+=['','## 一个具体的读出与行为分离例子','']
        for s,items in saved.items():
            if not items:continue
            e=items[0]
            lines.append(f"- {s} query {e['original_query_id']}：source {e['source']}，原目标 {e['old_target']}（图标签 {e['old_graph_label']}），移入目标 {e['new_target']}（图标签 {e['new_graph_label']}）。L1/rank32 祖先方向编辑后，源点成员读出从 {e['ancestor_readout_before']:.3f} 变为 {e['ancestor_readout_after_direction_edit']:.3f}，模型仍输出 {e['ancestor_direction_answer']}；只编辑其余方向后读出为 {e['ancestor_readout_after_complement_edit']:.3f}，模型却输出 {e['complement_answer']}。未校准分数不是概率；实例是事后按特征选出的说明，不是独立确认实验。")
        lines.append('完整少量实例及原始 query IDs 见 counterexamples.json；record_examples.py 可复现，没有 activation 文件。')
    for s,r in results.items():
        assert all(v<1e-4 for v in r['causal']['checks'].values())
        for z in r['causal']['curves']:
            if z['method']=='ancestor_subspace':
                other=next(o for o in r['causal']['curves'] if o['split']==z['split'] and o['layer']==z['layer'] and o['rank']==z['rank'] and o['method']=='random_equal_norm')
                assert np.isclose(z['result']['mean_edit_norm'],other['result']['mean_edit_norm'],rtol=1e-6,atol=1e-6)
        assert follow[s]['attention']['observer_margin_error']<1e-4
        assert follow[s]['exact_ancestor_readout_nullspace']['max_all_ancestor_score_change']<1e-4
        lines.append(f"\n{s} checkpoint SHA256: {r['manifest']['checkpoint_sha256']}")
    (HERE/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines))

if __name__=='__main__':main()

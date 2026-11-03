"""Compact figures/report and independent graph/metric verification."""
from analyze import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def bfs(adj,s):
    seen={s};todo=[s]
    for u in todo:
        for v in adj[u]:
            if v not in seen:seen.add(v);todo.append(v)
    return seen

def enrich(setting):
    p=HERE/f'{setting}.json';r=json.loads(p.read_text())
    cfg=load_config([Path(r['manifest']['run'])/'resolved_config.yaml'])
    tok=Tokenizer.from_config(cfg);world=ParametricWorld(Path(cfg.data.root)/'main-seed0')
    adj=world.adj();nodes=world.node_vocab_ids(tok);inv={int(v):i for i,v in enumerate(nodes)}
    reach=np.eye(100,dtype=bool)
    assert all(v>u for u in range(100) for v in adj[u])
    for u in reversed(range(100)):
        for v in adj[u]:reach[u]|=reach[v]
    for u in range(100):assert set(np.flatnonzero(reach[u]))==bfs(adj,u)
    first=[]
    for line in gzip.open(OLD/f'{setting}_path_queries.jsonl.gz','rt'):
        row=json.loads(line)
        if row['split']!=2 or row['label']!=1 or len(row['trace'])<2:continue
        u=row['source'];t=row['target'];v=inv.get(row['trace'][1],-1)
        assert inv[row['trace'][0]]==u
        good={v for v in adj[u] if reach[v,t]}
        if not good or len(good)==len(adj[u]):continue
        first.append(dict(q=row['query_ids'],u=u,t=t,v=v,good=good,correct=v in good,trace=row['trace'],pred=row['pred']))
    pair_queries=set();bad_queries=set();badexamples=[];pair_count=0
    for a,b in itertools.combinations(first,2):
        if a['u']!=b['u'] or a['good']&b['good']:continue
        pair_count+=1;pair_queries|={a['q'],b['q']}
        for z in (a,b):
            if not z['correct']:
                bad_queries.add(z['q'])
                if len(badexamples)<5 and not any(e['query']==z['q'] for e in badexamples):
                    badexamples.append(dict(query=z['q'],current=tok.name(nodes[z['u']]),target=tok.name(nodes[z['t']]),
                        useful=[tok.name(nodes[v]) for v in sorted(z['good'])],chosen=tok.name(nodes[z['v']]),
                        generated=tok.decode(z['trace']),final_pred=z['pred']))
    assert pair_count==r['target_pairs']['n']
    r['target_pairs'].update(unique_queries=len(pair_queries),unique_wrong_queries=len(bad_queries),
        first_mixed_test_queries=len(first),first_mixed_correct=sum(z['correct'] for z in first),
        failure_examples=badexamples,warning='Pairs share sources and queries; pair percentages are descriptive, not independent samples.')
    # Numerical/causal-capture validation: observation hooks have no output effect.
    assert r['observation_hooks_vs_native']['max_abs_logit_difference']==0
    assert abs(linear.auc([0,1],[0,0])-.5)<1e-12
    check=ranking(np.array([[1,0,0]]),np.array([[2,1,0]]))
    assert check['auc']==1 and check['top1_hit']==1
    r['verification']=dict(reachability='reverse DP equals independent BFS for all 100 sources',
        pair_counts_reconstructed=True,observer_exactly_preserves_native_logits=True,
        attention='reconstructed QK weights from native SDPA inputs; train-only exact-distance centering',
        replay_warning='Full-sequence BF16 replay and cached free generation differ slightly; saved native tokens define behavioral choices.')
    write(p,r);return r

def fmt(x):return f'{100*x:.1f}%'

def main():
    results={s:enrich(s) for s in ('simple','hard')}
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for row,(setting,r) in enumerate(results.items()):
        ax=axes[row,0];sub=r['sublayer_logit_lens'];xx=np.arange(13)
        ax.plot(xx,[z['successor_node_conditional_mass'] for z in sub],'o-',label='Mass on true successors (node-conditioned)')
        ax.plot(xx,[z['useful_within_successors']['top1_hit'] for z in sub],'s-',label='Useful next choice within successors')
        ax.axhline(sub[-1]['useful_within_successors']['random_top1'],ls=':',color='grey',label='Uniform successor choice')
        ax.set(xticks=xx,xticklabels=['E']+[s for l in range(1,7) for s in (f'A{l}',f'F{l}')],ylim=(-.03,1.04),title=f'{setting}: trained output head, no probe fit',ylabel='Probability mass / hit rate')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
        ax=axes[row,1];features=r['probes']['query_split']['features']
        ax.plot(range(7),[z['within_successors']['auc'] for z in features['residual']['useful']['layers']],'o-',label='Residual useful-successor probe')
        for color,(name,label) in zip(('tab:orange','tab:green','tab:red'),[('current_embedding','Current embedding'),('target_embedding','Target embedding'),('random_target_embedding','Random target identity')]):
            ax.axhline(features[name]['useful']['layers'][0]['within_successors']['auc'],ls='--',color=color,label=label)
        ax.set(xticks=range(7),xticklabels=['E']+[f'L{l}' for l in range(1,7)],ylim=(.45,1.02),title=f'{setting}: useful vs dead REAL successors',ylabel='Macro decision AUROC')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(HERE/'next_node_stages.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for row,(setting,r) in enumerate(results.items()):
        group=r['attention']['groups']['mixed_useful_dead']
        for col,(key,title) in enumerate([('raw_mass','Raw attention mass'),('distance_centered_mass','Distance-centered mass (signed)')]):
            ax=axes[row,col];a=np.asarray(group[key]);lo,hi=(0,1) if col==0 else (-.9,.9)
            im=ax.imshow(a,vmin=lo,vmax=hi,cmap='viridis' if col==0 else 'RdBu_r',aspect='auto')
            ax.set(xticks=range(6),xticklabels=['syntax','source','target','bar','past path','current'],yticks=range(6),yticklabels=[f'L{i}' for i in range(1,7)],title=f'{setting}: {title}')
            for i in range(6):
                for j in range(6):ax.text(j,i,f'{a[i,j]:.2f}',ha='center',va='center',fontsize=9,color='white' if abs(a[i,j])>.5 else 'black')
            fig.colorbar(im,ax=ax,shrink=.8)
    fig.savefig(HERE/'attention_roles.png',dpi=170);plt.close(fig)
    lines=['# Fixed N100 Path 定向选择分析','',
        '范围：简单分层 DAG 与困难多尺度 DAG 各一张 N100 固定图；使用原有验证集最佳 Path checkpoint。每个图重用 4500 条原始自由生成查询。全部前向查询，不包含 reverse 查询。观察当前位置预测下一个节点的 activation；不包含终止位置、答案、目标到达后的 token，也不把非法历史之后的状态纳入探针。',
        '', '## 主要结论','',
        '结果支持一种候选机制：前层首先提升真实后继的输出评分，后层结合目标信息完成选择。两张图的层位置不同。简单图的第 1 层 FFN 后后继评分突出，第 2 层读取目标后选择改善；困难图第 1 层 FFN 后形成后继偏好，第 2 层主要读取分隔符，目标相关筛选主要发生在第 4、6 层。',
        '', '这是分层输出评分与注意力模式提供的机制线索，尚未通过激活替换证明必要性。不能把它表述为模型在执行显式搜索，也不能据此排除固定图上的邻接表、可达性或路径记忆。',
        '', '## 输出头观察（无需训练探针）','',
        'A/F 分别为 attention/FFN 残差之后。使用原模型 final_norm 与 LM head 的未校准 logit lens。后继概率是对本图 100 个节点重新归一化后的总概率，并非完整词表概率；有用选择率只在真实后继中比较，并仅统计同时存在有用与无用后继的位置。有用 = 该后继等于目标或能到达目标。所有聚合以决策位置为单位，长轨迹贡献更多位置，不是整条 query 的准确率。后继质量在全部 test 决策上平均，有用选择率在 mixed 子集上平均，两者样本分母不同。','',
        '| 图 | 位置 | 后继节点概率质量 | 后继内选中有用节点 |','|---|---|---:|---:|']
    for setting,r in results.items():
        for i in (0,1,2,3,4,6,8,12):
            z=r['sublayer_logit_lens'][i];lines.append(f"| {setting} | {r['sublayer_names'][i]} | {fmt(z['successor_node_conditional_mass'])} | {fmt(z['useful_within_successors']['top1_hit'])} |")
    lines+=['','## 线性探针与身份对照','',
        '冻结模型后训练 100 输出的仿射 ridge：后继集合、有用后继集合、实际生成的下一节点。只用原 query train 拟合、val 选正则强度，test 汇报。特征归一化只用 train。分数不是概率。主要指标采用每个决策内部的 AUROC、Top-|真实集合| recall；对于定向能力，严格只比较真实后继中的有用与无用节点，避免大量非邻接节点带来虚高分。',
        '', '| 图 | 后继探针 E recall | 后继探针 L6 recall | 有用探针 E AUC | L2 AUC | L4 AUC | L6 AUC | 仅目标 embedding AUC | 随机目标身份 AUC |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for setting,r in results.items():
        f=r['probes']['query_split']['features'];ss=f['residual']['successor']['layers'];uu=f['residual']['useful']['layers']
        values=[fmt(ss[0]['recall_at_true_k']),fmt(ss[6]['recall_at_true_k'])]+[f"{uu[l]['within_successors']['auc']:.3f}" for l in (0,2,4,6)]+[f"{f[n]['useful']['layers'][0]['within_successors']['auc']:.3f}" for n in ('target_embedding','random_target_embedding')]
        lines.append('| '+setting+' | '+' | '.join(values)+' |')
    lines+=['',
        '重要限制：100 个节点、128 维输入，线性探针足以把节点身份当作查表索引。随机身份向量也能得到很高分，因此“embedding 能读出后继/可达性”不能直接证明 embedding 已经编码这些结构，更不能证明模型使用探针读出的信息。另做同时留出 query 与当前节点身份的 60/20/20 探针对照（seed 17）；这些节点仍被原模型训练过。',
        '', '| 图 | query 留出后继 L6 recall | 同时留出当前节点 L6 recall | mixed test 决策数 |','|---|---:|---:|---:|']
    for setting,r in results.items():
        q=r['probes']['query_split'];h=r['probes']['query_and_current_heldout']
        lines.append(f"| {setting} | {fmt(q['features']['residual']['successor']['layers'][6]['recall_at_true_k'])} | {fmt(h['features']['residual']['successor']['layers'][6]['recall_at_true_k'])} | {q['mixed_test']} |")
    lines+=['','## 注意力','',
        '从原生 SDPA 实际输入重建 QK softmax，保留 RoPE，观察 hook 对原生输出 logits 的差异为 0。只用 train 查询的无 padding、无答案前缀，按每个固定相对距离汇总全部因果 token 对的平均权重；test 权重减去对应距离均值。校正后值有符号，且不要求归一化为 1；这是描述性校正，不能消除所有位置效应。图中专门使用混合候选决策。','']
    for setting,r in results.items():
        a=r['attention']['groups']['mixed_useful_dead'];raw=np.asarray(a['raw_mass']);cen=np.asarray(a['distance_centered_mass'])
        lines.append(f"- {setting}：目标 raw mass L1–L6 = {', '.join(fmt(v) for v in raw[:,2])}；距离校正后 = {', '.join(f'{v:+.3f}' for v in cen[:,2])}；分隔符 raw mass = {', '.join(fmt(v) for v in raw[:,3])}。")
    lines+=['','困难图 L2 对分隔符的高注意力只说明它读取该位置的隐藏状态；分隔符可能承载此前已整合的查询信息，不能解释为它只在读取标点。简单图 L1 的 syntax 包括 WORLD 与 QUERY，应与直接关注图信息区别开。','',
        '## 同一合法起始历史，不同目标','',
        '只取 test 正例、第一 Path token 已复制起点的合法历史，匹配相同起点、不同目标且有用后继集合互不相交的查询。前缀只有同一个当前节点，改变的是输入目标，未注入非法 CoT。选择来自已保存的原生自由生成 token，避免用重放的近似 logits 替代实际行为。','']
    for setting,r in results.items():
        p=r['target_pairs'];lines.append(f"- {setting}：{p['n']} 对，涉及 {p['unique_sources']} 个起点、{p['unique_queries']} 个独立查询；下一节点发生改变 {fmt(p['fraction_choice_changes'])}，双方都选中有用后继 {fmt(p['fraction_both_useful'])}。这些配对共享查询，不能当独立样本。失败配对涉及 {p['unique_wrong_queries']} 个原始错误查询；全部首步 mixed 查询 {p['first_mixed_correct']}/{p['first_mixed_test_queries']} 正确。")
        for e in p['examples'][:1]:lines.append(f"  例：当前 {e['current']}，目标 {e['target_a']} 时有用后继 {e['useful_a']}，选择 {e['next_a']}；目标改为 {e['target_b']} 时有用后继 {e['useful_b']}，选择 {e['next_b']}。")
        for e in p['failure_examples'][:2]:lines.append(f"  错例：query {e['query']}，{e['current']} → {e['target']}，有用 {e['useful']}，实际选择 {e['chosen']}；完整生成 `{e['generated']}`，最终答案 {e['final_pred']}。")
    lines+=['','## 样本、验证与复现','']
    for setting,r in results.items():
        lines.append(f"- {setting}：best step {r['manifest']['checkpoint_step']}；{r['n_decisions']} 个 next-node 决策，train/val/test = {r['split_decisions']}；与已保存 cache 自由生成 next token 的 BF16 完整序列重放一致率 {fmt(r['native']['test_node_argmax_matches_saved_next'])}。checkpoint SHA256 {r['manifest']['checkpoint_sha256']}。")
    lines+=['',
        '完整序列重放与逐 token KV decode 存在 BF16 数值差异，故行为配对使用原始 token；attention 与 probe 使用无干预的原生 SDPA 前缀重放，观察 hook 精确不改变输出。独立 BFS 验证全部 100 起点的可达标签，并检查配对计数、AUROC ties 和 ranking 指标。未保存 activation、探针权重或模型 checkpoint，未更改训练代码。冻结前向在 GPU 1，所有 ridge 拟合在 GPU 0；任务已结束。',
        '', '下一步若要确认因果机制，应分别消融 L1 FFN 后继方向与关键目标读取层，检查前者是否破坏边合法性、后者是否保留边合法性但破坏目标定向。当前结果还没有完成这一因果分离。',
        '', '关键图：next_node_stages.png（分层输出头与探针对照）、attention_roles.png（原始与距离校正注意力）、matched_targets.png（两个相同起点不同目标的实例，附 examples.json 分层评分）。',
        '', '复现：`.venv/bin/python studies/path_direction/analyze.py`，随后 `.venv/bin/python studies/path_direction/summarize.py` 与 `.venv/bin/python studies/path_direction/examples.py`。']
    (HERE/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines))

if __name__=='__main__':main()

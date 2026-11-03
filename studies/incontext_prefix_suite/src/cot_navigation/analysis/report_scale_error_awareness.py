"""Two key figures and a validation-selection-aware scale probe report."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def values(d, arch, population, head=2, kind=None):
    result=[]
    for fit in d['fits']:
        if fit['architecture']!=arch or fit[population] is None:
            continue
        metric=fit[population][head]
        if kind is not None:
            metric=metric['by_kind'].get(kind,{})
        if metric.get('auroc') is not None:
            result.append(metric['auroc'])
    return result


def stat(v):
    return (float(np.mean(v)),float(np.std(v))) if v else (None,None)


def fmt(v):
    m,s=stat(v)
    return 'N/A' if m is None else f'{m:.3f} ± {s:.3f}'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path)
    args=parser.parse_args();out=args.output
    selected=json.loads((out/'selection.json').read_text())['models']
    data=[json.loads((out/'models'/f'{e["id"]}.json').read_text()) for e in selected]
    rows=[];findings=[]
    for d in data:
        entry=d['entry']
        for population in ['synthetic','natural']:
            for arch in ['linear','mlp']:
                for h,head in enumerate(d['heads']):
                    mean,std=stat(values(d,arch,population,h))
                    rows.append(dict(model=entry['id'],protocol=entry['protocol'],best_step=entry['step'],
                                     population=population,architecture=arch,head=head,auroc=mean,seed_std=std))
        syn_l=stat(values(d,'linear','synthetic'))[0];syn_m=stat(values(d,'mlp','synthetic'))[0]
        nat_l=stat(values(d,'linear','natural'))[0];nat_m=stat(values(d,'mlp','natural'))[0]
        findings.append(dict(model=entry['id'],protocol=entry['protocol'],synthetic_gain=syn_m-syn_l,
                             natural_gain=None if nat_m is None else nat_m-nat_l,
                             natural_pairs=d['datasets']['natural']['pairs'],
                             natural_node_auroc=stat(values(d,'mlp','natural',2,'in_graph_node'))[0],
                             natural_node_control=stat(values(d,'mlp','natural',3,'in_graph_node'))[0],
                             natural_node_pairs=d['datasets']['natural']['categories'].get('in_graph_node',0),
                             natural_other_auroc=stat(values(d,'mlp','natural',2,'format_or_other'))[0],
                             weak_natural_generalization=nat_m is not None and nat_m<.6,
                             below_chance=nat_m is not None and nat_m<.45))
    with (out/'summary.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for plot in ['comparison','hard_nodes']:
        fig,axes=plt.subplots(2,3,figsize=(14,8),sharex=True,sharey=True)
        for row,kind in enumerate(['context_graph','parametric_graph']):
            for col,family in enumerate(['path','dfs','bfs']):
                ax=axes[row,col];group=[d for d in data if d['entry']['kind']==kind and d['entry']['family']==family]
                styles=([('linear','synthetic',2,'Linear synthetic','#4477AA','-'),
                         ('mlp','synthetic',2,'MLP synthetic','#CC6677','-'),
                         ('linear','natural',2,'Linear natural','#228833','--'),
                         ('mlp','natural',2,'MLP natural','#AA3377','--')]
                        if plot=='comparison' else
                        [('linear','synthetic',2,'Linear L6','#4477AA','-'),
                         ('mlp','synthetic',2,'MLP L6','#CC6677','-'),
                         ('mlp','synthetic',3,'MLP token control','#228833','--')])
                for arch,pop,head,label,color,linestyle in styles:
                    series=[stat(values(d,arch,pop,head,'in_graph_node' if plot=='hard_nodes' else None)) for d in group]
                    means=[np.nan if a is None else a for a,b in series]
                    stds=[0 if b is None else b for a,b in series]
                    ax.errorbar(range(len(group)),means,yerr=stds,label=label,color=color,
                                linestyle=linestyle,marker='o',capsize=2,markersize=4)
                ax.axhline(.5,color='gray',linestyle=':',linewidth=1)
                ax.set_title(('New graph' if row==0 else 'Fixed graph')+' / '+family.upper())
                ax.set_xticks(range(4),['N36','N64','N100','Curr.100'])
                ax.set_ylim(-.03,1.04);ax.set_ylabel('L6 AUROC');ax.grid(alpha=.15)
                node_scales=[['N36','N64','N100','Curr.100'][i] for i,d in enumerate(group)
                             if d['entry']['protocol'].endswith('_nodes')]
                if node_scales:
                    ax.text(.98,.94,'node variant: '+', '.join(node_scales),ha='right',
                            transform=ax.transAxes,fontsize=8)
                if plot=='comparison':
                    counts=[d['datasets']['natural']['pairs'] for d in group]
                    ax.set_xlabel('natural pairs: '+', '.join(map(str,counts)),fontsize=8)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncol=len(labels),fontsize=9)
        fig.suptitle(('Synthetic vs natural first-error detection' if plot=='comparison' else
                      'Synthetic in-graph wrong-node subset vs identity control')+'; error bars = probe seed SD',fontsize=12)
        fig.tight_layout(rect=(0,.06,1,.95));fig.savefig(out/f'scale_{plot}.png',dpi=160);plt.close(fig)
    improved=sum(f['synthetic_gain']>0 for f in findings)
    overview=[f'同一人工测试分布上，MLP相对线性探针提高了 {improved}/24 个设置的L6 AUROC。'
              '人工分布收益与自然错误泛化需要分开判断。']
    for kind,label in [('context_graph','新图'),('parametric_graph','固定图')]:
        eligible=[stat(values(d,'mlp','natural'))[0] for d in data
                  if d['entry']['kind']==kind and d['datasets']['natural']['pairs']>=10]
        if eligible:
            overview.append(f'{label}中自然错误不少于10对的 {len(eligible)} 个设置，'
                            f'MLP L6自然AUROC范围为 {min(eligible):.3f}–{max(eligible):.3f}。'
                            '该范围只是本次设置的描述，不是独立图种子统计。')
    lines=['# 图 scale 后的错误识别探针','',*overview,'',
        '![规模变化下的人工与自然错误读出](scale_comparison.png)','',
        '本轮覆盖新图/固定图的 N36、N64、N100 静态训练及课程至 N100，共 24 个设置。'
        '每个规模、每种图任务内分别按存储的验证 best_score 选择 Path、DFS、BFS 最佳变体及 best checkpoint；'
        '同分优先全状态版本。未使用 test 分数挑选。scale 批次固定图只有图 seed 0，三个探针初始化不等于三个图 seed。','',
        '读出严格停在最后已输入 token：读取 h[layer,t-1]，判断 x[t-1] 是否局部违规，x[t] 尚未采样。'
        'L4–L6 分别训练线性/64 hidden GELU MLP，token embedding+字段/位置匹配对照；每种三个初始化。'
        '冻结导航模型，训练仅人工错误，真实生成首次错误完全保留作外部测试。','',
        '## 数据与 scale 适配','',
        '- 人工数据目标：train 8192 对、val 2048 对、test 4096 对；每对是共享合法历史的一个正确 token 与一个错误 token。'
        '新图主规模池 train 50k、val/test 各512，因此探针最多使用4096/512/512个独立查询；固定图按现有划分取样。',
        '- 原始查询沿用旧探针 common.sample_ids 的label/distance分层抽样。自然评估来自原始held-out test split，'
        '不是原scale最终评估使用的同一组256个ID；同时本轮读取best而非latest，不能直接与旧最终答案表逐项比较。',
        '- 使用实际 ScalePreparedRows 的四种预生成轨迹，variant=index%4；固定图按训练时 world_roots 映射添加 WORLD token。'
        '每个 checkpoint 的探针仅在其主规模/主世界训练与评估，不借用其他世界的查询。',
        '- STOP 沿用原生4096上下文、cot_margin 和强制停止策略；不把 STOP、最终答案/EOS 或错误后的连锁 token 当成错误监督。'
        '只有合法修复集合非空的首次错误可进入自然配对测试。早停/漏掉整个后缀不由局部 token 探针覆盖。',
        '- 人工错误覆盖图内错误节点、额外节点、过早分隔符、错误动作；Path/节点版只有图内节点替换。'
        '每个原查询的所有改写保留在相同 split，按查询逆样本数加权。'
        'Path只检查起点与真实边；沿合法边走入不能到达目标的分支，不属于本轮的局部错误标签。',
        '- DFS/BFS 节点版使用精确的任意邻居优先级前缀语言。独立穷举全部4节点DAG，4950个合法前缀的允许下一节点集合一致。'
        '全状态沿用旧探针的语义口径：集合顺序和合法PUSH均接受，不等同于严格复现教师序列。',
        '- AdamW最多1000更新，人工验证BCE按各头选最优；人工/自然泛化必须分开解释。'
        '正负训练先验1:1，输出分数不是自然错误发生率的校准估计。','',
        '## L6结果','',
        '均值 ± 三个探针初始化的标准差，非测试样本置信区间。自然错误对少于10只作诊断。','',
        '| 设置 | 选中协议 / 步数 | 训练对/查询 | 自然对 | 人工 Linear | 人工 MLP | 自然 Linear | 自然 MLP | 自然 MLP 身份对照 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for d in data:
        e=d['entry'];tr=d['datasets']['train'];n=d['datasets']['natural']['pairs']
        cells=[fmt(values(d,a,p,h)) for p,a,h in [('synthetic','linear',2),('synthetic','mlp',2),
               ('natural','linear',2),('natural','mlp',2),('natural','mlp',3)]]
        lines.append(f"| {e['id']} | {e['protocol']} / {e['step']} | {tr['pairs']}/{tr['unique_queries']} | {n} | "+' | '.join(cells)+' |')
    lines+=['','## 图内错误节点子集','','排除格式/动作错误，但仍包含起点/状态抄写等局部一致性检查，不代表完整搜索能力。','',
            '![人工图内节点错误与身份对照](scale_hard_nodes.png)','',
            '| 设置 | Linear L6 | MLP L6 | MLP 身份对照 |','|---|---:|---:|---:|']
    for d in data:
        cells=[fmt(values(d,a,'synthetic',h,'in_graph_node')) for a,h in [('linear',2),('mlp',2),('mlp',3)]]
        lines.append('| '+d['entry']['id']+' | '+' | '.join(cells)+' |')
    lines+=['','## N100与课程：自然图内错误节点子集','',
            '只比较错误token和合法修复均为图内节点的情况，排除图外/格式差异。'
            'Path/节点版人工训练不含图外节点，图内标记等控制特征在训练中可能是常量；'
            '其全自然集的身份基线不能当成充分训练过的图外节点检测器。这里的子集避免该标记分布外变化，'
            '但仍存在自然前缀与随机替换之间的其他分布差异。','',
            '| 设置 | 自然图内错误对 | Linear L6 | MLP L6 | MLP 身份对照 |','|---|---:|---:|---:|---:|']
    for d in data:
        if d['entry']['n_nodes']!=100:
            continue
        n=d['datasets']['natural']['categories'].get('in_graph_node',0)
        cells=[fmt(values(d,a,'natural',h,'in_graph_node')) for a,h in [('linear',2),('mlp',2),('mlp',3)]]
        lines.append(f"| {d['entry']['id']} | {n} | "+' | '.join(cells)+' |')
    lines+=['','## 自然错误的层间读出与配对比较','',
            '逐层列出，不根据自然测试分数选择最佳层。配对胜率检查同一合法历史下，错误token的错误分是否高于合法修复；'
            '概率差为mean(p_wrong−p_repair)，应大于0才符合期望方向。','',
            '| 设置 | MLP L4 | MLP L5 | MLP L6 | L6配对胜率 | L6概率差 | MLP身份对照 | 本次导航答案准确率 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    for d in data:
        cells=[fmt(values(d,'mlp','natural',h)) for h in range(3)]
        fits=[f['natural'][2] for f in d['fits'] if f['architecture']=='mlp' and f['natural'] is not None]
        cells.append(fmt([f['paired_rank_accuracy'] for f in fits]))
        cells.append(fmt([f['wrong_probability']-f['correct_probability'] for f in fits]))
        cells.append(fmt(values(d,'mlp','natural',3)))
        inventory=d['natural_rollout']
        cells.append(f"{inventory['answer_correct']/inventory['queries']:.1%}")
        lines.append('| '+d['entry']['id']+' | '+' | '.join(cells)+' |')
    # Reconstruct a few graph facts from held-out query IDs, without loading
    # navigation models or saving whole prompts/activation caches.
    from cot_navigation.config import load_config
    from cot_navigation.data.dataset import ContextStore
    from cot_navigation.protocols.tokenizer import Tokenizer
    examples=[]
    for d in data:
        e=d['entry']
        if e['id'] not in {'context_l6_path','context_l10_path'}:
            continue
        cfg=load_config([Path.cwd()/e['run']/'resolved_config.yaml'])
        tok=Tokenizer.from_config(cfg);store=ContextStore(Path.cwd()/cfg.data.root)
        for case in d['interesting_cases']:
            if case['kind']!='in_graph_node':
                continue
            query=case['query'];nodes=store.node_vocab_ids(query,tok);adj=store.adj(query)
            tail=case['prefix_tail'].split()
            previous=tok.stoi.get(tail[-2]) if len(tail)>=2 else None
            if previous not in nodes:
                continue
            allowed=[nodes[v] for v in adj[nodes.index(previous)]]
            assert tok.id(case['wrong_token']) not in allowed
            assert tok.id(case['legal_repair']) in allowed
            examples.append(dict(model=e['id'],query=query,probe_seed=0,
                previous_node=tok.name(previous),wrong_node=case['wrong_token'],
                legal_successors=[tok.name(v) for v in allowed],legal_repair=case['legal_repair'],
                probe_error_probability=float(1/(1+np.exp(-case['error_logit'])))))
            break
    (out/'path_error_examples.json').write_text(json.dumps(examples,indent=2)+'\n')
    lines+=['','## 具体漏报例子','',
            '以下只是MLP seed0低错误分的诊断案例，不用于调参。根据原始测试查询重新核对了图上的出边。']
    for case in examples:
        lines.append(f"- {case['model']}，query={case['query']}：前一节点 `{case['previous_node']}`，"
                     f"合法后继为 `{', '.join(case['legal_successors'])}`，却生成 `{case['wrong_node']}`。"
                     f"探针对这个错误token的错误分仅 {case['probe_error_probability']:.3f}。")
    lines+=['','## 异常与限制','']
    for f in findings:
        if f['natural_node_pairs']>=20 and f['natural_node_auroc'] is not None and f['natural_node_auroc']>.6:
            control=f['natural_node_control']
            if control is not None and f['natural_node_auroc']-control>.05:
                lines.append(f"- 局部信号：{f['model']} 的 {f['natural_node_pairs']} 对自然图内节点错误，"
                             f"MLP AUROC {f['natural_node_auroc']:.3f}，身份对照 {control:.3f}。"
                             '保留这一积极结果，但不能把子集上的读出等同于全部自然错误识别或模型自发监控。')
        if f['natural_pairs']>=10 and f['natural_node_auroc'] is not None and f['natural_other_auroc'] is not None:
            if f['natural_node_auroc']<.5 and f['natural_other_auroc']>.85:
                lines.append(f"- 类型混合：{f['model']} 的自然图内节点 MLP AUROC 为 {f['natural_node_auroc']:.3f}，"
                             f"图外/格式类为 {f['natural_other_auroc']:.3f}；总体较高不能解释为语义节点错误识别成功。")
        if f['natural_pairs']>=10 and f['below_chance']:
            lines.append(f"- {f['model']}：自然测试 {f['natural_pairs']} 对，MLP AUROC低于0.45。"
                         '保留训练标签方向，未根据测试集反转分数。')
        elif f['natural_pairs']>=10 and f['synthetic_gain']>.05 and (f['natural_gain'] or 0)<0:
            lines.append(f"- {f['model']}：人工 AUROC 提高 {f['synthetic_gain']:.3f}，自然 AUROC 变化 {f['natural_gain']:.3f}。"
                         '人工分布收益未迁移到自然错误。')
    lines+=['','不同规模的训练步数、混合规模曝光、所选协议/最优步数不同，不能把曲线差异完全归因于图大小。'
            '原导航验证仅16例，固定图很多变体同分，best选择存在噪声。'
            '旧小图使用不同数据和教师轨迹，节点版oracle也更弱，跨轮不是严格控制变量比较。'
            '高分只支持 activation 中有可读出的局部违规相关信息，不能证明原模型自发监控或修复错误。','',
            '一种需要后续检验的解释是负例选择偏差：人工错误是从不合法替换中随机抽取，'
            '自然错误却是模型贪心解码实际选择的token，后者通常是模型认为更可能的选择。'
            '因此合成探针可能利用模型熟悉度/置信线索，却无法迁移到高置信自然错误。'
            '本轮没有用此假设解释全部落差，也没有更改测试标签或按测试分数选层；'
            '后续可固定评估集，比较随机负例与模型高概率的非法替换。','',
            '关键图：`scale_comparison.png`、`scale_hard_nodes.png`。'
            '`models/*.json` 保存各层、正负概率、类型/字段、训练曲线及少量解码异常前缀；'
            '`selection.json` 保存全部候选验证记录，`verification.json` 保存完整性校验。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (out/'findings.json').write_text(json.dumps(findings,indent=2)+'\n')
    print('REPORT',len(data),'models')


if __name__=='__main__':
    main()

"""Three compact figures and a Chinese report from the follow-up JSONs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('studies/mechanistic_followup/results'))
    args=p.parse_args();root=args.input
    models=[json.loads(x.read_text()) for x in sorted((root/'models').glob('*.json'))]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    def fmt(v,percent=True): return '—' if v is None else f'{v*100:.1f}%' if percent else f'{v:.3f}'
    def task(m): return 'New graph' if m['entry']['kind']=='context_graph' else 'Fixed graph'
    def primary(m): return not m['entry'].get('auxiliary',False)
    def average(values):
        values=[x for x in values if x is not None]
        return float(np.mean(values)) if values else None
    csv_rows=[]
    for m in models:
        for target,s in m.get('sets',{}).items():
            for mode in ['reference','generated']:
                for layer,v in zip(s['layers'],s.get(mode) or []):
                    csv_rows.append({'model':m['entry']['id'],'experiment':'sets','target':target,'mode':mode,'layer':layer,
                                     **{k:v[k] for k in ['n_rows','nontrivial_rows','mean_true_probability','mean_false_probability',
                                                       'false_positive_rate_at_half','pairwise_auc','strict_separation','topk_recall'] if k in v}})
        for readout,r in m.get('errors',{}).get('readouts',{}).items():
            for source,s in r.get('tests',{}).items():
                for layer,v in zip(r['layers'],s.get('all',[])):
                    csv_rows.append({'model':m['entry']['id'],'experiment':'errors','target':readout,'mode':source,'layer':layer,
                                     **{k:v.get(k) for k in ['n_rows','balanced_accuracy','auroc','average_precision','paired_wrong_higher_rate']}})
        for variant,b in m.get('blocking',{}).items():
            if not isinstance(b,dict):continue
            for condition,v in b.get('conditions',{}).items():
                csv_rows.append({'model':m['entry']['id'],'experiment':'blocking','target':variant,'mode':condition,
                                 **{k:v.get(k) for k in ['n','accuracy','clean_accuracy_same_cases','counterfactual_NO_rate','pushed_blocked_route_node_rate']}})
    if csv_rows:
        fields=list(dict.fromkeys(k for r in csv_rows for k in r))
        with (root/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(csv_rows)
    # Figure 1: true/false sigmoid scores and *within-state* ranking.
    fig,axes=plt.subplots(2,2,figsize=(12,7),sharey=True)
    families=['direct','path','filler','dfs','bfs']
    for row,kind in enumerate(['New graph','Fixed graph']):
        for col,target in enumerate(['successors','visited']):
            ax=axes[row,col];groups=[]
            for fam in families:
                ms=[m for m in models if primary(m) and task(m)==kind and m['entry']['family']==fam and target in m.get('sets',{})]
                if ms:groups.append((fam,[m['sets'][target]['generated'][2] for m in ms]))
            for i,(label,values) in enumerate(groups):
                for off,key,color,title in [(-.23,'mean_true_probability','#2574a9','True score'),(0,'mean_false_probability','#e89b43','False score'),(.23,'pairwise_auc','#3b9b74','Pairwise AUC')]:
                    yy=[v.get(key) for v in values if v.get(key) is not None]
                    ax.bar(i+off,np.mean(yy),.21,color=color,label=title if i==0 else None,
                           yerr=np.std(yy) if len(yy)>1 else None,capsize=2)
                strict=average([v.get('strict_separation') for v in values])
                if strict is not None:ax.plot(i,strict,'kx',label='All true > all false' if i==0 else None)
            ax.set_xticks(range(len(groups)),[g[0] for g in groups]);ax.set_ylim(0,1.04)
            ax.set_title(f'{kind}: {target} (L6)');ax.grid(axis='y',alpha=.15)
    axes[0,0].legend(fontsize=8,ncol=2)
    fig.suptitle('Membership scores and discrimination; fixed error bars = SD across graph worlds')
    fig.tight_layout();fig.savefig(root/'set_discrimination.png',dpi=170);plt.close(fig)
    # Figure 2: only natural first errors; paired repaired controls, actual next
    # token held fixed. No useful confidence interval from 1-2 events is implied.
    fig,axes=plt.subplots(2,2,figsize=(12,7),sharey=True)
    colors=['#376eab','#e1973e','#479363','#999999']
    for row,kind in enumerate(['New graph','Fixed graph']):
        for col,readout in enumerate(['wrong_token','next_token']):
            ax=axes[row,col];groups=[]
            for fam in ['path','dfs','bfs']:
                ms=[m for m in models if primary(m) and task(m)==kind and m['entry']['family']==fam]
                vals=[m.get('errors',{}).get('readouts',{}).get(readout,{}).get('tests',{}).get('natural',{}) for m in ms]
                vals=[v for v in vals if 'all' in v]
                if readout=='next_token':
                    vals=[{**v,'all':v['strict_repaired_next_legal'],
                           'n_pairs':v['strict_repaired_next_legal_pairs']} for v in vals
                          if v.get('strict_repaired_next_legal_pairs',0)>0]
                if vals:groups.append((fam,vals))
            for i,(label,values) in enumerate(groups):
                for l in range(4):
                    yy=[v['all'][l]['auroc'] for v in values if v['all'][l]['auroc'] is not None]
                    if yy:ax.bar(i+(l-1.5)*.19,np.mean(yy),.18,color=colors[l],label=['L4','L5','L6','Token control'][l] if i==0 else None,
                                  yerr=np.std(yy) if len(yy)>1 else None,capsize=2)
            ax.axhline(.5,color='black',ls='--',lw=.7)
            ax.set_xticks(range(len(groups)),[f'{g[0]}\nn={sum(v["n_pairs"] for v in g[1])}' for g in groups])
            ax.set_ylim(0,1.04);ax.set_title(f'{kind}: {readout}');ax.grid(axis='y',alpha=.15)
    axes[0,0].legend(ncol=4,fontsize=8)
    fig.suptitle('Mixed-trained error heads; +1: SAME next token, legal repaired control (small n exploratory)')
    fig.tight_layout();fig.savefig(root/'error_detection.png',dpi=170);plt.close(fig)
    # Figure 3: NO is obedience to the counterfactual machine, NOT correctness
    # on the unchanged graph query (whose true label remains YES).
    fig,axes=plt.subplots(2,2,figsize=(13,7),sharey=True)
    for row,kind in enumerate(['New graph','Fixed graph']):
        for col,variant in enumerate(['including_goal','excluding_goal']):
            ax=axes[row,col]
            ms=[m for m in models if task(m)==kind and m.get('blocking',{}).get(variant,{}).get('n_cases',0)]
            for i,m in enumerate(ms):
                b=m['blocking'][variant]
                for off,name,color,title in [(-.23,'all_routes','#b64a48','All routes'),(0,'duplicate_padding','#888888','Equal-length duplicate control'),(.23,'goal_only','#d5ac42','Goal only')]:
                    v=b['conditions'].get(name)
                    if v:ax.bar(i+off,v['counterfactual_NO_rate'],.21,color=color,label=title if i==0 else None)
            ax.set_xticks(range(len(ms)),[m['entry']['id'].replace('context_','').replace('fixed_seed','s').replace('_bfs_state_control',' BFS-state').replace('_dfs',' DFS')+f'\nn={m["blocking"][variant]["n_cases"]}' for m in ms],rotation=15,ha='right')
            ax.set_ylim(0,1.04);ax.set_title(f'{kind}: {variant}');ax.grid(axis='y',alpha=.15)
    axes[0,0].legend(fontsize=7)
    fig.suptitle('Answer NO after certified blocking (original graph label stays YES)')
    fig.tight_layout();fig.savefig(root/'all_route_blocking.png',dpi=170);plt.close(fig)
    lines=['# CoT 机理补充实验：全部路径阻断、错误探针、集合概率',
           '', '本报告复用上一轮验证集选择的 checkpoint。导航模型冻结，只在物理 GPU0 训练线性读出头。',
           '旧结果只读；不保存模型／探针 checkpoint、activation 缓存或 W&B run。', '',
           '## 1. 集合探针：正确节点高分，错误节点是否低分', '',
           '使用与上一轮相同的数据、位置、训练种子及训练配置重新拟合。下面均为实际自由生成前缀、L6。',
           '输出是独立 sigmoid membership score，经过正类加权训练，未做概率校准，不能解释为 softmax 下一 token 概率。',
           'AUC 是每个非空、非全集状态中，随机正确节点分数超过随机错误节点分数的概率，再对状态平均；平分计 0.5。',
           '严格分离要求最低正确分数高于最高错误分数。空集／全集不计入排序分母；指标仍保存这些状态的误报。',
           'Top-k 使用真实集合大小 k，只用于诊断。固定图的当前节点 embedding 控制本来就能记忆邻接表。', '',
           '| 模型 | 目标 | 正确分数 | 错误分数 | 错误节点误报率 | 配对 AUC | 严格分离 | 输入对照 AUC |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    findings=[]
    by_name={m['entry']['id']:m for m in models}
    path=by_name.get('context_path',{}).get('sets',{}).get('successors',{}).get('generated')
    if path:
        a=path[2]
        findings.append({'topic':'Path 邻接信息排序明显强于单看阈值 F1 的印象',
            'model':'context_path','layer':6,'metrics':a,
            'text':f'新图 Path 后继 L6：正确／错误 membership score {a["mean_true_probability"]:.3f}/{a["mean_false_probability"]:.3f}，状态内排序 AUC {a["pairwise_auc"]:.3f}，但错误节点误报率 {a["false_positive_rate_at_half"]:.1%}、严格分离仅 {a["strict_separation"]:.1%}。不能把排序可读等同完整邻接表被可靠解码。'})
    for name in ['context_dfs','context_bfs']:
        m=by_name.get(name,{})
        inventory=m.get('errors',{}).get('natural_inventory',{}).get('test')
        if inventory:
            n=m['rollout']['n'];wrong_final=round(n*(1-m['rollout']['accuracy']))
            bound=max(0,inventory['first_errors']-wrong_final)
            findings.append({'topic':'最终答案正确不保证 CoT 无局部错误','model':name,
                'n_test':n,'semantic_error_queries':inventory['first_errors'],
                'at_least_correct_answer_with_error':bound,
                'text':f'{name} 的 {n} 条测试中发现 {inventory["first_errors"]} 条第一处局部 CoT 错误；仅按总数即可保证至少 {bound} 条存在 CoT 错误但最终答对（这是下界，不是精确交叉表）。'})
    bfs=by_name.get('context_bfs',{}).get('errors',{}).get('readouts',{}).get('next_token',{}).get('tests',{}).get('natural',{})
    if bfs.get('strict_repaired_next_legal'):
        findings.append({'topic':'下一 token 错误检测高度依赖对照合法性','model':'context_bfs',
            'all_pairs':bfs['n_pairs'],'all_l6_auc':bfs['all'][2]['auroc'],
            'strict_pairs':bfs['strict_repaired_next_legal_pairs'],'strict_l6_auc':bfs['strict_repaired_next_legal'][2]['auroc'],
            'text':f'新图 BFS +1 处 L6 AUC 从全部 {bfs["n_pairs"]} 对的 {bfs["all"][2]["auroc"]:.3f}，降至合法修复对照 {bfs["strict_repaired_next_legal_pairs"]} 对的 {bfs["strict_repaired_next_legal"][2]["auroc"]:.3f}。未筛选结果不能直接当作稳定的错误觉察证据。'})
    for m in models:
        for variant,b in m.get('blocking',{}).items():
            if not isinstance(b,dict):continue
            c=b.get('paired_contrasts',{}).get('all_routes_minus_dead_distinct_matched')
            if c and c['ci95'][0]>0:
                findings.append({'topic':'与严格 visited 语义相反的匹配干预方向','model':m['entry']['id'],'variant':variant,'contrast':c,
                    'text':f'{m["entry"]["id"]}/{variant}：在 {c["n"]} 对等长匹配查询上，全路径标记比无关节点标记更常输出 YES，差 {c["original_answer_accuracy_difference"]:.1%}，95% 区间 [{c["ci95"][0]:.1%}, {c["ci95"][1]:.1%}]。这与简单服从 visited 的预期方向相反；也可能包含状态修复，不能只凭最终答案定位原因。'})
    (root/'findings.json').write_text(json.dumps(findings,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines[5:5]=['## 主要观察','']+[f'- {f["text"]}' for f in findings]+['']
    for m in models:
        for target,s in m.get('sets',{}).items():
            if not s.get('generated'):continue
            v,control=s['generated'][2],s['generated'][3]
            lines.append(f'| {m["entry"]["id"]} | {target} | {fmt(v["mean_true_probability"],False)} | {fmt(v["mean_false_probability"],False)} | {fmt(v["false_positive_rate_at_half"])} | {fmt(v.get("pairwise_auc"))} | {fmt(v.get("strict_separation"))} | {fmt(control.get("pairwise_auc"))} |')
    lines+=['','![集合分数与排序](set_discrimination.png)','','## 2. CoT 第一处错误的线性检测','','图规则标注第一处局部语义错误，避免把后续连锁错误当作独立样本。Path 允许任意合法路径；DFS/BFS state 允许集合任意排列和任意合法 PUSH；BFS 节点序列按最短距离层级检查，允许同层不同顺序。',
            '不把“没生成目标”自动当成局部错误；最终 YES/NO/EOS 不作为探针输入。FOUND/EXHAUST 是 CoT 动作，会检查是否提前终止。截断只表示未观察到后续，不能证明完整正确。',
            '训练数据混合：最多 256 条参考查询的图内节点错误／修复配对，加最多 128 条模型自由生成查询的第一处错误／修复配对。验证最多 128 条参考＋64 条自由生成查询。训练／验证自由生成最多 512 token；测试使用原生 4096 上下文。',
            '同一查询的错误、修复及两个读出位置始终处于同一 split。每层独立 affine head，AdamW lr=.03、wd=.001、250 步，验证 BCE 选点。',
            '错误 token 处：错误前缀与只修复该 token 的对照。下一个 token 处：将实际同一个后继 token 追加到两种前缀；标签仍是指定的前一个 token 是否错误，不能理解为当前整个前缀是否完全正确。',
            '无后续非终止 token 的错误不能进入 +1，因此两列样本可能很不同。语法／图外 token 与图内语义错误另分组；完整明细见 models/*.json。',
            '探针高分说明错误信息可被外部监督线性读出，不证明原模型会自主报错／修正。', '',
            '| 模型 | 位置 | 自然错误配对数 | L4 AUC | L5 AUC | L6 AUC | token 控制 AUC | 图内错误配对数 | 图内 L6 AUC |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for m in models:
        for readout,r in m.get('errors',{}).get('readouts',{}).items():
            v=r.get('tests',{}).get('natural',{})
            if 'all' not in v:continue
            a=v['all'];inside=v.get('by_kind',{}).get('in_graph_node')
            lines.append(f'| {m["entry"]["id"]} | {readout} | {v["n_pairs"]} | '+ ' | '.join(fmt(x['auroc']) for x in a)+f' | {inside[2]["n_rows"]//2 if inside else 0} | {fmt(inside[2]["auroc"]) if inside else "—"} |')
    lines+=['','下一个 token 的严格自然子集：要求修复前一个 token 后，追加同一个实际后继仍是合法前缀。上表的完整 +1 任务只预测指定的前一 token 是否错，可能包含修复后新产生的后继错误；下表排除此情况。',
            '', '| 模型 | 严格自然配对数 | L4 AUC | L5 AUC | L6 AUC | token 控制 AUC |',
            '|---|---:|---:|---:|---:|---:|']
    for m in models:
        v=m.get('errors',{}).get('readouts',{}).get('next_token',{}).get('tests',{}).get('natural',{})
        a=v.get('strict_repaired_next_legal')
        if a:lines.append(f'| {m["entry"]["id"]} | {v["strict_repaired_next_legal_pairs"]} | '+' | '.join(fmt(x['auroc']) for x in a)+' |')
    lines+=['','![自然错误检测](error_detection.png)','','人工图内错误的独立测试：','','| 模型 | 位置 | 配对数 | L6 AUC | token 控制 AUC | 错误配对分数更高比例 |','|---|---|---:|---:|---:|---:|']
    for m in models:
        for readout,r in m.get('errors',{}).get('readouts',{}).items():
            v=r.get('tests',{}).get('synthetic',{})
            if 'all' in v:lines.append(f'| {m["entry"]["id"]} | {readout} | {v["n_pairs"]} | {fmt(v["all"][2]["auroc"])} | {fmt(v["all"][3]["auroc"])} | {fmt(v["all"][2]["paired_wrong_higher_rate"])} |')
    controls_path=root/'error_training_controls.json'
    if controls_path.exists():
        controls=json.loads(controls_path.read_text())
        lines+=['','探针训练来源对照：自然错误与人工节点替换的分布可能不同。下面分别训练自然错误头、人工错误头，与前面的混合头比较。三种头在完全相同的自然错误测试配对上评分；不是给导航模型纠错训练。',
                '额外控制只读取指定错误／修复 token 的 embedding：+1 处记得前一 token 的身份，不等于识别它是否错误。',
                '', '| 模型 | 位置 | 头的训练来源 | 自然测试对数 | L6 AUC | 当前 token 控制 AUC | 前一 token 控制 AUC | 严格 +1 对数 | 严格 +1 L6 AUC |',
                '|---|---|---|---:|---:|---:|---:|---:|---:|']
        for name,m in controls['models'].items():
            for readout,r in m['readouts'].items():
                for source,v in r['heads'].items():
                    if 'natural_test' not in v:continue
                    a=v['natural_test'];strict=v.get('strict_test')
                    prev=fmt(a[4]['auroc']) if len(a)>4 else '—'
                    lines.append(f'| {name} | {readout} | {source} | {r["natural_test_pairs"]} | {fmt(a[2]["auroc"])} | {fmt(a[3]["auroc"])} | {prev} | {v.get("strict_pairs") or "—"} | {fmt(strict[2]["auroc"]) if strict else "—"} |')
        lines+=['','自然错误训练样本量及更严格的图内语义子集：','']
        for name,m in controls['models'].items():
            inv=m['inventory']
            lines.append(f'- {name}：自然错误训练／验证对数 {inv["train"]["natural_pairs"]}/{inv["val"]["natural_pairs"]}；人工错误训练／验证对数 {inv["train"]["synthetic_pairs"]}/{inv["val"]["synthetic_pairs"]}。')
            for source,v in m['readouts']['next_token']['heads'].items():
                a=v.get('strict_in_graph_test')
                if a:lines.append(f'  {source} 头：同时要求图内节点错误且修复后后继合法，n={v["strict_in_graph_pairs"]}，L6 AUC={fmt(a[2]["auroc"])}。小样本不作能力边界判断。')
    lines+=['','## 3. 阻断所有剩余可达路径','','只在真实 YES 查询、目标尚未位于 frontier、原状态按标准搜索仍能找到目标时干预。考虑整个 stack／queue，不只当前节点。',
            '将所有未访问、能从某个 frontier 到达且还能到目标的节点加入 visited。另做不加入目标的版本；只有图算法与完整状态机均验证已阻断所有路径才纳入。',
            '同时比较全部未访问节点入 visited、仅加入目标、等长重复已有 visited 节点的 padding；若有足够不同的死分支节点，另有等长 distinct-dead 对照。重复控制是分布外序列，不冒称合法状态控制。',
            '在 visited 结束处截断，重生成 candidates/action 和全部后续。原图答案仍是 YES；遵守被篡改状态的搜索机应回答 NO。所以下表 NO 率是反事实状态机服从率，不是原查询准确率。',
            '不能仅凭 NO 多认定正确搜索：还要结合是否继续 PUSH 被阻断节点、是否抛弃／修复 visited 以及匹配对照。', '',
            '| 模型 | 条件 | n | 平均插入数 | clean YES率 | 全路径阻断后 NO率 | 等长 padding NO率 | 仅目标 NO率 | 后续 PUSH 阻断节点率 |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for m in models:
        for variant,b in m.get('blocking',{}).items():
            if not isinstance(b,dict) or not b.get('n_cases'):continue
            c=b['conditions'];v=c['all_routes']
            lines.append(f'| {m["entry"]["id"]} | {variant} | {b["n_cases"]} | {b["mean_inserted_nodes"]:.1f} | {fmt(c["clean"]["accuracy"])} | {fmt(v["counterfactual_NO_rate"])} | {fmt(c["duplicate_padding"]["counterfactual_NO_rate"])} | {fmt(c.get("goal_only",{}).get("counterfactual_NO_rate"))} | {fmt(v["pushed_blocked_route_node_rate"])} |')
    lines+=['','更严格的等长、不同死分支节点对照（仅可匹配的同一批查询）：', '',
            '| 模型 | 版本 | 配对数 | 全路径标记 YES率 − 无关标记 YES率 | 配对 95% 区间 |',
            '|---|---|---:|---:|---|']
    for m in models:
        for variant,b in m.get('blocking',{}).items():
            if not isinstance(b,dict):continue
            c=b.get('paired_contrasts',{}).get('all_routes_minus_dead_distinct_matched')
            if c:lines.append(f'| {m["entry"]["id"]} | {variant} | {c["n"]} | {fmt(c["original_answer_accuracy_difference"])} | [{fmt(c["ci95"][0])}, {fmt(c["ci95"][1])}] |')
    lines+=['','正差表示标记有用节点后反而更常回答 YES；不能解释成更忠实地遵循 visited。', '',
            '生成失败／原生上下文上限（仅列出非零条件）：', '',
            '| 模型 | 版本 | 条件 | n | 无合法答案率 | 截断率 |','|---|---|---|---:|---:|---:|']
    for m in models:
        for variant,b in m.get('blocking',{}).items():
            if not isinstance(b,dict):continue
            for condition,c in b.get('conditions',{}).items():
                if c['failure_rate'] or c['limit_rate']:
                    lines.append(f'| {m["entry"]["id"]} | {variant} | {condition} | {c["n"]} | {fmt(c["failure_rate"])} | {fmt(c["limit_rate"])} |')
    lines+=['','![全部路径阻断](all_route_blocking.png)','','每个版本最多 40 个合格查询；两个版本的查询子集不一定相同。不能把不同样本的柱高直接相减。',
            'visited 编辑与先前历史可能矛盾，模型可能修复状态；另一些图中存在已在队列的替代分支，本实验已把全部 frontier 纳入阻断验证。','','## 验证与复用','','最多每模型保存 6 个错误配对和各阻断版本 8 个解码案例。所有 checkpoint SHA256、原测试重放一致率、旧集合 F1 复现差异、查询 ID 均保存。',
            '运行：`.venv/bin/python -m cot_navigation.analysis.followup --resume`；汇报：`.venv/bin/python -m cot_navigation.analysis.followup_report`。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('REPORT',root,len(models),'models',len(csv_rows),'rows')


if __name__=='__main__':main()

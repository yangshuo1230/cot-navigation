"""Compact, reproducible tables and two figures from the per-query audit."""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import re
import sys

ROOT = Path.cwd()
OUT = ROOT/'studies/curriculum_trace_errors'
sys.path.insert(0,str(OUT))
from audit import jobs, sha, write
import numpy as np

PROTOCOLS = ['path','dfs_state','bfs_state','dfs_nodes','bfs_nodes']
SHAPES = ['valid_complete','legal_prefix_stopped','termination_only',
          'state_history_mismatch','one_bad_unit','two_or_three_bad_units','four_or_more_bad_units']
SEEN_FIXED = {'n16_l4w4','n24_l4w6','n32_l4w8','n36_l6w6','n64_l8w8','n100_l10w10'}

def summarize(rows):
    first = Counter()
    local = Counter()
    shape = Counter()
    candidate_subtypes=Counter()
    prefix = []
    density = []
    for r in rows:
        first.update([r['first_error']['kind']] if r['first_error'] else [])
        if r['first_error'] and r['first_error']['kind']=='candidate_mismatch':
            d=r['first_error']['detail']
            expected=set(d.get('expected',[]));shown=d.get('shown',[])
            missing=expected-set(shown);extra=set(shown)-expected;duplicate=len(shown)!=len(set(shown))
            subtypes=[]
            if missing:subtypes.append('omitted_eligible')
            if extra:subtypes.append('extra_ineligible')
            if duplicate:subtypes.append('duplicate')
            candidate_subtypes.update(subtypes)
            r['first_candidate_error']={'subtypes':subtypes,'missing':sorted(missing),'extra':sorted(extra),'duplicate':duplicate}
        kinds = {k for e in r['events'] for k in e.get('kinds',[e.get('kind')]) if k}
        local.update(kinds)
        sh = r['failure_shape']
        if sh=='termination_only' and not r['prefix_valid']: sh='state_history_mismatch'
        shape[sh]+=1
        if r['first_error_fraction'] is not None: prefix.append(r['first_error_fraction'])
        if not r['complete_valid']: density.append(r['bad_unit_fraction'])
    n = len(rows)
    return dict(n=n,answer_correct=sum(r['correct'] for r in rows),
                complete_valid=sum(r['complete_valid'] for r in rows),
                action_valid=sum(r['action_valid'] for r in rows),
                invalid_but_executable=sum(not r['complete_valid'] and r['action_valid'] for r in rows),
                correct_answer_invalid=sum(r['correct'] and not r['complete_valid'] for r in rows),
                wrong_answer_valid=sum(not r['correct'] and r['complete_valid'] for r in rows),
                stop=sum(r['stop'] for r in rows),forced_stop=sum(r['forced_stop'] for r in rows),
                prefix_valid=sum(r['prefix_valid'] for r in rows),
                first_errors=dict(first),first_candidate_subtypes=dict(candidate_subtypes),local_error_query_counts=dict(local),failure_shapes=dict(shape),
                first_error_fraction_median=float(np.median(prefix)) if prefix else None,
                first_error_fraction_q25=float(np.quantile(prefix,.25)) if prefix else None,
                first_error_fraction_q75=float(np.quantile(prefix,.75)) if prefix else None,
                invalid_local_density_median=float(np.median(density)) if density else None,
                late_error_queries=sum(r['first_error_fraction'] is not None and r['first_error_fraction']>=.8 for r in rows))

def main():
    metadata=[]; grouped={}; detail=[]; checks=[]
    expected=0
    for entry in jobs():
        dest=OUT/'models'/entry['id']
        expected += 9 if entry['kind']=='context' else 6
        for path in sorted(dest.glob('*.json')):
            m=json.loads(path.read_text())
            with gzip.open(path.with_suffix('.jsonl.gz'),'rt') as f: rows=[json.loads(x) for x in f]
            assert len(rows)==m['n']
            assert sha(ROOT/entry['run']/'checkpoints/best.pt')==entry['checkpoint_sha256']
            query=json.loads((OUT/'queries'/entry['kind']/(m['case']+'.json')).read_text())
            assert [r['query'] for r in rows]==query['ids']
            dimensions=re.search(r'_l(\d+)w(\d+)_n(\d+)_m(\d+)',m['dataset'])
            layers,width,n,edges=map(int,dimensions.groups())
            case=m['case']; key=(entry['kind'],entry['protocol'],case)
            role = ('unseen_world' if case not in SEEN_FIXED else 'curriculum_scale') if entry['kind']=='fixed' else ('shape_or_scale_OOD' if case in {'deep','wide','larger'} else 'curriculum_scale')
            grouped[key]=rows
            for r in rows:
                assert r['n_nodes']==n
                if r['label']: assert r['shortest_path'] is not None
                else: assert r['shortest_path'] is None
                r.update(model=entry['id'],kind=entry['kind'],protocol=entry['protocol'],case=case,
                         layers=layers,width=width,edges=edges,evaluation_role=role)
            old_case = '' if case in {'n100','n100_l10w10'} else '-'+case
            old_path=ROOT/entry['run']/'eval'/('predictions_test_best'+old_case+'.jsonl')
            old={int(r['query_id']):r for r in map(json.loads,old_path.read_text().splitlines())} if old_path.exists() else {}
            overlap=[r for r in rows if r['query'] in old]
            mismatches=[r['query'] for r in overlap if r['pred']!=old[r['query']]['pred']]
            checks.append({'model':entry['id'],'case':case,'hash_unchanged':True,'old_prediction_compared':len(overlap),'old_prediction_mismatches':mismatches})
            m.update(layers=layers,width=width,n_nodes=n,edges=edges,evaluation_role=role,summary=summarize(rows))
            metadata.append(m)
            for scope,subset in [('all',rows)]+[(f'gap={gap},label={label}',[r for r in rows if r['gap']==gap and r['label']==label])
                                for gap,label in sorted({(r['gap'],r['label']) for r in rows})]:
                s=summarize(subset)
                detail.append(dict(model=entry['id'],kind=entry['kind'],protocol=entry['protocol'],case=case,
                                   n_nodes=n,layers=layers,width=width,evaluation_role=role,scope=scope,**s))
    primary_finished=sum(m['evaluation_role']!='unseen_world' for m in metadata)
    write(OUT/'verification.json',{'expected_primary_conditions':expected,'finished_primary_conditions':primary_finished,
        'supplementary_conditions':len(metadata)-primary_finished,
        'all_primary_finished':primary_finished==expected,'validator_tests':json.loads((OUT/'validator_tests.json').read_text()),
        'query_selection_identical_across_protocols':True,'checkpoint_checks':checks})
    write(OUT/'summary.json',metadata)
    fields=list(detail[0])
    with (OUT/'distributions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader()
        for r in detail: w.writerow({k:json.dumps(v) if isinstance(v,dict) else v for k,v in r.items()})
    with gzip.open(OUT/'per_query_errors.jsonl.gz','wt') as f:
        for rows in grouped.values():
            for r in rows: f.write(json.dumps(r)+'\n')
    figures(metadata,grouped)
    report(metadata,grouped,checks,expected)
    print(json.dumps({'primary_finished':primary_finished,'primary_expected':expected,'queries':sum(m['n'] for m in metadata)}))

def figures(metadata,grouped):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,5,figsize=(23,10),layout='constrained')
    for i,kind in enumerate(['context','fixed']):
        for j,protocol in enumerate(PROTOCOLS):
            ms=[m for m in metadata if m['model']['kind']==kind and m['model']['protocol']==protocol and m['evaluation_role']!='unseen_world']
            ms.sort(key=lambda m:(m['n_nodes'],m['layers'],m['width']))
            matrix=np.full((len(ms),19),np.nan)
            labels=[]
            for k,m in enumerate(ms):
                labels.append(f"N{m['n_nodes']} L{m['layers']} W{m['width']}")
                rows=grouped[kind,protocol,m['case']]
                for gap in set(r['gap'] for r in rows):
                    subset=[r for r in rows if r['gap']==gap]
                    if 1<=gap<=19: matrix[k,gap-1]=sum(not r['complete_valid'] for r in subset)/len(subset)
            ax=axes[i,j]
            if ms: im=ax.imshow(matrix,aspect='auto',vmin=0,vmax=1,cmap='magma')
            ax.set_yticks(range(len(labels)),labels,fontsize=7)
            ax.set_xticks(range(0,19,2),range(1,20,2)); ax.set_xlabel('Layer gap')
            ax.set_title(f'{kind}: {protocol}')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Invalid / incomplete trace fraction',shrink=.7)
    fig.savefig(OUT/'invalid_by_scale_gap.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,5,figsize=(23,10),layout='constrained')
    colors=['#4daf4a','#80b1d3','#ffd92f','#984ea3','#ffb74d','#e57373','#b71c1c']
    for i,kind in enumerate(['context','fixed']):
        for j,protocol in enumerate(PROTOCOLS):
            ms=[m for m in metadata if m['model']['kind']==kind and m['model']['protocol']==protocol and m['evaluation_role']!='unseen_world']
            ms.sort(key=lambda m:(m['n_nodes'],m['layers'],m['width']))
            bottom=np.zeros(len(ms))
            for shape,color in zip(SHAPES,colors):
                values=np.array([m['summary']['failure_shapes'].get(shape,0)/m['n'] for m in ms])
                axes[i,j].barh(range(len(ms)),values,left=bottom,color=color,label=shape)
                bottom+=values
            axes[i,j].set_yticks(range(len(ms)),[f"N{m['n_nodes']} L{m['layers']}W{m['width']}" for m in ms],fontsize=7)
            axes[i,j].set_xlim(0,1);axes[i,j].set_title(f'{kind}: {protocol}')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=4,fontsize=9)
    fig.savefig(OUT/'failure_shapes.png',dpi=160);plt.close(fig)

def report(metadata,grouped,checks,expected):
    lines=['# Graph scale 课程模型：CoT 非法轨迹细分测试', '',
        f"主实验完成 {sum(m['evaluation_role']!='unseen_world' for m in metadata)}/{expected} 个模型×图条件，共 {sum(m['n'] for m in metadata if m['evaluation_role']!='unseen_world'):,} 次查询重放；另保留 {sum(m['evaluation_role']=='unseen_world' for m in metadata)} 个已测未见固定图条件作为补充。",'',
        '使用新图、固定图课程训练的 Path、DFS-state、BFS-state、DFS-nodes、BFS-nodes 共 10 个模型；全部选择各自验证集最佳 best.pt。测试集不参与 checkpoint 选择。', '',
        '## 设计与统计口径','',
        '- 对每个现有 TEST 集按（层间距离，可达标签）分层，每格最多抽取 8 条、不放回；不足 8 条全部使用。种子 20261021，同一图条件的五种协议使用完全相同的查询。逐格样本数见 distributions.csv。小样本格仅用于描述，不能据此断言稳定差异。',
        '- 新图：N=16/24/32/36/64/100；另有 N100 深窄 L20W5、浅宽 L5W20，以及 N144 L12W12。固定图主分析只用 scale_roots 实际训练的上述六种规模常规图。world_roots 注册的其余九张图作为未见世界补充对照单列，不能解释为已学图上的规模泛化。',
        '- 正例独立 BFS 计算最短路，负例 shortest_path=null；gap 是数据集的层间距离，不能称为负例的最短路。当前分层图正例最短路等于层差，仍独立验证标签。',
        '- 冻结模型，原生 BF16、4096 上下文及 STOP 规则；只用 GPU0，不中断其他训练。推理 batch64；与历史 batch16 的重叠查询预测对照另列。没有重新训练、写 checkpoint 或创建 W&B run。',
        '- 完整合法性按真实标签对应的任务要求判定，答案准确率单列；不因最后答案错误而直接认定每个中间状态错误。Path 负例仅要求真实边随机游走终止于 sink，不是不可达证明。Path 正例必须到达目标；即使模型输出 NO，也不能以“合法负例随机游走”标准放过未到目标的轨迹。这比先按模型预测决定终点要求的旧口径更严格；合法边前缀率另列。',
        '- 状态协议：从起点严格重放，visited/candidates 按集合检查，栈/队列按顺序检查，允许任意合法候选优先级；不是逐字匹配参考轨迹。动作骨架单独重放，不依赖打印状态。',
        '- Node 协议输出首次扩展顺序，不是相邻节点路径；DFS 允许隐式回溯，BFS 允许合法的兄弟节点排列。顺序首错之后不再把推测出的队列/栈当真，只统计重复、目标后继续等可独立判断的违规。',
        '- 首错是相对真实初始状态的第一次错误；首错后的 local 统计只检查打印状态自身及相邻记录的一致性。后续局部一致不等于恢复正确搜索，也不能把所有后续全局失配重复计成新错误。',
        '- 若状态字段已含图外节点，该记录归入 invalid_state_nodes，不再依赖这份非法状态推断候选/PUSH 细类。后续这些局部细类是已确证的检出数，可能低估总数；候选首错则额外直接按真实 eligible 集合记录遗漏、多出和重复。',
        '- 错误类型非互斥；“只有一处坏记录”也可能包含多个字段错误。termination_only 指无结构违规但提前结束；合法 STOP 单列，不归入完整轨迹。首错分位数以实际输出记录/节点数归一化，终止处缺失对应 1。', '',
        '## 全部图条件：答案准确率 / 完整轨迹合法率','',
        '| 任务与图 | 每协议 n | Path | DFS-state | BFS-state | DFS-nodes | BFS-nodes |',
        '|---|---:|---:|---:|---:|---:|---:|']
    keys=sorted({(m['model']['kind'],m['n_nodes'],m['layers'],m['width'],m['case']) for m in metadata if m['evaluation_role']!='unseen_world'})
    lookup={(m['model']['kind'],m['case'],m['model']['protocol']):m for m in metadata}
    for kind,n,l,w,case in keys:
        cells=[]; counts=[]
        for p in PROTOCOLS:
            m=lookup.get((kind,case,p))
            if not m: cells.append('待完成');continue
            s=m['summary'];counts.append(m['n'])
            cells.append(f"{s['answer_correct']/m['n']:.1%} / {s['complete_valid']/m['n']:.1%}")
        lines.append(f"| {kind} N{n} L{l}W{w} | {counts[0] if counts else 0} | "+' | '.join(cells)+' |')
    lines += ['','## 错误分布与首错位置','',
        '下表合并同一任务下全部测试图条件；每条查询权重相同，因此是本次分层诊断面板的分布，不是原训练分布或自然总体的估计。首错百分位仅对非法/未完成轨迹计算。', '',
        '| 任务/协议 | n | 非法/未完成 | 首错位置 Q25/中位/Q75 | 检出1处违规 | 2–3处 | ≥4处 | 合法STOP |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    all_summaries={}
    for kind in ['context','fixed']:
        for p in PROTOCOLS:
            rows=[r for (k,pp,c),rr in grouped.items() if k==kind and pp==p for r in rr if r['evaluation_role']!='unseen_world']
            if not rows:continue
            s=summarize(rows); all_summaries[kind,p]=s
            n=s['n'];bad=n-s['complete_valid'];sh=s['failure_shapes']
            positions='/'.join(f"{s[k]:.0%}" if s[k] is not None else '—' for k in ['first_error_fraction_q25','first_error_fraction_median','first_error_fraction_q75'])
            lines.append(f"| {kind}/{p} | {n} | {bad} ({bad/n:.1%}) | {positions} | {sh.get('one_bad_unit',0)} | {sh.get('two_or_three_bad_units',0)} | {sh.get('four_or_more_bad_units',0)} | {sh.get('legal_prefix_stopped',0)} |")
    lines += ['','## N100 常规图：分跳数、分正负例','',
        '单元格为非法或未完成数 / 查询数。短、中、长分别为层差 1–3、4–6、7–9；固定图已有 TEST 不含一跳，短组实际为 2–3。', '',
        '| 任务/协议 | 正例短 | 正例中 | 正例长 | 负例短 | 负例中 | 负例长 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for kind,case in [('context','n100'),('fixed','n100_l10w10')]:
        for p in PROTOCOLS:
            rows=grouped.get((kind,p,case),[])
            values=[]
            for label in [1,0]:
                for low,high in [(1,3),(4,6),(7,9)]:
                    ss=[r for r in rows if r['label']==label and low<=r['gap']<=high]
                    values.append(f"{sum(not r['complete_valid'] for r in ss)}/{len(ss)}" if ss else '—')
            lines.append(f'| {kind}/{p} | '+' | '.join(values)+' |')
    for (kind,p),s in all_summaries.items():
        bad=s['n']-s['complete_valid']
        lines += ['',f'### {kind} / {p}','',
            f"答案正确但轨迹非法/未完成 {s['correct_answer_invalid']}/{s['n']}；轨迹完整合法但答案错误 {s['wrong_answer_valid']}/{s['n']}；STOP {s['stop']}（强制 {s['forced_stop']}）；动作骨架合法 {s['action_valid']}/{s['n']}；完整轨迹非法但动作骨架仍完整可执行 {s['invalid_but_executable']}/{s['n']}；结构合法前缀 {s['prefix_valid']}/{s['n']}。",'',
            '首错类别（互斥，分母为本协议非法/未完成数）：']
        for name,count in sorted(s['first_errors'].items(),key=lambda x:-x[1]):
            lines.append(f'- `{name}`：{count}/{bad}（{count/max(1,bad):.1%}）')
        if s['first_candidate_subtypes']:
            lines += ['','候选首错细分（非互斥）：'+ '；'.join(f'`{k}` {v}' for k,v in s['first_candidate_subtypes'].items())+'。遗漏是缺少真实未访问后继，多出是包含不属于该 eligible 集合的 token，后者可能是已访问节点、非边节点或图外 token。']
        lines += ['','局部错误（非互斥，分母为该协议全部查询数）：']
        for name,count in sorted(s['local_error_query_counts'].items(),key=lambda x:-x[1]):
            lines.append(f"- `{name}`：{count}/{s['n']}（{count/s['n']:.1%}）")
    lines += ['','## 典型错误窗口','',
        '每个协议选取不同失败形态；下面是原始输出窗口，事件含真实候选集合及动作期望。完整逐查询事件见压缩 JSONL。']
    for kind in ['context','fixed']:
        for p in PROTOCOLS:
            candidates=[(m,e) for m in metadata if m['model']['kind']==kind and m['model']['protocol']==p and m['evaluation_role']!='unseen_world' for e in m['examples']]
            chosen=[]; used=set()
            for m,e in sorted(candidates,key=lambda x:(-x[0]['n_nodes'],-len(x[1]['types']))):
                key=e['failure_shape']
                if key not in used:
                    chosen.append((m,e));used.add(key)
                if len(chosen)==2: break
            for m,e in chosen:
                lines += ['',f"### {kind}/{p}，{m['case']}，query={e['query']}", '',
                    f"{e['source']} → {e['target']}，层差 {e['gap']}，标签 {e['label']}，预测 {e['pred']}；形态 `{e['failure_shape']}`。",'',
                    '首错：`'+json.dumps(e['first_error'],ensure_ascii=False)+'`','',
                    '```text',e['window'],'```']
    lines += ['','## 补充：未见固定图对照','',
        '下列 WORLD token 虽已注册，但对应图不在实际课程 scale_roots 中。这里的失败同时包含未学图知识，不能用来判断已学固定图的搜索能力。主分布、两张图及上面的协议汇总均排除这些条件。', '',
        '| 未见图 | Path | DFS-state | BFS-state | DFS-nodes | BFS-nodes |',
        '|---|---:|---:|---:|---:|---:|']
    for n,l,w,case in sorted({(m['n_nodes'],m['layers'],m['width'],m['case']) for m in metadata if m['evaluation_role']=='unseen_world'}):
        cells=[]
        for p in PROTOCOLS:
            m=lookup.get(('fixed',case,p))
            cells.append(f"{m['summary']['complete_valid']/m['n']:.1%}" if m and m['n'] else '无测试查询' if m else '本轮未测')
        lines.append(f'| N{n} L{l}W{w} | '+' | '.join(cells)+' |')
    compared=sum(c['old_prediction_compared'] for c in checks)
    mismatch=sum(len(c['old_prediction_mismatches']) for c in checks)
    lines += ['','## 验证与局限','',
        f'- 所有已完成条件的 checkpoint SHA256 与原验证集最佳选择清单相同；查询选择跨协议一致。原历史预测可比 {compared} 条，batch64 与历史 batch16 的最终预测差异 {mismatch} 条（详见 verification.json）。',
        '- 检查器通过 7680 项参考断言、24576 项任意节点序列的旧/新判定等价检查，以及 FOUND→POP 目标漏检变异测试；参考状态完整性额外在全部四节点 DAG、端点和两种搜索下检查。',
        '- 仅现有 curriculum 的 graph seed0，不是多训练种子或多固定图种子的显著性结论；不同图规模也同时改变形状/拓扑，不能把差异完全归因于节点数。',
        '- 没有该系列课程训练的 Filler checkpoint；Direct 无图轨迹。此次结构合法性覆盖现有五种有轨迹协议。',
        '- Node 协议的“检出1处违规”仅表示检出了一个违规位置。首次顺序错误之后不能唯一重建隐式搜索状态，后续顺序不再判定；不能据此认定它确实只错一次。状态协议的打印字段允许持续进行局部检查，两者的坏单元计数不能直接当作同等完整的误差率比较。',
        '- 本实验是轨迹观察分析，不足以证明模型内部是否在执行搜索；局部违规的扩散形态也不能单凭观察认定因果。', '',
        '## 文件','',
        '- `distributions.csv`：每模型×图条件的总体及精确 gap×label 统计。',
        '- `per_query_errors.jsonl.gz`：所有查询首错、坏单元、局部事件、终止/STOP 和答案信息；不保存大批激活或完整 rollout。',
        '- `summary.json`：条件元信息、checkpoint、样本格及少量原始错误窗口。',
        '- `invalid_by_scale_gap.png` / `failure_shapes.png`：仅两张关键结果图。',
        '- `queries/`：固定 TEST 查询 ID 和每格可用数，后续可直接复用。']
    batch_path=OUT/'batch_sensitivity.json'
    if batch_path.exists():
        b=json.loads(batch_path.read_text())
        tested=sum(r['n'] for r in b['results'])
        matched=sum(r['original_batch16_reproduced'] for r in b['results'])
        lines += ['', '## BF16 分批敏感性复核','',
            f'在预先列出的五个模型×条件中，各取一个含较多差异查询的原始 batch16，共 {tested} 条；历史最终预测复现 {matched}/{tested}。这些是有意选择的诊断样本，不能用来估计总体差异率。主表保留统一 batch64 的全部结果，没有择优替换。分批差异和轨迹合法性分别记录，不能把数值敏感性误报为 checkpoint 改变。详细逐查询比较见 batch_sensitivity.json。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':main()

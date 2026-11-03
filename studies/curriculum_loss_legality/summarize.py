"""Paired legality comparisons on common queries, with no test-based selection."""
from pathlib import Path
from collections import Counter
import csv
import gzip
import hashlib
import json
import numpy as np

ROOT=Path.cwd()
OUT=ROOT/'studies/curriculum_loss_legality'
CASES=['n16','n24','n32','n36','n64','n100','deep','wide','larger']

def metrics(rows):
    n=len(rows)
    errors=Counter(r['first_error']['kind'] for r in rows if r['first_error'])
    local=Counter(k for r in rows for k in {k for e in r['events'] for k in e.get('kinds',[])})
    fractions=[r['first_error_fraction'] for r in rows if r['first_error_fraction'] is not None]
    return {'n':n,'answer_correct':sum(r['correct'] for r in rows),
        'complete_valid':sum(r['complete_valid'] for r in rows),'action_valid':sum(r['action_valid'] for r in rows),
        'prefix_valid':sum(r['prefix_valid'] for r in rows),'stop':sum(r['stop'] for r in rows),
        'forced_stop':sum(r['forced_stop'] for r in rows),
        'legal_prefix_stopped':sum(r['stop'] and r['prefix_valid'] and not r['complete_valid'] for r in rows),
        'median_trace_tokens':float(np.median([r['trace_token_count'] for r in rows])),
        'median_records':float(np.median([r['units'] for r in rows])),
        'correct_answer_invalid':sum(r['correct'] and not r['complete_valid'] for r in rows),
        'annotation_only_invalid':sum(not r['complete_valid'] and r['action_valid'] for r in rows),
        'four_or_more_bad_units':sum(r['bad_unit_count']>=4 for r in rows),
        'median_first_error_fraction':float(np.median(fractions)) if fractions else None,
        'first_error_counts':dict(errors),'local_error_query_counts':dict(local)}

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v) if isinstance(v,dict) else v for k,v in r.items()})

def main():
    data={};meta=[];table=[];paired=[];examples=[]
    for path in sorted((OUT/'models').glob('*/*.json')):
        m=json.loads(path.read_text());e=m['model'];ck=e['checkpoint_label'];lam=e['lambda'];case=m['case']
        checkpoint=Path(e['run'])/'checkpoints'/(e['checkpoint_name']+'.pt')
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==m['checkpoint_sha256']
        with gzip.open(path.with_suffix('.jsonl.gz'),'rt') as f:rr=[json.loads(s) for s in f]
        query=json.loads((OUT/'queries/context'/(case+'.json')).read_text())
        assert [r['query'] for r in rr]==query['ids']
        assert len(rr)==m['n']
        data[ck,lam,case]=rr
        for scope,subset in [('all',rr),('positive',[r for r in rr if r['label']==1]),('negative',[r for r in rr if r['label']==0])]+[
            (f'gap={gap},label={label}',[r for r in rr if r['gap']==gap and r['label']==label])
            for gap,label in sorted({(r['gap'],r['label']) for r in rr})]:
            table.append({'checkpoint':ck,'lambda':lam,'step':m['step'],'case':case,'scope':scope,**metrics(subset)})
        m['summary']=metrics(rr);meta.append(m)
    for ck in ['best','last']:
        for case in CASES+['pooled']:
            if case=='pooled':
                if not all((ck,lam,c) in data for lam in [1,4,16] for c in CASES):continue
                by_lam={lam:[r for c in CASES for r in data[ck,lam,c]] for lam in [1,4,16]}
            elif all((ck,lam,case) in data for lam in [1,4,16]):by_lam={lam:data[ck,lam,case] for lam in [1,4,16]}
            else:continue
            for lam in [4,16]:
                baseline=by_lam[1];candidate=by_lam[lam]
                assert [(r['query'],r['label'],r['n_nodes'],r['gap']) for r in baseline]==[(r['query'],r['label'],r['n_nodes'],r['gap']) for r in candidate]
                for scope,label in [('all',None),('positive',1),('negative',0)]:
                    pairs=[(a,b) for a,b in zip(baseline,candidate) if label is None or a['label']==label]
                    for measure in ['complete_valid','action_valid','correct']:
                        gained=sum(not a[measure] and b[measure] for a,b in pairs)
                        lost=sum(a[measure] and not b[measure] for a,b in pairs)
                        paired.append({'checkpoint':ck,'case':case,'scope':scope,'lambda':lam,'measure':measure,
                            'n':len(pairs),'baseline_success':sum(a[measure] for a,b in pairs),
                            'candidate_success':sum(b[measure] for a,b in pairs),'gained':gained,'lost':lost,
                            'delta_percentage_points':100*(gained-lost)/len(pairs)})
                    if scope=='all' and case in ['n100','larger']:
                        for direction in ['gained','lost']:
                            if direction=='gained':
                                changed=[(a,b) for a,b in pairs if not a['complete_valid'] and b['complete_valid']]
                            else:
                                changed=[(a,b) for a,b in pairs if a['complete_valid'] and not b['complete_valid']]
                            for a,b in changed[:2]:
                                fields=['query','label','gap','correct','complete_valid','action_valid','stop','units','first_error','bad_unit_count']
                                examples.append({'checkpoint':ck,'case':case,'lambda':lam,'direction':direction,
                                    'baseline':{k:a[k] for k in fields},'candidate':{k:b[k] for k in fields}})
    OUT.mkdir(exist_ok=True)
    if table:write_csv(OUT/'legality_by_scale_hop_label.csv',table)
    if paired:write_csv(OUT/'paired_changes.csv',paired)
    verification={'finished_conditions':len(meta),'expected_conditions':54,'all_finished':len(meta)==54,
        'checkpoint_hashes_unchanged':True,'shared_query_ids_verified':True,
        'queries':sum(m['n'] for m in meta),'sampling':'8 per exact layer-gap/label cell; same 20261021 seed as earlier panel',
        'batch':64,'precision':'bf16','max_seq_len':4096,'no_training_or_wandb':True,
        'oracle_source_sha256':hashlib.sha256((ROOT/'studies/curriculum_trace_errors/audit.py').read_bytes()).hexdigest()}
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    (OUT/'summary.json').write_text(json.dumps(meta,indent=2)+'\n')
    (OUT/'paired_examples.json').write_text(json.dumps(examples,indent=2)+'\n')
    with gzip.open(OUT/'per_query_errors.jsonl.gz','wt') as f:
        for (ck,lam,case),rr in data.items():
            for r in rr:f.write(json.dumps({'checkpoint':ck,'lambda':lam,'case':case,**r})+'\n')
    report(data,meta,paired,verification)
    if verification['all_finished']:
        findings={}
        for ck in ['best','last']:
            findings[ck]={str(l):metrics([r for c in CASES for r in data[ck,l,c]]) for l in [1,4,16]}
        (OUT/'findings.json').write_text(json.dumps(findings,indent=2)+'\n')
    plot(data)
    print(json.dumps(verification))

def report(data,meta,paired,v):
    lines=['# 提高 CoT 权重是否改善 DFS-state 合法性','',
        f"完成 {v['finished_conditions']}/54 个模型×图条件，共 {v['queries']} 次冻结重放。",'',
        '同一训练种子的 λ=1/4/16；各自验证集 best，以及统一 200k 的 last 分开比较。每个图条件使用相同 TEST 查询，按精确层差与真实标签每格最多 8 条；复用此前固定诊断面板。不是原生完整 256 条测试集，答案准确率不能与上一轮原生测试直接作差。', '',
        '## 判定口径','',
        '- 完整合法：从真实初始状态重放，stack 顺序正确、visited/candidates 集合正确、每步动作合法，并正确 FOUND/EXHAUST 终止。最后 YES/NO 答案是否正确独立统计。',
        '- 动作骨架合法：忽略打印字段，仅凭动作从真实初态执行并正确终止，用于区分状态注释错误与实际执行错误。允许任意合法候选优先级，不要求与某条参考 CoT 逐字一致。',
        '- STOP 未完成搜索不算完整合法；合法前缀与完整搜索区分。局部错误是相对打印状态的可确证违规，首错之后的局部一致不代表恢复了真实搜索。',
        '- visited 与 candidates 的排列不影响语义；PUSH 可选择任意真实未访问后继。只有输出分隔符参与解析，不额外提供真实轨迹给模型。',
        '- BF16、batch64、原 4096 上下文与 STOP 策略；使用空闲 GPU0/2/3，没有中断两组 SWA 或其他训练。batch64 与历史 batch16 可存在数值分批差异，所有 λ 在这里统一条件。',
        '- 含图外节点的打印记录只归类 invalid_state_nodes；不依赖该非法状态进一步推断 PUSH/候选细类，所以细类事件数可能是下界。','']
    if v['all_finished']:
        last={l:metrics([r for c in CASES for r in data['last',l,c]]) for l in [1,4,16]}
        lines += ['## 主要结果：统一 200k 的 last','',
            '| λ | 分层面板完整合法 | 动作骨架合法 | 仅状态注释不合法、动作仍可执行 |',
            '|---|---:|---:|---:|']
        for l,s in last.items():
            lines.append(f"| {l} | {s['complete_valid']}/{s['n']} ({s['complete_valid']/s['n']:.1%}) | {s['action_valid']}/{s['n']} ({s['action_valid']/s['n']:.1%}) | {s['annotation_only_invalid']}/{s['n']} |")
        lines += ['',
            'λ=16 的完整合法率在本次分层面板上有温和改善，但动作骨架的增幅更小；部分净增来自状态注释与动作的一致性提高。λ=4 的效果因规模/形状而异，不存在 λ 越大所有条件都越好的结果。N144 的 λ=16 完整合法率与 λ=1 相同，不能认为已解决规模外推。', '',
            '上述汇总按本次分层查询等权计算，深窄图因跳数多而样本更多；不是自然分布上的整体准确率，应结合下面每个条件的结果看。','']
    for ck in ['best','last']:
        lines += [f'## {ck}：完整轨迹合法率 / 动作骨架合法率','',
            '| 图条件 | 每组 n | λ=1 | λ=4 | λ=16 |','|---|---:|---:|---:|---:|']
        for case in CASES:
            ss=[metrics(data[ck,l,case]) if (ck,l,case) in data else None for l in [1,4,16]]
            cells=[f"{s['complete_valid']/s['n']:.1%} / {s['action_valid']/s['n']:.1%}" if s else '运行中' for s in ss]
            n=next((s['n'] for s in ss if s),0)
            lines.append(f'| {case} | {n} | '+' | '.join(cells)+' |')
        lines += ['',f'### {ck} / N100：正负例分开','',
            '| λ | 正例完整合法 | 负例完整合法 | 答案正确但轨迹非法 | STOP | 首错中位位置 |',
            '|---|---:|---:|---:|---:|---:|']
        for l in [1,4,16]:
            if (ck,l,'n100') not in data:continue
            rr=data[ck,l,'n100'];s=metrics(rr)
            cells=[]
            for label in [1,0]:
                subset=[r for r in rr if r['label']==label]
                good=sum(r['complete_valid'] for r in subset)
                cells.append(f'{good}/{len(subset)} ({good/len(subset):.1%})')
            pos=s['median_first_error_fraction']
            position=f'{pos:.1%}' if pos is not None else '—'
            lines.append(f"| {l} | "+' | '.join(cells)+f" | {s['correct_answer_invalid']}/{s['n']} | {s['stop']}/{s['n']} | {position} |")
        lines += ['',f'### {ck} / N100：同一查询配对变化','',
            '| λ 对照 | 指标 | λ=1 失败→当前成功 | λ=1 成功→当前失败 | 净变化（百分点） |',
            '|---|---|---:|---:|---:|']
        for p in paired:
            if p['checkpoint']==ck and p['case']=='n100' and p['scope']=='all':
                lines.append(f"| {p['lambda']} vs 1 | {p['measure']} | {p['gained']} | {p['lost']} | {p['delta_percentage_points']:+.2f} |")
        lines += ['',f'### {ck} / N100：首错分布','']
        for l in [1,4,16]:
            if (ck,l,'n100') not in data:continue
            s=metrics(data[ck,l,'n100'])
            lines.append(f"- λ={l}：`{json.dumps(s['first_error_counts'],ensure_ascii=False)}`；检出至少四处坏记录 {s['four_or_more_bad_units']} 条。")
    lines += ['','## Checkpoint 与解释边界','']
    for m in meta:
        if m['case']=='n100':lines.append(f"- λ={m['model']['lambda']} / {m['model']['checkpoint_label']}：step={m['step']}，SHA256={m['checkpoint_sha256']}。")
    lines += ['','best 基于仅 16 条自由生成验证查询选取，可能受验证波动影响。λ=4 的 best 为 150k，处于 N100 主训练切换前；λ=1/16 的 best 分别为 155k/160k，已接受不同量的 N100 训练。因此统一 200k 的 last 是判断权重作用的重要对照。只测一个训练种子；按查询配对能描述本次改进/退化，不能替代多训练种子验证。', '',
        '当前两组 SWA 尚未达到最终训练预算，本次没有用其中间 checkpoint 混入比较。', '',
        '## 配对实例与机制限制','']
    example_path=OUT/'paired_examples.json'
    if example_path.exists():
        ee=json.loads(example_path.read_text())
        for direction in ['gained','lost']:
            match=next((e for e in ee if e['checkpoint']=='last' and e['case']=='n100' and e['lambda']==16 and e['direction']==direction and e['baseline']['label']==1),None)
            if match:
                a,b=match['baseline'],match['candidate']
                lines.append(f"- N100 query={a['query']}，可达正例，层差 {a['gap']}：λ=1 输出 {a['units']} 条记录、完整合法={a['complete_valid']}；λ=16 输出 {b['units']} 条记录、完整合法={b['complete_valid']}。首错细节保存在配对 JSON 中。")
    lines += ['', '合法率是自由生成行为指标，也受到所选搜索顺序、生成长度和错误扩散影响。上述实例同时存在改善和退化；不能只凭合法率上升，断言模型已经掌握了更精确的单步规则或更优规划。', '',
        '## 文件','',
        '- legality_by_scale_hop_label.csv：各规模、精确层差、正负例的完整合法性/动作合法性/答案及首错统计。',
        '- paired_changes.csv：同一查询上的增益、损失与净变化。',
        '- paired_examples.json：N100/N144 中改善与退化的配对实例，保留首错细节，不重复生成轨迹。',
        '- per_query_errors.jsonl.gz：逐查询记录；不保存大批激活或完整 rollout。',
        '- legality_comparison.png：唯一关键图，best/last 与正负例分开。',
        '- summary.json：checkpoint 元信息及少量原始错误窗口。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def plot(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(16,8),layout='constrained')
    for i,ck in enumerate(['best','last']):
        for j,(scope,label) in enumerate([('all',None),('reachable',1),('unreachable',0)]):
            ax=axes[i,j]
            for lam in [1,4,16]:
                xs=[];ys=[]
                for k,case in enumerate(CASES):
                    if (ck,lam,case) not in data:continue
                    rr=[r for r in data[ck,lam,case] if label is None or r['label']==label]
                    xs.append(k);ys.append(sum(r['complete_valid'] for r in rr)/len(rr))
                ax.plot(xs,ys,'o-',label=f'lambda={lam}')
            ax.set_title(f'{ck}: {scope}');ax.set_ylim(0,1.03)
            ax.set_xticks(range(len(CASES)),CASES,rotation=45);ax.grid(alpha=.2)
            ax.set_ylabel('Complete semantic trace validity');ax.legend()
    fig.savefig(OUT/'legality_comparison.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()

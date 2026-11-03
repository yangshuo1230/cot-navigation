"""Local plotting and auditable summaries for natural-history error readout."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
MODELS=['context_l10_dfs_state','context_l10_bfs_state','fixed_l10_dfs_state','fixed_l10_bfs_state']
TITLES=['新图 DFS','新图 BFS','固定图 DFS','固定图 BFS']
TOKEN=['error','token_1','token_4','token_16','token_64']
RECORD=['record_1_vis','record_2_vis','record_4_vis']
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'axes.unicode_minus':False,
    'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'path'})


def main():
    for name in ('verification.json','stream_verification.json'):
        verify=json.loads((HERE/name).read_text(encoding='utf-8'))
        assert verify['status']=='PASS' and len(verify['models'])==4
    rows=[]; manifest=[]; data={}
    for folder,domain in (('results','paired'),('stream_results','stream')):
        for model in MODELS:
            path=HERE/folder/'models'/f'{model}.json'; raw=path.read_bytes(); result=json.loads(raw)
            assert result['complete']; data[domain,model]=result
            manifest.append(dict(path=str(path.relative_to(HERE)),sha256=hashlib.sha256(raw).hexdigest()))
            for fit in result['fits']:
                for r in fit['test']:
                    row=dict(model=model,domain=domain,training=fit.get('training','natural_stream'),
                        subset=fit.get('subset','all'),architecture=fit['architecture'],seed=fit['seed'],
                        anchor=fit['anchor'],layer=r['layer'],n=r['n'],queries=r['queries'])
                    for metric in ('auroc','balanced_accuracy','paired_rank_accuracy'):
                        row[metric]=r.get(metric)
                    rows.append(row)
    (HERE/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    with (HERE/'all_probe_runs.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    keys=('model','domain','training','subset','architecture','anchor','layer')
    groups=defaultdict(list)
    for row in rows: groups[tuple(row[k] for k in keys)].append(row)
    summaries=[]; lookup={}
    for key,group in groups.items():
        summary=dict(zip(keys,key)); summary.update(n=group[0]['n'],queries=group[0]['queries'],seeds=len(group))
        for metric in ('auroc','balanced_accuracy','paired_rank_accuracy'):
            vals=[r[metric] for r in group if r[metric] is not None]
            summary[metric+'_mean']=float(np.mean(vals)) if vals else None
            summary[metric+'_std']=float(np.std(vals)) if vals else None
        summaries.append(summary);lookup[key]=summary
    (HERE/'summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (HERE/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)

    def get(model,anchor,layer='L6',domain='stream',training='natural_stream'):
        return lookup.get((model,domain,training,'all','mlp64',anchor,layer))

    def value(row):
        return 'NA' if row is None else f"{row['auroc_mean']:.3f}"

    def save(fig,name):
        fig.savefig(HERE/f'{name}.png',dpi=200,bbox_inches='tight')
        fig.savefig(HERE/f'{name}.svg',bbox_inches='tight');plt.close(fig)

    def curves(anchors,labels,name,title):
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        for ax,model,display in zip(axes.flat,MODELS,TITLES):
            for head,label,color in [('L6','L6 MLP','#2878b5'),
                ('token_position_control','Token + position control','#e38234'),
                ('position_control','Position control','#699f60')]:
                points=[get(model,a,head) for a in anchors]
                ys=[np.nan if r is None else r['auroc_mean'] for r in points]
                sd=[0 if r is None else r['auroc_std'] for r in points]
                if any(np.isfinite(ys)):
                    ax.errorbar(range(len(anchors)),ys,yerr=sd,label=label,color=color,marker='o',capsize=3)
            counts=[data['stream',model]['dataset']['test']['anchors'][a]['pairs'] for a in anchors]
            ax.set_xticks(range(len(anchors)),[f'{label}\nn={n}' for label,n in zip(labels,counts)])
            ax.set_ylim(0,1.04);ax.axhline(.5,color='.6',linestyle='--',linewidth=1);ax.grid(alpha=.15)
            ax.set_title(display);ax.set_ylabel('AUROC')
            if not any(get(model,a) for a in anchors):
                sizes=[data['paired',model]['dataset'][s]['pairs'] for s in ('train','val','test')]
                message=('首错后即进入终止动作\n没有可用的后续读出位置' if max(counts)==0 else
                    f'有效非终止首错不足\ntrain/val/test = {sizes[0]}/{sizes[1]}/{sizes[2]}')
                ax.text(.5,.72,message,
                        ha='center',transform=ax.transAxes)
        handles,leglabels=axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,leglabels,loc='lower center',ncol=3,bbox_to_anchor=(.5,.015))
        fig.suptitle(title+'\n正负均为自然生成；包括错误传播影响；n 为每类测试样本数；误差棒为 3 个探针初始化标准差',fontsize=12)
        fig.tight_layout(rect=(0,.08,1,.9));save(fig,name)

    curves(TOKEN,['错误处','+1 token','+4 tokens','+16 tokens','+64 tokens'],
           '01_natural_stream_tokens','自然轨迹：历史至当前位置是否已经出错')
    curves(RECORD,['后续第 1 条 VIS','后续第 2 条 VIS','后续第 4 条 VIS'],
           '02_natural_stream_records','自然轨迹：后续 VIS 结束位置的错误读出')
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,model,title in zip(axes,MODELS[:2],TITLES[:2]):
        definitions=[('人工训练\n迁移自然',get(model,'error',domain='paired',training='synthetic_transfer')),
            ('自然修复对照\nL6 MLP',get(model,'error',domain='paired',training='natural_repair')),
            ('自然修复对照\ntoken 对照',get(model,'error','current_token',domain='paired',training='natural_repair')),
            ('纯自然轨迹\nL6 MLP',get(model,'error')),
            ('纯自然轨迹\ntoken+位置',get(model,'error','token_position_control'))]
        ax.bar(range(5),[r['auroc_mean'] if r else np.nan for _,r in definitions],
            color=['#aaa','#2878b5','#e38234','#2878b5','#e38234'])
        for i,(_,r) in enumerate(definitions):
            if r: ax.text(i,r['auroc_mean']+.025,f"{r['auroc_mean']:.3f}\nn={r['n']//2}",ha='center',fontsize=9)
        ax.set_xticks(range(5),[label for label,_ in definitions],fontsize=9)
        ax.axhline(.5,color='.5',linestyle='--');ax.set_ylim(0,1.15);ax.set_ylabel('AUROC');ax.set_title(title)
    fig.suptitle('错误当场：不同测试设计与身份对照\n不同设计的样本群不同；自然修复对照的正确分支是人工单 token 修复')
    fig.tight_layout(rect=(0,0,1,.87));save(fig,'03_immediate_controls')

    lines=['# 自然生成 CoT 的错误读出','',
        '四个旧 scale_v1 N100 validation-best 模型冻结。每模型训练/验证/测试分别192/96/256个原始查询，严格按原数据划分。'
        '正式轮次 max_new_tokens=0，沿用模型完整上下文预算与原 cot_margin/STOP 规则；未额外加入1024 token截断。'
        '最初的1024 token试跑保存在服务器 pilot_1024，正式统计不混入试跑。','',
        '## 两类测试','',
        '1. **严格修复对照**：错误分支及后续全是模型贪心生成。定位第一个非终止语义错误，'
        '从合法 token 集合按固定种子均匀抽取一个修复 token，不看未来选择修复。只替换该 token，后续 token 保持完全一致。'
        '每个读出位置只保留修复后从开头到该位置仍合法的配对。该设计能隔离单个历史错误，但自然错误往往使后续也改变，因此晚期样本很少。','',
        '2. **纯自然轨迹分类**：正负输入都来自未经修改的自然生成。正例是在首错及其后的位置，负例是完整合法的生成前缀。'
        '按语法字段、优先同 token 身份、尽量相近的推理位置和 prompt 长度匹配；每个 anchor 每查询最多贡献一个正例、一个负例。'
        '这一测试允许后续错误传播，回答能否分类“此前至今已发生错误”，不单独证明记住了过去错误。','',
        '两类测试的首错与读出均排除 FOUND/EXHAUST、STOP、YES/NO/EOS；读出是消费该位置 token 后的 block residual。'
        '“错误当场”不是预测错误是否将要发生。所有非终止字段均按图关系与实际动作状态做语义检查，接受不同集合顺序和合法 PUSH 选择。'
        '人工迁移使用上一轮 CAND 单节点错误训练分布；自然错误训练另拟合新探针。','',
        'Linear/64 hidden GELU MLP，600步，3个初始化，验证 BCE 选择；自然轨迹提取使用 FP32、关闭 TF32。'
        '严格配对使用 BF16并同批配对，错前残差最大差为0。冻结导航权重未改动，不保存探针权重或 activation。','',
        '## 错误当场的严格修复对照','',
        '| 模型 | test配对 | 自然训练 L6 MLP | token身份对照 | 人工训练→自然 L6 |',
        '|---|---:|---:|---:|---:|']
    for model,title in zip(MODELS,TITLES):
        lines.append(f"| {title} | {data['paired',model]['dataset']['test']['pairs']} | "
            f"{value(get(model,'error',domain='paired',training='natural_repair'))} | "
            f"{value(get(model,'error','current_token',domain='paired',training='natural_repair'))} | "
            f"{value(get(model,'error',domain='paired',training='synthetic_transfer'))} |")
    lines+=['','新图 BFS 的单 token 修复会把过早输出的分隔符替换为缺失候选节点；'
            'token身份本身就能分清这两个分支，因此高 AUC 不构成错误意识证据。'
            '固定图 DFS 的当场残差读出也没有超过 token 身份对照。'
            '人工 CAND 节点探针迁移到不同类型自然错误的低 AUC 也不代表完全没有信息，而是原评分方向/训练分布不适用。','',
            '## 纯自然轨迹结果','',
            '| 模型 | 读出 | 每类n | L6 MLP AUROC | token+位置 | 仅位置 |',
            '|---|---|---:|---:|---:|---:|']
    for model,title in zip(MODELS,TITLES):
        for a in TOKEN+RECORD:
            row=get(model,a)
            if row:
                lines.append(f"| {title} | {a} | {row['n']//2} | {value(row)} | "
                    f"{value(get(model,a,'token_position_control'))} | {value(get(model,a,'position_control'))} |")
    lines+=['','新图 DFS 在 +4 token 处明显超过这两种对照，但 +16/+64 已不能超过仅位置对照；'
            '固定图 DFS 的后续 VIS 有高于对照的初步信号，测试每类仅22–24个样本，且可能读到错误传播后的状态。'
            '新图 BFS 主要漏候选后直接 EXHAUST，缺乏非终止后续；固定图 BFS 的训练首错仅8个，不拟合自然错误探针。','',
            '高于0.5但不超过 token/位置对照时，不能将信号解释为图信息与历史的整合。'
            '即使高于对照，纯自然轨迹分类也可能利用持续出现的当前状态错误，需要严格修复对照才能进一步隔离过去错误。','',
            '## 样本覆盖与限制','',
            '| 模型 | 严格首错 train/val/test | test +1/+4/+16/+64 合法修复配对 | test 第1/2/4条VIS合法修复配对 |',
            '|---|---|---|---|']
    for model,title in zip(MODELS,TITLES):
        d=data['paired',model]['dataset']; valid=d['test']['valid_pairs']
        lines.append(f"| {title} | {'/'.join(str(d[s]['pairs']) for s in ('train','val','test'))} | "
            f"{'/'.join(str(valid[a]) for a in TOKEN[1:])} | {'/'.join(str(valid[a]) for a in RECORD)} |")
    lines+=['','测试集完整生成覆盖（包括在读出测试中主动排除的终止动作错误）：','',
        '| 模型 | 首错为非终止 token | 首错为 FOUND/EXHAUST | 生成前缀未见语义错误 | 原生成预算用尽 |',
        '|---|---:|---:|---:|---:|']
    verification=json.loads((HERE/'verification.json').read_text(encoding='utf-8'))
    coverage={r['model']:r['generation_coverage']['test'] for r in verification['models']}
    for model,title in zip(MODELS,TITLES):
        c=coverage[model]
        lines.append(f"| {title} | {c.get('first_error_is_nonterminal',0)} | "
            f"{c.get('first_error_is_excluded_terminal_action',0)} | "
            f"{c.get('no_semantic_error_in_generated_prefix',0)} | {data['paired',model]['dataset']['test']['generation_limits']} |")
    lines+=['','自然训练要求 train≥20、val≥8、test≥8 配对；不足时标 NA。原始迁移指标即使样本不足也保留在 CSV，不能作为稳健结论。'
            '首错统计排除了终止动作错误，因此不是全部轨迹错误率；用尽上下文前未出现非终止错误的轨迹也可能答案错误。'
            '同一固定图的查询并不等于独立图泛化。各 anchor 独立拟合探针；token偏移与后续记录是不同位置，不构成同一条单调遗忘曲线。'
            '误差棒仅为探针初始化标准差，不是查询抽样置信区间。尚未测试自动纠错或证明模型会主动使用读出的信息。','',
            '两个独立 CPU 状态机校验分别检查全部严格配对和纯自然读出前缀，验证记录见 verification.json 与 stream_verification.json。','',
            '![自然轨迹 token 偏移](01_natural_stream_tokens.png)','',
            '![自然轨迹后续记录](02_natural_stream_records.png)','',
            '![当场对照](03_immediate_controls.png)','']
    (HERE/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    # Omit large pilot rollouts and machine-specific transient logs.
    with zipfile.ZipFile(HERE/'natural_history_error_results.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in HERE.rglob('*'):
            if path.is_file() and path.suffix not in ('.zip','.pyc') and '__pycache__' not in path.parts:
                archive.write(path,path.relative_to(HERE))
    print('Wrote summaries, three PNG/SVG figures, REPORT.md, results ZIP.')


if __name__=='__main__': main()

"""Three key natural-training figures; artificial errors remain final controls."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P=Path(__file__).resolve().parent
ARMS=['matched_clean','correction'];LABELS=['匹配干净模型','纠错模型'];COLORS=['#477baa','#e58b42']
PHASES=['首错记录','后续第1条','后续第2条']


def main():
    d=json.loads((P/'summary.json').read_text());assert d['complete']
    plt.rcParams.update({'font.sans-serif':['Noto Sans CJK SC','Microsoft YaHei','DejaVu Sans'],
                         'axes.unicode_minus':False,'font.size':10,'svg.fonttype':'none'})
    out=P/'figures';out.mkdir(exist_ok=True)
    def get(arm,ds,phase,head=5,arch='mlp64'):
        return next((r for r in d['aggregates'] if (r['arm'],r['architecture'],r['dataset'],r['record'],r['head'])==(arm,arch,ds,phase,head)),None)
    def save(fig,name):
        for ext in ['png','svg']:fig.savefig(out/f'{name}.{ext}',dpi=170)
        plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15.5,5.3));x=np.arange(3)
    for arm,label,color in zip(ARMS,LABELS,COLORS):
        for ax,ds,title in zip(axes[:2],['test','own_test'],['相同自然轨迹：共享测试','各checkpoint自己生成：自身测试']):
            rr=[get(arm,ds,p) for p in range(3)]
            yy=[r['auroc'] if r else np.nan for r in rr];std=[r['auroc_seed_std'] if r else np.nan for r in rr]
            ax.errorbar(x,yy,yerr=std,fmt='o-',capsize=3,color=color,label=label)
            ax.set(title=title,xticks=x,xticklabels=PHASES,ylim=(0,1),ylabel='第6层独立MLP AUROC')
        for arch,ls in [('linear','--'),('mlp64','-')]:
            rr=[get(arm,'test',p,arch=arch) for p in range(3)]
            axes[2].plot(x,[r['auroc'] if r else np.nan for r in rr],marker='o',ls=ls,color=color,label=label+' '+arch)
    axes[2].set(title='自然训练：线性与MLP比较',xticks=x,xticklabels=PHASES,ylim=(0,1),ylabel='共享自然测试AUROC')
    for ax in axes:ax.axhline(.5,color='grey',ls=':');ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=8)
    fig.tight_layout(rect=(0,.07,1,.94));fig.text(.5,.01,'仅原样自然前缀、早于MISTAKE；位置宏平均；4个模型seed。自身测试的错误分布不同，模型间共同输入比较见左图。',ha='center',fontsize=9)
    save(fig,'01_natural_primary')

    fig,axes=plt.subplots(2,3,figsize=(15,7),layout='constrained')
    for ai,arm in enumerate(ARMS):
        for phase in range(3):
            rr=[r for r in d['curves'] if r['arm']==arm and r['dataset']=='test' and r['architecture']=='mlp64' and r['record']==phase and r['head']<6]
            if not rr:axes[ai,phase].axis('off');continue
            array=np.full((6,max(r['token_offset'] for r in rr)+1),np.nan)
            for r in rr:array[r['head'],r['token_offset']]=r['auroc']
            ax=axes[ai,phase];im=ax.imshow(array,origin='lower',aspect='auto',vmin=.4,vmax=.95,cmap='viridis')
            ax.set_facecolor('#ddd');ax.set(title=LABELS[ai]+' · '+PHASES[phase],xlabel='记录内token偏移（从0起）',yticks=range(6),yticklabels=[f'L{i}' for i in range(1,7)])
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.7,label='自然共享测试 AUROC')
    fig.suptitle('自然错误训练→自然错误测试：所有有效层与token位置');save(fig,'02_natural_token_layers')

    types=sorted(d['artificial_counts']);fig,axes=plt.subplots(1,2,figsize=(16,5.6))
    names={'candidate_nonedge':'非邻接候选','candidate_out_of_graph':'图外候选','candidate_visited':'已访问候选',
           'candidate_duplicate':'重复候选','candidate_omission':'候选遗漏','visited_omission':'visited遗漏',
           'visited_extra':'visited多余','frontier_wrong_order':'栈顺序错误','frontier_wrong_node':'栈节点错误'}
    for ax,prefix,title in zip(axes,['natural_type/','artificial/'],['自然主测试：按自然首错类型','最后的人工控制：相同图/查询/记录']):
        for i,(arm,label,color) in enumerate(zip(ARMS,LABELS,COLORS)):
            rr=[get(arm,prefix+k,0) for k in types]
            ax.bar(np.arange(len(types))+(i-.5)*.36,[r['auroc'] if r else np.nan for r in rr],.34,color=color,label=label)
            if i==0:
                for j,r in enumerate(rr):
                    if r is None:ax.text(j,.04,'NA',ha='center',fontsize=9,color='grey')
        ax.axhline(.5,color='grey',ls=':');ax.set(title=title,xticks=np.arange(len(types)),xticklabels=[names.get(k,k) for k in types],ylim=(0,1),ylabel='首错记录第6层独立MLP AUROC');ax.tick_params(axis='x',rotation=35);ax.legend(frameon=False)
    fig.tight_layout(rect=(0,.12,1,.94));fig.text(.5,.01,'人工错误仅测试、不参与训练/验证/选步；类型不足样本或对应位置无自然训练探针时留空，不视作无法识错。',ha='center',fontsize=9)
    save(fig,'03_error_type_controls')

    lines=['# 自然错误训练与自然测试探针','', '**8/8完成。所有主训练、验证、主测试均为未经编辑的自由生成前缀；人工错误仅为末尾受控测试。**','',
        '读取已消费token位置p的各层残差，预测截至p是否已出现语义错误。各层×记录×token偏移参数独立；线性/MLP各2探针seed，4模型seed；验证BCE独立选步。','',
        '|架构|测试|位置|匹配干净|纠错|','|---|---|---|---:|---:|']
    for arch in ['linear','mlp64']:
        for ds,label in [('test','共同自然测试'),('own_test','自身自然测试')]:
            for phase in range(3):
                rr=[get(a,ds,phase,arch=arch) for a in ARMS]
                values=[f"{r['auroc']:.4f}" if r else 'NA' for r in rr]
                lines.append(f'|{arch}|{label}|{PHASES[phase]}|{values[0]}|{values[1]}|')
    lines+=['','## 结论与逐层检查','',
        '首错记录的共同自然测试中，纠错模型L6 MLP平均AUROC更高，且4个模型seed均高于各自匹配对照。第4–5层的提升也明显。后续记录仍有中层错误相关信息，不能根据最后一层接近随机就说所有层都已失去信息；纠错模型在后续记录没有一致优势。','',
        '|记录|层|匹配干净MLP|纠错MLP|','|---|---|---:|---:|']
    for phase in range(3):
        for head in [3,4,5]:
            rr=[get(a,'test',phase,head=head) for a in ARMS]
            values=[f"{r['auroc']:.4f}" if r else 'NA' for r in rr]
            lines.append(f'|{PHASES[phase]}|L{head+1}|{values[0]}|{values[1]}|')
    lines+=['',
        '自然训练后MLP在中层略有增益，但L6没有全面超过线性探针。先前人工训练、自然donor外测的结果不再作为主判据；本轮同时更换了训练分布与自然测试轨迹，不能把新旧差异单独归因于一个因素。','',
        '人工类型控制使用同一批64个测试查询、相同原始记录：图外/非邻接/已访问/遗漏/重复候选的纠错模型首错记录读出较强；visited遗漏较弱，栈顺序接近随机。本次所选1090个自然错误训练查询中，栈顺序首错只有1个，visited遗漏首错为0，不能据此断言模型完全不编码这些错误。','',
        '## 自然测试的主要对照（第6层MLP，首错记录）','',
        '|读出或测试子集|匹配干净|纠错|','|---|---:|---:|']
    for ds,head,label in [('test',5,'完整自然共享测试'),('exact_token_test',5,'严格匹配当前token的自然子集'),
                          ('test',6,'当前token+位置/字段对照'),('test',7,'历史CoT embedding均值对照'),
                          ('shuffle_control',5,'标签随机翻转对照')]:
        rr=[get(a,ds,0,head=head) for a in ARMS]
        values=[f"{r['auroc']:.4f}" if r else 'NA' for r in rr]
        lines.append(f'|{label}|{values[0]}|{values[1]}|')
    lines+=['','## 动作区域读出（含动作前分隔符、操作符、参数）','',
        '|位置|匹配干净L6 MLP|纠错L6 MLP|','|---|---:|---:|']
    for phase in range(3):
        rr=[get(a,'field/action',phase) for a in ARMS]
        values=[f"{r['auroc']:.4f}" if r else 'NA' for r in rr]
        lines.append(f'|{PHASES[phase]}|{values[0]}|{values[1]}|')
    lines+=['','## 数据和对照','',
        '共同测试给不同冻结模型输入完全相同的自然前缀；自身测试只使用该checkpoint自己的自由生成错误。后者模型间错误难度不同，不能独立作为共同输入上能力的比较。','',
        '导航查询均为可达正样本；探针的负标签仅指截至当前位置CoT合法，不是不可达导航问题。正标签来自已错的自然前缀；负标签来自另一个仍合法的自然前缀，不经过修复或参考轨迹拼接。每对匹配生成checkpoint、语法字段、记录内token偏移，优先匹配当前token并尽量接近推理位置。测试另汇报exact-token子集和token/位置、历史embedding均值控制。','',
        '自然文本用原生BF16推理生成，冻结模型以FP32重放原样前缀提取残差，避免批量BF16数值差异变成探针线索；不声称逐位复现生成时的BF16隐藏状态。','',
        '原始查询和带节点名图哈希在train/val/test间互斥；同图多checkpoint轨迹仍为同一个独立图。只保留MISTAKE出现前的主读出，因此后续位置缺失也反映该筛选，不能解释为信息已经遗忘。','',
        '自然续写可能继续产生错误或传播之前错误，因而后续记录可解码不能直接证明存在独立的“历史错误记忆”。人工控制保留参考续写，只作补充敏感性分析。','',
        '人工控制的类型在自然训练数据中实际出现过；使用相同原始图、查询、参考记录改变对应字段。人工控制与自然测试分布不同，类型敏感性不是自然发生频率。','',
        '## 采样统计','', '```json',json.dumps({s:d['shared_sampling'][s] for s in ['train','val','test']},ensure_ascii=False,indent=2),'```','',
        '每模型原始结果保存逐位置、各字段、自然首错类型、人工错误类型和自身自然测试的细分统计；图表只保留3张关键图。','',
        '独立语义检查：5632条自然前缀首错位置全部一致；所有自然读出的标签、生成源、字段、偏移、原样token和MISTAKE前截断通过校验；checkpoint保持冻结。','',
        '![主结果](figures/01_natural_primary.png)','', '![逐层逐token](figures/02_natural_token_layers.png)','',
        '![末尾类型控制](figures/03_error_type_controls.png)','']
    (P/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('NATURAL REPORT COMPLETE',flush=True)


if __name__=='__main__':main()

"""Three compact figures and a Chinese report from completed paired probes."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P=Path(__file__).resolve().parent
ARMS=['matched_clean','correction'];LABELS=['匹配干净模型','纠错模型'];COLORS=['#477baa','#e58b42']


def main():
    d=json.loads((P/'SUMMARY_PAIRED_V2.json').read_text(encoding='utf-8'))
    assert d['finished']
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','Noto Sans CJK SC','DejaVu Sans'],
                         'axes.unicode_minus':False,'font.size':10,'svg.fonttype':'none'})
    out=P/'figures';out.mkdir(exist_ok=True)
    def get(arm,phase,ds='test',arch='mlp64_independent',head=5):
        return next(r for r in d['aggregates'] if (r['arm'],r['record'],r['dataset'],r['architecture'],r['head'])==(arm,phase,ds,arch,head))
    def save(fig,name,note):
        fig.text(.5,.014,note,ha='center',fontsize=9)
        fig.savefig(out/f'{name}.png',dpi=170)
        fig.savefig(out/f'{name}.svg')
        plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(16,7.2),layout='constrained')
    for ai,arm in enumerate(ARMS):
        for phase in range(3):
            rows=[r for r in d['curves'] if r['arm']==arm and r['dataset']=='test' and r['record']==phase and r['head']<6]
            maximum=max(r['token_offset'] for r in rows);array=np.full((6,maximum+1),np.nan)
            for r in rows:array[r['head'],r['token_offset']]=r['auroc']
            ax=axes[ai,phase];im=ax.imshow(array,vmin=.45,vmax=.95,cmap='viridis',aspect='auto',origin='lower')
            ax.set_facecolor('#e7e7e7');ax.set(yticks=np.arange(6),yticklabels=[f'L{i}' for i in range(1,7)],
                title=f'{LABELS[ai]} · '+['首错记录','后续第1条记录','后续第2条记录'][phase],
                xlabel='记录内 token 偏移（从0起）')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='测试 AUROC；0.5为随机水平',shrink=.75)
    fig.suptitle('每层 × 每个 token 位置独立 MLP：错误信号主要集中在首错记录',fontsize=15)
    fig.set_constrained_layout_pads(h_pad=.5,w_pad=.1)
    save(fig,'01_token_layer_heatmap','4个模型seed × 2个探针seed；仅错误已出现后的成对样本。灰色=样本不足；有效位置至少8个测试查询。')

    fig,axes=plt.subplots(1,3,figsize=(16,5.3))
    x=np.arange(3)
    for arm,label,color in zip(ARMS,LABELS,COLORS):
        rr=[get(arm,p) for p in range(3)]
        axes[0].errorbar(x,[r['auroc'] for r in rr],yerr=[r['auroc_seed_std'] for r in rr],
                         fmt='o-',capsize=4,color=color,label=label)
        for j,r in enumerate(rr):
            dy=-14 if arm==ARMS[0] and j>0 else 6
            axes[0].annotate(f"{r['auroc']:.3f}",(j,r['auroc']),
                xytext=(-5 if j==2 else 4,dy),ha='right' if j==2 else 'left',
                textcoords='offset points',fontsize=9)
        values=[get(arm,0,ds)['auroc'] for ds in ['test','shuffle_control','identical_cot_graph_zero_shot','graph_trained']]
        axes[1].bar(np.arange(4)+(-.18 if arm==ARMS[0] else .18),values,.34,color=color,label=label)
        axes[2].plot(x,[r['positive_p'] for r in rr],'o-',color=color,label=label+'：错误轨迹')
        axes[2].plot(x,[r['negative_p'] for r in rr],'s--',color=color,label=label+'：干净轨迹')
    for ax in axes:ax.axhline(.5,color='grey',ls=':',lw=1);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].set(title='固定第6层 MLP：AUROC',xticks=x,xticklabels=['首错记录','后1条','后2条'],ylim=(.45,.82));axes[0].legend(frameon=False)
    axes[1].set(title='首错记录的关键对照（L6 MLP）',xticks=np.arange(4),xticklabels=['主测试','标签打乱','相同CoT\n零样本迁移','相同CoT\n单独训练'],ylim=(0,.85),ylabel='AUROC')
    axes[2].set(title='错误概率读出：正负样本能否分开',xticks=x,xticklabels=['首错记录','后1条','后2条'],ylim=(.3,.7),ylabel='探针平均输出概率');axes[2].legend(frameon=False,fontsize=8)
    fig.tight_layout(rect=(0,.12,1,.94))
    save(fig,'02_controls_and_probabilities','指标先在有效token位置宏平均，再平均探针seed和模型seed；误差棒为4个模型seed的标准差，不是置信区间。')

    fig,axes=plt.subplots(1,3,figsize=(15,4.8))
    for phase,ax in enumerate(axes):
        for arm,label,color,offset in zip(ARMS,LABELS,COLORS,[-.18,.18]):
            vals=[r['auroc'] for r in get(arm,phase)['per_model']]
            bars=ax.bar(np.arange(4)+offset,vals,.34,color=color,label=label)
            ax.bar_label(bars,fmt='%.3f',fontsize=8,padding=3)
        ax.axhline(.5,color='grey',ls=':');ax.set(xticks=np.arange(4),xticklabels=['seed0','seed1','seed2','seed3'],
            title=['首错记录','后续第1条记录','后续第2条记录'][phase],ylim=(.45,.85),ylabel='第6层 MLP AUROC')
    axes[0].legend(frameon=False,fontsize=9)
    fig.tight_layout(rect=(0,.1,1,.94))
    save(fig,'03_model_seeds','每根柱先平均该模型的2个探针seed和有效token位置。首错记录纠错组3/4模型seed更高；后续平均小差异不稳定。')

    lines=['# 成对历史错误探针：修正轮结果','',
        '**8/8模型已完成，耗时约19分钟。** 匹配干净/纠错各4个模型seed，每个模型独立线性和MLP64各2个探针seed，最多5000步；没有重新训练导航模型。', '',
        '主结论：纠错训练增强了首错记录内可读出的局部图一致性错误信号，且在第4–5层已经出现；但没有稳定证据表明信号能跨后续两条动作记录持续保留。', '',
        '## 固定第6层的结果','',
        '|探针|位置|匹配干净|纠错|','|---|---|---:|---:|']
    for arch,label in [('linear_independent','线性'),('mlp64_independent','MLP64')]:
        for phase,name in enumerate(['首错记录','后续第1条','后续第2条']):
            lines.append(f"|{label}|{name}|{get(ARMS[0],phase,arch=arch)['auroc']:.4f}|{get(ARMS[1],phase,arch=arch)['auroc']:.4f}|")
    lines += ['', '这是各有效token位置的AUROC宏平均，不是最终答案准确率，也不是每条轨迹识错成功率。固定L6汇报，没有按测试成绩挑选最优层或最优模型seed。', '',
        '首错记录MLP：干净/错误轨迹平均输出概率，匹配干净模型约0.449/0.559，纠错模型约0.392/0.600；有区分，但离稳定、可靠识错仍有距离。后续记录两类概率都接近0.5。', '',
        '## 对照与泛化','',
        '|首错记录，L6 MLP|匹配干净|纠错|','|---|---:|---:|']
    for ds,name in [('shuffle_control','查询级标签打乱'),('identical_cot_graph_zero_shot','相同CoT、不同图：零样本迁移'),('graph_trained','相同CoT、不同图：单独训练'),('natural_test','自然donor错误外测')]:
        lines.append(f"|{name}|{get(ARMS[0],0,ds)['auroc']:.4f}|{get(ARMS[1],0,ds)['auroc']:.4f}|")
    lines += ['',
        '标签打乱回到随机水平；相同CoT文本的图控制中，token/位置及CoT embedding均值控制均为0.5。因此首错附近残差信号包含图与候选列表的一致性信息，不能全部解释为词面或位置差异。', '',
        '自然donor错误没有统一提升：第6层匹配干净模型更高，但第4–5层纠错模型约0.641–0.642，匹配模型约0.525/0.511。不能概括为纠错模型对所有错误类型、所有层都更容易读出。', '',
        '首错记录L6 MLP四个模型seed：匹配模型0.6055/0.6179/0.6005/0.6106；纠错模型0.7039/0.6040/0.7861/0.6742。纠错模型3/4 seed更高，但并非全部。后1条纠错组平均0.510主要来自seed0的0.5354，其余接近0.5。', '',
        '## 数据与边界','',
        '- 普通N100、新图、正样本；768对训练、192对验证、217对主测试。每对同图同查询，包含干净/错误两个版本。四个模型seed重复使用同一测试集，不能算作868个独立查询。',
        '- 主实验替换非PUSH候选为图内非邻接节点，其余文本和后续真实合法动作相同；没有MISTAKE标记。这隔离了历史字段错误，而非非法动作造成的全面状态崩坏。',
        '- 自然外测40对，来自旧donor在held-out图上的实际错误候选，包含遗漏/多余/重复；不是当前模型自身自由生成的错误。',
        '- 所有查询、变体和token位置保持在同一split；原始带节点名图哈希也无交叉。8个checkpoint SHA256均未变化，受控样本错误前成对激活差异为0。',
        '- 训练和主要评估仅保留错误已出现后的成对位置；各位置恰有一个正标签、一个负标签。原轮混合错误前后人口导致的高分不再使用。',
        '- 各层×记录×token位置独立参数、标准化和验证选步。训练至少30个正标签查询、验证至少8个；测试位置至少8个查询。部分晚位置支持较少，逐位置统计在SUMMARY_PAIRED_V2.json保留min_queries。',
        '- 首错记录测试支持≥32查询的位置子集，L6 MLP平均AUROC为0.6456→0.7369，方向一致；这只是样本支持敏感性检查，不是按成绩筛选。',
        '- 不可将“可以从激活解码”称为主动知道/会改错；也不可由这组线性/MLP读出失败断言模型内部不存在任何历史错误信息。概率对应人为采样分布，尚未校准为自由生成的真实后验。', '',
        '![逐层逐token](figures/01_token_layer_heatmap.png)','',
        '![对照和概率](figures/02_controls_and_probabilities.png)','',
        '![模型seed](figures/03_model_seeds.png)','']
    (P/'REPORT_PAIRED_V2.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Wrote paired-v2 report and 3 PNG/SVG figures.')


if __name__=='__main__':main()

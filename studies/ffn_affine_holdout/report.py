"""One key figure plus an evidence-calibrated report."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
DATA={g:json.loads((HERE/f'{g}.json').read_text()) for g in ('simple','hard')}
SCOPE='shared_strict_queries'
METHODS=['native','ffn_off','ridge_full','ridge_core','unitwise_full','unitwise_core','constant_norm_native']
LABELS=['Native','FFN off','Dense affine\nLN removed','Dense affine\nLN kept','Unitwise affine\nLN constant','Unitwise affine\nLN kept','Original GELU\nLN constant']
PRETTY={'native':'原模型','ffn_off':'FFN删除','ridge_full':'重拟合仿射：LN前→增量','ridge_core':'重拟合仿射：LN后→增量',
        'unitwise_full':'保留W，GELU逐元仿射＋常数LN','unitwise_core':'保留W，GELU逐元仿射＋原LN',
        'constant_norm_native':'原GELU＋常数LN','taylor_full':'训练均值处Taylor：含LN', 'taylor_core':'训练均值处Taylor：仅GELU核心'}

def records(g,scope=SCOPE):return DATA[g]['scopes'][scope]
def ba(g,role,method):return [r['roles'][role]['functional'][method]['constrained_ba'] for r in records(g)]
def readout(g,role,relation,method,fixed=False):
    key={'native':'native_post','ffn_off':'after_attention'}.get(method,method+'_post')
    field='native_decoder_fixed' if fixed else 'fresh'
    return [r['roles'][role]['relations'][relation][field][key]['auc'] for r in records(g)]
def avg(v):return float(np.mean(v))
def fmt(v):return f'{np.mean(v):.3f}'

plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(2,2,figsize=(15,9),layout='constrained');x=np.arange(len(METHODS))
for j,(fixed,name) in enumerate([(False,'Fresh decoder'),(True,'Fixed native decoder')]):
    v=[readout('hard','source','descendants',m,fixed) for m in METHODS]
    ax[0,0].bar(x+(j-.5)*.36,np.mean(v,1),.36,yerr=np.std(v,1),capsize=2,label=name)
ax[0,0].set(xticks=x,xticklabels=LABELS,ylim=(.7,1.01),ylabel='Heldout-node AUROC',title='Hard source: reachable-descendant readout');ax[0,0].legend()
for j,role in enumerate(('source','target')):
    v=[ba('hard',role,m) for m in METHODS]
    ax[0,1].bar(x+(j-.5)*.36,np.mean(v,1),.36,yerr=np.std(v,1),capsize=2,label=role.title())
ax[0,1].set(xticks=x,xticklabels=LABELS,ylim=(.5,1.03),ylabel='YES/NO balanced accuracy',title='Hard: replace one heldout endpoint at L1');ax[0,1].legend()
for j,g in enumerate(('simple','hard')):
    v=[[r['joint_both_heldout'][m]['constrained_ba'] for r in records(g)] for m in METHODS]
    ax[1,0].bar(x+(j-.5)*.36,np.mean(v,1),.36,yerr=np.std(v,1),capsize=2,label=g.title())
ax[1,0].set(xticks=x,xticklabels=LABELS,ylim=(.3,1.05),ylabel='YES/NO balanced accuracy',title='Replace BOTH heldout endpoints (small test panels)');ax[1,0].legend()
for j,g in enumerate(('simple','hard')):
    v=[]
    for m in METHODS:
        key={'native':'native_post','ffn_off':'after_attention'}.get(m,m+'_post')
        v.append([r['roles']['source']['original_head']['successors'][key]['recall_at_true_k'] for r in records(g)])
    ax[1,1].bar(x+(j-.5)*.36,np.mean(v,1),.36,yerr=np.std(v,1),capsize=2,label=g.title())
ax[1,1].set(xticks=x,xticklabels=LABELS,ylim=(0,1.05),ylabel='Original-head successor recall@k',title='Source: successor retrieval (no fitted probe)');ax[1,1].legend()
fig.savefig(HERE/'ffn_role.png',dpi=170);plt.close(fig)

lines=['# 留出节点的 L1 FFN 仿射替换与功能判断','',
'结论：FFN 的大部分当前可达性答题作用，可以由保留原W_up/W_down、线性近似GELU的替代保留，且拟合时严格排除测试节点。困难图起点原FFN增强的后代probe表现没有被同等保留；这个非线性表示效应与最终答案表现可以分离。不能据此证明后代事实存于FFN，也不能证明所有仿射替代都不可能达到原表现。','',
'## 数据划分与适用范围','',
'沿用简单/困难N100 Fixed DAG Direct各一个验证最佳checkpoint，模型冻结，CPU FP32推理，闭式ridge仅GPU0。不训练主模型、不保存activation/probe/model checkpoint或W&B run。只替换第一层节点位置的FFN输出；attention、协议token位置、后续层全部保留。因此没有把整个网络线性化。','',
'节点seed17/31/47，60训练/20验证/20测试。与原query train/val/test取交集。原模型训练见过全部节点；留出约束本次新拟合的映射和probe，不是原模型未见节点泛化。','',
'两种拟合支持均保存：','',
'- source_only：只拟合训练节点位于起点位置时的FFN，源位置看不到后面的目标；有效独立输入约53–55种。',
'- shared_strict_queries（主表）：只允许训练查询的两个端点都属于训练节点集；验证查询两个端点都属于验证节点集。用起点、目标两种位置拟合一套共享仿射映射，尽量符合原共享FFN。训练输入没有测试节点token，样本不仅是60个固定embedding，还包含由attention产生的上下文变体。','',
'图关系probe仍沿用原query划分与焦点身份划分；probe训练中的另一个端点可能为其他节点集合，目标activation可能包含其上下文。映射拟合的更严格双端点排除与probe焦点留出不能混淆。主要起点probe看不到目标。','',
'主共享映射的训练样本数为简单1878–1988个位置、困难2408–2490个位置；严格验证仅简单26–34条查询、困难18–22条。每seed实际训练映射覆盖60个节点，测试与之不交叠。有效输入秩约简单119–120、困难90–95（source-only约52–55），存在样本不足/病态外推的限制。','',
'## 替代方式','',
'记R为attention后残差，H=norm2(R)，原增量D=W_down GELU(W_up H)。所有拟合只用训练节点，验证节点选ridge；不使用后继/祖先标签拟合FFN替代。','',
'| 方法 | 实际替代 | 仍保留的非线性 |','|---|---|---|',
'| ridge_full | D≈AR+b | attention与后续网络；当前norm2/GELU都被替代 |',
'| ridge_core | D≈AH+b | 当前norm2及attention、后续网络 |',
'| unitwise_core | 每个GELU(z_j)≈a_j z_j+b_j，保留原up/down矩阵 | 当前norm2及attention、后续网络 |',
'| unitwise_full | unitwise_core，再将norm2逐样本均值/标准差换成训练统计常数 | attention与后续网络；该FFN分支严格仿射于R |',
'| constant_norm_native | 原GELU，norm2改训练统计常数 | GELU、attention、后续网络 |',
'| taylor_core/full | 在固定训练均值处的一阶Taylor映射 | 依所取输入保留norm2；不在测试点重新线性化 |','',
'逐元GELU近似只拟合512条标量最小二乘直线；原up/down矩阵含原模型训练得到的信息，未被重新估计。它既是更强的权重保持基线，也不能被称为原模型没见过测试节点。','',
'## 当前答案能保留多少','',
'主共享拟合下，各自只替换一个测试留出端点位置，其他L1位置保持native，完整后续网络重算。下表每格为三个划分的BA均值；各列测试查询子集不同，必须只在列内与native比较。所有主表替代的原生输出仍100%是YES/NO。','',
'| 方法 | 简单起点 | 简单目标 | 困难起点 | 困难目标 |','|---|---:|---:|---:|---:|']
for m in METHODS:
    lines.append('| '+PRETTY[m]+' | '+' | '.join(fmt(ba(g,role,m)) for g in ('simple','hard') for role in ('source','target'))+' |')
lines+=['','保留W的逐元仿射替代明显优于重新估计整张矩阵，说明“新仿射拟合失败”混有估计与先验问题，不应直接归因于不可约非线性。困难起点仍有约1–1.5pp损失；样本少且每图单checkpoint，不宣称显著性或全部线性映射的理论上限。','',
'## 两个端点同时是留出节点、同时替换','',
'这组比单端点干预更严格，但测试样本很少：简单三划分分别34/31/40条（正例17/8/8），困难26/18/25条（正例11/8/19）。跨seed有重叠，不把总数当独立样本，不给出跨训练置信区间。','',
'| 方法 | 简单 BA | 困难 BA |','|---|---:|---:|']
for m in METHODS:
    lines.append('| '+PRETTY[m]+' | '+' | '.join(fmt([r['joint_both_heldout'][m]['constrained_ba'] for r in records(g)]) for g in ('simple','hard'))+' |')
lines+=['','困难常数norm2保留原GELU时，三划分全部测试预测与native逐条一致；其余方法的逐条改变比例、非法token率与正负支持数都在JSON。简单略高于native的点估计不当作已证实改进。','',
'## 困难起点：额外后代读出增益是否保留','',
'probe为固定候选跨留出焦点AUROC，排除拓扑上不可能成立的关系、自身及另一端点；每seed各自解码器，先聚合重复焦点查询，不跨折拼接分数。重新拟合probe测其学习效果；固定native probe测同一个读出对变化后状态的反应，后者也有坐标/分布偏移，均不直接等于语义中介。','',
'| 替代后残差 | 新probe | 原native probe固定 |','|---|---:|---:|']
for m in METHODS:
    lines.append('| '+PRETTY[m]+' | '+fmt(readout('hard','source','descendants',m))+' | '+fmt(readout('hard','source','descendants',m,True))+' |')
lines+=['','原norm2输入的新probe为0.858。保留norm2、逐元线性化GELU，后代新probe从0.948降到0.872；保留原GELU、固定norm2则仍有0.946。额外后代readout效应在本组替代中主要与GELU非线性有关，而不是norm2逐样本归一化。','',
'但固定native probe时，逐元核心仿射仍有0.923，仅比native0.948低约2.5pp；新probe的7.6pp下降包含估计方式的影响，不能全当作信息消失。对整张仿射拟合的失败，也不能排除更好仿射映射或任务相关拟合目标。','',
'注意答案表现与该probe并不一一对应：困难起点逐元全仿射BA0.965，而后代新probe仅0.874；原FFN BA0.980、probe0.948。数据更支持“多数当前YES/NO不需要完整保留这份额外probe增益”，不能据此断言模型不用后代信息。','',
'## 目标祖先与直接后继对照','',
'下表为主共享条件下的新probe AUROC；关系难度/密度不同，不把跨任务AUROC直接当作信息量排序。','',
'| 图/位置/关系 | 原FFN | 整体仿射 | 保留LN仿射核心 | 逐元全仿射 | 逐元仿射核心 |','|---|---:|---:|---:|---:|---:|']
for g in ('simple','hard'):
    for role,rel in [('target','ancestors'),('source','successors'),('source','descendants')]:
        lines.append(f'| {g}/{role}/{rel} | '+' | '.join(fmt(readout(g,role,rel,m)) for m in ('native','ridge_full','ridge_core','unitwise_full','unitwise_core'))+' |')
lines+=['','困难目标祖先约0.979在仿射替代后仍约0.978–0.980，目标答案也保留较好；简单目标祖先probe替代后反而较高。仍不能据此指定祖先为答案因果中介。','',
'## 重建误差与方法限制','',
'以下R²是对原FFN输出增量的留出节点重建，参照均值只用实际映射训练样本，非主模型准确率。','',
'| 方法 | 简单起点 | 简单目标 | 困难起点 | 困难目标 |','|---|---:|---:|---:|---:|']
for m in ('ridge_full','ridge_core','unitwise_full','unitwise_core','constant_norm_native','taylor_full','taylor_core'):
    lines.append('| '+PRETTY[m]+' | '+' | '.join(fmt([r['roles'][role]['heldout_delta_reconstruction'][m]['r2'] for r in records(g)]) for g in ('simple','hard') for role in ('source','target'))+' |')
lines+=['','单一训练均值处Taylor在困难图明显失败，不能拿它作为最强线性替代。固定权重、逐元全局GELU线性近似强得多。仿射重建不完美但答案仍好，说明重建全部增量细节与保持当前答案不是同一个目标。','',
'全仿射ridge后残差是R的可逆仿射变换：各条件rank均128。因此它不扩大R上的仿射probe函数类；其新probe得分变化可能来自先验/正则/统计估计。已将该条件probe解析折回R，所有测试分数误差<1e-3，无测试标签再拟合。这个控制不证明原非线性FFN未增加有用的可读表示。','',
'## 原输出头的直接后继检索','',
'不拟合新probe，直接将起点残差通过原final_norm与tied输出头。按每个留出焦点聚合查询，排除自身、另一端点及拓扑上不合法候选，在非平凡非空集合上取recall@真实出度。这是source位置的诊断logit lens，不是实际在分隔符生成答案，也不是完整WALK/PATH自由生成。与旧的全测试查询加权数字分母不同，不直接横比。','',
'| 条件 | 简单后继recall | 困难后继recall |','|---|---:|---:|']
for m in METHODS:
    key={'native':'native_post','ffn_off':'after_attention'}.get(m,m+'_post')
    lines.append('| '+PRETTY[m]+' | '+' | '.join(fmt([r['roles']['source']['original_head']['successors'][key]['recall_at_true_k'] for r in records(g)]) for g in ('simple','hard'))+' |')
lines+=['','简单图原头后继检索在FFN前仅0.014，FFN后0.694；逐元全仿射后降到0.522，尽管起点答案BA仍保留0.926（native0.934）。因此“多数YES/NO可由仿射替代保留”不等于全部FFN功能、后继检索与完整行为都等价。困难图FFN前原头已能读到0.791、FFN后0.903，说明后继原生读出并非总要等到FFN；不能把知识只定位到FFN。','',
'## 对FFN作用的判断','',
'1. 有跨节点可推广的特征变换作用，而不只是“见过100个地址才线性插值”：严格60节点训练、另外20节点验证、20节点测试下，原矩阵保持的仿射替代保留了大部分答案。',
'2. 当前两个模型的大部分可达性答题作用不要求保留L1节点FFN的全部非线性；尤其逐元近似比新dense拟合强，权重先验很重要。',
'3. 困难起点原GELU对后代probe的额外提升有可测影响；这个表示效应与答案必要性应分开。只固定norm2却不去GELU，几乎保留原表示/答案。',
'4. 不把阴性拟合结果当作所有仿射映射不可能，也不把近似成功当作知识只在embedding。原W_up/W_down仍保留训练过的图知识/先验；权重存储与输入解码的唯一分解仍未确定。','',
'![FFN留出节点仿射替换](ffn_role.png)','',
'复现：repo root `.venv/bin/python studies/ffn_affine_holdout/run.py`，然后运行同目录report.py。只生成两个结果JSON、一个PNG和本报告，不保存activation或探针/模型权重。所有数值覆盖每图一个Direct checkpoint，不外推其他协议/seed。','']
(HERE/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
print('Wrote REPORT.md and ffn_role.png')

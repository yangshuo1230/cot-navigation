"""Matched continuation/interpolation analysis; no checkpoint selection on test."""
from pathlib import Path
from collections import defaultdict,Counter
import json
import statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).parent
E=json.loads((ROOT/'evidence.json').read_text(encoding='utf-8-sig'))
assert E['complete']
RUNS=E['runs']
MODES=('parent','short','long','endpoints')
LABEL={'parent':'原50k模型','short':'续训仅76','long':'续训仅220','endpoints':'续训76/220各半'}
CASES=['id','id_disconnected_pad32','train_length220','extrap_length292','wide32_branch2',
       'wide32_densityhalf','deep16_sameprompt','deep24','deep32','deep48','large64']
CN={'id':'原图76','id_disconnected_pad32':'独立子图补长148（内插）','train_length220':'独立子图补长220（训练端点）',
    'extrap_length292':'独立子图补长292（外推）','wide32_branch2':'实际加宽32/48/3','wide32_densityhalf':'实际加宽加密32/96/3',
    'deep16_sameprompt':'同长度加深16/24/7','deep24':'加深24/40/5','deep32':'加深32/56/7',
    'deep48':'加深48/88/11','large64':'加宽加深64/112/7'}


def stats(rows):
    n=len(rows)
    old=[r for r in rows if r['hop']<=3]
    new=[r for r in rows if r['hop']>3]
    partial=lambda a:{k:sum(bool(r[k]) for r in a)/len(a) for k in ('correct','execution_valid','trace_valid')}
    return {'n':n,**{k:sum(bool(r[k]) for r in rows)/n for k in
        ('correct','trace_valid','execution_valid','format_fail','limit')},
        'errors':dict(Counter(r['trace_error'] for r in rows if r['trace_error'])),
        'first_error_record0':sum(not r['trace_valid'] and r['valid_records']==0 for r in rows),
        'by_hop':{str(h):{'n':len(a),'accuracy':sum(r['correct'] for r in a)/len(a),
                         'execution_valid':sum(r['execution_valid'] for r in a)/len(a),
                         'trace_valid':sum(r['trace_valid'] for r in a)/len(a)}
                  for h in sorted(set(r['hop'] for r in rows)) for a in [[r for r in rows if r['hop']==h]]},
        'old_hops':partial(old),'new_hops':partial(new) if new else {}}


S={}
for key,run in RUNS.items():
    assert len(run['predictions'])==1760,(key,len(run['predictions']))
    assert len({(r['case'],r['index']) for r in run['predictions']})==1760
    groups=defaultdict(list)
    for row in run['predictions']:groups[row['case']].append(row)
    assert set(groups)==set(CASES)
    S[key]={c:stats(rows) for c,rows in groups.items()}
for p in ('bfs_state','dfs_state'):
    v=[E['continuation_verification'][p+'/'+m] for m in ('short','long','endpoints')]
    assert {x['step'] for x in v}=={5000}
    assert {x['queries_seen'] for x in v}=={320000}
    assert len({x['query_sampler_sha256'] for x in v})==1
    assert len({x['edge_rng_sha256'] for x in v})==1
    assert len({RUNS[p+'/'+m]['experiment']['parent_sha256'] for m in ('short','long','endpoints')})==1
    # Every arm's example identity/hop/label/prompt matches; do not compare
    # the original192-case audit directly to the current96-case subset.
    anchors={(r['case'],r['index']):(r['hop'],r['label'],r['prompt_len']) for r in RUNS[p+'/parent']['predictions']}
    for mode in ('short','long','endpoints'):
        now={(r['case'],r['index']):(r['hop'],r['label'],r['prompt_len']) for r in RUNS[p+'/'+mode]['predictions']}
        assert now==anchors
(ROOT/'summary.json').write_text(json.dumps(S,indent=2),encoding='utf-8')
(ROOT/'completion.json').write_text(json.dumps({'continuations':6,'additional_updates_each':5000,
    'frozen_controls':2,'eval_outputs':14080,'all_queries_and_edge_rng_matched':True,
    'heldout_interpolation_prompt_length':148},indent=2))


def pct(x):return f'{100*x:.2f}%'


def table(head,rows):
    return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])+'\n'


interp=table(['模型','训练方式','n','148答案正确','148动作骨架合法','148完整状态合法'],
    [[p,LABEL[m],S[p+'/'+m]['id_disconnected_pad32']['n']]+[pct(S[p+'/'+m]['id_disconnected_pad32'][k])
      for k in ('correct','execution_valid','trace_valid')] for p in ('bfs_state','dfs_state') for m in MODES])
alltables=[]
for p in ('bfs_state','dfs_state'):
    alltables += ['## '+p+'：答案 / 动作骨架 / 完整状态\n',
        table(['条件']+[LABEL[m] for m in MODES],
            [[CN[c]]+[' / '.join(pct(S[p+'/'+m][c][k]) for k in ('correct','execution_valid','trace_valid'))
                      for m in MODES] for c in CASES])]
(ROOT/'ALL_CONDITIONS.md').write_text('\n'.join(alltables),encoding='utf-8')
hoplines=[]
for p in ('bfs_state','dfs_state'):
    for c in ('deep16_sameprompt','deep32','deep48'):
        hoplines += ['## '+p+' / '+CN[c]+'\n',table(['hop']+[LABEL[m] for m in MODES],
            [[h]+[' / '.join(pct(S[p+'/'+m][c]['by_hop'][h][k]) for k in ('accuracy','execution_valid')) for m in MODES]
             for h in S[p+'/parent'][c]['by_hop']])]
(ROOT/'BY_HOP.md').write_text('每格：答案准确率 / 动作骨架合法率。每个hop正负各16例。\n\n'+'\n'.join(hoplines),encoding='utf-8')
urls=[(key,r['wandb_status']['url']) for key,r in RUNS.items() if r.get('wandb_status',{}).get('url')]
report=f'''# 轨迹错误与prompt长度内插：已执行的控制实验

本次在原50k checkpoint基础上完成6个独立5k续训（BFS/DFS×short/long/endpoints），另对两个原checkpoint做相同集合的冻结对照。每组最终评估1760条，共14080条自由生成。没有根据test选择checkpoint；均用预先指定的最终5k权重。没有修改主训练框架、原始数据集或原批次输出。

## 实验针对的假设

原训练prompt固定76token，但CoT输出和总训练序列长度本来就不固定。要检验的是固定图输入规模/查询边界的训练支持范围，而不是“模型从没在148这个位置计算过”。边列表补长同时改变相对位置、无关节点数和注意力竞争；本实验不单独证明某个位置编码机制是原因。

保持查询组件16节点24边、原节点名、原边的相对顺序、查询、标签及整条监督轨迹完全不变；只加入不相连的16节点24边组件并交错边列表。每个组件增加72个prompt token，没有新增CoT token，也不是Filler。

- short：只训练76token，无额外组件。
- long：只训练220token，即额外两个组件。
- endpoints：每个逻辑batch严格32条76token+32条220token。
- 测试148token是额外一个组件，两种训练端点均不包含它；292token是额外三个组件，用于外推。

每组追加5000步×64查询=320000次相同查询曝光；原邻居重排RNG及最终query sampler状态哈希已核对相同；同一协议三组父权重SHA256相同。重置优化器，AdamW/SDPA/bf16/torch.compile沿用原实现，LR=1e-4，前100步warmup。旧训练序列与缓存不修改；新prompt的loss mask整体正确平移。CPU预检48条通过且没有保存smoke记录或checkpoint。

不同组的输入token/FLOPs不同，不是等计算量实验；long比endpoints平均输入更长，可以帮助判断改善是否仅来自混合短长。新图seed仍为单个，续训不等价于从零训练长度混合模型。

## 首次错误细分

详见ERROR_ANALYSIS.md和error_breakdown.json。基于上一轮的同一192条配对ID/padding样本：

- 原分布BFS43条状态非法，其中38条首次错误包含漏合法候选；动作首次错误包括25次过早POP、10次PUSH不存在的边、4次重复PUSH、4次无候选却PUSH。
- padding后BFS首次错误144次为visited不一致，其中127次含重复节点；另30次为候选错误。动作首次错误包括67次过早POP、21次走不存在的边等。
- padding后DFS192/192在第一条状态就出现5字段而非4字段的记录；175/192还在有候选时直接EXHAUST。它不是先正确执行很长再逐步积累误差。
- 固定图seed0普通DFS的64条抽样，动作错误为21次到了目标却未FOUND，以及6次未到目标就FOUND；普通BFS也有相同数量的这两类错误，另有两次PUSH错误。这里主要是目标/终止判断，不等同于新图padding的格式错误。

其他形式也要区分“走错”与“没完成”：新图DFS节点序列192例中，149例是终止不合适、32例不是合法DFS展开顺序；BFS节点序列有98例不满足BFS展开顺序、26例重复节点。Path负例在同prompt长度的8层图中，严格按生成器走到sink仅50.22%合法，但若只要求路径前缀逐边正确则96.65%合法；这类短负路径不应直接当成走错边。相关例子和较宽松指标见NONSTATE_ERRORS.md。错误码non_node_token也可能是生成了本图没有的节点，而非特殊token。

## 未见148长度的内插结果

{interp}

每组96条完全相同的held-out查询，正负均衡、hop1..3各32条。不能与上一轮192条或完整10000条test直接相减。

长度覆盖显著缓解骨架与状态错误，尤其DFS。固定长训练也能向较短148泛化，因此不能把改善归功于“两个端点混合”这一项本身。还需要看它是否损伤原76输入：

{table(['模型','方式','76答案','76动作骨架','76完整状态'],[[p,LABEL[m]]+[pct(S[p+'/'+m]['id'][k]) for k in ('correct','execution_valid','trace_valid')]
       for p in ('bfs_state','dfs_state') for m in MODES])}

DFS仅长训练在76输入上完整状态合法率降至29.17%，端点混合保留86.46%，因此混合长度对避免短长度行为遗忘有实际价值。BFS两组短长度的表现接近，没有看到同样严重的遗忘。

## 真正改变图结构/路径深度的结果

{table(['条件','BFS短对照答案/骨架','BFS端点答案/骨架','DFS短对照答案/骨架','DFS端点答案/骨架'],
       [[CN[c]]+[' / '.join(pct(S[p+'/'+m][c][k]) for k in ('correct','execution_valid'))
                for p,m in [('bfs_state','short'),('bfs_state','endpoints'),('dfs_state','short'),('dfs_state','endpoints')]]
        for c in ('wide32_branch2','wide32_densityhalf','deep16_sameprompt','deep24','deep32','deep48','large64')])}

完整四组结果见ALL_CONDITIONS.md，逐hop见BY_HOP.md。所有组仍仅训练查询hop1..3，并没有偷偷加入长路径监督。因此真大图/新增hop的改善是迁移，但若仍失败，说明输入长度覆盖不能替代更复杂状态和更长搜索的训练。

尤其同prompt76的8层图：BFS短对照答案83.48%/骨架57.14%，端点组82.59%/57.14%；DFS也基本不变。这是“长度覆盖不能解决所有长路径问题”的直接对照。真实48节点11跳条件下，BFS端点组答案71.59%，但骨架仅30.97%，还不能称为稳定搜索。

## 内插集的状态错误分布

{table(['模型','方式','首次状态错误分布','首错在record0的条数'],[[p,LABEL[m],str(S[p+'/'+m]['id_disconnected_pad32']['errors']),S[p+'/'+m]['id_disconnected_pad32']['first_error_record0']]
       for p in ('bfs_state','dfs_state') for m in MODES])}

BFS固定短对照的visited首错66/96，端点组为0/96；DFS固定短组第一条就格式错误96/96，端点组格式首错为0/96。剩余错误主要回到候选集合或动作/终止判断，而不是查询一结束就进入错误格式。

## 当前可以支持的结论

固定短prompt/查询边界的训练分布是一个重要失败来源：相同额外步数、相同查询和边顺序下，短长度对照不能修复内插骨架，而长长度暴露与端点混合均显著修复。端点混合还能保留DFS短输入能力。

但这不证明全部错误都由位置问题造成。补长同时提供了无关子图干扰训练，且各组FLOPs不同；候选集合、状态规模和超出3跳的执行仍有独立困难。单seed、96例内插集的探索性结果需要重复验证，尤其BFS答案提高几个百分点不应当成精确效应量；骨架和完整状态的提升更大。

附加KV-cache检查：在prompt76/148/268/292各抽查33个预测位置，FP32完整前向与逐token cache前向的argmax全部一致；最大logit差约4.7e-4，最小top2间隔大于1.5。未发现这些例子存在明显的cache长度索引错误，记录见cache_validation.json；不声称这是穷尽的实现正确性证明。

## 结果文件、W&B与复现

W&B group：length_interp_v1_20261017。六个正式训练run已使用原项目的在线配置，训练损失和长度评估指标上传，未上传checkpoint artifact。

{chr(10).join('- ['+k+']('+u+')' for k,u in urls)}

服务器目录studies/length_interpolation/runs/PROTOCOL/MODE/含experiment.json、resolved_config.yaml、checkpoint、生成结果和完成标记。checkpoint中的step5000是追加阶段步数，父checkpoint50000在experiment.json中单独标记。

从项目根目录执行，例如：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=1 .venv/bin/python studies/length_interpolation/experiment.py --protocol bfs_state --mode endpoints --steps 5000
```

这会拒绝覆盖已有checkpoint；训练恢复可传--resume PATH。当前完整结果不需要重跑。新重复应使用新output/seed配置，不能把相同seed重跑当独立重复。

本地原始快照evidence.json、汇总summary.json、错误细分ERROR_ANALYSIS.md、图interpolation.png。短输入对照用于排除仅追加训练步数这一解释；更多seed和从零训练的覆盖对照尚未执行。
'''
(ROOT/'REPORT.md').write_text(report,encoding='utf-8')

plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10})
fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
lengths=[76,148,220,292];cases=['id','id_disconnected_pad32','train_length220','extrap_length292']
styles={'parent':('#999999',':'),'short':('#cd8832','-'),'long':('#4979ad','-'),'endpoints':('#329578','-')}
for i,p in enumerate(('bfs_state','dfs_state')):
    for j,(metric,title) in enumerate([('correct','答案准确率'),('execution_valid','动作骨架合法率')]):
        ax=axes[i,j]
        for mode in MODES:
            color,ls=styles[mode]
            ax.plot(lengths,[100*S[p+'/'+mode][c][metric] for c in cases],marker='o',label=LABEL[mode],color=color,ls=ls)
        ax.set_xticks(lengths);ax.set_ylim(0,105);ax.axvspan(76,220,color='#329578',alpha=.05)
        ax.set_title(p+'：'+title);ax.set_xlabel('prompt长度（查询组件与轨迹不变）');ax.set_ylabel('%');ax.legend(fontsize=9)
fig.savefig(ROOT/'interpolation.png',dpi=160);plt.close(fig)
print(interp)
print('ALL CONDITIONS')
print('\n'.join(alltables))

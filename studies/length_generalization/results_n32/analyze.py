"""Summarize frozen-model trajectory audits and extrapolation results."""
from collections import Counter,defaultdict
from pathlib import Path
import json
import math
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).parent
META=json.loads((ROOT/'metadata.json').read_text())
ROWS=[json.loads(x) for x in (ROOT/'predictions.jsonl').read_text().splitlines()]
assert len(ROWS)==len({(r['model'],r['case'],r['index']) for r in ROWS})
sys.path.insert(0,str(ROOT.parent/'retrain_v3'/'src'))
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.protocols.dfs import dfs_search
from cot_navigation.protocols.bfs import bfs_search,first_expansion_order
from cot_navigation.protocols.render import render_steps
from trace_audit import validate_execution,self_test
tok=Tokenizer()
postprocess_checks=self_test(tok,dfs_search,bfs_search,render_steps,first_expansion_order)
INPUTS={(c,r['index']):r for c,rs in META['inputs'].items() for r in rs}
for r in ROWS:
    if r['protocol'] in ('dfs_state','bfs_state'):
        original=INPUTS[r['case'],r['index']]
        adj={v:[] for v in original['nodes']}
        for u,v in original['edges']:adj[original['nodes'][u]].append(original['nodes'][v])
        r['execution_valid']=validate_execution(r['trace_tokens'],r['pred'],adj,r['source'],r['target'],tok,r['protocol']=='bfs_state')
        assert not r['trace_valid'] or r['execution_valid']
        assert not r['execution_valid'] or r['correct']
    else:r['execution_valid']=r['trace_valid']
(ROOT/'postprocess_validation.json').write_text(json.dumps({'exhaustive_checks':postprocess_checks}))
GROUPS=defaultdict(list)
for row in ROWS:GROUPS[row['model'],row['case']].append(row)
for (model,case),rows in GROUPS.items():
    assert len(rows)==META['records'][case],(model,case,len(rows))
assert len(GROUPS)==89 and len(ROWS)==28736
(ROOT/'completion.json').write_text(json.dumps({'outputs':len(ROWS),'model_condition_groups':len(GROUPS),
    'all_expected_counts_verified':True,'duplicates':0},indent=2))
CASES=[x[0] for x in META['shapes']]
LABELS={'id':'原分布16/24/3','id_disconnected_pad32':'独立子图补长32/48/3',
        'wide32_branch2':'加宽32/48/3','wide32_densityhalf':'加宽加密32/96/3',
        'deep16_sameprompt':'同输入长度16/24/7','deep24':'加深24/40/5','deep32':'加深32/56/7',
        'deep48':'加深48/88/11','large64':'加宽加深64/112/7','fixed0':'固定图seed0'}
MODELS=['path','path_negative','dfs_state','bfs_state','dfs_nodes','bfs_nodes']


def rate(rows,key):return sum(bool(r[key]) for r in rows)/len(rows) if rows else None


def stats(rows):
    return {'n':len(rows),'answer_accuracy':rate(rows,'correct'),'trajectory_valid':rate(rows,'trace_valid'),
            'execution_valid':rate(rows,'execution_valid'),
            'answer_and_trace_valid':sum(r['correct'] and r['trace_valid'] for r in rows)/len(rows),
            'exact_reference':rate(rows,'exact_reference'),'format_fail':rate(rows,'format_fail'),
            'generation_limit':rate(rows,'limit'),
            'gold_over_context':sum(not r['reference_fits_context'] for r in rows),
            'mean_generated_tokens':float(np.mean([r['generated_trace_len'] for r in rows])),
            'mean_reference_tokens':float(np.mean([r['reference_trace_len'] for r in rows])),
            'yes_recall':rate([r for r in rows if r['label']==1],'correct'),
            'no_recall':rate([r for r in rows if r['label']==0],'correct'),
            'errors':dict(Counter(r['trace_error'] for r in rows if r['trace_error'])),
            'by_hop':{str(h):{'n':len(a),'answer_accuracy':rate(a,'correct'),
                             'trajectory_valid':rate(a,'trace_valid')}
                      for h in sorted(set(r['hop'] for r in rows))
                      for a in [[r for r in rows if r['hop']==h]]},
            'within_training_hops':{'n':len(a:=[r for r in rows if r['hop']<=3]),
                                    'answer_accuracy':rate(a,'correct'),'trajectory_valid':rate(a,'trace_valid')},
            'new_hops':{'n':len(a:=[r for r in rows if r['hop']>3]),
                        'answer_accuracy':rate(a,'correct'),'trajectory_valid':rate(a,'trace_valid')}}


SUMMARY={model:{case:stats(rows) for (m,case),rows in GROUPS.items() if m==model}
         for model in sorted(set(m for m,c in GROUPS))}
(ROOT/'summary.json').write_text(json.dumps(SUMMARY,indent=2),encoding='utf-8')
paired={}
for model in SUMMARY:
    if not model.startswith('context/'):continue
    first={r['index']:r for r in GROUPS[model,'id']}
    padding={r['index']:r for r in GROUPS[model,'id_disconnected_pad32']}
    assert first.keys()==padding.keys()
    for index,a in first.items():
        b=padding[index]
        assert all(a[k]==b[k] for k in ('source','target','label','hop','reference_trace_len'))
    paired[model]={'n':len(first),'answer_lost':sum(a['correct'] and not padding[i]['correct'] for i,a in first.items()),
                   'answer_gained':sum(not a['correct'] and padding[i]['correct'] for i,a in first.items()),
                   'reference_lengths_unchanged':True}
(ROOT/'paired_padding_control.json').write_text(json.dumps(paired,indent=2))


def pct(value):return '—' if value is None else f'{value*100:.2f}%'


def table(head,rows):
    return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])+'\n'


context=table(['图条件：节点/边/最大跳数','n']+MODELS,
    [[LABELS[c],META['records'][c]]+[pct(SUMMARY['context/'+m][c]['answer_accuracy']) for m in MODELS] for c in CASES])
trajectory=table(['模型','原分布n','答案正确','完整轨迹合法','动作骨架合法','答案且轨迹正确','完全匹配参考'],
    [[m,192]+[pct(SUMMARY['context/'+m]['id'][k]) for k in
             ('answer_accuracy','trajectory_valid','execution_valid','answer_and_trace_valid','exact_reference')] for m in MODELS])
fixed=table(['固定图seed0模型','n','答案正确','完整轨迹合法','动作骨架合法','答案且轨迹正确','完全匹配参考'],
    [[m.split('/')[1],SUMMARY[m]['fixed0']['n']]+[pct(SUMMARY[m]['fixed0'][k]) for k in
             ('answer_accuracy','trajectory_valid','execution_valid','answer_and_trace_valid','exact_reference')]
     for m in SUMMARY if m.startswith('fixed/') and m!='fixed/direct'])
bfs=table(['图条件','答案正确','完整轨迹合法','动作骨架合法','1–3跳答案','4+跳答案','格式失败','生成触顶','gold超窗数'],
    [[LABELS[c]]+[pct(SUMMARY['context/bfs_state'][c][k]) for k in ('answer_accuracy','trajectory_valid','execution_valid')]+
     [pct(SUMMARY['context/bfs_state'][c][region]['answer_accuracy']) for region in ('within_training_hops','new_hops')]+
     [pct(SUMMARY['context/bfs_state'][c][k]) for k in ('format_fail','generation_limit')]+
     [SUMMARY['context/bfs_state'][c]['gold_over_context']] for c in CASES])
hop_lines=[]
for c in CASES:
    hop_lines += ['## '+LABELS[c]+'\n',table(['hop','n','BFS答案','BFS轨迹合法','DFS答案','DFS轨迹合法','负例Path答案'],
        [[h,v['n'],pct(v['answer_accuracy']),pct(v['trajectory_valid']),
          pct(SUMMARY['context/dfs_state'][c]['by_hop'][h]['answer_accuracy']),
          pct(SUMMARY['context/dfs_state'][c]['by_hop'][h]['trajectory_valid']),
          pct(SUMMARY['context/path_negative'][c]['by_hop'][h]['answer_accuracy'])]
         for h,v in SUMMARY['context/bfs_state'][c]['by_hop'].items()])]
(ROOT/'BY_HOP.md').write_text('\n'.join(hop_lines),encoding='utf-8')

bins=[(0,32),(33,64),(65,128),(129,256),(257,512),(513,1024),(1025,4096),(4097,100000)]
lengthlines=['# 按参考轨迹token长度分桶\n',
    '混合图条件的描述性统计，不作为长度变化的因果效应。不同协议的参考长度定义不同，不能直接横向当等计算量比较。\n']
for m in MODELS:
    lengthlines += ['## '+m+'\n',table(['参考trace长度','n','答案正确','轨迹合法','生成触顶'],
        [[f'{lo}..{hi}',len(a),pct(rate(a,'correct')),pct(rate(a,'trace_valid')),pct(rate(a,'limit'))]
         for lo,hi in bins for a in [[r for r in ROWS if r['model']=='context/'+m and lo<=r['reference_trace_len']<=hi]] if a])]
(ROOT/'BY_TRACE_LENGTH.md').write_text('\n'.join(lengthlines),encoding='utf-8')

# Detailed reproducible failure cases, with the expected state at the first
# invalid record reconstructed from the model's OWN preceding legal actions.
names=['<PAD>','<EOS>','<EDGE>','<QUERY>','|',';','<YES>','<NO>','<PUSH>','<POP>','<FOUND>','<EXHAUST>','<WALK>','<DOT>']+[f'<V{i}>' for i in range(256)]
decode=lambda ids:' '.join(names[i] for i in ids)
examples=[]
for model,case,kind in [('context/bfs_state','id','correct_but_invalid'),
                        ('context/bfs_state','id_disconnected_pad32','correct_but_invalid'),
                        ('context/bfs_state','wide32_branch2','correct_but_invalid'),
                        ('context/bfs_state','deep48','correct_but_invalid'),
                        ('context/dfs_state','id_disconnected_pad32','invalid'),
                        ('context/path_negative','deep48','invalid')]:
    selected=next((r for r in GROUPS[model,case] if not r['trace_valid'] and (kind!='correct_but_invalid' or r['correct'])),None)
    if selected is None:continue
    source=next(x for x in META['inputs'][case] if x['index']==selected['index'])
    prompt=[t for u,v in source['edges'] for t in (2,source['nodes'][u],source['nodes'][v])]+[3,selected['source'],selected['target'],4]
    snippet={'model':model,'case':case,'record':selected,'input':source,'prompt_text':decode(prompt)}
    if selected['protocol'].endswith('_state'):
        records=[];start=0
        for end,t in enumerate(selected['trace_tokens']):
            if t==5:records.append(selected['trace_tokens'][start:end]);start=end+1
        n=selected.get('valid_records',0) or 0
        frontier,vis=[selected['source']],[selected['source']]
        fifo=selected['protocol']=='bfs_state'
        for rec in records[:n]:
            action=rec[rec.index(4,rec.index(4,rec.index(4)+1)+1)+1:]
            if action[0]==8:frontier.append(action[1]);vis.append(action[1])
            elif action[0]==9:frontier.pop(0 if fifo else -1)
        adj={v:[] for v in source['nodes']}
        for u,v in source['edges']:adj[source['nodes'][u]].append(source['nodes'][v])
        cur=frontier[0] if fifo else frontier[-1]
        snippet['first_bad_record_index']=n
        snippet['first_bad_record_text']=decode(records[n]) if n<len(records) else '<no complete record>'
        snippet['expected_before_record']={'frontier':decode(frontier),'visited':decode(vis),
                                          'candidate_set':decode(sorted(set(adj[cur])-set(vis)))}
    examples.append(snippet)
(ROOT/'FAILURE_EXAMPLES.json').write_text(json.dumps(examples,indent=2),encoding='utf-8')
exampletext=['# 真实生成错误例子\n','全部是冻结模型自由生成，不是teacher forcing。完整trace token保存在JSON。\n']
for e in examples:
    r=e['record'];exampletext += ['## '+e['model']+' / '+e['case']+'\n',
        f"index={r['index']}，hop={r['hop']}，label={r['label']}，pred={r['pred']}，error={r['trace_error']}。\n",
        '完整输入：\n```text\n'+e['prompt_text']+'\n```\n']
    if 'first_bad_record_text' in e:
        exampletext += [f"第一个错误record（从0计）：{e['first_bad_record_index']}。\n",
            '模型输出：\n```text\n'+e['first_bad_record_text']+'\n```\n',
            '依据模型此前合法动作推得的正确状态：\n```json\n'+json.dumps(e['expected_before_record'],ensure_ascii=False,indent=2)+'\n```\n']
    else:exampletext += ['模型trace：\n```text\n'+decode(r['trace_tokens'])+'\n```\n']
(ROOT/'FAILURE_EXAMPLES.md').write_text('\n'.join(exampletext),encoding='utf-8')

report=f'''# CoT轨迹审计与新图长度泛化

冻结本批所有模型的50k latest checkpoint，使用原项目greedy KV-cache decoder、bf16、原始256节点词表、原4096上下文上限。没有训练、修改权重、修改训练数据或W&B上传。服务器仅使用空闲GPU0，不干扰GPU1..7的其他任务。

## BFS是否随机

新图BFS配置shuffle_context_edges=true、random_search_order=false。实际Trainer._training_task给transform传入边重排RNG；transform重排EDGE三元组，对搜索协议从新prompt重建邻接表，再重新渲染BFS。因此新图BFS每次采样都会随机改变邻居优先级。固定图普通BFS两项均false，使用固定邻接顺序；固定图只有dfs_random单独启用访问顺序随机化，没有对应的随机BFS训练分支。

## 参考与生成轨迹的核对标准

参考缓存检查：{json.dumps(META['reference_audit'],ensure_ascii=False)}。全部通过；另穷举4节点DAG和端点的{META['validator_checks']}项验证检查通过。

- 全状态DFS/BFS：独立重放模型自己的动作，检查初始状态、stack/queue、visited顺序、候选集合、PUSH选择、POP和终止。允许不同合法候选排序，不要求匹配唯一参考。状态列表顺序是协议的一部分，不能随意更改。
- DFS/BFS节点序列：检查其能否由相应搜索的首次展开顺序产生，且不重复节点。DFS回溯和BFS兄弟节点之间不一定存在直接边，不能把这种序列当路径验证。
- Path：检查起点、真实边、无重复；YES必须到达目标。负例随机Path按当前协议必须走到sink，但单条随机路径无法证明不可达。原Path的空NO输出也不构成搜索证明。
- “轨迹合法”“答案正确”“二者同时成立”分别报告；Path的合法负例路径可能伴随错误NO。Direct不含轨迹，其“空轨迹合法”不代表任何搜索能力。
- 全状态额外检查“动作骨架合法”：忽略展示的visited/candidate/frontier注释，仅依照真实图重放PUSH/POP/终止动作。这样能区分状态抄写错误与实际搜索动作错误；FOUND时漏写目标的出邻居也会降低完整轨迹合法率，但未必影响动作骨架。后处理独立穷举检查共{postprocess_checks}项通过。

新图原分布192条抽样（每hop/label32条）：

{trajectory}

固定图seed0的64条held-out多跳查询（其他图的生成轨迹本次未重测）：

{fixed}

## 长度泛化设计

每个条件中，每个hop都有32 YES和32 NO，均为独立图/查询；同一条件所有模型看到完全相同输入。原分布来自原test；OOD复用原图生成规则和标签平衡方式，排除已有结构哈希。共有{sum(META['records'][c] for c in CASES)}个新图条件记录，9个模型共{sum(len(a) for (m,c),a in GROUPS.items() if m.startswith('context/'))}次自由生成。固定图8个模型各64条，合计{len(ROWS)}条输出。

表中最大跳数是图层数减1；实际最长路径也记录在metadata，正例h的实际路径长恰为h。图是相邻层DAG，原模型训练最大hop=3。

- 独立子图补长：对原分布同一查询图增加一张不相连子图；查询、标签、原节点名和原组件邻居顺序完全不变，prompt76→148。它改变输入长度/干扰项，不增加查询搜索难度。
- 加宽32/48：非末层平均出度约2，保持原分支规模；加宽32/96保持原0.5边密度，分支规模增大。两种都不要求超过3跳。
- 同输入长度16/24/7：改为8层×2节点，节点数、边数、prompt76保持不变，最大hop增至7。形状/密度同时变化，不能把差异完全归为hop的单一因果效应。
- 加深24/40/5、32/56/7、48/88/11：保持宽4、边密度0.5及平均出度约2。
- 加宽加深64/112/7：8层×8节点，平均出度约2。

## 最终答案准确率

{context}

关键观察：BFS是保留能力最强的CoT，但没有保持原分布水平；DFS与节点序列在输入明显变长时大多回到接近chance。Path负例在deep48和large64出现95.60%/100%的格式失败，低答案分数不能简单解释成只是在YES/NO间选错。

原Path的100%“轨迹合法”只是模型始终直接输出空NO，协议允许空负例；这不是100%的有效推理或搜索。应看答案与轨迹联合正确率，并结合协议本身能否证明NO。

## BFS全状态：答案、过程与新hop分开看

{bfs}

两个关键对照：

- 同一192个原分布查询加入不相连子图后，BFS答案95.83%→86.46%，完整状态77.60%→8.85%，动作骨架77.60%→38.02%。配对统计21条由对变错、3条由错变对。查询节点名、标签和参考轨迹长度均经断言确认不变；这些样本生成触顶和gold超窗均为0。
- 同输入长度16/24/7中，BFS在1..3跳为99.48%，4/5/6/7跳依次93.75%/81.25%/57.81%/50.00%。这说明能部分扩展到4跳，但没有稳定延伸到7跳；不能把此处下降归因于prompt token变多。拓扑宽度/密度也变化，仍需避免把该对照当成只改变hop的纯因果实验。
- 上述7跳的50%是全部回答NO：YES召回0/32，NO召回32/32。4/5/6跳YES召回分别29/32、20/32、5/32，下降主要来自长距离可达正例。
- 固定图随机DFS本次64例答案100%、完整状态96.88%、动作骨架98.44%，但仅15.62%精确匹配随机参考。因此对随机顺序训练，精确匹配明显低估合法轨迹；对新图较大的输入，放宽为语义与动作重放后仍存在大量真实错误。

上述数字均为本次探索性样本，不能与上一轮完整test的97.42%当作同一集合比较。原分布可匹配原正式test的1,600条模型-查询预测均完全一致（见same_id_consistency.json），没有发现新评估入口或batch变化造成答案差异。

新增hop的结果不应与原1..3hop混为同一分布；BY_HOP.md给出完整逐hop结果。BY_TRACE_LENGTH.md给出混合条件的参考token长度分桶，仅作描述性统计。

生成预算没有依据gold长度设定，也没有强制答案。max_new_tokens=0沿用原解码器语义，生成到首个YES/NO、提前EOS或4096上下文耗尽。gold超窗样本未过滤，独立标记，避免通过丢弃长轨迹美化结果。先看同prompt深度和无关子图对照，再解释极大图上的窗口因素。

## 失败定位与解释边界

summary.json按模型/条件记录第一种语义错误分布、逐hop结果、格式失败和触顶比例；FAILURE_EXAMPLES.md展示真实生成中第一个非法状态及正确期望状态。对于“答案对、状态错”的样本，最终分类得分不能充当合法搜索证据。

这些是单个训练checkpoint在每hop64查询上的探索性测试，没有训练种子重复；原分布抽样分数不替代此前完整10000条test。图大小、层数、稀疏度与gold轨迹长度并非全部独立，两个额外对照用于减少混淆。现象提示需要研究模型是否依赖固定prompt位置/图规模与固定状态模板，但本次没有定位具体神经机制。

## 文件与复现

脚本：run_eval.py、trace_audit.py；冻结checkpoint入口与输入记录：metadata.json；逐样本生成：predictions.jsonl；汇总：summary.json；图：length_generalization.png。

服务器从项目根目录执行：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python studies/length_generalization/run_eval.py --per-label 32 --batch 32 --output studies/length_generalization/results_n32
```

现有output会跳过已经写出的(model,case,index)，可续评；要做独立重复应更换seed实现及新output，不能把同seed重复当独立样本。
'''
(ROOT/'REPORT.md').write_text(report,encoding='utf-8')

plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10})
fig,axes=plt.subplots(1,3,figsize=(19,6),layout='constrained')
matrix=np.array([[SUMMARY['context/'+m][c]['answer_accuracy']*100 for m in MODELS] for c in CASES])
im=axes[0].imshow(matrix,vmin=0,vmax=100,cmap='YlGnBu',aspect='auto')
axes[0].set_xticks(range(6),['Path','负例Path','DFS状态','BFS状态','DFS节点','BFS节点'],rotation=40,ha='right')
axes[0].set_yticks(range(len(CASES)),[LABELS[c] for c in CASES]);axes[0].set_title('答案准确率 / %')
for i in range(len(CASES)):
    for j in range(6):axes[0].text(j,i,f'{matrix[i,j]:.1f}',ha='center',va='center',color='white' if matrix[i,j]>65 else 'black',fontsize=8)
for ax,case in zip(axes[1:],['deep16_sameprompt','deep48']):
    for model,metric,label,color,ls in [('bfs_state','answer_accuracy','BFS答案','#347eac','-'),
                                        ('bfs_state','trajectory_valid','BFS轨迹合法','#347eac','--'),
                                        ('dfs_state','answer_accuracy','DFS答案','#cd8730','-'),
                                        ('dfs_state','trajectory_valid','DFS轨迹合法','#cd8730','--')]:
        d=SUMMARY['context/'+model][case]['by_hop'];h=list(map(int,d))
        ax.plot(h,[d[str(x)][metric]*100 for x in h],marker='o',label=label,color=color,ls=ls)
    ax.axvline(3.5,color='gray',ls=':');ax.set_ylim(0,105);ax.set_xticks(h);ax.set_xlabel('查询hop（每hop正负各32条）')
    ax.set_title(LABELS[case]+'\n虚线竖线右侧超出训练hop');ax.legend(fontsize=9)
fig.savefig(ROOT/'length_generalization.png',dpi=160);plt.close(fig)
print(context)
print(trajectory)
print(fixed)
print(bfs)

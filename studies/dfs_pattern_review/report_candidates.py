"""Report full candidate-field generation separately from whole DFS accuracy."""
from pathlib import Path
from collections import Counter
import json
HERE=Path(__file__).resolve().parent
runs=json.loads((HERE/'candidate_generation.json').read_text())
examples=json.loads((HERE/'candidate_token_examples.json').read_text())
lines=['# DFS 候选列表生成：Fixed 与 In-Context 的具体差别','',
       '候选构建、候选排序、PUSH选分支是三个不同的问题。本次只检查候选字段生成与历史读取；不将目标信息未被证明用于排序等同于没有执行DFS。','',
       '## 独立生成整段候选字段','',
       '从正确的当前stack和visited之后开始，不提供任何gold候选token，模型自由生成直到自己输出分隔符|。允许任何合法顺序；必须集合完全正确、无重复且正确结束。不是整条自由DFS合法率。','',
       '每模型沿用20个test查询，按fresh/repeat×empty/nonempty每query保留首个状态；剔除FOUND，候选起点≤4096，FP32，GPU0，冻结参数。','',
       '历史屏蔽：保留全部输入token、绝对位置、原始EDGE/QUERY/WORLD以及当前stack/visited；仅禁止当前记录的所有attention查询行在所有层读取更早CoT的key，包括生成候选时。当前stack/visited的隐藏状态也在此屏蔽条件下重新计算，避免它们先偷偷读旧历史再向后传递。','',
       '|模型|完整历史：整字段正确|屏蔽旧CoT：整字段正确|','|---|---:|---:|']
for name,r in runs.items():
    a=r['results']['full']['rows'];b=r['results']['no_old_cot']['rows']
    lines.append(f"|{name}|{sum(x['set_correct'] for x in a)}/{len(a)} ({100*sum(x['set_correct'] for x in a)/len(a):.1f}%)|{sum(x['set_correct'] for x in b)}/{len(b)} ({100*sum(x['set_correct'] for x in b)/len(b):.1f}%)|")
    assert all(c['argmax_drift']==0 and c['max_logit_error']==0 for c in r['identity_checks'])
lines += ['', '首次fresh_nonempty状态在该选择规则下通常就是根状态，之前没有CoT，因此屏蔽不改变它是设计上的identity对照，不能作为新结论。各状态并不独立；另保存了按query先平均的结果，表格采用直观的状态计数。','',
          '## 回溯后的非空候选','', '|模型|完整历史|屏蔽旧历史|','|---|---:|---:|']
for name,r in runs.items():
    a=[x for x in r['results']['full']['rows'] if x['bucket']=='repeat_nonempty']
    b=[x for x in r['results']['no_old_cot']['rows'] if x['bucket']=='repeat_nonempty']
    lines.append(f"|{name}|{sum(x['set_correct'] for x in a)}/{len(a)}|{sum(x['set_correct'] for x in b)}/{len(b)}|")
lines += ['', '两种任务都依赖历史，不能将Fixed描述为只给当前节点查静态后继表就完成候选输出。屏蔽是较强的分布变化，不能据此判定模型无法从显式状态计算。特别是困难Fixed屏蔽后的53个失败字段直接输出<PUSH>而不是候选；历史也参与字段/协议阶段控制，不能把下降全解释成图关系或visited排重计算失败。','',
          '## 合法候选顺序不唯一','', '|模型/query/记录|参考候选|实际从字段开头自行生成|','|---|---|---|']
for r in examples:
    for s in r['states']:
        lines.append(f"|{r['model']}/{r['query_id']}/R{s['record']}|{' '.join(s['candidates']) or '空'}|{' '.join(s['autoreg_generated']) or '空'} {'（已输出结束符）' if s['autoreg_closed'] else '（未结束）'}|")
lines += ['', '因此前面给定参考CAND=[V13,V15,V14]后PUSH参数输出V13，与模型自行生成CAND=[V14,V13,V15]并不矛盾：两者前缀条件不同。单例先输出目标V14也不能证明目标导向排序；它同时是原参考邻居表最后一项，需要专门的目标配对对照。','',
          '## 具体差异1：首次候选的知识来源','',
          'In-Context需要读取本例输入图中相关边的表示/移位位置；先前20-query分析在首次扩展某节点时，L5对destination+3位置的平均质量N16为80.1%、N100为74.2%。Fixed没有EDGE输入，其图关系来源于训练参数与已有上下文；首个候选预测中常强读当前节点所在的visited/stack位置，但尚未定位DFS自身的邻接记忆究竟在哪个FFN/embedding子空间。','',
          '## 具体差异2：回溯后如何列剩余候选','',
          'In-Context query50520：初始V11候选[V13,V15,V14]；给定已经探索V13并POP的历史，生成[V15,V14]。预测第一个V15时，L5强读旧PUSH及参数；续写V14时，L5强读旧参数V13@85（0.632），L6直接读旧候选V14@82（0.307）。结束列表时，该位置L6强读本轮已输出V15@101（0.772）。这是多个读取位置共同参与，不是每个token都沿单一固定通路。','',
          '困难Fixed query780：初始V70候选[V82,V72,V76,V80]；给定探索V82并POP的合法历史，再生成[V72,V76,V80]。此时L4依次读取旧候选列表中下一项的位置：','',
          '|输入位置|下一输出|L4读取旧位置|注意力|','|---|---|---|---:|',
          '|新列表前的分隔符@29|V72|旧V72@9|0.997|',
          '|新V72@30|V76|旧V76@10|0.813|',
          '|新V76@31|V80|旧V80@11|0.956|',
          '|新V80@32|分隔符|旧列表结束符@12|0.990|','',
          '该例符合“按历史候选列表的剩余部分续写”，不支持简单概括为每次都重新取Adj(u)再做完整集合差；但它也不能排除其他层同时进行集合/状态计算。','',
          '## 具体差异3：Fixed候选确实会随状态改变','',
          '简单Fixed query151中，当前节点V82、当前stack=[V65,V74,V82]和query保持相同：visited=[V65,V74,V82]时自行生成[V96]；探索V96并POP回来，visited增加V96时自行生成空列表。说明不是不看任何状态、永远输出V82的静态出邻居。两次的完整历史和位置也不同，因此尚不能将差异单独归因于显式visited token。','',
          '## 结论边界','',
          '区别更明确地表现为图知识获取接口、历史内容的承载位置、候选列表的续写和终止方式。两种任务都利用历史，且都能在这些给定正确状态下生成合法候选。原子操作正确不等于整条DFS正确，历史读取路线也不等于已证明的通用DFS算法。','',
          '无新训练、checkpoint、W&B、激活或完整attention矩阵。候选字段是短结果，逐状态生成与所有错误保留于candidate_generation.json；具体逐token路线保留于candidate_token_examples.json。']
(HERE/'CANDIDATE_GENERATION.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Wrote candidate-generation report')

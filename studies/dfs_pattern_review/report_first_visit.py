"""Observed before-first-error candidate hazards, not marginal causal rates."""
from pathlib import Path
import json
HERE=Path(__file__).resolve().parent
p=HERE/'candidate_first_visit.json';r=json.loads(p.read_text())
assert r['version']==2 and len(r['native_panels'])==5
# Earlier runner stored empty defaults when only error phase was retained.
# They are not observed empty candidate fields: remove those placeholders.
for v in r['native_panels'].values():
    for ex in v['examples']:
        ex.pop('expected',None);ex.pop('shown',None)
    v['examples_are_phase_only']=True
    for key,values in v['by_phase'].items():assert 0<=values['bad']<=values['n']
    assert sum(v['first_candidate_error_counts'].values())==sum(v['by_phase'][key]['bad'] for key in ('root','fresh_pushed','revisit'))
p.write_text(json.dumps(r,indent=2)+'\n')
def entry(a):return f"{a['bad']}/{a['n']} ({100*a['rate']:.2f}%)" if a['n'] else '—'
lines=['# 候选首错是否主要发生在回溯后','',
       '目前不能作此一般化。原Fixed全池有相当比例的错误发生于新入栈节点；新增In-Context面板反而显示首次处理新节点的候选错误率更高。不能将“节点首次处理”与“轨迹早期”混同，也不能将首个语义错误称为此前错误的累计传播。','',
       '## 定义与统计方法','',
       '- fresh：当前节点此前未作为栈顶处理过；root另列，fresh_pushed仅指PUSH后首次处理的新节点。revisit：回溯后再次作为当前栈顶。',
       '- 首次到新节点时也可能已有很长visited，并已有出邻居在别的分支访问过；DAG有汇合，所以fresh也需要全局排重。',
       '- 5个validation-best checkpoint，每模型128个分层test查询，原生BF16 greedy、原始上下文和STOP预算；仅GPU0冻结推理，无训练。',
       '- 只在当前frontier与visited已通过独立重放、此前没有语义错误时，进入候选字段就计一次机会。包含名单不匹配、重复、图外节点、非法token和未闭合字段。预算强制中止的未完字段作删失。',
       '- 若只等完整记录才计数，In-Context输出图外节点会在旧检查器中先标non_protocol_token，从而漏掉一批候选错误；本次按字段重新定位。',
       '- 报告的是“已经合法走到该状态”的条件错误率。首次错即停止，存在幸存者选择与其他错误的竞争风险，不是配对因果比较。每模型单次训练，状态来自相同查询，不独立。','',
       '## 原Fixed完整测试池：候选集合首错的发生阶段','',
       '|图|根节点第一条|PUSH后新节点首次|回溯重访|总候选集合首错|','|---|---:|---:|---:|---:|']
for s,a in r['original_fixed_full_pool'].items():lines.append(f"|{s}|{a['root']}|{a['fresh_pushed']}|{a['revisit']}|{a['candidate_errors']}|")
lines += ['', '这些是原全池精确分类，未给状态机会数作分母；不能据错误条数直接比较哪类风险更大。fresh判定使用DFS发现顺序：fresh时当前节点恰为内部visited发现序列的最后一项。该不变量在独立原生重放及含汇合的小图测试中通过验证。','',
          '## 新面板：应生成非空候选时的条件出错率','',
          '排除根节点，并仅比较应输出非空列表的状态，避免fresh和revisit的空列表比例不同。分母为候选字段尝试次数，含尚未闭合就发生的可确认错误。','',
          '|模型|PUSH后新节点首次|回溯后重访|','|---|---:|---:|']
for name,v in r['native_panels'].items():
    lines.append(f"|{name}|{entry(v['by_nonempty']['fresh_pushed_nonempty'])}|{entry(v['by_nonempty']['revisit_nonempty'])}|")
lines += ['', '普通In-Context N16/N100在这些合法前缀中没有观察到revisit候选首错；0次不是错误率被证明为0。困难In-Context也不是首次更安全。Fixed两类机会的差别很小且只有少量错误，不支持统一断言回溯更难。','',
          '## 全部候选机会（含空列表）','',
          '|模型|根节点|新PUSH节点|重访|','|---|---:|---:|---:|']
for name,v in r['native_panels'].items():lines.append(f"|{name}|{entry(v['by_phase']['root'])}|{entry(v['by_phase']['fresh_pushed'])}|{entry(v['by_phase']['revisit'])}|")
lines += ['', '## 两个真实反例','',
          '1. 困难Fixed query313，第2条刚到新节点V5；visited只有[V2,V5]。正确候选[V6,V7,V9,V11]，生成[V9,V9,V6,V11]，重复V9、漏V7。此时既是节点首次处理，也是搜索极早期，不能统一解释为长历史累积。',
          '2. 简单Fixed query159，第64条首次到V18；visited已有33个节点，其中V22已访问。真实后继[V21,V22,V25,V29]，应列[V21,V25,V29]，却列[V29,V22,V21]。说明首次处理的新节点也会面临旧分支造成的排重负担。','',
          '## 为什么不能直接叫累计误差','',
          '本次以首次语义错误为终点，候选首错前的显式记录都已通过图与状态重放。因此没有观察到“此前已错→后续传播”的链条。长历史、更多visited、未见状态组合、合法但不常见的搜索顺序、检索难度，均可能让后期更难，但它们不等于先前已有错误累计。',
          '即使发现错误率随步骤增长，也需控制图结构、候选数、visited大小和状态类型后，才能进一步讨论长度效应；本次按record分箱的原始机会数保留在JSON，没有拟合因果关系。','',
          '## 解释性假设','',
          'In-Context首次处理当前节点时要从输入图抽取新的后继集合；回溯后可利用先前生成的候选和操作表示。这个差别与新面板的错误分布相容，但不是这条机制已被证明导致错误的结论。Fixed也可能用旧列表、节点表示和其他历史路线，不能从“首次/重访”标签单独确定其内部算法。','',
          '额外的合成验证覆盖：首次/重访计数、首次到达也需排除旧visited后继、图外节点、未闭合候选、非法字段token、visited错误不进入候选分母、动作错误不抹去正确候选字段。见candidate_first_visit_verification.json。']
(HERE/'CANDIDATE_FIRST_VISIT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Wrote first-visit report for five 128-query panels')

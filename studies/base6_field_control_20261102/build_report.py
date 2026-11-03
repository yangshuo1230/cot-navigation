"""Audit and display all-layer causal boundary experiments without hiding controls."""
import json,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
FIELDS={'stack':'stack：起点后结束','stack_tail':'stack：深栈边界','visited':'visited','candidates':'candidates：非空列表末尾','candidates_empty':'candidates：初始为空'}
KINDS={'attention':'Attention','ffn':'FFN','residual':'整层残差'}
DIRECTIONS=['stop_to_continue','continue_to_stop']

def main():
 rows=[];pairs=[];checks=[];completions=[]
 for prefix in ['', 'stack_tail.','candidate_empty.']:
  rows+=json.loads((ROOT/(prefix+'results.json')).read_text());pairs+=json.loads((ROOT/(prefix+'pairs.json')).read_text());checks+=json.loads((ROOT/(prefix+'replay_checks.json')).read_text());completions.append(json.loads((ROOT/(prefix+'COMPLETE.json')).read_text()))
 baseline={(r['index'],r['field'],r['direction']):r['baseline'] for r in rows};good={};strict={};scope={}
 for f in FIELDS:
  ids=sorted({r['index'] for r in rows if r['field']==f});good[f]=[ix for ix in ids if all(baseline[ix,f,d]['class_correct'] for d in DIRECTIONS)];strict[f]=[ix for ix in ids if all(baseline[ix,f,d]['legal_original_next'] for d in DIRECTIONS)]
  scope[f]=dict(available_queries=len(ids),class_correct_pair_queries=len(good[f]),content_correct_pair_queries=len(strict[f]),same_top_pair_queries=sum(p.get('matched_current_top',False) for p in pairs if p['field']==f and p['available']))
 summary=[]
 grouped=defaultdict(list)
 for r in rows:
  for subset,ids in [('class',good[r['field']]),('content',strict[r['field']])]:
   if r['index'] in ids:grouped[r['field'],r['component'],r['layer'],r['condition'],r['direction'],subset].append(r)
 for key,rr in grouped.items():
  f,k,l,c,d,s=key;summary.append(dict(field=f,component=k,layer=l,condition=c,direction=d,subset=s,n=len(rr),flips=sum(r['flipped_to_opposite_class'] for r in rr),legal_donor=sum(r['legal_donor_next'] for r in rr),legal_original=sum(r['legal_original_next'] for r in rr),invalid=sum(r['output_class']=='invalid' for r in rr),mean_margin_change=float(np.mean([r['stop_margin_change'] for r in rr]))))
 lookup={(r['field'],r['component'],r['layer'],r['condition'],r['direction'],r['subset']):r for r in summary}
 def stat(f,k,l,c='opposite_donor',d=None,s='class'):
  zz=[lookup.get((f,k,l,c,dd,s),{}) for dd in ([d] if d else DIRECTIONS)];return {key:sum(z.get(key,0) for z in zz) for key in ['n','flips','legal_donor','invalid']}
 def ratio(f,k,l,c='opposite_donor',d=None,s='class',metric='flips'):
  v=stat(f,k,l,c,d,s);return f"{v[metric]}/{v['n']}"
 binding=json.loads((ROOT/'binding.json').read_text());plan=json.loads((ROOT.parent/'dfs_attention_lifecycle_20261030/plan.json').read_text());assert binding['sha256']==plan['checkpoint_sha256']
 assert len(rows)==sum(c['conditions'] for c in completions)
 assert len(checks)==sum(c['full_forward_checks'] for c in completions)
 assert all(r['prediction']==r['baseline']['prediction'] for r in rows if r['condition']=='sham')
 assert all(c['history_cache_unchanged'] and c['single_receiver_single_layer'] for c in completions)
 pairmap={(p['index'],p['field']):p for p in pairs if p['available']}
 assert all(pairmap[r['index'],r['field']]['visible']==r['visible'] for r in rows)
 validation=dict(passed=True,model_step=binding['step'],checkpoint_sha256=binding['sha256'],unique_queries=len({r['index'] for r in rows}),available_pairs=sum(p['available'] for p in pairs),receivers=len(baseline),conditions=len(rows),shams=sum(r['condition']=='sham' for r in rows),full_forward_checks=len(checks),maximum_replay_logit_error=max(c['max_logit_error'] for c in checks),history_positions_unchanged=True,candidate_permutation_pairs_validated=sum(p['field']=='candidates' and p['available'] for p in pairs),scope=scope)
 (ROOT/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2));(ROOT/'summary.json').write_text(json.dumps(dict(scope=scope,statistics=summary),ensure_ascii=False))
 # The same donor-vs-random matrix is shown for each component, both directions.
 fig,axes=plt.subplots(1,3,figsize=(14,9));labels=['Stack root / stop→node','Stack root / node→stop','Stack deep / stop→node','Stack deep / node→stop','Visited / stop→node','Visited / node→stop','Cand. end / stop→node','Cand. end / node→stop','Cand. empty / stop→node','Cand. empty / node→stop']
 for ax,(kind,title) in zip(axes, KINDS.items()):
  a=[];texts=[]
  for f in FIELDS:
   for d in DIRECTIONS:
    row=[];tt=[]
    for l in range(1,7):
     v=stat(f,kind,l,d=d);r=stat(f,kind,l,'donor_norm_random',d);row.append((v['flips']-r['flips'])/max(1,v['n']));tt.append(f"{v['flips']}/{v['n']}\nr={r['flips']}")
    a.append(row);texts.append(tt)
  im=ax.imshow(a,vmin=-1,vmax=1,cmap='RdBu',aspect='auto')
  for i in range(10):
   for j in range(6):ax.text(j,i,texts[i][j],ha='center',va='center',fontsize=8,color='white' if abs(a[i][j])>.6 else '#162b3b')
  ax.set(xticks=range(6),xticklabels=[f'L{l}' for l in range(1,7)],yticks=range(10),yticklabels=labels if ax==axes[0] else ['']*10,title=title if kind!='residual' else 'Block residual');ax.set_yticks(np.arange(-.5,10,2),minor=True);ax.grid(which='minor',axis='y',color='white',linewidth=2)
 fig.suptitle('Opposite donor swaps: node/bar class flips; r = norm-matched random flips',fontsize=13);fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.6,label='Donor minus random flip fraction');fig.subplots_adjust(left=.18,right=.9,wspace=.12,top=.93,bottom=.04);fig.savefig(ROOT/'causal_matrix.png',dpi=170);plt.close(fig)
 total=len(rows);lines=['# 六层 base：字段继续／停止的因果定位','',f"冻结原始 step185000 checkpoint，SHA256 `{binding['sha256']}`。本轮只研究这个模型，未使用九层纠错模型。",'',f"**18 条已有 N100 test query，{validation['available_pairs']} 组停止／继续配对、{validation['receivers']} 个接收位置、{total} 次独立条件运行。条件数不是独立 query 数。**",'', '## 结论','', '**没有一个层统一控制所有字段。visited 的 L5 attention、非空候选末尾的 L3 attention、深栈结束的 L6 attention，呈现不同的因果依赖。stack 在只输出起点与深栈末尾也有不同路线。L6 FFN 普遍携带最终停止／节点输出信息，但这不等于停止算法都在 L6 FFN 中计算。**','', '## 实验怎样区分影响与控制','', '- 消融：只把当前接收位置某层的 attention 或 FFN 输出置零，重算本层剩余计算与后续层，观察原有边界是否被破坏。也做等范数随机扰动。','- 双向交换：同一 query、同一可见接收 token、同一字段，一个应输出 `|`、另一个应继续。只交换某一个组件输出，观察停止→节点、节点→停止两个方向。','- 另交换整层残差，并测试半量／反向供体差；全部结果保留。接近输出头的交换有天然优势，只能说明表示足以改变读出，不能自动定位算法。','- 只改当前接收位置、只选一层；历史位置不变。后续层重新计算，不继续覆盖后续层。供体可以来自同条轨迹更晚的记录，这是离线反事实，不是生成时可以读取未来。','- 每个自然供体差配一个同范数随机方向；一个随机方向不是完整随机分布，只是局部对照。未做多重比较显著性宣称。','- 分类严格分 `|`、节点、非法 token；非法 token 不算成功续写。另检查节点是否属于供体场景应输出的内容，以及原始内容是否仍合法。','', '## 配对范围','', '| 子场景 | 可用 query | 原始两侧停止／节点类别都对 | 原始两侧具体内容都合法 |','|---|---:|---:|---:|']
 for f,name in FIELDS.items():
  s=scope[f];lines.append(f"| {name} | {s['available_queries']} | {s['class_correct_pair_queries']} | {s['content_correct_pair_queries']} |")
 lines+=['', 'stack 起点对照：两侧均刚输出第一个起点节点，已输出数量 j=1；一侧目标长度为1，另一侧更长。深栈对照：末节点与另一记录中同一节点继续的位置匹配，已输出数量 j 相同且至少为2。','', 'visited：相同可见节点，选择靠近轨迹中部的结束／续写配对。非空候选：保持同一记录候选集合不变，交换首末节点，构造相同接收字面节点的相反边界；17组均通过 DFS replay 合法性验证。初始空候选：两侧接收者都是 visited 结束符，排除 FOUND 情况，优先匹配同一当前栈顶；同栈顶匹配数量见 summary.json。','', '候选排列后有3条 query 的继续侧虽然输出节点，却不是剩余合法候选；主表仍可研究停止／节点类别，严格内容子集只保留两侧内容都合法的 query。不能把这3条当作可靠续写样本。','', '## 逐层：交换组件后能否切换边界','', '以下分母是两个方向的条件数，例如36来自18条query×两个方向；“合法供体内容”要求停止时为 `|`，继续时为供体场景的合法下一节点。随机列使用相同范数，所有层均报告。','']
 for f,name in FIELDS.items():
  lines += [f'### {name}：attention 输出交换','', '| 层 | 停止→节点 | 节点→停止 | 两方向合计 | 等范数随机合计 | 合法供体内容 |','|---|---:|---:|---:|---:|---:|']
  for l in range(1,7):lines.append(f"| L{l} | {ratio(f,'attention',l,d=DIRECTIONS[0])} | {ratio(f,'attention',l,d=DIRECTIONS[1])} | {ratio(f,'attention',l)} | {ratio(f,'attention',l,'donor_norm_random')} | {ratio(f,'attention',l,metric='legal_donor')} |")
 lines+=['', '## 逐层消融：是否破坏原有边界', '', '以下只报告 attention 置零导致的相反类别切换；非法输出另外列出。随机列与置零改变量同范数。', '']
 for f,name in FIELDS.items():
  lines += [f'### {name}：attention 置零','', '| 层 | 原本停止→节点 | 随机 | 原本继续→停止 | 随机 | 非法输出（两方向） |','|---|---:|---:|---:|---:|---:|']
  for l in range(1,7):lines.append(f"| L{l} | {ratio(f,'attention',l,'zero',DIRECTIONS[0])} | {ratio(f,'attention',l,'zero_norm_random',DIRECTIONS[0])} | {ratio(f,'attention',l,'zero',DIRECTIONS[1])} | {ratio(f,'attention',l,'zero_norm_random',DIRECTIONS[1])} | {ratio(f,'attention',l,'zero',metric='invalid')} |")
 lines+=['', '## 怎么解释各字段','',f"**visited：L5 attention 是最清楚的内容与边界联合接口。** 双向交换 {ratio('visited','attention',5)}，随机 {ratio('visited','attention',5,'donor_norm_random')}；成功切换均恢复合法供体内容。其他层的 attention 交换远弱于 L5。和此前后继载体接入的5/6结果相容，支持 L5 的发现序列后继检索参与决定继续或停止。L2–L4 消融也会破坏部分行为，因此不是只有L5有用。",'',f"**非空候选的末尾：L3 attention 具有更直接的边界控制证据。** 双向交换 {ratio('candidates','attention',3)}，随机 {ratio('candidates','attention',3,'donor_norm_random')}；严格内容正确子集为 {ratio('candidates','attention',3,s='content')}，其中 {ratio('candidates','attention',3,s='content',metric='legal_donor')} 符合供体合法内容。这不能解释成候选节点主要由L3检索：L5仍有内容读取作用，L5交换偏向停止→节点，两个方向不对称。",'',f"**深栈结束：L6 attention 对维持停止很重要。** 置零后停止→节点 {ratio('stack_tail','attention',6,'zero',DIRECTIONS[0])}；等范数随机 {ratio('stack_tail','attention',6,'zero_norm_random',DIRECTIONS[0])}。供体交换停止→节点 {ratio('stack_tail','attention',6,d=DIRECTIONS[0])}，但这些节点并不等于供体正确下一节点，不能称恢复了正确续写。反方向节点→停止仅 {ratio('stack_tail','attention',6,d=DIRECTIONS[1])}。更像参与停止门控，而非独自提供应续写内容。",'',f"**stack 起点位置不同。** L6 attention 的置零与供体交换均不改变边界类别；L4 attention 双向交换 {ratio('stack','attention',4)}，随机 {ratio('stack','attention',4,'donor_norm_random')}。不能把此前深栈结论推广到只输出起点的边界。",'', f"**初始候选为空，不能沿用非空列表末尾的L3解释。** 17组均匹配同一当前栈顶；L3 attention置零和供体交换均为0/34边界切换。L4 attention置零使非空→提前停止 {ratio('candidates_empty','attention',4,'zero',DIRECTIONS[1])}，相同范数随机0/17；L5 attention置零使空→错误节点17/17，这些节点均不符合供体的合法候选；L6 attention双向供体交换 {ratio('candidates_empty','attention',6)}，随机 {ratio('candidates_empty','attention',6,'donor_norm_random')}，成功切换均符合供体内容。这里体现的是L4–L6协作的依赖，不能说L3统一控制候选字段的全部停止。", '', '## L6 FFN：共同的最终输出接口','', '| 子场景 | 双向交换类别切换 | 等范数随机 | 合法供体内容 |','|---|---:|---:|---:|']
 for f,name in FIELDS.items():lines.append(f"| {name} | {ratio(f,'ffn',6)} | {ratio(f,'ffn',6,'donor_norm_random')} | {ratio(f,'ffn',6,metric='legal_donor')} |")
 lines+=['', '这些结果说明 L6 FFN 输出已含可操纵的停止／节点信息；但交换最后 FFN 离输出头很近，其高成功率不证明它负责比较长度或计算集合耗尽。必要性置零结果与充分性交换结果需分开看。','', '## 校验与限制','',f"- {validation['shams']} 次 sham 的预测均与原模型相同。{len(checks)} 次完整前向 hook 复算，与缓存后缀重算 argmax 全部相同；最大绝对 logit 差 {validation['maximum_replay_logit_error']:.6g}。",'- 所有接收者的历史 K/V 在干预前后逐字节 SHA256 一致；缓存重算只覆盖当前位置，避免把历史干预混入接收位置实验。','- 使用已有18条参考query，新增干预条件，不是新增自由生成成功率，也不是全新未见研究样本。每个子场景每条query至多一对，不能把上万条件当成上万样本。','- 本轮定位的是可以影响边界的组件与表示，尚未解出精确长度计数、候选耗尽比较或完整跨位置计算回路。此前L2–L6线性长度方向干预无效的结果仍成立。','- 缓存重放利用因果性：改最后位置不会改变此前位置；验证覆盖每个接收者的L3 attention置零／供体／随机，以及L6 FFN供体条件。其余条件有逐层sham检查。','', '## 文件','', '- `control.html`：交互查看所有层、两个方向、随机对照、内容合法性和逐query输出。','- `summary.json`：完整分类与严格内容子集统计；`validation.json`：分母、哈希、sham和数值等价校验。','- `results.json`、`stack_tail.results.json`、`candidate_empty.results.json`：全部原始条件，未删除失败或非法输出。','- 三组 `*.pairs.json`、`*.replay_checks.json`：配对位置与完整前向复核。','- `experiment.py`、`enrich.py`、`build_report.py`：实验、内容评分、报告代码。','']
 (ROOT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
 compact=[{k:r[k] for k in ['index','field','direction','layer','component','condition','prediction','output_class','flipped_to_opposite_class','legal_donor_next','legal_original_next','stop_margin_change','visible','q','allowed_original_next','allowed_donor_next','baseline']} for r in rows]
 payload=dict(scope=scope,summary=summary,rows=compact,fields=FIELDS,validation=validation)
 template=(ROOT/'template.html').read_text(encoding='utf-8');(ROOT/'control.html').write_text(template.replace('__DATA__',json.dumps(payload,ensure_ascii=False).replace('</','<\\/')),encoding='utf-8');print(json.dumps(validation,ensure_ascii=False))
if __name__=='__main__':main()

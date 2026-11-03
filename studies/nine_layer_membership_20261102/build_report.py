"""Query-balanced summaries and a compact, auditable research report."""
import sys,json,html,re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from probe import evaluations

def mean_query(rows,fn):
 groups={}
 for r in rows:
  v=fn(r)
  if v is not None:groups.setdefault(r['index'],[]).append(float(v))
 values=np.array([np.mean(v) for v in groups.values()]);rng=np.random.default_rng(20261102)
 ci=np.quantile(rng.choice(values,(4000,len(values)),replace=True).mean(1),[.025,.975]) if len(values) else [None,None]
 return dict(mean=None if not len(values) else float(values.mean()),ci95=list(ci),queries=len(values),events=sum(map(len,groups.values())))

def main():
 rows=json.loads((ROOT/'interventions.json').read_text());route=json.loads((ROOT/'route_aligned.interventions.json').read_text());rows += [dict(r,condition='route_'+r['condition']) for r in route];probes=json.loads((ROOT/'probe_results.json').read_text());validation=json.loads((ROOT/'validation.json').read_text());summary={}
 for model in ['corr9_s0','corr9_s1']:
  edge=json.loads((ROOT/(model+'.edge_reads.json')).read_text());val=[r for r in edge if r['split']=='val'];test=[r for r in edge if r['split']=='test'];layer=max(range(9),key=lambda l:mean_query(val,lambda r:r['attention']['relevant'][l])['mean']);edge_summary={k:mean_query(test,lambda r:r['attention'][k][layer]) for k in ['relevant','position_matched_wrong','same_destination_true','same_destination_wrong']}
  for k,n in [('eligible','n_eligible'),('excluded','n_excluded')]:edge_summary[k+'_per_edge']=mean_query(test,lambda r:r['attention'][k][layer]/r[n] if r['n_eligible'] and r['n_excluded'] else None)
  edge_summary.update(layer=layer+1,legal_prediction=mean_query(test,lambda r:r['predicted_is_legal_remaining']))
  rr=[r for r in rows if r['model']==model];baseline={r['index']:r for r in rr if r['condition']=='baseline'};interventions={}
  for condition in dict.fromkeys(r['condition'] for r in rr):
   allrows=[r for r in rr if r['condition']==condition];clean=[r for r in allrows if baseline[r['index']]['original_correct']]
   def stats(z):
    o=dict(n=len(z),expected_correct=sum(r['expected_correct'] for r in z),original_correct=sum(r['original_correct'] for r in z),excluded_returned=sum(r['excluded_returned'] for r in z),changed=sum(r['output']!=baseline[r['index']]['output'] for r in z),STOP=sum(r['STOP'] for r in z))
    if z and 'membership_shift' in z[0]:
     effective=[r for r in z if abs(r['membership_shift'])>1e-6];o.update(effective_membership_interventions=len(effective),membership_bit_flipped=sum((r['probe_membership']>=r['probe_threshold'])!=(r['probe_membership']+r['membership_shift']>=r['probe_threshold']) for r in effective),max_count_shift=max(abs(r['count_shift']) for r in z),max_identity_margin_shift=max(abs(r['identity_margin_shift']) for r in z),identity_argmax_changed=sum(r['probe_identity_correct']!=r['identity_after_correct'] for r in z))
     for key in ['membership_shift','relative_norm','first_query_kl','first_query_logit_max_change','first_query_excluded_probability_change']:
      if key in z[0]:o[key]=dict(mean=float(np.mean([r[key] for r in z])),max_abs=float(max(abs(r[key]) for r in z)))
    return o
   interventions[condition]=dict(all=stats(allrows),baseline_correct=stats(clean))
  selection=probes[model]['selected'];site=0 if selection['site']=='action' else 1;meta=json.loads((ROOT/(model+'.push_metadata.json')).read_text());features=np.load(ROOT/(model+'.push.npz'))['features'];B=np.load(ROOT/(model+'.probe_weights.npz'))[selection['site']+'_L'+str(selection['layer'])];mask=np.array([r['split']=='test' for r in meta]);mm=[r for r in meta if r['split']=='test'];scores=np.c_[features[mask,selection['layer'],site],np.ones(mask.sum())]@B
  perquery=[]
  for ix in sorted(set(r['index'] for r in mm)):
   mi=[i for i,r in enumerate(mm) if r['index']==ix]
   if not any(set(mm[i]['visited'])-set([mm[i]['source'],mm[i]['pushed']]) for i in mi):continue
   m=evaluations(scores[mi], [mm[i] for i in mi]);
   if m['n_query']:perquery.append(dict(index=ix,value=m['auc']))
  selected_test=next(r['test'] for r in probes[model]['curves'] if r['site']==selection['site'] and r['layer']==selection['layer']);summary[model]=dict(edge=edge_summary,selected_probe=selection,probe_test=selected_test,probe_auc_uncertainty=mean_query(perquery,lambda r:r['value']),interventions=interventions,baseline_correct=sum(r['original_correct'] for r in baseline.values()),states=len(baseline))
 (ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
 fig,axes=plt.subplots(2,2,figsize=(13,9));colors=['#2274a5','#d77a24']
 for i,(model,s) in enumerate(summary.items()):
  e=s['edge'];bars=axes[0,0].bar(np.array([0,1])+i*.32,[e['relevant']['mean'],e['position_matched_wrong']['mean']],width=.3,label=model,color=colors[i]);axes[0,0].bar_label(bars,labels=[f"{e[k]['mean']:.2%}" for k in ['relevant','position_matched_wrong']],padding=5);sel=s['selected_probe'];curves=[r for r in probes[model]['curves'] if r['site']==sel['site']];axes[0,1].plot([r['layer'] for r in curves],[r['test']['auc'] for r in curves],'-o',label=model+' '+sel['site'],color=colors[i]);axes[0,1].scatter([sel['layer']],[s['probe_test']['auc']],s=180,facecolors='none',edgecolors=colors[i])
  for j,key in enumerate(['visited_remove_excluded','edge_current_source','push_remove_norm_0.1','push_remove_norm_0.25']):
   v=s['interventions'][key]['baseline_correct'];axes[1,0].bar(j+i*.32,v['expected_correct']/v['n'],width=.3,color=colors[i]);axes[1,0].text(j+i*.32,v['expected_correct']/v['n']+.025,f"{v['expected_correct']}/{v['n']}",ha='center',fontsize=9)
  axes[1,1].bar(np.array([0,1])+i*.32,[e['eligible_per_edge']['mean'],e['excluded_per_edge']['mean']],width=.3,label=model,color=colors[i])
 axes[0,0].set(xticks=[.16,1.16],xticklabels=['Current outgoing','Matched wrong source'],ylabel='Attention mass',title='Edge selection (validation-selected layer)',ylim=(0,1));axes[0,0].legend();axes[0,1].axhline(.5,color='gray',ls='--');axes[0,1].set(xlabel='Residual stage',ylabel='Visited membership AUROC',title='PUSH membership probe; circle = dev-selected',ylim=(.45,.65));axes[0,1].legend()
 axes[1,0].set(xticks=np.arange(4)+.16,xticklabels=['Visited remove','Current edge edit','PUSH 10% norm','PUSH 25% norm'],ylim=(0,1.15),ylabel='Counterfactual set success',title='Main experiment: dev-selected PUSH layers');axes[1,1].set(xticks=[.16,1.16],xticklabels=['Unvisited destination','Visited destination'],ylabel='Attention per edge',title='Mixed outgoing sets: both are still read');axes[1,1].legend()
 for ax in axes.ravel():ax.spines[['top','right']].set_visible(False)
 fig.tight_layout();fig.savefig(ROOT/'evidence.png',dpi=170);plt.close(fig)
 lines=['# 九层纠错模型：新节点首次展开时的边读取与 visited 过滤','', '实验日期：2026-11-02。范围：`l9_p30/s0` 与 `l9_p30/s1` 的 best checkpoint；不能外推至所有 DFS 模型。checkpoint SHA256 已与上次分析绑定核对。','', '**结论：读取当前节点的具体出边有强证据；当前显式 visited 的作用有 seed 差异；PUSH 上只有较弱的 visited 线性可解码性，干预未获得稳定且特异的过滤因果效果。尚不能证明 PUSH 保存了完整 visited 并据此执行过滤。**','', '![实验汇总](evidence.png)','', '## 统计口径与实验隔离','', '- 36 条原始 query：18 val、18 test，沿用前轮研究的 query 面板，不是全新独立样本。每个模型提取 775 个 PUSH 状态，两个位置 × L0–L9；两个模型使用相同前缀。多个 token/状态来自同一 query，不作为独立 query。','- 边读取测试有 319 个 token 预测事件，来自 17 条有适用事件的 test query；每条 query 内先平均，再跨 query 平均。逐边数值为实际 attention，未减距离基线。','- 过滤干预从 18 条 test query 中固定选出 13 条“首次展开，后继中包含非源节点的已访问节点”的 query，每条一个状态，每个模型均为 13 状态。原始候选列表集合正确均为 10/13。主因果分母使用这 10 个原本正确的状态。','- 两模型合计 26 个模型／状态组合、720 次条件运行（主实验 404，路线对齐补充 316），不是 720 条独立样本。额外强剂量是在初轮未见输出变化后增加的探索性压力测试。','- **visited-only：只修改当前 visited 的 token；PUSH 残差不加 hook。PUSH-only：输入 token 与原始前缀 SHA256 完全相同，只改选定一个历史 PUSH 位置的一个层输出；每次条件结束卸载 hook。两个干预从不叠加。**','- 单层干预后，后续层、后续 token 的表示会自然重新计算；可能影响当前 visited 的隐藏表示。这属于 PUSH 干预的下游传播，不能解释为同时主动修改 visited。','- 当前 visited 的孤立编辑与此前历史不一致，是人为冲突实验；阴性结果不能证明正常运行时 visited 完全没用。','', '## 1. 边列表读的是相应边吗？','', '| 模型／读边层 | 当前节点全部出边质量 | 位置匹配无关边质量 | 同目标真边／错来源边：每边均值 |','|---|---:|---:|---:|']
 for model,s in summary.items():
  e=s['edge'];lines.append(f"| {model} L{e['layer']} | {e['relevant']['mean']:.2%} | {e['position_matched_wrong']['mean']:.2%} | {e['same_destination_true']['mean']:.2%} / {e['same_destination_wrong']['mean']:.2%} |")
 lines+=['', '读边层由 val 上“当前出边 attention 总量”最大确定。位置匹配控制与真出边数量相同，并匹配目标是否已访问；同目标控制保持目标节点相同，只换来源节点，其覆盖为 17 query、290 事件。不是只比较所有图 token 与所有历史 token。','', '这里 attention 落在 destination 后一位 carrier；先前独立分析已显示其早层读取前一位 destination。下列输入编辑检验的是原始边的 destination token，而非只按 carrier 标签推断。','', '| 操作：原始集合正确的状态 | s0 | s1 |','|---|---:|---:|']
 def frac(model,key,field='expected_correct'):
  v=summary[model]['interventions'][key]['baseline_correct'];return f"{v[field]}/{v['n']}"
 for label,key in [('当前节点的边 u→v（v 已访问）改成 u→w（w 未访问），完整生成新候选集合','edge_current_source'),('同样把 v 改成 w，但改另一来源节点的边；保持原候选集合','edge_wrong_source_same_destination')]:lines.append(f"| {label} | {frac('corr9_s0',key)} | {frac('corr9_s1',key)} |")
 lines+=['', '42 次边编辑全部保持 DAG，且此前完成的 DFS records 经独立 replay 验证合法。错来源控制只选择此前未展开的来源，不破坏此前搜索；该控制只有 8/13 状态可构造，两模型原始正确子集分别为 7、6，因此分母不同。s1 有一个错来源编辑也引起输出变化，说明存在少量非局部敏感性，不能声称控制绝对无效。','', '同时存在可保留与应排除后继的 13 query、57 事件中，单条边的 attention 均值：']
 for model,s in summary.items():lines.append(f"- {model}：未访问目标 {s['edge']['eligible_per_edge']['mean']:.2%}，已访问目标 {s['edge']['excluded_per_edge']['mean']:.2%}。")
 lines+=['', '**解释：这是按当前来源节点检索出边的清晰 pattern，但不是把 visited 目标在 attention 权重上彻底屏蔽。过滤还可能在 value、残差或后续层发生。**','', '## 2. 只改当前显式 visited 会怎样？','', '| 操作／按反事实完整集合正确计数 | s0 | s1 |','|---|---:|---:|',f"| 删除一个应被排除的 v：用非后继未访问 w 替换，保持字段长度；期待 v 重新出现 | {frac('corr9_s0','visited_remove_excluded')} | {frac('corr9_s1','visited_remove_excluded')} |",f"| 加入一个原本可用后继：替换另一非后继 visited 节点，保持字段长度；期待排除它 | {frac('corr9_s0','visited_add_eligible')} | {frac('corr9_s1','visited_add_eligible')} |",'', 's0 删除实验 10/10 输出完全不变；s1 2/10 按预期返回 v，其他 8/10 不变。加入实验两个模型在可构造且原始正确的 7 个状态均不变。不能简单总结为“visited 完全不起作用”；更准确是 s0 在本组冲突实验中依赖其他历史线索，s1 在部分状态使用当前 visited 信息。','', '## 3. PUSH 中能解码 visited 吗？','', '目标是该次 PUSH **之前**的 visited membership，不包括刚推入节点。评估在该图真实节点内进行，排除始终已访问的 query source 和刚 PUSH 的节点，避免只识别当前节点就拿高分。岭回归输出 256 维 membership 位图；按整条 query 划分 train/dev，alpha 和层／位置只用 dev 选择，之后在全部 18 val 重拟合，在 18 test 评估。775 PUSH 中 test 有 388 状态；适用 AUROC 的是 17 query、370 状态。','', '| 模型／dev 选择的位置层 | 测试 AUROC | 95% query bootstrap 区间 | 后继子集 AUROC | size/位置/身份控制 | 打乱 membership 标签控制 |','|---|---:|---|---:|---:|---:|']
 for model,s in summary.items():
  ci=s['probe_auc_uncertainty']['ci95'];p=probes[model];lines.append(f"| {model} {s['selected_probe']['site']} L{s['selected_probe']['layer']} | {s['probe_test']['auc']:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] | {s['probe_test']['neighbor_auc']:.3f} | {p['control_size_position_identity']['auc']:.3f} | {p['shuffled_membership_labels']['auc']:.3f} |")
 lines+=['', '后继子集只使用同时含 visited/unvisited 后继的 PUSH 状态：13 query、50 状态。size/位置/身份控制使用 L0 的 PUSH 参数表示、oracle visited 数量、相对 token 位置。打乱标签控制在每个状态图内随机分配相同数量的 visited。AUROC 0.5 为随机排序；这里是微弱排序信号，不是完整 visited 集合精确恢复。bootstrap 区间反映 query 采样波动，不含 checkpoint、层选择和重复研究不确定性。','', 'L0 的 action 也有约 0.54 AUROC，提示位置等混杂因素；18 条训练 query 中某些节点从未成为 visited 正例，位图 probe 对这部分节点没有有效方向。当前线性位图设计及样本量不足以排除更复杂或分布式 visited 编码。不能把测试曲线中事后最高层当成新的确证结果。','', '## 4. 在 dev 最可解码层干预 PUSH 的 visited 方向','', '只用 dev 选出的单位置、单层。沿目标 v 的 membership 读出方向移动，同时用线性约束保持 visited-count 读出，以及当前推入节点与原最高竞争节点的身份 margin。后者不保证全部身份 logits 不变。保留 L9 probe 供观察，但不在历史 token 的 L9 输出做此干预：最后一层残差不再被后续层读取，无法这样影响后面的 token。','', '- 初轮：membership score 目标移动 ±0.25、±0.5，残差改变量上限为原表示范数的 25%；配等范数、与上述读出方向正交的随机控制。','- 13 个状态中只有 9 个目标节点有非零可操控方向；另 4 个必须视为“干预未实现”，不能计为有效阴性。两模型这 9 个中原始集合正确均为 7 个。','- 补充：有效方向上按原残差范数的 10%、25% 改动，配同等强度反方向和随机控制。这是较强的离分布压力测试，仍不能代表天然编码。','- 干预固定过去的 PUSH 位置；每次生成新候选 token 的完整前向传播都会应用相同 hook，不混用任何 edited visited 输入。','', '| 原始正确且可有效改变 membership 的状态 | s0 | s1 |','|---|---:|---:|',f"| 降低 membership，25% 范数强剂量：v 回归且完整反事实集合正确 | {frac('corr9_s0','push_remove_norm_0.25')} | {frac('corr9_s1','push_remove_norm_0.25')} |",f"| 同一强剂量导致候选序列发生任何变化 | {frac('corr9_s0','push_remove_norm_0.25','changed')} | {frac('corr9_s1','push_remove_norm_0.25','changed')} |",f"| 等范数随机控制导致候选序列变化 | {frac('corr9_s0','push_random_norm_0.25','changed')} | {frac('corr9_s1','push_random_norm_0.25','changed')} |",'', '**此处仅指 dev 最可解码层：初轮及强剂量的加减方向均未改变候选序列。** 下文另做实际读取路线对齐的补充检验；这不是“已经证明 PUSH 不含 visited”，也不是“证明任何 PUSH 内容均无作用”。PUSH 还可能提供当前节点身份、字段路由或分布式历史摘要，真正有用的成分未必与本 probe 方向一致。','', '首个候选预测的概率变化也单独记录，避免把 argmax 不变误当成整个分布不变：']
 for model,s in summary.items():
  v=s['interventions']['push_remove_norm_0.25']['baseline_correct'];lines.append(f"- {model} 强剂量降低方向：首预测 KL 均值 {v['first_query_kl']['mean']:.3g}，目标 v 概率变化均值 {v['first_query_excluded_probability_change']['mean']:.3g}，最大绝对 logit 改变 {v['first_query_logit_max_change']['max_abs']:.3g}。")
 lines+=['', '## 5. 补充：与实际 attention 读取路线对齐', '', '最容易解码的层不一定被候选 query 读取。依据前轮相同模型 attention 路线，s0 在 L9 读取此前 PUSH 参数，故修改参数位置的 L8 输出；s1 在 L6 读取此前 PUSH 参数，故修改 L5 输出。两者仍只修改一个层的一个历史位置，输入保持原始 visited；完整 probe 权重均已在同一 val 数据拟合。', '', '| 路线对齐，25% 范数；原始正确且有效方向 | s0：argument L8 | s1：argument L5 |','|---|---:|---:|']
 for label,key,field in [('降低 membership：v 回归且完整新集合正确','route_push_remove_norm_0.25','expected_correct'),('降低 membership：输出序列有任何变化','route_push_remove_norm_0.25','changed'),('等范数随机方向：v 回归','route_push_random_norm_0.25','excluded_returned'),('等范数随机方向：输出序列有任何变化','route_push_random_norm_0.25','changed')]:
  lines.append(f"| {label} | {frac('corr9_s0',key,field)} | {frac('corr9_s1',key,field)} |")
 lines+=['', 's0 虽有 1/7 按目标返回 v，但等范数随机方向也在该状态返回 v，并在另外两个状态返回 v（3/7）；不能据此确认 membership 方向有特异因果作用。s1 定向改变仍为 0/7，随机扰动有 1/7 改变其他输出。初轮 score 位移及 10% 范数补充结果见 summary.json。', '', '**因此结论是“尚无稳定且特异的 PUSH membership → 过滤因果证据”，而非“所有 PUSH 干预都无效”。路线对齐能显示 s0 的该历史表示确实有输出敏感性，但尚未把敏感成分归因给 visited。**']
 lines+=['', '## 6. 一个可直接核对的长轨迹状态', '', 'query 50638：刚 PUSH 的节点是 `<V205>`，它的唯一后继是此前已访问的 `<V46>`，因此原始候选列表为空。该 PUSH 位于第 32 条 record；下表观察紧接着的首次展开，两个 seed 使用相同输入。', '', '| 独立条件 | s0 候选输出 | s1 候选输出 |','|---|---|---|','| 原始 | 空 | 空 |','| 只在当前 visited 中把 V46 换成未访问的非后继 V228 | 空 | V46 |','| 只把边 V205→V46 改成 V205→V228 | V228 | V228 |','| 路线对齐：降低 PUSH 中 V46 membership，25% 范数 | 空 | 空 |','| 路线对齐：等范数随机 PUSH 扰动 | V46 | 空 |', '', '这个例子同时展示了：两个 seed 都使用具体出边；当前显式 visited 的影响有 seed 差异；随机扰动也能让被排除节点回归，因而仅观察“干预后出现 V46”还不足以确认 visited 的特异编码。该例用于解释现象，整体结论仍以上述全部状态计数为准。']
 lines+=['', '## 7. 当前能支持的信息流图景','', '1. 当前新节点身份 → 找到它的对应出边：有 attention 选择性和逐边输入编辑两类证据。','2. 读取出边时，已访问目标仍会被读到；“读到哪些边”与“最终输出哪些候选”应分开理解。','3. 显式 visited 与历史轨迹的使用存在 seed 差异，不能用原始六层模型或另一个九层 seed 的结果替代。','4. PUSH 上弱 visited 解码信号尚无干预支持；不能画成已经证实的“PUSH visited bitmap → 过滤器”因果箭头。','5. 仍待确认：较完整的合法历史反事实、历史 selected-candidate / PUSH 参数的九层模型独立干预、逐头／value 子空间路径、更多 query 的目标条件 probe。以上均未在本轮得到证明。','', '## 文件与复核','', '- `interventions.json`：404 次主条件运行；`route_aligned.interventions.json`：316 次路线对齐补充运行，均含，原始／干预输入 SHA256、修改位置、逐 token 输出、反事实集合、probe 位移、范数及首预测分布变化。','- `summary.json`：全状态与原始正确子集分开汇总。`validation.json`：隔离断言、sham 一致性、42 次图编辑 DAG 与此前 DFS 合法性检查。','- `probe_results.json` / `selected_probe.json`：完整逐层结果与 dev 选择；两个 `*.probe_weights.npz`：可复核线性读出。','- `*.edge_reads.json`：逐事件、逐层三条最大权重边的来源／目标身份与控制统计。','- `models.json`：本轮 checkpoint 绑定，本地与远端均保存。','- 代码：`common.py`、`collect.py`、`probe.py`、`ridge.py`、`intervene.py`、`verify.py`、`build_report.py`。实验目录同步至远端项目的 `studies/nine_layer_membership_20261102/`。','']
 report='\n'.join(lines);(ROOT/'REPORT.md').write_text(report,encoding='utf-8')
 # Self-contained text, with the standalone figure referenced locally over HTTP/file.
 body=[];inlist=False;intable=False
 def inline(text):
  text=html.escape(text);text=re.sub(r'\*\*(.*?)\*\*',r'<strong>\1</strong>',text);return re.sub(r'`(.*?)`',r'<code>\1</code>',text)
 for line in lines:
  if line.startswith('|'):
   if not intable:body.append('<table>');intable=True
   if set(line.replace('|','').replace('-','').replace(':','').strip())==set():continue
   body.append('<tr>'+''.join('<td>'+inline(c.strip())+'</td>' for c in line.strip('|').split('|'))+'</tr>');continue
  if intable:body.append('</table>');intable=False
  if line.startswith('!['):body.append('<img src="evidence.png" alt="Evidence summary">');continue
  if line.startswith('# '):body.append('<h1>'+html.escape(line[2:])+'</h1>')
  elif line.startswith('## '):body.append('<h2>'+html.escape(line[3:])+'</h2>')
  elif line:body.append('<p>'+inline(line)+'</p>')
 if intable:body.append('</table>')
 (ROOT/'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>九层 DFS：边读取与 visited 因果检验</title><style>body{max-width:1150px;margin:35px auto;padding:0 22px;font:16px/1.8 system-ui;background:#f5f7fa;color:#192b3d}h1{font-size:28px}h2{margin-top:42px}table{border-collapse:collapse;background:white;width:100%;font-size:14px}td{border:1px solid #c9d3dd;padding:10px}tr:first-child{background:#e5edf6;font-weight:600}img{width:100%;background:white}p{overflow-wrap:anywhere}</style>'+''.join(body)+'</html>',encoding='utf-8')
 print(json.dumps(dict(models=2,conditions=len(rows),graph_failures=validation['graph_failures'],files=['REPORT.md','report.html','summary.json','evidence.png'])))
if __name__=='__main__':main()

"""Produce one key figure and a Chinese report; retain compact aggregates."""
import json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
CASES=['n16','n24','n32','n36','main']

def sanitize(x):
 if isinstance(x,dict):return {k:sanitize(v) for k,v in x.items()}
 if isinstance(x,list):return [sanitize(v) for v in x]
 if isinstance(x,float) and not math.isfinite(x):return None
 return x

def main():
 settings=json.loads((P/'settings.json').read_text());roles=settings['roles'];features=settings['features']
 data={c:json.loads((P/(c+'.json')).read_text()) for c in CASES}
 channel={c:json.loads((P/('channels_'+c+'.json')).read_text()) for c in ['n16','n32','main']}
 mass={c:json.loads((P/('mass_'+c+'.json')).read_text()) for c in ['n16','n32','main']}
 indirect={c:json.loads((P/('indirect_'+c+'.json')).read_text()) for c in ['n16','n32','main']}
 def att(case,arm,layer,feature,role='candidate_nonempty'):
  return data[case]['attention'][arm][layer-1][roles.index(role)][features.index(feature)]
 def pct(v):return f'{100*v:.2f}%'
 lines=['# 错误数据训练的读图注意力与收益来源','',
 '## 核心发现','',
 '新模型在候选预测位置形成了集中于第5层的直接出边读取通路：更多 attention 落在当前节点的出边终点，且大部分落在尚未 visited 的合法候选。距离校正后选择性仍然明显；局部遮挡、总量匹配和保总量打乱实验均表明，这条通路在新模型中具有实际功能。','',
 '但这不是“全模型 attention 越多越好”的一般结论。干净模型更依赖先前当前节点表示中的读图过程，低直接注意力并不意味着没有使用图。当前比较也未独立排除不同词表/初始化、难例重采样等训练差异。','',
 '## 对照与方法','',
 '- 干净 DFS λ16 与 30% 自然错误 DFS λ16；均为 last 200k、6层、每层1个头、128维、RoPE。',
 '- 五种图：N16、N24、N32、N36、N100。每种256个固定测试查询，最多4个相同的正确搜索状态，reference variant=0；排除 FOUND 状态的候选探测。',
 '- 两模型接收完全相同的图、节点命名、边顺序、CoT前缀与绝对位置。预测候选使用候选尚未生成时的前一位置 activation。',
 '- 从标准 SDPA 前向的实际 Q/K 投影提取选定行，按原 RoPE 变换计算 FP32 softmax；提取 hook 不修改前向结果，每个 reader 的首批检查 logits 逐元素一致。',
 '- 每模型、层、头、探测角色和精确相对距离的平均 attention，用独立256个验证查询估计；测试使用 A(q,k)-mean[A|q-k]。没有用测试结果拟合距离均值。',
 '- 图 attention 包含输入边列表的 EDGE、起点、终点；另行统计真实出边终点及合法候选终点，避免把协议符号的注意力当成邻居读取。',
 '- 先在每查询内平均采样状态，再跨查询平均；95%区间为查询级配对 bootstrap（500次），不是多训练seed置信区间。','',
 '## 生成第一个非空候选时：第5层','',
 '下表均为该查询位置的总 attention 质量；合法候选终点只取属于“当前节点出边，且未 visited”的输入边终点位置。','',
 '|图|读整个边列表：干净→新模型|读合法候选终点：干净→新模型|','|---|---:|---:|']
 for c in CASES:
  lines.append('|'+('N100' if c=='main' else c.upper())+'|'+pct(att(c,'clean',5,'graph_mass'))+' → '+pct(att(c,'error',5,'graph_mass'))+'|'+pct(att(c,'clean',5,'eligible_dest_mass'))+' → '+pct(att(c,'error',5,'eligible_dest_mass'))+'|')
 lines+=['','这不是所有层都同比增加。例如 N100 的第2层读图总量下降，第5层对合法后继终点则大幅增加，表明读取位置和功能分工发生了变化。','',
 '## 距离校正后的选择性','',
 '比较相关出边终点与无关边终点的平均校正 attention 差。该量是每key的残差密度差，可以为负，不是概率。','',
 '|图|第5层：干净|第5层：新模型|模型差的95%查询bootstrap区间|校正覆盖率（新模型）|','|---|---:|---:|---:|---:|']
 fi=features.index('residual_dest_density_contrast');ri=roles.index('candidate_nonempty')
 for c in CASES:
  a=data[c]['attention']
  lines.append(f"|{'N100' if c=='main' else c.upper()}|{a['clean'][4][ri][fi]:.5f}|{a['error'][4][ri][fi]:.5f}|[{a['ci_low'][4][ri][fi]:.5f}, {a['ci_high'][4][ri][fi]:.5f}]|{pct(att(c,'error',5,'residual_coverage'))}|")
 lines+=['','距离校正是描述性去均值：它不能完美区分 RoPE 与内容效应，广泛的位置/类别偏好也可能被一起减去。因此主要依据相关边选择性和下面的配对干预，而非残差图总量单独下结论。','',
 '## 数量与位置：只干预第5层的候选预测位置','',
 '下表只取应输出非空候选的状态。指标为“下一token是否属于合法候选集合”，允许任意合法邻居优先级，不是完整候选字段或最终导航答案准确率。每例保持相同前缀和batch；identity同样使用选定行FP32 attention作为无干预参照。','',
 '- 总量匹配：把新模型该位置的图attention质量，逐例改为干净模型同位置的质量；保持图内、图外各自的相对权重。',
 '- 打乱边：按固定随机种子重排整个边三元组的attention权重，保持EDGE/起点/终点角色及图attention总量；value不重排。',
 '- 清零图：只在第5层该位置去除输入边列表key，并重归一化其他key。','',
 '|图|新模型原始合法率|匹配干净模型总量后|总量不变、打乱边后|清零图后|图总量：原始→匹配后|','|---|---:|---:|---:|---:|---:|']
 for c in ['n16','n32','main']:
  v=mass[c]['result']['error'][0]
  lines.append(f"|{'N100' if c=='main' else c.upper()}|{pct(v[0][1])}|{pct(v[1][1])}|{pct(v[3][1])}|{pct(v[2][1])}|{pct(v[0][2])} → {pct(v[1][2])}|")
 lines+=['','在这个新模型中，更大的有效读图质量和正确的边定位都重要：只降低质量会显著损害输出，保留质量但打乱语义位置也会损害输出。这个干预不能推出“给任意模型增加attention总量就能提高能力”。','',
 '## 干净模型为什么直接读图少却仍会导航？','',
 '补做两种通路干预：仅阻断候选预测位置对图的读取；或仅阻断更早出现的当前节点token对图的读取（包括query和CoT中的当前节点）。后者会改变这些节点表示，随后可能通过CoT被候选位置读取。另以数量相同、位置接近的其他历史token作控制。','',
 '下表混合空/非空候选状态，均为query级平均下一token合法率；与上一表的非空子集口径不同。','',
 '|图/模型|identity|阻断候选位置读图|阻断当前节点表示读图|阻断匹配历史位置读图|两条通路同时阻断|','|---|---:|---:|---:|---:|---:|']
 for c in ['n16','n32','main']:
  for arm,label in [('clean','干净'),('error','新模型')]:
   v=channel[c]['lesion'][arm]
   lines.append('|'+('N100' if c=='main' else c.upper())+'/'+label+'|'+'|'.join(pct(x[1]) for x in v)+'|')
 lines+=['','N100 干净模型对当前节点表示的读图更敏感，而新模型对候选位置的直接读图更敏感，支持两模型信息路由不同。匹配历史位置也可能有功能，不能将该干预解释为完全隔离的单一电路。','',
 '对应的注意力模式：干净模型在较早当前节点表示上读图，候选位置第4/6层更多读取CoT中的当前节点；新模型候选位置第2层更多读取当前visited/节点信息，第5层再集中读取真实出边终点。这个流程与数据一致，但尚未完整重建value/FFN编码。','',
 '## 校验与解释边界','',
 '- 所有模型冻结，未训练、未写checkpoint、未创建W&B run；GPU0小批量前向，现有训练未停止。',
 '- 干预中的math identity与SDPA有少量BF16 argmax差异，检查数、最大logit差异逐实验记录；干预相互比较都使用相同计算路径。',
 '- 相同正确前缀避免了自由生成历史差异，但它研究的是局部候选能力。完整导航提升还涉及错误累积、路径选择及纠错，不能把下一token损失直接折算成最终准确率下降。',
 '- 本实验能确认新模型的直接相关边读取有功能；不能独立确认这条路由由错误数据而非初始化/难例重采样等因素造成，也没有把最终净增答案全部归因于它。',
 '- Attention质量不等于信息贡献大小，value、输出投影和残差也影响结果；因此保留了局部因果干预，而不只展示注意力热图。','',
 '## 结果文件','',
 '- n16/n24/n32/n36/main.json：各层、角色、原始/距离校正attention，查询级区间及相关边遮挡。',
 '- mass_*.json：第5层逐例总量匹配、保总量打乱边、清零图的干预。',
 '- channels_*.json：直接读图与当前节点表示的间接读图通路。',
 '- indirect_*.json：当前节点token本身的attention模式。',
 '- key_results.png：唯一关键结果图；无完整attention矩阵或activation dump。','']
 (P/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
 # Compact 3-panel figure: quantity, distance-corrected specificity, controls.
 fig,axes=plt.subplots(1,3,figsize=(16,4.7))
 matrix=np.asarray([[100*(att(c,'error',l,'graph_endpoint_mass')-att(c,'clean',l,'graph_endpoint_mass')) for l in range(1,7)] for c in CASES])
 im=axes[0].imshow(matrix,cmap='RdBu_r',vmin=-50,vmax=50,aspect='auto')
 axes[0].set_xticks(range(6),range(1,7));axes[0].set_yticks(range(5),['N16','N24','N32','N36','N100']);axes[0].set_xlabel('Layer');axes[0].set_title('Change in graph-endpoint attention (pp)')
 for y in range(5):
  for x in range(6):axes[0].text(x,y,f'{matrix[y,x]:.0f}',ha='center',va='center',color='white' if abs(matrix[y,x])>28 else 'black',fontsize=9)
 fig.colorbar(im,ax=axes[0],fraction=.046,pad=.04)
 for arm,color,label in [('clean','#4c78a8','Clean'),('error','#f58518','Error-trained')]:
  v=[att('main',arm,l,'residual_dest_density_contrast') for l in range(1,7)]
  axes[1].plot(range(1,7),v,'o-',color=color,label=label)
 axes[1].axhline(0,color='gray',lw=.7);axes[1].set_xticks(range(1,7));axes[1].set_xlabel('Layer');axes[1].set_ylabel('Per-key residual attention contrast');axes[1].set_title('N100: related vs unrelated edge targets');axes[1].legend()
 v=mass['main']['result']['error'][0];order=[0,1,3,2]
 axes[2].bar(range(4),[100*v[i][1] for i in order],color=['#f58518','#54a24b','#b279a2','#e45756'])
 axes[2].set_xticks(range(4),['Identity','Match clean mass','Shuffle edges','Zero graph'],rotation=23,ha='right');axes[2].set_ylim(0,110);axes[2].set_ylabel('Legal next-candidate token (%)');axes[2].set_title('N100: layer-5 quantity / placement controls')
 for x,i in enumerate(order):axes[2].text(x,100*v[i][1]+2,f'{100*v[i][1]:.1f}',ha='center')
 fig.tight_layout();fig.savefig(P/'key_results.png',dpi=200);plt.close(fig)
 # Strict JSON for downstream reuse; no NaN literals.
 for f in P.glob('*.json'):
  x=json.loads(f.read_text());f.write_text(json.dumps(sanitize(x),indent=2,allow_nan=False)+'\n')
 print('Verified outputs; report and one key figure written.')

if __name__=='__main__':main()

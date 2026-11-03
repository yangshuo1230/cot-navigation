"""Generate the single updated figure and a compact, explicit audit report."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parent
d=json.loads((ROOT/'results.json').read_text(encoding='utf-8'))
font=Path('C:/Windows/Fonts/msyh.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.size':10,'axes.unicode_minus':False,'pdf.fonttype':42})
names=['fixed_l10_dfs_nodes','fixed_l10_bfs_nodes','context_curriculum_dfs_nodes','context_curriculum_bfs_nodes']
names=[n for n in names if n in d['models']]
labels={'fixed_l10_dfs_nodes':'Fixed · DFS','fixed_l10_bfs_nodes':'Fixed · BFS',
        'context_curriculum_dfs_nodes':'In-Context · DFS','context_curriculum_bfs_nodes':'In-Context · BFS'}
left=[];right=[];ylabels=[];records=[]
for n in names:
    m=d['models'][n];a=m['tasks']['allowed'];f=m['tasks']['filter']
    left.append([a['scores'][5].get('pairwise_auc'),a['neighbor_only'][5].get('pairwise_auc'),f['scores'][5].get('pairwise_auc')])
    pm=f['matched'];v=pm.get('directional_accuracy',[None]*9)
    right.append([v[5],v[7],v[8]])
    ylabels.append(labels[n]+f"\n{pm['n_pairs']} 对 / {pm['n_queries']} 查询")
    records.append(dict(model=n,auc=left[-1],matched=pm,exact_stack=f['matched_exact_stack'],
                        mixed_support=f['scores'][5],shuffled_auc=m['shuffled_filter'][5].get('pairwise_auc')))
fig,axs=plt.subplots(1,2,figsize=(14.5,7.3))
fig.subplots_adjust(left=.16,right=.94,top=.74,bottom=.29,wspace=.65)
for ax,vals,title,cols,rows in [
    (axs[0],left,'候选排序：全图高分是否保留到真实后继内？',
     ['全图候选\n全图训练探针','仅真实后继\n同一个探针','仅真实后继\n专项探针'],[labels[n] for n in names]),
    (axs[1],right,'合法历史配对：未访问时分数是否更高？',
     ['L6 activation','当前节点\nembedding','历史节点\n集合'],ylabels)]:
    vals=np.array([[np.nan if v is None else v for v in row] for row in vals])*100
    im=ax.imshow(vals,cmap='Blues',vmin=0,vmax=100,aspect='auto')
    ax.set_xticks(range(3),cols);ax.set_yticks(range(len(names)),rows)
    ax.tick_params(length=0,pad=10)
    ax.set_title(title,pad=18,fontsize=12)
    for i in range(len(names)):
        for j in range(3):
            v=vals[i,j]
            ax.text(j,i,'无样本' if np.isnan(v) else f'{v:.1f}%',ha='center',va='center',fontsize=13,
                    weight='bold',color='white' if v>70 else '#17324d')
    ax.set_xticks(np.arange(-.5,3,1),minor=True)
    ax.set_yticks(np.arange(-.5,len(names),1),minor=True)
    ax.grid(which='minor',color='white',lw=1.5);ax.tick_params(which='minor',length=0)
    for s in ax.spines.values():s.set_visible(False)
cax=fig.add_axes([.957,.29,.01,.45]);fig.colorbar(im,cax=cax,ticks=[0,50,100])
fig.text(.035,.95,'后继关系与 visited 过滤：分开检验',fontsize=20,weight='bold',color='#17324d')
fig.text(.035,.88,'旧简单 DAG · N100 / M150 · 每查询 8 种合法搜索顺序 · 当前节点输出后的 L6 残差',fontsize=12,color='#536575')
fig.text(.035,.18,'左：状态内 AUC，只在同时含已访问和未访问后继的状态计算“后继内”指标；随机水平 50%。\n'
         '右：同图、同起终点、同当前节点；比较同一后继在未访问／已访问历史中的探针分数，平分计 0.5。',
         fontsize=10,linespacing=1.8,color='#38495a')
fig.text(.035,.06,'BFS 此处测“当前展开节点的未访问后继”，与旧图的“合法下一展开分组”不同。\n'
         '使用合法参考前缀，非自由生成；查询留出，无节点留出；配对共享节点与查询，不是独立样本。',
         fontsize=10,linespacing=1.8,color='#637181')
for ext in ['png','pdf']:fig.savefig(ROOT/f'search_filter_comparison.{ext}',dpi=200,facecolor='white')
plt.close(fig)
(ROOT/'summary.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# 搜索候选过滤与合法历史配对复核','',
       '冻结旧 scale_v1 N100/M150 模型；所有拟合 GPU0；不保存权重或 activation。',
       '每模型原查询 train/val/test=256/96/128，每查询8种随机合法邻居顺序。不同顺序不跨查询划分。',
       '主要位置改为当前展开节点已输出后的残差，并保留输出前 L6。BFS 预测当前节点的未发现后继，不是旧图的下一展开 cohort。',
       '所有首展开状态从独立节点前缀重放验证 visited/frontier/candidates；排除起点与当前目标节点。',
       '集合成员标签采用 multi-hot，不评估完整顺序；完整 top-k 使用真实 k，空集和全集不参与 AUC。','',
       '“同一个探针”指本次全图训练的候选探针仅修改评分 mask，不是重用历史已丢弃的探针权重。',
       '|模型|全图候选 AUC|同探针后继内 AUC|专项后继内 AUC|打乱后继标签 AUC|混合状态数/查询数/当前节点token种类|',
       '|---|---:|---:|---:|---:|---|']
def pct(v):return '无定义' if v is None else f'{100*v:.1f}%'
for r in records:
    q=r['mixed_support']
    lines.append('|'+labels[r['model']]+'|'+'|'.join(pct(v) for v in r['auc'])+'|'+pct(r['shuffled_auc'])+f"|{q['nontrivial_rows']}/{q['n_queries_mixed']}/{q['n_current_nodes_mixed']}|")
lines+=['','## 合法历史配对','',
        '同一查询、当前节点、被比较后继；从8种合法顺序中确定性选择一对，每查询/当前节点/后继最多一对，不按模型效果选样。',
        '“未访问”一侧减“已访问”一侧分数大于0为成功，等于0计0.5。当前节点 embedding 对照理论上恒为0.5。',
        'DFS 另保留完整有序栈相同的更严格子集；BFS 不强行要求相同完整队列。','',
        '|模型|配对数/查询数/当前节点token种类|专项 L6|当前 embedding|历史集合|DFS同完整栈：配对数/查询数/当前token种类，L6|',
        '|---|---|---:|---:|---:|---|']
for r in records:
    p=r['matched'];e=r['exact_stack'];v=p.get('directional_accuracy',[None]*9)
    support=f"{p['n_pairs']}/{p['n_queries']}/{p.get('n_current_nodes',0)}"
    es='不适用或无配对' if not e['n_pairs'] else f"{e['n_pairs']}/{e['n_queries']}/{e['n_current_nodes']}，{pct(e['directional_accuracy'][5])}"
    lines.append(f"|{labels[r['model']]}|{support}|{pct(v[5])}|{pct(v[7])}|{pct(v[8])}|{es}|")
lines+=['','## 专项过滤的层与对照','',
        '|模型|L1后|L2后|L3后|L6后|L6当前节点前|当前embedding|历史节点集合|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
for name in names:
    ss=d['models'][name]['tasks']['filter']['scores']
    lines.append('|'+labels[name]+'|'+'|'.join(pct(ss[i].get('pairwise_auc')) for i in [0,1,2,5,6,7,8])+'|')
lines+=['','## 结果解释','',
        '- 全图候选 AUC 与后继内过滤衡量不同问题。将同一个全图探针限制到后继，才可判断其高分是否仅来自排除非邻居。',
        '- 专项训练不保证必然超过全图训练；训练 mask、类别比例和优化目标改变，本次均沿用250步、验证 BCE 选择，未按测试调参。',
        '- 配对方向分数高但状态内 AUC 不高并不矛盾：前者比较同一后继在不同历史中的分数；后者比较同一状态内不同后继的分数，后者还需要跨节点校准。',
        '- Fixed 的历史节点集合专项对照很强，activation 的动态信号不能直接解释成完整搜索状态机。',
        '- 当前节点 embedding 在配对中恒相同，配对得分严格50%；历史配对超过它只能排除纯当前节点静态身份解释。']
lines+=['','## 限制','',
        '- 这批为合法参考上下文专项诊断，不能冒充旧自由生成前缀结果；前后读出位置也不同。',
        '- 只有同时含已访问和未访问后继的状态才检验过滤，必须结合支持数量，不能将同节点多种顺序当独立节点。',
        '- 参考合法历史中的读出差异不是因果中介证明，也没有测原模型原生下一 token 是否按该特征决策。',
        '- 本次不进行节点留出；固定图主模型见过全部节点。控制当前节点不变排除了仅依赖其静态身份的配对解释，仍可能有其他历史相关特征。',
        '- 配对不额外匹配前缀长度或 visited 数量，不能排除进度等历史相关线索。当前节点计数指唯一 token ID 种类，新图中不同图的同名 token 不代表同一个图节点。',
        '- 历史节点集合对照在固定图上可能很强；DFS visited 是历史展开节点集合，BFS discovered 还包含已入队未展开节点。',
        '- 节点共享、查询共享与单张固定图意味着不能把配对数当独立样本数。不报告伪精确显著性。']
(ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))

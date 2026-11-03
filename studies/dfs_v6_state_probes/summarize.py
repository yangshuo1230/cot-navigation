"""One result figure and report for the repeated v6 state probes."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

root=Path(__file__).resolve().parent
d=json.loads((root/'results.json').read_text())
models=list(d['models'].values())
panels=['short_test','long_AA_test','long_TA_test','long_AT_test','long_TT_test']
labels=['短程','长程 A→A','长程 T→A','长程 A→T','长程 T→T']
tasks=['succ','visited','frontier','filter']
columns=['全部后继','visited 集合','栈成员集合','后继内过滤']

def mean(values):
    v=[x for x in values if x is not None]
    return float(np.mean(v)) if v else None

def metric(task,source,panel,key='pairwise_auc',head=5):
    return mean([m['tasks'][task][source][panel]['scores'][head].get(key) for m in models])

summary={}
for p in panels:
    s={source:{t:{key:metric(t,source,p,key) for key in ['pairwise_auc','topk_exact','f1','threshold_exact_set_accuracy']}
               for t in ['succ','visited','frontier','allowed','filter']} for source in ['reference','free']}
    s['reference_filter']={}
    for name,head in [('L6',5),('before_L6',6),('current_embedding',7),('history_bag',8)]:
        s['reference_filter'][name]=metric('filter','reference',p,head=head)
    s['reference_filter']['full_probe_neighbor_auc']=mean([m['tasks']['allowed']['reference'][p]['neighbor_only'][5].get('pairwise_auc') for m in models])
    s['reference_controls']={t:{label:metric(t,'reference',p,head=head) for label,head in [('L1',0),('L6',5),('embedding',7),('history',8)]} for t in ['visited','frontier','filter']}
    for k in ['matched','matched_exact_stack']:
        s[k]={}
        s[k]['support']=[{key:m['tasks']['filter']['reference'][p][k].get(key) for key in ['n_pairs','n_queries','n_current_nodes']} for m in models]
        for name,head in [('L6',5),('current_embedding',7),('history_bag',8)]:
            s[k][name]=mean([m['tasks']['filter']['reference'][p][k].get('directional_accuracy',[None]*9)[head] for m in models])
    s['shuffled_filter']=mean([m['shuffled_filter'][p][5].get('pairwise_auc') for m in models])
    s['free_by_completion']={}
    for completed in ['True','False']:
        s['free_by_completion'][completed]={}
        for t in ['succ','visited','frontier','filter']:
            entries=[m['tasks'][t]['free_by_completion'][p].get(completed) for m in models]
            s['free_by_completion'][completed][t]={
                'pairwise_auc':mean([e['scores'][5].get('pairwise_auc') for e in entries if e]),
                'L1_pairwise_auc':mean([e['scores'][0].get('pairwise_auc') for e in entries if e]),
                'topk_exact':mean([e['scores'][5].get('topk_exact') for e in entries if e]),
                'L1_topk_exact':mean([e['scores'][0].get('topk_exact') for e in entries if e]),
                'queries_by_seed':[e['queries'] if e else 0 for e in entries]}
    summary[p]=s
(root/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

font=Path('C:/Windows/Fonts/msyh.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.size':11,'axes.unicode_minus':False,'pdf.fonttype':42})
fig,axs=plt.subplots(1,3,figsize=(17.5,6.8),gridspec_kw={'width_ratios':[4,4,4]})
fig.subplots_adjust(left=.08,right=.95,top=.75,bottom=.29,wspace=.32)
for ax,source,title in zip(axs[:2],['reference','free'],['合法参考前缀（每查询8种顺序）','模型自由生成的合法前缀']):
    values=np.array([[metric(t,source,p) for t in tasks] for p in panels],float)*100
    im=ax.imshow(values,vmin=0,vmax=100,cmap='Blues',aspect='auto')
    ax.set_xticks(range(4),columns,fontsize=10);ax.set_yticks(range(5),labels)
    ax.set_title(title,pad=16,fontsize=13);ax.tick_params(length=0,pad=10)
    for i in range(5):
        for j in range(4):
            val=values[i,j];ax.text(j,i,'NA' if np.isnan(val) else f'{val:.1f}%',ha='center',va='center',fontsize=13,
                                  weight='bold',color='white' if val>70 else '#17324d')
    ax.set_xticks(np.arange(-.5,4,1),minor=True);ax.set_yticks(np.arange(-.5,5,1),minor=True)
    ax.grid(which='minor',color='white',lw=1.5);ax.tick_params(which='minor',length=0)
    for spine in ax.spines.values():spine.set_visible(False)
control=summary['long_TT_test']['reference_controls']
vals=np.array([[control[t][key] for key in ['L1','L6','embedding','history']] for t in ['visited','frontier','filter']],float)*100
ax=axs[2];ax.imshow(vals,vmin=0,vmax=100,cmap='Blues',aspect='auto')
ax.set_title('T→T 参考：早层、末层与对照',pad=16,fontsize=12)
ax.set_xticks(range(4),['L1','L6','当前节点\nembedding','历史节点\n集合'],fontsize=9)
ax.set_yticks(range(3),['visited','栈成员','过滤'],fontsize=10);ax.tick_params(length=0,pad=8)
for i in range(3):
    for j in range(4):ax.text(j,i,f'{vals[i,j]:.1f}%',ha='center',va='center',fontsize=12,weight='bold',color='white' if vals[i,j]>70 else '#17324d')
ax.set_xticks(np.arange(-.5,4,1),minor=True);ax.set_yticks(np.arange(-.5,3,1),minor=True)
ax.grid(which='minor',color='white',lw=1.5);ax.tick_params(which='minor',length=0)
for spine in ax.spines.values():spine.set_visible(False)
cax=fig.add_axes([.965,.29,.008,.46]);fig.colorbar(im,cax=cax,ticks=[0,50,100])
fig.text(.045,.95,'v6 DFS nodes：visited 与搜索状态是否仍可读？',fontsize=19,weight='bold',color='#17324d')
fig.text(.045,.88,f'N80 固定图 · final 50k · {len(models)}个初始化均值 · 当前节点输出后的 L6 残差 · 状态内 AUC',fontsize=11,color='#536575')
fig.text(.045,.17,'探针仅拟合短程／A→A训练查询，验证也仅用短程／A→A；T端点长程查询只用于测试。\n'
         '后继内过滤仅比较真实后继中的已访问／未访问节点；AUC随机水平50%，高分不等于整集合恢复。',fontsize=10,color='#536575',linespacing=1.8)
fig.text(.045,.045,'自由生成仅保留首次非法展开之前的状态，排除起点及当前目标位置；包含最终失败查询的先前合法状态。\n'
         '标签为集合，不评价栈顺序；未新增节点身份留出。高可读性不证明原生输出使用该信息。',fontsize=10,color='#697783',linespacing=1.8)
for ext in ['png','pdf']:fig.savefig(root/f'v6_state_probes.{ext}',dpi=200,facecolor='white')
plt.close(fig)

def pct(x):return 'NA' if x is None else f'{x:.1%}'
lines=['# v6 DFS nodes：复跑状态探针','',
       f'完成{len(models)}/4个初始化，固定final50k；N80 graph0。模型冻结，所有拟合GPU0，无checkpoint/activation缓存。',
       '复用search_filter_audit的提取、250步仿射头、训练标准化、query加权BCE、验证BCE早停与合法配对规则。',
       '当前节点之后L1–L6以及当前节点之前L6、当前embedding、完整历史节点集合对照。读出标签为全部后继、visited、栈成员、合法候选及后继内专项过滤。',
       f"Probe train={len(d['train_ids'])}（192短程+64长AA），val={len(d['val_ids'])}（72短程+24长AA）；原query划分互斥。测试不用于拟合或选步。", 
       '参考顺序严格复用v6冻结bank中8个变体，并逐条断言节点序列一致，不替换成旧批次的顺序生成器。',
       'free生成无oracle；提取合法展开前缀。排除起点和目标节点，首次非法展开及其后无合法状态标签。失败组不包含错误节点本身的错误后缀。','',
       '## L6状态内AUC','',
       '|面板|上下文|全部后继|visited|栈成员|合法候选全图|后继内过滤|',
       '|---|---|---:|---:|---:|---:|---:|']
for p,label in zip(panels,labels):
    for source in ['reference','free']:
        lines.append('|'+label+'|'+source+'|'+'|'.join(pct(summary[p][source][t]['pairwise_auc']) for t in ['succ','visited','frontier','allowed','filter'])+'|')
lines+=['','## 完整集合恢复（真实k的top-k；不恢复顺序）','',
        '|面板|上下文|visited整集合|栈成员整集合|后继内过滤整集合|','|---|---|---:|---:|---:|']
for p,label in zip(panels,labels):
    for source in ['reference','free']:
        lines.append('|'+label+'|'+source+'|'+'|'.join(pct(summary[p][source][t]['topk_exact']) for t in ['visited','frontier','filter'])+'|')
lines+=['','## 参考上下文：状态读出的静态身份与历史对照','',
        '|面板|任务|L1|L6|当前节点embedding|历史节点集合|','|---|---|---:|---:|---:|---:|']
for p,label in zip(panels,labels):
    for t in ['visited','frontier','filter']:
        c=summary[p]['reference_controls'][t]
        lines.append('|'+label+'|'+t+'|'+'|'.join(pct(c[k]) for k in ['L1','L6','embedding','history'])+'|')
lines+=['','## T→T参考：逐层读出','',
        '|任务/指标|L1|L2|L3|L4|L5|L6|','|---|---:|---:|---:|---:|---:|---:|']
for t in ['visited','frontier','filter']:
    for key in ['pairwise_auc','topk_exact']:
        lines.append('|'+t+'/'+key+'|'+'|'.join(pct(metric(t,'reference','long_TT_test',key,head=i)) for i in range(6))+'|')
lines+=['','## 参考后继内过滤与合法历史配对','',
        '|面板|全图探针限制后继|专项L6|当前embedding|历史集合|打乱标签|同节点配对方向L6|同完整栈配对L6|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
for p,label in zip(panels,labels):
    s=summary[p];f=s['reference_filter']
    lines.append('|'+label+'|'+'|'.join(pct(v) for v in [f['full_probe_neighbor_auc'],f['L6'],f['current_embedding'],f['history_bag'],s['shuffled_filter'],s['matched']['L6'],s['matched_exact_stack']['L6']])+'|')
lines+=['','## 自由生成：完整合法／最终失败查询的先前合法状态','',
        '|面板|完整合法？|每seed支持查询数|visited L1 AUC|visited L6 AUC|栈 L6 AUC|过滤 L6 AUC|visited L6整集合|',
        '|---|---|---|---:|---:|---:|---:|---:|']
for p,label in zip(panels,labels):
    for completed in ['True','False']:
        s=summary[p]['free_by_completion'][completed]
        lines.append(f"|{label}|{completed}|{s['visited']['queries_by_seed']}|"+'|'.join(pct(v) for v in [s['visited']['L1_pairwise_auc'],s['visited']['pairwise_auc'],s['frontier']['pairwise_auc'],s['filter']['pairwise_auc'],s['visited']['topk_exact']])+'|')
lines+=['','## 解释限制','',
        '- 原主模型见过所有节点；此处是query留出与长程端点角色测试，不是原模型或新解码器的未见节点测试。',
        '- N80与先前N100图、训练数据、上下文不同，不能将跨批读出差异单独归因于曝光约束。',
        '- visited、frontier是集合，不证明完整有序栈。过滤高分不证明模型原生选择或终止读了同一方向。',
        '- 失败组的合法前缀通常较短，成功/失败之间不构成相同状态的因果比较；首次出错位置之后未测。',
        '- 参考合法顺序与自由生成使用相同已拟合探针；不在free测试或失败子集上重训。',
        '- 同节点合法配对排除单纯当前身份解释，未额外匹配前缀长度/visited数，仍可能存在进度等相关性。',
        '- 只有一张图，四模型初始化；pair共享查询/节点，均值不是独立图泛化或置信区间。',
        '- 向量化AUC/严格分离/top-k与原慢实现对随机分数、并列分数、空/全集及变mask逐项一致，验证结果见verification.json。']
lines+=['','## 后继内评分支持（每seed有效混合状态行数）','',
        '只在同一状态同时有已访问后继和未访问后继时计算过滤AUC。自由生成失败分组的query支持数是存在合法锚点的query数量，不表示每条query都有过滤混合状态。',
        '', '|面板|参考混合行数|自由生成混合行数|','|---|---|---|']
for p,label in zip(panels,labels):
    rr=[m['tasks']['filter']['reference'][p]['scores'][5]['nontrivial_rows'] for m in models]
    ff=[m['tasks']['filter']['free'][p]['scores'][5]['nontrivial_rows'] for m in models]
    lines.append(f'|{label}|{rr}|{ff}|')
(root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Summarized',len(models),'models')

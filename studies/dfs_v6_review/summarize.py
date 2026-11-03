"""Summarize immutable final-step v6 pilot evaluations, not test-selected runs."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

root=Path(__file__).resolve().parent
d=json.loads((root/'collected.json').read_text(encoding='utf-8-sig'))
runs={p:sorted([r for r in d['runs'] if r['protocol']==p],key=lambda r:r['seed']) for p in ['direct','dfs_nodes']}
assert all(len(r)==4 for r in runs.values())
assert all(r['step']==50000 and r['complete'] for r in d['runs'])
panels=['short_test','long_AA_test','long_TA_test','long_AT_test','long_TT_test']
summary={}
for name in panels:
    s={}
    for p,rr in runs.items():
        a=[r['panels'][name]['all'] for r in rr]
        s[p]={key:float(np.mean([x[key] for x in a])) for key in ['ba','accuracy','yes_recall','no_recall']}
        s[p]['n_per_seed']=a[0]['n']
        s[p]['ba_by_seed']=[x['ba'] for x in a]
        if p=='dfs_nodes':
            s[p].update({key:float(np.mean([x[key] for x in a])) for key in
                ['trace_valid_rate','positive_trace_valid_rate','negative_trace_valid_rate']})
            s[p]['legal_by_seed']=[x['trace_valid_rate'] for x in a]
            errors=Counter()
            for r in rr:errors.update(r['panels'][name]['errors'])
            s[p]['errors_pooled']=dict(errors)
            assert all(x['truncated']==0 for x in a)
    summary[name]=s
for r in runs['dfs_nodes']:
    for p in r['exposure']['panels'].values():
        assert p['direct_query_label_seen']==0
        assert p['positive_n']==p['semantic_tree_relation_seen']
        assert p['macro_distance_counts']=={'1':p['positive_n']}
(root/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

font=Path('C:/Windows/Fonts/msyh.ttc')
font_manager.fontManager.addfont(str(font))
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(font)).get_name(),'pdf.fonttype':42,'axes.unicode_minus':False})
fig,ax=plt.subplots(figsize=(11.5,5.9))
fig.subplots_adjust(left=.08,right=.98,top=.76,bottom=.27)
x=np.arange(5);w=.24
for offset,p,key,color,label in [(-w,'direct','ba','#8c9dad','Direct 答案 BA'),
                                 (0,'dfs_nodes','ba','#2878b5','DFS nodes 答案 BA'),
                                 (w,'dfs_nodes','trace_valid_rate','#e59b43','DFS nodes 完整轨迹合法率')]:
    vals=[100*summary[n][p][key] for n in panels]
    ax.bar(x+offset,vals,width=w*.94,color=color,label=label)
    for xx,yy in zip(x+offset,vals):ax.text(xx,yy+1,f'{yy:.1f}',ha='center',fontsize=9)
ax.set_xticks(x,['短程测试\n1–3跳','长程 A→A\n31条','长程 T→A\n168条','长程 A→T\n168条','长程 T→T\n84条'])
ax.set_ylim(0,108);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('百分比（4个初始化种子均值）')
ax.spines[['top','right']].set_visible(False);ax.set_axisbelow(True);ax.grid(axis='y',alpha=.15)
fig.legend(loc='upper center',bbox_to_anchor=(.52,.86),ncol=3,frameon=False)
fig.text(.055,.95,'限制长程端点角色后：目标角色迁移仍有明显缺口',fontsize=17,weight='bold',color='#17324d')
fig.text(.055,.13,'N80 单张简单 DAG · 固定 50k 步 · 4个初始化种子 · 原生自由生成，无 oracle 修复。\n'
         'A：长程训练端点；T：长程端点留出，但可出现在短程训练、ADJ 和搜索中间位置。',fontsize=10,color='#526171',linespacing=1.7)
fig.text(.055,.035,'曝光审计：所有面板正例关系均被训练发现树语义覆盖；这不是严格无 witness 暴露实验。',fontsize=10,color='#8c5320')
for ext in ['png','pdf']:fig.savefig(root/f'v6_dfs_nodes_results.{ext}',dpi=200,facecolor='white')
plt.close(fig)

lines=['# v6 DFS nodes：长程端点角色留出结果复核','',
       '来源：outputs/fixed_navigation_protocols_v6/pilot/n80/g0_s{0,1,2,3}/{direct,dfs_nodes}/final.json 与 actual_exposure.json。',
       'N80 单张图，四个初始化；每模型均固定50k步完成，无测试选checkpoint。DFS state尚未完成，未混入比较。',
       '短程1–3跳只留出query；长程4–9跳训练端点只用A，T不作长程训练query端点，但允许在短程、ADJ和搜索内部出现。',
       '所有结果为已有完整原生自由生成测试，未重新推理或拟合探针。','',
       '|面板|每seed查询数|Direct BA|DFS nodes BA|完整合法率|正例完整合法率|负例完整合法率|',
       '|---|---:|---:|---:|---:|---:|---:|']
for n in panels:
    a=summary[n];b=a['dfs_nodes']
    lines.append(f"|{n}|{b['n_per_seed']}|{a['direct']['ba']:.1%}|{b['ba']:.1%}|{b['trace_valid_rate']:.1%}|{b['positive_trace_valid_rate']:.1%}|{b['negative_trace_valid_rate']:.1%}|")
lines+=['','## 曝光约束核查','',
        '四seed所有测试面板 direct_query_label_seen=0；但 semantic_tree_relation_seen=positive_n，宏关系最小拼接数均为1。',
        '例如TT每seed57个正例全部被训练发现树祖先—后代关系覆盖。训练树关系不是连续节点token组成的直接边，重建需要已知图；因此不能把这项保守语义覆盖冒充直接query答案监督或证明模型确实使用这些关系。',
        '结论是：端点角色留出成立，严格未见关系/未见witness不成立。不能声称已经排除图关系记忆或训练子搜索复用。','',
        '## 对先前结论的影响','',
        '- 目标T的长程迁移明显比起点T更难；不能将固定图内高探针成绩推广成对新端点角色的稳定完整DFS。',
        '- TT正例完整合法率约49.6%，负例约79.6%；目标识别/终止和合法展开两类问题都存在。',
        '- AT中首错合计：正例非法展开101、到目标后继续86、提前终止24；TT对应66/35/10。计数跨seed，同一查询复用，不能视为独立样本。',
        '- AT完整合法率随gap4到9约76.6/67.5/57.8/54.2/40.6/28.1%，但最长桶没有负例，不能作受控长度因果解释。',
        '- Direct的TT BA约86.1%，DFS nodes约84.3%；本轮没有显示nodes在留出端点关系判断上稳定优于Direct。',
        '- 没有对这轮checkpoint重跑同位置线性探针；因此不能据行为失败断言activation没有visited信息。',
        '- 目前只一张80节点图、四初始化，不是四张独立图；AA只有31条查询，不作过强泛化结论。','',
        '复核断言：8个run全为50k且COMPLETE；同面板4seed支持一致；nodes无截断；曝光直接query为0、发现树语义覆盖100%。']
(root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('PASS: completed runs, 4-seed aggregate, no truncation, exposure constraints verified')

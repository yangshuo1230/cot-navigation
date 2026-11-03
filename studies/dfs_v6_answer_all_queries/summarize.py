"""Percentage-progress figures for the all-query external-probe split."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

root=Path(__file__).resolve().parent
d=json.loads((root/'results.json').read_text());models=list(d['models'].values())
phases=[p for p,_ in d['phases']];xx=[int(100*v) for _,v in d['phases']]
panels=list(d['test_panels'])

def stat(phase,family,source,panel,pop,head,key='balanced_accuracy'):
    vals=[];support=[]
    for m in models:
        item=m['phases'][phase][family]
        if 'status' in item:continue
        r=item[source][panel][pop]
        if not r['heads']:continue
        v=r['heads'][head][key]
        if v is not None:vals.append(v)
        support.append({k:r.get(k,0) for k in ['n_rows','n_queries','n_positive_queries','n_negative_queries']})
    return (float(np.mean(vals)) if vals else None),support

summary={}
for panel in panels:
    s={'n_test_queries':len(d['test_panels'][panel]),'phases':{}}
    native=[m['native_final'][panel]['balanced_accuracy'] for m in models]
    s['native_final_BA']=float(np.mean([v for v in native if v is not None])) if any(v is not None for v in native) else None
    for phase in phases:
        p={}
        for source in ['reference','free']:
            p[source]={}
            for family,pop in [('all','all'),('all','no_target_yet'),('active','active_legal')]:
                value={}
                for h,label in enumerate(d['heads']):
                    value[label]={key:stat(phase,family,source,panel,pop,h,key)[0] for key in ['accuracy','balanced_accuracy','yes_recall','no_recall']}
                value['support']=stat(phase,family,source,panel,pop,5)[1]
                p[source][family+'/'+pop]=value
        p['shuffled_active_L6_BA']=stat(phase,'active','shuffled_reference',panel,'active_legal',5)[0]
        s['phases'][phase]=p
    summary[panel]=s
(root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')

font=Path('C:/Windows/Fonts/msyh.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.size':10,'axes.unicode_minus':False,'pdf.fonttype':42})
fig,axes=plt.subplots(2,3,figsize=(13.5,8.8),sharex=True,sharey=True)
fig.subplots_adjust(left=.08,right=.98,top=.78,bottom=.23,wspace=.12,hspace=.32)
series=[('active','active_legal',5,'L6：合法且未结束','#2675b9','-','o'),
        ('active','active_legal',0,'L1：合法且未结束','#327d70','-','^'),
        ('all','all',5,'L6：全部位置（含终止）','#e59b43','--','s'),
        ('active','active_legal',7,'起终点 embedding 对照','#8b8d94',':',None),
        ('active','active_legal',8,'长度等可观测特征对照','#a35494','-.',None)]
for row,source in enumerate(['reference','free']):
    for col,(panel,label) in enumerate(zip(['all','AT','TT'],['全部探针测试查询','长程 A→T','长程 T→T'])):
        ax=axes[row,col]
        for family,pop,head,legend,color,style,marker in series:
            vals=[stat(p,family,source,panel,pop,head)[0] for p in phases]
            ax.plot(xx,[np.nan if v is None else 100*v for v in vals],color=color,ls=style,marker=marker,ms=3,lw=1.6,label=legend)
        ax.set_title(f"{label} n={len(d['test_panels'][panel])}\n"+('合法参考轨迹' if row==0 else '模型自由生成'),fontsize=10)
        ax.set_ylim(40,102);ax.set_yticks([50,60,70,80,90,100]);ax.set_xticks([0,20,40,60,80,100])
        ax.axhline(50,color='#aaaaaa',lw=.8);ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
        if col==0:ax.set_ylabel('答案探针 balanced accuracy（%）')
        if row==1:ax.set_xlabel('CoT 生成进度（%）')
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.53,.875),ncol=3,frameon=False,fontsize=9)
fig.text(.04,.96,'CoT 百分比进度：探针训练覆盖全部查询设置',fontsize=19,weight='bold',color='#17324d')
counts={k:len(v) for k,v in d['query_ids'].items()}
fig.text(.04,.915,f"全部2712查询重划分：train {counts['train']} / val {counts['val']} / test {counts['test']} · {len(models)}个初始化均值",fontsize=11,color='#536575')
fig.text(.04,.13,'训练覆盖短程及 AA／VA／AV／VV／TA／AT／TT；同一 query 的8种轨迹始终在同一划分。横轴仅用百分比。\n'
         '蓝／绿线排除已终止、已到目标和非法历史；100%处无有效参考状态。紫线检验进度定位下的长度混杂。',fontsize=10,color='#536575',linespacing=1.8)
fig.text(.04,.035,'这是外部探针的新划分，导航模型未重训；部分探针测试 query 已被导航模型见过，原见过／未见结果分开记录。\n'
         'A→T、T→T子集样本少，仅作描述。各进度有效query变化，不能把曲线直接解释为CoT的因果收益。',fontsize=9,color='#697783',linespacing=1.8)
for ext in ['png','pdf']:fig.savefig(root/f'answer_progress_all_queries.{ext}',dpi=200,facecolor='white')
plt.close(fig)

def pct(v):return 'NA' if v is None else f'{v:.1%}'
lines=['# 全查询重划分：CoT百分比进度答案探针','',
       f"完成{len(models)}/4个v6 DFS nodes final50k，模型权重不变。探针训练物理GPU0，冻结推理GPU1。", 
       '用户要求：探针训练覆盖所有查询设置，不只AA；横轴只使用CoT百分比。',
       f"原数据全部2712条query，以角色/层差/标签分层后，探针train={counts['train']}、val={counts['val']}、test={counts['test']}。小分层调整以保留样本支持，实际比例见下表。", 
       '同一query的8种离线轨迹不跨划分；reference训练/验证，reference及原生free测试使用同一拟合头。各百分比/层独立仿射分类器，250步上限，val BCE选步、val BA选阈值。',
       '0%为prompt末尾；百分比通过事后完整CoT长度定位，不输入YES/NO/EOS。active族训练、验证和测试都只保留合法未终止位置。参考终止之前两族训练数据相同，复用完全相同的拟合头以避免重复计算。',
       '注意：这是外部probe数据的新划分，不是重新训练导航模型。probe train可包含原主模型test query的标签；probe test仍与probe train/val严格分离。','',
       '|query设置|probe train|probe val|probe test|','|---|---:|---:|---:|']
for role,c in d['coverage'].items():lines.append(f"|{role}|{c['train']}|{c['val']}|{c['test']}|")
lines+=['','## 全部探针测试查询：BA','',
        '|上下文|CoT进度|L6全部位置|L6合法未结束|L1合法未结束|query embedding对照|长度等可观测对照|','|---|---:|---:|---:|---:|---:|---:|']
for source in ['reference','free']:
    for phase in phases:
        r=summary['all']['phases'][phase][source];a=r['all/all'];b=r['active/active_legal']
        lines.append('|'+source+'|'+phase[1:]+'%|'+'|'.join(pct(v) for v in [a['L6']['balanced_accuracy'],b['L6']['balanced_accuracy'],b['L1']['balanced_accuracy'],b['query_embeddings']['balanced_accuracy'],b['observable_control']['balanced_accuracy']])+'|')
lines+=['','## 所有设置与原导航query见过/未见分组：active读出','',
        '|设置|上下文|进度|L6 accuracy|L6 BA|YES recall|NO recall|L1 BA|每seed正/负query支持|','|---|---|---:|---:|---:|---:|---:|---:|---|']
for panel in panels:
    for source in ['reference','free']:
        for phase in phases:
            b=summary[panel]['phases'][phase][source]['active/active_legal'];h=b['L6'];support=[(v['n_positive_queries'],v['n_negative_queries']) for v in b['support']]
            lines.append(f"|{panel}|{source}|{phase[1:]}%|"+'|'.join(pct(v) for v in [h['accuracy'],h['balanced_accuracy'],h['yes_recall'],h['no_recall'],b['L1']['balanced_accuracy']])+f'|{support}|')
lines+=['','## 打乱标签对照：全部query、参考active L6 BA','']
for phase in phases:lines.append(f"- {phase[1:]}%：{pct(summary['all']['phases'][phase]['shuffled_active_L6_BA'])}")
lines+=['','## 边界','',
        '- 不再要求探针跨AA→T迁移。本结果回答覆盖全部角色监督后答案可读性如何变化，不能与旧探针泛化测试直接等同。',
        '- 只有一张N80图，4个模型初始化；AT31、TT14个probe测试查询，不作稳定排名或显著性断言。',
        '- 查询权重相等，BA平衡正负recall；active查询支持随进度改变。',
        '- 相对进度由未来长度定位，未来长度未作为显式输入，但已生成token数仍可能关联标签；长度等可观测特征对照必须一同解释。',
        '- 起终点embedding仿射头接受probe标签监督，可能拟合节点身份/角色先验，不证明模型原生已经使用答案。',
        '- 终点出现后可直接揭示YES，高分不是提前规划证据；线性探针也不是原模型中途回答准确率。',
        '- 所有checkpoint/activation/探针权重不新保存，仅保留关键汇总JSON、报告与一张主图。']
(root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Summarized',len(models),'models',counts)

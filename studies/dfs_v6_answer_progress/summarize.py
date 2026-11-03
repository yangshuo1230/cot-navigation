"""Report phase-specific answer probes with terminal and cohort controls."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

root=Path(__file__).resolve().parent
d=json.loads((root/'results.json').read_text());models=list(d['models'].values())
panels=['short_test','long_AA_test','long_TA_test','long_AT_test','long_TT_test']
ratios=['r0','r25','r50','r75','r100'];absolute=['t1','t2','t4','t8','t16','t32','t64']

def entries(phase,family,source,panel,population,head,metric):
    values=[];support=[]
    for m in models:
        item=m['phases'][phase].get(family,{})
        if 'status' in item:continue
        e=item[source][panel][population]
        if not e['heads']:continue
        v=e['heads'][head][metric]
        if v is not None:values.append(v)
        support.append({k:e.get(k,0) for k in ['n_rows','n_queries','n_positive_queries','n_negative_queries']})
    return values,support


def mean_value(*args):
    v,_=entries(*args)
    return float(np.mean(v)) if v else None


summary={}
for panel in panels:
    s={'native_final_BA':float(np.mean([m['native_final'][panel]['balanced_accuracy'] for m in models])), 'phases':{}}
    for phase,_ in d['phases']:
        p={}
        for source in ['reference','free']:
            p[source]={}
            for family,population in [('all','all'),('all','no_target_yet'),('active','active_legal')]:
                key=family+'/'+population
                result={}
                for h,label in enumerate(d['heads']):
                    result[label]={metric:mean_value(phase,family,source,panel,population,h,metric)
                                   for metric in ['accuracy','balanced_accuracy','yes_recall','no_recall']}
                result['support']=entries(phase,family,source,panel,population,5,'balanced_accuracy')[1]
                p[source][key]=result
        p['shuffled_active_L6_BA']=mean_value(phase,'active','shuffled_reference',panel,'active_legal',5,'balanced_accuracy')
        s['phases'][phase]=p
    summary[panel]=s
(root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')

font=Path('C:/Windows/Fonts/msyh.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.size':10,'axes.unicode_minus':False,'pdf.fonttype':42})
fig,axes=plt.subplots(2,3,figsize=(13.5,8.5),sharex=True,sharey=True)
fig.subplots_adjust(left=.08,right=.98,top=.79,bottom=.22,wspace=.12,hspace=.32)
xx=[0,25,50,75,100]
series=[('active','active_legal',5,'L6：合法且未结束','#2675b9','-','o'),
        ('active','active_legal',0,'L1：合法且未结束','#327d70','-','^'),
        ('all','all',5,'L6：全部位置（含终止）','#e59b43','--','s'),
        ('active','active_legal',7,'起终点 embedding 对照','#8b8d94',':',None),
        ('active','active_legal',8,'长度等可观测特征对照','#a35494','-.',None)]
for row,source in enumerate(['reference','free']):
    for col,(panel,label) in enumerate(zip(['long_AA_test','long_AT_test','long_TT_test'],['长程 A→A','长程 A→T','长程 T→T'])):
        ax=axes[row,col]
        for family,population,head,legend,color,style,marker in series:
            vv=[mean_value(p,family,source,panel,population,head,'balanced_accuracy') for p in ratios]
            yy=[np.nan if v is None else 100*v for v in vv]
            ax.plot(xx,yy,label=legend,color=color,ls=style,marker=marker,ms=4,lw=1.8)
        ax.axhline(50,color='#aaaaaa',lw=.8)
        ax.set_ylim(40,102);ax.set_yticks([50,60,70,80,90,100]);ax.set_xticks(xx)
        ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
        ax.set_title(label+' · '+('合法参考轨迹' if row==0 else '模型自由生成'),fontsize=11)
        if col==0:ax.set_ylabel('答案探针 balanced accuracy（%）')
        if row==1:ax.set_xlabel('已生成 CoT 占整条轨迹长度（%）')
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.53,.88),ncol=3,frameon=False,fontsize=9)
fig.text(.04,.96,'CoT 进展中的查询答案可读性',fontsize=20,weight='bold',color='#17324d')
fig.text(.04,.91,f'v6 DFS nodes · N80 · final 50k · {len(models)}初始化均值 · 各进度独立线性探针，验证集选择阈值',fontsize=11,color='#536575')
fig.text(.04,.12,'蓝／绿线：训练、验证和测试均剔除已到目标、已穷尽或历史非法的位置；100%处参考搜索已结束，因此无点。\n'
         '橙线允许终止状态，但所有输入均排除 YES／NO／EOS。紫线若很高，表明比例定位下长度本身已强关联答案。',fontsize=10,color='#536575',linespacing=1.8)
fig.text(.04,.035,'参考 train/val 拟合后直接测试自由生成，无测试重训；0%是 prompt 末尾。比例进度由事后轨迹长度定位，另附固定token步数对照。\n'
         '这是外部解码器成绩，不是让原模型在中间提前输出答案的正确率；各进度的有效query支持不同，见报告。',fontsize=9,color='#697783',linespacing=1.8)
for ext in ['png','pdf']:fig.savefig(root/f'answer_progress.{ext}',dpi=200,facecolor='white')
plt.close(fig)

# Fixed-token progress is necessary to distinguish answer information from the
# normalized-progress length shortcut. The surviving query cohort still varies.
fig,axes=plt.subplots(2,3,figsize=(13.5,8.2),sharex=True,sharey=True)
fig.subplots_adjust(left=.08,right=.98,top=.78,bottom=.22,wspace=.12,hspace=.32)
steps=['r0','t1','t2','t4','t8','t16','t32']
fixed_series=[(5,'L6：合法且未结束','#2675b9','-','o'),
              (0,'L1：合法且未结束','#327d70','-','^'),
              (7,'起终点 embedding 对照','#8b8d94',':',None),
              (8,'可观测特征对照','#a35494','-.',None)]
for row,source in enumerate(['reference','free']):
    for col,(panel,label) in enumerate(zip(['long_AA_test','long_AT_test','long_TT_test'],['长程 A→A','长程 A→T','长程 T→T'])):
        ax=axes[row,col]
        for head,legend,color,style,marker in fixed_series:
            values=[mean_value(p,'active',source,panel,'active_legal',head,'balanced_accuracy') for p in steps]
            ax.plot(range(len(steps)),[np.nan if v is None else 100*v for v in values],label=legend,color=color,ls=style,marker=marker,lw=1.8,ms=4)
        ax.set_title(label+' · '+('合法参考轨迹' if row==0 else '模型自由生成'),fontsize=11)
        ax.set_ylim(30,102);ax.set_yticks([40,50,60,70,80,90,100]);ax.set_xticks(range(len(steps)),['0','1','2','4','8','16','32'])
        ax.axhline(50,color='#aaaaaa',lw=.8);ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
        if col==0:ax.set_ylabel('答案探针 balanced accuracy（%）')
        if row==1:ax.set_xlabel('已生成 CoT token 数（0为prompt末尾）')
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.53,.88),ncol=2,frameon=False,fontsize=10)
fig.text(.04,.96,'固定生成步数：答案可读性是否随 CoT 增强？',fontsize=19,weight='bold',color='#17324d')
fig.text(.04,.91,f'v6 DFS nodes · {len(models)}初始化均值 · 每步独立拟合、验证校准 · 同一步数比较正负query',fontsize=11,color='#536575')
fig.text(.04,.12,'仅保留图上合法且搜索尚未结束的位置；同一步数不再由未来轨迹长度定位。\n'
         '随着步数增加，已终止和非法轨迹退出；32步时部分面板支持很少，不能把曲线当固定query群体的因果变化。',fontsize=10,color='#536575',linespacing=1.8)
fig.text(.04,.035,'外部线性解码器，非原模型中途原生答案；参考 train/val 拟合后直接测试自由生成。\n'
         '只展示0至32步，详细正负query支持、accuracy、BA和两类recall见报告。',fontsize=9,color='#697783',linespacing=1.8)
for ext in ['png','pdf']:fig.savefig(root/f'answer_progress_fixed_tokens.{ext}',dpi=200,facecolor='white')
plt.close(fig)

def pct(v):return 'NA' if v is None else f'{v:.1%}'
lines=['# v6 DFS nodes：CoT生成进展与查询答案探针','',
       f'完成{len(models)}/4初始化，固定50k checkpoint。冻结推理物理GPU1，仿射探针训练物理GPU0。无导航训练、checkpoint、activation缓存或W&B。',
       '使用上一轮状态探针同一批 train256/val96，以及全部short/AA/TA/AT/TT测试query。训练查询仅短程和长AA，验证也仅短程/长AA；不使用T长程查询拟合。',
       '参考采用冻结bank全部8变体，自由生成仅测试。所有位置均为输入当前位置残差，读取后预测原始图可达标签；不输入最终答案/EOS。',
       '每phase/每层独立仿射头，250步上限，按val BCE选步；val BA选择阈值。随机标签按query整组打乱，不拆散同query变体。',
       '各query总权重相同，再按正负query分别计算recall，BA取两者平均。既报告accuracy也报告BA，防止类别比例变化造成错觉。',
       'all族包含终止位置；active族在训练、验证、测试均只保留图上合法且搜索尚未结束、目标尚未出现的状态。自由生成也用同一个参考拟合头，不在测试重训。',
       '每phase独立读出意味着曲线不是同一个固定解码器的演化，也不是原模型原生中途答案正确率。','',
       '## 比例进度：答案BA','',
       '|面板|上下文|进度|L6全部位置|L6未终止合法|L1未终止合法|query embedding未终止|',
       '|---|---|---|---:|---:|---:|---:|']
for panel in panels:
    for source in ['reference','free']:
        for phase in ratios:
            p=summary[panel]['phases'][phase][source];a=p['all/all'];b=p['active/active_legal']
            lines.append('|'+panel+'|'+source+'|'+phase+'|'+'|'.join(pct(v) for v in [a['L6']['balanced_accuracy'],b['L6']['balanced_accuracy'],b['L1']['balanced_accuracy'],b['query_embeddings']['balanced_accuracy']])+'|')
lines+=['','## 固定生成token数：未终止合法位置','',
        '|面板|上下文|token数|L6 accuracy|L6 BA|YES recall|NO recall|query embedding BA|每seed正/负query支持|',
        '|---|---|---|---:|---:|---:|---:|---:|---|']
for panel in panels:
    for source in ['reference','free']:
        for phase in absolute:
            b=summary[panel]['phases'][phase][source]['active/active_legal'];h=b['L6']
            support=[(s['n_positive_queries'],s['n_negative_queries']) for s in b['support']]
            lines.append('|'+panel+'|'+source+'|'+phase+'|'+'|'.join(pct(v) for v in [h['accuracy'],h['balanced_accuracy'],h['yes_recall'],h['no_recall'],b['query_embeddings']['balanced_accuracy']])+f'|{support}|')
lines+=['','## T→T比例进度：有效支持与终点泄露对照','',
        '|进度|上下文|未终止每seed正/负query|全位置observable对照BA|未见目标的全位置头BA|active随机标签L6 BA|',
        '|---|---|---|---:|---:|---:|']
for phase in ratios:
    for source in ['reference','free']:
        p=summary['long_TT_test']['phases'][phase];r=p[source]
        support=[(s['n_positive_queries'],s['n_negative_queries']) for s in r['active/active_legal']['support']]
        lines.append(f'|{phase}|{source}|{support}|'+pct(r['all/all']['observable_control']['balanced_accuracy'])+'|'+pct(r['all/no_target_yet']['L6']['balanced_accuracy'])+'|'+pct(p['shuffled_active_L6_BA'])+'|')
lines+=['','## 原模型最终答案（全部测试query，供区分口径）','']
for panel in panels:lines.append(f"- {panel}: BA {pct(summary[panel]['native_final_BA'])}")
lines+=['','## 核心观察','',
        '- 比例进度的中途BA有明显长度混杂：active状态下没有目标/终止标志，observable对照主要靠已生成token数及是否已经输出节点即可获得高分。',
        '- 固定步数active探针没有呈现稳定的单调增长；不能用本结果证明CoT逐步把最终答案写入当前位置。',
        '- 起终点embedding对照本身很强；它是接受query标签训练的外部仿射头，也可能拟合节点角色先验，不能直接解释为原模型已使用的答案。',
        '- 终点处的高可读性可由轨迹是否出现目标等特征解释；不是提前规划证据。']
lines+=['','## 限制','',
        '- 比例进度用完整轨迹长度事后定位，未来长度不作为探针输入；但样本定位仍可能引入长度/标签相关选择，因此必须结合固定token数及query embedding/observable对照。',
        '- active有效样本随进展减少，且非法或提前终止的轨迹会退出；后段曲线不是固定查询群体的因果改善。',
        '- 节点身份未新增留出，原主模型见过所有节点；这是固定图query泛化与长程端点角色划分。',
        '- 探针高于原生答案不等于模型本身已经算出了同一个答案；探针失败也不等于没有任何可用信息。',
        '- 只有一张80节点图、四初始化；不将多变体和多进度当独立模型。',
        '- 无额外因果干预，本实验不证明CoT导致能力提升。']
(root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Summarized',len(models),'models')

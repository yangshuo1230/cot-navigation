"""Two standalone figures for the final token-matched all-position grid."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

P=Path(__file__).resolve().parent
OUT=P/'figures';OUT.mkdir(exist_ok=True)
data=json.loads((P/'check_summary.json').read_text(encoding='utf-8'))
assert len(data)==12 and all(len(r['tests'])==5 for r in data)
fonts={f.name for f in font_manager.fontManager.ttflist}
font=next((f for f in ['Microsoft YaHei','Noto Sans CJK SC','SimHei'] if f in fonts),'DejaVu Sans')
plt.rcParams.update({'font.family':font,'font.size':11,'axes.unicode_minus':False,
    'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'path'})
tasks=['main-n32','main-n64','main','main-deep','main-hard']
labels=['简单图\nN32','简单图\nN64','简单图\nN100','窄深图\nN100','复杂图\nN100']
colors=['#487AAB','#E19D39','#BD5364']

for key,title,population,name in [
    ('positive','答案准确率','仅可达正样本；每任务每模型256个查询','answer_accuracy'),
    ('legal','CoT 合法率','正负样本合并；每任务每模型512个查询','cot_legality')]:
    fig,axes=plt.subplots(1,2,figsize=(13.8,6),sharey=True)
    fig.subplots_adjust(left=.065,right=.985,top=.75,bottom=.25,wspace=.12)
    fig.suptitle(f'全位置纠错训练｜{title}',x=.065,ha='left',y=.965,fontsize=20,fontweight='bold')
    fig.text(.065,.88,f'token预算匹配（约101.6亿token/模型） · {population}',fontsize=11,color='#526273')
    for ax,depth in zip(axes,[6,9]):
        x=np.arange(5)
        ax.set_title(f'{depth}层 Transformer',fontsize=14,pad=14)
        for j,(rate,color) in enumerate(zip([0,30,60],colors)):
            rows=sorted([r for r in data if r['run'].split('/')[1]==f'l{depth}_p{rate}'],key=lambda r:r['run'])
            assert len(rows)==2
            values=np.array([[100*r['tests'][task][key] for r in rows] for task in tasks])
            centers=x+(j-1)*.25
            ax.bar(centers,values.mean(1),width=.22,color=color,alpha=.9,label=f'{rate}% 错误轨迹',zorder=2)
            # Two actual seeds, no confidence interval inferred from n=2.
            for i in range(5):
                ax.plot([centers[i],centers[i]],[values[i].min(),values[i].max()],color='#26364A',lw=.9,zorder=3)
            for seed,marker in enumerate(['o','D']):
                ax.scatter(centers+(seed-.5)*.045,values[:,seed],s=23,marker=marker,
                    facecolor='white',edgecolor='#26364A',linewidth=.8,zorder=4)
        ax.set_xticks(x,labels);ax.set_ylim(0,106);ax.set_yticks(range(0,101,20))
        ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True);ax.tick_params(axis='x',length=0,pad=9)
    axes[0].set_ylabel(f'{title}（%）',labelpad=9)
    h,l=axes[0].get_legend_handles_labels()
    fig.legend(h,l,loc='lower center',bbox_to_anchor=(.52,.11),ncol=3,frameon=False)
    fig.text(.065,.055,'柱高：两个初始化seed均值；圆点/菱形：seed0/seed1；细线：两seed范围（不是置信区间）。',fontsize=9,color='#526273')
    if key=='legal':
        fig.text(.065,.015,'合法率要求完整、语义有效的CoT，允许 <MISTAKE> 取消错误并成功修复；与上图正样本准确率的分母不同。',fontsize=9,color='#526273')
    else:
        fig.text(.065,.015,'0%/30%/60%分别训练262,370 / 226,929 / 200,000步；逐课程阶段匹配预期token量。',fontsize=9,color='#526273')
    fig.savefig(OUT/f'{name}.png',dpi=220,bbox_inches='tight',facecolor='white')
    fig.savefig(OUT/f'{name}.svg',bbox_inches='tight',facecolor='white')
    plt.close(fig)
print(OUT)

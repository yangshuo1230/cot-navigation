"""Plot completed generalization scalars; no inference, training or network."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

P=Path(__file__).resolve().parent
NODES=[32,64,100,144,169,196]
ARMS=['matched_clean','correction']
COLORS=['#326B9B','#D76537']
LABELS=['Baseline','纠错训练（30%）']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',type=Path,default=P/'summary.json')
    parser.add_argument('--output',type=Path,default=P/'figures')
    args=parser.parse_args()
    data=json.loads(args.summary.read_text(encoding='utf-8'))
    assert data['complete_cells']==data['expected_cells']==96
    args.output.mkdir(parents=True,exist_ok=True)
    available={f.name for f in font_manager.fontManager.ttflist}
    font=next((f for f in ['Microsoft YaHei','Noto Sans CJK SC','SimHei'] if f in available),'DejaVu Sans')
    plt.rcParams.update({'font.family':font,'axes.unicode_minus':False,'font.size':11,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'path','savefig.facecolor':'white'})

    def values(family,arm,key):
        result=[]
        for n in NODES:
            rr=sorted([r for r in data['results'] if r['population']=='positive' and
                r['family']==family and r['n_nodes']==n and r['arm']==arm],key=lambda r:r['seed'])
            assert len(rr)==4 and [r['seed'] for r in rr]==list(range(4))
            result.append([100*r['metrics'][key] for r in rr])
        return np.array(result)

    def save(fig,name):
        fig.savefig(args.output/f'{name}.png',dpi=220,bbox_inches='tight')
        fig.savefig(args.output/f'{name}.svg',bbox_inches='tight')
        plt.close(fig)

    for key,title,ylabel,name in [('accuracy','答案准确率','答案准确率（%）','answer_accuracy'),
                                  ('legality','CoT 合法率','CoT 合法率（%）','cot_legality')]:
        fig,axes=plt.subplots(1,2,figsize=(12.5,5.5),sharey=True)
        fig.subplots_adjust(left=.075,right=.98,bottom=.22,top=.77,wspace=.12)
        fig.suptitle(f'DFS 图规模泛化｜{title}',x=.075,ha='left',y=.97,fontsize=19,fontweight='bold')
        fig.text(.075,.895,'6层模型 · baseline vs 30%纠错训练 · 仅可达正样本',fontsize=11,color='#526273')
        for ax,family,subtitle in zip(axes,['simple','complex'],['简单图｜相邻层 DAG，M=floor(1.5N)','复杂图｜长短边与汇合 DAG，M=2N']):
            x=np.arange(6)
            for arm,color,label in zip(ARMS,COLORS,LABELS):
                v=values(family,arm,key);mean=v.mean(1);sd=v.std(1,ddof=1)
                ax.fill_between(x,np.maximum(0,mean-sd),np.minimum(100,mean+sd),color=color,alpha=.12,lw=0)
                ax.plot(x,mean,'o-',color=color,lw=2.3,ms=5.5,label=label,zorder=3)
                for seed in range(4):
                    ax.scatter(x+(seed-1.5)*.035,v[:,seed],s=15,color=color,alpha=.48,zorder=2)
            ax.set_title(subtitle,fontsize=12,pad=13)
            ax.set_xticks(x,NODES);ax.set_xlabel('图节点数 N',labelpad=9)
            ax.set_ylim(0,103);ax.set_yticks(np.arange(0,101,20));ax.grid(axis='y',alpha=.17)
            ax.set_axisbelow(True)
        axes[0].set_ylabel(ylabel)
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.52,.075),ncol=2,frameon=False)
        fig.text(.075,.025,'实线：4个seed均值；阴影：±1个seed标准差；小点：各seed。每个设置256个固定正样本，所有模型共用。',fontsize=9,color='#526273')
        if key=='legality':
            fig.text(.075,-.015,'合法率允许 <MISTAKE> 取消错误记录并成功修复；预算截断不视为完整合法轨迹。',fontsize=9,color='#526273')
        save(fig,name)

    fig,axes=plt.subplots(1,2,figsize=(12.5,3.9))
    fig.subplots_adjust(left=.1,right=.9,bottom=.25,top=.68,wspace=.25)
    fig.suptitle('纠错训练相对 Baseline 的变化',x=.1,ha='left',y=.97,fontsize=18,fontweight='bold')
    fig.text(.1,.84,'仅可达正样本 · 四seed均值之差 · 单位：百分点（pp）',fontsize=11,color='#526273')
    arrays=[]
    for key in ['accuracy','legality']:
        arrays.append(np.stack([values(f,'correction',key).mean(1)-values(f,'matched_clean',key).mean(1) for f in ['simple','complex']]))
    limit=max(1,max(abs(a).max() for a in arrays))
    for ax,a,title in zip(axes,arrays,['答案准确率变化','CoT 合法率变化']):
        im=ax.imshow(a,cmap='RdBu',vmin=-limit,vmax=limit,aspect='auto')
        ax.set_title(title,pad=12);ax.set_xticks(range(6),NODES);ax.set_yticks([0,1],['简单图','复杂图'])
        ax.set_xlabel('图节点数 N',labelpad=9)
        for i in range(2):
            for j in range(6):ax.text(j,i,f'{a[i,j]:+.2f}',ha='center',va='center',color='white' if abs(a[i,j])>.62*limit else '#172D42',fontsize=11)
        for spine in ax.spines.values():spine.set_visible(False)
    cax=fig.add_axes([.925,.25,.014,.43]);fig.colorbar(im,cax=cax,label='百分点')
    fig.text(.1,.065,'蓝色表示纠错模型更高，红色表示更低。这是描述性均值差，未作显著性检验。',fontsize=9,color='#526273')
    save(fig,'correction_gain')
    print('Saved 3 figures (PNG + SVG) to',args.output)


if __name__=='__main__':main()

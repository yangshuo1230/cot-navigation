"""Standalone PNG/SVG figures from an immutable compact snapshot."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

GROUPS=['l6_p30','l6_p60','l6_p90','l9_p30','l9_p60','l9_p90']
LABELS=['6层\n30%','6层\n60%','6层\n90%','9层\n30%','9层\n60%','9层\n90%']
TASKS=['n32','n64','main','deep','hard']
NAMES=['N32 · 4×8 · M48','N64 · 8×8 · M96','普通N100 · 10×10 · M150','窄深N100 · 20×5 · M150','困难N100 · M200']
BLUE='#397a9d';ORANGE='#dc8b3d'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--snapshot',type=Path,required=True);ap.add_argument('--font',default='C:/Windows/Fonts/msyh.ttc');ap.add_argument('--allow-deprecated',action='store_true');a=ap.parse_args()
    font_manager.fontManager.addfont(a.font)
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=a.font).get_name(),'font.size':10,'axes.unicode_minus':False,'svg.fonttype':'path','axes.spines.top':False,'axes.spines.right':False})
    d=json.loads(a.snapshot.read_text(encoding='utf-8'));out=a.snapshot.parent/'figures';out.mkdir(exist_ok=True)
    deprecated=d.get('lifecycle')=='deprecated' or (a.snapshot.parent/'DEPRECATED.json').exists()
    if deprecated and not a.allow_deprecated:raise ValueError('Withdrawn cohort: only explicitly watermarked archival figures allowed')
    def save(fig,name):
        if deprecated:
            fig.text(.5,.52,'已废弃 · 不得引用',ha='center',va='center',rotation=15,fontsize=66,color='#c62828',alpha=.27)
            fig.text(.5,.995,'本批模型及派生结果已于2026-11-02撤回，仅供追溯',ha='center',va='top',fontsize=12,color='#c62828')
        fig.savefig(out/(name+'.png'),dpi=180,facecolor='white')
        fig.savefig(out/(name+'.svg'),facecolor='white');plt.close(fig)
    for metric,title,file in [('correct','回答准确率','01_answer_accuracy'),('valid','CoT语义合法率','02_cot_validity')]:
        fig,axes=plt.subplots(2,3,figsize=(16,8.8));fig.subplots_adjust(top=.84,bottom=.12,hspace=.42,wspace=.23)
        fig.suptitle(title+'：层数 × 错误数据比例 × 图设置',fontsize=19,y=.97)
        fig.text(.5,.91,'可达测试查询；每柱2个模型seed × 256查询。圆点为各seed。无同批0%干净对照。',ha='center',color='#475569')
        for task,name,ax in zip(TASKS,NAMES,axes.flat):
            x=np.arange(6);natural=[100*d['natural'][task][g][metric]/512 for g in GROUPS]
            ax.bar(x-.18,natural,.34,color=BLUE,label='自然生成')
            for i,g in enumerate(GROUPS):
                ax.text(i-.18,natural[i]+1.1,f'{natural[i]:.1f}',ha='center',fontsize=8,color=BLUE)
                values=d['natural'][task][g]['seed_accuracy' if metric=='correct' else 'seed_validity']
                ax.scatter([i-.23,i-.13],np.array(values)*100,s=13,c='#173b50',zorder=4)
                oracle=d['oracle'].get(task,{}).get(g)
                if oracle:
                    value=100*oracle[metric]/512;ax.bar(i+.18,value,.34,color=ORANGE)
                    ax.text(i+.18,value+1.1,f'{value:.1f}',ha='center',fontsize=8,color='#a65b15')
                    vals=oracle['seed_accuracy' if metric=='correct' else 'seed_validity'];ax.scatter([i+.13,i+.23],np.array(vals)*100,s=13,c='#7b4217',zorder=4)
                else:ax.text(i+.18,4,'待完成',rotation=90,va='bottom',ha='center',fontsize=7,color='#8793a4')
            ax.set_title(name,fontsize=12,pad=10);ax.set_xticks(x);ax.set_xticklabels(LABELS);ax.set_ylim(0,108);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel(title+'（%）');ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
        ax=axes.flat[-1];ax.axis('off')
        ax.bar(0,0,color=BLUE,label='自然生成：全部完成');ax.bar(0,0,color=ORANGE,label='强制MISTAKE：仅完整双seed配对')
        ax.legend(loc='upper left',frameon=False,fontsize=11)
        pairs=sum(len(v) for v in d['oracle'].values());ax.text(0,.64,f'干预配对覆盖：{pairs}/30组\n\n空白表示尚未完成，不表示0。\n所有结果来自固定快照，后续可重建。',transform=ax.transAxes,va='top',color='#475569')
        fig.text(.04,.035,'干预保留错误字段，只把错误动作替换为MISTAKE；无额外上下文预算。每种图独立统计。',fontsize=10,color='#475569')
        save(fig,file)
    fig,axes=plt.subplots(2,5,figsize=(22,9));fig.subplots_adjust(top=.84,bottom=.13,wspace=.27,hspace=.42)
    fig.suptitle('自然生成的首错位置分布',fontsize=20,y=.97)
    fig.text(.5,.91,'仅包含实际出现首错的轨迹；包括后来被MISTAKE取消的首错。箱体=25–75%，横线=中位数；须=1.5×IQR，离群点不画。',ha='center',fontsize=11)
    for col,(task,name) in enumerate(zip(TASKS,NAMES)):
        for row,key in enumerate(['first_tokens','first_steps']):
            ax=axes[row,col];values=[d['natural'][task][g][key] for g in GROUPS]
            boxes=ax.boxplot(values,patch_artist=True,showfliers=False,widths=.58,medianprops={'color':'#172b3a','linewidth':1.7})
            for i,b in enumerate(boxes['boxes']):b.set_facecolor(BLUE if i<3 else ORANGE);b.set_alpha(.65)
            labels=[f'{LABELS[i]}\nn={len(v)}' for i,v in enumerate(values)]
            ax.set_xticks(np.arange(1,7));ax.set_xticklabels(labels,fontsize=8);ax.set_yscale('log');ax.grid(axis='y',alpha=.18,which='both');ax.set_axisbelow(True)
            ax.set_title(name if row==0 else '首错前实际搜索步',fontsize=11)
            ax.set_ylabel('CoT token位置（对数轴）' if row==0 else '搜索步位置（对数轴）',fontsize=9)
    fig.text(.035,.035,'输入边列表不计入CoT位置。蓝色=6层，橙色=9层；各组箱体包含2个seed。无错轨迹不以最大长度填充。',fontsize=11,color='#475569')
    save(fig,'03_first_error_positions')
    metrics=[('raw_mistakes','MISTAKE原始出现次数'),('true_marks','成功标注的错误记录数'),('error_records','可核验错误记录总数'),('immediate_repairs','紧接下一条记录修复成功次数'),('repaired_whole_trajectories','修复后整条合法且答对的轨迹数'),('complete_correct','全部CoT合法且答对的轨迹数')]
    fig,axes=plt.subplots(2,3,figsize=(18,10));fig.subplots_adjust(top=.86,bottom=.12,hspace=.35,wspace=.30)
    fig.suptitle('自然生成：出错 → 标注 → 局部修复 → 整条成功',fontsize=20,y=.97)
    fig.text(.5,.915,'每格512个可达查询（2个seed）；数字为实际计数。每个面板独立色标，不能跨面板按颜色比较数量。',ha='center',fontsize=11)
    for ax,(key,title) in zip(axes.flat,metrics):
        values=np.array([[d['natural'][t][g][key] for g in GROUPS] for t in TASKS])
        im=ax.imshow(values,cmap='Blues',aspect='auto',vmin=0,vmax=max(1,values.max()))
        for i in range(5):
            for j in range(6):ax.text(j,i,str(values[i,j]),ha='center',va='center',color='white' if values[i,j]>values.max()*.55 else '#173b50',fontsize=11)
        ax.set_title(title,fontsize=12,pad=10);ax.set_xticks(range(6));ax.set_xticklabels(LABELS,fontsize=9);ax.set_yticks(range(5));ax.set_yticklabels(['N32','N64','普通N100','窄深N100','困难N100'],fontsize=10)
        fig.colorbar(im,ax=ax,fraction=.035,pad=.025)
    fig.text(.035,.045,'错误总数按完整记录计：同记录多种错误只计一次；非法动作后的不可核验后缀不计入错误分母。',fontsize=10,color='#475569')
    fig.text(.035,.02,'MISTAKE原始次数包含未核验标记；“全部完整轨迹正确”也包含从未出错的轨迹，不等于“修复后整条正确”。',fontsize=10,color='#475569')
    save(fig,'04_error_marker_repair_counts')
    with (out/'figure_data.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['graph','model','condition','queries','accuracy_pct','validity_pct']+[key for key,_ in metrics]+['lifecycle'])
        for condition in ['natural','oracle']:
            for task,gg in d[condition].items():
                for group,v in gg.items():writer.writerow([task,group,condition,v['n'],100*v['correct']/v['n'],100*v['valid']/v['n']]+[v.get(key,'') for key,_ in metrics]+['deprecated' if deprecated else 'active'])
    print('Wrote four PNG/SVG figures and figure_data.csv to',out)


if __name__=='__main__':main()

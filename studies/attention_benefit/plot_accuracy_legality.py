"""Compare complete free-navigation answers and effective legal DFS traces.

Recompute all counts from per-query retest outputs. Gray is a differently
trained reference; the blue/orange pair shares training graph family,
trajectory weight and context limit, but is not an initialization-matched
causal training control. No inference or training is launched.
"""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=Path(__file__).resolve().parent/'hard_retest')
    parser.add_argument('--out',type=Path,default=None,help='Output stem, without extension')
    args=parser.parse_args()
    stem=args.out or args.input_dir/'accuracy_cot_legality'
    stem.parent.mkdir(parents=True,exist_ok=True)
    models=['simple_clean','simple_error','hard_clean']
    data=[json.loads((args.input_dir/(m+'_navigation.json')).read_text(encoding='utf-8')) for m in models]
    ids=[r['query_id'] for r in data[0]['rows']]
    assert len(ids)==len(set(ids))==256
    assert all(d['finished'] and [r['query_id'] for r in d['rows']]==ids for d in data)
    assert all([r['label'] for r in d['rows']]==[r['label'] for r in data[0]['rows']] for d in data)
    n=len(ids)
    counts=np.array([[sum(r['correct'] for r in d['rows']),sum(r['effective_complete'] for r in d['rows'])] for d in data])
    assert all(not r['effective_complete'] or r['correct'] for d in data for r in d['rows'])

    available={f.name for f in font_manager.fontManager.ttflist}
    chosen=next((f for f in ['Microsoft YaHei','Noto Sans CJK SC','SimHei'] if f in available),None)
    if not chosen:raise RuntimeError('Install a Chinese font before rendering this Chinese figure')
    plt.rcParams.update({'font.family':chosen,'axes.unicode_minus':False,'svg.fonttype':'path',
                         'font.size':11,'axes.titlesize':15})
    fig,axes=plt.subplots(1,2,figsize=(13.6,6.2))
    fig.subplots_adjust(left=.065,right=.98,bottom=.27,top=.79,wspace=.16)
    colors=['#3077B5','#E98B2A','#A5ACB4']
    positions=np.array([0,1,2.35])
    labels=['Baseline（无纠错）\n简单图训练 · λ=16 · 4K',
            '加入 30% 纠错数据\n简单图训练 · λ=16 · 4K',
            '困难图训练 Baseline\n参考：λ=1 · 32K']
    for metric,ax in enumerate(axes):
        rates=counts[:,metric]/n*100
        ax.bar(positions,rates,width=.65,color=colors,zorder=3)
        for x,y,count in zip(positions,rates,counts[:,metric]):
            ax.text(x,y+2,f'{y:.2f}%\n{count}/{n}',ha='center',va='bottom',fontsize=11,linespacing=1.4)
        ax.set_xticks(positions,labels,fontsize=10)
        ax.set_xlim(-.65,3.0);ax.set_ylim(0,105);ax.set_yticks(range(0,101,20))
        ax.set_yticklabels([f'{v}%' for v in range(0,101,20)])
        ax.grid(axis='y',color='#DFE4EA',linewidth=.8,zorder=0)
        ax.spines[['top','right','left']].set_visible(False)
        ax.spines['bottom'].set_color('#CBD2DA');ax.tick_params(axis='both',length=0,pad=10)
        ax.axvline(1.73,color='#D5DAE0',linestyle='--',linewidth=1,zorder=1)
        ax.set_title(['最终回答准确率','完整 CoT 合法率'][metric],pad=20)
        delta=rates[1]-rates[0]
        if metric==1:
            y=max(rates[:2])+18
            ax.plot([0,0,1,1],[y-1.5,y,y,y-1.5],color='#525D6A',linewidth=1)
            ax.text(.5,y+2,f'纠错模型 +{delta:.2f} 个百分点',ha='center',color='#525D6A',fontsize=10)
        else:
            ax.text(.5,.99,f'纠错模型 +{delta:.2f} 个百分点',transform=ax.transAxes,
                    ha='center',color='#525D6A',fontsize=10)
    fig.suptitle('困难 DAG：回答准确率与完整 CoT 合法性',fontsize=20,y=.965,fontweight='bold')
    fig.text(.5,.885,'非严格分层 DAG · N=100，M=200 · 同一批 256 个测试查询 · 完整自由生成',ha='center',fontsize=11,color='#56616E')
    fig.text(.065,.145,'CoT 合法：整条状态与动作通过原图重放，并正确完成搜索；允许 MISTAKE 取消错误记录，未完成不计合法。',fontsize=10,color='#46515E')
    fig.text(.065,.095,'蓝 / 橙为原有简单图训练模型；灰色为困难图训练参考，训练设置不同。全部为单训练 seed、last 200k checkpoint。',fontsize=10,color='#46515E')
    fig.text(.065,.045,'最终答案正确不要求 CoT 完全正确；本图未展示跨训练 seed 的不确定性。',fontsize=9.5,color='#687481')
    for ext in ('png','svg'):fig.savefig(stem.with_suffix('.'+ext),dpi=200,facecolor='white')
    plt.close(fig)
    print('Verified counts:',dict(zip(models,counts.tolist())))
    print('Wrote:',stem.with_suffix('.png'),stem.with_suffix('.svg'))


if __name__=='__main__':main()

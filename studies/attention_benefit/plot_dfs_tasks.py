"""Two separate, same-task-axis DFS figures from complete batch16 audits.

Counts are recomputed from paired per-query free-generation outputs.
CoT validity uses state/action validity and a completed FOUND/EXHAUST,
independently of the final answer token. No inference runs from this script.
"""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

P=Path(__file__).resolve().parent
TASKS=[
    ('n16','N16\n4层 × 4宽\nM=24',0.),
    ('n24','N24\n4层 × 6宽\nM=36',1.),
    ('n32','N32\n4层 × 8宽\nM=48',2.),
    ('n36','N36\n6层 × 6宽\nM=54',3.),
    ('n64','N64\n8层 × 8宽\nM=96',4.),
    ('main','N100\n10层 × 10宽\nM=150',5.),
    ('deep','深窄 N100\n20层 × 5宽\nM=150',6.4),
    ('wide','宽浅 N100\n5层 × 20宽\nM=150',7.4),
    ('larger','N144\n12层 × 12宽\nM=216',8.8),
    ('hard_dag','困难 DAG N100\n非严格分层\nM=200',10.2),
]


def load_data():
    output=[]
    for case,label,x in TASKS:
        pair=[]
        for arm in ('clean','error'):
            if case in ('n16','n24','n32','n36'):
                path=P.parent/'accuracy_source/small'/f'dfs_{arm}_{case}.json'
            elif case=='main':
                path=P.parent/'accuracy_source/strict'/f'dfs_{arm}_main.json'
            elif case=='hard_dag':
                path=P/'hard_retest'/f'simple_{arm}_navigation.json'
            else:
                path=P/'task_comparison/batch16_audits'/f'dfs_{arm}_{case}.json'
            d=json.loads(path.read_text(encoding='utf-8'));rows=d['rows']
            assert d['batch_size']==16 and len(rows)==256 and len({r['query_id'] for r in rows})==256,path
            assert all(r.get('matches_saved',True) for r in rows),f'Answer drift: {path}'
            if 'finished' in d:assert d['finished'],path
            legal=[bool(r['effective_prefix_valid'] and r['terminal'] in ('FOUND','EXHAUST')) for r in rows]
            pair.append(dict(model=arm,n=len(rows),correct=sum(r['correct'] for r in rows),legal=sum(legal),
                legal_wrong_answer=sum(valid and not r['correct'] for valid,r in zip(legal,rows)),
                ids=[r['query_id'] for r in rows],labels=[r['label'] for r in rows],source=str(path),
                checkpoint=d.get('checkpoint',d.get('metadata',{}).get('checkpoint')),
                expected_checkpoint='full_l16/checkpoints/latest.pt' if arm=='clean' else 'p30_l16/checkpoints/latest.pt'))
            assert pair[-1]['checkpoint'].replace('\\','/').endswith(pair[-1]['expected_checkpoint']),path
        assert pair[0]['ids']==pair[1]['ids'] and pair[0]['labels']==pair[1]['labels'],case
        for item in pair:item.pop('ids');item.pop('labels')
        output.append(dict(task=case,label=label,x=x,models=pair))
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,default=P/'task_comparison')
    args=parser.parse_args();args.out_dir.mkdir(exist_ok=True,parents=True)
    data=load_data()
    fonts={f.name for f in font_manager.fontManager.ttflist}
    chosen=next((f for f in ('Microsoft YaHei','Noto Sans CJK SC','SimHei') if f in fonts),None)
    if not chosen:raise RuntimeError('A Chinese font is required')
    plt.rcParams.update({'font.family':chosen,'axes.unicode_minus':False,'svg.fonttype':'path'})
    positions=np.array([d['x'] for d in data]);colors=['#3077B5','#E98B2A']
    for metric,title,filename in [('correct','回答准确率','answer_accuracy_by_task'),
                                 ('legal','完整 CoT 合法率','cot_legality_by_task')]:
        fig,ax=plt.subplots(figsize=(19,7.2))
        fig.subplots_adjust(left=.058,right=.985,bottom=.30,top=.77)
        for mi,legend in enumerate(('Baseline：无纠错数据','加入 30% 纠错数据')):
            ys=np.array([d['models'][mi][metric]/d['models'][mi]['n']*100 for d in data])
            xs=positions+(mi-.5)*.35
            ax.bar(xs,ys,width=.32,color=colors[mi],label=legend,zorder=3)
            for x,y in zip(xs,ys):ax.text(x,y+1.3,f'{y:.2f}%',ha='center',va='bottom',fontsize=9)
        ax.set_ylim(0,112);ax.set_yticks(range(0,101,20));ax.set_yticklabels([f'{n}%' for n in range(0,101,20)],fontsize=11)
        ax.set_xlim(-.7,10.9);ax.set_xticks(positions,[d['label'] for d in data],fontsize=10)
        ax.tick_params(axis='both',length=0,pad=9)
        ax.grid(axis='y',color='#DEE4EA',linewidth=.8,zorder=0)
        ax.spines[['top','right','left']].set_visible(False);ax.spines['bottom'].set_color('#C8D0D9')
        for boundary in (5.7,8.1,9.5):ax.axvline(boundary,color='#CBD3DD',linestyle='--',linewidth=1,zorder=1)
        for a,b,text in [(-.5,5.5,'常规图规模'),(5.95,7.85,'图形状迁移'),(8.3,9.3,'规模外推'),(9.7,10.7,'图家族迁移')]:
            ax.text((a+b)/2,1.035,text,transform=ax.get_xaxis_transform(),ha='center',fontsize=11,color='#56616F')
        deltas=[]
        for x,d in zip(positions,data):
            diff=(d['models'][1][metric]-d['models'][0][metric])/256*100;deltas.append(diff)
            ax.text(x,-.315,f'{diff:+.2f}',transform=ax.get_xaxis_transform(),ha='center',fontsize=10,
                    color='#26734D' if diff>0 else '#B3473E' if diff<0 else '#65717E')
        fig.text(.018,.147,'Δ / pp',fontsize=9,color='#65717E')
        fig.suptitle('DFS：各图规模与形状的'+title,fontsize=22,fontweight='bold',y=.975)
        fig.text(.5,.91,'同一对简单分层图课程训练模型 · λ=16 · 4K 上下文 · last 200k · 每任务 256 个相同查询 · BF16 / batch16',ha='center',fontsize=12,color='#56616F')
        fig.legend(loc='upper center',bbox_to_anchor=(.5,.875),ncol=2,frameon=False,fontsize=12)
        if metric=='correct':
            definition='回答准确率：仅提供边列表和 QUERY，自由生成整条 CoT 后判断最终 YES / NO；中间过程出错也可能答对。'
        else:
            definition='完整 CoT 合法率：独立重放全部字段和动作，完成合法 FOUND / EXHAUST；允许 MISTAKE 作废错误记录，未完成搜索不计合法。'
        fig.text(.058,.09,definition,fontsize=10.5,color='#46515F')
        fig.text(.058,.04,'Δ 为纠错模型 − 无纠错基线（百分点）。三个迁移组均使用原有同一对 checkpoint；每个设置仅一个训练 seed。',fontsize=10.5,color='#65717E')
        for ext in ('png','svg'):fig.savefig(args.out_dir/(filename+'.'+ext),dpi=180,facecolor='white')
        plt.close(fig)
    (args.out_dir/'metrics.json').write_text(json.dumps(dict(tasks=data,
        definition='CoT validity is checked independently of final answer; complete effective semantic DFS with cancellations',
        n_per_task=256,batch_size=16,checkpoint='last200k'),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('Rendered two figures for',len(data),'paired DFS tasks.')
    for d in data:print(d['task'],[(m['correct'],m['legal']) for m in d['models']])


if __name__=='__main__':main()

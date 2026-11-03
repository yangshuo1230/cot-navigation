"""Position-only natural error curves, comparing two validated cohorts.

CPU-only aggregation of frozen record annotations: no model inference/refitting.
Incomplete records and suffixes after illegal actions remain censored exactly as
in the shared semantic audit. Position zero is the first CoT token, not prompt.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/analysis/correction_position_risk_comparison'
COHORTS=[('legacy6_s0123_supplement_20261102','native_supplement','旧批次 · 6层 · 4 seeds'),
         ('all_positions_tokenmatched_20261103','','新批次 · 6/9层 · 2 seeds')]
TASKS=['n32','n64','main','deep','hard']
EDGES=np.arange(0,4097,128)
COLORS={0:'#277DA1',30:'#E76F51',60:'#459B76'}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate():
    OUT.mkdir(parents=True,exist_ok=True);result=dict(version=1,bin_edges=EDGES.tolist(),groups=[],sources={},
        bootstrap='500 crossed query x model-seed resamples; percentile 95% CI',
        min_records_or_entered=20,min_distinct_queries=8,
        record_definition='complete scored records by starting CoT token; includes wrong cancelled records; censor after illegal action',
        token_definition='first-error events / actually observed error-free-prefix token decisions, including first error; exclude artificial budget STOP')
    task_identity={};total_files=0
    rng=np.random.default_rng(12037)
    for cohort,subdir,title in COHORTS:
        mp=ROOT/'studies/correction_behavior/cohorts'/f'{cohort}.json';manifest=json.loads(mp.read_text())
        assert manifest.get('lifecycle')!='deprecated';result['sources'][str(mp.relative_to(ROOT))]=sha(mp)
        base=ROOT/manifest['output']/subdir;summary_path=base/'summary.json'
        summary=json.loads(summary_path.read_text());assert summary['complete']
        result['sources'][str(summary_path.relative_to(ROOT))]=sha(summary_path)
        bygroup=defaultdict(list)
        for e in manifest['models']:bygroup[e['group']].append(e)
        for task in TASKS:
            for group,entries in sorted(bygroup.items()):
                ts=[]
                for e in entries:
                    path=base/'results'/f"{e['id']}__{task}.json.gz"
                    result['sources'][str(path.relative_to(ROOT))]=sha(path)
                    with gzip.open(path,'rt') as f:blob=json.load(f)
                    assert blob['complete'] and blob['identity']['model']['id']==e['id']
                    identity=blob['identity']['task_identity']
                    shared=(identity['fingerprint'],tuple(sorted(identity['queries'])))
                    if task in task_identity:assert shared==task_identity[task]
                    else:task_identity[task]=shared
                    assert len(blob['rows'])==256
                    ts.extend(dict(t,seed=e['seed']) for t in blob['rows']);total_files+=1
                qs=sorted({t['query'] for t in ts});seeds=sorted({t['seed'] for t in ts})
                qi={q:i for i,q in enumerate(qs)};si={s:i for i,s in enumerate(seeds)}
                shape=(len(qs),len(seeds),len(EDGES)-1)
                panels={k:dict(n=np.zeros(shape,np.int64),bad=np.zeros(shape,np.int64),entered=np.zeros(shape,np.int64))
                        for k in ['all_records','no_prior_error_records','first_error_token']}
                for t in ts:
                    q,s=qi[t['query']],si[t['seed']]
                    for r in t['records']:
                        assert 0<=r['token_start']<4096
                        b=int(r['token_start']//128)
                        for key in ['all_records']+(['no_prior_error_records'] if r['prior_bad_records']==0 else []):
                            panels[key]['n'][q,s,b]+=1;panels[key]['bad'][q,s,b]+=bool(r['bad']);panels[key]['entered'][q,s,b]=1
                    first=t['first']['token']-1 if t['first'] is not None else None
                    end=first+1 if first is not None else t['trace_tokens']-int(t['interrupted'])
                    assert 0<=end<=4096
                    for b,(lo,hi) in enumerate(zip(EDGES[:-1],EDGES[1:])):
                        panels['first_error_token']['n'][q,s,b]=max(0,min(end,int(hi))-int(lo))
                        panels['first_error_token']['entered'][q,s,b]=int(end>lo)
                        panels['first_error_token']['bad'][q,s,b]=int(first is not None and lo<=first<hi)
                expected=summary['tasks'][task][group]
                assert int(panels['all_records']['n'].sum())==expected['all']['n']
                assert int(panels['all_records']['bad'].sum())==expected['all']['bad']
                assert int(panels['no_prior_error_records']['n'].sum())==expected['by_history']['0']['n']
                assert int(panels['no_prior_error_records']['bad'].sum())==expected['by_history']['0']['bad']
                assert int(panels['first_error_token']['bad'].sum())==sum(h['events'] for h in expected['hazard'])
                assert int(panels['first_error_token']['n'].sum())==sum(h['exposure'] for h in expected['hazard'])
                qw=rng.multinomial(len(qs),np.full(len(qs),1/len(qs)),size=500)
                sw=rng.multinomial(len(seeds),np.full(len(seeds),1/len(seeds)),size=500)
                data={}
                for key,p in panels.items():
                    den=p['n'].sum((0,1));num=p['bad'].sum((0,1));entered=p['entered'].sum((0,1))
                    distinct=(p['entered'].sum(1)>0).sum(0)
                    bn=np.einsum('rq,qsb,rs->rb',qw,p['n'],sw,optimize=True)
                    by=np.einsum('rq,qsb,rs->rb',qw,p['bad'],sw,optimize=True)
                    values=[]
                    for b in range(len(den)):
                        rates=by[bn[:,b]>0,b]/bn[bn[:,b]>0,b]
                        ci=np.quantile(rates,[.025,.975]).tolist() if len(rates) else None
                        values.append(dict(lo=int(EDGES[b]),hi=int(EDGES[b+1]),denominator=int(den[b]),errors=int(num[b]),
                            model_queries=int(entered[b]),distinct_queries=int(distinct[b]),
                            rate=float(num[b]/den[b]) if den[b] else None,ci95=ci,
                            supported=bool((entered[b] if key=='first_error_token' else den[b])>=20 and distinct[b]>=8)))
                    data[key]=values
                result['groups'].append(dict(cohort=cohort,title=title,task=task,group=group,layers=entries[0]['layers'],
                    error_rate=entries[0]['error_rate'],seeds=seeds,trajectories=len(ts),panels=data))
    assert total_files==100
    result['audited_files']=total_files;result['matched_graph_query_sets']=True
    (OUT/'position_risk.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('AGGREGATED',total_files,'files;',len(result['groups']),'cohort/group/graph cells; all totals match frozen summaries')


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'font.size':10,'axes.unicode_minus':False,'svg.fonttype':'none','savefig.facecolor':'white'})
    d=json.loads((OUT/'position_risk.json').read_text(encoding='utf-8'))
    rows=[(COHORTS[0][0],6,'旧批次 · 6层 · 4 seeds'),(COHORTS[1][0],6,'新批次 · 6层 · 2 seeds'),(COHORTS[1][0],9,'新批次 · 9层 · 2 seeds')]
    tasknames=['N32 · 4×8 · M48','N64 · 8×8 · M96','N100 · 普通 · M150','N100 · 窄深 · M150','N100 · 困难 · M200']
    specs=[('all_records','01_record_error_by_position','仅按 CoT 位置：当前动作记录的出错概率','当前记录错误率',
            '不按既往错误分组；分母是起点落在该区间内的完整可评分动作记录。'),
           ('no_prior_error_records','02_clean_prefix_record_error','尚未出错时：当前位置首次出错的记录风险','当前记录错误率（进入记录前无错）',
            '仅要求进入当前记录前无错，保留当前首次出错记录；避免混入已经出错后的传播。'),
           ('first_error_token','03_first_error_token_hazard','按实际风险 token 数归一化：首错概率随位置变化','每个风险 token 的首错概率',
            '分母是实际观察到的首错前 token 决策数，包含首错 token；不是各位置首错次数的占比。')]
    for key,name,title,ylabel,definition in specs:
        fig,aa=plt.subplots(3,5,figsize=(20,12),sharex='col',sharey='col' if key=='first_error_token' else True)
        fig.subplots_adjust(top=.85,bottom=.19,left=.09,right=.98,hspace=.37,wspace=.25)
        for row,(cohort,depth,rowname) in enumerate(rows):
            for col,task in enumerate(TASKS):
                ax=aa[row,col]
                all_task=[g for g in d['groups'] if g['task']==task]
                limits=[v['hi'] for g in all_task for panel in g['panels'].values() for v in panel if v['supported']]
                xmax=max(512,int(np.ceil(max(limits)/256)*256))
                ymax=1
                if key=='first_error_token':
                    high=[v['ci95'][1] for g in all_task for v in g['panels'][key] if v['supported'] and v['ci95']]
                    ymax=max(.005,np.ceil(max(high)*1000)/1000*1.05)
                selected=[g for g in d['groups'] if g['cohort']==cohort and g['layers']==depth and g['task']==task]
                for g in sorted(selected,key=lambda g:g['error_rate']):
                    vals=g['panels'][key];c=COLORS[round(g['error_rate']*100)]
                    x=np.array([(v['lo']+v['hi'])/2 for v in vals]);valid=np.array([v['supported'] for v in vals])
                    y=np.array([v['rate'] if v['rate'] is not None else np.nan for v in vals]);yy=np.where(valid,y,np.nan)
                    ax.plot(x,yy,marker='o',ms=3,lw=1.8,color=c)
                    lo=np.array([v['ci95'][0] if v['ci95'] else np.nan for v in vals]);hi=np.array([v['ci95'][1] if v['ci95'] else np.nan for v in vals])
                    ax.fill_between(x,np.where(valid,lo,np.nan),np.where(valid,hi,np.nan),color=c,alpha=.10)
                ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
                ax.set_ylim(-.015*ymax,ymax*1.04);ax.set_xlim(0,xmax)
                ticks=np.linspace(0,xmax,5).astype(int);ax.set_xticks(ticks,[str(x) for x in ticks],fontsize=8)
                ax.tick_params(axis='y',labelsize=8)
                ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=1 if key=='first_error_token' else 0))
                if row==0:ax.set_title(tasknames[col],fontsize=11,fontweight='bold')
                if col==0:ax.set_ylabel(rowname+'\n'+ylabel,fontsize=10)
                if row==2:ax.set_xlabel('CoT token 位置（不含输入图/查询）',fontsize=9)
        fig.suptitle(title,fontsize=18,fontweight='bold',y=.98)
        fig.text(.5,.93,definition,ha='center',fontsize=11,color='#555555')
        handles=[Line2D([0],[0],color=c,marker='o',label='普通模型 0%' if r==0 else f'纠错模型 {r}%') for r,c in COLORS.items()]
        handles.append(Line2D([0],[0],ls=':',color='#999999',label='支持不足区间留空'))
        fig.legend(handles=handles,ncol=4,loc='lower center',bbox_to_anchor=(.5,.085),frameon=False,fontsize=11)
        fig.text(.5,.035,'每128 token一个区间；同一列共享横纵轴尺度，列间横轴及首错概率纵轴可不同。阴影为查询×seed交叉bootstrap 95%区间。\n'
            '连线要求≥20条记录/风险模型查询且≥8个独立查询；非法动作后停止严格评分，未完成记录和未生成后缀不当成正确。\n'
            '后段存在幸存/删失选择，曲线不是位置的因果效应。',ha='center',fontsize=9,color='#555555')
        fig.savefig(OUT/(name+'.png'),dpi=190);fig.savefig(OUT/(name+'.svg'));plt.close(fig)
        print('PLOT',name)
    lines=['# 两批实验：CoT 位置与自然出错概率',
        '旧批次6层0/30%，各4seed；新批次6/9层0/30/60%，各2seed。只含有效批次，可达正查询，五种图分别显示。',
        '100个已完成自然模型–图文件全部审计；两批使用相同图fingerprint和查询集合。无需新生成或重新训练。',
        '第一张图只按位置分桶，不按错误历史/状态/记录编号分组。每条完整记录按起点分桶；包含有错误但被MISTAKE取消的记录。',
        '第二张图只过滤进入记录前已经出错的情况，当前首错仍在分子/分母。第三张图按真实风险token数计算首错hazard。',
        '不要用首错发生次数的分布代替风险率：较晚位置仍被观察到的轨迹较少。也不要把三个指标的分母混用。',
        '阴影是500次查询和初始化seed交叉重采样的95%区间；新版仅两个seed，不足以证明初始化稳定。低支持区间留空，不显示不稳定点估计。',
        'position_risk.json保存每个桶的分子、分母、独立查询数、模型查询数、置信区间以及100个源文件哈希。']
    for _,name,title,_,_ in specs:lines+=['',f'## {title}',f'[PNG]({name}.png) · [SVG]({name}.svg)',f'![{title}]({name}.png)']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--aggregate-only',action='store_true');p.add_argument('--plot-only',action='store_true');a=p.parse_args()
    if not a.plot_only:aggregate()
    if not a.aggregate_only:plot()

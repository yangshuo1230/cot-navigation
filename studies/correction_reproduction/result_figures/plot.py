"""Reproducible compact figures for the validated legacy 6L 0%/30% cohort.

Read existing summaries only. No model inference, fitting, or result mutation.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/analysis/correction_summary_figures_legacy6'
BEHAVIOR=ROOT/'outputs/analysis/correction_behavior/legacy6_s0123_supplement_20261102'
STATE=ROOT/'outputs/analysis/state_readouts/legacy6_s0123_supplement_20261102'
COLORS={'l6_p0':'#277DA1','l6_p30':'#E76F51'}
NAMES={'l6_p0':'普通模型 0%','l6_p30':'纠错模型 30%'}
TASKS=['n32','n64','main','deep','hard']
TITLES=['N32 · 4×8 · M48','N64 · 8×8 · M96','N100 · 10×10 · M150','N100 · 20×5 · M150','N100 · 困难图 · M200']
SOURCES={};SERIES={}


def load(path):
    raw=path.read_bytes();SOURCES[str(path.relative_to(ROOT))]=hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def style():
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,
        'axes.titleweight':'bold','font.size':10,'axes.unicode_minus':False,
        'savefig.facecolor':'white','figure.facecolor':'white','svg.fonttype':'none'})


def axes(ax,xlabel=None,ylabel=None):
    ax.grid(axis='y',alpha=.16,zorder=0);ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    if xlabel:ax.set_xlabel(xlabel)
    if ylabel:ax.set_ylabel(ylabel)


def finish(fig,name,title,subtitle,footnote,legend=True):
    fig.suptitle(title,y=.985,fontsize=18,fontweight='bold')
    fig.text(.5,.90 if fig.get_figheight()<7 else .93,subtitle,ha='center',fontsize=10,color='#555555')
    if legend:
        handles,labels=fig.axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.075),ncol=3 if len(labels)==6 else min(4,len(labels)),frameon=False)
    fig.text(.5,.025,footnote,ha='center',va='bottom',fontsize=9,color='#555555')
    fig.savefig(OUT/(name+'.png'),dpi=210)
    fig.savefig(OUT/(name+'.svg'))
    plt.close(fig)


def forced():
    d=load(BEHAVIOR/'oracle_mistake/comparison.json');assert d['complete'] and d['completed_tasks']==40
    fig,aa=plt.subplots(1,3,figsize=(17,5.6));fig.subplots_adjust(top=.80,bottom=.28,wspace=.24)
    for ax,(metric,title) in zip(aa,[('correct','回答准确率'),('valid','完整 CoT 合法率'),('cut','生成预算耗尽率')]):
        for group in COLORS:
            for mode,ls,marker in [('native','-','o'),('forced','--','s')]:
                y=[d['groups'][t][group][mode+'_'+metric]/d['groups'][t][group]['n'] for t in TASKS]
                label=NAMES[group]+(' · 自然' if mode=='native' else ' · 强制 MISTAKE')
                ax.plot(range(5),y,ls=ls,marker=marker,color=COLORS[group],label=label,lw=2)
                SERIES.setdefault('01_forced',[]).append(dict(metric=metric,group=group,mode=mode,value=y,n=1024))
        ax.set_title(title);ax.set_xticks(range(5),['N32','N64','N100\n普通','N100\n窄深','N100\n困难']);axes(ax)
        ax.set_ylim(0,1 if metric!='cut' else .30)
    finish(fig,'01_forced_mistake','强制 MISTAKE：准确率、CoT 合法率与预算耗尽',
        '有效 6 层模型；0% / 30%，各 4 个种子；每种图设置每组 1,024 条模型–查询结果',
        '发现错误记录后替换动作，后续仍由模型自由生成；不提供正确修复。固定 4K 上下文。')


def persistence():
    d=load(BEHAVIOR/'native_supplement/summary.json');assert d['complete']
    fig,aa=plt.subplots(2,5,figsize=(20,9));fig.subplots_adjust(top=.85,bottom=.21,hspace=.52,wspace=.23)
    keys=['0','1','2','3-4','5+']
    for col,(task,title) in enumerate(zip(TASKS,TITLES)):
        supports={}
        for group in COLORS:
            g=d['tasks'][task][group];assert g['seeds']==[0,1,2,3]
            rows=[g['by_history'][k] for k in keys]
            y=np.array([r['bad']/r['n'] if r['n'] else np.nan for r in rows]);ci=[r.get('ci95') for r in rows]
            low=[max(0,v-c[0]) if c and np.isfinite(v) else 0 for v,c in zip(y,ci)]
            high=[max(0,c[1]-v) if c and np.isfinite(v) else 0 for v,c in zip(y,ci)]
            aa[0,col].errorbar(np.arange(5),y,yerr=[low,high],color=COLORS[group],marker='o',lw=1.8,capsize=2,label=NAMES[group])
            lags=g['first_error_lags'];rr=[]
            for lag in range(1,6):
                # The summary includes an 'all' row AND type-specific rows.
                # Use 'all' exactly once, rather than double-counting support.
                part=[r for r in lags if r['lag']==lag and r['first_type']=='all']
                assert len(part)==1
                n=sum(r['n'] for r in part);bad=sum(r['bad'] for r in part)
                rr.append(dict(lag=lag,n=n,bad=bad))
            yy=[r['bad']/r['n'] if r['n'] else np.nan for r in rr]
            aa[1,col].plot(range(1,6),yy,color=COLORS[group],marker='o',lw=1.8,label=NAMES[group])
            supports[group]=dict(history=[r['n'] for r in rows],lags=[r['n'] for r in rr])
            SERIES.setdefault('02_persistence',[]).append(dict(task=task,group=group,history=[dict(history=k,n=r['n'],bad=r['bad'],ci95=r.get('ci95')) for k,r in zip(keys,rows)],lags=rr))
        aa[0,col].set_title(title,fontsize=11)
        for row,kind,labels in [(0,'history',keys),(1,'lags',[str(i) for i in range(1,6)])]:
            ticks=range(5) if row==0 else range(1,6)
            counts=[f'{label}\n{supports["l6_p0"][kind][i]}\n{supports["l6_p30"][kind][i]}' for i,label in enumerate(labels)]
            aa[row,col].set_xticks(ticks,counts,fontsize=8)
        aa[0,col].set_xlabel('此前错误记录数');aa[1,col].set_xlabel('首错后第几条记录')
        for row in range(2):axes(aa[row,col]);aa[row,col].set_ylim(-.03,1.08)
    aa[0,0].set_ylabel('当前记录错误率');aa[1,0].set_ylabel('后续记录错误率')
    finish(fig,'02_error_persistence','错误持续性：错误累积与首错后的继续生成',
        '按图规模与形状分开；刻度下两行 n 依次为普通/纠错模型有效记录数；上排误差线为已有 95% 区间',
        '自然生成的关联统计，不能单独作因果解释。非法动作后停止严格评分；后续样本减少包含删失与提前结束。')


def error_probes():
    d=load(ROOT/'studies/correction_reproduction/natural_history_probes/early_offsets_summary.json');assert d['complete']
    fig,aa=plt.subplots(2,2,figsize=(13,9));fig.subplots_adjust(top=.85,bottom=.20,hspace=.36,wspace=.24)
    for row,arch in enumerate(['linear','mlp64']):
        for col,(dataset,subtitle) in enumerate([('test','共同自然前缀留出测试'),('exact_token_test','严格同 token 自然子集')]):
            ax=aa[row,col]
            for arm,group in [('matched_clean','l6_p0'),('correction','l6_p30')]:
                rr=[r for r in d['curves'] if r['architecture']==arch and r['dataset']==dataset and r['arm']==arm and r['token_offset'] in range(6) and r.get('min_positive_queries',0)>=8]
                arr=np.full((4,8),np.nan)
                for head in range(8):
                    for seed in range(4):
                        vals=[(p['tpr']+1-p['fpr'])/2 for r in rr if r['head']==head for p in r['per_model'] if p['model_seed']==seed]
                        if vals:arr[seed,head]=np.mean(vals)
                mean=np.nanmean(arr,0);std=np.nanstd(arr,0)
                ax.plot(range(1,7),mean[:6],marker='o',lw=2,color=COLORS[group],label=NAMES[group])
                ax.fill_between(range(1,7),mean[:6]-std[:6],mean[:6]+std[:6],color=COLORS[group],alpha=.12)
                ax.axhline(mean[6],color=COLORS[group],ls='--',alpha=.65,label=NAMES[group]+' · token/位置对照')
                SERIES.setdefault('03_error_probes',[]).append(dict(architecture=arch,dataset=dataset,group=group,mean=mean.tolist(),std=std.tolist(),per_model=arr.tolist()))
            ax.axhline(.5,color='#888888',ls=':',lw=1);ax.set_title(('线性' if arch=='linear' else 'MLP64')+' · '+subtitle)
            ax.set_xticks(range(1,7));axes(ax,'层','探针分类准确率');ax.set_ylim(.40,1.01)
    finish(fig,'03_error_history_probing','Probing：模型能否识别前序 CoT 已经出错？',
        'N100 普通图；首错 token 对齐 +0…+5 宏平均；固定阈值 0.5；阴影为 4 个模型种子的标准差',
        '已消费错误 token 后的历史错误探针，不是输出前的出错倾向探针。自然正负配对平衡，测试图/查询留出。')


def state_probes():
    d=load(STATE/'summary.json');assert d['complete']
    specs=[('field_node','accuracy','节点 token 的字段身份'),('field_bar','accuracy','相同 | token 的字段身份'),
        ('frontier','balanced_accuracy','已输出栈的成员集合'),('visited','balanced_accuracy','已输出 visited 的成员集合'),
        ('listed_candidates','balanced_accuracy','已输出候选的成员集合'),('successors_pre','balanced_accuracy','读入候选前：真实后继集合')]
    fig,aa=plt.subplots(2,3,figsize=(16,9));fig.subplots_adjust(top=.85,bottom=.24,hspace=.36,wspace=.24)
    for ax,(task,metric,title) in zip(aa.flat,specs):
        for group in COLORS:
            for arch,ls in [('mlp64','-'),('linear',':')]:
                r=next(r for r in d['probes'] if r['group']==group and r['task']==task and r['metric']==metric and r['architecture']==arch)
                y=np.array(r['mean']);sd=np.array(r['std'])
                ax.plot(range(1,7),y[:6],ls=ls,marker='o' if arch=='mlp64' else None,lw=2 if arch=='mlp64' else 1.5,
                    color=COLORS[group],label=NAMES[group]+(' · MLP64' if arch=='mlp64' else ' · 线性'))
                if arch=='mlp64':
                    ax.fill_between(range(1,7),y[:6]-sd[:6],y[:6]+sd[:6],color=COLORS[group],alpha=.10)
                    ax.axhline(y[7],color=COLORS[group],ls='--',alpha=.65,label=NAMES[group]+' · token/位置对照')
                SERIES.setdefault('04_state_probes',[]).append(r)
        ax.set_title(title);ax.set_xticks(range(1,7));axes(ax,'层','准确率' if task.startswith('field') else '集合成员平衡准确率')
        ax.set_ylim(.20 if task.startswith('field') else .40,1.02 if task.startswith('field') else .85)
    finish(fig,'04_field_state_probing','Probing：字段身份与搜索状态的可读出性',
        'N100 普通图；逐层独立探针；图/查询留出；集合阈值 0.5；阴影为 4 个模型种子的标准差',
        '集合成员指标同时检查正确和错误节点，不等于完整集合恢复率；栈只检测成员，不检测顺序。visited 的位置对照不可忽略。')


def raw_lens():
    d=load(STATE/'early_clean_lens/summary.json');assert d['complete']
    fig,aa=plt.subplots(1,3,figsize=(16,5.6));fig.subplots_adjust(top=.80,bottom=.28,wspace=.23)
    for ax,(task,title) in zip(aa,[('field_node','节点 token 位置'),('field_bar','字段分隔符位置'),('state','动作字段之前（第3个 |）')]):
        for group in COLORS:
            r=next(r for r in d['results'] if r['group']==group and r['task']==task and r['variant']=='raw')
            y=np.array(r['mean']);sd=np.array(r['std'])
            ax.plot(range(1,7),y,marker='o',lw=2,color=COLORS[group],label=NAMES[group])
            ax.fill_between(range(1,7),np.maximum(0,y-sd),np.minimum(1,y+sd),color=COLORS[group],alpha=.13)
            SERIES.setdefault('05_raw_lens',[]).append(r)
        ax.set_title(title);ax.set_xticks(range(1,7));axes(ax,'层','与最终层预测的一致率');ax.set_ylim(-.02,1.04)
    finish(fig,'05_basic_logit_lens','基础 logit lens：各层何时接近最终预测？',
        '原始残差 → 冻结 final norm → 冻结输出头；不训练校准映射；N100 普通图；阴影为种子标准差',
        '一致率不是 CoT 语义合法率。最终层与自身一致率恒为 100%；图中不把它解释为正确率。')


def semantic_lens():
    d=load(STATE/'legality_transitions/summary.json');assert d['complete'] and d['models']==8
    fig,aa=plt.subplots(1,3,figsize=(17,5.8));fig.subplots_adjust(top=.80,bottom=.28,wspace=.32)
    for ax,(task,title) in zip(aa[:2],[('all_unique_positions','全部唯一抽样位置'),('candidate_prediction','候选字段预测位置')]):
        for group in COLORS:
            for kind,ls in [('raw','--'),('calibrated','-')]:
                r=next(r for r in d['results'] if r['group']==group and r['task']==task and r['kind']==kind and r['population']=='all_legal_prefix')
                y=[(r['legal_legal']+r['legal_illegal'])/r['n'],(r['legal_legal']+r['illegal_legal'])/r['n']]
                ax.plot([5,6],y,marker='o',ls=ls,lw=2,color=COLORS[group],label=NAMES[group]+(' · 原始' if kind=='raw' else ' · 校准'))
                SERIES.setdefault('06_semantic_lens',[]).append(r)
        ax.set_title(title);ax.set_xticks([5,6],['第5层','最终第6层']);axes(ax,ylabel='下一 token 的 CoT 语义合法率');ax.set_ylim(0,1.05)
    ax=aa[2]
    xx=np.arange(2);width=.34
    for j,group in enumerate(COLORS):
        r=next(r for r in d['results'] if r['group']==group and r['task']=='candidate_prediction' and r['kind']=='calibrated' and r['population']=='all_legal_prefix')
        y=[r['illegal_to_legal_fraction'],r['legal_to_illegal_fraction']]
        bars=ax.bar(xx+(j-.5)*width,y,width,color=COLORS[group],label=NAMES[group])
        for bar,v in zip(bars,y):ax.text(bar.get_x()+bar.get_width()/2,v+.018,f'{100*v:.2f}%',ha='center',fontsize=10)
    ax.set_xticks(xx,['第5层非法\n→最终合法','第5层合法\n→最终非法']);ax.set_title('候选预测：校准 lens 的层间转变');axes(ax,ylabel='占本组全部位置');ax.set_ylim(0,.73)
    finish(fig,'06_semantic_legality_transitions','语义合法性：第5层到最终层的读出变化',
        'N100 普通图；保留即将产生首错的位置；此前已错的前缀排除；接受合法的不同候选顺序',
        '校准 lens 仅用训练/验证的早期正确位置拟合。层间读出转变不能直接证明网络主动执行了内部纠错。')


def revisit_persistence():
    d=load(BEHAVIOR/'native_supplement/summary.json')
    fig,aa=plt.subplots(1,3,figsize=(17,5.8));fig.subplots_adjust(top=.79,bottom=.28,wspace=.25)
    specs=[('same_extra','extra_returns','回溯后，同一个错误额外候选再次出现'),
        ('same_missing','missing_returns','回溯后，仍需要的同一后继继续被遗漏'),
        ('clean_then_bad','clean_returns','原本正确的候选列表，回访时变错')]
    xx=np.arange(5);width=.34
    for ax,(num,den,title) in zip(aa,specs):
        maxy=0
        for j,group in enumerate(COLORS):
            rr=[d['tasks'][t][group]['structure']['revisits'] for t in TASKS]
            y=[r[num]/r[den] if r.get(den,0) else np.nan for r in rr]
            valid=[v for v in y if np.isfinite(v)];maxy=max([maxy]+valid)
            bars=ax.bar(xx+(j-.5)*width,y,width,color=COLORS[group],label=NAMES[group])
            for bar,r,v in zip(bars,rr,y):
                text=f'{r.get(num,0)}/{r.get(den,0)}' if r.get(den,0) else 'NA'
                ax.annotate(text,(bar.get_x()+bar.get_width()/2,v if np.isfinite(v) else 0),
                    xytext=(0,5 if group=='l6_p0' else 20),textcoords='offset points',ha='center',fontsize=8,color=COLORS[group])
            SERIES.setdefault('07_revisits',[]).append(dict(group=group,numerator=num,denominator=den,
                counts=[dict(task=t,numerator=r.get(num,0),denominator=r.get(den,0)) for t,r in zip(TASKS,rr)]))
        ax.set_ylim(0,1.18 if maxy>.5 else max(.03,maxy*1.4))
        ax.set_xticks(xx,['N32','N64','N100\n普通','N100\n窄深','N100\n困难']);ax.set_title(title,fontsize=11);axes(ax,ylabel='占符合条件的回访事件')
    finish(fig,'07_backtrack_error_persistence','错误持续性：合法离开再回溯后，错误会不会重现？',
        '五种图分别统计；柱顶为事件数 / 有效回访数；分母为 0 显示 NA，不当作 0%',
        '只统计真实离开再回溯同一节点；MISTAKE 原地重试不算回溯。仅描述已发生的有效回访事件。')


def main():
    OUT.mkdir(parents=True,exist_ok=True);style()
    for fn in [forced,persistence,error_probes,state_probes,raw_lens,semantic_lens,revisit_persistence]:
        fn();print('WROTE',fn.__name__,flush=True)
    (OUT/'plot_data.json').write_text(json.dumps(dict(cohort='legacy6_s0123_supplement_20261102',source_sha256=SOURCES,series=SERIES),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# 有效旧版 6 层模型：结果图',
        '普通模型 0% 与纠错模型 30%，各 seed0–3；不含已废弃的 6/9 层 × 30/60/90% 批次。',
        '图 1–2、7 覆盖五种图；图 3–6 只覆盖 N100 普通图。全部主数据为可达正样本。',
        '每张图附 PNG 与 SVG，plot_data.json 固定输入文件 SHA256 和绘图数值。无需训练或重新推理。']
    for p in sorted(OUT.glob('*.png')):lines+=['',f'## {p.stem}',f'[PNG]({p.name}) · [SVG]({p.stem}.svg)',f'![{p.stem}]({p.name})']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()

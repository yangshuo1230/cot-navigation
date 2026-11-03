"""Compact figures for arbitrary manifest groups; no checkpoint inference."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib import font_manager

# 本地中文字体优先；Linux 可通过 COT_PLOT_FONT 指定同一字体。
import os
_font=Path(os.environ.get('COT_PLOT_FONT','C:/Windows/Fonts/msyh.ttc'))
if _font.exists():
    font_manager.fontManager.addfont(str(_font))
    plt.rcParams['font.family']=font_manager.FontProperties(fname=str(_font)).get_name()
plt.rcParams.update({'axes.unicode_minus':False,'svg.fonttype':'none'})

ROOT=Path(__file__).resolve().parents[2]
COLORS={0:'#277DA1',30:'#E76F51',60:'#459B76'}
TASKS=['n32','n64','main','deep','hard'];LABELS=['N32','N64','N100\n普通','N100\n窄深','N100\n困难']


def read(p):return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}


def label(group):return '错误掺入 '+group.split('_p',1)[1]+'%'


def color(entry):return COLORS[round(entry['error_rate']*100)]


def fmt(ax):
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
    ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_ylim(0,1.04)


def save(fig,out,name,title,note):
    fig.suptitle(title,fontsize=16,fontweight='bold',y=.99)
    fig.text(.5,.025,note,ha='center',fontsize=9,color='#555555')
    h,l=fig.axes[0].get_legend_handles_labels()
    if h:fig.legend(h,l,loc='lower center',bbox_to_anchor=(.5,.065),ncol=min(6,len(l)),frameon=False)
    fig.savefig(out/(name+'.png'),dpi=200);fig.savefig(out/(name+'.svg'));plt.close(fig)


def history_curve(d,arch,dataset):
    byhead=defaultdict(list)
    for run in d['runs']:
        if run['architecture']!=arch:continue
        for r in run.get('tests',{}).get(dataset,[]):byhead[r['head']].append(r['accuracy'])
    return [float(np.mean(byhead[i])) if byhead[i] else np.nan for i in range(d['model']['layers']+2)]


def plot(manifest,behavior,oracle,states,histories,out):
    out.mkdir(exist_ok=True,parents=True)
    entries={m['group']:m for m in manifest['models']};depths=sorted({m['layers'] for m in manifest['models']})
    groups={depth:sorted([g for g,e in entries.items() if e['layers']==depth],key=lambda g:entries[g]['error_rate']) for depth in depths}
    fig,aa=plt.subplots(len(depths),2,figsize=(13,8),squeeze=False);fig.subplots_adjust(top=.87,bottom=.22,hspace=.36)
    for row,depth in enumerate(depths):
        for col,(metric,title) in enumerate([('correct','回答准确率'),('valid','整条 CoT 合法率')]):
            ax=aa[row,col]
            for g in groups[depth]:
                if oracle.get('complete'):
                    for mode,ls in [('native','-'),('forced','--')]:
                        yy=[oracle['groups'].get(t,{}).get(g,{}) for t in TASKS]
                        y=[r.get(mode+'_'+metric,0)/r['n'] if r.get('n') else np.nan for r in yy]
                        ax.plot(range(5),y,ls=ls,marker='o',color=color(entries[g]),label=label(g)+(' 自然生成' if mode=='native' else ' 强制标记'))
                else:
                    yy=[behavior.get('tasks',{}).get(t,{}).get(g,{}) for t in TASKS]
                    y=[r.get(metric,0)/r['n'] if r.get('n') else np.nan for r in yy]
                    ax.plot(range(5),y,marker='o',color=color(entries[g]),label=label(g))
            ax.set_xticks(range(5),LABELS);ax.set_title(f'{depth}层 / {title}');fmt(ax)
    if oracle.get('complete'):
        save(fig,out,'01_forced_mistake','强制 MISTAKE 与自然生成对比','仅可达正样本；同层数对照；每组两个初始化种子；4K 上下文；不提供正确修复。')
    else:
        save(fig,out,'01_natural_accuracy_validity','自然生成：回答准确率与整条 CoT 合法率','仅可达正样本；同层数对照；每组两个初始化种子；4K 上下文；暂不做强制 MISTAKE。')
    if behavior.get('complete'):
        fig,aa=plt.subplots(len(depths),5,figsize=(19,8),squeeze=False);fig.subplots_adjust(top=.87,bottom=.22,hspace=.40,wspace=.25)
        bins=['0','1','2','3-4','5+']
        for row,depth in enumerate(depths):
            for col,task in enumerate(TASKS):
                ax=aa[row,col]
                for g in groups[depth]:
                    rs=behavior.get('tasks',{}).get(task,{}).get(g,{}).get('by_history',{})
                    yy=[rs.get(b,{}) for b in bins];y=[r['bad']/r['n'] if r.get('n') else np.nan for r in yy]
                    ax.plot(range(5),y,marker='o',color=color(entries[g]),label=label(g))
                ax.set_title(f'{depth}层 / {LABELS[col].replace(chr(10)," ")}');ax.set_xticks(range(5),bins);fmt(ax)
                ax.set_xlabel('前序错误记录数')
                if col==0:ax.set_ylabel('当前记录出错率')
        save(fig,out,'02_error_persistence','自然生成：前序错误数量与后续出错率','观察性统计，不代表因果；非法动作后停止严格评分；完整计数与不确定性保留在行为汇总中。')
        fig,aa=plt.subplots(len(depths),2,figsize=(13,8),squeeze=False);fig.subplots_adjust(top=.87,bottom=.22,hspace=.35)
        for row,depth in enumerate(depths):
            for col,(numer,denom,title) in enumerate([('same_extra','extra_returns','未使用的错误候选再次出现'),('same_missing','missing_returns','仍需访问的后继再次遗漏')]):
                ax=aa[row,col]
                for g in groups[depth]:
                    rr=[behavior.get('tasks',{}).get(t,{}).get(g,{}).get('structure',{}).get('revisits',{}) for t in TASKS]
                    y=[r.get(numer,0)/r[denom] if r.get(denom) else np.nan for r in rr]
                    ax.plot(range(5),y,marker='o',color=color(entries[g]),label=label(g))
                ax.set_title(f'{depth}层：{title}',fontsize=11);ax.set_xticks(range(5),LABELS);fmt(ax);ax.set_ylabel('重复出现比例')
        save(fig,out,'03_backtrack_persistence','自然生成：合法回溯到同一节点时的错误持续性','仅统计实际回溯；排除 MISTAKE 原地重试；无分母处留空，不记为零。')
    fig,aa=plt.subplots(len(depths),2,figsize=(13,8),squeeze=False);fig.subplots_adjust(top=.87,bottom=.22,hspace=.35)
    for row,depth in enumerate(depths):
        for col,arch in enumerate(['linear','mlp64']):
            ax=aa[row,col]
            for g in groups[depth]:
                dd=[d for d in histories if d['model']['group']==g and d['alignment']=='error_token_0_5']
                arrays=[history_curve(d,arch,'test') for d in dd]
                if not arrays:continue
                arr=np.array(arrays);mean=np.nanmean(arr,0);sd=np.nanstd(arr,0);x=np.arange(1,depth+1)
                ax.plot(x,mean[:depth],marker='o',color=color(entries[g]),label=label(g))
                ax.fill_between(x,mean[:depth]-sd[:depth],mean[:depth]+sd[:depth],color=color(entries[g]),alpha=.12)
                ax.axhline(mean[depth],color=color(entries[g]),ls='--',alpha=.5)
            ax.set_title(f'{depth}层 / '+{'linear':'线性探针','mlp64':'MLP64 探针'}[arch]);ax.set_xticks(range(1,depth+1));fmt(ax);ax.set_ylim(.4,1.02);ax.set_xlabel('层数');ax.set_ylabel('阈值 0.5 的预测准确率')
    save(fig,out,'04_history_probing','自然历史错误探针：留出共同前缀测试','首错偏移 +0～+5 平均；已读入错误 token；虚线为 token/位置对照；阴影为两个模型种子的标准差。')
    specs=[('field_node','accuracy','节点 token 的字段身份'),('visited','balanced_accuracy','已访问集合成员'),
           ('listed_candidates','balanced_accuracy','候选集合成员'),('successors_pre','balanced_accuracy','候选输出前的真实后继')]
    fig,aa=plt.subplots(len(depths),4,figsize=(18,8),squeeze=False);fig.subplots_adjust(top=.87,bottom=.22,hspace=.35,wspace=.28)
    for row,depth in enumerate(depths):
        for col,(task,metric,title) in enumerate(specs):
            ax=aa[row,col]
            for g in groups[depth]:
                dd=[d for d in states if d['model']['group']==g];arr=[]
                for d in dd:
                    r=next(p for p in d['probes'] if p['task']==task and p['architecture']=='mlp64')
                    arr.append([h[metric] for h in r['test']])
                if not arr:continue
                arr=np.array(arr);y=arr.mean(0);sd=arr.std(0);x=np.arange(1,depth+1)
                ax.plot(x,y[:depth],marker='o',color=color(entries[g]),label=label(g))
                ax.fill_between(x,y[:depth]-sd[:depth],y[:depth]+sd[:depth],color=color(entries[g]),alpha=.1)
                ax.axhline(y[depth+1],color=color(entries[g]),ls='--',alpha=.5)
            ax.set_title(f'{depth}层：{title}',fontsize=11);ax.set_xticks(range(1,depth+1));ax.set_xlabel('层数');ax.set_ylabel('准确率' if col==0 else '平衡准确率');fmt(ax);ax.set_ylim(.2 if col==0 else .4,1.03 if col==0 else .9)
    save(fig,out,'05_field_state_probing','逐层独立 MLP64：字段身份与搜索状态探针','普通 N100；自身自然测试窗口；虚线为 token/位置对照；集合阈值 0.5；线性结果保留在 JSON。')
    fig,aa=plt.subplots(len(depths),2,figsize=(13,8),squeeze=False);fig.subplots_adjust(top=.87,bottom=.22,hspace=.35)
    for row,depth in enumerate(depths):
        for col,kind in enumerate(['raw','calibrated']):
            ax=aa[row,col]
            for g in groups[depth]:
                arr=[[r['final_agreement'] for r in d['logit_lens'] if r['task']=='field_node' and r['kind']==kind] for d in states if d['model']['group']==g]
                if not arr:continue
                arr=np.array(arr);y=arr.mean(0);sd=arr.std(0);x=np.arange(1,depth+1)
                ax.plot(x,y,marker='o',color=color(entries[g]),label=label(g));ax.fill_between(x,np.maximum(0,y-sd),np.minimum(1,y+sd),color=color(entries[g]),alpha=.12)
            ax.set_title(f'{depth}层 / '+{'raw':'原始读出','calibrated':'早期正确位置校准'}[kind]);ax.set_xticks(range(1,depth+1));fmt(ax);ax.set_ylabel('与最终层预测的一致率');ax.set_xlabel('层数')
    save(fig,out,'06_logit_lens','逐层词表读出：原始与早期正确位置校准','普通 N100 节点输入位置；一致率不等于 CoT 合法率；校准仅使用与测试图隔离的训练/验证位置。')
    fig,aa=plt.subplots(len(depths),2,figsize=(13,8),squeeze=False);fig.subplots_adjust(top=.87,bottom=.22,hspace=.35)
    for row,depth in enumerate(depths):
        for col,task in enumerate(['all_unique_positions','candidate_prediction']):
            ax=aa[row,col]
            for g in groups[depth]:
                dd=[d for d in states if d['model']['group']==g]
                if not dd:continue
                for kind,ls in [('raw','--'),('calibrated','-')]:
                    stages=sorted({5,depth-1});ys=[]
                    for stage in stages:
                        rr=[r for d in dd for r in d['semantic_transitions'][str(stage)]['transitions'] if r['task']==task and r['population']=='all_legal_prefix' and r['kind']==kind]
                        n=sum(r['n'] for r in rr);ys.append(sum(r['legal_legal']+r['legal_illegal'] for r in rr)/n if n else np.nan)
                    n=sum(r['n'] for r in rr);ys.append(sum(r['legal_legal']+r['illegal_legal'] for r in rr)/n if n else np.nan)
                    ax.plot(stages+[depth],ys,marker='o',ls=ls,color=color(entries[g]),label=label(g)+(' 原始' if kind=='raw' else ' 校准'))
            ax.set_title(f'{depth}层 / { {'all_unique_positions':'全部去重位置','candidate_prediction':'候选预测位置'}[task]}',fontsize=11);ax.set_xticks(sorted({5,depth-1,depth}));fmt(ax);ax.set_ylabel('下一 token 的语义合法率')
    save(fig,out,'07_semantic_legality','深层读出的下一 token 语义合法率','仅读入前缀合法的位置，包含首错预测位置；接受其他合法 token 顺序；不等于整条生成轨迹合法率。')


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);a=p.parse_args()
    import os
    os.chdir(ROOT);manifest=read(a.manifest);base=Path(manifest['output'])
    out=ROOT/'outputs/analysis/correction_mechanisms'/manifest['cohort']
    behavior=read(base/'summary.json');oracle=read(base/'oracle_mistake/comparison.json')
    states=[read(out/'state'/(m['id']+'.json')) for m in manifest['models']];states=[d for d in states if d]
    histories=[read(out/'history'/(m['id']+'__'+alignment+'.json')) for m in manifest['models'] for alignment in ['error_token_0_5','record_0_2']];histories=[d for d in histories if d]
    choices=read(out/'execution_choices.json');natural_skipped=choices.get('skip_natural',False);forced_skipped=choices.get('skip_forced',False)
    complete=bool((natural_skipped or behavior.get('complete')) and (forced_skipped or oracle.get('complete')) and len(states)==len(manifest['models']) and len(histories)==2*len(manifest['models']))
    summary=dict(complete=complete,cohort=manifest['cohort'],behavior_tasks=behavior.get('complete_tasks',0),expected_behavior_tasks=len(manifest['models'])*5,
        state_models=len(states),history_jobs=len(histories),oracle_tasks=oracle.get('completed_tasks',0),natural_analysis_skipped=natural_skipped,forced_mistake_deferred=forced_skipped,
        models=[dict(id=m['id'],layers=m['layers'],error_rate=m['error_rate'],checkpoint_step=m['checkpoint_step']) for m in manifest['models']],
        warning='Two model seeds per group. Missing unsupported probe cells are not zero. Same-depth controls only.')
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    plot(manifest,{} if natural_skipped else behavior,{} if forced_skipped else oracle,states,histories,out/'figures')
    markers=read(out/'marker_counts.json')
    if markers and not natural_skipped:
        from marker_summary import plot as plot_markers
        assert markers['complete'] and markers['cohort']==manifest['cohort']
        plot_markers(markers,manifest,out/'figures')
    lines=['# 全位置错误池 v2：6/9层机制实验',f'完整状态：{complete}；自然行为 {summary["behavior_tasks"]}/60；强制MISTAKE {summary["oracle_tasks"]}/60；状态/lens {len(states)}/12；错误探针 {len(histories)}/24。',
        '同深度内比较0/30/60%，各两个初始化种子；按token曝光匹配的最终checkpoint。禁止混入废弃或等步数旧批次。',
        '自然行为与强制MISTAKE覆盖五种图；探针和lens覆盖普通N100可达正查询。自然错误探针有共同前缀、自身错误、同token与随机标签对照。',
        'logit lens最终一致率不等于CoT合法率；语义合法性独立重放，不把自然错误token作为正确目标。语义转变不是主动纠错的因果证据。',
        '原始JSON保留每seed/位置支持量，基础行为保留长度调整、删失、首错、历史计数、回溯复现等全部模块。']
    if natural_skipped:lines.insert(2,'用户已取消后续自然行为分析；现有自然结果仅为未完成的历史快照，不补跑、不生成自然持续性图，也不计作本次完成要求。')
    if forced_skipped:lines.insert(2,'强制MISTAKE按用户要求暂缓，不运行、不生成干预图；本轮完成标准是探针/lens、完整自然分析和结果图。')
    titles=['回答准确率与 CoT 合法率','前序错误与后续出错率','回溯后的错误持续性',
            '自然历史错误探针','字段身份与搜索状态探针','逐层词表读出',
            '深层语义合法率','自然 MISTAKE 标注与修复','原始 MISTAKE 全量与可核验计数']
    if markers and not natural_skipped:
        lines+=['','MISTAKE 图的前五种图设置使用可评分完整记录计数；非法动作后的后缀和不完整记录无法判定标对或修复，不算作无错误。',
                '只有普通 N100 保留了完整原始 token，可按轨迹哈希核验原始 MISTAKE 总量；其他图的原始总量未知。',
                '下一条修复：标对后紧接的记录合法、无错误且不再取消。整条修复：存在上述修复且最终整条 CoT 合法并答对，按轨迹去重。',
                '[完整计数（含误报、每种子与原始计数覆盖）](marker_counts.json)']
    for f in sorted((out/'figures').glob('*.png')):
        index=int(f.name.split('_',1)[0])-1
        title=titles[index] if 0<=index<len(titles) else f.stem
        lines+=['',f'## {title}',f'[PNG](figures/{f.name}) · [SVG](figures/{f.stem}.svg)',f'![{title}](figures/{f.name})']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    assert complete,'Full pipeline must not finish with incomplete analyses'
    print('COMPLETE',summary,flush=True)


if __name__=='__main__':main()

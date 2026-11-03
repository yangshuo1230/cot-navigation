"""CPU-only summaries of naturally occurring error histories, graph by graph."""
from collections import Counter,defaultdict
import gzip
import html
import json
from pathlib import Path

import numpy as np

P=Path(__file__).resolve().parent
BINS=['0','1','2','3-4','5+']
NAMES={'n32':'N32 / 4×8 / M48','n64':'N64 / 8×8 / M96',
       'main':'N100 / 10×10 / M150','deep':'窄深N100 / 20×5 / M150',
       'hard':'困难N100 / M200'}
ARMS=['matched_clean','correction']


def counts(rows):
    n=len(rows)
    return dict(n=n,queries=len({(r['seed'],r['query']) for r in rows}),
        bad=sum(r['bad'] for r in rows),
        annotation_bad=sum(r['annotation_bad'] for r in rows),
        action_bad=sum(r['action_bad'] for r in rows),
        illegal_action=sum(not r['action_legal'] and not r['cancel'] for r in rows),
        true_cancel=sum(r['true_cancel'] for r in rows),
        false_cancel=sum(r['false_cancel'] for r in rows),
        recovered=sum(r['recovered_after_error'] for r in rows),
        unmarked_recovery=sum(r['unmarked_recovery'] for r in rows),
        bad_nodes=sum(r['bad_node_count'] for r in rows),
        inherited_bad_nodes=sum(r['inherited_bad_nodes'] for r in rows),
        illegal_push=sum(r['illegal_push'] for r in rows),
        illegal_push_from_candidates=sum(r['illegal_push_from_current_candidates'] for r in rows),
        error_types=dict(sum((Counter(r['errors']) for r in rows),Counter())),
        bad_rate=sum(r['bad'] for r in rows)/n if n else None,
        action_bad_rate=sum(r['action_bad'] for r in rows)/n if n else None)


def agebin(n):return '1-2' if n<=2 else '3-4' if n<=4 else '5-8' if n<=8 else '9-16' if n<=16 else '17+'


def history_bin(n):return str(n) if n<3 else '3-4' if n<5 else '5+'


def matched_history(rows):
    """Descriptive common-support comparison, never labelled a causal effect.

    Within same seed, query distance, record age, visited size, depth, at-goal.
    Weight overlap cells by harmonic support; report coverage, not extrapolate.
    """
    cells=defaultdict(lambda:[[],[]])
    for r in rows:
        key=(r['seed'],r['distance'],agebin(r['record']),agebin(r['visited_size']),agebin(r['search_depth']),r['at_goal'])
        cells[key][int(r['prior_bad_records']>0)].append(r)
    used=[]
    for key,(clean,bad) in cells.items():
        if min(len(clean),len(bad))<3:continue
        if min(len({r['query'] for r in clean}),len({r['query'] for r in bad}))<2:continue
        weight=len(clean)*len(bad)/(len(clean)+len(bad))
        used.append((weight,np.mean([r['bad'] for r in clean]),np.mean([r['bad'] for r in bad]),len(clean),len(bad)))
    den=sum(x[0] for x in used)
    return dict(cells=len(used),clean_records=sum(x[3] for x in used),post_records=sum(x[4] for x in used),
        clean_rate=sum(w*a for w,a,b,_,_ in used)/den if den else None,
        post_rate=sum(w*b for w,a,b,_,_ in used)/den if den else None,
        delta=sum(w*(b-a) for w,a,b,_,_ in used)/den if den else None,
        causal=False)


def transitions(trajectories):
    rows=defaultdict(list);denominators=Counter();lost=defaultdict(Counter)
    for trajectory in trajectories:
        rr=trajectory['records'];first=next((i for i,r in enumerate(rr) if r['bad']),None)
        if first is None:continue
        kind=next(iter(rr[first]['errors']))
        for lag in range(1,6):
            for typ in ['all',kind]:
                denominators[typ,lag]+=1
            if first+lag<len(rr):
                r=rr[first+lag]
                rows[('all',lag)].append(r);rows[(kind,lag)].append(r)
            else:
                reason='illegal_action' if trajectory['stopped_on_illegal_action'] else 'budget' if trajectory['interrupted'] else 'other_end'
                for typ in ['all',kind]:lost[typ,lag][reason]+=1
    return [dict(first_type=k,lag=l,initial_error_trajectories=denominators[k,l],
                 unavailable=dict(lost[k,l]),**counts(rows[k,l])) for k,l in sorted(denominators)]


def bootstrap(rows,key,metric,seed=731):
    """Crossed resampling of shared queries and model seeds; record-weighted rate."""
    selected=[r for r in rows if r['prior_bin']==key]
    if not selected:return None
    queries=sorted({r['query'] for r in rows});qi={q:i for i,q in enumerate(queries)}
    n=np.zeros((len(queries),4));y=n.copy()
    for r in selected:
        n[qi[r['query']],r['seed']]+=1;y[qi[r['query']],r['seed']]+=r[metric]
    rng=np.random.default_rng(seed);rates=[]
    for _ in range(1000):
        qs=rng.integers(len(queries),size=len(queries));ss=rng.integers(4,size=4)
        den=n[np.ix_(qs,ss)].sum()
        if den:rates.append(y[np.ix_(qs,ss)].sum()/den)
    return np.quantile(rates,[.025,.975]).tolist() if rates else None


def summarize():
    summary=dict(complete=False,schema=1,tasks={},population='positive reachable official test queries',
        history_unit='earlier erroneous complete records; current record excluded',
        scoring='true-graph state follows legal actions, ignores printed state; censor after illegal action',
        interpretation='observational natural continuations, not causal identification')
    files=list((P/'results').glob('*.json.gz'))
    grouped=defaultdict(list)
    for file in files:
        with gzip.open(file,'rt') as f:data=json.load(f)
        grouped[(data['identity']['task'],data['identity']['arm'])].append(data)
    for task in NAMES:
        summary['tasks'][task]=dict(name=NAMES[task],groups={})
        complete_query_sets=[{r['query'] for r in b['rows']} for arm in ARMS for b in grouped[task,arm] if b['complete']]
        assert not complete_query_sets or all(q==complete_query_sets[0] for q in complete_query_sets)
        for arm in ARMS:
            blobs=grouped[task,arm]
            if not blobs:continue
            ts=[];rr=[];seeds=[]
            for b in blobs:
                seed=b['identity']['model_seed'];seeds.append(seed)
                for tr in b['rows']:
                    rs=[dict(r,seed=seed,query=tr['query'],distance=tr['distance']) for r in tr['records']]
                    event_count=0;prior_types=Counter()
                    for r in rs:
                        r['prior_error_events']=event_count
                        r['event_bin']=str(event_count) if event_count<3 else '3-4' if event_count<5 else '5+'
                        r['prior_type_counts']=dict(prior_types)
                        event_count+=r['bad_token_events']
                        prior_types.update(r['errors'].keys())
                    ts.append(dict(tr,records=rs,seed=seed));rr.extend(rs)
            bins={k:counts([r for r in rr if r['prior_bin']==k]) for k in BINS}
            for k in BINS:
                bins[k]['bad_rate_ci95']=bootstrap(rr,k,'bad')
            strata=[]
            for k in BINS:
                for typ in sorted({r['first_type'] for r in rr if r['prior_bin']==k}):
                    strata.append(dict(prior=k,first_type=typ,**counts([r for r in rr if r['prior_bin']==k and r['first_type']==typ])))
            type_strata=[]
            for typ in sorted({typ for r in rr for typ in r['prior_type_counts']}):
                for k in BINS:
                    subset=[r for r in rr if history_bin(r['prior_type_counts'].get(typ,0))==k]
                    type_strata.append(dict(prior=k,first_type=typ,**counts(subset)))
            summary['tasks'][task]['groups'][arm]=dict(complete=len(blobs)==4 and all(b['complete'] for b in blobs),
                seeds=sorted(seeds),trajectories=len(ts),unique_queries=len({tr['query'] for tr in ts}),
                correct=sum(tr['correct'] for tr in ts),trace_valid=sum(tr['trace_valid'] for tr in ts),
                error_trajectories=sum(any(r['bad'] for r in tr['records']) for tr in ts),
                first_error_in_partial_record=sum(bool(tr['first']) and tr['first']['record']>len(tr['records']) for tr in ts),
                no_post_first_record=sum(bool(tr['records']) and tr['records'][-1]['bad'] and sum(r['bad'] for r in tr['records'])==1 for tr in ts),
                illegal_action_censored=sum(tr['stopped_on_illegal_action'] for tr in ts),
                partial_censored=sum(tr['trailing_partial'] for tr in ts),
                interrupted=sum(tr['interrupted'] for tr in ts),
                numerical_pred_changes=sum(tr['pred']!=tr['historical_pred'] for tr in ts),
                all=counts(rr),by_history=bins,by_first_type_history=strata,
                by_type_specific_count=type_strata,
                by_history_excluding_goal={k:counts([r for r in rr if r['prior_bin']==k and not r['at_goal']]) for k in BINS},
                by_event_history={k:counts([r for r in rr if r['event_bin']==k]) for k in BINS},
                by_current_candidate={k:counts([r for r in rr if any(e.startswith('candidate_') for e in r['errors'])==bad]) for k,bad in [('correct',False),('incorrect',True)]},
                by_unresolved={k:counts([r for r in rr if r['unresolved_bin']==k]) for k in BINS},
                by_seed={s:{k:counts([r for r in rr if r['seed']==s and r['prior_bin']==k]) for k in BINS} for s in seeds},
                common_support=matched_history(rr),first_error_lags=transitions(ts))
    summary['complete']=all(len(t['groups'])==2 and all(g['complete'] for g in t['groups'].values()) for t in summary['tasks'].values())
    (P/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    return summary


def pct(x):return 'NA' if x is None else f'{100*x:.1f}%'


def render(s):
    text=['# 自然CoT错误之后的继续生成','',
        '**完成状态：'+('全部40个模型-图任务完成。' if s['complete'] else '部分结果，不能作为4seed最终结论。')+'**','',
        '只分析可达查询；baseline=matched_clean。错误次数按此前错误完整记录数计，当前记录不计入历史。',
        '逐条重建真实图上的执行状态，允许候选/visited任意合法顺序；字段写错不会修改参考状态。',
        '首个非法动作后停止严格语义评分，未完成记录不当成正确。自然MISTAKE不推进搜索状态。',
        '这是自然轨迹关联分析，无法单独识别前文错误的因果影响；不同模型的错误子集不同。',
        '固定上下文4096，所有样本原生贪心生成，不编辑token、不强制纠错；4个模型seed。','']
    for task,t in s['tasks'].items():
        text+=['## '+t['name'],'','|模型|此前错误记录数|后续记录数|模型-查询数|当前记录错误率|95%区间|动作错误率|','|---|---:|---:|---:|---:|---|---:|']
        for arm,g in t['groups'].items():
            for k,r in g['by_history'].items():
                ci=r['bad_rate_ci95'];c='NA' if ci is None else '–'.join(pct(v) for v in ci)
                text.append(f"|{arm}|{k}|{r['n']}|{r['queries']}|{pct(r['bad_rate'])}|{c}|{pct(r['action_bad_rate'])}|")
        text+=['','共同支持分层比较：按模型seed、查询距离、记录进度、visited大小、栈深度和是否已达终点分层，仅比较有两种历史的单元；仍非因果估计。','']
        for arm,g in t['groups'].items():
            m=g['common_support'];a=g['all'];p=g['by_history']
            delta='NA' if m['delta'] is None else f"{100*m['delta']:+.1f} pp"
            text.append(f"- {arm}：seed={g['seeds']}，{g['trajectories']}条轨迹，{g['unique_queries']}个不同查询；{g['illegal_action_censored']}条因非法动作删失，{g['interrupted']}条预算中断。共同支持{m['cells']}个单元，后错−无错风险差{delta}，覆盖{m['post_records']}条后错记录。")
            text.append(f"  非法PUSH {a['illegal_push']}次，其中{a['illegal_push_from_candidates']}次推入当前错误候选列表中的节点。")
        text+=['']
    text+=['## 解释限制','','- 后错率按记录加权，不是最终答案准确率；置信区间对共享查询和4个模型seed交叉重采样。',
        '- 错误类型统计允许同记录多种错误；遗漏按字段事件计数，不把一个错误列表等同于多个独立错误。',
        '- 一条错误字段记录仍可能执行合法动作；只有这种可继续核验的路径进入后续风险集。非法动作后不再编造真状态。',
        '- 全部历史与未取消错误历史分别统计；MISTAKE取消当前记录但不会抹掉历史上曾发生错误的事实。',
        '- 大量已错样本会终止或被删失，高错误次数组是幸存子集；记录数、删失率和各seed结果应与比率一起看。',
        '- 错误节点再次出现、或从错误候选PUSH，只是传播兼容现象；不能排除共享困难或独立重复错误。',
        '- 只保留紧凑记录级标注，不保存激活、checkpoint、完整token轨迹，不修改训练代码。','']
    (P/'REPORT.md').write_text('\n'.join(text),encoding='utf-8')
    payload=json.dumps(s,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
    template=(P/'dashboard_template.html').read_text(encoding='utf-8')
    (P/'error_continuation_dashboard.html').write_text(template.replace('__DATA__',payload),encoding='utf-8')


if __name__=='__main__':
    data=summarize();render(data);print('REPORT',data['complete'],flush=True)

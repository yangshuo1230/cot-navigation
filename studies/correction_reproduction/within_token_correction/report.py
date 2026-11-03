"""Query/model-paired summaries for before-emission correction, no rank metrics."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(key,'2')
from collections import defaultdict
import gzip
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

P=Path(__file__).resolve().parent;OUT=P/'current'


def avg(values):
    values=[x for x in values if x is not None]
    return float(np.mean(values)) if values else None


def summarize(blobs,selection,verification):
    probes=defaultdict(list);transitions=defaultdict(list);groups=defaultdict(list);null_index={}
    calibration=[]
    for d in blobs:
        calibration.append(dict(arm=d['arm'],seed=d['model_seed'],**d['calibration']))
        for r in d['probes']+d['shuffle_probes']:
            key=(d['arm'],r['target'],r['lens'],r['layer'],r['architecture'],r['control'],r['population'])
            probes[key].append(dict(r,seed=d['model_seed']))
        for r in d['transitions']:
            for pop in r['populations']:
                for layer in range(6):
                    key=(d['arm'],r['lens'],pop,r['requires_node'],layer+1)
                    transitions[key].append(dict(seed=d['model_seed'],query=r['query'],
                        legal=r['legal'][layer],node_legal=r['node_legal'][layer],is_node=r['provisional_is_node'][layer],
                        final=r['actual_legal'],wrong_node=r['provisional_is_node'][layer] and not r['legal'][layer],
                        spoiled=r['provisional_is_node'][layer] and r['legal'][layer] and not r['actual_legal']))
        for r in d['interventions']:
            r=dict(r,arm=d['arm'],seed=d['model_seed'])
            cohorts=['all']
            if r['tentative_is_node'] and not r['tentative_legal']:cohorts.append('wrong_node')
            if not r['tentative_node_legal']:cohorts.append('wrong_conditional_node')
            if not r['correct_before']:cohorts.append('future_error')
            if r['tentative_is_node'] and not r['tentative_legal'] and r['correct_before']:cohorts.append('sham_corrected')
            for pop in r['populations']:
                for cohort in cohorts:
                    key=(r['arm'],r['probe_target'],r['lens'],r['layer'],r['mode'],r['condition'],pop,cohort)
                    groups[key].append(r)
            null_index[d['arm'],d['model_seed'],r['query'],r['prefix_hash'],r['probe_target'],r['lens'],r['layer'],r['mode'],r['condition']]=r
    probe_rows=[]
    for key,rr in sorted(probes.items()):
        row=dict(zip(['arm','target','lens','layer','architecture','control','population'],key))
        row.update({m:avg([r.get(m) for r in rr]) for m in ['accuracy','balanced_accuracy','tpr','tnr','majority_accuracy']})
        row.update(models=len(rr),min_positive=min(r['positive'] for r in rr),min_negative=min(r['negative'] for r in rr),
            min_queries=min(r['queries'] for r in rr),threshold=.5,per_model=rr)
        probe_rows.append(row)
    transition_rows=[]
    for key,rr in sorted(transitions.items()):
        per_model=[]
        for seed in sorted({r['seed'] for r in rr}):
            vv=[r for r in rr if r['seed']==seed];wrong=[r for r in vv if r['wrong_node']]
            per_model.append(dict(seed=seed,n=len(vv),queries=len({r['query'] for r in vv}),
                provisional_legal=avg([r['legal'] for r in vv]),conditional_node_legal=avg([r['node_legal'] for r in vv]),
                node_tendency_fraction=avg([r['is_node'] for r in vv]),final_legal=avg([r['final'] for r in vv]),
                wrong_node_count=len(wrong),corrected_count=sum(r['final'] for r in wrong),
                repair_rate=avg([r['final'] for r in wrong]),spoiled_count=sum(r['spoiled'] for r in vv)))
        row=dict(zip(['arm','lens','population','requires_node','layer'],key))
        row.update({m:avg([r[m] for r in per_model]) for m in ['provisional_legal','conditional_node_legal','node_tendency_fraction','final_legal','repair_rate']})
        row.update(models=len(per_model),per_model=per_model)
        transition_rows.append(row)
    estimates=[]
    def metric(r,name,after):
        if name=='accuracy':return float(r['correct_after' if after else 'correct_before'])
        if name=='legal_mass':return r['legal_mass_after' if after else 'legal_mass_before']
        if name=='mlp_signal':return r['mlp_after' if after else 'mlp_before']
        if name=='linear_signal':return r['linear_after' if after else 'linear_before']
        return r[name] if after else 0.
    for number,(key,rr) in enumerate(sorted(groups.items())):
        arm,target,lens,layer,mode,condition,pop,cohort=key
        seeds=sorted({r['seed'] for r in rr});queries=sorted({r['query'] for r in rr})
        q_index={q:i for i,q in enumerate(queries)};s_index={s:i for i,s in enumerate(seeds)}
        rng=np.random.default_rng(30261101+number)
        qw=rng.multinomial(len(queries),np.full(len(queries),1/len(queries)),size=1000).astype(np.float32)
        sw=rng.multinomial(len(seeds),np.full(len(seeds),1/len(seeds)),size=1000).astype(np.float32)
        min_queries=min(len({r['query'] for r in rr if r['seed']==s}) for s in seeds)
        def interval(matrix):
            if min_queries<8:return None
            count=qw@np.isfinite(matrix).T
            sampled=np.divide(qw@np.nan_to_num(matrix).T,count,out=np.zeros_like(count),where=count>0)
            valid=count>0;boots=(sampled*sw).sum(1)/(sw*valid).sum(1).clip(1)
            return np.quantile(boots,[.025,.975]).tolist()
        for name in ['accuracy','legal_mass','mlp_signal','linear_signal','relative_norm','backtrack']:
            values=np.full((len(seeds),len(queries)),np.nan);base=values.copy();null=values.copy()
            by_query=defaultdict(list)
            for r in rr:by_query[r['seed'],r['query']].append(r)
            for (seed,q),rows in by_query.items():
                i=s_index[seed];j=q_index[q]
                values[i,j]=avg([metric(r,name,True) for r in rows]);base[i,j]=avg([metric(r,name,False) for r in rows])
                if condition in ['legal','error']:
                    matched=[null_index[arm,seed,r['query'],r['prefix_hash'],target,lens,layer,mode,'random_'+condition] for r in rows]
                    null[i,j]=avg([metric(r,name,True) for r in matched])
            means=np.nanmean(values,1);bases=np.nanmean(base,1);delta=values-base
            row=dict(zip(['arm','target','lens','layer','mode','condition','population','cohort'],key))
            row.update(metric=name,models=len(seeds),min_queries=min_queries,max_queries=max(len({r['query'] for r in rr if r['seed']==s}) for s in seeds),
                positions=len(rr),mean=float(means.mean()),baseline=float(bases.mean()),delta=float(np.nanmean(delta,1).mean()),
                paired_ci95=interval(delta) if name=='accuracy' else None,
                per_model=[dict(seed=s,mean=float(means[i]),baseline=float(bases[i]),delta=float(np.nanmean(delta[i]))) for i,s in enumerate(seeds)])
            if condition in ['legal','error']:
                adjusted=values-null
                row.update(null_adjusted_delta=float(np.nanmean(adjusted,1).mean()),
                    null_adjusted_ci95=interval(adjusted) if name=='accuracy' else None)
            estimates.append(row)
    return dict(complete=True,models=len(blobs),counts=selection['counts'],probes=probe_rows,transitions=transition_rows,
        estimates=estimates,calibration=calibration,verification=verification,
        total_interventions=sum(len(d['interventions']) for d in blobs),seconds=sum(d['seconds'] for d in blobs),
        target_definitions=dict(prospective='Will the actual final next token be illegal? Same target at all layers, before emission.',
            provisional='Is the node actually preferred by this layer readout illegal? Missing class support is NA.'),
        aggregation='query mean, then model-seed mean; 1000 crossed paired query/model bootstrap replicates',
        primary_endpoint='Unconstrained actual final next-token argmax belongs to the true legal-next-token set')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render-only',action='store_true',help='Rebuild presentation from the verified saved summary')
    args=parser.parse_args()
    if args.render_only:
        data=json.loads((OUT/'summary.json').read_text())
        assert data['complete'] and data['verification']['passed']
        render(data);return
    files=sorted((OUT/'results').glob('*.gz'));assert len(files)==8
    blobs=[json.load(gzip.open(f,'rt')) for f in files]
    assert all(d['complete'] and d['checkpoint_unchanged'] for d in blobs)
    verification=json.loads((OUT/'verification.json').read_text());assert verification['passed']
    selection=json.loads((OUT/'selection.json').read_text());data=summarize(blobs,selection,verification)
    text=json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False)
    (OUT/'summary.json').write_text(text,encoding='utf-8')
    render(data)
    print('WITHIN TOKEN REPORT COMPLETE',len(data['estimates']),data['total_interventions'],flush=True)


def render(data):
    text=json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False)
    template=(P/'dashboard_template.html').read_text(encoding='utf-8')
    html=template.replace('__WITHIN_TOKEN_DATA__',text.replace('<','\\u003c').replace('&','\\u0026'))
    (P/'within_token_dashboard.html').write_text(html,encoding='utf-8')
    # Preserve bookmarked entries without duplicating the embedded data three times.
    redirect='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>已更新为当前节点输出前实验</title><p>结果已更新：<a href="../within_token_correction/within_token_dashboard.html">打开当前节点输出前的探针与层间修正</a></p><script>location.replace('../within_token_correction/within_token_dashboard.html'+location.hash)</script></html>'''
    (P.parent/'causal_error_signal/causal_intervention_dashboard.html').write_text(redirect,encoding='utf-8')
    (P.parent/'natural_history_probes/natural_probe_dashboard.html').write_text(redirect,encoding='utf-8')
    write_report(data)
    print('WITHIN TOKEN PAGES RENDERED',flush=True)


def write_report(d):
    def get(arm,layer,condition,metric,mode='preserve_tendency',target='provisional'):
        return next((r for r in d['estimates'] if r['arm']==arm and r['layer']==layer and r['lens']=='raw' and
            r['condition']==condition and r['metric']==metric and r['mode']==mode and r['target']==target and
            r['population']=='main' and r['cohort']=='all'),None)
    def pp(x):return f'{x*100:+.2f} pp' if x is not None else 'NA'
    lines=['# 当前 token 输出前的层间修正', '', '[交互结果](within_token_dashboard.html)', '',
        '读取 h[p−1]，预测尚未输入的 x[p]。全部输入前缀合法且不含 MISTAKE；本轮终点是实际最终输出层的当前 token 合法率。', '',
        '## 主要目标：该层临时候选是否合法', '',
        '原输出头读出、仍需节点、该层确实倾向某个节点。阈值0.5；合法/非法训练验证类不足时为NA，不用辅助前瞻分数替代。', '',
        '| 模型 | 层 | MLP平衡准确率 | 可用模型 |', '|---|---|---:|---:|']
    for arm in ['matched_clean','correction']:
        for layer in [3,4,5,6]:
            r=next((r for r in d['probes'] if r['arm']==arm and r['target']=='provisional' and r['lens']=='raw' and
                r['layer']==layer and r['architecture']=='mlp64' and not r['control'] and r['population']=='main'),None)
            lines.append(f"| {arm} | L{layer} | {r['balanced_accuracy']*100:.2f}% | {r['models']}/4 |" if r else f'| {arm} | L{layer} | NA | 0/4 |')
    lines+=['', '## 辅助目标：预测本次实际最终输出是否会错', '',
        '各模型标签由其真实最终当前token决定，跨模型错误样本可能不同；这里的差值只作描述性比较，不能替代主要临时候选目标。', '',
        '| 模型 | 层 | 线性平衡准确率 | MLP平衡准确率 |', '|---|---|---:|---:|']
    for arm in ['matched_clean','correction']:
        for layer in [3,4,5,6]:
            rr=[next((r for r in d['probes'] if r['arm']==arm and r['target']=='prospective' and r['layer']==layer and
                r['architecture']==arch and not r['control'] and r['population']=='main'),None) for arch in ['linear','mlp64']]
            vals=[f"{r['balanced_accuracy']*100:.2f}%（{r['models']}模型）" if r else 'NA' for r in rr]
            lines.append(f'| {arm} | L{layer} | '+ ' | '.join(vals)+' |')
    lines+=['', '## 保持该层临时节点与全词表 argmax 不变的因果干预', '',
        '以下为主位置集，原输出头读出，向合法训练均值钳制。数值为当前 token 合法率相对同范数随机对照的变化。先对查询内部位置平均，再对模型seed平均。', '',
        '| 模型 | 层 | 准确率变化 | 配对95%区间 | MLP错误概率变化 | 模型数 |', '|---|---|---:|---|---:|---:|']
    for arm in ['matched_clean','correction']:
        for layer in [4,5,6]:
            r=get(arm,layer,'legal','accuracy');m=get(arm,layer,'legal','mlp_signal')
            ci=' ～ '.join(pp(v) for v in r['null_adjusted_ci95']) if r and r['null_adjusted_ci95'] else 'NA'
            lines.append(f"| {arm} | L{layer} | {pp(r['null_adjusted_delta']) if r else 'NA'} | {ci} | {pp(m['delta']) if m else 'NA'} | {r['models'] if r else 0} |")
    lines+=['', '## 必须保留的解释边界', '',
        '- 原输出头和校准读出都是对中间表示的诊断代理，不能断言模型在浅层实际作出了一个离散决定。校准只模仿训练集的冻结最终logits，不训练正确节点标签。',
        '- 一些模型的浅层临时节点几乎全非法；对应临时候选探针缺少合法类，严格标为NA。不能用“总判错”证明识错能力。前瞻探针使用真实最终当前token合法性标签，与临时候选标签分开。',
        '- 保持倾向的L6干预不改变实际输出，是由约束与计算图决定的结构负对照，不能单独当作“最后一层太晚”的实验证据。',
        '- 非约束干预可能直接改变浅层候选。保持倾向干预则固定该层全词表与节点argmax；信号与随机对照采用共同回退步长，不按干预成功筛掉样本。',
        '- 主因果位置要求仍有合法后继节点；结束列表位置单独展示读出统计。合法集合允许任意剩余后继，不要求固定路径或排序。',
        '- 主位置与自然首错位置分开展示；首错位置经过富集，不代表自然发生率。所有查询可达，查询和图在读出校准、探针训练、各自验证与测试间隔离。',
        '- 探针能解码未来错误不等于内部显式错误检测，错误预测方向也可能包含节点或图结构信息。', '',
        f"8个冻结模型；{d['total_interventions']}个当前token干预结果。独立语义、时序、范数与真实checkpoint缓存重算验证通过。未保存激活或探针权重。"]
    (P/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()

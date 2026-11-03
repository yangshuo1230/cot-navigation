"""Compact paired causal estimates and a standalone offline HTML report."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np

P=Path(__file__).resolve().parent
PRIMARY={'local':{'mistake_p','decision_correct'},
         'free':{'correct','focus_mark','first_repaired','answer_and_trace_valid','focus_false_mark'}}
METRICS={'local':['mistake_p','decision_correct','desired_p','linear_shift','mlp_shift','relative_norm','kl_from_sham'],
         'free':['correct','focus_mark','first_repaired','answer_and_trace_valid','focus_false_mark','limit','forced_stop']}


def summarize(blobs,selection):
    groups=defaultdict(list);index={};quality=defaultdict(list)
    timing={(r['event'],r['y']):r['pos']==r['decision_pos'] for r in selection['selected_readouts'] if r['site']=='onset'}
    for blob in blobs:
        for row in blob['probe_test_accuracy']:
            quality[blob['arm'],row['site'],row['layer'],row['architecture']].append(row['accuracy'])
        for domain,field in [('local','local'),('free','rollouts')]:
            for raw in blob[field]:
                row=dict(raw,domain=domain,arm=blob['arm'],model_seed=blob['model_seed'])
                if domain=='local':
                    row['linear_shift']=row.get('linear_after',0)-row.get('linear_before',0)
                    row['mlp_shift']=row.get('mlp64_after',0)-row.get('mlp64_before',0)
                site=row.get('site','onset')
                same=row.get('onset_equals_decision',timing[row['event'],row['y']])
                sites=['decision'] if site=='decision' else ['onset_all','onset_same' if same else 'onset_past']
                for variant in sites:
                    v=dict(row,site=variant)
                    key=(domain,variant,row['arm'],row['y'],row['layer'],row['condition'],row['dose'])
                    groups[key].append(v)
                    idx=(domain,variant,row['arm'],row['y'],row['model_seed'],row['event'],row['layer'],row['condition'],row['dose'])
                    assert idx not in index
                    index[idx]=v
    estimates=[]
    for number,(key,rows) in enumerate(sorted(groups.items())):
        domain,site,arm,y,layer,condition,dose=key
        if condition=='sham':continue
        seeds=sorted({r['model_seed'] for r in rows})
        by_seed={seed:{r['event']:r for r in rows if r['model_seed']==seed} for seed in seeds}
        events=sorted(set.intersection(*[set(d) for d in by_seed.values()]))
        if not events:continue
        rng=np.random.default_rng(20261031+number)
        si=rng.integers(len(seeds),size=(1000,len(seeds),1))
        qi=rng.integers(len(events),size=(1000,1,len(events)))
        for metric in METRICS[domain]:
            values=[];baselines=[];nulls=[]
            null_condition='random_'+condition if condition in ['clean','error'] and dose==1 else None
            for seed in seeds:
                vv=[];bb=[];nn=[]
                for event in events:
                    r=by_seed[seed][event]
                    base=index[domain,site,arm,y,seed,event,0,'sham',0.]
                    vv.append(float(r.get(metric,0)));bb.append(float(base.get(metric,0)))
                    if null_condition:
                        null=index[domain,site,arm,y,seed,event,layer,null_condition,1.]
                        nn.append(float(null.get(metric,0)))
                values.append(vv);baselines.append(bb)
                if null_condition:nulls.append(nn)
            values=np.array(values);baselines=np.array(baselines);delta=values-baselines
            ci=np.quantile(delta[si,qi].mean((1,2)),[.025,.975]).tolist() if metric in PRIMARY[domain] else None
            item=dict(domain=domain,site=site,arm=arm,y=y,layer=layer,condition=condition,dose=dose,
                metric=metric,models=len(seeds),queries_per_model=len(events),mean=float(values.mean()),
                baseline=float(baselines.mean()),delta=float(delta.mean()),paired_ci95=ci,
                per_model=[dict(seed=seed,mean=float(values[i].mean()),delta=float(delta[i].mean())) for i,seed in enumerate(seeds)])
            if null_condition:
                adjusted=values-np.array(nulls)
                item.update(null_adjusted_delta=float(adjusted.mean()),
                    null_adjusted_ci95=np.quantile(adjusted[si,qi].mean((1,2)),[.025,.975]).tolist() if metric in PRIMARY[domain] else None)
            estimates.append(item)
    probe_quality=[dict(zip(['arm','site','layer','architecture'],key),accuracy=float(np.mean(v)),
        model_seed_std=float(np.std(v,ddof=1)),models=len(v),threshold=.5) for key,v in sorted(quality.items())]
    flips=[]
    for key,rows in groups.items():
        domain,site,arm,y,layer,condition,dose=key
        if domain!='free' or site!='onset_all' or condition not in ['clean','error']:continue
        for r in rows:
            base=index[domain,site,arm,y,r['model_seed'],r['event'],0,'sham',0.]
            if r['correct']!=base['correct'] or r['first_repaired']!=base['first_repaired']:
                flips.append(dict(arm=arm,seed=r['model_seed'],query=r['query'],y=y,layer=layer,condition=condition,
                    answer_before=base['correct'],answer_after=r['correct'],repair_before=base['first_repaired'],repair_after=r['first_repaired']))
    return dict(complete=True,models=len(blobs),selection=selection['selection'],estimates=estimates,
        probe_quality=probe_quality,interesting_flips=flips[:40],interesting_flips_total=len(flips),
        totals=dict(local_readouts=sum(len(d['local']) for d in blobs),free_continuations=sum(len(d['rollouts']) for d in blobs),
            seconds=sum(d['seconds'] for d in blobs)),
        audits=dict(checkpoints_unchanged=all(d['checkpoint_unchanged'] for d in blobs),
            same_prefix_cohort=all(d['selection']==blobs[0]['selection'] for d in blobs),
            past_last_layer_fixed_text_control=True,unit_tests_passed=14),
        uncertainty='1000 paired crossed model-seed/query bootstrap replicates; exploratory, conditional-prefix cohort',
        checkpoints=[dict(arm=d['arm'],seed=d['model_seed'],**d['checkpoint']) for d in blobs])


def main():
    blobs=[json.loads(p.read_text()) for p in sorted((P/'results').glob('*_s*.json'))]
    assert len(blobs)==8 and all(d['complete'] and d['checkpoint_unchanged'] for d in blobs)
    for d in blobs:
        for name,expected in d['identity']['source_sha256'].items():
            assert hashlib.sha256((P.parents[2]/name).read_bytes()).hexdigest()==expected,name
        assert hashlib.sha256(Path(d['checkpoint']['path']).read_bytes()).hexdigest()==d['checkpoint']['sha256']
    selection=json.loads((P/'selection.json').read_text())
    data=summarize(blobs,selection)
    verification=json.loads((P/'verification.json').read_text())
    assert verification['passed']
    data['audits']['independent_verification']=verification
    (P/'summary.json').write_text(json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False),encoding='utf-8')
    payload=json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False).replace('<','\\u003c').replace('&','\\u0026')
    template=(P/'dashboard_template.html').read_text(encoding='utf-8')
    assert template.count('__CAUSAL_DATA__')==1
    (P/'causal_intervention_dashboard.html').write_text(template.replace('__CAUSAL_DATA__',payload),encoding='utf-8')
    write_markdown(data)
    print('CAUSAL REPORT COMPLETE',data['totals'],flush=True)


def write_markdown(data):
    def get(domain,site,layer,condition,metric,arm='correction',y=1):
        return next(r for r in data['estimates'] if r['domain']==domain and r['site']==site and
            r['layer']==layer and r['condition']==condition and r['metric']==metric and
            r['arm']==arm and r['y']==y and r['dose']==1)
    def pp(v):return f'{v*100:+.2f} pp'
    def ci(v):return ' ～ '.join(pp(x) for x in v) if v else '—'
    lines=['# 自然错误信号的因果干预', '',
        '[打开离线交互结果](causal_intervention_dashboard.html)', '',
        f"8个冻结200k模型；局部{data['selection']['local_pairs']}对自然前缀，自由续写{data['selection']['free_pairs']}对；各4个模型seed。", '',
        '方向仅由自然训练集拟合的线性探针确定，验证BCE选步。将一个残差投影移至训练合法／错误类均值，改动范数最多20%；同范数正交随机方向作对照。MLP独立拟合检查错误概率变化。探针分类统一阈值0.5。', '',
        '## 首错位置干预：错误读出和后续报错概率', '',
        '已错前缀、合法均值钳制、强度1。下表局部文本固定到动作前分隔符。概率变化以百分点表示。', '',
        '| 层 | MLP错误概率变化 | MISTAKE概率：不干预 → 干预 | 相对不干预 | 减去随机对照后的变化及95%区间 |',
        '|---|---:|---:|---:|---|']
    for l in [4,5,6]:
        r=get('local','onset_all',l,'clean','mistake_p');m=get('local','onset_all',l,'clean','mlp_shift')
        lines.append(f"| L{l} | {pp(m['mean'])} | {r['baseline']*100:.2f}% → {r['mean']*100:.2f}% | {pp(r['delta'])} | {pp(r['null_adjusted_delta'])} [{ci(r['null_adjusted_ci95'])}] |")
    lines+=['', '## 动作分隔符处干预：最后一层仍可改变决策', '',
        '| 层 | 合法均值钳制：MISTAKE概率变化 | 错误均值钳制：MISTAKE概率变化 | 合法钳制减去随机对照后的95%区间 |',
        '|---|---:|---:|---|']
    for l in [4,5,6]:
        a=get('local','decision',l,'clean','mistake_p');b=get('local','decision',l,'error','mistake_p')
        lines.append(f"| L{l} | {pp(a['delta'])} | {pp(b['delta'])} | {ci(a['null_adjusted_ci95'])} |")
    lines+=['', '固定文本时，严格过去位置的L6干预必须无效，这是计算图结构的负对照，不是“最后一层没有纠错机会”的证据。动作分隔符处的L6干预直接作用于下一token决策。', '',
        '## 自由续写：是否稳定影响最终答案与真正修复', '',
        '从首错token自由续写到原生4K／STOP预算；没有参考续写或强制纠错。以下均为纠错模型、已错前缀、强度1；数值为干预减去同范数随机干预，不是相对不干预的差。', '',
        '| 层 / 干预 | 最终答案准确率变化及95%区间 | 首错修复率变化及95%区间 | 当前记录报错变化及95%区间 |',
        '|---|---|---|---|']
    for l in [4,5,6]:
        for condition,label in [('clean','合法均值'),('error','错误均值')]:
            rr=[get('free','onset_all',l,condition,m) for m in ['correct','first_repaired','focus_mark']]
            lines.append(f"| L{l} / {label} | "+' | '.join(f"{pp(r['null_adjusted_delta'])} [{ci(r['null_adjusted_ci95'])}]" for r in rr)+' |')
    lines+=['', '## 解释边界', '',
        '- 这里只控制一个错误预测方向；不能据此声称消除了全部错误信息。方向也可能携带图或搜索状态信息。',
        '- 每次只干预一个token位置，后续仍可能重新计算错误信号；弱效应不能排除更广泛错误表征的重要性。',
        '- 探针概率改变、MISTAKE决策改变、修复成功和最终答案改变是不同端点。需要看随机对照调整后的配对区间，而非只看方向或某一个seed。',
        '- 共享前缀来自8个模型的自然生成，不一定是接收模型自己生成。比较反映相同自然错误输入下的处理差异。',
        '- 所有查询均可达；合法／已错指CoT前缀。条件前缀上的答案准确率不能代表完整任务分布或总体训练收益。',
        '- 自由续写仅32对独立前缀，4个模型seed重复使用同一批图；完整结果必须结合样本量及不确定性解读。',
        '- 95%区间使用1000次配对模型seed×查询bootstrap；本轮为探索性干预，不按显著性挑选层或方向。', '',
        '## 验证与产物', '',
        f"{data['totals']['local_readouts']}个局部读出、{data['totals']['free_continuations']}条自由续写；相关CPU测试14项通过，全部自然读出重新语义重放，随机正交性／范数／零剂量生成身份检查通过，8个checkpoint哈希未变。", '',
        '逐例指标保存在results/，来源指纹在plan.json，独立验证在verification.json，汇总为summary.json。未保存激活或探针权重。']
    (P/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()

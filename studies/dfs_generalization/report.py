"""Incremental, offline report from compact semantic audits; no model inference."""
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

P=Path(__file__).resolve().parent
sys.path.insert(0,str(P))
from run import ROOT, OUT, DATA, ARMS, NODES
from cot_navigation.experiments.scheduler import file_lock, write_json


def metrics(rows):
    n=len(rows)
    if not n:
        return None
    reasons=Counter(r['first']['kind'] for r in rows if r['first'])
    fields=Counter(r['first']['field'] for r in rows if r['first'])
    positions=Counter()
    for r in rows:
        if r['first']:
            step=r['first']['search_step']
            positions['1' if step==1 else '2' if step==2 else '3–4' if step<=4 else
                      '5–8' if step<=8 else '9–16' if step<=16 else '17–32' if step<=32 else '33+']+=1
    def rate(key):
        return sum(bool(r[key]) for r in rows)/n
    return dict(n=n, accuracy=rate('correct'), legality=rate('trace_valid'),
        literal_legality=rate('literal_valid'), answer_and_trace_valid=rate('answer_and_trace_valid'),
        invalid_but_correct=sum(r['correct'] and not r['trace_valid'] for r in rows)/n,
        interrupted=rate('interrupted'), forced_stop=rate('forced_stop'),
        reference_fit_rate=rate('reference_fits'),
        reference_fit_subset=metrics_core([r for r in rows if r['reference_fits']]),
        first_errors=dict(reasons), first_fields=dict(fields), first_positions=dict(positions),
        first_error_n=sum(reasons.values()), outcomes=dict(Counter(r['outcome'] for r in rows)),
        no_error_incomplete=sum(not r['first'] and not r['trace_valid'] for r in rows),
        mean_trace_tokens=sum(r['trace_length'] for r in rows)/n,
        mistake_tokens=sum(r['raw_mistake_tokens'] for r in rows),
        repairs=sum(r['repairs'] for r in rows))


def metrics_core(rows):
    n=len(rows)
    return dict(n=n,accuracy=sum(r['correct'] for r in rows)/n,
                legality=sum(r['trace_valid'] for r in rows)/n) if n else None


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with file_lock(OUT/'.report.lock'):
        manifest=json.loads((DATA/'manifest.json').read_text())
        results=[]
        for path in sorted((OUT/'results').glob('*.json.gz')):
            with gzip.open(path,'rt',encoding='utf-8') as handle:
                item=json.load(handle)
            assert item['complete']
            for population,label in [('positive',1),('negative',0),('all',None)]:
                rows=[r for r in item['rows'] if label is None or r['label']==label]
                results.append(dict(arm=item['model']['arm'],seed=item['model']['seed'],
                    cell=item['cell']['name'],n_nodes=item['cell']['n'],family=item['cell']['family'],
                    population=population,metrics=metrics(rows)))
        summary=dict(complete_cells=len(results)//3,expected_cells=96,
            primary_population='positive',manifest=manifest,results=results,
            notes=['同一固定查询及批次用于全部8个模型；每组4个seed。',
                   '合法率允许MISTAKE取消错误记录后的有效修复；另报告完全无错合法率。',
                   '首错分布避免将错误动作后的连锁失效重复计数；缺少完整终止但无首错单独统计。',
                   '简单图M=floor(1.5N)，复杂图M=2N；图族和边数共同变化，不能单独归因于拓扑。',
                   '全部使用原生4K预算；截断率与参考DFS可完成子集单独报告。参考长度依赖搜索顺序。',
                   '跨规模的rank-band跨度分布不同，正样本和负样本分别查看。'])
        write_json(OUT/'summary.json',summary)
        payload=json.dumps(summary,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
        template=(P/'dashboard_template.html').read_text(encoding='utf-8')
        (OUT/'dashboard.html').write_text(template.replace('__DATA__',payload),encoding='utf-8')
        lines=['# DFS泛化评估','',f'已完成 {summary["complete_cells"]}/96 个模型×图设置。',
               '',*['- '+n for n in summary['notes']],'',
               '| 图 | N | baseline准确率 | 纠错准确率 | baseline合法率 | 纠错合法率 |',
               '|---|---:|---:|---:|---:|---:|']
        for cell in manifest['cells']:
            vals=[]
            for key in ('accuracy','legality'):
                for arm in ARMS:
                    rr=[r['metrics'][key] for r in results if r['population']=='positive' and
                        r['cell']==cell['name'] and r['arm']==arm]
                    vals.append(f'{100*sum(rr)/len(rr):.2f}% ({len(rr)}/4)' if rr else '待运行')
            lines.append(f'| {cell["family"]} | {cell["n"]} | '+' | '.join(vals)+' |')
        (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        print('REPORT',summary['complete_cells'],'/96',flush=True)


if __name__=='__main__':
    main()

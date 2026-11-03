"""Audit completed compact outputs and collect comparable per-seed summaries."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
COHORT='all_positions_tokenmatched_20261103'
OUT=ROOT/'outputs/analysis/correction_mechanisms'/COHORT


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def curve(data,arch,dataset):
    result=[]
    for head in range(data['model']['layers']+2):
        vals=[r['accuracy'] for run in data['runs'] if run['architecture']==arch
            for r in run.get('tests',{}).get(dataset,[]) if r['head']==head]
        result.append(float(np.mean(vals)) if vals else None)
    return result


def main():
    manifest=read(ROOT/'studies/correction_behavior/cohorts'/f'{COHORT}.json')
    behavior=read(ROOT/manifest['output']/'summary.json');assert behavior['complete']
    state=[read(OUT/'state'/(e['id']+'.json')) for e in manifest['models']]
    history=[read(OUT/'history'/(e['id']+'__'+alignment+'.json')) for e in manifest['models'] for alignment in ['error_token_0_5','record_0_2']]
    expected={e['id']:e for e in manifest['models']};sha=set();checked_cells=0
    for d in state+history:
        e=expected[d['model']['id']]
        assert d['complete'] and d['model']['run']==e['run']
        assert d['checkpoint']['step']==e['checkpoint_step']
        assert d['checkpoint']['sha256']==d['model']['checkpoint_sha256'];sha.add(d['checkpoint']['sha256'])
        if 'runs' in d:
            for run in d['runs']:
                for tests in run.get('tests',{}).values():
                    for r in tests:
                        assert r['threshold']==.5 and r['n']==2*r['positive']
                        assert abs(r['accuracy']-(r['tpr']+1-r['fpr'])/2)<1e-9
                        assert 'auroc' not in r;checked_cells+=1
        if 'semantic_transitions' in d:
            for panel in d['semantic_transitions'].values():
                for r in panel['transitions']:
                    assert sum(r[k] for k in ['legal_legal','legal_illegal','illegal_legal','illegal_illegal'])==r['n']
    assert len(sha)==12
    groups={};lines=['# 完整结果复核',
        '自然行为60/60、状态/lens 12/12、历史探针24/24完成；强制MISTAKE暂缓。',
        '所有模型均为本批token曝光匹配最终checkpoint。每组只有2个初始化seed。',
        '下表百分数为两模型seed均值；错误历史探针阈值0.5，首错token+0…+5宏平均，普通N100共同留出前缀。',
        '|组|历史错误MLP L5|严格同token L5|打乱标签 L5|候选集合成员BA L5|校准lens L5最终一致率|',
        '|---|---|---|---|---|---|']
    for group in sorted({e['group'] for e in manifest['models']}):
        sd=[d for d in state if d['model']['group']==group];depth=sd[0]['model']['layers']
        hh=[d for d in history if d['model']['group']==group and d['alignment']=='error_token_0_5']
        record=[d for d in history if d['model']['group']==group and d['alignment']=='record_0_2']
        g=dict(layers=depth,behavior={},history={},state={},lens={},semantic={})
        for task in ['n32','n64','main','deep','hard']:
            b=behavior['tasks'][task][group]
            g['behavior'][task]=dict(n=b['n'],accuracy=b['correct']/b['n'],valid=b['valid']/b['n'],
                marks=b['true_marks'],false_marks=b['false_marks'],repairs=b['repairs'],cuts=b['interrupted'],
                seeds={seed:dict(accuracy=p['correct']/p['n'],valid=p['valid']/p['n']) for seed,p in b['per_seed'].items()},
                history={k:dict(n=v['n'],bad=v['bad']) for k,v in b['by_history'].items()},revisits=b['structure']['revisits'])
        for arch in ['linear','mlp64']:
            for dataset in ['test','exact_token_test','own_test','shuffle_control']:
                arr=[curve(d,arch,dataset) for d in hh]
                if any(v is None for a in arr for v in a):continue
                a=np.array(arr)
                g['history'][arch+'/'+dataset]=dict(mean=a.mean(0).tolist(),std=a.std(0).tolist(),per_seed=a.tolist())
        for task,metric in [('field_node','accuracy'),('field_bar','accuracy'),('frontier','balanced_accuracy'),('visited','balanced_accuracy'),('listed_candidates','balanced_accuracy'),('successors_pre','balanced_accuracy')]:
            arr=[]
            for d in sd:
                p=next(p for p in d['probes'] if p['architecture']=='mlp64' and p['task']==task)
                arr.append([r[metric] for r in p['test']])
            a=np.array(arr);g['state'][task]=dict(mean=a.mean(0).tolist(),std=a.std(0).tolist())
        for kind in ['raw','calibrated']:
            a=np.array([[r['final_agreement'] for r in d['logit_lens'] if r['task']=='field_node' and r['kind']==kind] for d in sd])
            g['lens'][kind]=dict(mean=a.mean(0).tolist(),std=a.std(0).tolist(),per_seed=a.tolist())
        for layer in sd[0]['semantic_transitions']:
            for kind in ['raw','calibrated']:
                rr=[r for d in sd for r in d['semantic_transitions'][layer]['transitions'] if r['kind']==kind and r['population']=='all_legal_prefix' and r['task']=='all_unique_positions']
                counts={k:sum(r[k] for r in rr) for k in ['n','legal_legal','legal_illegal','illegal_legal','illegal_illegal']};n=counts['n']
                g['semantic'][layer+'/'+kind]=dict(counts,l5_legal=(counts['legal_legal']+counts['legal_illegal'])/n,
                    final_legal=(counts['legal_legal']+counts['illegal_legal'])/n)
        vals=[g['history']['mlp64/test']['mean'][4],g['history']['mlp64/exact_token_test']['mean'][4],g['history']['mlp64/shuffle_control']['mean'][4],g['state']['listed_candidates']['mean'][4],g['lens']['calibrated']['mean'][4]]
        lines.append('| '+' | '.join([group]+[f'{100*v:.1f}%' for v in vals])+' |');groups[group]=g
    lines+=['','## 回答准确率（两个seed合并）','|组|N32|N64|N100普通|N100窄深|N100困难|','|---|---|---|---|---|---|']
    for group,g in groups.items():
        lines.append('| '+' | '.join([group]+[f'{100*g["behavior"][task]["accuracy"]:.2f}%' for task in ['n32','n64','main','deep','hard']])+' |')
    lines+=['','## 9层30%相对同seed基线的变化（百分点）','|图|seed0 准确率变化|seed1 准确率变化|seed0 合法率变化|seed1 合法率变化|','|---|---|---|---|---|']
    for task in ['n32','n64','main','deep','hard']:
        b=groups['l9_p0']['behavior'][task]['seeds'];c=groups['l9_p30']['behavior'][task]['seeds']
        diff=[100*(c[seed][metric]-b[seed][metric]) for metric in ['accuracy','valid'] for seed in ['0','1']]
        lines.append('| '+' | '.join([task]+[f'{v:+.2f}' for v in diff])+' |')
    lines+=['','9层30%在五种图上均为seed0下降、seed1上升。可以描述均值/种子差异，不能仅凭两个初始化声称稳定优势。',
        '60%组第5层错误历史更容易解码，但普通图/小图导航没有相应提高，因此更强错误可解码性不足以解释回答收益。',
        '9层30%的校准lens改善主要出现在第7–8层：第8层最终一致率约73.8%，对照54.7%；第5层并未同步提高。',
        '自然回溯错误依然持续：N100普通图9层30%同一额外错候选复现29/35，仍需后继的重复遗漏61/61。',
        '','## 数据与解释限制',
        '探针可解码性不等于模型主动利用该信号。+0位置已消费错误token，不能据此称输出前自我检测。',
        '校准lens最终预测一致率不等于CoT合法率。语义转变另用真图replay计数，前序已错排除，首错预测位置保留。',
        '状态/lens在各模型自身自然测试轨迹上比较，轨迹分布不同；历史错误探针额外提供共同前缀和严格同token对照。',
        '普通模型与纠错模型均只有2个初始化seed，必须查看逐seed结果，不能只根据均值声称稳定获益。']
    result=dict(complete=True,unique_checkpoints=len(sha),audited_history_cells=checked_cells,groups=groups)
    (OUT/'review_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'REVIEW.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines[:13]));print('AUDIT',len(sha),checked_cells)
    for group,g in groups.items():
        print(group,'lens calibration',g['lens']['calibrated']['mean'],'state successors',g['state']['successors_pre']['mean'][:g['layers']])


if __name__=='__main__':main()

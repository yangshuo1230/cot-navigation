"""Read-only completed-run audit and matched comparison with the 200k suite."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from statistics import mean
from collections import Counter
import json

ROOT=Path('/mnt/ningxuefei/yangshuo/cot-navigation-unified')
OUT=ROOT/'outputs/fine_node_curriculum_20261028'
OLD=ROOT.with_name('cot-navigation')/'outputs/node_family_recheck_20261028'
LEVELS=[12,16,24,32,36,48,64,80,100]

def read_rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()]

runs=[]
for p in sorted(OUT.glob('s*/*/*')):
    if not (p/'COMPLETE.json').is_file():continue
    name=str(p.relative_to(OUT));seed,protocol,family=name.split('/')
    final=json.loads((p/'eval/free_test_last.json').read_text())
    predictions=read_rows(p/'eval/predictions_test_last.jsonl')
    old_predictions=read_rows(OLD/name/'eval/predictions_final_own_n100.jsonl')
    old_sem=json.loads((OLD/name/'eval/semantic_own_n100.json').read_text())
    old_by_id={x['query_id']:x for x in old_predictions}
    matched=len(predictions)==len(old_predictions) and all(
        r['query_id'] in old_by_id and
        all(r[k]==old_by_id[r['query_id']][k] for k in ['label','prompt_tokens'])
        for r in predictions)
    assert matched,name
    old_pos=[r for r in old_predictions if r['label']==1]
    pos_ids={r['query_id'] for r in old_pos}
    old_metrics=dict(ba=sum(r['correct'] for r in old_predictions)/len(old_predictions),
        positive_accuracy=sum(r['correct'] for r in old_pos)/len(old_pos),
        positive_valid=sum(x['trace_valid'] for x in old_sem['records'] if x['query_id'] in pos_ids)/len(old_pos))
    validation=[]
    for path in sorted((p/'eval').glob('free_0*.json')):
        a=json.loads(path.read_text());stage=min((a['step']-1)//100000,8)
        validation.append(dict(step=a['step'],nodes=LEVELS[stage],
            ba=a['overall']['balanced_accuracy'],positive_accuracy=a['overall']['yes_recall'],
            positive_valid=a['semantic']['positive_valid_rate']))
    logs=[x for x in read_rows(p/'metrics.jsonl') if 'answer_loss' in x]
    windows=[]
    for end in range(820000,1000001,20000):
        subset=[x for x in logs if end-20000<x['step']<=end]
        windows.append(dict(end=end,**{k:mean(x[k] for x in subset) for k in
            ['answer_loss','trace_loss','data_time','update_time']}))
    retention=[]
    for path in sorted((p/'eval').glob('retention_01000000_level*.json')):
        level=int(path.stem.split('level')[-1]);a=json.loads(path.read_text())
        retention.append(dict(nodes=LEVELS[level],ba=a['overall']['balanced_accuracy'],
                              positive_valid=a['semantic']['positive_valid_rate']))
    stages=[]
    for x in validation:
        if x['step'] in [100000*i for i in range(1,9)]+[1000000]:stages.append(x)
    runs.append(dict(job=name,seed=int(seed[1:]),protocol=protocol,family=family,
                     final=final,old=old_metrics,identical_test_prompts=matched,
                     stage_ends=stages,validation=validation,windows=windows,retention=retention))

groups=[]
for protocol in ['dfs_nodes','bfs_nodes']:
    for family in ['simple','hard']:
        chosen=[r for r in runs if r['protocol']==protocol and r['family']==family]
        group=dict(protocol=protocol,family=family,
                   ba=mean(r['final']['overall']['balanced_accuracy'] for r in chosen),
                   positive_accuracy=mean(r['final']['overall']['yes_recall'] for r in chosen),
                   positive_valid=mean(r['final']['semantic']['positive_valid_rate'] for r in chosen),
                   all_valid=mean(r['final']['semantic']['valid_rate'] for r in chosen),
                   old={k:mean(r['old'][k] for r in chosen) for k in ['ba','positive_accuracy','positive_valid']})
        errors=Counter()
        for r in chosen:errors.update(r['final']['semantic']['positive_errors'])
        group['positive_errors']=dict(errors)
        group['stages']=[dict(nodes=LEVELS[i],**{key:mean(r['stage_ends'][i][key] for r in chosen)
                         for key in ['ba','positive_valid']}) for i in range(9)]
        group['retention']=[dict(nodes=LEVELS[i],**{key:mean(r['retention'][i][key] for r in chosen)
                            for key in ['ba','positive_valid']}) for i in range(8)]
        group['n100_timepoints']=[dict(step=step,**{key:mean(next(x[key] for x in r['validation'] if x['step']==step)
                for r in chosen) for key in ['ba','positive_valid']}) for step in [820000,900000,1000000]]
        groups.append(group)
result=dict(checked_at_beijing=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            status=json.loads((OUT/'status.json').read_text()),runs=runs,groups=groups)
(OUT/'results_check_20261029.json').write_text(json.dumps(result,indent=2))
print('All 12 tests exactly match old query IDs, labels and edge/query prompt tokens.')
for g in groups:
    print(g['protocol'],g['family'],{k:round(g[k]*100,2) for k in ['ba','positive_accuracy','positive_valid','all_valid']},
          'old',{k:round(v*100,2) for k,v in g['old'].items()},'errors',g['positive_errors'])
    print('Stages',[(x['nodes'],round(x['ba']*100,2),round(x['positive_valid']*100,2)) for x in g['stages']])
    print('Retention',[(x['nodes'],round(x['ba']*100,2),round(x['positive_valid']*100,2)) for x in g['retention']])
    print('N100',g['n100_timepoints'])
for r in runs:
    if r['family']=='hard':print('LOSS',r['job'],[(x['end'],round(x['answer_loss'],4),round(x['trace_loss'],4)) for x in r['windows'] if x['end'] in [820000,900000,1000000]])

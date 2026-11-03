"""Read-only result aggregation; never stop or modify training."""
from pathlib import Path
import json
import statistics
from datetime import datetime,timezone

ROOT=Path(str(next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'pyproject.toml').is_file() and (_p / 'studies').is_dir())))
SUITE=ROOT/'studies/fixed_exposure_suite'
OUT=ROOT/'outputs/exposure_grokking_v2'
result=dict(collected_at_utc=datetime.now(timezone.utc).isoformat(),fixed=[],grokking=[])
keys=['accuracy','balanced_accuracy','yes_recall','no_recall','n']
def brief(x):return {k:x.get(k) for k in keys}
for folder in sorted((OUT/'fixed').glob('*/*/*')):
    if not (folder/'final.json').exists():continue
    final=json.loads((folder/'final.json').read_text())
    evaluations=sorted(folder.glob('eval_*.json'))
    last=json.loads(evaluations[-1].read_text())
    result['fixed'].append(dict(id=str(folder.relative_to(OUT)),final=brief(final),
        last_train=brief(last['train']),last_monitor=brief(last['monitor'])))
for folder in sorted((OUT/'grokking').iterdir()):
    if not folder.is_dir():continue
    curves=[]
    for path in sorted(folder.glob('eval_*.json')):
        d=json.loads(path.read_text())
        curves.append(dict(step=d['step'],phase=d['phase'],train=brief(d['train']),monitor=brief(d['monitor'])))
    lines=(folder/'metrics.jsonl').read_text().splitlines()
    latest=json.loads(lines[-1])
    top=max(curves,key=lambda x:x['monitor']['balanced_accuracy'])
    failures=[]
    for d in curves:
        if d['train']['accuracy']<.99:failures.append('train')
        if d['monitor']['balanced_accuracy']<.95:failures.append('monitor_ba')
        if min(d['monitor']['yes_recall'],d['monitor']['no_recall'])<.95:failures.append('class_recall')
    recent=json.loads(sorted(folder.glob('eval_*.json'))[-1].read_text())
    mistakes=[p for p in recent['monitor']['predictions'] if not p['correct']]
    from collections import Counter
    result['grokking'].append(dict(id=folder.name,phase1_completed=(folder/'PHASE1_GROKKED.json').exists(),
        latest_metrics=latest,evaluations=len(curves),first=curves[0],latest=curves[-1],best=top,
        curves=curves,failed_conditions=dict(Counter(failures)),
        latest_error_count=len(mistakes),latest_errors=mistakes,
        every_eval_train_perfect=all(c['train']['accuracy']>=.99 for c in curves)))
summary=[]
for protocol in ['direct','path']:
    for arm in ['R0','R2','R3']:
        rows=[r for r in result['fixed'] if r['id'].split('/')[-2:]==[protocol,arm]]
        summary.append(dict(protocol=protocol,arm=arm,n=len(rows),
            final_ba_mean=statistics.mean(r['final']['balanced_accuracy'] for r in rows),
            final_ba_sd=statistics.stdev(r['final']['balanced_accuracy'] for r in rows),
            yes_recall_mean=statistics.mean(r['final']['yes_recall'] for r in rows),
            no_recall_mean=statistics.mean(r['final']['no_recall'] for r in rows),
            train_accuracy_mean=statistics.mean(r['last_train']['accuracy'] for r in rows)))
result['fixed_summary']=summary
(SUITE/'results_snapshot_20261024.json').write_text(json.dumps(result,indent=2))
print(json.dumps(dict(fixed_summary=summary,grokking=[{k:v for k,v in d.items() if k not in ['curves','latest_errors']} for d in result['grokking']]),indent=2))

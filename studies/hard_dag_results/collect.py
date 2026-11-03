"""Read saved reports only; do not evaluate or resume any checkpoint."""
import json
from pathlib import Path
import yaml

base = Path('outputs/hard_dag_v1/launches/hard_dag_v1_s0')
results = []
for cfgpath in sorted(base.glob('**/resolved_config.yaml')):
    out = cfgpath.parent
    cfg = yaml.safe_load(cfgpath.read_text())
    if cfg['train']['stage'] == 'A':
        continue
    reports = {p.stem.removeprefix('free_test_'): json.loads(p.read_text())
               for p in sorted((out/'eval').glob('free_test*.json'))}
    curve = []
    for p in sorted((out/'eval').glob('free_*.json')):
        suffix = p.stem.removeprefix('free_')
        if not suffix.isdigit():
            continue
        row = json.loads(p.read_text())
        tfp = p.with_name('teacher_forced_'+suffix+'.json')
        tf = json.loads(tfp.read_text()) if tfp.exists() else {}
        curve.append(dict(step=int(suffix), overall=row.get('overall'),
                          teacher_forced=tf.get('overall'),
                          free_seconds=p.stat().st_mtime-tfp.stat().st_mtime if tfp.exists() else None))
    partial = {}
    for p in (out/'eval').glob('predictions_test*.jsonl'):
        tag = p.stem.removeprefix('predictions_test_')
        if tag not in reports:
            rows = [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
            partial[tag] = dict(n=len(rows), correct=sum(r['correct'] for r in rows),
                                forced_stop=sum(r.get('forced_stop', False) for r in rows))
    last_train = {}
    metrics = out/'metrics.jsonl'
    if metrics.exists():
        with metrics.open('rb') as f:
            f.seek(max(0, metrics.stat().st_size-200000))
            for line in f.read().splitlines():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if 'answer_loss' in r or 'trace_loss' in r:
                    last_train = r
    results.append(dict(job=str(out.relative_to(base)), config=cfg, reports=reports,
                        curve=curve, partial=partial, last_train=last_train))

study = Path('studies/hard_dag_results')
study.mkdir(exist_ok=True)
(study/'summary.json').write_text(json.dumps(dict(batch='hard_dag_v1_s0', runs=results), indent=2)+'\n')
for r in results:
    c = r['config']
    selection = c['evaluation']['selection_metric']
    candidates = [v for v in r['curve'] if v['step'] >= c['evaluation'].get('selection_min_step', 0)]
    best = max(candidates, key=lambda v: v['overall'].get(selection) or 0) if candidates else None
    print(json.dumps(dict(job=r['job'], selection=selection, best_val=best,
        last_val=r['curve'][-1] if r['curve'] else None,
        test={k: dict(overall=v['overall'], checkpoint_step=v.get('checkpoint_step'),
                      training_step=v.get('training_step'), query_view=v.get('query_view'))
              for k, v in r['reports'].items() if k in ['best', 'last']},
        partial=r['partial'], last_train={k:v for k,v in r['last_train'].items()
                    if k in ['step', 'answer_loss', 'trace_loss', 'walk_loss', 'step_time']})))

"""Read existing fixed-DAG histories; never evaluate or modify training jobs."""
from pathlib import Path
import json
import statistics
import yaml

ROOT = Path(str(next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'pyproject.toml').is_file() and (_p / 'studies').is_dir())))
OUT = ROOT / 'studies/fixed_learning_speed'
OUT.mkdir(parents=True, exist_ok=True)

def read(p):
    return json.loads(p.read_text())

def compact(x):
    return {k: v for k, v in x.items() if not isinstance(v, (dict, list))}

runs = []
for p in sorted((ROOT / 'outputs').rglob('resolved_config.yaml')):
    c = yaml.safe_load(p.read_text())
    if c.get('data', {}).get('knowledge') != 'parametric_graph' or c.get('train', {}).get('stage') == 'A':
        continue
    d = p.parent
    curve = []
    for e in sorted((d / 'eval').glob('free_*.json')):
        if not e.stem[5:].isdigit():
            continue
        obj = read(e)
        curve.append(dict(step=obj.get('step', int(e.stem[5:])), **compact(obj['overall'])))
    logs = []
    if (d / 'metrics.jsonl').exists():
        for line in (d / 'metrics.jsonl').read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if 'step_time' in r:
                logs.append(r)
    times = [r['step_time'] for r in logs if r.get('step', 0) >= 1000]
    for point in curve:
        near = min(logs, key=lambda r: abs(r['step']-point['step'])) if logs else {}
        point['queries_seen'] = near.get('queries_seen')
    tests = {}
    for e in (d / 'eval').glob('free_test*.json'):
        obj = read(e)
        tests[e.stem] = dict(overall=compact(obj['overall']), step=obj.get('checkpoint_step'), view=obj.get('query_view'))
    runs.append(dict(path=str(d.relative_to(ROOT)), config=c, curve=curve, tests=tests,
                     median_step_seconds=statistics.median(times) if times else None,
                     last_log=compact(logs[-1]) if logs else {}))

for p in sorted((ROOT/'outputs/exposure_grokking_adj_v3/fixed').glob('*/*/*/resolved_job.json')):
    d=p.parent
    curve=[]
    for e in sorted(d.glob('eval_*.json')):
        obj=read(e)
        curve.append(dict(step=obj['step'], **{k:compact(v) for k,v in obj.items() if isinstance(v,dict)}))
    logs=[read_line for line in (d/'metrics.jsonl').read_text().splitlines() if (read_line:=json.loads(line))]
    runs.append(dict(path=str(d.relative_to(ROOT)), job=read(p), curve=curve,
                     final=compact(read(d/'final.json')) if (d/'final.json').exists() else None,
                     log_sample=logs[:1]+logs[-1:]))

(OUT/'inventory.json').write_text(json.dumps(runs,indent=2)+'\n')
for r in runs:
    print(r['path'], 'points',len(r['curve']), 'last',r['curve'][-1] if r['curve'] else None)

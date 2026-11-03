"""Read-only final report collection; never loads weights or launches runs."""
import json
from pathlib import Path

ROOT = Path.home()/'cot-navigation'
out = ROOT/'studies/navigation_results_check'
out.mkdir(parents=True, exist_ok=True)
result = {'suites': {}}
for suite, protocols in [('simple', ['direct','dfs_state','dfs_nodes']),
                         ('hard', ['direct','path','dfs_state','dfs_nodes'])]:
    base = ROOT/'outputs'/('fixed_navigation_protocols_v6' if suite == 'simple' else 'fixed_navigation_hard_v7')/'pilot'
    group = {'status': json.loads((base/'status.json').read_text()), 'runs': []}
    for protocol in protocols:
        for seed in range(4):
            folder = base/f'n80/g0_s{seed}'/protocol
            row = {'protocol': protocol, 'seed': seed}
            for name in ('COMPLETE','resolved_job','final','adjacency'):
                p = folder/f'{name}.json'
                row[name] = json.loads(p.read_text()) if p.exists() else None
            metrics = [json.loads(s) for s in (folder/'metrics.jsonl').read_text().splitlines()]
            row['last_training'] = metrics[-1]
            vals = [json.loads(s) for s in (folder/'validation.jsonl').read_text().splitlines()]
            row['validation_curve'] = [{'step': v['step'], 'panels': {
                k: d['metrics']['all'] for k,d in v['panels'].items()}} for v in vals]
            group['runs'].append(row)
    result['suites'][suite] = group
(out/'collected.json').write_text(json.dumps(result, separators=(',',':')))
print(json.dumps({s: {'complete':len(g['status']['completed']), 'failed':g['status']['failed'],
    'runs':[(r['protocol'],r['seed'],r['COMPLETE']['step'],r['last_training']['optimizer_step']) for r in g['runs']]}
    for s,g in result['suites'].items()}))

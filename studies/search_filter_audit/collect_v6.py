"""Read-only collection of completed v6 pilot scores and exposure audits."""
import json
from pathlib import Path
root=Path(str(next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'pyproject.toml').is_file() and (_p / 'studies').is_dir())))
base=root/'outputs/fixed_navigation_protocols_v6/pilot/n80'
doc={'runs':[],'active':[]}
for seed in range(4):
    for protocol in ['direct','dfs_nodes','dfs_state']:
        run=base/f'g0_s{seed}'/protocol
        final=run/'final.json'
        if not final.exists():
            metrics=run/'metrics.jsonl'
            tail=json.loads(metrics.read_text().splitlines()[-1]) if metrics.exists() else None
            doc['active'].append({'seed':seed,'protocol':protocol,'last_metric':tail})
            continue
        d=json.loads(final.read_text())
        out={'seed':seed,'protocol':protocol,'step':d['step'],'run':str(run),'panels':{},
             'complete':(run/'COMPLETE.json').exists()}
        for name,panel in d['panels'].items():
            out['panels'][name]={'all':panel['metrics']['all'],'by_gap':{k:v for k,v in panel['metrics'].items() if k!='all'},
                                 'errors':panel['errors'],'examples':panel['examples']}
        ep=run/'actual_exposure.json'
        out['exposure']=json.loads(ep.read_text()) if ep.exists() else None
        doc['runs'].append(out)
print(json.dumps(doc))

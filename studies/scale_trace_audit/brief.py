import json
from pathlib import Path
root=Path(str(next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'pyproject.toml').is_file() and (_p / 'studies').is_dir())) + '/studies/scale_trace_audit')
rows=[json.loads(p.read_text()) for p in (root/'results').rglob('*.json') if p.name!='complete.json']
print('Finished conditions',len(rows),'finished models',len(list((root/'results').rglob('complete.json'))))
for r in rows:
 if r['job'] in ['context/curriculum/dfs_state','context/curriculum/bfs_state','fixed/l10/stage_b/dfs_state','fixed/curriculum/stage_b/bfs_state']:
  print(r['job'],r['case'],r['n'],'answer',r['answer_accuracy'],'trace',r['complete_trace_accuracy'],'prefix',r['prefix_valid_rate'],'skeleton',r['skeleton_accuracy'],'counts',r['counts'],'errors',r['errors'])

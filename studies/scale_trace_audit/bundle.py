from pathlib import Path
import json,tarfile
root=Path(str(next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'pyproject.toml').is_file() and (_p / 'studies').is_dir())));folder=root/'studies/scale_trace_audit'
rows=[json.loads(p.read_text()) for p in (folder/'results').rglob('*.json') if p.name!='complete.json']
assert len(rows)==84 and len(list((folder/'results').rglob('complete.json')))==51
v=json.loads((folder/'verification.json').read_text());v.update(conditions=84,total_queries=sum(r['n'] for r in rows),all_conditions_answers_match_original=all(r['counts']['old_prediction_matches']==r['counts']['old_prediction_compared']==r['n'] for r in rows))
(folder/'verification.json').write_text(json.dumps(v,indent=2))
archive=folder/'reports_bundle.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
 for p in (folder/'results').rglob('*.json'):tar.add(p,arcname=str(p.relative_to(folder)))
 for name in ['REPORT.md','summary.json','failure_details.json','verification.json','plan.json','launch.json']:
  tar.add(folder/name,arcname=name)
print(json.dumps(v))

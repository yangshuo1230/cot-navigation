from pathlib import Path
import json,os
root=Path(str(next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'pyproject.toml').is_file() and (_p / 'studies').is_dir())) + '/studies/scale_trace_audit')
rows=[]
for p in (root/'results').rglob('*.json'):
 if p.name=='complete.json':continue
 r=json.loads(p.read_text());rows.append(r)
print('conditions',len(rows),'models',len(list((root/'results').rglob('complete.json'))))
for x in json.loads((root/'launch.json').read_text()):
 try:os.kill(x['pid'],0);live=True
 except ProcessLookupError:live=False
 print('worker',x['worker'],'alive',live,'tail',(root/f"worker_{x['worker']}.log").read_text().splitlines()[-1:])
for r in rows:
 if r['case']=='main':print(r['job'],r['n'],round(100*r['answer_accuracy'],2),round(100*r['complete_trace_accuracy'],2),round(100*r['skeleton_accuracy'],2),r['errors'], 'oldmatch',r['counts']['old_prediction_matches'],r['counts']['old_prediction_compared'])

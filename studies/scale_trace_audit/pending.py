from pathlib import Path
import json
R=Path(str(next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'pyproject.toml').is_file() and (_p / 'studies').is_dir())) + '/studies/scale_trace_audit')
J=json.loads((R/'plan.json').read_text())['jobs']
for w in range(7):
 print(w,[j['name'] for i,j in enumerate(J) if i%7==w and not(R/'results'/j['name']/'complete.json').exists()])

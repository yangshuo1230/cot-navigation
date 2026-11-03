from pathlib import Path
import json
root=Path(str(next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'pyproject.toml').is_file() and (_p / 'studies').is_dir())))
folder=root/'outputs/task_only_n32_v1/launches/task_only_n32_s0'
for p in folder.glob('task_only/*/metrics.jsonl'):
 rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
 train=[x for x in rows if 'answer_loss' in x]
 mode=[{k:v for k,v in r.items() if k in ['event','requested_mode','effective_mode','wandb_url','run_id']} for r in rows if any(k in r for k in ['effective_mode','requested_mode'])]
 print(p.parent.name,'last',train[-1] if train else 'initializing','tracking',mode)
print('status',json.loads((folder/'status.json').read_text()))

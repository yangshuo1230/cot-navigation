from pathlib import Path
import json,torch
root=Path(str(next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'pyproject.toml').is_file() and (_p / 'studies').is_dir())))
folder=root/'outputs/task_only_n32_v1/launches/task_only_n32_s0'
rows=[]
for p in (folder/'task_only').iterdir():
 if not p.is_dir():continue
 modes=json.loads((p/'wandb_status.json').read_text()) if (p/'wandb_status.json').exists() else {}
 latest=p/'checkpoints/latest.pt'
 item={'protocol':p.name,'wandb_effective':modes.get('effective'),'wandb_url':modes.get('url')}
 if latest.exists():
  blob=torch.load(latest,map_location='cpu',weights_only=False)
  assert blob['arch']['task_only'] and blob['walks_seen']==0 and blob['walk_sampler'] is None
  item.update(checkpoint_step=blob['step'],walks_seen=blob['walks_seen'],queries_seen=blob['queries_seen'])
  del blob
 vals=sorted((p/'eval').glob('free_0*.json'))
 if vals:
  v=json.loads(vals[-1].read_text());item.update(val_step=v['step'],val_n=v['overall']['n'],val_answer_accuracy=v['overall']['answer_accuracy'])
 rows.append(item)
output=root/'studies/task_only_n32/launch_verification.json'
output.write_text(json.dumps({'status':json.loads((folder/'status.json').read_text()),'task_only':rows},indent=2))
print(json.dumps(rows))

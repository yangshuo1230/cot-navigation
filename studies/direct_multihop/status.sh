#!/usr/bin/env bash
set -euo pipefail
cd /mnt/ningxuefei/yangshuo/cot-navigation
.venv/bin/python - <<'PY'
from pathlib import Path
import json
root=Path('studies/direct_multihop/runs/direct_multihop_v1')
s=json.loads((root/'status.json').read_text());ledger=json.loads((root/'ledger.json').read_text())
print('Updated UTC:',s['updated_at'])
print('Completed:',len(s['completed']),'Running:',len(s['running']),'Pending:',len(s['pending']),'Failed:',s['failed'])
print('Process GPU-hours:',round(sum(v['charged_seconds']+v.get('live_seconds',0) for v in ledger['jobs'].values())/3600,3),'/ 30 allocated')
for name,meta in s['running'].items():
 folder=root/name;files=sorted((folder/'eval').glob('step_*.json'))
 f=json.loads(files[-1].read_text()) if files else {}
 p=folder/'wandb_status.json';wb=json.loads(p.read_text()) if p.exists() else {}
 print(name,'GPU',meta['gpu'],'step',f.get('step'),'stage',f.get('stage'),
       'val by hop',{h:round(v['accuracy'],4) for h,v in f.get('val',{}).items()},'W&B',wb.get('effective'))
PY

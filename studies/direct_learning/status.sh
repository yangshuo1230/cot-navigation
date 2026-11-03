#!/usr/bin/env bash
set -euo pipefail
cd /mnt/ningxuefei/yangshuo/cot-navigation
.venv/bin/python - <<'PY'
from pathlib import Path
import json
root=Path('studies/direct_learning/runs/direct_learning_v1')
s=json.loads((root/'status.json').read_text())
print('Updated UTC:',s['updated_at'])
print('Completed:',len(s['completed']),'Running:',len(s['running']),'Pending:',len(s['pending']),'Failed:',s['failed'])
ledger=json.loads((root/'ledger.json').read_text())
hours=sum(x['charged_seconds']+x.get('live_seconds',0) for x in ledger['jobs'].values())/3600
print('Consumed process GPU-hours (includes live):',round(hours,3),'/ allocated 80')
for name,info in s['running'].items():
    folder=root/name
    reports=sorted((folder/'eval').glob('step_*.json'))
    v=json.loads(reports[-1].read_text()) if reports else {}
    wb=json.loads((folder/'wandb_status.json').read_text()) if (folder/'wandb_status.json').exists() else {}
    print(name,'GPU',info['gpu'],'step',v.get('step'),'stage',v.get('stage'),
          'train',v.get('train',{}).get('accuracy'),'val',v.get('val',{}).get('accuracy'),
          'W&B',wb.get('effective'),wb.get('url'))
PY

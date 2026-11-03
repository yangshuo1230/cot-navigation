"""Start the authorized N32 batch only after every scale audit finishes."""
from pathlib import Path
import json
import os
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
AUDIT=ROOT/'studies/scale_trace_audit'
OUT=ROOT/'studies/task_only_n32'
jobs=json.loads((AUDIT/'plan.json').read_text())['jobs']
workers=json.loads((AUDIT/'launch.json').read_text())
while True:
    missing=[j['name'] for j in jobs if not (AUDIT/'results'/j['name']/'complete.json').exists()]
    if not missing:
        break
    for w in workers:
        owned=[j['name'] for i,j in enumerate(jobs) if i%len(workers)==w['worker']]
        unfinished=[n for n in missing if n in owned]
        if not unfinished:
            continue
        try:
            os.kill(w['pid'],0)
        except ProcessLookupError:
            raise RuntimeError(f'Audit worker {w["worker"]} exited before completion: {unfinished}')
    (OUT/'waiting_for_audit.json').write_text(json.dumps({'remaining_models':len(missing)}))
    time.sleep(15)
argv=['bash','scripts/task_only_n32.sh','--run-name','task_only_n32_s0',
      '--gpus','1','2','3','4','5','6','7','--wait','--background']
subprocess.run(argv,cwd=ROOT,check=True)
(OUT/'started_after_audit.json').write_text(json.dumps({'command':argv,'audited_models':len(jobs),'started_unix':time.time()},indent=2))
print('All 51 frozen audits finished; N32 matched task-only/WALK batch launched.',flush=True)

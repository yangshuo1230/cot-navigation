"""Start the reviewed suite detached, once; do not interrupt other GPU jobs."""
from pathlib import Path
import json
import os
import subprocess
import sys

ROOT=Path(str(next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'pyproject.toml').is_file() and (_p / 'studies').is_dir())))
HERE=ROOT/'studies/fixed_adjacency_suite'
OUT=ROOT/'outputs/exposure_grokking_adj_v3'
assert (HERE/'manifest.json').is_file()
assert (HERE/'train_suite.py').is_file()
import wandb
sys.path.insert(0,str(ROOT/'src'))
from cot_navigation.tracking.logger import _has_wandb_credentials
if not _has_wandb_credentials(wandb):raise RuntimeError('W&B online credentials unavailable')
if not hasattr(wandb.sdk.wandb_run.Run,'offline'):
    raise RuntimeError('W&B runtime differs; audit run.offline before dispatch')
OUT.mkdir(parents=True,exist_ok=True)
receipt=OUT/'launch_receipt.json'
if receipt.exists():
    previous=json.loads(receipt.read_text())
    try:os.kill(previous['pid'],0)
    except ProcessLookupError:pass
    else:raise RuntimeError('Existing scheduler is alive; do not launch twice')
env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','TORCHINDUCTOR_COMPILE_THREADS':'2'}
with (OUT/'launcher.log').open('a') as log:
    command=[sys.executable,str(HERE/'launch.py'),'--repo-root',str(ROOT),'--execute']
    proc=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,
                          stderr=subprocess.STDOUT,start_new_session=True,env=env)
    result=dict(pid=proc.pid,command=command,status_file=str(OUT/'status.json'),
                log=str(OUT/'launcher.log'),formal_launch_requested=True)
    receipt.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

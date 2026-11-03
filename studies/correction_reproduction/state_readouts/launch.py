"""Launch one resumable GPU0 analysis without retaining probe checkpoints."""
from pathlib import Path
import json
import subprocess
import sys
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--early-clean',action='store_true')
parser.add_argument('--legality-transitions',action='store_true')
args=parser.parse_args()

root=Path(__file__).resolve().parents[3]
out=root/'outputs/analysis/state_readouts/legacy6_s0123_supplement_20261102'
if args.early_clean:out=out/'early_clean_lens'
if args.legality_transitions:out=root/'outputs/analysis/state_readouts/legacy6_s0123_supplement_20261102/legality_transitions'
out.mkdir(parents=True,exist_ok=True)
with (out/'run.log').open('a') as log:
    script='legality_transitions.py' if args.legality_transitions else ('early_clean_lens.py' if args.early_clean else 'run.py')
    child=subprocess.Popen([sys.executable,str(Path(__file__).with_name(script))],cwd=root,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(out/'process.json').write_text(json.dumps(dict(pid=child.pid,log=str(out/'run.log'))))
print(child.pid)

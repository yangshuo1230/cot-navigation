"""Sequential natural corpus -> probe training -> compact report, GPU0 only."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

P=Path(__file__).resolve().parent;ROOT=P.parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=5000)
    args=parser.parse_args();os.chdir(ROOT)
    os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['WANDB_MODE']='disabled'
    ready=P/'prepared.json'
    existing=P/'prepare_process.json'
    if not ready.exists() and existing.exists():
        pid=json.loads(existing.read_text())['pid']
        def alive():
            command=Path(f'/proc/{pid}/cmdline')
            return command.exists() and 'natural_history_probes/prepare.py' in command.read_text(errors='replace')
        if alive():
            print('WAIT existing natural preparation',pid,flush=True)
            while not ready.exists() and alive():time.sleep(10)
    if not ready.exists():subprocess.run([sys.executable,'-u',str(P/'prepare.py')],check=True,cwd=ROOT)
    for script,extra in [('run.py',['--steps',str(args.steps)]),('summarize.py',[]),('report.py',[])]:
        print('STAGE',script,flush=True)
        subprocess.run([sys.executable,'-u',str(P/script),*extra],check=True,cwd=ROOT)
    (P/'complete.json').write_text(json.dumps(dict(complete=True,finished=time.time())))


if __name__=='__main__':main()

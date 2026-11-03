"""Finish the optional OOD report when existing frozen workers complete."""
from pathlib import Path
import os
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
while len(list((HERE/'results').rglob('complete.json')))<51:
    time.sleep(15)
for script in ['postprocess.py','summarize.py']:
    subprocess.run([sys.executable,str(HERE/script)],cwd=HERE.parents[1],
                   env={**os.environ,'CUDA_VISIBLE_DEVICES':''},check=True)
print('All 84 audit conditions complete; final report refreshed.',flush=True)

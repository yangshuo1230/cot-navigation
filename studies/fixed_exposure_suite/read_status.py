from pathlib import Path
import json
import subprocess
ROOT=Path(str(next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'pyproject.toml').is_file() and (_p / 'studies').is_dir())))
OUT=ROOT/'outputs/exposure_grokking_v2'
if (OUT/'status.json').exists():print('STATUS', (OUT/'status.json').read_text())
print('LAUNCHER_TAIL','\n'.join((OUT/'launcher.log').read_text().splitlines()[-12:]))
for p in sorted(OUT.glob('*.log')):
    if p.name=='launcher.log':continue
    lines=p.read_text(errors='replace').splitlines()
    bad=[line for line in lines if 'Traceback' in line or 'Error:' in line]
    print('WORKER',p.name,'errors',bad[-3:],'tail',lines[-3:])
for p in sorted(OUT.glob('**/metrics.jsonl')):
    lines=p.read_text().splitlines()
    if lines:
        row=json.loads(lines[-1])
        print('PROGRESS',str(p.parent.relative_to(OUT)),{k:row.get(k) for k in ['step','phase','loss','elapsed']})
print(subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True))

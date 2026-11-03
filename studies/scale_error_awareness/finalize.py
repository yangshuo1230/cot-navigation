"""Finish this one-off study after its analysis process completes."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
STUDY=Path(__file__).resolve().parent
OUT=STUDY/'results'
PYTHON=ROOT/'.venv/bin/python'


def main():
    pid=int((STUDY/'pid').read_text())
    while True:
        if len(list((OUT/'models').glob('*.json')))==24 and 'COMPLETE' in (STUDY/'run.log').read_text()[-100:]:
            break
        proc=Path(f'/proc/{pid}')
        if not proc.exists() or proc.joinpath('stat').read_text().split()[2]=='Z':
            raise RuntimeError('Analysis exited before completing 24 models; inspect run.log')
        time.sleep(15)
    commands=[
        ['-m','pytest','tests/test_scale_error_awareness.py','tests/test_error_awareness.py',
         'tests/test_mechanistic_followup.py','tests/test_mechanistic_analysis.py','-q'],
        ['-m','cot_navigation.analysis.verify_scale_error_awareness'],
        ['-m','cot_navigation.analysis.report_scale_error_awareness',str(OUT)],
    ]
    for command in commands:
        subprocess.run([str(PYTHON),*command],cwd=ROOT,check=True)
    files={str(p.relative_to(STUDY)):p for p in OUT.rglob('*') if p.is_file()}
    for name in ['scale_error_awareness.py','report_scale_error_awareness.py','verify_scale_error_awareness.py']:
        files['source/'+name]=ROOT/'src/cot_navigation/analysis'/name
    files['source/test_scale_error_awareness.py']=ROOT/'tests/test_scale_error_awareness.py'
    files['source/finalize.py']=STUDY/'finalize.py'
    if (STUDY/'run_parallel.py').exists():
        files['source/run_parallel.py']=STUDY/'run_parallel.py'
    files['provenance/reference_sampler_source.py']=STUDY/'reference_sampler_source.py'
    manifest={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in sorted(files.items())}
    (STUDY/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('FINALIZED',len(manifest),'files',flush=True)


if __name__=='__main__':
    main()

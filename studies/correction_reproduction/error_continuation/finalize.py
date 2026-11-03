"""Refresh the same compact report when another model/graph task completes."""
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import time

P=Path(__file__).resolve().parent


def main():
    pid=int(sys.argv[1]);previous=-1;generation=0
    while True:
        try:
            running=b'error_continuation/run.py' in Path(f'/proc/{pid}/cmdline').read_bytes()
        except FileNotFoundError:
            running=False
        files=list((P/'results').glob('*.json.gz'))
        complete=sum(bool(json.load(gzip.open(f,'rt'))['complete']) for f in files)
        if complete!=previous or not running:
            result=subprocess.run([sys.executable,str(P/'report.py')],cwd=P.parents[2],check=False)
            generation+=1
            state=dict(generation=generation,running=running,complete_tasks=complete,total_tasks=40,
                complete=complete==40 and result.returncode==0,report_returncode=result.returncode,
                failed=not running and complete!=40)
            temp=P/'finalization.tmp';temp.write_text(json.dumps(state),encoding='utf-8');temp.replace(P/'finalization.json')
            previous=complete
        if not running:break
        time.sleep(30)


if __name__=='__main__':main()

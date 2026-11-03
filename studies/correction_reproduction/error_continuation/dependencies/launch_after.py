"""Run detailed natural replay after the original GPU0 study releases the GPU."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

P=Path(__file__).resolve().parent


def main():
    old_pid=int(sys.argv[1])
    def state(**kw):
        temp=P/'status.tmp';temp.write_text(json.dumps(kw));temp.replace(P/'status.json')
    state(phase='queued',waiting_for_pid=old_pid)
    while True:
        try:running=b'error_continuation/run.py' in Path(f'/proc/{old_pid}/cmdline').read_bytes()
        except FileNotFoundError:running=False
        if not running:break
        time.sleep(30)
    subprocess.run([sys.executable,str(P.parent/'length_analysis.py')],check=True)
    with (P/'run.log').open('a') as log:
        child=subprocess.Popen([sys.executable,'-u',str(P/'run.py'),'--seeds','0','1'],cwd=P.parents[3],stdout=log,stderr=subprocess.STDOUT)
        state(phase='running',pid=child.pid)
        previous=-1;generation=0
        while True:
            files=list((P/'results').glob('*.json.gz'))
            import gzip
            completed=sum(bool(json.load(gzip.open(f,'rt'))['complete']) for f in files)
            code=child.poll()
            if completed!=previous or code is not None:
                result=subprocess.run([sys.executable,str(P/'report.py')],check=False)
                generation+=1
                state(phase='complete' if code==0 and completed==20 and result.returncode==0 else 'failed' if code is not None else 'running',
                    pid=child.pid,generation=generation,complete_tasks=completed,total_tasks=20,returncode=code)
                previous=completed
            if code is not None:break
            time.sleep(30)


if __name__=='__main__':main()

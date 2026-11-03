from types import SimpleNamespace
from pathlib import Path
import sys
sys.path.insert(0,'scripts')
import task_only_n32 as t
args=SimpleNamespace(run_name='task_only_n32_s0',protocols=list(t.PROTOCOLS),steps=50000,tracking='online',with_walk_control=True)
out,jobs=t.build_jobs(args)
t.verify_cached_data(jobs,Path('studies/task_only_n32'),{})
print('All 13 job configs, cached query/WALK data, splits and vocab verified')

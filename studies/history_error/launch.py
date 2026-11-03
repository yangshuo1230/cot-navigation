"""Start the controlled pilot on idle GPU7; never preempt other jobs."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys

root = Path.cwd()
study = root / "studies/history_error"
lock = (study / "launch.lock").open("w")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
raw = subprocess.check_output(["nvidia-smi", "--id=7", "--query-gpu=memory.used,utilization.gpu",
                               "--format=csv,noheader,nounits"], text=True)
memory, utilization = map(int, raw.strip().split(","))
if memory > 500 or utilization > 10:
    raise RuntimeError(f"GPU7 busy: {raw}")
fp32 = "--fp32" in sys.argv
with (study / ("fp32.log" if fp32 else "run.log")).open("w") as log:
    command = [sys.executable, "-u", str(study / "run.py"), "--gpu", "7"]
    if fp32:
        command += ["--precision", "fp32", "--output", "studies/history_error/results_fp32"]
    process = subprocess.Popen(command,
                               cwd=root, stdout=log, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, start_new_session=True)
(study / ("process_fp32.json" if fp32 else "process.json")).write_text(json.dumps({"pid": process.pid, "gpu": 7,
    "models": ["context_l10_dfs_state", "context_l10_bfs_state", "fixed_l10_dfs_state", "fixed_l10_bfs_state"]}, indent=2) + "\n")
print("STARTED", process.pid, "GPU7", flush=True)

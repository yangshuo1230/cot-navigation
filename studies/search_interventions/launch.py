"""Run independent frozen inference jobs on the currently idle server GPUs."""
import json
from pathlib import Path
import subprocess
import sys

root = Path.cwd()
folder = root / "studies/search_interventions"
entries = json.loads((root / "studies/scale_mechanistic/results/selection.json").read_text())["models"]
names = [e["id"] for e in entries if e["protocol"] in {"dfs_state", "bfs_state"}]
jobs = []
for gpu in range(8):
    models = names[gpu::8]
    log = folder / f"gpu{gpu}.log"
    with log.open("w") as stream:
        process = subprocess.Popen([sys.executable, str(folder / "run.py"), "--gpu", str(gpu),
                                    "--cases", "24", "--batch", "48", "--models", *models],
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    jobs.append({"gpu": gpu, "pid": process.pid, "models": models, "log": str(log)})
(folder / "processes.json").write_text(json.dumps(jobs, indent=2) + "\n")
print(json.dumps(jobs, indent=2))

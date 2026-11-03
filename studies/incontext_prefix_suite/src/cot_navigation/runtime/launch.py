"""Independent single-GPU launcher.

The default is a dry run. A GPU with any compute process is not idle, and this
launcher will not start work on it. It does not start a multi-GPU DDP job.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from cot_navigation.runtime.envcheck import compute_apps, nvidia_smi_gpus


def idle_gpus(min_free_mb: float = 20000) -> list[dict]:
    gpus = nvidia_smi_gpus()
    busy = {app["uuid"] for app in compute_apps()}
    idle = []
    for gpu in gpus:
        gpu = dict(gpu)
        gpu["busy"] = gpu["uuid"] in busy
        gpu["idle"] = (not gpu["busy"]) and gpu["memory_free_mb"] >= min_free_mb
        idle.append(gpu)
    return idle


def plan_launch(commands: list[str], gpus: list[int] | None, output_root: str, dry_run: bool = True) -> dict:
    snapshot = idle_gpus()
    free = [row["index"] for row in snapshot if row["idle"]]
    idle_set = set(free)
    chosen = list(gpus) if gpus is not None else free[: len(commands)]
    blocked = [gpu for gpu in chosen if gpu not in idle_set]
    report = {"dry_run": dry_run, "gpus": snapshot, "assignments": [], "started": False}
    if blocked or len(chosen) < len(commands):
        report["error"] = (
            f"need {len(commands)} idle GPUs, found {len(free)} idle "
            f"(requested {chosen}, busy among them {blocked}). Refusing to share a busy GPU."
        )
        report["exit_code"] = 2
        return report
    root = Path(output_root)
    for command, gpu in zip(commands, chosen):
        argv = shlex.split(command)
        ephemeral = "smoke" in argv or any(Path(arg).stem.startswith("smoke_") for arg in argv)
        log_path = root / f"gpu{gpu}.log"
        pid_path = root / f"gpu{gpu}.pid"
        env_assignment = f"CUDA_VISIBLE_DEVICES={gpu}"
        report["assignments"].append(
            {
                "gpu": gpu,
                "command": command,
                "log": None if ephemeral else str(log_path),
                "pid_file": None if ephemeral else str(pid_path),
                "env": env_assignment,
            }
        )
    if dry_run:
        report["exit_code"] = 0
        return report
    if any(item["log"] for item in report["assignments"]):
        root.mkdir(parents=True, exist_ok=True)
    for item in report["assignments"]:
        log_handle = open(item["log"], "w") if item["log"] else None
        proc = subprocess.Popen(
            shlex.split(item["command"]),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(item["gpu"])},
            start_new_session=True,
        )
        if item["pid_file"]:
            Path(item["pid_file"]).write_text(str(proc.pid) + "\n")
        if log_handle:
            log_handle.close()
        item["pid"] = proc.pid
    report["started"] = True
    report["exit_code"] = 0
    return report


def main_preview(argv: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in [sys.executable, "-m", "cot_navigation.cli", *argv])

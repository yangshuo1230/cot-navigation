#!/usr/bin/env python3
"""Formal single-GPU experiment runner; never runs smoke or a timing job.

Build/validate shared datasets once, schedule independent runs on idle GPUs,
and initialize every fixed-graph protocol from its common knowledge model.
Use --dry-run to inspect the plan without creating files or starting jobs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cot_navigation.config import load_config, to_plain
from cot_navigation.runtime.launch import idle_gpus

MODES = ("direct", "path", "dfs_state", "filler")
# Filler is retired from new experiment batches. Keep explicit selection so
# historical four-protocol plans can still be resumed without changing them.
DEFAULT_MODES = ("direct", "path", "dfs_state")


@dataclass
class Job:
    name: str
    config: Path
    overrides: list[str]
    output: Path
    steps: int
    depends_on: str | None = None

    def command(self, checkpoint: Path | None = None, parent: Path | None = None) -> list[str]:
        argv = [sys.executable, "-m", "cot_navigation.cli", "train", "--config", str(self.config)]
        for value in self.overrides:
            argv += ["--set", value]
        if checkpoint is not None:
            argv += ["--resume", str(checkpoint)]
        elif parent is not None:
            argv += ["--init-from", str(parent)]
        return argv


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--setting", choices=("context", "fixed", "all"), default="all")
    p.add_argument("--protocols", nargs="+", choices=MODES, default=list(DEFAULT_MODES))
    p.add_argument("--graph-seeds", nargs="+", type=int, default=[0], help="Fixed-graph seeds; default: 0")
    p.add_argument("--model-seed", type=int, default=0)
    p.add_argument("--gpus", nargs="+", type=int, help="Physical GPU indices; default: all visible in nvidia-smi")
    p.add_argument("--wait", action="store_true", help="Wait for an idle GPU if none are available")
    p.add_argument("--run-name", default=None, help="Stable name required with --resume")
    p.add_argument("--output-root", default="outputs/first_round_v2/launches")
    p.add_argument("--resume", action="store_true", help="Skip finished runs and resume latest checkpoints")
    p.add_argument("--dry-run", action="store_true", help="Print plan only; no data, run files or GPU work")
    p.add_argument("--background", action="store_true", help="Create log directories and start a detached launcher")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tracking", choices=("online", "offline", "disabled"), default="online")
    p.add_argument("--context-steps", type=int, help="Default from config: 2000")
    p.add_argument("--knowledge-steps", type=int, help="Default from config: 1000")
    p.add_argument("--task-steps", type=int, help="Default from config: 2000")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Additional typed config override")
    args = p.parse_args(argv)
    if args.resume and not args.run_name:
        p.error("--resume requires the original --run-name")
    for key in ("context_steps", "knowledge_steps", "task_steps"):
        if getattr(args, key) is not None and getattr(args, key) < 1:
            p.error(f"--{key.replace('_', '-')} must be positive")
    owned = {"output_dir", "protocol.name", "train.stage", "train.max_steps", "train.model_seed",
             "data.graph_seed", "runtime.smoke", "train.persist_outputs", "tracking.mode"}
    for override in args.set:
        if "=" not in override or override.split("=", 1)[0] in owned:
            p.error(f"runner owns this option or override is invalid: {override}")
    args.protocols = list(dict.fromkeys(args.protocols))
    args.graph_seeds = list(dict.fromkeys(args.graph_seeds))
    if args.gpus is not None and (len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 0):
        p.error("--gpus must contain distinct nonnegative indices")
    args.run_name = args.run_name or datetime.now(ZoneInfo("Asia/Shanghai")).strftime("run_%Y%m%d_%H%M%S")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_name) or args.run_name in {".", ".."}:
        p.error("--run-name must be a plain name using letters, digits, underscore, dot or hyphen")
    return args


def build_jobs(args, root=ROOT) -> tuple[Path, list[Job]]:
    output = Path(args.output_root)
    output = (output if output.is_absolute() else root / output) / args.run_name
    shared = list(args.set) + [f"model.compile={str(args.compile).lower()}",
        f"tracking.mode={args.tracking}", f"train.model_seed={args.model_seed}",
        "runtime.device=cuda", "runtime.smoke=false", "train.persist_outputs=true",
        f"tracking.group={json.dumps(args.run_name)}"]
    jobs = []

    def add(name, preset, steps, dependency=None, extra=()):
        out = output / name
        overrides = shared + list(extra) + [f"output_dir={json.dumps(str(out))}",
                                            f"tracking.name={json.dumps(args.run_name + '/' + name)}"]
        if steps is not None:
            overrides += [f"train.max_steps={steps}"]
        cfg_path = root / "configs/experiments" / (preset + ".yaml")
        cfg = load_config([cfg_path], overrides)
        if cfg.train.warmup_steps >= cfg.train.max_steps:
            raise ValueError(f"{name}: warmup must be shorter than training; override train.warmup_steps")
        jobs.append(Job(name, cfg_path, overrides, out, cfg.train.max_steps, dependency))

    if args.setting in {"context", "all"}:
        for mode in args.protocols:
            add(f"context/{mode}", f"main_context_{mode}", args.context_steps)
    if args.setting in {"fixed", "all"}:
        fixed = load_config([root / "configs/experiments/main_stage_a.yaml"], args.set)
        for seed in args.graph_seeds:
            if seed not in fixed.data.graph_seeds:
                raise ValueError(f"graph seed {seed} is not in generated data.graph_seeds={fixed.data.graph_seeds}")
            parent = f"fixed/seed{seed}/stage_a"
            add(parent, "main_stage_a", args.knowledge_steps, extra=[f"data.graph_seed={seed}"])
            for mode in args.protocols:
                add(f"fixed/seed{seed}/stage_b/{mode}", f"main_stage_b_{mode}", args.task_steps,
                    parent, [f"data.graph_seed={seed}"])
    return output, jobs


def plan_record(jobs: list[Job]) -> dict:
    return {job.name: {"depends_on": job.depends_on,
                      "config": to_plain(load_config([job.config], job.overrides))} for job in jobs}


def check_resume_plan(previous: dict, current: dict):
    """Only extend step counts; execution backend changes do not alter weights."""
    old = json.loads(json.dumps(previous))
    new = json.loads(json.dumps(current))
    if set(old) != set(new):
        raise ValueError("resume must use the original settings, protocols and graph seeds")
    for name in new:
        a, b = old[name]["config"], new[name]["config"]
        # These newly introduced knobs preserve the historical behavior at
        # their defaults. Old launch plans omitted them, unlike new resolved
        # configs; normalize only these known additions before comparison.
        additions = {
            "protocol": {"negative_random_path": False, "random_search_order": False},
            "train": {"shuffle_context_edges": False, "query_hops": "all",
                      "curriculum_stage_steps": [], "walk_loss_mode": "fixed",
                      "walk_loss_weight": 1.0, "walk_balance_decay": 0.98},
            "tracking": {"variant": "baseline"},
        }
        for section, defaults in additions.items():
            for cfg in (a, b):
                for key, value in defaults.items():
                    cfg[section].setdefault(key, value)
        if a["train"]["stage"] == "A" and b["train"]["max_steps"] != a["train"]["max_steps"]:
            raise ValueError("knowledge-stage length is fixed for existing branches; use a new --run-name")
        if b["train"]["max_steps"] < a["train"]["max_steps"]:
            raise ValueError("resume cannot decrease training steps")
        a["train"].pop("max_steps"); b["train"].pop("max_steps")
        for cfg in (a, b):
            cfg["model"].pop("compile"); cfg["model"].pop("compile_mode")
    if old != new:
        raise ValueError("resume configuration differs; use original options or a new --run-name")


@contextmanager
def file_lock(path: Path, *, blocking=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            raise RuntimeError(f"another launcher owns {path}") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def prepare_data(jobs, output, env):
    groups = {}
    for job in jobs:
        cfg = load_config([job.config], job.overrides)
        root = Path(cfg.data.root)
        entry = groups.setdefault(root, {"job": job, "kind": cfg.data.knowledge, "protocols": set()})
        if cfg.train.stage != "A":
            entry["protocols"].add(cfg.protocol.name)
    for root, entry in groups.items():
        job = entry["job"]
        # prepare owns the dataset lock and reuses a verified receipt. No
        # repeated generation/full DFS audit on every subsequent launch.
        argv = [sys.executable, "-m", "cot_navigation.cli", "data", "prepare",
                "--config", str(job.config), "--protocols", *sorted(entry["protocols"] or {"direct"})]
        for value in job.overrides:
            argv += ["--set", value]
        log = output / f"data_{entry['kind']}_{root.name}_prepare.log"
        print(f"Data prepare/reuse: {root} -> {log}", flush=True)
        with log.open("w") as handle:
            subprocess.run(argv, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def checkpoint_step(path):
    from cot_navigation.training.checkpoint import load_checkpoint
    return int(load_checkpoint(path, map_location="cpu")["step"])


def job_complete(job):
    """A full checkpoint is not completion if required test reports failed."""
    latest = job.output / "checkpoints/latest.pt"
    if not latest.exists() or checkpoint_step(latest) < job.steps:
        return False
    cfg = load_config([job.config], job.overrides)
    if cfg.train.stage != "A":
        for tag in cfg.evaluation.final_checkpoints:
            path = job.output / "eval" / f"free_test_{tag}.json"
            try:
                report = json.loads(path.read_text())
                if report.get("training_step", -1) < job.steps or report.get("overall", {}).get("n", 0) < 1:
                    return False
            except (OSError, ValueError):
                return False
    return True


def progress(job, started, resumed_step=0, metrics_mtime=0):
    path = job.output / "metrics.jsonl"
    if not path.exists() or path.stat().st_mtime <= metrics_mtime:
        return f"{job.name}: initializing / compiling"
    try:
        row = json.loads(path.read_text().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return f"{job.name}: waiting for next metrics row"
    step = int(row["step"])
    elapsed = time.monotonic() - started
    message = f"{job.name}: {step}/{job.steps}, elapsed {elapsed/60:.1f} min"
    new_steps = step - resumed_step
    if new_steps >= 100:
        # Includes startup and any validation already seen. Future validation
        # can cost more, so this is explicitly a rough ETA, not a promise.
        remaining = elapsed * max(0, job.steps-step) / new_steps
        message += f", rough ETA {remaining/60:.1f} min"
    return message


def run_jobs(args, jobs, output, selected, env):
    done, pending, running = set(), list(jobs), {}
    lookup = {job.name: job for job in jobs}
    checkpoints = {}
    for job in jobs:
        latest = job.output / "checkpoints/latest.pt"
        if args.resume and latest.exists():
            checkpoints[job.name] = latest
            if job_complete(job):
                done.add(job.name); pending.remove(job)
                print(f"Already completed: {job.name}", flush=True)
        elif job.output.exists() and any(job.output.iterdir()):
            raise RuntimeError(f"{job.output} has records but no resumable checkpoint; use a new --run-name")
    last_status = 0
    try:
        while pending or running:
            for gpu, item in list(running.items()):
                code = item["proc"].poll()
                if code is None:
                    continue
                item["log"].close()
                del running[gpu]
                if code:
                    raise RuntimeError(f"{item['job'].name} exited {code}; see {item['job'].output/'train.log'}")
                latest = item["job"].output / "checkpoints/latest.pt"
                if not job_complete(item["job"]):
                    raise RuntimeError(f"{item['job'].name} did not finish its checkpoint/test reports")
                done.add(item["job"].name)
                print(f"Finished {item['job'].name} in {(time.monotonic()-item['started'])/60:.1f} min", flush=True)
            ready = [job for job in pending if job.depends_on is None or job.depends_on in done]
            idle = [row["index"] for row in idle_gpus() if row["idle"] and row["index"] in selected
                    and row["index"] not in running] if ready else []
            for gpu, job in zip(idle, ready):
                parent = lookup[job.depends_on].output / "checkpoints/latest.pt" if job.depends_on else None
                argv = job.command(checkpoints.get(job.name), parent)
                job.output.mkdir(parents=True, exist_ok=True)
                log = (job.output / "train.log").open("a")
                metrics = job.output / "metrics.jsonl"
                metrics_mtime = metrics.stat().st_mtime if metrics.exists() else 0
                resumed_step = checkpoint_step(checkpoints[job.name]) if job.name in checkpoints else 0
                started = time.monotonic()
                try:
                    proc = subprocess.Popen(argv, cwd=ROOT, env={**env, "CUDA_VISIBLE_DEVICES": str(gpu)},
                                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                except BaseException:
                    log.close()
                    raise
                running[gpu] = {"proc": proc, "job": job, "log": log, "started": started,
                                "resumed_step": resumed_step, "metrics_mtime": metrics_mtime}
                pending.remove(job)
                print(f"Started {job.name} on GPU {gpu}, PID {proc.pid}: {shlex.join(argv)}", flush=True)
            if pending and not running and not args.wait:
                raise RuntimeError("No selected GPU is idle. Free a card, choose another --gpus list, or add --wait.")
            if time.monotonic() - last_status >= 30:
                for item in running.values():
                    print(progress(item["job"], item["started"], item["resumed_step"], item["metrics_mtime"]), flush=True)
                if pending and not running:
                    print("Waiting for an idle GPU; existing jobs will not be interrupted.", flush=True)
                write_json(output / "status.json", {"completed": sorted(done),
                    "running": {str(gpu): {"job": x["job"].name, "pid": x["proc"].pid} for gpu,x in running.items()},
                    "pending": [job.name for job in pending]})
                last_status = time.monotonic()
            if pending or running:
                time.sleep(5)
        write_json(output / "status.json", {"completed": sorted(done), "running": {}, "pending": []})
    finally:
        # Only terminate process groups spawned by this invocation.
        for item in running.values():
            if item["proc"].poll() is None:
                os.killpg(item["proc"].pid, signal.SIGTERM)
        for item in running.values():
            try:
                item["proc"].wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(item["proc"].pid, signal.SIGKILL)
                item["proc"].wait()
            item["log"].close()


def start_background(args, output, argv, script_path=None):
    """Open the log after creating its directory, before spawning the launcher.

    External shell redirection happens before this script can run. Owning the
    redirection here makes a fresh checkout work without a manual mkdir/nohup.
    Only the launcher is detached; it retains ownership of its training jobs.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    log = output.parent / f"{args.run_name}.launch.log"
    child_args = [value for value in argv if value != "--background"]
    # Preserve a generated timestamp name across the parent/child boundary.
    child_args += ["--run-name", args.run_name]
    with log.open("a") as handle:
        proc = subprocess.Popen([sys.executable, str(script_path or Path(__file__).resolve()), *child_args],
                                cwd=ROOT, env={**os.environ, "PYTHONUNBUFFERED": "1"},
                                stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                                start_new_session=True)
    print(f"Background launcher PID: {proc.pid}\nLog: {log}\n"
          f"Follow: tail -f {shlex.quote(str(log))}", flush=True)
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    os.chdir(ROOT)
    output, jobs = build_jobs(args)
    return execute(args, output, jobs, argv)


def execute(args, output, jobs, argv, *, script_path=None):
    """Shared scheduling engine; data suites supply their own immutable plan."""
    inventory = idle_gpus()
    selected = args.gpus if args.gpus is not None else [row["index"] for row in inventory]
    unknown = set(selected) - {row["index"] for row in inventory}
    if unknown:
        raise ValueError(f"GPU indices do not exist: {sorted(unknown)}")
    plan = plan_record(jobs)
    lookup = {job.name: job for job in jobs}
    print(f"Run: {args.run_name}\nOutput: {output}\nJobs: {len(jobs)}", flush=True)
    for job in jobs:
        parent = lookup[job.depends_on].output / "checkpoints/latest.pt" if job.depends_on else None
        print(f"  {job.name}: {job.steps} updates; depends on {job.depends_on or '-'}", flush=True)
        if args.dry_run:
            print("    " + shlex.join(job.command(parent=parent)), flush=True)
    print("Idle selected GPUs: " + str([row["index"] for row in inventory if row["idle"] and row["index"] in selected]), flush=True)
    if args.dry_run:
        return 0
    if not selected or (not args.wait and not any(row["idle"] and row["index"] in selected for row in inventory)):
        raise RuntimeError("No selected GPU is idle; nothing started. Use --wait to queue.")
    if output.exists() and not args.resume:
        raise RuntimeError(f"{output} already exists; use --resume with matching options or a new --run-name")
    if args.resume:
        path = output / "launch_plan.json"
        if not path.exists():
            raise RuntimeError("--resume requires an existing launch_plan.json")
        check_resume_plan(json.loads(path.read_text()), plan)
    if args.background:
        return start_background(args, output, argv, script_path)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "2"),
           "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "2"),
           "TORCHINDUCTOR_COMPILE_THREADS": os.environ.get("TORCHINDUCTOR_COMPILE_THREADS", "2")}
    with file_lock(output / ".launch.lock", blocking=False):
        write_json(output / "launch_plan.json", plan)
        prepare_data(jobs, output, env)
        run_jobs(args, jobs, output, selected, env)
    return 0


if __name__ == "__main__":
    def interrupt(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; use the same --run-name and --resume after a checkpoint has been saved.", file=sys.stderr)
        raise SystemExit(130)
    except (RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)

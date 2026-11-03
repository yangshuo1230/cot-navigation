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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from cot_navigation.config import load_config, to_plain
from cot_navigation.runtime.launch import idle_gpus

MODES = ("direct", "path", "dfs_state", "filler")
# Filler is retired from new experiment batches. Keep explicit selection so
# historical four-protocol plans can still be resumed without changing them.
DEFAULT_MODES = ("direct", "path", "dfs_state")


from cot_navigation.experiments.scheduler import *

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


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    os.chdir(ROOT)
    output, jobs = build_jobs(args)
    return execute(args, output, jobs, argv, script_path=ROOT / 'scripts/train_first_round.py')



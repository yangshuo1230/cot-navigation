#!/usr/bin/env python3
"""Retrain with controlled query coverage and independent context graph counts.

No training is started with --dry-run or --prepare-only. Each job owns one GPU;
the scheduler reuses idle cards and each fixed graph has one shared Stage A.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import train_first_round as engine
from cot_navigation.config import load_config

VARIANTS = {
    # Balanced means the original per-distance downsample. Coverage retains
    # every multi-hop training pair and balances labels by cycling both pools.
    "legacy": ("balanced", False, False),
    "coverage": ("full", True, False),
    "onehop": ("balanced", True, True),
    "combined": ("full", True, True),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--setting", choices=("all", "context", "fixed"), default="all")
    p.add_argument("--protocols", nargs="+", choices=engine.MODES, default=list(engine.DEFAULT_MODES))
    p.add_argument("--context-sizes", nargs="+", type=int, default=[10000, 50000, 150000])
    p.add_argument("--fixed-variants", nargs="+", choices=tuple(VARIANTS), default=["combined"])
    p.add_argument("--one-hop-fraction", type=float, default=0.25)
    p.add_argument("--graph-seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--model-seed", type=int, default=0)
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--context-val-size", type=int, default=10000)
    p.add_argument("--context-test-size", type=int, default=10000)
    p.add_argument("--context-steps", type=int, default=50000)
    p.add_argument("--knowledge-steps", type=int, default=1000)
    p.add_argument("--task-steps", type=int, default=50000)
    p.add_argument("--gpus", nargs="+", type=int)
    p.add_argument("--wait", action="store_true")
    p.add_argument("--run-name")
    p.add_argument("--output-root", default="outputs/retrain_v3/launches")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--background", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--prepare-only", action="store_true", help="prepare verified datasets/caches; no run directory or GPUs")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tracking", choices=("online", "offline", "disabled"), default="online")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = p.parse_args(argv)
    if args.resume and not args.run_name:
        p.error("--resume requires the original --run-name")
    if args.prepare_only and (args.resume or args.background):
        p.error("--prepare-only is separate from --resume/--background")
    for key in ("context_sizes",):
        if any(n < 2 or n % 2 for n in getattr(args, key)):
            p.error("context sizes must be positive even counts")
    if any(getattr(args, k) < 2 or getattr(args, k) % 2 for k in ("context_val_size", "context_test_size")):
        p.error("context val/test sizes must be positive even counts")
    if any(getattr(args, k) < 1 for k in ("context_steps", "knowledge_steps", "task_steps")):
        p.error("training steps must be positive")
    if not 0 < args.one_hop_fraction < 1:
        p.error("--one-hop-fraction must be in (0,1); select coverage/legacy for zero")
    if args.gpus is not None and (len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 0):
        p.error("--gpus must contain distinct nonnegative indices")
    owned = {"output_dir", "protocol.name", "train.stage", "train.max_steps", "train.model_seed",
             "data.graph_seed", "data.data_seed", "data.root", "data.splits.train", "data.splits.val",
             "data.splits.test", "data.include_one_hop_queries", "data.context_holdout_first",
             "train.fixed_query_pool", "train.balance_query_labels", "train.one_hop_fraction",
             "runtime.smoke", "train.persist_outputs", "tracking.mode", "tracking.name", "tracking.group"}
    for item in args.set:
        if "=" not in item or item.split("=", 1)[0] in owned:
            p.error(f"runner owns this option or override is invalid: {item}")
    for key in ("protocols", "context_sizes", "graph_seeds", "fixed_variants"):
        setattr(args, key, list(dict.fromkeys(getattr(args, key))))
    args.run_name = args.run_name or datetime.now(ZoneInfo("Asia/Shanghai")).strftime("retrain_%Y%m%d_%H%M%S")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_name) or args.run_name in {".", ".."}:
        p.error("--run-name must be a plain name")
    return args


def build_jobs(args, root=ROOT):
    output = Path(args.output_root)
    output = (output if output.is_absolute() else root / output) / args.run_name
    shared = args.set + [f"model.compile={str(args.compile).lower()}", f"tracking.mode={args.tracking}",
        f"train.model_seed={args.model_seed}", f"data.data_seed={args.data_seed}",
        "runtime.device=cuda", "runtime.smoke=false", "train.persist_outputs=true",
        f"tracking.group={json.dumps(args.run_name)}"]
    jobs = []

    def add(name, preset, steps, extra, parent=None):
        out = output / name
        overrides = shared + list(extra) + [f"train.max_steps={steps}",
            f"output_dir={json.dumps(str(out))}", f"tracking.name={json.dumps(args.run_name+'/'+name)}"]
        path = root / "configs/experiments" / (preset + ".yaml")
        cfg = load_config([path], overrides)
        if cfg.train.warmup_steps >= cfg.train.max_steps:
            raise ValueError(f"{name}: warmup must be shorter than training")
        jobs.append(engine.Job(name, path, overrides, out, steps, parent))

    if args.setting in {"all", "context"}:
        for n in args.context_sizes:
            data_root = (f"data/retrain_v3/context_n16_m24_train{n}_val{args.context_val_size}"
                         f"_test{args.context_test_size}_s{args.data_seed}")
            for mode in args.protocols:
                add(f"context/train{n}/{mode}", "retrain_context", args.context_steps,
                    [f"data.root={data_root}", f"data.splits.train={n}",
                     f"data.splits.val={args.context_val_size}", f"data.splits.test={args.context_test_size}",
                     f"protocol.name={mode}"])
    if args.setting in {"all", "fixed"}:
        data_root = f"data/retrain_v3/parametric_n32_m48_onehop_s{args.data_seed}"
        for seed in args.graph_seeds:
            extra = [f"data.root={data_root}", f"data.graph_seed={seed}"]
            cfg = load_config([root / "configs/experiments/retrain_stage_a.yaml"], shared + extra)
            if seed not in cfg.data.graph_seeds:
                raise ValueError(f"graph seed {seed} is not in data.graph_seeds={cfg.data.graph_seeds}")
            parent = f"fixed/seed{seed}/stage_a"
            add(parent, "retrain_stage_a", args.knowledge_steps, extra)
            for variant in args.fixed_variants:
                pool, balance, one_hop = VARIANTS[variant]
                for mode in args.protocols:
                    add(f"fixed/seed{seed}/stage_b/{variant}/{mode}", "retrain_stage_b", args.task_steps,
                        extra + [f"protocol.name={mode}", f"train.fixed_query_pool={pool}",
                                 f"train.balance_query_labels={str(balance).lower()}",
                                 f"train.one_hop_fraction={args.one_hop_fraction if one_hop else 0.0}"], parent)
    return output, jobs


def prepare_only(jobs):
    from cot_navigation.data.prepared import prepare_dataset
    groups = {}
    for job in jobs:
        cfg = load_config([job.config], job.overrides)
        entry = groups.setdefault(cfg.data.root, {"cfg": cfg, "protocols": set()})
        if cfg.train.stage != "A":
            entry["protocols"].add(cfg.protocol.name)
    for root, entry in groups.items():
        report = prepare_dataset(entry["cfg"], sorted(entry["protocols"] or {"direct"}))
        print(json.dumps({"root": root, "reused": report["reused"], "validation": report["validation"]}), flush=True)
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    os.chdir(ROOT)
    output, jobs = build_jobs(args)
    if args.prepare_only and not args.dry_run:
        return prepare_only(jobs)
    return engine.execute(args, output, jobs, argv, script_path=Path(__file__).resolve())


if __name__ == "__main__":
    def interrupt(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; resume with the same run-name and options.", file=sys.stderr)
        raise SystemExit(130)
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)

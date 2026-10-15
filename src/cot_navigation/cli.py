"""Command line.

Overrides use ``--set dotted.key=value`` and are parsed as YAML scalars.
``--dry-run`` prints the resolved config and does not generate, train, or launch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cot_navigation.config import dump_config, load_config
from cot_navigation.data.generate import generate_context, generate_parametric
from cot_navigation.data.splits import per_distance_quotas
from cot_navigation.data.validate import validate_context, validate_parametric
from cot_navigation.runtime.benchmark import run_benchmark
from cot_navigation.runtime.envcheck import probe_environment
from cot_navigation.runtime.launch import plan_launch


def _configs(args) -> list[str]:
    return list(args.config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cot-nav")
    sub = parser.add_subparsers(dest="command", required=True)

    env_p = sub.add_parser("env", help="print host, torch, CUDA, and kernel probe")
    env_p.add_argument("--probe-kernels", action="store_true")

    data = sub.add_parser("data")
    data_sub = data.add_subparsers(dest="data_cmd", required=True)
    for name in ("generate", "validate"):
        cmd = data_sub.add_parser(name)
        cmd.add_argument("--config", action="append", required=True)
        cmd.add_argument("--set", action="append", default=[])
        cmd.add_argument("--dry-run", action="store_true")
        if name == "generate":
            cmd.add_argument("--resume", dest="resume", action="store_true", default=True)
            cmd.add_argument("--no-resume", dest="resume", action="store_false")

    train_p = sub.add_parser("train")
    _add_common(train_p)
    train_p.add_argument("--resume", default=None)
    train_p.add_argument("--init-from", default=None)

    eval_p = sub.add_parser("evaluate")
    _add_common(eval_p)
    eval_p.add_argument("--checkpoint", required=True)
    eval_p.add_argument("--split", default="test")

    bench_p = sub.add_parser("benchmark")
    _add_common(bench_p)
    bench_p.add_argument("--batch", type=int, default=8)
    bench_p.add_argument("--length", type=int, default=256)
    bench_p.add_argument("--warmup", type=int, default=3)
    bench_p.add_argument("--steps", type=int, default=10)

    smoke_p = sub.add_parser("smoke", help="ephemeral full pipeline; never persists runs/checkpoints")
    smoke_p.add_argument("--config", action="append", default=[])
    smoke_p.add_argument("--set", action="append", default=[])
    smoke_p.add_argument("--steps", type=int, default=2)

    launch_p = sub.add_parser("launch")
    # dest cannot be "command": that name is the subparser and would be overwritten.
    launch_p.add_argument("--command", dest="jobs", action="append", required=True, help="shell-free argv string, repeat per job")
    launch_p.add_argument("--gpus", default="", help="comma separated GPU indices; default = idle GPUs")
    launch_p.add_argument("--output-root", default="outputs/launch")
    launch_p.add_argument("--execute", action="store_true", help="actually start processes; default is dry-run")

    args = parser.parse_args(argv)
    if args.command == "env":
        print(json.dumps(probe_environment(probe_kernels=args.probe_kernels), indent=2, default=str))
        return 0
    if args.command == "launch":
        gpus = [int(part) for part in args.gpus.split(",") if part.strip()] or None
        report = plan_launch(args.jobs, gpus, args.output_root, dry_run=not args.execute)
        print(json.dumps(report, indent=2))
        return int(report.get("exit_code", 0))
    configs = _configs(args) or ["configs/experiments/smoke_context.yaml"]
    cfg = load_config(configs, args.set)
    if args.command == "smoke":
        from cot_navigation.runtime.smoke import run_smoke

        print(json.dumps(run_smoke(cfg, args.steps), indent=2))
        return 0
    if args.dry_run:
        print(dump_config(cfg))
        if args.command == "data" and args.data_cmd == "generate":
            _print_quotas(cfg)
        return 0
    if args.command == "data" and args.data_cmd == "generate":
        if cfg.data.knowledge == "context_graph":
            manifest = generate_context(cfg.data, resume=args.resume)
        else:
            manifest = generate_parametric(cfg.data, resume=args.resume)
        print(json.dumps({"root": cfg.data.root, "kind": manifest["kind"]}, indent=2))
        return 0
    if args.command == "data" and args.data_cmd == "validate":
        if cfg.data.knowledge == "context_graph":
            report = validate_context(cfg.data)
        else:
            report = validate_parametric(cfg.data)
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "benchmark":
        report = run_benchmark(cfg, batch_size=args.batch, length=args.length, warmup=args.warmup, steps=args.steps)
        print(json.dumps({k: report[k] for k in report if k != "env"}, indent=2, default=str))
        return 0
    from cot_navigation.training.trainer import Trainer

    if args.command == "train":
        trainer = Trainer(cfg, resume=args.resume, init_from=args.init_from)
        try:
            metrics = trainer.fit()
        finally:
            trainer.tracker.finish()
        print(json.dumps({"step": trainer.completed, "last": metrics}, indent=2, default=str))
        return 0
    if args.command == "evaluate":
        trainer = Trainer(cfg, evaluation_checkpoint=args.checkpoint)
        try:
            report = trainer.evaluate_split(args.split)
        finally:
            trainer.tracker.finish()
        summary = report.get("overall", {})
        print(json.dumps({"split": args.split, "overall": summary, "trace": report.get("trace", {})}, indent=2))
        return 0
    raise SystemExit(f"unhandled command {args.command}")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")


def _print_quotas(cfg) -> None:
    n_shapes = len(cfg.data.shapes)
    for split, count in ("train", cfg.data.splits.train), ("val", cfg.data.splits.val), ("test", cfg.data.splits.test):
        per_shape = count // n_shapes
        print(f"# {split} total={count} per_shape={per_shape}")
        for name, shape in cfg.data.shapes.items():
            quotas = per_distance_quotas(per_shape, shape.layers - 1)
            print(f"#   {name} " + ", ".join(f"d{d}:{pos}/{neg}" for d, pos, neg in quotas))


if __name__ == "__main__":
    sys.exit(main())

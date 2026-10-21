#!/usr/bin/env python3
"""Scale ladder: square graphs from 4x4 to 10x10, one seed, GPUs 1-7.

Training mixes smaller graphs (including N24/N32) into 70% main-scale data.
Context tests include deep/wide/larger graphs; main fixed tests use trained
worlds. Two fixed probes expose additional worlds through WALK only.
A curriculum changes the main scale every 50k steps.

Data is prepared before the scheduler starts. --dry-run only prints the plan.
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

LADDER = (4, 6, 8, 10)
LENGTH_LAYERS = 12
MAIN_FRACTION = 0.7
STEP_UNIT = 50000
KNOWLEDGE_STEPS = 1000
CONTEXT_TRAIN = 50000
CONTEXT_VAL = 512
CONTEXT_TEST = 512
EVAL_TRAIN = 2
EVAL_VAL = 2
EVAL_TEST = 512
COT_MARGIN = 8
MIX_SHAPES = ((4, 4), (4, 6), (4, 8), (6, 6), (8, 8), (10, 10))
PROTOCOLS = (
    ("direct", ()),
    ("path", ("protocol.negative_random_path=true",)),
    ("dfs_state", ("protocol.random_search_order=true",)),
    ("bfs_state", ("protocol.random_search_order=true",)),
    ("dfs_nodes", ("protocol.random_search_order=true",)),
    ("bfs_nodes", ("protocol.random_search_order=true",)),
)
DEFAULT_GPUS = [1, 2, 3, 4, 5, 6, 7]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--setting", choices=("all", "context", "fixed"), default="all")
    parser.add_argument("--gpus", nargs="+", type=int, default=DEFAULT_GPUS)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--run-name")
    parser.add_argument("--output-root", default="outputs/scale_v1/launches")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument('--context-train', type=int, default=CONTEXT_TRAIN)
    parser.add_argument('--variants', type=int, default=4)
    parser.add_argument('--transfer-probes', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tracking", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    if args.resume and not args.run_name:
        parser.error("--resume requires the original --run-name")
    if args.prepare_only and (args.background or args.resume or args.dry_run):
        parser.error("--prepare-only cannot be combined with dry-run, background, or resume")
    if args.gpus is not None and (not args.gpus or min(args.gpus) < 0 or len(set(args.gpus)) != len(args.gpus)):
        parser.error("--gpus must contain distinct nonnegative indices")
    if not set(args.gpus) <= set(DEFAULT_GPUS):
        parser.error("GPU 0 is left free; choose GPUs from 1-7")
    if args.context_train < 2 or args.context_train % 2 or args.variants < 1:
        parser.error('context-train must be positive/even and variants must be positive')
    owned = {
        "output_dir", "protocol.name", "protocol.negative_random_path", "protocol.random_search_order",
        "protocol.cot_margin", "train.stage", "train.max_steps", "train.model_seed", "train.fixed_query_pool",
        "train.balance_query_labels", "train.one_hop_fraction", "train.shuffle_context_edges",
        "train.persist_outputs", "train.eval_interval", "train.save_interval", "train.eval_samples",
        "train.free_gen_samples", "train.scale_roots", "train.scale_weights", "train.scale_stage_ends",
        "train.scale_stage_weights", "runtime.smoke", "data.root", "data.n_nodes", "data.n_edges",
        "data.shapes", "data.graph_shape", "data.graph_seed", "data.graph_seeds", "data.data_seed",
        "data.splits.train", "data.splits.val", "data.splits.test", "data.include_one_hop_queries",
        "data.context_holdout_first", "data.audit_fraction", "data.max_attempts",
        "evaluation.trace_diagnostics", "evaluation.test_samples", "evaluation.selection_min_step",
        "evaluation.extra_sets", "tracking.mode", "tracking.group", "tracking.name", "tracking.variant",
        "tracking.tags",
    }
    owned |= {'train.world_roots', 'train.scale_walk_weights', 'protocol.trajectory_variants', 'runtime.cuda_device'}
    for item in args.set:
        if "=" not in item or item.split("=", 1)[0] in owned or item.split("=", 1)[0].startswith("data.shapes."):
            parser.error(f"runner owns this option or invalid override: {item}")
    args.run_name = args.run_name or datetime.now(ZoneInfo("Asia/Shanghai")).strftime("scale_%Y%m%d_%H%M%S")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_name) or args.run_name in {".", ".."}:
        parser.error("--run-name must be a plain name")
    return args


def layers_width(kind: str, size: int) -> tuple[int, int]:
    if kind == "square":
        return size, size
    if kind == "deep":
        return size * 2, size // 2
    if kind == "wide":
        return size // 2, size * 2
    raise ValueError(kind)


def counts(kind: str, size: int) -> tuple[int, int, int]:
    if kind == "square" and size in LADDER:
        return CONTEXT_TRAIN, CONTEXT_VAL, CONTEXT_TEST
    return EVAL_TRAIN, EVAL_VAL, EVAL_TEST


def data_root(setting: str, layers: int, width: int, train_n: int, seed=0) -> str:
    nodes = layers * width
    edges = nodes * 3 // 2
    if setting == "context":
        return f"data/scale_v1/context_l{layers}w{width}_n{nodes}_m{edges}_train{train_n}_s{seed}"
    return f"data/scale_v1/fixed_l{layers}w{width}_n{nodes}_m{edges}_s{seed}"


def square_root(setting: str, size: int) -> str:
    train_n, _val, _test = counts("square", size)
    return data_root(setting, size, size, train_n)


def _json_weights(values) -> str:
    return json.dumps([round(float(value), 10) for value in values])


def mix_weights(n_members: int, main_index: int) -> list[float]:
    weights = [0.0] * n_members
    if main_index == 0:
        weights[0] = 1.0
        return weights
    share = (1.0 - MAIN_FRACTION) / main_index
    for index in range(main_index):
        weights[index] = share
    weights[main_index] = MAIN_FRACTION
    return weights


def curriculum_weights() -> list[float]:
    flat = []
    for main in range(len(LADDER)):
        flat.extend(mix_weights(len(MIX_SHAPES), [x[0] * x[1] for x in MIX_SHAPES].index(LADDER[main] ** 2)))
    return flat


def static_steps(layers: int) -> int:
    return STEP_UNIT * layers // 4


def geometries():
    """Every dataset the tests read, including shapes that are never trained."""
    specs = []
    for setting in ("context", "fixed"):
        for size in (*LADDER, LENGTH_LAYERS):
            kinds = ("square", "deep", "wide") if size in LADDER else ("square",)
            for kind in kinds:
                layers, width = layers_width(kind, size)
                train_n, val_n, test_n = counts(kind, size)
                specs.append((setting, layers, width, train_n, val_n, test_n))
        specs += [(setting, l, w, CONTEXT_TRAIN, CONTEXT_VAL, CONTEXT_TEST) for l, w in ((4, 6), (4, 8))]
    return specs


def data_overrides(setting: str, layers: int, width: int, train_n: int, val_n: int, test_n: int, args=None) -> list[str]:
    nodes = layers * width
    edges = nodes * 3 // 2
    seed = args.data_seed if args else 0
    if args and (layers, width) in MIX_SHAPES:
        train_n = args.context_train
    root = data_root(setting, layers, width, train_n, seed)
    overrides = [
        f"data.root={root}",
        f"data.n_nodes={nodes}",
        f"data.n_edges={edges}",
        f"data.shapes={{main: {{layers: {layers}, width: {width}}}}}",
        "data.graph_shape=main",
        "data.max_attempts=2000",
        "protocol.cot_margin=8",
        "protocol.name=direct",
        f'data.data_seed={seed}',
    ]
    if setting == "context":
        overrides += [
            f"data.splits.train={train_n}",
            f"data.splits.val={val_n}",
            f"data.splits.test={test_n}",
            "data.context_holdout_first=true",
            "data.audit_fraction=0.01",
        ]
    else:
        overrides += [
            "data.graph_seed=0",
            "data.graph_seeds=[0]",
            "data.include_one_hop_queries=true",
            "data.audit_fraction=1.0",
        ]
    return overrides


def roots_for(args, setting, shapes):
    return [data_root(setting, l, w, args.context_train, args.data_seed) for l, w in shapes]


def world_roots(args):
    return [data_root('fixed', l, w, n, args.data_seed) + '/main-seed0'
            for setting, l, w, n, _v, _t in geometries() if setting == 'fixed']


def extra_sets(setting, main, args=None, transfer=False):
    args = args or parse_args(['--dry-run'])
    shapes = [(f'n{l*w}', l, w) for l, w in MIX_SHAPES if l*w < main*main]
    if setting == 'context' or transfer:
        shapes += [('deep', *layers_width('deep', main)), ('wide', *layers_width('wide', main)),
                   ('larger', main + 2, main + 2)]
    encoded = []
    for name, l, w in shapes:
        n = args.context_train if (l, w) in MIX_SHAPES else EVAL_TRAIN
        root = data_root(setting, l, w, n, args.data_seed)
        encoded.append(f'{name}={root}' + ('/main-seed0' if setting == 'fixed' else ''))
    return encoded


def build_jobs(args, root=ROOT):
    output = Path(args.output_root)
    output = (output if output.is_absolute() else root / output) / args.run_name
    shared = args.set + [f'model.compile={str(args.compile).lower()}',
        f'tracking.mode={args.tracking}', f'train.model_seed={args.model_seed}',
        'runtime.device=cuda', 'runtime.cuda_device=0', 'runtime.smoke=false',
        'train.persist_outputs=true', 'protocol.cot_margin=8',
        f'protocol.trajectory_variants={args.variants}', 'evaluation.trace_diagnostics=false',
        'evaluation.test_samples=256', 'train.eval_interval=5000', 'train.save_interval=5000',
        'train.eval_samples=128', 'train.free_gen_samples=16',
        f'tracking.group={json.dumps(args.run_name)}', 'tracking.tags=["scale_v1"]']
    jobs = []

    def add(name, preset, steps, extra, parent=None):
        out = output / name
        overrides = shared + extra + [f'train.max_steps={steps}',
            f'output_dir={json.dumps(str(out))}', f'tracking.name={json.dumps(args.run_name + "/" + name)}']
        path = root / 'configs/experiments' / f'{preset}.yaml'
        cfg = load_config([path], overrides)
        if cfg.train.warmup_steps >= cfg.train.max_steps:
            raise ValueError(f'{name}: warmup must be shorter than the stage budget')
        jobs.append(engine.Job(name, path, overrides, out, steps, parent))

    all_worlds = world_roots(args)
    for setting in ('context', 'fixed'):
        if args.setting not in ('all', setting):
            continue
        for size, curriculum in [(s, False) for s in LADDER] + [(10, True)]:
            shapes = list(MIX_SHAPES) if curriculum else [p for p in MIX_SHAPES if p[0]*p[1] <= size*size]
            roots = roots_for(args, setting, shapes)
            weights = mix_weights(len(roots), len(roots)-1)
            mixture = [f'train.scale_roots={json.dumps(roots)}',
                       f'evaluation.extra_sets={json.dumps(extra_sets(setting, size, args))}']
            if curriculum:
                ends = [50000, 100000, 150000, 200000]
                mixture += [f'train.scale_stage_ends={json.dumps(ends)}',
                    f'train.scale_stage_weights={_json_weights(curriculum_weights())}',
                    'evaluation.selection_min_step=150000']
                steps, label = 200000, 'curriculum'
            else:
                mixture += [f'train.scale_weights={_json_weights(weights)}']
                steps, label = static_steps(size), f'l{size}'
            geometry = data_overrides(setting, size, size, *counts('square', size), args=args)
            if setting == 'context':
                for protocol, flags in PROTOCOLS:
                    add(f'context/{label}/{protocol}', 'retrain_context', steps,
                        geometry + mixture + ['train.shuffle_context_edges=true', f'protocol.name={protocol}',
                        f'tracking.variant={label}-{protocol}', *flags])
            else:
                world_ids = [f'train.world_roots={json.dumps(all_worlds)}']
                parent = f'fixed/{label}/stage_a'
                initial_weights = curriculum_weights()[:len(roots)] if curriculum else weights
                add(parent, 'retrain_stage_a', KNOWLEDGE_STEPS, geometry + world_ids + [
                    f'train.scale_roots={json.dumps(roots)}', f'train.scale_weights={_json_weights(initial_weights)}',
                    'tracking.variant=knowledge'])
                for protocol, flags in PROTOCOLS:
                    add(f'fixed/{label}/stage_b/{protocol}', 'retrain_stage_b', steps,
                        geometry + world_ids + mixture + ['train.fixed_query_pool=full',
                        'train.balance_query_labels=true', 'train.one_hop_fraction=0.25',
                        f'protocol.name={protocol}', f'tracking.variant={label}-{protocol}', *flags], parent)

    # Two explicitly knowledge-exposed transfer probes, separate from the main
    # fixed-world setting. All test worlds contribute WALK, never test queries.
    if args.setting in ('all', 'fixed') and args.transfer_probes:
        roots = [str(Path(p).parent) for p in all_worlds]
        geometry = data_overrides('fixed', 10, 10, *counts('square', 10), args=args)
        worlds = [f'train.world_roots={json.dumps(all_worlds)}', f'train.scale_roots={json.dumps(roots)}']
        knowledge = [1 / len(roots)] * len(roots)
        parent = 'fixed/knowledge_transfer/stage_a'
        add(parent, 'retrain_stage_a', KNOWLEDGE_STEPS, geometry + worlds + [
            f'train.scale_weights={_json_weights(knowledge)}', 'tracking.variant=all-worlds-knowledge'])
        train_roots = roots_for(args, 'fixed', MIX_SHAPES)
        train_weights = dict(zip(train_roots, mix_weights(len(train_roots), len(train_roots)-1)))
        query_weights = [train_weights.get(p, 0) for p in roots]
        for protocol in ('direct', 'bfs_state'):
            flags = dict(PROTOCOLS)[protocol]
            add(f'fixed/knowledge_transfer/stage_b/{protocol}', 'retrain_stage_b', static_steps(10),
                geometry + worlds + [f'train.scale_weights={_json_weights(query_weights)}',
                f'train.scale_walk_weights={_json_weights(knowledge)}',
                f'evaluation.extra_sets={json.dumps(extra_sets("fixed", 10, args, transfer=True))}',
                'train.fixed_query_pool=full', 'train.balance_query_labels=true', 'train.one_hop_fraction=0.25',
                f'protocol.name={protocol}', f'tracking.variant=knowledge-transfer-{protocol}', *flags], parent)
    return output, jobs


def prepare_all(args):
    from cot_navigation.data.prepared import prepare_dataset, _inventory, _preparation_lock
    from cot_navigation.data.scale_prepared import prepare_variants
    from cot_navigation.data.dataset import ContextStore, ParametricWorld
    from cot_navigation.data.shards import atomic_json
    from cot_navigation.protocols.tokenizer import Tokenizer

    for setting, layers, width, train_n, val_n, test_n in geometries():
        if args.setting not in ('all', setting):
            continue
        if setting == 'fixed' and not args.transfer_probes and (layers, width) not in MIX_SHAPES:
            continue
        overrides = args.set + data_overrides(setting, layers, width, train_n, val_n, test_n, args=args)
        overrides += [f'protocol.trajectory_variants={args.variants}']
        if setting == 'fixed':
            overrides += [f'train.world_roots={json.dumps(world_roots(args))}']
        preset = 'retrain_context.yaml' if setting == 'context' else 'retrain_stage_b.yaml'
        cfg = load_config([ROOT / 'configs/experiments' / preset], overrides)
        # Raw generation/validation once; do not generate unbounded legacy DFS
        # caches, nor repeat the whole audit independently for every protocol.
        report = prepare_dataset(cfg, ['direct'])
        source = ContextStore(cfg.data.root) if setting == 'context' else ParametricWorld(Path(cfg.data.root) / 'main-seed0')
        tok = Tokenizer.from_config(cfg)
        # Serialize cache writers across launchers sharing the same dataset.
        with _preparation_lock(Path(cfg.data.root)):
            for protocol, _flags in PROTOCOLS:
                prepare_variants(source, cfg, tok, protocol)
            report['inventory'] = _inventory(Path(cfg.data.root))
            atomic_json(Path(cfg.data.root) / 'prepared_data.json', {k: v for k, v in report.items() if k != 'reused'})
        print(json.dumps({'root': cfg.data.root, 'raw_reused': report['reused'], 'variants': args.variants}), flush=True)
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    os.chdir(ROOT)
    output, jobs = build_jobs(args)
    if args.dry_run:
        return engine.execute(args, output, jobs, argv, script_path=Path(__file__).resolve(),
                          data_preparer=lambda _jobs, _out, _env: prepare_all(args))
    if args.prepare_only:
        return prepare_all(args)
    return engine.execute(args, output, jobs, argv, script_path=Path(__file__).resolve(),
                          data_preparer=lambda _jobs, _out, _env: prepare_all(args))


if __name__ == "__main__":
    def interrupt(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; use the same --run-name and --resume with the same options.", file=sys.stderr)
        raise SystemExit(130)
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)

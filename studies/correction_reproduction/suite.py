"""Prepare by default. Launch only after review with an explicit --launch.

12 trials: 4 paired seeds x original clean / matched clean / correction.
Existing immutable graphs, trajectories and observed-error pools are reused.
"""
import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'TORCHINDUCTOR_COMPILE_THREADS']:
    os.environ.setdefault(key, '2')
import yaml
from cot_navigation.experiments import scheduler as engine
from cot_navigation.config import apply_training_profile, load_config
from trace_weight import build_policy

ARMS = ['matched_clean', 'correction', 'original_clean']
SOURCE = 'outputs/natural_correction_v1/launches/natural_correction_s0/context/dfs_state/p30_l16/resolved_config.yaml'


@dataclass
class Job(engine.Job):
    arm: str = ''

    def command(self, checkpoint=None, parent=None):
        assert parent is None
        argv = [sys.executable, str(HERE / 'worker.py'), '--config', str(self.config),
                '--arm', self.arm, '--execute-training']
        if checkpoint is not None: argv += ['--resume', str(checkpoint)]
        return argv


def build(args):
    out = ROOT / 'outputs/correction_reproduction_v1/launches' / args.run_name
    generated = ROOT / 'configs/generated/correction_reproduction' / args.run_name
    generated.mkdir(parents=True, exist_ok=True)
    template = apply_training_profile(yaml.safe_load((ROOT / SOURCE).read_text()))
    policy = build_policy()
    policy_text = json.dumps(policy, indent=2) + '\n'
    policy_path = HERE / 'trace_weight_policy.json'
    if args.launch and policy_path.read_text() != policy_text:
        raise ValueError('per-example length policy changed; prepare for review again')
    if not args.launch: policy_path.write_text(policy_text)
    jobs = []
    # Prioritize the four matched comparisons in wave 1; original controls
    # queue behind them. All arms are trained from scratch for the full 200K.
    for arm in ARMS:
        for seed in args.seeds:
            cfg = json.loads(json.dumps(template))
            name = f'dfs_state/{arm}/s{seed}'
            cfg['output_dir'] = str(out / name)
            cfg['train'].update(model_seed=seed, exposure_seed=seed, max_steps=200000,
                trace_loss_lambda=1., free_gen_samples=64, keep_last=3)
            cfg['model'].update(max_seq_len=4096, compile=True)
            cfg['protocol'].update(enable_mistake=arm != 'original_clean',
                mistake_rate=0. if arm == 'original_clean' else .3, mistake_seed=seed,
                mistake_mask_errors=False, mistake_source='synthetic' if arm == 'original_clean' else 'natural', natural_complex_share=.5,
                mistake_cache_root='data/natural_correction/natural_correction_s0')
            cfg['evaluation'].update(seed=1729, test_samples=512, final_checkpoints=['last'],
                free_batch_size=16, mistake_samples=0, trace_diagnostics=False,
                save_trace_tokens=False, extra_sets=[
                    'n32=data/scale_v1/context_l4w8_n32_m48_train50000_s0',
                    'n64=data/scale_v1/context_l8w8_n64_m96_train50000_s0',
                    'deep=data/scale_v1/context_l20w5_n100_m150_train2_s0',
                    'hard=data/hard_dag_v1/context_r10b10_n100_m200_train50000_s0'])
            cfg['tracking'].update(mode=args.tracking, group=args.run_name,
                name=f'{args.run_name}/{name}', variant=arm, upload_artifacts=False,
                tags=['correction_reproduction', 'dfs_state', arm, f'seed{seed}'],
                notes='Frozen original natural-error pool; paired model/exposure/error-sampling seeds. '
                      'matched_clean retains MISTAKE vocabulary and identical error-pool query sampling, '
                      'but trains canonical clean rows. Full loss with per-example '
                      'lambda=clip(actual CoT tokens/128,1,16); last200K is primary endpoint.')
            path = generated / f'{arm}_s{seed}.yaml'
            text = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
            if path.exists() and path.read_text() != text and args.launch:
                raise ValueError(f'config changed: {path}; prepare for review again')
            path.write_text(text); load_config([path])
            jobs.append(Job(name, path, [], out / name, 200000, arm=arm))
    return out, jobs


def source_hashes():
    paths = [HERE / n for n in ['suite.py', 'worker.py', 'audit.py',
                              'trace_weight.py', 'trace_weight_policy.json']]
    paths += [ROOT / n for n in [SOURCE, 'configs/train/throughput.yaml',
        'src/cot_navigation/training/trainer.py', 'src/cot_navigation/data/natural_mistakes.py',
        'src/cot_navigation/models/attention.py', 'src/cot_navigation/evaluation/rollout.py',
        'studies/accuracy_source/run.py']]
    paths += sorted((ROOT / 'src/cot_navigation').rglob('*.py'))
    paths += [ROOT / 'scripts/train_first_round.py']
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def check_sources():
    saved = json.loads((HERE / 'review_plan.json').read_text())['source_hashes']
    if saved != source_hashes():
        raise RuntimeError('reviewed code/config sources changed: prepare and audit again before launch')


def prepare_data(jobs, output, env):
    # Never remine or regenerate data as a side effect of starting training.
    check_sources()
    import subprocess
    subprocess.run([sys.executable, str(HERE / 'audit.py'), '--verify-existing'],
                   cwd=ROOT, env=env, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-name', default='dfs_correction_repro_20261028')
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3])
    p.add_argument('--gpus', type=int, nargs='+', default=list(range(8)))
    p.add_argument('--tracking', choices=['online', 'offline', 'disabled'], default='online')
    p.add_argument('--launch', action='store_true')
    p.add_argument('--background', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--wait', action='store_true')
    args = p.parse_args()
    import re
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run_name): p.error('invalid run-name')
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0: p.error('invalid seeds')
    if len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 0: p.error('invalid GPUs')
    if args.background and not args.launch: p.error('--background requires --launch')
    os.chdir(ROOT)
    output, jobs = build(args)
    args.dry_run = not args.launch
    if not args.launch:
        plan = dict(run_name=args.run_name, launched=False, seeds=args.seeds, jobs=engine.plan_record(jobs),
                    source_hashes=source_hashes(), primary='last checkpoint, 200000 updates',
                    fixed_corpus='data/natural_correction/natural_correction_s0',
                    pending_review=True)
        (HERE / 'review_plan.json').write_text(json.dumps(plan, indent=2))
    else:
        check_sources()
        reviewed = json.loads((HERE / 'review_plan.json').read_text())
        if reviewed['run_name'] != args.run_name or reviewed['jobs'] != engine.plan_record(jobs):
            raise RuntimeError('job plan differs from reviewed plan; prepare for review again')
    return engine.execute(args, output, jobs, sys.argv[1:], script_path=Path(__file__),
                          data_preparer=prepare_data)


if __name__ == '__main__':
    raise SystemExit(main())

"""Read-only online tracking and launch readiness; never creates a W&B run."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from data import HERE, digest

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--repo-root', type=Path, required=True)
args = p.parse_args()
cfg = json.loads((HERE / 'experiment.json').read_text())
manifest = json.loads((HERE / 'manifest.json').read_text())
verified = json.loads((HERE / 'verification.json').read_text())
assert digest(cfg) == manifest['config_sha256']
assert verified['status'] == 'PASS' and verified['supervision_checked']
assert verified['manifest_sha256'] == digest(manifest)
from train import source_hash
assert verified['source_sha256'] == source_hash(args.repo_root), 'Runtime verification is stale'
import wandb
entity = wandb.Api(timeout=20).default_entity
assert entity, 'No authenticated W&B account found'
plan = subprocess.check_output([sys.executable, str(HERE / 'launch.py'), '--repo-root',
                                str(args.repo_root), '--mode', 'pilot'], text=True)
print(json.dumps({'status': 'READY', 'wandb_entity': entity,
                  'wandb_project': cfg['wandb_project'], 'wandb_group': cfg['wandb_group'],
                  'pilot_jobs': len(json.loads(plan)['jobs']),
                  'formal_training_started': False, 'wandb_run_created': False}))

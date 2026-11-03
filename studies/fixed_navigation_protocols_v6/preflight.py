"""Read-only readiness and authenticated W&B check; never creates a run."""
import argparse
import json
from pathlib import Path
from prepare import HERE, digest
from common import source_hash
from launch import jobs_for

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--repo-root', type=Path, required=True)
args = p.parse_args()
cfg = json.loads((HERE / 'experiment.json').read_text())
manifest = json.loads((HERE / 'manifest.json').read_text())
report = json.loads((HERE / 'verification.json').read_text())
assert report['status'] == 'PASS' and report['manifest_sha256'] == digest(manifest)
assert report['source_sha256'] == source_hash(args.repo_root)
import wandb
entity = wandb.Api(timeout=20).default_entity
assert entity
print(json.dumps({'status': 'READY', 'wandb_entity': entity, 'project': cfg['wandb_project'],
                  'group': cfg['wandb_group'], 'pilot_tasks': len(jobs_for('pilot', cfg, cfg['protocols'])),
                  'formal_tasks': len(jobs_for('formal', cfg, cfg['protocols'])), 'formal_training_started': False}))

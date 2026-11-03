"""Read-only proof of new-arm configuration, W&B and first real updates."""
from pathlib import Path
import json

ROOT=Path.cwd()
OUT=ROOT/'outputs/task_only_n32_v1/launches/joint_walk_n32_s0'
plan=json.loads((OUT/'launch_plan.json').read_text());rows=[]
assert len(plan)==6 and all(spec['depends_on'] is None for spec in plan.values())
for name,spec in plan.items():
    cfg=spec['config'];folder=Path(cfg['output_dir'])
    assert cfg['train']['skip_knowledge_stage'] and not cfg['train']['task_only']
    assert cfg['train']['walk_loss_mode']=='fixed' and cfg['train']['walk_loss_weight']==1
    assert not (folder/'init_from.json').exists()
    tracking=folder/'wandb_status.json'
    status=json.loads(tracking.read_text()) if tracking.exists() else {}
    metrics=folder/'metrics.jsonl'
    records=[json.loads(s) for s in metrics.read_text().splitlines() if s.strip()] if metrics.exists() else []
    train=[r for r in records if 'answer_loss' in r]
    row={'job':name,'effective_tracking':status.get('effective'),'wandb_url':status.get('url'),
         'parent_run_id':status.get('parent_run_id'),'initialization':'random','stage_a_dependency':False}
    if train:
        last=train[-1]
        assert last['walks_seen']==last['step']*16 and last['queries_seen']==last['step']*64
        assert 'walk_loss' in last
        row.update({k:last[k] for k in ['step','queries_seen','walks_seen','answer_loss','walk_loss']})
    rows.append(row)
receipt={'runs':rows,'status':json.loads((OUT/'status.json').read_text())}
(ROOT/'studies/joint_walk_n32/running_verification.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))

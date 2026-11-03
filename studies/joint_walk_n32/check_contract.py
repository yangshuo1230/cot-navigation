"""CPU-only initialization/data/resume checks; never call fit or save models."""
from pathlib import Path
from types import SimpleNamespace
import copy
import hashlib
import json
import sys
import torch

ROOT=Path.cwd();sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import task_only_n32 as runner
from cot_navigation.experiments import scheduler as engine
from cot_navigation.config import load_config,ConfigError
from cot_navigation.training.trainer import Trainer,TrainError,_arch_snapshot,_check_resume
from cot_navigation.models.transformer import build_model

args=SimpleNamespace(arm='joint_from_scratch',run_name='joint_walk_n32_s0',protocols=list(runner.PROTOCOLS),
                     steps=50000,tracking='online',with_walk_control=True)
output,jobs=runner.build_jobs(args)
assert len(jobs)==6 and all(job.depends_on is None for job in jobs)
runner.verify_cached_data(jobs,ROOT/'studies/joint_walk_n32',{})

cfg=load_config([jobs[0].config],jobs[0].overrides)
assert cfg.train.skip_knowledge_stage and not cfg.train.task_only
assert cfg.train.walk_loss_mode=='fixed' and cfg.train.walk_loss_weight==1
assert cfg.train.effective_query_batch_size==64 and cfg.train.walk_batch_size==16
assert cfg.train.max_steps==50000
try:Trainer(cfg,init_from='not-a-checkpoint')
except TrainError as exc:assert 'pretrained' in str(exc)
else:raise AssertionError('joint random init may not load pretrained weights')

cfg.train.persist_outputs=False;cfg.model.compile=False;cfg.train.precision='fp32';cfg.runtime.device='cpu';cfg.runtime.cuda_device=None
trainer=Trainer(cfg)
assert trainer.query_sampler is not None and trainer.walk_sampler is not None
assert trainer.completed==trainer.queries_seen==trainer.walks_seen==0 and trainer.parent_run_id is None
torch.manual_seed(cfg.train.model_seed)
expected=build_model(cfg.model,len(trainer.tok),trainer.tok.pad)
assert all(torch.equal(v,expected.state_dict()[k]) for k,v in trainer.raw.state_dict().items())
initial_hash=hashlib.sha256()
for k,v in trainer.raw.state_dict().items():initial_hash.update(k.encode());initial_hash.update(v.detach().numpy().tobytes())

# Joint training must reuse the QUERY, WALK and augmentation streams of the
# existing WALK control. Initialization mode must not change data exposure.
control=load_config([ROOT/'configs/experiments/task_only_n32_v1/control_direct.yaml'])
reference=Trainer.__new__(Trainer);reference.cfg=control;reference.completed=0
reference.store=reference.world=None;reference.scale_members=[]
reference._load_data();reference._init_samplers()
for _ in range(10):
    assert trainer.query_sampler.next()==reference.query_sampler.next()
    assert trainer.walk_sampler.next()==reference.walk_sampler.next()
for key in trainer.augmentation_rngs:
    assert trainer.augmentation_rngs[key].bit_generator.state==reference.augmentation_rngs[key].bit_generator.state

# Saved runs omitted this new knob. Both old launcher plans and checkpoints
# must remain resumable with skip_knowledge_stage=false, never as this arm.
oldargs=SimpleNamespace(arm='paired',run_name='task_only_n32_s0',protocols=list(runner.PROTOCOLS),
                        steps=50000,tracking='online',with_walk_control=True)
oldout,oldjobs=runner.build_jobs(oldargs)
oldplan=json.loads((oldout/'launch_plan.json').read_text())
engine.check_resume_plan(oldplan,engine.plan_record(oldjobs))
blob=torch.load(oldout/'task_only/direct/checkpoints/latest.pt',map_location='cpu',weights_only=False)
oldcfg=load_config([oldout/'task_only/direct/resolved_config.yaml'])
saved=copy.deepcopy(blob['arch'])
_check_resume(saved,_arch_snapshot(oldcfg,trainer.tok,blob['arch']['manifest_hash']))
bad=copy.deepcopy(oldcfg);bad.train.task_only=False;bad.train.skip_knowledge_stage=True
try:_check_resume(copy.deepcopy(blob['arch']),_arch_snapshot(bad,trainer.tok,blob['arch']['manifest_hash']))
except TrainError:pass
else:raise AssertionError('may not reinterpret an old run as the new arm')
trainer.tracker.finish()
receipt=dict(jobs=6,stage_a_jobs=0,initialization='seeded_random',initial_state_sha256=initial_hash.hexdigest(),
             model_seed=cfg.train.model_seed,query_batch=64,walk_batch=16,steps=50000,
             data_streams_match_walk_control=True,legacy_plan_and_checkpoint_resume=True,optimizer_updates=0)
(ROOT/'studies/joint_walk_n32/verification.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))

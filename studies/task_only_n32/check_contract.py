"""CPU constructor/config/resume contracts; no optimizer updates or output runs."""
from pathlib import Path
import sys
import copy
import torch
ROOT=Path.cwd()
sys.path.insert(0,str(ROOT/'scripts'))
import task_only_n32
from cot_navigation.config import load_config,ConfigError
from cot_navigation.training.trainer import Trainer,TrainError,_arch_snapshot,_check_resume
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.models.transformer import build_model

path=ROOT/'configs/experiments/task_only_n32_v1/direct.yaml'
cfg=load_config([path])
assert cfg.train.task_only and cfg.train.walk_loss_mode=='none'
bad=copy.deepcopy(cfg);bad.train.walk_loss_mode='fixed'
try:bad.validate()
except ConfigError:pass
else:raise AssertionError('WALK must be forbidden with task_only')
bad=copy.deepcopy(cfg);bad.train.task_only=False
try:Trainer(bad)
except TrainError as e:assert 'stage B requires' in str(e)
else:raise AssertionError('legacy Stage-B guard must remain')
try:Trainer(cfg,init_from='not-a-real-checkpoint')
except TrainError as e:assert 'pretrained' in str(e)
else:raise AssertionError('task_only must reject pretrained initialization')

cfg.train.persist_outputs=False;cfg.model.compile=False;cfg.train.precision='fp32';cfg.runtime.device='cpu';cfg.runtime.cuda_device=None
t=Trainer(cfg)
assert t.walk_sampler is None and t.query_sampler is not None
assert t.walks_seen==t.queries_seen==t.completed==0
assert len(t.train_ids)==288 and len(t.scale_members)==1
torch.manual_seed(cfg.train.model_seed)
expected=build_model(cfg.model,len(t.tok),t.tok.pad)
assert all(torch.equal(v,expected.state_dict()[k]) for k,v in t.raw.state_dict().items())
snapshot=_arch_snapshot(cfg,t.tok,t.manifest_hash)
assert snapshot['task_only']
legacy=copy.deepcopy(snapshot);legacy.pop('task_only')
current=copy.deepcopy(snapshot);current['task_only']=False
_check_resume(legacy,current)
try:_check_resume(copy.deepcopy(legacy),snapshot)
except TrainError:pass
else:raise AssertionError('task-only may not resume a pretrained run')
t.tracker.finish()
print('PASS: random init, no WALK sampler, 288 disjoint train queries, legacy guard/resume compatibility; zero training updates')

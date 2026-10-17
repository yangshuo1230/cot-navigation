"""Independent graph oracles, sampling quotas, online RNG and exact resume."""
import copy
from collections import deque, Counter
import importlib.util
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import numpy as np
import pytest
import torch

from cot_navigation.config import load_config, ShapeSpec, SplitsSpec
from cot_navigation.data.generate import generate_context, generate_parametric
from cot_navigation.data.prepared import prepare_dataset, PreparedRows
from cot_navigation.data.dataset import ContextStore
from cot_navigation.protocols.bfs import bfs_search, first_expansion_order
from cot_navigation.protocols.dfs import dfs_search
from cot_navigation.protocols.render import render_task, context_prompt
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.protocols.validate_trace import parse_completed_steps, compare_completed
from cot_navigation.training.augmentation import random_path, transform
from cot_navigation.training.sampler import HopCurriculumSampler
from cot_navigation.training.trainer import Trainer

ROOT = Path(__file__).resolve().parents[1]


def oracle(adj, src, dst):
    queue, seen, order = deque([src]), {src}, []
    while queue:
        cur = queue.popleft()
        order.append(cur)
        if cur == dst:
            return True, order
        for nxt in adj[cur]:
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return False, order


def test_bfs_and_unique_sequences_match_independent_fifo_oracle():
    rng = np.random.default_rng(551)
    tok = Tokenizer(32)
    nodes = [tok.node_id(i) for i in range(7)]
    for _ in range(100):
        adj = [[v for v in range(u+1,7) if rng.random()<.35] for u in range(7)]
        for src in range(7):
            for dst in range(7):
                reachable, order = oracle(adj,src,dst)
                steps = bfs_search(adj,src,dst)
                assert (steps[-1].action=='FOUND') == reachable
                assert first_expansion_order(steps)==order
                for protocol in ('dfs_nodes','bfs_nodes'):
                    row = render_task(tok,protocol,int(reachable),[tok.id('|')],adj,src,dst,nodes)
                    body = row.tokens[row.prompt_len:row.answer_index]
                    assert len(body)==len(set(body)) and body[0]==nodes[src]
                    assert set(body)<=set(nodes)
                row = render_task(tok,'bfs_state',int(reachable),[tok.id('|')],adj,src,dst,nodes)
                parsed = parse_completed_steps(row.tokens[1:row.answer_index],tok,fifo=True)
                assert compare_completed(steps,parsed,nodes)['complete_match']


def test_random_negative_paths_are_legal_sink_terminated_and_resampled():
    adj = [[1,2],[3],[4],[],[],[]]
    rng = np.random.default_rng(22)
    paths = [random_path(adj,0,rng) for _ in range(64)]
    assert {tuple(p) for p in paths}=={(0,1,3),(0,2,4)}
    for path in paths:
        assert not adj[path[-1]]
        assert all(v in adj[u] for u,v in zip(path,path[1:]))
    assert random_path(adj,5,rng)==[5]
    tok = Tokenizer(16); nodes = [tok.node_id(i) for i in range(6)]
    row = render_task(tok,'path',0,[tok.id('|')],adj,0,5,nodes)
    changed = transform(row,'path',tok,adj,nodes,0,5,negative_rng=rng)
    assert row.answer_index==1 and changed.answer_index==4
    assert changed.tokens[changed.answer_index]==tok.no
    assert changed.trace_indices==[1,2,3,5]


def test_edge_augmentation_keeps_edge_set_and_search_state_consistent():
    tok = Tokenizer(16); nodes = [tok.node_id(i) for i in range(5)]
    adj = [[1,2],[3],[3],[],[]]
    edges = [(u,v) for u,vs in enumerate(adj) for v in vs]
    prompt = context_prompt(tok,nodes,edges,0,4)
    row = render_task(tok,'dfs_state',0,prompt,adj,0,4,nodes)
    rng = np.random.default_rng(7)
    variants = [transform(row,'dfs_state',tok,adj,nodes,0,4,edge_rng=rng) for _ in range(20)]
    assert len({tuple(r.tokens[:r.prompt_len]) for r in variants})>1
    original = sorted(map(tuple,np.asarray(prompt[:-4]).reshape(-1,3)))
    for r in variants:
        triples = np.asarray(r.tokens[:r.prompt_len-4]).reshape(-1,3)
        assert sorted(map(tuple,triples))==original
        ordered = [[] for _ in nodes]
        for _,u,v in triples:ordered[nodes.index(u)].append(nodes.index(v))
        reference = dfs_search(ordered,0,4)
        parsed = parse_completed_steps(r.tokens[r.prompt_len:r.answer_index],tok)
        assert compare_completed(reference,parsed,nodes)['complete_match']
    assert row.tokens[:row.prompt_len]==prompt  # no cached-row mutation


def test_curriculum_stages_replay_hops_and_labels_equally_and_resume():
    distances = np.repeat([1,2,3],12)
    labels = np.tile([0,1],18)
    sampler = HopCurriculumSampler(distances,labels,64,5,(3,3,3))
    for stage in (1,2,3):
        counts = Counter()
        for _ in range(3):
            ids = sampler.next()
            counts.update(zip(distances[ids],labels[ids]))
        assert set(h for h,l in counts)==set(range(1,stage+1))
        assert max(counts.values())-min(counts.values())<=1
    saved = sampler.state_dict()
    expected = sampler.next()
    resumed = HopCurriculumSampler(distances,labels,64,5,(3,3,3))
    resumed.load_state_dict(saved)
    assert resumed.next()==expected


def suite():
    spec = importlib.util.spec_from_file_location('explore_suite',ROOT/'scripts/explore.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def test_suite_plan_excludes_filler_and_shares_fixed_knowledge(tmp_path):
    m = suite()
    args = m.parse_args(['--run-name','test','--dry-run','--output-root',str(tmp_path/'absent')])
    out,jobs = m.build_jobs(args)
    assert len(jobs)==60 and not out.exists()
    parents = [j for j in jobs if j.name.endswith('/stage_a')]
    assert len(parents)==3
    for j in jobs:
        cfg = m.load_config([j.config],j.overrides)
        assert cfg.protocol.name!='filler'
        if cfg.train.stage=='B':assert j.depends_on in {p.name for p in parents}
        if cfg.train.query_hops=='curriculum':
            budgets=cfg.train.curriculum_stage_steps
            assert sum(budgets)==50000 and max(budgets)-min(budgets)<=1
        if cfg.train.stage=='context':assert cfg.train.shuffle_context_edges
    _,variants = m.build_jobs(m.parse_args(['--run-name','shapes','--shapes','wide','deep','--experiments','baseline']))
    geometries = {(m.load_config([j.config],j.overrides).data.n_nodes,
                   tuple((v.layers,v.width) for v in m.load_config([j.config],j.overrides).data.shapes.values())) for j in variants}
    assert geometries=={(32,((4,8),)),(32,((8,4),))}


@pytest.fixture
def context_cfg(tmp_path):
    cfg=load_config([ROOT/'configs/experiments/retrain_context.yaml'],[
        f'data.root={tmp_path/"data"}','data.workers=1','data.splits.train=72','data.splits.val=24','data.splits.test=24',
        'model.layers=1','model.d_model=16','model.d_ff=32','model.compile=false',
        'runtime.device=cpu','runtime.probe_kernels=false','train.precision=fp32',
        'train.max_steps=6','train.warmup_steps=0','train.effective_query_batch_size=6',
        'train.microbatch_size=2','train.eval_interval=0','train.save_interval=0',
        'train.persist_outputs=false','tracking.mode=disabled','evaluation.final_checkpoints=[]'])
    generate_context(cfg.data)
    return cfg


@pytest.mark.parametrize('protocol', ['bfs_state','dfs_nodes','bfs_nodes'])
def test_new_protocol_prepared_tokens_preserve_trace_masks(context_cfg,protocol):
    cfg=context_cfg
    prepare_dataset(cfg,[protocol])
    source=ContextStore(cfg.data.root);tok=Tokenizer(cfg.data.node_vocab)
    cached=PreparedRows(source,protocol,tok)
    for i in range(len(source)):
        a,b=source.render(i,protocol,tok),cached.row(i)
        assert a.tokens==b.tokens and a.trace_indices==b.trace_indices


def test_singlehop_training_and_evaluation_use_only_singlehop(context_cfg):
    cfg=context_cfg;cfg.train.query_hops='one_hop'
    tr=Trainer(cfg)
    assert set(tr.store.arrays['distance'][tr.train_ids])=={1}
    assert set(tr.store.arrays['distance'][tr._selected_ids('test',0)])=={1}
    assert set(tr.store.arrays['distance'][tr.train_ids[tr.query_sampler.next()]])=={1}


def test_online_edge_rng_curriculum_resume_matches_weights(context_cfg):
    cfg=context_cfg
    cfg.train.shuffle_context_edges=True
    cfg.train.query_hops='curriculum'
    cfg.train.curriculum_stage_steps=(2,2,2)
    full=Trainer(copy.deepcopy(cfg));full.fit()
    partial=Trainer(copy.deepcopy(cfg))
    for _ in range(3):
        partial._update();partial.completed+=1;partial._touch_rng()
    with TemporaryDirectory(prefix='cot-test-resume-') as scratch:
        ckpt=Path(scratch)/'resume.pt';torch.save(partial._payload(),ckpt)
        resumed=Trainer(copy.deepcopy(cfg),resume=str(ckpt));resumed.fit()
        assert all(torch.equal(v,resumed.raw.state_dict()[k]) for k,v in full.raw.state_dict().items())


@pytest.mark.parametrize('mode,protocol', [('balanced','path'),('none','path'),('balanced','dfs_state')])
def test_fixed_walk_control_and_dynamic_path_resume(tmp_path,mode,protocol):
    cfg=load_config([ROOT/'configs/experiments/retrain_stage_a.yaml'],[
        f'data.root={tmp_path/"data"}','data.graph_seeds=[0]','data.n_walks=32','data.min_edge_coverage=2',
        'model.layers=1','model.d_model=16','model.d_ff=32','model.compile=false',
        'runtime.device=cpu','runtime.probe_kernels=false','train.precision=fp32',
        'train.max_steps=4','train.warmup_steps=0','train.effective_query_batch_size=4',
        'train.walk_batch_size=4','train.microbatch_size=2','train.eval_interval=0',
        'train.save_interval=0','train.persist_outputs=false','tracking.mode=disabled'])
    generate_parametric(cfg.data)
    initial=Trainer(copy.deepcopy(cfg)).initial_state()
    cfg.train.stage='B';cfg.protocol.name=protocol
    cfg.protocol.negative_random_path=protocol=='path'
    cfg.protocol.random_search_order=protocol=='dfs_state'
    cfg.train.walk_loss_mode=mode
    if mode=='none':cfg.train.walk_batch_size=0
    full=Trainer(copy.deepcopy(cfg),init_state=initial)
    if mode=='none':assert full.walk_sampler is None and full.prepared_walks is None
    full.fit()
    partial=Trainer(copy.deepcopy(cfg),init_state=initial)
    for _ in range(2):
        metrics=partial._update();partial.completed+=1;partial._touch_rng()
    if mode=='balanced':
        assert 0<=metrics['walk_loss_share']<=1 and metrics['walk_weight']>0
        assert abs(metrics['total_loss']-metrics['task_loss']-metrics['weighted_walk_loss'])<1e-5
        assert partial.loss_balance_ema is not None
    with TemporaryDirectory(prefix='cot-test-resume-') as scratch:
        ckpt=Path(scratch)/'resume.pt';torch.save(partial._payload(),ckpt)
        resumed=Trainer(copy.deepcopy(cfg),resume=str(ckpt));resumed.fit()
        assert all(torch.equal(v,resumed.raw.state_dict()[k]) for k,v in full.raw.state_dict().items())

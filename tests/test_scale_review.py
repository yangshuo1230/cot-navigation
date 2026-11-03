"""Focused review contracts; no training updates, W&B or GPU execution."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import numpy as np
import torch

from cot_navigation.config import load_config
from cot_navigation.models.transformer import build_model
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.protocols.render import render_task, render_walk, context_prompt
from cot_navigation.training.trainer import Trainer, _arch_snapshot, _check_resume

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import scale


def test_legacy_checkpoint_and_new_vocab():
    tok = Tokenizer()
    assert len(tok) == 270 and tok.node_lo == 14
    extended = Tokenizer(enable_stop=True, world_tokens=15)
    assert tok.itos == extended.itos[:len(tok)]
    folder = ROOT / 'studies/minimal_query_smallgraph/runs/minimal_query_smallgraph_v1/direct_single'
    if folder.exists():
        blob = torch.load(folder / 'checkpoints/latest.pt', map_location='cpu', weights_only=False)
        assert blob['arch']['vocab_hash'] == tok.hash()
        cfg = load_config([folder / 'resolved_config.yaml'])
        model = build_model(cfg.model, len(tok), tok.pad)
        model.load_state_dict(blob['model'], strict=True)
        _check_resume(dict(blob['arch']), _arch_snapshot(cfg, tok, blob['arch']['manifest_hash']))


def test_plan_and_knowledge_boundaries():
    args = scale.parse_args(['--dry-run', '--run-name', 'review', '--data-seed', '3'])
    _, jobs = scale.build_jobs(args)
    assert len(jobs) == 68
    assert all(x in range(1, 8) for x in args.gpus)
    configs = {j.name: load_config([j.config], j.overrides) for j in jobs}
    for name, cfg in configs.items():
        assert cfg.data.n_edges * 2 == cfg.data.n_nodes * 3
        assert cfg.protocol.cot_margin == 8
        assert '_s3' in cfg.data.root
        assert cfg.protocol.trajectory_variants == 4
        if name.startswith('fixed/'):
            assert len(cfg.train.world_roots) == 15
            if cfg.train.stage == 'B':
                assert cfg.train.one_hop_fraction == .25 and cfg.data.include_one_hop_queries
    cfg = configs['context/l6/direct']
    assert len(cfg.train.scale_roots) == 4  # N16,24,32,36
    assert list(cfg.train.scale_weights) == [.1, .1, .1, .7]
    assert any('n24_' in root for root in cfg.train.scale_roots)
    assert any('n32_' in root for root in cfg.train.scale_roots)
    for name, cfg in configs.items():
        if name.startswith('fixed/') and 'knowledge_transfer' not in name:
            assert not any(x.startswith(('deep=', 'wide=', 'larger=')) for x in cfg.evaluation.extra_sets)
    probe = configs['fixed/knowledge_transfer/stage_b/bfs_state']
    assert len(probe.train.scale_roots) == len(probe.train.scale_walk_weights) == 15
    assert sum(x > 0 for x in probe.train.scale_weights) == 6
    assert all(x > 0 for x in probe.train.scale_walk_weights)
    for root, weight in zip(probe.train.scale_roots, probe.train.scale_weights):
        if any(root + '/main-seed0' == s.split('=', 1)[1] for s in probe.evaluation.extra_sets if s.startswith(('deep=', 'wide=', 'larger='))):
            assert weight == 0


def test_world_prefix_masks():
    tok = Tokenizer(8, enable_stop=True, world_tokens=2)
    shim = Trainer.__new__(Trainer)
    shim.tok = tok
    shim.cfg = SimpleNamespace(train=SimpleNamespace(world_roots=['/tmp/world0', '/tmp/world1']))
    nodes = [tok.node_id(i) for i in range(3)]
    row = render_task(tok, 'direct', 1, [tok.id('<QUERY>'), nodes[0], nodes[1], tok.id('|')], [[1], [], []], 0, 1, nodes)
    shim.world = SimpleNamespace(root=Path('/tmp/world0'))
    a = shim._world_prefix(row)
    shim.world = SimpleNamespace(root=Path('/tmp/world1'))
    b = shim._world_prefix(row)
    assert a.tokens[1:] == b.tokens[1:] and a.tokens[0] != b.tokens[0]
    assert a.answer_index == row.answer_index + 1
    walk = render_walk(tok, nodes, [0, 1])
    changed = shim._world_prefix(walk)
    assert changed.walk_indices == [i + 1 for i in walk.walk_indices]
    assert changed.answer_index == -1


def test_offline_cache_prefix_and_randomness(tmp_path):
    from cot_navigation.data.dataset import ContextStore
    from cot_navigation.data.scale_prepared import prepare_variants, ScalePreparedRows
    from cot_navigation.data.dataset import collate_examples
    tok = Tokenizer(7, enable_stop=True)
    store = ContextStore.__new__(ContextStore)
    store.root, store.fingerprint, store.shape_names = tmp_path, 'synthetic', ['test']
    edges = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5)]
    store.arrays = dict(sample_id=np.arange(2), node_token=np.tile(np.arange(7), (2, 1)),
        edge_src=np.tile([a for a,b in edges], (2, 1)), edge_dst=np.tile([b for a,b in edges], (2, 1)),
        s_topo=np.array([0, 0]), t_topo=np.array([5, 6]), label=np.array([1, 0]),
        distance=np.array([3, 3]), shape_id=np.array([0, 0]), split_id=np.array([0, 0]))
    cfg = SimpleNamespace(model=SimpleNamespace(max_seq_len=64), train=SimpleNamespace(world_roots=()),
        protocol=SimpleNamespace(name='direct', trajectory_variants=8, cot_margin=8))
    for protocol in ('direct', 'path', 'dfs_state', 'bfs_state', 'dfs_nodes', 'bfs_nodes'):
        cfg.protocol.name = protocol
        prepare_variants(store, cfg, tok, protocol)
        prepared = ScalePreparedRows(store, cfg, tok)
        bodies = []
        for i in range(2):
            for v in range(prepared.variants(i)):
                r = prepared.row(i, v)
                assert len(r.tokens) <= 56
                assert r.tokens[r.answer_index:] == [tok.label_id(r.label), tok.id('<EOS>')]
                packed = collate_examples([r], tok.pad)
                assert packed['answer_mask'].sum() == 1
                if protocol == 'path' and i == 1:
                    assert r.tokens[r.prompt_len] == tok.node_id(0)
                if r.meta.get('truncated') and protocol.endswith('state'):
                    assert r.tokens[r.answer_index-1] == tok.id('<STOP>')
                    assert r.answer_index-1 == r.prompt_len or r.tokens[r.answer_index-2] == tok.id(';')
                if i == 0:
                    bodies.append(tuple(r.tokens[r.prompt_len:r.answer_index]))
        if protocol in ('dfs_state', 'bfs_state'):
            assert len(set(bodies)) > 1


def test_decode_budget_rollback_and_model_answer():
    from cot_navigation.evaluation.rollout import generate
    tok = Tokenizer(2, enable_stop=True)
    class Fake(torch.nn.Module):
        max_seq_len = 40
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
        def new_cache(self, batch, device, dtype):
            return SimpleNamespace(lengths=torch.zeros(batch, dtype=torch.long), max_len=40)
        def forward(self, tokens, **kwargs):
            logits = torch.zeros((*tokens.shape, len(tok)))
            for i in range(len(tokens)):
                token = int(tokens[i, -1])
                nxt = tok.id(';') if token == tok.node_id(0) else tok.node_id(0)
                logits[i, -1, nxt] = 20
                if token == tok.id('<STOP>'):
                    logits[i, -1, tok.no] = 10
                    logits[i, -1, tok.yes] = 1
            return logits
    prompt = [tok.id('<QUERY>'), tok.node_id(0), tok.node_id(1), tok.id('|')]
    result = generate(Fake(), [prompt], 'bfs_state', tok, cot_margin=8)[0]
    assert result['pred'] == 0 and result['forced_stop'] and result['answer_constrained']
    assert result['trace_tokens'][-2:] == [tok.id(';'), tok.id('<STOP>')]
    assert len(result['tokens']) <= 32


def test_completion_requires_extra_reports(tmp_path, monkeypatch):
    from cot_navigation.experiments import scheduler as engine
    cfgpath = ROOT / 'configs/experiments/retrain_context.yaml'
    job = SimpleNamespace(config=cfgpath, output=tmp_path, steps=50000,
                          overrides=['evaluation.extra_sets=["wide=data/test"]'])
    (tmp_path/'checkpoints').mkdir()
    (tmp_path/'checkpoints/latest.pt').touch()
    (tmp_path/'eval').mkdir()
    monkeypatch.setattr(engine, 'checkpoint_step', lambda p: 50000)
    for tag in ('best', 'last'):
        (tmp_path/'eval'/f'free_test_{tag}.json').write_text(json.dumps(dict(training_step=50000, overall={'n': 1})))
    assert not engine.job_complete(job)
    for tag in ('best', 'last'):
        (tmp_path/'eval'/f'free_test_{tag}-wide.json').write_text(json.dumps(dict(training_step=50000, overall={'n': 1})))
    assert engine.job_complete(job)

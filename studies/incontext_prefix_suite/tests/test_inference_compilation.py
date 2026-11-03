"""Regressions for the first full validation after compiled training."""
from types import SimpleNamespace

import pytest
import torch
from torch._dynamo.exc import RecompileLimitExceeded

from cot_navigation.config import EvaluationConfig, ModelConfig, TrainConfig
from cot_navigation.models.compiled_inference import CompiledInference
from cot_navigation.models.transformer import build_model
from cot_navigation.training.trainer import Trainer


def make_model(order='parallel', graph='bidirectional', device='cpu'):
    cfg = ModelConfig(architecture='dual', attention_order=order, graph_attention=graph,
                      layers=1, heads=2, d_model=16, d_ff=32, max_seq_len=1536,
                      graph_max_tokens=868)
    return build_model(cfg, 40).to(device)


def prompt(device='cpu'):
    x = torch.full((8, 106), 16, dtype=torch.long, device=device)
    x[:, 102:] = torch.tensor([3, 16, 17, 4], device=device)
    return x


def test_only_cache_limit_uses_persistent_phase_fallback(monkeypatch):
    calls = []
    def compile_stub(fn, **kw):
        if fn.__name__ != '_decode':
            return fn
        def fail(*args):
            calls.append('limit')
            raise RecompileLimitExceeded('cache_size_limit reached')
        return fail
    monkeypatch.setattr(torch, 'compile', compile_stub)
    m = make_model().eval()
    wrapped = CompiledInference(m)
    x = prompt()
    c = m.new_cache(8, 'cpu', torch.float32)
    with torch.no_grad():
        wrapped(x, cache=c, cache_mode='prefill')
        c.valid_length = 107
        positions = torch.full((8, 1), 106)
        with pytest.warns(RuntimeWarning, match='cache limit'):
            a = wrapped(x[:, :1], positions=positions, cache=c, cache_mode='decode')
        b = wrapped(x[:, :1], positions=positions, cache=c, cache_mode='decode')
        torch.testing.assert_close(a, b)
    assert calls == ['limit']
    assert wrapped.compile_fallbacks == {'decode'}
    def wrong(*args):
        raise RuntimeError('unrelated model error')
    wrapped._calls['full'] = wrong
    with pytest.raises(RuntimeError, match='unrelated model error'):
        wrapped(x)


def test_checkpoint_precedes_failing_validation():
    t = object.__new__(Trainer)
    t.cfg = SimpleNamespace(train=TrainConfig(max_steps=1, eval_interval=1,
        free_eval_interval=1, save_interval=1, log_interval=1))
    t.raw, t.completed, saved = torch.nn.Linear(1, 1), 0, []
    t._update = lambda **kw: {}
    t._touch_rng = lambda: None
    t.tracker = SimpleNamespace(log=lambda *args: None)
    t._save = lambda: saved.append(t.completed)
    def fail(**kw):
        assert saved == [1]
        raise RuntimeError('validation failed')
    t._evaluate = fail
    with pytest.raises(RuntimeError, match='validation failed'):
        t.fit()
    assert saved == [1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA compilation regression')
@pytest.mark.parametrize('order,graph', [('parallel', 'bidirectional'), ('serial', 'causal')])
def test_training_then_long_decode_and_compaction(order, graph):
    # >868 cached positions crosses the real graph-cache bound. Batch sizes
    # 8,5,2,1 exercise dynamic rows and the singleton specialization.
    # Production uses one architecture per process. Isolate parameterized test
    # cases too, since Dynamo's function-code cache spans module instances.
    torch._dynamo.reset()
    torch.manual_seed(41)
    m = make_model(order, graph, 'cuda')
    train = torch.compile(m, dynamic=True, fullgraph=True)
    x = prompt('cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        train(x).float().square().mean().backward()
    m.eval()
    wrapped = CompiledInference(m)
    cache = m.new_cache(8, 'cuda', torch.bfloat16)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        torch.testing.assert_close(wrapped(x), m(x), atol=.006, rtol=.03)
        wrapped(x, cache=cache, cache_mode='prefill')
        cache.lengths.fill_(106)
        for i in range(900):
            if i in (50, 200, 500):
                size = {50: 5, 200: 2, 500: 1}[i]
                cache.select_batch(torch.arange(size, device='cuda'))
            pos = cache.lengths[:, None]
            cache.valid_length = 107 + i
            token = torch.full_like(pos, 18)
            got = wrapped(token, positions=pos, cache=cache, cache_mode='decode')
            if i in (0, 51, 201, 501, 760, 899):
                expected = m(token, positions=pos, cache=cache, cache_mode='decode')
                torch.testing.assert_close(got, expected, atol=.006, rtol=.03)
                assert torch.isfinite(got).all()
            cache.lengths += 1
        # Length growth must reuse graphs, rather than specializing each token.
        assert not wrapped.compile_fallbacks
    m.train()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        train(torch.cat((x, x[:, :1]), dim=1)).float().square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())

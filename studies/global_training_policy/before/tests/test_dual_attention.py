"""Information-flow, incremental decoding and gradient checks for both orders."""
import pytest
import torch

from cot_navigation.config import ModelConfig
from cot_navigation.models.transformer import build_model


def model(order='parallel', graph='causal', device='cpu'):
    return build_model(ModelConfig(architecture='dual', attention_order=order,
        graph_attention=graph, graph_max_tokens=16, layers=2, heads=2,
        d_model=32, d_ff=64, max_seq_len=48), 40).to(device)


def rows(device='cpu'):
    return torch.tensor([[2, 16, 17, 3, 16, 18, 4, 16, 4, 17, 5, 6],
                         [2, 18, 19, 2, 19, 20, 3, 18, 20, 4, 18, 4]], device=device)


@pytest.mark.parametrize('order', ['parallel', 'serial'])
@pytest.mark.parametrize('graph', ['causal', 'bidirectional'])
def test_no_future_cot_or_graph_feedback(order, graph):
    torch.manual_seed(2)
    m = model(order, graph).eval()
    x = rows()
    capture, changed = {}, {}
    y = m(x, capture=capture)
    altered = x.clone(); altered[:, 11] = 30
    z = m(altered, capture=changed)
    torch.testing.assert_close(y[:, :11], z[:, :11], atol=1e-6, rtol=1e-6)
    for a, b in zip(capture['resid'], changed['resid']):
        torch.testing.assert_close(a[0, :7], b[0, :7], atol=0, rtol=0)
        torch.testing.assert_close(a[1, :10], b[1, :10], atol=0, rtol=0)
    y.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


@pytest.mark.parametrize('order', ['parallel', 'serial'])
@pytest.mark.parametrize('graph', ['causal', 'bidirectional'])
def test_mixed_prefix_cache_and_compaction(order, graph):
    torch.manual_seed(4)
    m = model(order, graph).eval()
    x = rows()
    with torch.no_grad():
        full = m(x)
        # Prefill may include CoT; graph memory must still stay fixed.
        cache = m.new_cache(2, 'cpu', torch.float32)
        prefill = m(x[:, :10], cache=cache, cache_mode='prefill')
        torch.testing.assert_close(prefill, full[:, :10], atol=2e-6, rtol=2e-5)
        for pos in (10, 11):
            cache.valid_length = pos + 1
            out = m(x[:, pos:pos+1], positions=torch.full((2, 1), pos), cache=cache, cache_mode='decode')
            torch.testing.assert_close(out[:, 0], full[:, pos], atol=2e-6, rtol=2e-5)
        cache.select_batch(torch.tensor([1, 0]))
        assert cache.prefix_lengths.tolist() == [10, 7]
        # Rewind/overwrite as used by forced STOP; stale later tokens must be hidden.
        cache.valid_length = 11
        out = m(x[[1, 0], 10:11], positions=torch.full((2, 1), 10), cache=cache, cache_mode='decode')
        torch.testing.assert_close(out[:, 0], full[[1, 0], 10], atol=2e-6, rtol=2e-5)


def test_graph_direction_and_matched_parameters():
    torch.manual_seed(8)
    causal, bidir = model(), model(graph='bidirectional')
    bidir.load_state_dict(causal.state_dict())
    x = rows()[:1]
    z = x.clone(); z[:, 5] = 29
    torch.testing.assert_close(causal(x)[:, 0], causal(z)[:, 0], atol=0, rtol=0)
    assert not torch.allclose(bidir(x)[:, 0], bidir(z)[:, 0], atol=1e-7)
    assert sum(p.numel() for p in causal.parameters()) == sum(p.numel() for p in model('serial').parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('order', ['parallel', 'serial'])
def test_compiled_cuda_decode(order):
    torch.manual_seed(9)
    m = model(order, 'bidirectional', 'cuda').eval()
    compiled = torch.compile(m, dynamic=True, fullgraph=True)
    x = rows('cuda')
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        expected = m(x)
        actual = compiled(x)
        torch.testing.assert_close(actual, expected, atol=.006, rtol=.03)
        cache = m.new_cache(2, 'cuda', torch.bfloat16)
        compiled(x[:, :10], cache=cache, cache_mode='prefill')
        cache.valid_length = 11
        out = compiled(x[:, 10:11], positions=torch.full((2, 1), 10, device='cuda'), cache=cache, cache_mode='decode')
        torch.testing.assert_close(out[:, 0], expected[:, 10], atol=.006, rtol=.03)


def test_rollout_retention_is_opt_in(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from cot_navigation.config import EvaluationConfig, ProtocolConfig, TrainConfig
    from cot_navigation.protocols.tokenizer import Tokenizer
    from cot_navigation.training.trainer import Trainer
    import cot_navigation.training.trainer as module
    t = object.__new__(Trainer)
    t.raw = model()
    t.device = torch.device('cpu')
    t.cfg = SimpleNamespace(evaluation=EvaluationConfig(save_trace_tokens=True, trace_diagnostics=False),
        protocol=ProtocolConfig(name='dfs_state'), train=TrainConfig(precision='fp32'),
        data=SimpleNamespace(n_nodes=16))
    t.out, t.persist, t.completed, t.tok = tmp_path, True, 100, Tokenizer(16)
    row = SimpleNamespace(tokens=[3, 16, 17, 4, 16, 6, 1], prompt_len=4,
                          label=1, filler_tokens=0, meta={'index': 7})
    def fake(*args, **kwargs):
        return [dict(tokens=row.tokens, trace_tokens=[16], pred=1,
                     format_fail=False, generation_limit_reached=False)]
    monkeypatch.setattr(module, 'generate', fake)
    t._free([row])
    saved = json.loads((tmp_path / 'eval/predictions_val_00000100.jsonl').read_text())
    assert saved['prompt_tokens'] == row.tokens[:4]
    assert saved['trace_tokens'] == [16]

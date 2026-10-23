"""Regression checks for decode cropping, row compaction and STOP rewind."""
import pytest
import torch
from torch import nn

from cot_navigation.evaluation.rollout import generate, _decode_step
from cot_navigation.models.attention import KVCache
from cot_navigation.models.transformer import TransformerLM
from cot_navigation.protocols.tokenizer import Tokenizer


@pytest.mark.parametrize('attn', ['math', 'sdpa'])
def test_cropped_logits_ignore_future_cache_and_match_reference(attn):
    torch.manual_seed(8)
    model = TransformerLM(80, 2, 1, 16, 32, 64, attn_impl=attn).eval()
    caches = [model.new_cache(2, 'cpu', torch.float32) for _ in range(2)]
    tokens = torch.randint(1, 50, (2, 8))
    with torch.no_grad():
        for cache in caches:
            model(tokens, positions=torch.arange(8).expand(2, -1), cache=cache, cache_mode='prefill')
            cache.lengths.copy_(torch.tensor([4, 8]))
            # Make unused future entries nonzero: a missing mask must fail.
            for k, v in zip(cache.k, cache.v):
                k[:, :, 9:] = 100
                v[:, :, 9:] = -100
        caches[1].valid_length = 8
        step = torch.tensor([[11], [12]])
        a = _decode_step(model, caches[0], step, 'fp32')
        b = _decode_step(model, caches[1], step, 'fp32')
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


class Scheduled(nn.Module):
    """Row identity is cached; position-driven outputs survive row removal."""
    def __init__(self, tok):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.tok, self.max_seq_len = tok, 40

    def new_cache(self, batch, device, dtype):
        cache = KVCache(1, batch, 1, self.max_seq_len, 4, device, dtype)
        cache.record_tokens = torch.zeros(batch, self.max_seq_len, dtype=torch.long, device=device)
        return cache

    def forward(self, tokens, positions=None, cache=None, cache_mode=None):
        if cache_mode == 'prefill':
            cache.record_tokens[:, 0] = tokens[:, 0]
        logits = torch.full((*tokens.shape, len(self.tok)), -30.)
        for row in range(len(tokens)):
            marker = int(cache.record_tokens[row, 0])
            pos = int(positions[row, -1])
            if marker == self.tok.node_id(0):
                picked = self.tok.yes
            elif marker == self.tok.node_id(1) and pos > 10:
                picked = self.tok.id('<EOS>')
            else:
                picked = self.tok.id(';') if pos % 7 == 0 else self.tok.node_id(3)
            logits[row, -1, picked] = 30
            # STOP must use model NO/YES logits, not an injected gold answer.
            logits[row, -1, self.tok.no] = max(float(logits[row, -1, self.tok.no]), -10.)
        return logits


@pytest.mark.parametrize('protocol', ['path', 'dfs_state', 'bfs_state'])
@pytest.mark.parametrize('crop,compact', [(True,False), (False,True), (True,True)])
def test_mixed_finish_eos_stop_rewind_and_output_order(protocol, crop, compact):
    tok = Tokenizer(8, enable_stop=True)
    prompts = [[tok.node_id(0)]*4, [tok.node_id(1)]*7, [tok.node_id(2)]*2]
    reference = generate(Scheduled(tok), prompts, protocol, tok, cot_margin=8,
                         crop_cache=False, compact_finished=False)
    actual = generate(Scheduled(tok), prompts, protocol, tok, cot_margin=8,
                      crop_cache=crop, compact_finished=compact)
    assert actual == reference
    assert actual[0]['pred'] == 1
    assert actual[1]['format_fail']
    assert actual[2]['forced_stop'] and actual[2]['pred'] == 0


def test_select_batch_preserves_cache_positions_and_record_history():
    cache = KVCache(2, 3, 1, 12, 4, 'cpu', torch.float32)
    cache.lengths.copy_(torch.tensor([3, 8, 5]))
    cache.record_tokens = torch.arange(36).reshape(3, 12)
    for k in cache.k:
        k.copy_(torch.arange(3)[:, None, None, None])
    cache.select_batch(torch.tensor([2, 0]))
    assert cache.lengths.tolist() == [5, 3]
    assert cache.record_tokens[:, 0].tolist() == [24, 0]
    assert cache.k[0][:, 0, 0, 0].tolist() == [2, 0]

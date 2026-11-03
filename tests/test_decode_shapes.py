"""Reuse one decoder across graph scales, cache alignment, compaction and rewind."""
import pytest
import torch

from cot_navigation.config import ModelConfig
from cot_navigation.models.compiled_inference import CompiledInference
from cot_navigation.models.transformer import build_model


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA dynamic decode regression')
def test_repeated_scales_and_compaction_do_not_exhaust_decode_cache():
    torch._dynamo.reset()
    cfg = ModelConfig(architecture='dual', layers=1, heads=2, d_model=16, d_ff=32,
                      max_seq_len=1536, graph_max_tokens=868, graph_attention='bidirectional')
    m = build_model(cfg, 40).cuda().eval()
    wrapped = CompiledInference(m)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for plen, batch in ((100, 64), (220, 16), (604, 32), (868, 8), (148, 1), (100, 64)):
            x = torch.full((batch, plen), 16, dtype=torch.long, device='cuda')
            x[:, -4:] = torch.tensor([3, 16, 17, 4], device='cuda')
            cache = m.new_cache(batch, 'cuda', torch.bfloat16)
            wrapped(x, cache=cache, cache_mode='prefill')
            cache.lengths.fill_(plen)
            end = plen
            for step in range(420):
                if step in (50, 150, 300):
                    n = min(cache.lengths.numel(), {50: 7, 150: 2, 300: 1}[step])
                    cache.select_batch(torch.arange(n, device='cuda'))
                if step == 200:
                    # STOP rewinds, leaving later keys populated but invisible.
                    end -= 37
                    cache.lengths.fill_(end)
                pos = cache.lengths[:, None]
                cache.valid_length = end + 1
                token = torch.full_like(pos, 18)
                got = wrapped(token, positions=pos, cache=cache, cache_mode='decode')
                if step in (0, 51, 151, 201, 301, 419):
                    reference = m(token, positions=pos, cache=cache, cache_mode='decode')
                    torch.testing.assert_close(got, reference, atol=.006, rtol=.03)
                end += 1
                cache.lengths += 1
        # Exercise the clamped final read view at the end of the context.
        for end in (1527, 1528, 1534, 1535):
            cache.valid_length = end + 1
            cache.lengths.fill_(end)
            token = torch.full((1, 1), 18, device='cuda', dtype=torch.long)
            got = wrapped(token, positions=cache.lengths[:, None], cache=cache, cache_mode='decode')
            assert torch.isfinite(got).all()
        assert not wrapped.compile_fallbacks

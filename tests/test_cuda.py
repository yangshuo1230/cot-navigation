"""Optional accelerator numerics; never creates a training run or checkpoint."""

import pytest
import torch

from cot_navigation.models.transformer import TransformerLM


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA verification is optional on CPU")
def test_compiled_bf16_sdpa_matches_fp32_reference_gradients():
    torch.manual_seed(11)
    reference = TransformerLM(80, 2, 1, 32, 64, 128, attn_impl="math").cuda().eval()
    raw = TransformerLM(80, 2, 1, 32, 64, 128, attn_impl="sdpa").cuda().eval()
    raw.load_state_dict(reference.state_dict())
    optimized = torch.compile(raw)
    tokens = torch.randint(1, 80, (2, 32), device="cuda")
    ref = reference(tokens)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = optimized(tokens)
    ref[:, -1].float().square().mean().backward()
    actual[:, -1].float().square().mean().backward()
    assert float((actual.float() - ref).abs().max()) < .05
    grad_ref = torch.cat([p.grad.flatten() for p in reference.parameters()])
    grad_actual = torch.cat([p.grad.flatten() for p in raw.parameters()])
    relative = (grad_actual - grad_ref).norm() / grad_ref.norm()
    assert float(relative) < .05

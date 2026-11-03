"""Cache compaction/rewind and input-only evaluation batching regressions."""
from types import SimpleNamespace
import pytest
import torch

from cot_navigation.evaluation.rollout import _decode_step, generate
from cot_navigation.models.transformer import TransformerLM
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.training.trainer import Trainer, _generation_chunks


def test_prefix_cache_compaction_matches_full_copy_after_growth_and_rewind():
    torch.manual_seed(32)
    model=TransformerLM(40,2,1,16,32,32,attn_impl='math').eval()
    caches=[model.new_cache(3,'cpu',torch.float32) for _ in range(2)]
    with torch.no_grad():
        for layer in range(2):
            k=torch.randn_like(caches[0].k[layer]);v=torch.randn_like(k)
            for c in caches:c.k[layer].copy_(k);c.v[layer].copy_(v)
        for c in caches:c.lengths.copy_(torch.tensor([6,4,5]))
        index=torch.tensor([2,0])
        reference,fast=caches
        reference.k=[x.index_select(0,index) for x in reference.k]
        reference.v=[x.index_select(0,index) for x in reference.v]
        reference.lengths=reference.lengths.index_select(0,index)
        pointer=fast.k[0].data_ptr()
        fast.valid_length=6;fast.select_batch(index)
        reference.valid_length=6
        assert fast.k[0].data_ptr()==pointer
        for layer in range(2):
            torch.testing.assert_close(fast.k[layer][:,:,:6],reference.k[layer][:,:,:6])
        for step in range(8):
            if step==4:
                # STOP may rewind before the last complete record. A later
                # decode must overwrite stale future keys before reading them.
                for c in caches:c.lengths[0]=4;c.valid_length=9
            token=torch.tensor([[7+step],[8+step]])
            a=_decode_step(model,reference,token,'fp32')
            b=_decode_step(model,fast,token,'fp32')
            torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)


def test_generation_chunking_preserves_order_and_uses_only_prompt_lengths():
    rows=[SimpleNamespace(prompt_len=n,tokens=[0]*gold) for n,gold in [(10,1000),(100,101),(10,9000),(10,10)]]
    chunks=list(_generation_chunks(iter(rows),3,120))
    assert [len(c) for c in chunks]==[1,1,2]
    assert [id(r) for c in chunks for r in c]==list(map(id,rows))
    rows[0].tokens=[0];rows[2].tokens=[0]
    assert [[id(r) for r in c] for c in chunks]==[[id(r) for r in c] for c in _generation_chunks(rows,3,120)]


def test_free_batch_limit_respects_cache_budget_and_direct_needs_no_cache():
    trainer=object.__new__(Trainer)
    trainer.device=torch.device('cuda')
    trainer.cfg=SimpleNamespace(evaluation=SimpleNamespace(free_batch_size=64,batch_size=16,free_max_cache_mb=1024),
        protocol=SimpleNamespace(name='dfs_state'),train=SimpleNamespace(precision='bf16'),
        model=SimpleNamespace(layers=6,d_model=128,max_seq_len=32768))
    assert trainer._free_batch_size()==10
    trainer.cfg.protocol.name='direct'
    assert trainer._free_batch_size()==64


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA compiled generation')
def test_compiled_fp32_generation_matches_reference_with_mixed_prompt_lengths():
    torch.manual_seed(9)
    torch.backends.cuda.matmul.allow_tf32=False
    tok=Tokenizer(16,enable_stop=True)
    model=TransformerLM(len(tok),2,1,16,32,64,attn_impl='sdpa').cuda().eval()
    compiled=torch.compile(model,dynamic=True,fullgraph=True)
    prompts=[[tok.node_id(1)]*4,[tok.node_id(2)]*7,[tok.node_id(3)]*9]
    reference=generate(model,prompts,'dfs_state',tok,'fp32',cot_margin=8)
    actual=generate(compiled,prompts,'dfs_state',tok,'fp32',cot_margin=8)
    assert actual==reference
    # Final evaluation reuses the compiled wrapper while loading best/last
    # into the same parameter objects; it must read the newly loaded weights.
    torch.manual_seed(10)
    model.reset_parameters()
    reference=generate(model,prompts,'dfs_state',tok,'fp32',cot_margin=8)
    actual=generate(compiled,prompts,'dfs_state',tok,'fp32',cot_margin=8)
    assert actual==reference


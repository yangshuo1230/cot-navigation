"""Causal sliding attention measured in complete DFS/BFS records, not tokens.

Input prompt stays visible. A semicolon completes its record, so the hidden
state at ';' predicts the NEXT record using the last K completed records.
Other positions additionally see their current partial record. Absolute RoPE
positions and cached KV states are retained; this does not erase indirect
information carried by recent states.
"""
import torch
from cot_navigation.protocols.tokenizer import Tokenizer

_tok = Tokenizer()
QUERY = _tok.id('<QUERY>')
SEMI = _tok.id(';')

# Separate compilation lets the existing trainer use its normal eager wrapper
# while FlexAttention lowers the custom mask to a fused Triton kernel.
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
_compiled_flex = torch.compile(flex_attention, dynamic=False,
                               options={"max_autotune": True})
_compiled_mask = torch.compile(create_block_mask, dynamic=False)


def compiled_flex(*args, **kwargs):
    # Torch 2.6 dynamic-shape backward has no valid kernel for some sizes.
    # Static specialization is used instead. Max autotuning also supplies
    # four-warp backward candidates: 2.6 skips its default eight-warp candidate
    # for the actual BF16/head-dim-128 model. Curriculum length buckets
    # exceed Dynamo's default eight specializations; scope the larger cache
    # to this call rather than changing existing full-attention experiments.
    with torch._dynamo.config.patch(cache_size_limit=256, accumulated_cache_size_limit=1024,
                                    automatic_dynamic_shapes=False):
        # Conservative tiles fit A800 shared memory at head dimension 128.
        kwargs.setdefault('kernel_options', {'BLOCK_M':32,'BLOCK_N':32,
            'BLOCK_M1':32,'BLOCK_N1':32,'BLOCK_M2':32,'BLOCK_N2':32})
        return _compiled_flex(*args, **kwargs)


def record_metadata(history, positions):
    """O(B*L) boundary metadata; cached suffix past each query stays invisible."""
    length = history.shape[1]
    keys = torch.arange(length, device=history.device)
    is_query = history == QUERY
    # First QUERY belongs to the immutable input, even if output is malformed.
    has_query = is_query.any(-1)
    prompt_end = is_query.to(torch.int64).argmax(-1) + 4
    body = keys[None, :] >= prompt_end[:, None]
    semis = ((history == SEMI) & body & has_query[:, None]).to(torch.int64)
    completed = semis.cumsum(-1)
    key_record = completed - semis
    query_record = completed.gather(1, positions)
    return has_query, prompt_end, key_record, query_record


def record_mask(history, positions, keep_records, *, use_flex=False):
    has_query, prompt_end, key_record, query_record = record_metadata(history, positions)
    length = history.shape[1]
    if not use_flex:
        keys = torch.arange(length, device=history.device)
        causal = keys[None, None, :] <= positions[:, :, None]
        visible = ((keys[None, None, :] < prompt_end[:, None, None]) |
                   (key_record[:, None, :] >= query_record[:, :, None]-keep_records) |
                   ~has_query[:, None, None])
        return (causal & visible)[:, None, :, :]

    # Flex may inspect padded block indices; clamp all metadata loads and mask
    # those lanes explicitly. The same mask is reused by every model layer.
    q_length = positions.shape[1]
    def mask_mod(b, h, q, k):
        qi = q.clamp(max=q_length-1)
        ki = k.clamp(max=length-1)
        causal = k <= positions[b, qi]
        visible = ((k < prompt_end[b]) |
                   (key_record[b, ki] >= query_record[b, qi]-keep_records) |
                   ~has_query[b])
        return (q < q_length) & (k < length) & causal & visible
    with torch._dynamo.config.patch(cache_size_limit=256, accumulated_cache_size_limit=1024,
                                    automatic_dynamic_shapes=False):
        return _compiled_mask(mask_mod, history.shape[0], None, q_length, length,
                              device=str(history.device), _compile=False)

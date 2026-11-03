"""Two half-width attention branches with immutable graph/QUERY memory.

The graph branch encodes prefix tokens and reads only that prefix. The history
branch reads only causal CoT tokens (including the consumed current token).
Parallel adds both deltas; serial applies history before querying the graph.
Both orders have exactly the same parameters. Absolute RoPE positions are kept.
Graph queries are split from the long tail, avoiding a [B,L,L] graph mask.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .attention import KVCache, apply_rope
from .ffn import build_ffn
from .transformer import TransformerLM


class DualCache(KVCache):
    def select_batch(self, indices):
        super().select_batch(indices)
        self.prefix_lengths = self.prefix_lengths.index_select(0, indices)


class Branch(nn.Module):
    def __init__(self, width, rope_base):
        super().__init__()
        self.dim = width // 2
        self.rope_base = rope_base
        self.qkv = nn.Linear(width, 3 * self.dim, bias=False)
        self.proj = nn.Linear(self.dim, width, bias=False)

    def project(self, x, positions):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = apply_rope(q[:, None], positions, self.rope_base)
        k = apply_rope(k[:, None], positions, self.rope_base)
        return q, k, v[:, None]

    def output(self, out):
        return self.proj(out[:, 0])


class DualBlock(nn.Module):
    def __init__(self, width, d_ff, order, graph_attention, graph_max_tokens, rope_base):
        super().__init__()
        self.order, self.graph_attention = order, graph_attention
        self.graph_max_tokens = graph_max_tokens
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.graph, self.history = Branch(width, rope_base), Branch(width, rope_base)
        self.ffn = build_ffn('standard', width, d_ff)

    def forward(self, x, positions, prefix, cache=None, layer_idx=0,
                cache_mode=None, capture=None, return_attn=False):
        if return_attn:
            raise ValueError('dual attention weight export is not implemented')
        decode = cache is not None and cache_mode == 'decode'
        norm = self.norm1(x)
        hq, hk, hv = self.history.project(norm, positions)
        # Each branch owns one cache head; graph keys never read the CoT slots.
        if decode:
            b = torch.arange(x.shape[0], device=x.device)[:, None]
            cache.k[layer_idx][b, 1, positions] = hk[:, 0]
            cache.v[layer_idx][b, 1, positions] = hv[:, 0]
            # SDPA selects different kernels for aligned/unaligned K lengths.
            # Round the read view only; absolute-position masks exclude unused
            # slots and STOP rewinds continue to overwrite their own positions.
            valid = cache.max_len if cache.valid_length is None else cache.valid_length
            end = min(cache.max_len, ((valid + 7) // 8) * 8)
            hk, hv = cache.k[layer_idx][:, 1:2, :end], cache.v[layer_idx][:, 1:2, :end]
        elif cache is not None:
            # Graph head is written below once the serial/parallel query is built.
            b = torch.arange(x.shape[0], device=x.device)[:, None]
            cache.k[layer_idx][b, 1, positions] = hk[:, 0]
            cache.v[layer_idx][b, 1, positions] = hv[:, 0]
        key = torch.arange(hk.shape[2], device=x.device)
        hkeep = key[None, None, None, :] >= prefix[:, None, None, None]
        if decode:
            hkeep = hkeep & (key[None, None, None, :] <= positions[:, None, :, None])
        hout = F.scaled_dot_product_attention(hq, hk, hv, attn_mask=hkeep, is_causal=not decode)
        hdelta = self.history.output(hout)
        # Explicit zero also makes empty prefix histories safe across SDPA backends.
        hdelta = hdelta * (positions >= prefix[:, None])[:, :, None]
        intermediate = x + hdelta
        gnorm = self.norm1(intermediate) if self.order == 'serial' else norm
        gq, gk, gv = self.graph.project(gnorm, positions)
        if cache is not None and not decode:
            cache.k[layer_idx][b, 0, positions] = gk[:, 0]
            cache.v[layer_idx][b, 0, positions] = gv[:, 0]
        if decode:
            # The immutable graph prefix fits this static bound. Using the
            # history length here introduced another pair of decode graphs.
            size = min(cache.max_len, self.graph_max_tokens)
            gk, gv = cache.k[layer_idx][:, 0:1, :size], cache.v[layer_idx][:, 0:1, :size]
        else:
            size = min(x.shape[1], self.graph_max_tokens)
            gk, gv = gk[:, :, :size], gv[:, :, :size]
        gkey = torch.arange(size, device=x.device)
        gkeep = gkey[None, None, None, :] < prefix[:, None, None, None]
        if self.graph_attention == 'causal' and not decode:
            # Only prefix queries are causal; every CoT query sees the full graph.
            first_pos = positions[:, :size]
            first_keep = gkeep & ((first_pos[:, None, :, None] >= prefix[:, None, None, None])
                                     | (gkey[None, None, None, :] <= first_pos[:, None, :, None]))
            first = F.scaled_dot_product_attention(gq[:, :, :size], gk, gv, attn_mask=first_keep)
            if x.shape[1] > size:
                tail = F.scaled_dot_product_attention(gq[:, :, size:], gk, gv, attn_mask=gkeep)
                gout = torch.cat((first, tail), dim=2)
            else:
                gout = first
        else:
            gout = F.scaled_dot_product_attention(gq, gk, gv, attn_mask=gkeep)
        x = intermediate + self.graph.output(gout)
        x = x + self.ffn(self.norm2(x))
        if capture is not None:
            capture.setdefault('resid', []).append(x)
        return x


class DualAttentionLM(TransformerLM):
    def __init__(self, cfg, vocab, pad_id=0):
        # Build only the actual modules, preserving the established LM/cache API.
        nn.Module.__init__(self)
        self.vocab, self.n_layers, self.n_heads = vocab, cfg.layers, 2
        self.d_model, self.d_ff = cfg.d_model, cfg.d_ff
        self.max_seq_len, self.pad_id = cfg.max_seq_len, pad_id
        self.ffn_kind, self.attn_impl = cfg.ffn, cfg.attn_impl
        self.swa_records, self.swa_backend = 0, cfg.swa_backend
        self.attention_order, self.graph_attention = cfg.attention_order, cfg.graph_attention
        self.graph_max_tokens = cfg.graph_max_tokens
        self.embed = nn.Embedding(vocab, cfg.d_model, padding_idx=pad_id)
        self.blocks = nn.ModuleList([
            DualBlock(cfg.d_model, cfg.d_ff, cfg.attention_order, cfg.graph_attention,
                      cfg.graph_max_tokens, cfg.rope_base) for _ in range(cfg.layers)])
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, vocab, bias=False)
        self.lm_head.weight = self.embed.weight
        self.reset_parameters()

    def new_cache(self, batch, device, dtype):
        return DualCache(self.n_layers, batch, 2, self.max_seq_len, self.d_model // 2, device, dtype)

    def forward(self, tokens, positions=None, cache=None, cache_mode=None,
                capture=None, return_attn=False):
        if positions is None:
            positions = torch.arange(tokens.shape[1], device=tokens.device)[None].expand(tokens.shape[0], -1)
        if cache is not None and cache_mode == 'decode':
            prefix = cache.prefix_lengths
        else:
            # QUERY ID is stable (3); prefix ends after QUERY source target |.
            query = tokens == 3
            prefix = query.to(torch.int64).argmax(dim=1) + 4
            torch._assert_async(query.any(dim=1).all(), 'dual attention requires QUERY prompts')
            torch._assert_async((prefix <= self.graph_max_tokens).all(), 'graph_max_tokens is too small')
            if cache is not None:
                cache.prefix_lengths = prefix
        x = self.embed(tokens)
        for i, block in enumerate(self.blocks):
            x = block(x, positions, prefix, cache, i, cache_mode, capture, return_attn)
        return self.lm_head(self.final_norm(x))

    def report(self):
        return {**super().report(), 'architecture': 'dual',
                'attention_order': self.attention_order, 'graph_attention': self.graph_attention,
                'branch_dim': self.d_model // 2, 'graph_max_tokens': self.graph_max_tokens}

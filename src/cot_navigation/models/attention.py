"""Causal attention with RoPE, an SDPA path, and a math reference path.

Training sequences are right-padded. With a square causal mask, a real token
only attends left, so padding on the right is invisible to those tokens and
``is_causal=True`` is valid. Decode is different: ``q_len`` is 1 and ``k_len``
is the cache length. A top-left causal mask would hide the cache. Decode
therefore uses an explicit "key index <= this token's absolute position" mask
and never sets ``is_causal`` when the lengths differ.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def apply_rope(x: torch.Tensor, positions: torch.Tensor, base: float) -> torch.Tensor:
    """Rotate ``x`` of shape ``[B, H, L, D]`` at absolute ``positions`` ``[B, L]``."""
    dim = x.shape[-1]
    if dim % 2 != 0:
        raise ValueError("RoPE head dimension must be even")
    half = dim // 2
    freq = base ** (-torch.arange(0, half, device=x.device, dtype=torch.float32) / half)
    angles = positions.to(torch.float32)[:, None, :, None] * freq[None, None, None, :]
    cos = angles.cos().to(dtype=x.dtype)
    sin = angles.sin().to(dtype=x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


def math_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    keep: torch.Tensor | None,
    return_weights: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Reference attention. ``keep`` is bool ``[B, 1, Q, K]`` or None (full causal already applied)."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    if keep is not None:
        scores = scores.masked_fill(~keep, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1)
    out = torch.matmul(weights, v.float()).to(dtype=q.dtype)
    return out, weights.to(dtype=q.dtype) if return_weights else None


class KVCache:
    """Packed absolute-position cache. Padding slots may hold stale keys; the mask ignores them."""

    def __init__(self, n_layers: int, batch: int, heads: int, max_len: int, dim: int, device, dtype):
        self.k = [torch.zeros(batch, heads, max_len, dim, device=device, dtype=dtype) for _ in range(n_layers)]
        self.v = [torch.zeros(batch, heads, max_len, dim, device=device, dtype=dtype) for _ in range(n_layers)]
        self.lengths = torch.zeros(batch, dtype=torch.long, device=device)
        self.max_len = max_len

    def write(self, layer: int, k: torch.Tensor, v: torch.Tensor, positions: torch.Tensor) -> None:
        batch, heads, length, _dim = k.shape
        b_idx = torch.arange(batch, device=k.device)[:, None, None]
        h_idx = torch.arange(heads, device=k.device)[None, :, None]
        pos = positions[:, None, :].expand(batch, heads, length)
        self.k[layer][b_idx, h_idx, pos] = k
        self.v[layer][b_idx, h_idx, pos] = v


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, rope_base: float, attn_impl: str, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must divide into heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rope_base = rope_base
        self.attn_impl = attn_impl
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        cache: KVCache | None = None,
        layer_idx: int = 0,
        cache_mode: str | None = None,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, length, _ = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = apply_rope(q.transpose(1, 2), positions, self.rope_base)
        k = apply_rope(k.transpose(1, 2), positions, self.rope_base)
        v = v.transpose(1, 2)
        weights = None
        if cache is not None and cache_mode == "decode":
            cache.write(layer_idx, k, v, positions)
            k_all = cache.k[layer_idx]
            v_all = cache.v[layer_idx]
            key_index = torch.arange(k_all.shape[2], device=x.device)
            # Absolute position of the newest query. q_len may be 1 while k_len is longer.
            newest = positions[:, -1]
            keep = key_index.view(1, 1, 1, -1) <= newest.view(-1, 1, 1, 1)
            out, weights = self._attend(q, k_all, v_all, keep, force_math=return_attn or self.attn_impl == "math")
        elif cache is not None and cache_mode == "prefill":
            cache.write(layer_idx, k, v, positions)
            out, weights = self._attend(q, k, v, None, force_math=return_attn or self.attn_impl == "math", causal=True)
        else:
            out, weights = self._attend(q, k, v, None, force_math=return_attn or self.attn_impl == "math", causal=True)
        out = out.transpose(1, 2).contiguous().view(batch, length, -1)
        return self.drop(self.proj(out)), weights

    def _attend(self, q, k, v, keep, force_math: bool, causal: bool = False):
        if force_math or self.attn_impl == "math":
            if causal and keep is None:
                q_len, k_len = q.shape[-2], k.shape[-2]
                q_idx = torch.arange(q_len, device=q.device)
                k_idx = torch.arange(k_len, device=q.device)
                # Square training mask. This branch is not used for q_len=1 decode.
                keep = (k_idx.view(1, 1, 1, k_len) <= q_idx.view(1, 1, q_len, 1))
            return math_attention(q, k, v, keep, return_weights=True if force_math else False)
        if causal and keep is None and q.shape[-2] == k.shape[-2]:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return out, None
        if causal and keep is None:
            raise RuntimeError("refusing is_causal when q_len != k_len; pass an explicit mask")
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=keep)
        return out, None

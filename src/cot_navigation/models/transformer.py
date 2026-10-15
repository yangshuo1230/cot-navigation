"""Pre-LN causal Transformer.

Activations are exported only when the caller passes a ``capture`` dict.
Training does not retain attention matrices. ``return_attn=True`` forces the
math reference inside attention so weights can be inspected; logits are then
checked against the SDPA path by tests.
"""

from __future__ import annotations

import torch
from torch import nn

from cot_navigation.models.attention import CausalSelfAttention, KVCache
from cot_navigation.models.ffn import build_ffn


class Block(nn.Module):
    def __init__(self, d_model: int, attn: CausalSelfAttention, ffn: nn.Module, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = attn
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = ffn
        self.drop = nn.Dropout(dropout)

    def forward(self, x, positions, cache=None, layer_idx=0, cache_mode=None, return_attn=False, capture=None):
        delta, weights = self.attn(
            self.norm1(x),
            positions,
            cache=cache,
            layer_idx=layer_idx,
            cache_mode=cache_mode,
            return_attn=return_attn,
        )
        x = x + self.drop(delta)
        x = x + self.drop(self.ffn(self.norm2(x)))
        if capture is not None:
            capture.setdefault("resid", []).append(x)
        return x, weights


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab: int,
        layers: int,
        heads: int,
        d_model: int,
        d_ff: int,
        max_seq_len: int,
        ffn: str = "standard",
        attn_impl: str = "sdpa",
        dropout: float = 0.0,
        rope_base: float = 10000.0,
        pad_id: int = 0,
    ):
        super().__init__()
        self.vocab = vocab
        self.n_layers = layers
        self.n_heads = heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.ffn_kind = ffn
        self.attn_impl = attn_impl
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab, d_model, padding_idx=pad_id)
        self.blocks = nn.ModuleList()
        shared = build_ffn(ffn, d_model, d_ff) if ffn == "shared_dense" else None
        for _ in range(layers):
            attn = CausalSelfAttention(d_model, heads, rope_base, attn_impl, dropout)
            block_ffn = shared if shared is not None else build_ffn("standard", d_model, d_ff)
            self.blocks.append(Block(d_model, attn, block_ffn, dropout))
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)
        self.lm_head.weight = self.embed.weight
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.embed.weight, std=0.02)
        with torch.no_grad():
            self.embed.weight[self.pad_id].zero_()
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.lm_head:
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        cache: KVCache | None = None,
        cache_mode: str | None = None,
        capture: dict | None = None,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list]:
        if positions is None:
            positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0).expand(tokens.shape[0], -1)
        x = self.embed(tokens)
        weights = []
        for index, block in enumerate(self.blocks):
            x, layer_w = block(
                x,
                positions,
                cache=cache,
                layer_idx=index,
                cache_mode=cache_mode,
                return_attn=return_attn,
                capture=capture,
            )
            if return_attn:
                weights.append(layer_w)
        logits = self.lm_head(self.final_norm(x))
        if return_attn:
            return logits, weights
        return logits

    def new_cache(self, batch: int, device, dtype) -> KVCache:
        return KVCache(self.n_layers, batch, self.n_heads, self.max_seq_len, self.d_model // self.n_heads, device, dtype)

    def report(self) -> dict:
        unique = sum(p.numel() for p in self.parameters())
        d, ff, layers = self.d_model, self.d_ff, self.n_layers
        # Multiply-adds counted as 2 FLOPs. Shared FFN parameters are stored once,
        # but every layer still applies the block, so compute stays proportional to depth.
        proj = layers * (2 * (3 * d * d + d * d) + 2 * (d * ff + ff * d))
        return {
            "unique_parameters": unique,
            "ffn": self.ffn_kind,
            "d_model": d,
            "d_ff": ff,
            "layers": layers,
            "heads": self.n_heads,
            "projection_flops_per_token": proj,
            "attn_impl": self.attn_impl,
        }


def build_model(cfg, vocab: int, pad_id: int = 0) -> TransformerLM:
    return TransformerLM(
        vocab=vocab,
        layers=cfg.layers,
        heads=cfg.heads,
        d_model=cfg.d_model,
        d_ff=cfg.d_ff,
        max_seq_len=cfg.max_seq_len,
        ffn=cfg.ffn,
        attn_impl=cfg.attn_impl,
        dropout=cfg.dropout,
        rope_base=cfg.rope_base,
        pad_id=pad_id,
    )

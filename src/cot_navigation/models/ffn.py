"""Feed-forward blocks.

``shared_dense`` is a normal dense FFN. Sharing is the transformer's choice to
reuse one instance across layers, not a second algorithm hidden in this class.
MoE is intentionally absent: calling it raises rather than silently running a
dense FFN.
"""

from __future__ import annotations

import torch
from torch import nn


class StandardFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        # Exact erf GELU. The tanh approximation would change the reference path.
        self.act = nn.GELU(approximate="none")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.act(self.up(x)))


def build_ffn(kind: str, d_model: int, d_ff: int) -> nn.Module:
    if kind in {"standard", "shared_dense"}:
        return StandardFFN(d_model, d_ff)
    if kind == "moe":
        raise NotImplementedError(
            "MoE is not implemented. Register a new module in models/ffn.py:build_ffn. "
            "There is no silent fallback to a dense FFN."
        )
    raise ValueError(f"unknown ffn kind {kind}")

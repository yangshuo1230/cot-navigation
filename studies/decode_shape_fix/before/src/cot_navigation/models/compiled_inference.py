"""Separate compilation caches for full inference, prefill, and cached decode.

Compiling the LM.forward entry again shares Dynamo's per-code cache with
training. Variable batch sizes, grad modes, and KV lengths can exhaust that
cache. These entry functions are distinct from one another and from training.
If a phase still reaches the compiler's limit, only that phase uses eager
execution thereafter. Model/data errors and unrelated compiler errors propagate.
"""
from __future__ import annotations

import warnings

import torch
from torch import nn
from torch._dynamo.exc import RecompileLimitExceeded


class CompiledInference(nn.Module):
    def __init__(self, raw):
        super().__init__()
        self.raw = raw  # same parameter objects: best/last loads remain visible
        self.max_seq_len = raw.max_seq_len
        self.compile_fallbacks = set()
        self._calls = {
            'full': torch.compile(self._full, dynamic=True, fullgraph=True),
            'prefill': torch.compile(self._prefill, dynamic=True, fullgraph=True),
            'decode': torch.compile(self._decode, dynamic=True, fullgraph=True),
        }

    def new_cache(self, batch, device, dtype):
        return self.raw.new_cache(batch, device, dtype)

    def _full(self, tokens, positions, cache):
        return self.raw(tokens, positions=positions)

    def _prefill(self, tokens, positions, cache):
        return self.raw(tokens, positions=positions, cache=cache, cache_mode='prefill')

    def _decode(self, tokens, positions, cache):
        return self.raw(tokens, positions=positions, cache=cache, cache_mode='decode')

    def forward(self, tokens, positions=None, cache=None, cache_mode=None):
        phase = cache_mode if cache is not None else 'full'
        if phase not in self._calls:
            raise ValueError(f'unsupported inference cache mode: {phase}')
        try:
            return self._calls[phase](tokens, positions, cache)
        except RecompileLimitExceeded:
            warnings.warn(f'{phase} inference reached the compilation cache limit; '
                          'continuing this phase with eager execution.', RuntimeWarning)
            self.compile_fallbacks.add(phase)
            self._calls[phase] = getattr(self, '_' + phase)
            return self._calls[phase](tokens, positions, cache)

"""Stable RNG derivation.

Sample content is a function of explicit integer seeds. Python's built-in
``hash()`` is process-salted and must not decide data. Worker count, shard
size, and resume point are not inputs to ``derive_seed``.
"""

from __future__ import annotations

import hashlib

import numpy as np
from numpy.random import Generator, PCG64

_MASK63 = (1 << 63) - 1


def derive_seed(*parts: int) -> int:
    """Mix integers into a 63-bit seed with BLAKE2s.

    The same parts always yield the same seed, including across processes.
    """
    hasher = hashlib.blake2s(digest_size=8)
    for part in parts:
        hasher.update(int(part).to_bytes(8, "little", signed=True))
    return int.from_bytes(hasher.digest(), "little") & _MASK63


def make_rng(*parts: int) -> Generator:
    """Return a PCG64 generator whose stream depends only on ``parts``."""
    return Generator(PCG64(derive_seed(*parts)))


def spawn_rng(rng: Generator) -> Generator:
    """Fork a child generator without using the global NumPy state."""
    seed = int(rng.integers(0, _MASK63))
    return Generator(PCG64(seed))


def permutation(rng: Generator, n: int) -> np.ndarray:
    return rng.permutation(n).astype(np.int64, copy=False)

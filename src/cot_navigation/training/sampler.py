"""Order of query and walk exposure.

The sampler seed does not include the protocol name, so the four stage-B
branches that share ``exposure_seed`` see the same query and walk order.
Evaluation must not draw from these generators.
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator, PCG64


class IndexSampler:
    def __init__(self, n_items: int, batch_size: int, seed: int):
        if n_items < 1:
            raise ValueError("sampler requires at least one item")
        if batch_size < 1:
            raise ValueError("batch_size")
        self.n_items = int(n_items)
        self.batch_size = int(batch_size)
        self.rng = Generator(PCG64(int(seed)))
        self.epoch = 0
        self.cursor = 0
        self.order = self.rng.permutation(self.n_items)

    def next(self) -> list[int]:
        picked: list[int] = []
        while len(picked) < self.batch_size:
            if self.cursor >= self.n_items:
                self.epoch += 1
                self.order = self.rng.permutation(self.n_items)
                self.cursor = 0
            take = min(self.batch_size - len(picked), self.n_items - self.cursor)
            picked.extend(int(i) for i in self.order[self.cursor : self.cursor + take].tolist())
            self.cursor += take
        return picked

    def state_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "cursor": self.cursor,
            "order": np.asarray(self.order, dtype=np.int64),
            "rng": self.rng.bit_generator.state,
            "n_items": self.n_items,
            "batch_size": self.batch_size,
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state["n_items"]) != self.n_items or int(state["batch_size"]) != self.batch_size:
            raise ValueError("sampler shape does not match the checkpoint")
        self.epoch = int(state["epoch"])
        self.cursor = int(state["cursor"])
        self.order = np.asarray(state["order"], dtype=np.int64)
        self.rng.bit_generator.state = state["rng"]


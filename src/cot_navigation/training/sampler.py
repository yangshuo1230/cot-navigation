"""Order of query and walk exposure.

The sampler seed does not include the protocol name, so the four stage-B
branches that share ``exposure_seed`` see the same query and walk order.
Evaluation must not draw from these generators.
"""

from __future__ import annotations

import numpy as np
import warnings
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
        return self.take(self.batch_size)

    def take(self, count: int) -> list[int]:
        """Draw a quota without throwing away unconsumed items in an epoch."""
        if count < 0:
            raise ValueError("count must be nonnegative")
        picked: list[int] = []
        while len(picked) < count:
            if self.cursor >= self.n_items:
                self.epoch += 1
                self.order = self.rng.permutation(self.n_items)
                self.cursor = 0
            take = min(count - len(picked), self.n_items - self.cursor)
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


class FixedQuerySampler:
    """Mix atomic/multi-hop tasks and optionally balance labels by resampling.

    Returns local indices into the trainer's train_ids, never held-out ids.
    Every item in each selected stratum is visited before its next shuffle.
    Separate RNGs keep query/walk exposure identical across output protocols.
    """

    def __init__(self, distances, labels, batch_size, seed, one_hop_fraction=0.0, balance_labels=False):
        self.batch_size = int(batch_size)
        self.fraction = float(one_hop_fraction)
        self.balance_labels = bool(balance_labels)
        self.step = 0
        self.rng = Generator(PCG64(seed + 101))
        self.groups, self.samplers = {}, {}
        distances, labels = np.asarray(distances), np.asarray(labels)
        for hop, enabled in (("one", self.fraction > 0), ("multi", True)):
            if not enabled:
                continue
            mask = distances == 1 if hop == "one" else distances >= 2
            for label in ((0, 1) if self.balance_labels else (-1,)):
                key = f"{hop}/{label}"
                ids = np.flatnonzero(mask & (labels == label if label >= 0 else True))
                if not len(ids):
                    raise ValueError(f"empty training query stratum {key}; change data/sampling configuration")
                self.groups[key] = ids
                self.samplers[key] = IndexSampler(len(ids), batch_size, seed + 200 + len(self.groups))

    def next(self):
        # Cumulative rounding realizes fractional quotas without a long-run
        # bias even when fraction * batch_size is not an integer.
        one = int((self.step + 1) * self.batch_size * self.fraction + 1e-9) - int(
            self.step * self.batch_size * self.fraction + 1e-9)
        picked = []
        for hop, count in (("one", one), ("multi", self.batch_size - one)):
            if not count:
                continue
            quotas = { -1: count }
            if self.balance_labels:
                extra = self.step % 2
                quotas = {0: count // 2 + int(count % 2 and extra == 0),
                          1: count // 2 + int(count % 2 and extra == 1)}
            for label, quota in quotas.items():
                key = f"{hop}/{label}"
                local = self.samplers[key].take(quota)
                picked.extend(self.groups[key][local].tolist())
        self.step += 1
        return self.rng.permutation(picked).tolist()

    def state_dict(self):
        return {"kind": "fixed_query_v1", "step": self.step, "batch_size": self.batch_size,
                "fraction": self.fraction, "balance_labels": self.balance_labels,
                "rng": self.rng.bit_generator.state,
                "groups": {k: v.tolist() for k, v in self.groups.items()},
                "samplers": {k: v.state_dict() for k, v in self.samplers.items()}}

    def load_state_dict(self, state):
        expected = self.state_dict()
        for key in ("kind", "batch_size", "fraction", "balance_labels", "groups"):
            if state.get(key) != expected[key]:
                raise ValueError(f"query sampler resume mismatch: {key}")
        self.step = int(state["step"])
        self.rng.bit_generator.state = state["rng"]
        for key, sampler in self.samplers.items():
            sampler.load_state_dict(state["samplers"][key])


class HopCurriculumSampler:
    """Equal stage durations are configured explicitly; equal per-hop/label replay.

    Stage k assigns equal slots to hops 1..k, balancing YES/NO per slot.
    If hop h has no negative training query, its negative slots draw from
    shorter-hop training negatives. Positive hop difficulty is never relaxed;
    neither future hops nor held-out queries enter the fallback pool.
    Cumulative round-robin quotas differ by at most one even for batch=64/k.
    Every stratum cycles without dropping rows. No held-out ids are supplied.
    """
    def __init__(self, distances, labels, batch_size, seed, stage_steps):
        self.batch_size = int(batch_size)
        self.stage_steps = tuple(int(n) for n in stage_steps)
        self.ends = np.cumsum(self.stage_steps)
        self.step = 0
        self.rng = Generator(PCG64(seed + 503))
        self.groups, self.samplers = {}, {}
        self.negative_fallbacks = {}
        self.label_counts = {hop: 0 for hop in range(1, len(stage_steps)+1)}
        distances, labels = np.asarray(distances), np.asarray(labels)
        for hop in self.label_counts:
            for label in (0, 1):
                key = f"{hop}/{label}"
                ids = np.flatnonzero((distances == hop) & (labels == label))
                if not len(ids) and label == 0:
                    ids = np.flatnonzero((distances >= 1) & (distances < hop) & (labels == 0))
                    if len(ids):
                        source_hops = sorted(int(h) for h in np.unique(distances[ids]))
                        self.negative_fallbacks[key] = {"source_hops": source_hops, "n_items": len(ids)}
                        warnings.warn(
                            f"curriculum stratum {key} is empty; sampling its negative slots "
                            f"from training hops {source_hops} ({len(ids)} queries)",
                            RuntimeWarning, stacklevel=2)
                if not len(ids):
                    raise ValueError(f"empty curriculum stratum {key}")
                self.groups[key] = ids
                self.samplers[key] = IndexSampler(len(ids), batch_size, seed + hop*2 + label)

    @property
    def active_hops(self):
        return min(int(np.searchsorted(self.ends, self.step, side="right")) + 1, len(self.stage_steps))

    def next(self):
        hops = self.active_hops
        start = 0 if hops == 1 else int(self.ends[hops-2])
        cursor = (self.step-start) * self.batch_size
        quotas = np.bincount(np.arange(cursor, cursor+self.batch_size) % hops, minlength=hops)
        picked = []
        for hop, count in enumerate(quotas, 1):
            seen = self.label_counts[hop]
            label_quota = np.bincount(np.arange(seen, seen+count) % 2, minlength=2)
            self.label_counts[hop] += int(count)
            for label, quota in enumerate(label_quota):
                key = f"{hop}/{label}"
                picked.extend(self.groups[key][self.samplers[key].take(int(quota))].tolist())
        self.step += 1
        return self.rng.permutation(picked).tolist()

    def state_dict(self):
        return {"kind": "hop_curriculum_v1", "step": self.step, "batch_size": self.batch_size,
                "stage_steps": self.stage_steps, "rng": self.rng.bit_generator.state,
                "label_counts": dict(self.label_counts),
                "groups": {k: v.tolist() for k, v in self.groups.items()},
                "negative_fallbacks": dict(self.negative_fallbacks),
                "samplers": {k: v.state_dict() for k, v in self.samplers.items()}}

    def load_state_dict(self, state):
        expected = self.state_dict()
        for key in ("kind", "batch_size", "stage_steps", "groups"):
            if state.get(key) != expected[key]:
                raise ValueError(f"curriculum resume mismatch: {key}")
        self.step = int(state["step"])
        self.rng.bit_generator.state = state["rng"]
        self.label_counts = dict(state["label_counts"])
        for key, sampler in self.samplers.items():
            sampler.load_state_dict(state["samplers"][key])


class BufferedSampler:
    """Adapt a balanced logical-batch sampler to variable scale quotas."""
    def __init__(self, sampler):
        self.sampler, self.pending = sampler, []

    def take(self, count):
        while len(self.pending) < count:
            self.pending.extend(self.sampler.next())
        result, self.pending = self.pending[:count], self.pending[count:]
        return result

    def state_dict(self):
        return {'sampler': self.sampler.state_dict(), 'pending': self.pending}

    def load_state_dict(self, state):
        self.sampler.load_state_dict(state['sampler'])
        self.pending = list(state['pending'])


class ScaleMixtureSampler:
    """Draw a batch from several datasets using cumulative scale quotas.

    Returns ``(member_index, local_index)`` pairs. A zero weight does not
    advance that member, so a later stage can start it from the first epoch.
    """

    def __init__(self, sizes, batch_size, seed, weights):
        if batch_size < 1 or any(int(size) < 1 for size in sizes):
            raise ValueError("scale mixture needs a positive batch and nonempty members")
        self.batch_size = int(batch_size)
        self.samplers = [IndexSampler(int(size), self.batch_size, int(seed) + 17 * (i + 1)) for i, size in enumerate(sizes)]
        self.rng = Generator(PCG64(int(seed) + 900))
        self.step = 0
        self.set_weights(weights)

    def set_weights(self, weights):
        values = np.asarray(list(weights), dtype=np.float64)
        if values.shape != (len(self.samplers),) or np.any(values < 0) or float(values.sum()) <= 0:
            raise ValueError("scale weights must be nonnegative and not all zero")
        self.weights = values / values.sum()

    def _cuts(self, steps_done):
        total = int(steps_done) * self.batch_size
        cumulative = 0.0
        cuts = []
        for index, weight in enumerate(self.weights):
            cumulative += float(weight)
            cuts.append(total if index + 1 == len(self.weights) else int(total * cumulative + 1e-9))
        return cuts

    def next(self):
        before, after = self._cuts(self.step), self._cuts(self.step + 1)
        counts, previous = [], 0
        for end_before, end_after in zip(before, after):
            counts.append((end_after - end_before) - previous)
            previous = end_after - end_before
        picked = []
        for index, count in enumerate(counts):
            if count:
                picked.extend((index, local) for local in self.samplers[index].take(count))
        self.step += 1
        order = self.rng.permutation(len(picked))
        return [picked[int(i)] for i in order]

    def state_dict(self):
        return {"kind": "scale_mix_v1", "step": self.step, "batch_size": self.batch_size,
                "weights": self.weights.tolist(), "rng": self.rng.bit_generator.state,
                "samplers": [sampler.state_dict() for sampler in self.samplers]}

    def load_state_dict(self, state):
        if state.get("kind") != "scale_mix_v1" or int(state["batch_size"]) != self.batch_size:
            raise ValueError("scale mixture resume mismatch")
        if len(state["samplers"]) != len(self.samplers):
            raise ValueError("scale mixture member count mismatch")
        self.step = int(state["step"])
        self.set_weights(state["weights"])
        self.rng.bit_generator.state = state["rng"]
        for sampler, saved in zip(self.samplers, state["samplers"]):
            sampler.load_state_dict(saved)


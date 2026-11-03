"""Compact resident tokens + ragged indexing; length-sorted logical batches.

The frozen corpus is loaded once. Only token IDs and row metadata are resident
on device; masks are created for each microbatch. No graph search in this file.
"""
import math
import numpy as np
import torch
from cot_navigation.experiments.artifacts import sha_file, digest
import json


class PackedBank:
    def __init__(self, meta, protocol, device, verify_hash=True, *, bank_root, world_root):
        self.protocol, self.device, self.meta = protocol, device, meta
        self.world = json.loads((world_root / meta['base_path']).read_text())
        assert digest(self.world) == meta['world_hash']
        self.info = meta['banks'][protocol]
        path = bank_root / self.info['path']
        if verify_hash:
            assert sha_file(path) == self.info['sha256'], 'Prepared bank changed'
        with np.load(path) as source:
            self.cpu = {k: source[k] for k in source.files}
        a = self.cpu
        self.lengths = np.diff(a['offsets'])
        self.qptr = a['qptr']
        self.keys = [f'q:{q}:{v}' if q >= 0 else f'adj:{v}' for q, v in zip(a['qids'], a['variants'])]
        # int16 gather is supported on CUDA; all token IDs are below32768.
        assert a['tokens'].max() < 32768
        self.flat = torch.tensor(a['tokens'].astype(np.int16), device=device)
        self.offset = torch.tensor(a['offsets'][:-1], device=device)
        self.length = torch.tensor(self.lengths, device=device)
        self.answer = torch.tensor(a['answers'].astype(np.int64), device=device)
        self.group = torch.tensor(a['groups'].astype(np.int64), device=device)
        self.adj_ids = np.flatnonzero(a['groups'] == 2).tolist()
        self.cells = {}
        for panel in ('short_train', 'long_train'):
            cells = {}
            for qid in self.world['panels'][panel]:
                r = self.world['records'][qid]
                assert self.qptr[qid + 1] > self.qptr[qid]
                cells.setdefault((r['gap'], r['y']), []).append(qid)
            # Match the frozen PATH sampler's gap/label order as well as its
            # separate QUERY/variant RNG streams for paired exposures.
            self.cells[panel] = {key: cells[key] for key in sorted(cells)}

    def sample(self, rng, variant_rng, cfg):
        result = []
        short = int(cfg['query_batch'] * cfg['short_fraction'])
        for panel, count in [('short_train', short), ('long_train', cfg['query_batch'] - short)]:
            cells = self.cells[panel]
            for j in range(count):
                keys = [k for k in cells if k[1] == j % 2]
                qid = rng.choice(cells[rng.choice(keys)])
                result.append(variant_rng.randrange(int(self.qptr[qid]), int(self.qptr[qid + 1])))
        result.extend(rng.choices(self.adj_ids, k=cfg['adjacency_rows']))
        return result

    @staticmethod
    def width(length):
        return int(length) if length < 64 else int(math.ceil(length / 16) * 16)

    def microbatches(self, ids, cfg):
        result, current = [], []
        total = 0
        for row in sorted(ids, key=lambda i: self.lengths[i]):
            width = self.width(self.lengths[row])
            if width > cfg['microbatch_token_budget']:
                raise ValueError('Token budget below an untruncated reference; increase it, never clip data')
            padded = width * (len(current) + 1)
            real = total + int(self.lengths[row])
            if current and (padded > cfg['microbatch_token_budget'] or padded / real > cfg['max_padding_ratio']):
                result.append(current)
                current, total = [], 0
            current.append(row)
            total += int(self.lengths[row])
        if current:
            result.append(current)
        return result

    def batch(self, ids):
        index = torch.tensor(ids, dtype=torch.long, device=self.device)
        width = self.width(max(self.lengths[ids]))
        position = torch.arange(width, device=self.device)[None, :]
        length = self.length[index]
        lookup = self.offset[index, None] + torch.minimum(position, length[:, None] - 1)
        tokens = self.flat[lookup].long()
        tokens = torch.where(position < length[:, None], tokens, 0)
        return tokens, length, self.answer[index], self.group[index]

    def denominators(self, ids, cfg):
        knowledge = sum(int(self.lengths[i]) - 3 for i in ids if self.cpu['groups'][i] == 2)
        short = int(cfg['query_batch'] * cfg['short_fraction'])
        return torch.tensor([cfg['query_batch'], knowledge, short, cfg['query_batch'] - short],
                            dtype=torch.float32, device=self.device)


from cot_navigation.training.packed_loss import logical_loss

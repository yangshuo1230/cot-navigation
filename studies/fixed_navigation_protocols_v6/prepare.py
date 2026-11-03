"""Freeze protocol-specific ragged banks on the original v5 query/node splits.

DFS state and DFS Nodes share neighbor-order variants and discovery trees.
No full-reference trace is truncated. Corpus rows are uint16 tokens, not padded
float masks, to keep storage/GPU memory proportional to real token counts.
"""
from __future__ import annotations
from array import array
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
from semantics import reference, replay_state, replay_nodes

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'fixed_navigation_v5'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.writing')
    temporary.write_text(json.dumps(value, separators=(',', ':')))
    temporary.replace(path)


class Builder:
    def __init__(self, protocol):
        self.protocol = protocol
        self.tokens, self.facts = array('H'), array('I')
        self.offsets, self.fact_offsets, self.qptr = [0], [0], [0]
        self.qids, self.variants, self.answers, self.groups = [], [], [], []
        self.max_length, self.max_qid = 0, None

    def add(self, tokens, qid, variant, answer, group, facts=()):
        self.tokens.extend(tokens)
        self.facts.extend(facts)
        self.offsets.append(len(self.tokens))
        self.fact_offsets.append(len(self.facts))
        self.qids.append(qid)
        self.variants.append(variant)
        self.answers.append(answer)
        self.groups.append(group)
        if len(tokens) > self.max_length:
            self.max_length, self.max_qid = len(tokens), qid

    def write(self, path, cfg):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_suffix('.writing').open('wb') as stream:
            np.savez_compressed(stream,
                tokens=np.frombuffer(self.tokens, dtype=np.uint16), offsets=np.asarray(self.offsets, np.int64),
                qptr=np.asarray(self.qptr, np.int64), qids=np.asarray(self.qids, np.int32),
                variants=np.asarray(self.variants, np.int16), answers=np.asarray(self.answers, np.int32),
                groups=np.asarray(self.groups, np.int8), facts=np.frombuffer(self.facts, dtype=np.uint32),
                fact_offsets=np.asarray(self.fact_offsets, np.int64))
        path.with_suffix('.writing').replace(path)
        capacity = max(256, 1 << (self.max_length + cfg['context_headroom'] - 1).bit_length())
        return {'path': path.relative_to(HERE).as_posix(), 'sha256': sha_file(path),
                'rows': len(self.qids), 'tokens': len(self.tokens), 'max_reference_length': self.max_length,
                'max_reference_query': self.max_qid, 'max_seq_len': capacity,
                'max_new_tokens': capacity - 4, 'bytes': path.stat().st_size,
                'no_truncation': True, 'semantic_tree_fact_count': len(self.facts)}


def prepare_world(meta, cfg, repo):
    from cot_navigation.protocols.tokenizer import Tokenizer
    from cot_navigation.protocols.dfs import dfs_search
    world = json.loads((BASE / meta['path']).read_text())
    assert digest(world) == meta['sha256']
    tok = Tokenizer(256)
    nodes = [tok.node_id(i) for i in world['node_names']]
    inverse = {t: i for i, t in enumerate(nodes)}
    used = {q for values in world['panels'].values() for q in values}
    builders = {p: Builder(p) for p in cfg['protocols']}
    started = time.perf_counter()
    variants_checked = 0
    for qid, r in enumerate(world['records']):
        if qid in used:
            prompt = [tok.id('<QUERY>'), nodes[r['s']], nodes[r['t']], tok.id('|')]
            group = int(r['gap'] >= 4)
            if 'direct' in builders:
                builders['direct'].add(prompt + [tok.label_id(r['y']), tok.id('<EOS>')], qid, 0, 4, group)
            for variant in range(cfg['variants']):
                rng = random.Random(71000000 + world['width'] * 100000 + world['seed'] * 10000 + qid * 37 + variant)
                adj = [vs[:] for vs in world['adj']]
                for vs in adj:
                    rng.shuffle(vs)
                steps = reference(adj, r['s'], r['t'])
                # Independent original project DFS must agree on the machine.
                other = dfs_search(adj, r['s'], r['t'])
                assert [(list(s.stack), list(s.visited), list(s.cands), s.action, s.arg) for s in other] == steps
                assert (steps[-1][3] == 'FOUND') == bool(r['y'])
                state, facts = [], set()
                for stack, visited, candidates, action, arg in steps:
                    for field in (stack, visited, candidates):
                        state.extend(nodes[u] for u in field)
                        state.append(tok.id('|'))
                    state.append(tok.id(f'<{action}>'))
                    if arg is not None:
                        state.append(nodes[arg])
                    state.append(tok.id(';'))
                    facts.update(u * world['n_nodes'] + v for j, u in enumerate(stack) for v in stack[j + 1:])
                order = steps[-1][1]
                node_body = [nodes[u] for u in order]
                a = replay_state(world['adj'], r['s'], r['t'], state, tok, inverse)
                b = replay_nodes(world['adj'], r['s'], r['t'], node_body, inverse)
                assert a['valid'] and b['valid'] and a['terminal'] == b['terminal'] == steps[-1][3]
                assert a['visited'] == b['visited'] == order
                for protocol, body in [('dfs_state', state), ('dfs_nodes', node_body)]:
                    if protocol in builders:
                        tokens = prompt + body + [tok.label_id(r['y']), tok.id('<EOS>')]
                        builders[protocol].add(tokens, qid, variant, len(prompt) + len(body), group, sorted(facts))
                variants_checked += 1
        for builder in builders.values():
            builder.qptr.append(len(builder.qids))
        if qid % 2000 == 0:
            print(json.dumps({'world': meta['path'], 'qid': qid, 'elapsed': time.perf_counter() - started}), flush=True)
    for u, successors in enumerate(world['adj']):
        body = sorted(nodes[v] for v in successors) or [tok.id('<EXHAUST>')]
        for builder in builders.values():
            builder.add([tok.id('<EDGE>'), nodes[u], tok.id('|')] + body + [tok.id('<EOS>')], -1, u, -1, 2)
    name = Path(meta['path']).stem
    banks = {p: b.write(HERE / 'data' / f'{name}_{p}.npz', cfg) for p, b in builders.items()}
    result = {'name': name, 'base_path': meta['path'], 'world_hash': meta['sha256'],
              'n_nodes': world['n_nodes'], 'graph_seed': world['seed'], 'banks': banks,
              'checked_queries': len(used), 'checked_search_variants': variants_checked,
              'elapsed': time.perf_counter() - started}
    print(json.dumps(result), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root', type=Path, required=True)
    p.add_argument('--mode', choices=['pilot', 'all'], default='pilot')
    args = p.parse_args()
    sys.path.insert(0, str(args.repo_root / 'src'))
    cfg = json.loads((HERE / 'experiment.json').read_text())
    base_manifest = json.loads((BASE / 'manifest.json').read_text())
    output = args.repo_root / 'outputs/fixed_navigation_protocols_v6'
    if output.exists() and any(output.rglob('resolved_job.json')):
        raise RuntimeError('Refusing preparation after a v6 training run has started')
    worlds = [w for w in base_manifest['worlds'] if args.mode == 'all' or w['n_nodes'] == 80]
    result = [prepare_world(w, cfg, args.repo_root) for w in worlds]
    atomic_json(HERE / 'manifest.json', {'version': 6, 'config_sha256': digest(cfg),
                'base_manifest_sha256': digest(base_manifest), 'worlds': result,
                'protocols': cfg['protocols'], 'semantic_replay': 'PASS'})


if __name__ == '__main__':
    main()

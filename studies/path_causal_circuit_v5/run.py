"""Frozen next-decision causal screen and natural target-state interchange.

GPU0 only, no training/checkpoints/activation files. FP32 intervention analysis;
native BF16 rollouts supply natural, legal prefixes. Effects concern ONE next
decision, not whole-trajectory success. Natural interchange controls avoid
interpreting an arbitrary zero ablation as a pure semantic memory deletion.
"""
from __future__ import annotations
from collections import defaultdict
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
import torch
import torch.nn.functional as F
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
ROOT = Path.cwd()
HERE = ROOT / 'studies/path_causal_circuit_v5'
SUITE = ROOT / 'studies/fixed_navigation_v5'
sys.path[:0] = [str(ROOT / 'src'), str(SUITE)]
from data import atomic_json, bfs, digest
from cot_navigation.models.transformer import TransformerLM
from cot_navigation.models.attention import apply_rope
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.evaluation.rollout import generate

PANELS = ['short_test', 'long_AA_test', 'long_TA_test', 'long_AT_test', 'long_TT_test']


def pad(rows, tok):
    length = max(map(len, rows))
    tokens = torch.full((len(rows), length), tok.pad, dtype=torch.long, device='cuda:0')
    for b, seq in enumerate(rows):
        tokens[b, :len(seq)] = torch.tensor(seq, device=tokens.device)
    last = torch.tensor([len(s) - 1 for s in rows], device=tokens.device)
    return tokens, last


def query_mean(values, ids):
    if not len(values):
        return None
    return float(np.mean([np.asarray(values)[ids == q].mean() for q in np.unique(ids)]))


def metrics(logits, states, nodes, adjacency, reach):
    probability = torch.tensor(logits).softmax(-1).numpy()
    chosen = logits.argmax(-1)
    inverse = {token: u for u, token in enumerate(nodes)}
    legal, useful, legal_mass, useful_mass, cond, chance, qids = [], [], [], [], [], [], []
    for i, row in enumerate(states):
        successors = adjacency[row['current']]
        good = [v for v in successors if reach[v, row['target']]]
        v = inverse.get(int(chosen[i]), -1)
        plegal = float(probability[i, [nodes[v] for v in successors]].sum())
        pgood = float(probability[i, [nodes[v] for v in good]].sum())
        legal.append(v in successors)
        useful.append(v in good)
        legal_mass.append(plegal)
        useful_mass.append(pgood)
        scores = logits[i, [nodes[v] for v in successors]]
        pp = np.exp(scores - scores.max())
        cond.append(float(pp[[v in good for v in successors]].sum() / pp.sum()))
        chance.append(len(good) / len(successors))
        qids.append(row['qid'])
    qids = np.asarray(qids)
    result = {'decisions': len(states), 'queries': len(np.unique(qids))}
    for name, values in [('legal_next', legal), ('useful_next', useful), ('legal_mass', legal_mass),
                         ('useful_mass', useful_mass), ('useful_given_legal', cond), ('uniform_baseline', chance)]:
        result[name] = query_mean(values, qids)
    return result


@torch.inference_mode()
def collect(model, world, tok, nodes, reach):
    inverse = {token: u for u, token in enumerate(nodes)}
    states = []
    for panel in PANELS:
        ids = world['panels'][panel]
        for start in range(0, len(ids), 128):
            part = ids[start:start + 128]
            prompts = [[tok.id('<QUERY>'), nodes[world['records'][i]['s']],
                        nodes[world['records'][i]['t']], tok.id('|')] for i in part]
            output = generate(model, prompts, 'path', tok, 'bf16', max_new_tokens=16)
            for qid, prompt, generated in zip(part, prompts, output):
                r, trace = world['records'][qid], generated['trace_tokens']
                if not r['y']:
                    continue
                for j, token in enumerate(trace):
                    current = inverse.get(token, -1)
                    if current < 0 or (j == 0 and current != r['s']):
                        break
                    if j and current not in world['adj'][inverse[trace[j - 1]]]:
                        break
                    if current == r['t']:
                        break
                    successors = world['adj'][current]
                    good = [v for v in successors if reach[v, r['t']]]
                    if good and len(good) < len(successors) and r['t'] // world['width'] - current // world['width'] >= 2:
                        states.append({'qid': qid, 'panel': panel, 'source': r['s'], 'target': r['t'],
                            'target_layer': r['t'] // world['width'], 'current': current,
                            'prefix': prompt + trace[:j + 1], 'first': j == 0,
                            'native_next': generated['generated'][j + 1], 'useful': good})
    return states


@torch.inference_mode()
def ablation_forward(model, rows, tok, condition):
    output = []
    max_sham_difference = 0.
    for start in range(0, len(rows), 128):
        part = rows[start:start + 128]
        tokens, last = pad([r['prefix'] for r in part], tok)
        batch = torch.arange(len(part), device=tokens.device)
        handles = []
        kind, layer = condition
        selected = range(6) if layer == 'all' else [layer]
        def attention_hook(module, args, result):
            delta, weights = result
            changed = delta.clone()
            if kind == 'zero_attention':
                changed[batch, last] = 0
            else:
                x, positions = args[:2]
                B, L, _ = x.shape
                qkv = module.qkv(x).view(B, L, 3, module.n_heads, module.head_dim)
                q, k, v = qkv.unbind(2)
                q = apply_rope(q.transpose(1, 2), positions, module.rope_base)
                k = apply_rope(k.transpose(1, 2), positions, module.rope_base)
                v = v.transpose(1, 2)
                q = q[batch, :, last, :].unsqueeze(2)
                keep = torch.arange(L, device=x.device)[None, :] <= last[:, None]
                banned = {'drop_target': 2, 'drop_separator': 3, 'drop_source': 1}.get(kind)
                if banned is not None:
                    keep[:, banned] = False
                recomputed = F.scaled_dot_product_attention(q, k, v, attn_mask=keep[:, None, None, :])
                projection = module.proj(recomputed.transpose(1, 2).reshape(B, 1, -1))[:, 0]
                changed[batch, last] = projection
            return changed, weights
        def ffn_hook(module, args, result):
            changed = result.clone()
            changed[batch, last] = 0
            return changed
        if kind != 'baseline':
            for l in selected:
                block = model.blocks[l]
                handles.append(block.ffn.register_forward_hook(ffn_hook) if kind == 'zero_ffn'
                               else block.attn.register_forward_hook(attention_hook))
        try:
            logits = model(tokens)[batch, last].cpu().numpy()
        finally:
            for h in handles:
                h.remove()
        if kind == 'sham':
            native = model(tokens)[batch, last].cpu().numpy()
            max_sham_difference = max(max_sham_difference, float(abs(native - logits).max()))
        output.append(logits)
    return np.concatenate(output), max_sham_difference


@torch.inference_mode()
def capture_pair_states(model, rows, tok):
    captured, handles = {}, []
    def before(layer):
        def hook(module, args):
            captured['input', layer] = args[0]
        return hook
    def after(layer):
        def hook(module, args, output):
            captured['attention', layer] = captured['input', layer] + output[0]
        return hook
    for l, block in enumerate(model.blocks):
        handles.extend([block.register_forward_pre_hook(before(l)), block.attn.register_forward_hook(after(l))])
    results, logits = [], []
    try:
        for start in range(0, len(rows), 128):
            part = rows[start:start + 128]
            tokens, last = pad([r['prefix'] for r in part], tok)
            b = torch.arange(len(part), device=tokens.device)
            cap = {}
            out = model(tokens, capture=cap)
            stages = [model.embed(tokens)]
            for l in range(6):
                stages.extend([captured['attention', l], cap['resid'][l]])
            # Store only three relevant positions, in RAM.
            x = torch.stack(stages)
            positions = torch.stack([torch.full_like(last, 2), torch.full_like(last, 3), last,
                                     torch.full_like(last, 1)], dim=1)
            results.append(x[:, b[:, None], positions].cpu().numpy().transpose(1, 0, 2, 3))
            logits.append(out[b, last].cpu().numpy())
    finally:
        for handle in handles:
            handle.remove()
    return np.concatenate(results), np.concatenate(logits)


@torch.inference_mode()
def patch_forward(model, recipient, donors, tok, stage, role):
    result = []
    for start in range(0, len(recipient), 128):
        part = recipient[start:start + 128]
        tokens, last = pad([r['prefix'] for r in part], tok)
        b = torch.arange(len(part), device=tokens.device)
        position = torch.full_like(last, {'target': 2, 'separator': 3, 'source': 1}.get(role, 0))
        if role == 'current':
            position = last
        role_index = {'target': 0, 'separator': 1, 'current': 2, 'source': 3}[role]
        value = torch.tensor(donors[start:start + len(part), stage, role_index], device=tokens.device)
        handles, block_input = [], {}
        def embed_hook(module, args, output):
            changed = output.clone()
            changed[b, position] = value
            return changed
        def before(module, args):
            block_input['x'] = args[0]
        def attention_hook(module, args, output):
            delta, weights = output
            changed = delta.clone()
            changed[b, position] = value - block_input['x'][b, position]
            return changed, weights
        def block_hook(module, args, output):
            x, weights = output
            changed = x.clone()
            changed[b, position] = value
            return changed, weights
        if stage == 0:
            handles.append(model.embed.register_forward_hook(embed_hook))
        else:
            layer = (stage - 1) // 2
            block = model.blocks[layer]
            if stage % 2:
                handles.extend([block.register_forward_pre_hook(before), block.attn.register_forward_hook(attention_hook)])
            else:
                handles.append(block.register_forward_hook(block_hook))
        try:
            result.append(model(tokens)[b, last].cpu().numpy())
        finally:
            for handle in handles:
                handle.remove()
    return np.concatenate(result)


@torch.inference_mode()
def run(seed):
    world = json.loads((SUITE / 'data/n80_g0.json').read_text())
    folder = ROOT / f'outputs/fixed_navigation_v5/pilot/n80/g0_s{seed}/path'
    path = folder / 'latest.pt'
    blob = torch.load(path, map_location='cpu', weights_only=False)
    assert blob['step'] == 50000 and blob['world_hash'] == digest(world)
    tok = Tokenizer(256)
    nodes = [tok.node_id(i) for i in world['node_names']]
    model = TransformerLM(len(tok), **blob['job']['model'], dropout=0.).to('cuda:0').eval().requires_grad_(False)
    model.load_state_dict(blob['model'])
    del blob
    reach = np.eye(world['n_nodes'], dtype=bool)
    for u in range(len(nodes)):
        reach[u, list(bfs(world['adj'], u))] = True
    rows = collect(model, world, tok, nodes, reach)
    base_logits, _ = ablation_forward(model, rows, tok, ('baseline', 0))
    baseline = metrics(base_logits, rows, nodes, world['adj'], reach)
    result = {'seed': seed, 'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'scope': 'One next decision at native nonlocal mixed legal prefixes; not closed-loop rollout accuracy',
              'baseline': baseline, 'ablations': {}, 'patches': {}}
    native_next = np.array([r['native_next'] for r in rows])
    result['FP32_vs_BF16_next_token_mismatch'] = float((base_logits.argmax(1) != native_next).mean())
    first = np.array([r['first'] for r in rows])
    conditions = [(kind, l) for kind in ('zero_attention', 'zero_ffn', 'drop_target', 'drop_separator', 'drop_source', 'sham')
                  for l in list(range(6)) + ['all']]
    for kind, layer in conditions:
        logits, error = ablation_forward(model, rows, tok, (kind, layer))
        entry = {'all': metrics(logits, rows, nodes, world['adj'], reach),
                 'first': metrics(logits[first], [r for r in rows if r['first']], nodes, world['adj'], reach)}
        if kind == 'sham':
            entry['max_logit_error_vs_native'] = error
            assert error < 1e-3
        result['ablations'][f'{kind}_{layer}'] = entry
    print(f'seed{seed} component/attention-edge ablations complete', flush=True)
    bins = defaultdict(list)
    for i, r in enumerate(rows):
        key = (tuple(r['prefix'][:2] + [-1] + r['prefix'][3:]), r['target_layer'])
        bins[key].append(i)
    pairs = []
    for ids in bins.values():
        for j, a in enumerate(ids):
            for b in ids[j + 1:]:
                if not set(rows[a]['useful']) & set(rows[b]['useful']):
                    pairs.append((a, b))
    if pairs:
        # No choice/accuracy-based pair selection. Each donor is a natural
        # query with the same source, prefix, depth and disjoint useful set.
        rec_indices = [i for pair in pairs for i in pair]
        donor_indices = [i for a, b in pairs for i in (b, a)]
        recipient = [rows[i] for i in rec_indices]
        donor_rows = [rows[i] for i in donor_indices]
        states, pair_logits = capture_pair_states(model, recipient, tok)
        donor_states = states[np.arange(len(rec_indices)) ^ 1]
        donor_logits = pair_logits[np.arange(len(rec_indices)) ^ 1]
        inverse = {token: u for u, token in enumerate(nodes)}
        def score_patch(logits):
            chosen = [inverse.get(int(t), -1) for t in logits.argmax(1)]
            return {'recipient_useful': float(np.mean([v in r['useful'] for v, r in zip(chosen, recipient)])),
                    'donor_useful': float(np.mean([v in r['useful'] for v, r in zip(chosen, donor_rows)])),
                    'legal': float(np.mean([v in world['adj'][r['current']] for v, r in zip(chosen, recipient)]))}
        result['pair_panel'] = {'unordered_pairs': len(pairs), 'directed_cases': len(recipient),
            'unique_queries': len({r['qid'] for r in recipient}), 'unique_current_nodes': len({r['current'] for r in recipient}),
            'baseline': score_patch(pair_logits), 'donor_baseline': score_patch(donor_logits),
            'examples': [{k: rows[i][k] for k in ('qid', 'source', 'target', 'current', 'prefix', 'useful')} for i in rec_indices[:4]]}
        result['pair_balanced_ablations'] = {}
        for kind, layer in [('zero_attention', 'all'), ('zero_ffn', 0), ('drop_target', 'all'),
                            ('drop_separator', 'all'), ('drop_source', 'all')]:
            ablated, _ = ablation_forward(model, recipient, tok, (kind, layer))
            result['pair_balanced_ablations'][f'{kind}_{layer}'] = score_patch(ablated)
            if kind == 'zero_attention':
                error = float(abs(ablated[::2] - ablated[1::2]).max())
                result['pair_balanced_ablations'][f'{kind}_{layer}']['same_current_different_target_logit_error'] = error
                assert error < 1e-3
        for role in ('target', 'separator', 'current', 'source'):
            stages = range(13) if role == 'current' else [0] + [2*i for i in range(1, 7)]
            curve = []
            for stage in stages:
                logits = patch_forward(model, recipient, donor_states, tok, stage, role)
                curve.append({'stage': stage, **score_patch(logits)})
                if role == 'target' and stage == 0 or role == 'current' and stage == 12:
                    assert float(abs(logits - donor_logits).max()) < 1e-3
                if role == 'current' and stage == 0 or role in ('target', 'separator') and stage == 12 or role == 'source':
                    assert float(abs(logits - pair_logits).max()) < 1e-3
            result['patches'][role] = curve
    HERE.mkdir(parents=True, exist_ok=True)
    atomic_json(HERE / f'seed{seed}.json', result)
    print(f'seed{seed} COMPLETE; pairs={len(pairs)}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3])
    for seed in p.parse_args().seeds:
        run(seed)

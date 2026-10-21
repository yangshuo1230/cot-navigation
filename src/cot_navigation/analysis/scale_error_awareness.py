"""Last-consumed-token error probes on validation-selected scale checkpoints.

Read h[layer,t-1] before x[t] exists. Reuse frozen-model probe optimizers from
error_awareness; keep old studies and source files unchanged. Physical GPU0.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from cot_navigation.config import load_config
from cot_navigation.data.dataset import ContextStore, ParametricWorld
from cot_navigation.data.scale_prepared import ScalePreparedRows
from cot_navigation.evaluation.rollout import generate
from cot_navigation.models.transformer import build_model
from cot_navigation.protocols.tokenizer import Tokenizer
from .common import AnalysisModel, Example, sha256, write_json
from .error_detection import semantic_slots as state_slots
from .error_awareness import train_probe, predict, evaluate, dataset_summary, FIELDS, SEED

SCALES = [('l6', 36), ('l8', 64), ('l10', 100), ('curriculum', 100)]
PROBE_FIELDS = FIELDS+['node_expansion']


def node_slots(ex, protocol, body):
    """Exact first-expansion prefix language for arbitrary neighbor priorities.

    BFS queue blocks encode all admissible sibling permutations without
    enumerating orders. DFS backtracks only after unvisited neighbors exhaust.
    Unlike the old BFS distance-tier oracle, this enforces parent queue order.
    """
    neighbors = {ex.nodes[u]: set(ex.nodes[v] for v in vs) for u, vs in enumerate(ex.adj)}
    source, target = ex.nodes[ex.source], ex.nodes[ex.target]
    seen, stack, blocks, terminal = set(), [], deque([{source}]), False
    for at, token in enumerate(body):
        if terminal:
            allowed = set()
        elif protocol == 'bfs_nodes':
            allowed = set(blocks[0]) if blocks else set()
        elif not stack and not seen:
            allowed = {source}
        else:
            while stack and not neighbors[stack[-1]]-seen:
                stack.pop()
            allowed = neighbors[stack[-1]]-seen if stack else set()
        yield at, allowed, 'node_expansion'
        if token not in allowed:
            return
        if protocol == 'bfs_nodes':
            blocks[0].remove(token)
            if not blocks[0]:
                blocks.popleft()
            children = neighbors[token]-seen-{token}
            # Discovery occurs at enqueue time, including unexpanded siblings.
            if children:
                blocks.append(children)
            seen.add(token); seen.update(children)
        else:
            stack.append(token); seen.add(token)
        terminal = token == target


def slots(ex, protocol, tok, body):
    if protocol in {'dfs_nodes', 'bfs_nodes'}:
        yield from node_slots(ex, protocol, body)
    else:
        yield from state_slots(ex, protocol, tok, body)


def body_tokens(ex, tok, tokens):
    excluded = {tok.yes, tok.no, tok.id('<EOS>')}
    if '<STOP>' in tok.stoi:
        excluded.add(tok.id('<STOP>'))
    end = next((i for i in range(ex.row.prompt_len, len(tokens)) if tokens[i] in excluded), len(tokens))
    return list(tokens[ex.row.prompt_len:end])


class ScaleModel(AnalysisModel):
    """Read the actual offline variants and prepend the correct WORLD token."""
    def __init__(self, entry, root, device='cuda:0'):
        self.entry, self.root, self.run = entry, root, root/entry['run']
        self.cfg = load_config([self.run/'resolved_config.yaml'])
        self.tok = Tokenizer.from_config(self.cfg)
        self.protocol, self.filler = self.cfg.protocol.name, 0
        self.device = torch.device(device)
        self.model = build_model(self.cfg.model, len(self.tok), self.tok.pad).to(device).eval().requires_grad_(False)
        ckpt = self.run/'checkpoints/best.pt'
        blob = torch.load(ckpt, map_location='cpu', weights_only=False)
        self.model.load_state_dict(blob['model'], strict=True)
        self.checkpoint = dict(sha256=sha256(ckpt), step=int(blob['step']), best_score=float(blob['best_score']))
        del blob
        self.layers = list(range(self.model.n_layers-3, self.model.n_layers))
        self.context = self.cfg.data.knowledge == 'context_graph'
        data_root = root/self.cfg.data.root
        self.store = (ContextStore(data_root) if self.context else
                      ParametricWorld(data_root/f'{self.cfg.data.graph_shape}-seed{self.cfg.data.graph_seed}'))
        self.prepared = ScalePreparedRows(self.store, self.cfg, self.tok)
        self.world_prefix = []
        if not self.context and self.cfg.train.world_roots:
            roots = [(root/p).resolve() for p in self.cfg.train.world_roots]
            index = roots.index(self.store.root.resolve())
            self.world_prefix = [self.tok.id(f'<WORLD_{index}>')]

    def example(self, index, split):
        a = self.store.arrays if self.context else self.store.queries
        adj = self.store.adj(index) if self.context else self.store.adj()
        nodes = self.store.node_vocab_ids(index, self.tok) if self.context else self.store.node_vocab_ids(self.tok)
        # Diverse but deterministic actual training variants; query split is
        # unchanged and the natural prompt is independent of the variant.
        variant = index % self.cfg.protocol.trajectory_variants
        row = self.prepared.row(index, variant)
        n = len(self.world_prefix)
        if n:
            row = replace(row, tokens=self.world_prefix+row.tokens,
                          prompt_len=row.prompt_len+n, answer_index=row.answer_index+n,
                          trace_indices=[p+n for p in row.trace_indices],
                          walk_indices=[p+n for p in row.walk_indices])
        return Example(int(index), split, row.label, adj, nodes,
                       int(a['s_topo'][index]), int(a['t_topo'][index]), row)


def make_event(ex, body, at, good, bad, field, kind, source, rng):
    legal = int(rng.choice(sorted(good)))
    prefix = list(ex.prompt)+body[:at]
    return dict(query=ex.index, position=len(prefix), field=field, kind=kind,
                source=source, correct=prefix+[legal], wrong=prefix+[int(bad)],
                correct_token=legal, wrong_token=int(bad))


class IndexedCorruptions:
    """Shuffle the same candidate order without allocating every bad tuple.

    Each row stores a slot and its possible replacements. NumPy shuffles the
    virtual flat indices with the same RNG draws as shuffling the expanded
    Python list. Only actually sampled tuples are materialized.
    """
    def __init__(self, rows, rng):
        self.rows = rows
        self.ends = np.cumsum([len(row[2]) for row in rows])
        self.order = np.arange(int(self.ends[-1]))
        rng.shuffle(self.order)
        self.remaining = len(self.order)

    def __len__(self):
        return self.remaining

    def pop(self):
        self.remaining -= 1
        index = int(self.order[self.remaining])
        slot = int(np.searchsorted(self.ends,index,side='right'))
        at, good, options, field, kind = self.rows[slot]
        begin = int(self.ends[slot-1]) if slot else 0
        return at,good,options[index-begin],field,kind


def synthetic_events(examples, protocol, tok, count, seed):
    rng = np.random.default_rng(seed)
    quota = min(64, max(2, math.ceil(count/max(1, len(examples)))))
    result = []
    for ex in examples:
        body = body_tokens(ex, tok, ex.row.tokens)
        groups = defaultdict(list)
        for at, good, field in slots(ex, protocol, tok, body):
            if body[at] not in good:
                raise ValueError(f'Invalid prepared reference {ex.index}:{at}:{field}')
            options = sorted(set(ex.nodes)-good)
            if options:
                kind = 'in_graph_node' if body[at] in ex.nodes else 'excess_node'
                groups[kind, field].append((at, good, options, field, kind))
            if field in {'frontier','visited','candidates'} and tok.id('|') not in good:
                groups['premature_delimiter',field].append((at,good,[tok.id('|')],field,'premature_delimiter'))
            if field == 'action':
                for name in ['PUSH','POP','FOUND','EXHAUST']:
                    bad = tok.id(f'<{name}>')
                    if bad not in good:
                        groups['wrong_action',field].append((at,good,[bad],field,'wrong_action'))
        keys = list(groups); rng.shuffle(keys)
        for key in keys:
            groups[key] = IndexedCorruptions(groups[key],rng)
        picked, seen = 0, set()
        while keys and picked < quota:
            for key in list(keys):
                if picked >= quota:
                    break
                if not groups[key]:
                    keys.remove(key); continue
                at, good, bad, field, kind = groups[key].pop()
                event = make_event(ex,body,at,good,bad,field,kind,'synthetic',rng)
                signature = (at,bad,event['correct_token'])
                if signature in seen:
                    continue
                seen.add(signature); result.append(event); picked += 1
    rng.shuffle(result)
    return result[:count]


def natural_events(am, examples, batch_size=16):
    """Original STOP policy; true first violation, never later cascade errors."""
    rng, found, inventory = np.random.default_rng(SEED+8), [], Counter()
    for start in range(0, len(examples), batch_size):
        chunk = examples[start:start+batch_size]
        generated = generate(am.model,[e.prompt for e in chunk],am.protocol,am.tok,
                             am.cfg.train.precision,max_new_tokens=0,cot_margin=am.cfg.protocol.cot_margin)
        for ex, gen in zip(chunk,generated):
            inventory['queries'] += 1
            inventory['answer_correct'] += int(not gen['format_fail'] and gen['pred']==ex.label)
            inventory['stops'] += int(gen['cot_interrupted'])
            inventory['forced_stops'] += int(gen['forced_stop'])
            body = body_tokens(ex,am.tok,gen['tokens'])
            for at, good, field in slots(ex,am.protocol,am.tok,body):
                if body[at] in good:
                    continue
                inventory['first_errors'] += 1
                if not good:
                    inventory['unrepairable'] += 1
                    break
                bad = body[at]
                kind = 'in_graph_node' if bad in ex.nodes and all(v in ex.nodes for v in good) else 'format_or_other'
                found.append(make_event(ex,body,at,good,bad,field,kind,'natural',rng))
                break
        print('ROLLOUT',am.entry['id'],min(start+batch_size,len(examples)),len(examples),flush=True)
    return found, dict(inventory)


def extract(am, examples, event_list, batch_size=64, token_budget=16384):
    if not event_list:
        return None
    lookup = {ex.index:ex for ex in examples}
    rows = [(i,label,event,event[key]) for i,event in enumerate(event_list)
            for label,key in [(0,'correct'),(1,'wrong')]]
    rows.sort(key=lambda r:len(r[3]))
    output = {k:[] for k in ['x','y','query','event','kind','field']}
    begin = 0
    while begin < len(rows):
        end = begin+1
        while end<len(rows) and end-begin<batch_size and (end-begin+1)*len(rows[end][3])<=token_budget:
            end += 1
        chunk = rows[begin:end]
        _, residual, _ = am.forward([r[3] for r in chunk])
        for b,(event_id,label,event,seq) in enumerate(chunk):
            assert len(seq)==event['position']+1
            ex, pos = lookup[event['query']], len(seq)-1
            embedding = am.model.embed.weight[seq[-1]].detach().float()
            obs = embedding.new_zeros(4+len(PROBE_FIELDS))
            obs[:4] = obs.new_tensor([pos/4096,(pos-ex.row.prompt_len)/4096,
                                      float(seq[-1] in ex.nodes),float(seq[-1]==ex.nodes[ex.target])])
            obs[4+PROBE_FIELDS.index(event['field'])] = 1
            deep = torch.stack([residual[l][b,pos].float() for l in am.layers])
            x = torch.cat([F.pad(deep,(0,len(obs))),torch.cat([embedding,obs])[None]],0).cpu().numpy()
            item = dict(x=x,y=label,query=ex.index,event=event_id,kind=event['kind'],field=event['field'])
            for key in output:
                output[key].append(item[key])
        del residual
        begin = end
    return {k:np.asarray(v,dtype=np.float32 if k in {'x','y'} else None) for k,v in output.items()}


def select_models(root, output):
    base = root/'outputs/scale_v1/launches/scale_v1_s0'
    models, candidates = [], []
    for kind in ['context','fixed']:
        for scale,nodes in SCALES:
            for family in ['path','dfs','bfs']:
                options = [family] if family=='path' else [family+'_state',family+'_nodes']
                group = []
                for protocol in options:
                    run = base/kind/scale
                    if kind=='fixed':
                        run = run/'stage_b'
                    run = run/protocol
                    blob = torch.load(run/'checkpoints/best.pt',map_location='cpu',weights_only=False)
                    cfg = load_config([run/'resolved_config.yaml'])
                    group.append(dict(run=str(run.relative_to(root)),protocol=protocol,family=family,
                                      kind=cfg.data.knowledge,seed=0 if kind=='fixed' else None,
                                      scale=scale,n_nodes=nodes,score=float(blob['best_score']),
                                      step=int(blob['step']),sha256=sha256(run/'checkpoints/best.pt'),
                                      selection_metric=cfg.evaluation.selection_metric))
                    del blob
                # Stable tie break: full-state first, never consult test metrics.
                winner = max(group,key=lambda e:e['score']).copy()
                winner['id'] = f'{kind}_{scale}_{family}'
                models.append(winner); candidates.extend(group)
    write_json(output/'selection.json',dict(models=models,candidates=candidates,
               rule='max stored validation best_score within task/scale/family; state first on ties'))
    return models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('research/scale_error_awareness_20261019/results'))
    parser.add_argument('--train-pairs',type=int,default=8192)
    parser.add_argument('--val-pairs',type=int,default=2048)
    parser.add_argument('--test-pairs',type=int,default=4096)
    parser.add_argument('--train-queries',type=int,default=4096)
    parser.add_argument('--val-queries',type=int,default=1024)
    parser.add_argument('--test-queries',type=int,default=2048)
    parser.add_argument('--natural-queries',type=int,default=256)
    parser.add_argument('--steps',type=int,default=1000)
    parser.add_argument('--hidden',type=int,default=64)
    parser.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    parser.add_argument('--forward-batch',type=int,default=64)
    parser.add_argument('--forward-tokens',type=int,default=16384)
    parser.add_argument('--probe-batch',type=int,default=1024)
    parser.add_argument('--models',nargs='*')
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['WANDB_MODE']='disabled'
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    root, out = Path.cwd(), args.output
    out.mkdir(parents=True,exist_ok=True)
    settings = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in {'resume','models'}}
    settings.update(physical_gpu=0,readout='last consumed token h[layer,t-1], before sampling x[t]',
                    supervision='synthetic only; natural first errors are held-out test',
                    stop='preserve original cot_margin; STOP excluded from classification',
                    source_sha256={name:sha256(Path(__file__).parent/name) for name in
                                   ['scale_error_awareness.py','error_awareness.py','error_detection.py','common.py']})
    if (out/'settings.json').exists() and json.loads((out/'settings.json').read_text())!=settings:
        raise ValueError('Settings/source changed; use a new output')
    write_json(out/'settings.json',settings)
    entries = select_models(root,out)
    for entry in entries:
        if args.models and entry['id'] not in args.models:
            continue
        path = out/'models'/f'{entry["id"]}.json'
        if args.resume and path.exists():
            print('SKIP',entry['id'],flush=True);continue
        start = time.time();print('MODEL',entry['id'],entry['protocol'],flush=True)
        am = ScaleModel(entry,root)
        assert am.checkpoint['sha256']==entry['sha256']
        examples = {s:am.examples(s,getattr(args,s+'_queries')) for s in ['train','val','test']}
        ids = {s:{e.index for e in rows} for s,rows in examples.items()}
        assert not ids['train']&ids['val'] and not ids['train']&ids['test'] and not ids['val']&ids['test']
        data, summary = {}, {}
        for i,split in enumerate(['train','val','test']):
            ev = synthetic_events(examples[split],am.protocol,am.tok,getattr(args,split+'_pairs'),SEED+i)
            data[split] = extract(am,examples[split],ev,args.forward_batch,args.forward_tokens)
            summary[split] = dataset_summary(data[split],ev)
            print('DATA',split,summary[split],flush=True)
        natural_examples = am.examples('test',args.natural_queries)
        ev, inventory = natural_events(am,natural_examples)
        natural = extract(am,natural_examples,ev,args.forward_batch,args.forward_tokens)
        summary['natural'] = dataset_summary(natural,ev)
        ex = examples['train'][0];seq = ex.row.tokens[:ex.row.prompt_len+4]
        _,a,_ = am.forward([seq]);_,b,_ = am.forward([seq+[am.tok.yes]])
        delta = max(float((a[l][0,-1]-b[l][0,len(seq)-1]).abs().max()) for l in am.layers)
        del a,b
        result = dict(entry=entry,checkpoint=am.checkpoint,datasets=summary,
                      query_ids={s:sorted(v) for s,v in ids.items()},
                      heads=[f'L{l+1}' for l in am.layers]+['token_control'],
                      causal_mask_max_abs_delta=delta,natural_rollout=inventory,fits=[],interesting_cases=[])
        for arch,hidden in [('linear',0),('mlp',args.hidden)]:
            for seed in args.seeds:
                fitted = train_probe(data['train'],data['val'],hidden,SEED+seed,args.steps,args.probe_batch,am.device)
                syn_logits,nat_logits = predict(fitted,data['test']),predict(fitted,natural)
                fit = dict(architecture=arch,seed=seed,hidden=hidden,training=fitted[3],
                           synthetic=evaluate(data['test'],syn_logits),natural=evaluate(natural,nat_logits))
                result['fits'].append(fit)
                print('FIT',entry['id'],arch,seed,'synthetic',fit['synthetic'][2]['auroc'],
                      'natural',None if fit['natural'] is None else fit['natural'][2]['auroc'],flush=True)
                if arch=='mlp' and seed==args.seeds[0] and natural is not None:
                    wrong = np.flatnonzero(natural['y']==1)
                    for row in wrong[np.argsort(nat_logits[wrong,2])[:3]]:
                        event = ev[int(natural['event'][row])]
                        result['interesting_cases'].append(dict(query=event['query'],field=event['field'],
                            kind=event['kind'],wrong_token=am.tok.name(event['wrong_token']),
                            legal_repair=am.tok.name(event['correct_token']),error_logit=float(nat_logits[row,2]),
                            prefix_tail=am.tok.decode(event['wrong'][-40:])))
                del fitted
        result.update(seconds=time.time()-start,checkpoint_unchanged_after=sha256(am.run/'checkpoints/best.pt')==entry['sha256'])
        write_json(path,result);print('DONE',entry['id'],result['seconds'],flush=True)
        del am,data,natural,examples
        torch.cuda.empty_cache()
    print('COMPLETE',flush=True)


if __name__=='__main__':
    main()

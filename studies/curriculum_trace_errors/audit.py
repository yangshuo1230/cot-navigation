"""Frozen curriculum checkpoint audit; stratify TEST queries by gap and label.

Run from the project root. No optimizer, prepared-data writes or W&B sessions.
Local errors after the first global error are conditional on printed state;
they are not evidence that the original search has recovered.
"""
from pathlib import Path
from collections import Counter, deque
from types import SimpleNamespace
import argparse
import gzip
import hashlib
import json
import os
import re
import sys
import time

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'studies/scale_trace_audit')]
import numpy as np
import torch
from cot_navigation.config import load_config
from cot_navigation.data.dataset import ContextStore, ParametricWorld
from cot_navigation.models.transformer import build_model
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.protocols.render import context_prompt, parametric_prompt, render_steps
from cot_navigation.protocols.dfs import dfs_search
from cot_navigation.protocols.bfs import bfs_search, first_expansion_order
from cot_navigation.evaluation.rollout import generate
from cot_navigation.analysis.legal_history import state_audit
from legacy_trace_audit import validate_execution, self_test, validate_nodes

OUT = ROOT/'studies/curriculum_trace_errors'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(p)

def sequence_audit(body, adj, src, dst, label, protocol):
    """Chronological first error. BFS accepts arbitrary legal sibling order.

    Groups in FIFO order avoid looking ahead at an already invalid suffix.
    After an order error only graph-independent violations are counted.
    """
    events = []
    seen = set()
    stack = [src]
    groups = deque([{src}])
    discovered = {src}
    ordered = True
    goal_seen = False
    exhausted = False
    def add(i, kind):
        events.append({'unit': i, 'kind': kind})
    for i, node in enumerate(body):
        basic = []
        if node not in adj: basic.append('non_graph_node')
        if i == 0 and node != src: basic.append('wrong_start')
        if node in seen: basic.append('repeated_node')
        if goal_seen: basic.append('continued_after_target')
        if protocol == 'path' and i and node not in adj.get(body[i-1], set()):
            basic.append('non_edge')
        for kind in basic: add(i, kind)
        if basic: ordered = False
        if ordered and protocol == 'bfs_nodes':
            while groups and not groups[0]: groups.popleft()
            if not groups or node not in groups[0]:
                add(i, 'not_bfs_order'); ordered = False
            else:
                groups[0].remove(node)
                if node != dst:
                    eligible = adj[node]-discovered
                    if eligible: groups.append(set(eligible))
                    discovered.update(eligible)
        elif ordered and protocol == 'dfs_nodes' and i:
            while stack and not adj[stack[-1]]-seen: stack.pop()
            if not stack or node not in adj[stack[-1]]:
                add(i, 'not_dfs_order'); ordered = False
            else: stack.append(node)
        seen.add(node)
        goal_seen |= node == dst
    if ordered and body:
        if protocol == 'bfs_nodes': exhausted = not any(groups)
        elif protocol == 'dfs_nodes':
            while stack and not adj[stack[-1]]-seen: stack.pop()
            exhausted = not stack
        else: exhausted = not adj.get(body[-1], set())
    prefix_valid = not events
    complete = prefix_valid and bool(body) and (
        (label == 1 and body[-1] == dst) or (label == 0 and exhausted))
    if not complete and not events:
        add(len(body), 'empty_trace' if not body else
            'wrong_positive_endpoint' if label else 'unfinished_search')
    return dict(complete_valid=complete, prefix_valid=prefix_valid,
                units=len(body), valid_prefix_units=events[0]['unit'] if events else len(body),
                first_error=events[0] if events else None, events=events,
                action_valid=complete, order_checks_after_first_error=False)

def full_state_audit(ex, body, tok, fifo):
    """Grounded first error plus all independently checkable local errors."""
    strict = state_audit(ex, body, ex.label, tok, fifo)
    adj = {ex.nodes[u]:{ex.nodes[v] for v in vs} for u,vs in enumerate(ex.adj)}
    target = ex.nodes[ex.target]
    events = []
    previous = None
    start = 0
    units = 0
    action_names = {tok.id('<'+a+'>'):a for a in ('PUSH','POP','FOUND','EXHAUST')}
    for end, x in enumerate(body):
        if x != tok.id(';'): continue
        raw = body[start:end]
        begin = start
        start = end+1
        fields = [[]]
        for v in raw:
            if v == tok.id('|'): fields.append([])
            else: fields[-1].append(v)
        kinds = []
        details = {}
        if len(fields) != 4:
            kinds.append('malformed_fields'); previous = None
        else:
            ff, vv, cc, aa = fields
            if not ff or not set(ff+vv+cc) <= set(adj):
                kinds.append('invalid_state_nodes')
            else:
                cur = ff[0 if fifo else -1]
                eligible = adj[cur]-set(vv)
                if len(vv) != len(set(vv)): kinds.append('visited_duplicate')
                if not set(ff) <= set(vv): kinds.append('frontier_missing_from_visited')
                if len(ff) != len(set(ff)): kinds.append('frontier_duplicate')
                if len(cc) != len(set(cc)): kinds.append('candidate_duplicate')
                if set(cc)-adj[cur]: kinds.append('candidate_nonedge')
                if set(cc)&set(vv): kinds.append('candidate_already_visited')
                if eligible-set(cc): kinds.append('candidate_omitted')
                if cur == target and set(cc) != eligible: kinds.append('goal_candidate_annotation')
                expected = 'FOUND' if cur == target else 'PUSH' if eligible else 'POP' if len(ff)>1 else 'EXHAUST'
                name = action_names.get(aa[0]) if aa else None
                if name is None or len(aa) != (2 if name == 'PUSH' else 1):
                    kinds.append('malformed_action')
                else:
                    if name != expected:
                        kinds.append('missed_goal' if cur == target else
                                     'false_found' if name == 'FOUND' else
                                     'premature_exhaust' if name == 'EXHAUST' else
                                     'pop_with_candidates' if name == 'POP' and eligible else 'wrong_action')
                    if name == 'PUSH':
                        if aa[1] not in adj[cur]: kinds.append('nonedge_push')
                        if aa[1] in vv: kinds.append('revisited_push')
                        if aa[1] not in cc: kinds.append('push_not_displayed')
                details = {'current':cur,'target':target,'expected_candidates':sorted(eligible),
                           'shown_candidates':cc,'expected_action':expected,'shown_action':aa}
            if previous:
                pf, pv, pa = previous
                expected_f = list(pf)
                expected_v = set(pv)
                known = False
                if len(pa)==2 and pa[0]==tok.id('<PUSH>'):
                    expected_f.append(pa[1]); expected_v.add(pa[1]); known=True
                elif pa==[tok.id('<POP>')] and expected_f:
                    expected_f.pop(0 if fifo else -1); known=True
                elif pa in ([tok.id('<FOUND>')],[tok.id('<EXHAUST>')]):
                    kinds.append('record_after_terminal')
                if known:
                    if ff != expected_f: kinds.append('frontier_update_error')
                    if set(vv) != expected_v: kinds.append('visited_update_error')
            previous = ff,vv,aa
        if kinds:
            events.append({'unit':units,'token_start':begin,'token_end':end+1,
                           'kinds':sorted(set(kinds)),'detail':details})
        units += 1
    if start != len(body):
        events.append({'unit':units,'token_start':start,'token_end':len(body),'kinds':['unfinished_record']})
    first = strict['first_error']
    if first:
        first = {'unit':first['record'],'kind':first['kind'],'detail':first['detail']}
    elif not strict['complete_valid']:
        first = {'unit':units,'kind':'unfinished_search'}
    return dict(complete_valid=strict['complete_valid'], prefix_valid=strict['prefix_valid'],
                units=units, valid_prefix_units=strict['strict_valid_records'], first_error=first,
                events=events, local=strict['local'], action_valid=validate_execution(
                    body,ex.label,adj,ex.nodes[ex.source],target,tok,fifo))

def test_validators(tok):
    import itertools
    checked = self_test(tok,dfs_search,bfs_search,render_steps,first_expansion_order)
    nodes = [tok.node_id(i) for i in range(4)]
    pairs = list(itertools.combinations(range(4),2))
    equivalence = 0
    for mask in range(64):
        raw = [[] for _ in nodes]
        for k,(u,v) in enumerate(pairs):
            if mask>>k&1: raw[u].append(v)
        adj = {nodes[u]:{nodes[v] for v in vs} for u,vs in enumerate(raw)}
        for fifo,search in ((False,dfs_search),(True,bfs_search)):
            for s,t in itertools.permutations(range(4),2):
                steps = search(raw,s,t)
                label = int(steps[-1].action=='FOUND')
                ex = SimpleNamespace(nodes=nodes,adj=raw,source=s,target=t,label=label)
                assert full_state_audit(ex,render_steps(steps,nodes,tok),tok,fifo)['complete_valid']
                seq = [nodes[v] for v in first_expansion_order(steps)]
                assert sequence_audit(seq,adj,nodes[s],nodes[t],label,'bfs_nodes' if fifo else 'dfs_nodes')['complete_valid']
                for length in range(1,5):
                    for rest in itertools.permutations([n for n in nodes if n!=nodes[s]],length-1):
                        trial = [nodes[s],*rest]
                        old = validate_nodes(trial,label,adj,nodes[s],nodes[t],tok,fifo)['valid']
                        new = sequence_audit(trial,adj,nodes[s],nodes[t],label,'bfs_nodes' if fifo else 'dfs_nodes')['complete_valid']
                        assert old==new,(raw,s,t,fifo,trial,old,new)
                        equivalence += 1
    raw = [[1,2],[3],[],[]]
    ex = SimpleNamespace(nodes=nodes,adj=raw,source=0,target=3,label=1)
    trace = render_steps(dfs_search(raw,0,3),nodes,tok)
    broken = list(trace)
    broken[broken.index(tok.id('<FOUND>'))] = tok.id('<POP>')
    r = full_state_audit(ex,broken,tok,False)
    assert not r['complete_valid'] and any('missed_goal' in e['kinds'] for e in r['events'])
    return {'legacy_reference_assertions':checked,'node_language_equivalence':equivalence,'mutated_goal_detected':True}

def jobs():
    entries = json.loads((ROOT/'studies/scale_mechanistic/results/selection.json').read_text())['models']
    return [e for e in entries if e['scale']=='curriculum' and e['protocol']!='direct']

def select_ids(source, context, per_cell, seed):
    a = source.arrays if context else source.queries
    ids = source.indices('test')
    rng = np.random.default_rng(seed)
    selected = []
    cells = []
    for gap,label in sorted({(int(a['distance'][i]),int(a['label'][i])) for i in ids}):
        pool = np.asarray([int(i) for i in ids if int(a['distance'][i])==gap and int(a['label'][i])==label])
        take = sorted(rng.choice(pool,min(per_cell,len(pool)),replace=False).tolist())
        selected += take
        cells.append({'gap':gap,'label':label,'available':len(pool),'selected':len(take)})
    return selected,cells

def run(entry, per_cell, batch, include_unseen=False):
    folder = ROOT/entry['run']
    cfg = load_config([folder/'resolved_config.yaml'])
    tok = Tokenizer.from_config(cfg)
    checkpoint = folder/'checkpoints'/(entry.get('checkpoint_name','best')+'.pt')
    before = sha(checkpoint)
    assert before == entry['checkpoint_sha256']
    model = build_model(cfg.model,len(tok),tok.pad).cuda().eval().requires_grad_(False)
    blob = torch.load(checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(blob['model'],strict=True)
    step = int(blob['step']); del blob
    context = entry['kind']=='context'
    roots = cfg.train.scale_roots if context else cfg.train.world_roots
    if context:
        cases = [(f'n{n}',p) for n,p in zip([16,24,32,36,64,100],roots)]
    else:
        cases = []
        for p in roots:
            if not include_unseen and Path(p).parent not in {Path(r) for r in cfg.train.scale_roots}:
                continue
            layers,width,n = map(int,re.search(r'fixed_l(\d+)w(\d+)_n(\d+)',str(p)).groups())
            cases.append((f'n{n}_l{layers}w{width}',p))
    if context:
        cases += [tuple(x.split('=',1)) for x in cfg.evaluation.extra_sets if x.split('=',1)[0] in {'deep','wide','larger'}]
    dest = OUT/'models'/entry['id']
    dest.mkdir(parents=True,exist_ok=True)
    for case,root in cases:
        result = dest/(case+'.json')
        if result.exists(): continue
        started = time.time()
        source = ContextStore(root) if context else ParametricWorld(root)
        a = source.arrays if context else source.queries
        chosen,cells = select_ids(source,context,per_cell,20261021)
        write(OUT/'queries'/entry['kind']/(case+'.json'),{'root':root,'split':'test','per_gap_label':per_cell,'ids':chosen,'cells':cells})
        world = [] if context else [tok.id(f'<WORLD_{[Path(x).resolve() for x in roots].index(Path(root).resolve())}>')]
        rows = []
        examples = []
        example_types = Counter()
        for at in range(0,len(chosen),batch):
            prepared = []
            for idx in chosen[at:at+batch]:
                adj = source.adj(idx) if context else source.adj()
                nodes = source.node_vocab_ids(idx,tok) if context else source.node_vocab_ids(tok)
                s,t = int(a['s_topo'][idx]),int(a['t_topo'][idx])
                distances = {s:0}; queue = deque([s])
                while queue:
                    u = queue.popleft()
                    for v in adj[u]:
                        if v not in distances: distances[v]=distances[u]+1; queue.append(v)
                assert int(t in distances)==int(a['label'][idx])
                ex = SimpleNamespace(nodes=nodes,adj=adj,source=s,target=t,label=int(a['label'][idx]))
                if context:
                    edges = list(zip(a['edge_src'][idx].astype(int).tolist(),a['edge_dst'][idx].astype(int).tolist()))
                    prompt = context_prompt(tok,nodes,edges,s,t)
                else: prompt = world+parametric_prompt(tok,nodes,s,t)
                prepared.append((idx,ex,prompt,distances.get(t)))
            outputs = generate(model,[x[2] for x in prepared],entry['protocol'],tok,cfg.train.precision,
                               max_new_tokens=cfg.evaluation.max_new_tokens,cot_margin=cfg.protocol.cot_margin)
            for (idx,ex,prompt,shortest),out in zip(prepared,outputs):
                body = list(out['trace_tokens'])
                stop = tok.id('<STOP>') in body
                if stop: body = body[:body.index(tok.id('<STOP>'))]
                adjtok = {ex.nodes[u]:{ex.nodes[v] for v in vs} for u,vs in enumerate(ex.adj)}
                if entry['protocol'].endswith('_state'): r = full_state_audit(ex,body,tok,entry['protocol']=='bfs_state')
                else: r = sequence_audit(body,adjtok,ex.nodes[ex.source],ex.nodes[ex.target],ex.label,entry['protocol'])
                if stop and r['prefix_valid'] and not r['complete_valid']:
                    r['first_error'] = {'unit':r['units'],'kind':'legal_prefix_stopped'}
                bad_units = {e['unit'] for e in r['events']}
                first = r['first_error']
                item = dict(query=idx,label=ex.label,pred=out['pred'],correct=out['pred']==ex.label,
                            n_nodes=len(ex.nodes),gap=int(a['distance'][idx]),shortest_path=shortest,
                            stop=stop,forced_stop=bool(out['forced_stop']),format_fail=bool(out['format_fail']),
                            trace_token_count=len(body),**r)
                item['bad_unit_count'] = len(bad_units)
                item['bad_unit_fraction'] = len(bad_units)/max(1,r['units'])
                item['first_error_fraction'] = min(1,first['unit']/max(1,r['units'])) if first else None
                item['post_first_bad_fraction'] = (sum(u>first['unit'] for u in bad_units)/max(1,r['units']-first['unit']-1)) if first else None
                item['failure_shape'] = ('valid_complete' if r['complete_valid'] else 'legal_prefix_stopped' if stop and r['prefix_valid'] else
                    'termination_only' if not bad_units or all(e.get('kind') in {'empty_trace','wrong_positive_endpoint','unfinished_search'} for e in r['events']) else
                    'one_bad_unit' if len(bad_units)==1 else 'two_or_three_bad_units' if len(bad_units)<=3 else 'four_or_more_bad_units')
                rows.append(item)
                keys = set(k for e in r['events'] for k in e.get('kinds',[e.get('kind')]))
                if first: keys.add(first['kind'])
                novelty = sorted(k for k in keys if k and example_types[k]<1)
                if novelty and len(examples)<12:
                    focus = first['unit'] if first else 0
                    if entry['protocol'].endswith('_state'):
                        boundaries = [-1]+[i for i,v in enumerate(body) if v==tok.id(';')]
                        lo = boundaries[max(0,min(focus,len(boundaries)-1))-1]+1 if focus else 0
                        hi = boundaries[min(len(boundaries)-1,focus+2)]+1
                    else: lo,hi=max(0,focus-5),min(len(body),focus+8)
                    examples.append({'query':idx,'label':ex.label,'pred':out['pred'],'source':tok.name(ex.nodes[ex.source]),
                        'target':tok.name(ex.nodes[ex.target]),'gap':item['gap'],'first_error':first,
                        'failure_shape':item['failure_shape'],'types':novelty,'window':tok.decode(body[lo:hi][:240]),
                        'window_token_start':lo,'events':r['events'][:4]})
                    example_types.update(novelty)
            print(json.dumps({'model':entry['id'],'case':case,'done':min(at+batch,len(chosen)),'total':len(chosen)}),flush=True)
        with gzip.open(dest/(case+'.jsonl.gz'),'wt',encoding='utf-8') as f:
            for row in rows: f.write(json.dumps(row)+'\n')
        assert sha(checkpoint)==before
        write(result,{'model':entry,'case':case,'dataset':str(root),'checkpoint_sha256':before,'step':step,
                      'n':len(rows),'sampling_cells':cells,'elapsed_seconds':time.time()-started,'examples':examples})
        del source
    del model
    torch.cuda.empty_cache()

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',type=int,default=0)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--per-cell',type=int,default=8)
    parser.add_argument('--batch',type=int,default=64)
    parser.add_argument('--test-only',action='store_true')
    parser.add_argument('--include-unseen',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    if args.test_only:
        cfg=load_config([ROOT/jobs()[0]['run']/'resolved_config.yaml'])
        result=test_validators(Tokenizer.from_config(cfg))
        write(OUT/'validator_tests.json',result);print(result)
    else:
        for entry in jobs()[args.worker::args.workers]: run(entry,args.per_cell,args.batch,args.include_unseen)

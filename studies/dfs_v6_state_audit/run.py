"""Audit frozen v6 simple Fixed DFS State: diversity and first-error replay.

Only compact aggregates and a few reproducible failures are saved. Greedy native
generation is deterministic; diversity across queries is not sampling randomness.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['WANDB_MODE'] = 'disabled'
import sys, json, hashlib, math, time
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
V6 = ROOT / 'studies/fixed_navigation_protocols_v6'
sys.path[:0] = [str(V6), str(ROOT / 'src')]
from packed_bank import PackedBank
from semantics import replay_state
from cot_navigation.models.transformer import TransformerLM
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.evaluation.rollout import generate

OUT = Path(__file__).resolve().parent
PANELS = ['short_test', 'long_AA_test', 'long_TA_test', 'long_AT_test', 'long_TT_test']


def digest(seq):
    return hashlib.sha256(np.asarray(seq, dtype=np.uint16).tobytes()).hexdigest()


def audit(adj, source, target, body, tok, inverse):
    """Ground-truth state replay; record only opportunities before first error.

    Node IDs in this routine are topology indices. Saved example labels are
    converted to actual vocabulary IDs by the caller.
    """
    stack, history, seen = [source], [source], {source}
    visited_current = set()
    last_candidates = {}
    pos = 0
    events, pushes = [], [source]
    terminal = None
    first = None
    bar, semi = tok.id('|'), tok.id(';')
    actions = {tok.id('<' + a + '>'): a for a in ['PUSH', 'POP', 'FOUND', 'EXHAUST']}

    def field():
        nonlocal pos
        values = []
        while pos < len(body) and body[pos] != bar:
            values.append(inverse.get(body[pos], -body[pos]-1))
            pos += 1
        closed = pos < len(body)
        if closed:
            pos += 1
        return values, closed

    def fail(kind, **detail):
        nonlocal first
        first = dict(kind=kind, record=len(events), current=stack[-1],
                     stack=stack.copy(), visited=history.copy(), **detail)

    while pos < len(body) and first is None:
        if terminal:
            fail('tokens_after_terminal')
            break
        actual, closed = field()
        if not closed or actual != stack:
            fail('stack_mismatch' if closed else 'incomplete_stack')
            break
        actual, closed = field()
        if not closed or actual != history:
            fail('visited_mismatch' if closed else 'incomplete_visited')
            break
        cur = stack[-1]
        expected = set(adj[cur]) - seen
        candidates, closed = field()
        phase = 'root' if not events else 'fresh_pushed' if cur not in visited_current else 'revisit'
        event = dict(current=cur, phase=phase, nonempty=bool(expected),
                     record=len(events)+1, visited_n=len(seen),
                     blocked=len(set(adj[cur]) & seen), bad=False,
                     expected=sorted(expected), shown=candidates, at_goal=cur == target)
        events.append(event)
        visited_current.add(cur)
        flags = dict(missing=bool(expected-set(candidates)),
                     extra_non_neighbor=any(v not in adj[cur] for v in candidates if v >= 0),
                     extra_visited_neighbor=any(v in seen and v in adj[cur] for v in candidates if v >= 0),
                     duplicate=len(candidates) != len(set(candidates)),
                     outside_graph=any(v < 0 for v in candidates), unclosed=not closed)
        if any(flags.values()):
            event['bad'] = True
            previous = last_candidates.get(cur)
            missing = expected-set(candidates)
            fail('candidate_error', expected=sorted(expected), shown=candidates,
                 flags=flags, phase=phase, blocked=event['blocked'], at_goal=cur==target,
                 previous_candidates=previous,
                 stale_previous_list=previous is not None and candidates==previous,
                 full_unfiltered_adjacency=bool(set(candidates)==set(adj[cur]) and set(candidates)!=expected),
                 missing_target=target in missing,
                 missing_all_eligible=bool(expected and not set(candidates)&expected))
            break
        last_candidates[cur] = candidates.copy()
        if pos >= len(body) or body[pos] not in actions:
            fail('bad_action')
            break
        action = actions[body[pos]]
        pos += 1
        arg = None
        if action == 'PUSH':
            if pos >= len(body) or body[pos] not in inverse:
                fail('missing_push_node')
                break
            arg = inverse[body[pos]]
            pos += 1
        if pos >= len(body) or body[pos] != semi:
            fail('missing_record_end')
            break
        pos += 1
        required = 'FOUND' if cur == target else 'PUSH' if expected else 'POP' if len(stack)>1 else 'EXHAUST'
        if action != required:
            fail('illegal_action', shown_action=action, expected_action=required)
            break
        if action == 'PUSH':
            if not candidates or arg != candidates[0]:
                fail('push_not_first_candidate')
                break
            stack.append(arg)
            seen.add(arg)
            history.append(arg)
            pushes.append(arg)
        elif action == 'POP':
            stack.pop()
        else:
            terminal = action
    if first is None and terminal is None:
        fail('missing_terminal')
    return dict(valid=first is None, first=first, events=events, pushes=pushes, terminal=terminal)


def entropy(counts, k):
    total = sum(counts.values())
    return -sum(v/total * math.log(v/total) for v in counts.values())/math.log(k) if total and k > 1 else None


def rate(events):
    return dict(n=len(events), bad=sum(e['bad'] for e in events),
                rate=sum(e['bad'] for e in events)/len(events) if events else None)


def diversity(bank, counts, tok, inverse):
    bodies, orders = set(), set()
    per_query = defaultdict(list)
    exposed = 0
    training_ids = set(bank.world['panels']['short_train'] + bank.world['panels']['long_train'])
    push = tok.id('<PUSH>')
    for row, key in enumerate(bank.keys):
        if not key.startswith('q:') or counts.get(key, 0) <= 0:
            continue
        qid = int(bank.cpu['qids'][row])
        assert qid in training_ids
        start = int(bank.cpu['offsets'][row]) + 4
        end = int(bank.cpu['offsets'][row]) + int(bank.cpu['answers'][row])
        body = bank.cpu['tokens'][start:end].tolist()
        source = bank.world['records'][qid]['s']
        order = tuple([source] + [inverse[body[i+1]] for i, t in enumerate(body[:-1]) if t == push])
        bodies.add(digest(body))
        orders.add(order)
        # The first record has only root stack/VIS; read its candidate field.
        bars = [i for i, t in enumerate(body) if t == tok.id('|')][:3]
        cands = tuple(inverse[t] for t in body[bars[1]+1:bars[2]])
        per_query[qid].append((order, cands))
        exposed += 1
    branches = [v for v in per_query.values() if len(v[0][1]) > 1]
    output = dict(actual_exposed_variants=exposed, queries=len(per_query),
                  variants_per_query=dict(Counter(len(v) for v in per_query.values())),
                  unique_body_count=len(bodies), unique_node_orders=len(orders),
                  mean_unique_node_orders_per_query=float(np.mean([len({x[0] for x in v}) for v in per_query.values()])),
                  fraction_queries_multiple_node_orders=float(np.mean([len({x[0] for x in v})>1 for v in per_query.values()])),
                  root_branch_queries=len(branches),
                  root_branch_multiple_first_fraction=float(np.mean([len({x[1][0] for x in v})>1 for v in branches])),
                  root_branch_mean_normalized_entropy=float(np.mean([entropy(Counter(x[1][0] for x in v),len(v[0][1])) for v in branches])),
                  root_branch_mean_distinct_permutations=float(np.mean([len({x[1] for x in v}) for v in branches])))
    return output, bodies, orders


def self_test(tok):
    adj = [[1,2], [3], [3], [], []]
    inverse = {tok.node_id(i): i for i in range(5)}
    from semantics import reference
    def encode(rows):
        result = []
        for st, vi, ca, ac, ar in rows:
            for f in [st,vi,ca]:
                result.extend(tok.node_id(x) for x in f)
                result.append(tok.id('|'))
            result.append(tok.id('<'+ac+'>'))
            if ar is not None:
                result.append(tok.node_id(ar))
            result.append(tok.id(';'))
        return result
    rows = reference(adj,0,4)
    a = audit(adj,0,4,encode(rows),tok,inverse)
    assert a['valid'] and len(a['events']) == 7
    assert any(e['phase']=='fresh_pushed' and e['blocked'] for e in a['events'])
    altered = [(s[:],v[:],c[:],ac,ar) for s,v,c,ac,ar in rows]
    altered[0][2].append(1)
    b = audit(adj,0,4,encode(altered),tok,inverse)
    assert b['first']['flags']['duplicate'] and not b['valid']
    altered = [(s[:],v[:],c[:],ac,ar) for s,v,c,ac,ar in rows]
    altered[0][2].append(4)
    assert audit(adj,0,4,encode(altered),tok,inverse)['first']['flags']['extra_non_neighbor']


def main():
    torch.set_num_threads(2)
    tok = Tokenizer(256)
    self_test(tok)
    meta = next(w for w in json.loads((V6/'manifest.json').read_text())['worlds'] if w['name']=='n80_g0')
    bank = PackedBank(meta,'dfs_state',torch.device('cpu'))
    world = bank.world
    nodes = [tok.node_id(i) for i in world['node_names']]
    inverse = {v:i for i,v in enumerate(nodes)}
    result_path = OUT/'results.json'
    previous = json.loads(result_path.read_text()) if result_path.exists() else {}
    doc = previous if previous.get('version')==2 else dict(version=2,seeds=[])
    for seed in range(4):
        if any(s['seed']==seed for s in doc['seeds']):
            continue
        run = ROOT/f'outputs/fixed_navigation_protocols_v6/pilot/n80/g0_s{seed}/dfs_state'
        counts = json.loads((run/'exposure_counts.json').read_text())
        train, bodies, orders = diversity(bank,counts,tok,inverse)
        print('TRAIN',seed,train,flush=True)
        blob = torch.load(run/'latest.pt',map_location='cpu',weights_only=False)
        assert blob['step']==50000
        raw = TransformerLM(len(tok),**blob['job']['model'],dropout=0.).to('cuda:0')
        raw.load_state_dict(blob['model'])
        raw.eval().requires_grad_(False)
        model = torch.compile(raw,dynamic=True)
        result = dict(seed=seed,checkpoint_sha256=hashlib.sha256((run/'latest.pt').read_bytes()).hexdigest(),
                      train=train,panels={},examples=[],first_failures=[])
        original = json.loads((run/'final.json').read_text())
        root_choices = defaultdict(Counter)
        root_perms = defaultdict(Counter)
        negative_root_choices = defaultdict(Counter)
        test_orders = defaultdict(set)
        test_orders_by_source = defaultdict(set)
        for panel in PANELS:
            ids = world['panels'][panel]
            checks, errors, flags, events, matches = Counter(),Counter(),Counter(),[],Counter()
            patterns, action_errors = Counter(), Counter()
            for at in range(0,len(ids),64):
                qids = ids[at:at+64]
                prompts = [[tok.id('<QUERY>'),nodes[world['records'][q]['s']],nodes[world['records'][q]['t']],tok.id('|')] for q in qids]
                started = time.monotonic()
                outputs = generate(model,prompts,'dfs_state',tok,'bf16',max_new_tokens=bank.info['max_new_tokens'])
                for qid,out in zip(qids,outputs):
                    r = world['records'][qid]
                    body = out['trace_tokens']
                    ar = audit(world['adj'],r['s'],r['t'],body,tok,inverse)
                    baseline = replay_state(world['adj'],r['s'],r['t'],body,tok,inverse)
                    assert ar['valid']==baseline['valid'], (qid,ar['first'],baseline)
                    legal = ar['valid'] and not out['generation_limit_reached']
                    checks['n']+=1
                    checks['trace_valid']+=legal
                    checks['truncated']+=out['generation_limit_reached']
                    checks['correct']+=out['answer_token']==tok.label_id(r['y'])
                    checks[f"correct{r['y']}"]+=out['answer_token']==tok.label_id(r['y'])
                    errors[(ar['first'] or {}).get('kind','valid')]+=1
                    events.extend(ar['events'])
                    first = ar['first']
                    if first:
                        result['first_failures'].append(dict(panel=panel,query=qid,label=r['y'],target=r['t'],
                            answer_correct=out['answer_token']==tok.label_id(r['y']),**first))
                    if first and first['kind']=='illegal_action':
                        action_errors[first['expected_action']+'->'+first['shown_action']]+=1
                    if first and first['kind']=='candidate_error':
                        flags.update(k for k,v in first['flags'].items() if v)
                        patterns['at_goal' if first['at_goal'] else 'not_at_goal']+=1
                        for k in ['stale_previous_list','full_unfiltered_adjacency','missing_target','missing_all_eligible']:
                            patterns[k]+=first[k]
                        matches['candidate_first_failure']+=1
                        matches['candidate_answer_correct']+=out['answer_token']==tok.label_id(r['y'])
                        for kind,value in first['flags'].items():
                            if value and sum(e['category']==kind for e in result['examples'])<2:
                                ex = dict(category=kind,panel=panel,query=qid,label=r['y'],
                                          source=r['s'],target=r['t'],answer_correct=out['answer_token']==tok.label_id(r['y']),**first)
                                result['examples'].append(ex)
                    body_match = digest(body) in bodies
                    matches['body_exact']+=body_match
                    # Extract full printed PUSH order even after a semantic error.
                    push = tok.id('<PUSH>')
                    seq = tuple([r['s']]+[inverse.get(body[i+1],-1) for i,t in enumerate(body[:-1]) if t==push])
                    matches['push_order_exact']+=seq in orders
                    matches['legal_push_order_exact']+=legal and seq in orders
                    matches['legal_body_exact']+=legal and body_match
                    if legal:
                        test_orders[(r['s'],r['y'])].add(seq)
                        test_orders_by_source[r['s']].add(seq)
                    if ar['events'] and ar['events'][0]['phase']=='root':
                        root = ar['events'][0]
                        if not root['bad'] and len(root['expected'])>1 and not root['at_goal']:
                            root_choices[r['s']][root['shown'][0]]+=1
                            root_perms[r['s']][tuple(root['shown'])]+=1
                            if r['y']==0:
                                negative_root_choices[r['s']][root['shown'][0]]+=1
                print('BATCH',seed,panel,at,len(qids),'seconds',round(time.monotonic()-started,1),flush=True)
            old = original['panels'][panel]['metrics']['all']
            diffs = {k:dict(new=v,old=old.get(k,0)) for k,v in checks.items() if v!=old.get(k,0)}
            result['panels'][panel] = dict(checks=checks,errors=errors,candidate_flags=flags,matches=matches,
                candidate_patterns=patterns,action_errors=action_errors,
                candidate_phase={p:rate([e for e in events if e['phase']==p]) for p in ['root','fresh_pushed','revisit']},
                candidate_phase_nonempty={p:rate([e for e in events if e['phase']==p and e['nonempty']]) for p in ['root','fresh_pushed','revisit']},
                candidate_error_records=dict(Counter(str(e['record']) for e in events if e['bad'])),
                original_count_differences=diffs)
        roots = [s for s,c in root_choices.items() if sum(c.values())>=4]
        result['greedy_root_diversity'] = dict(sources_with_at_least4_queries=len(roots),
            fraction_sources_multiple_first=float(np.mean([len(root_choices[s])>1 for s in roots])),
            mean_normalized_entropy=float(np.mean([entropy(root_choices[s],len(world['adj'][s])) for s in roots])),
            mean_dominant_choice_fraction=float(np.mean([max(root_choices[s].values())/sum(root_choices[s].values()) for s in roots])),
            mean_distinct_candidate_permutations=float(np.mean([len(root_perms[s]) for s in roots])),
            per_source=[dict(source=s,counts=dict(root_choices[s]),orders={','.join(map(str,k)):v for k,v in root_perms[s].items()}) for s in roots])
        negative_roots=[s for s,c in negative_root_choices.items() if sum(c.values())>=4]
        result['greedy_negative_root_diversity']=dict(sources=len(negative_roots),
            fraction_sources_multiple_first=float(np.mean([len(negative_root_choices[s])>1 for s in negative_roots])),
            mean_normalized_entropy=float(np.mean([entropy(negative_root_choices[s],len(world['adj'][s])) for s in negative_roots])))
        neg = [v for (s,y),v in test_orders.items() if y==0]
        result['legal_negative_search_orders'] = dict(sources=len(neg),mean_distinct_orders=float(np.mean([len(v) for v in neg])))
        doc['seeds'].append(result)
        doc['node_vocabulary_names']=world['node_names']
        doc['verification']='Toy DAG cases PASS; every generated trace validity agrees with independent original replay_state.'
        result_path.write_text(json.dumps(doc,ensure_ascii=False,indent=2))
        del model,raw,blob
        torch.cuda.empty_cache()
    print('DONE',flush=True)


if __name__=='__main__':
    main()

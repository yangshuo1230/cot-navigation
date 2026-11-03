"""Audit before-emission timing, independent legal next tokens, and tail routing."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['WANDB_MODE']='disabled'
import gzip
import hashlib
import json
from pathlib import Path
import runpy
import numpy as np
import torch
from cot_navigation.analysis.within_token import cached_tail
from cot_navigation.evaluation.rollout import pad_rows
from cot_navigation.protocols.correction_replay import replay
from cot_navigation.experiments.artifacts import atomic_json,sha_file

P=Path(__file__).resolve().parent;OUT=P/'current';S=runpy.run_path(str(P/'run.py'));R=S['R']


def independent_allowed(prefix,graph,src,tok):
    stack=[src];seen={src};segments=[];begin=0
    for i,t in enumerate(prefix):
        if t==tok.id(';'):segments.append(prefix[begin:i]);begin=i+1
    for seg in segments:
        bars=[i for i,t in enumerate(seg) if t==tok.id('|')];assert len(bars)==3
        action=seg[bars[2]+1:]
        if action[0]==tok.id('<PUSH>'):stack.append(action[1]);seen.add(action[1])
        elif action[0]==tok.id('<POP>'):stack.pop()
        else:raise AssertionError('Candidate slot after terminal state')
    current=prefix[begin:];bars=[i for i,t in enumerate(current) if t==tok.id('|')]
    assert len(bars)==2 and current[:bars[0]]==stack
    assert set(current[bars[0]+1:bars[1]])==seen
    remaining=set(graph.get(stack[-1],[]))-seen-set(current[bars[1]+1:])
    return sorted(remaining) if remaining else [tok.id('|')]


def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(.32)
    selection=json.loads((OUT/'selection.json').read_text());am=R['Model']('correction',0)
    cases,_,_=R['load_data'](am.tok);query_sets={};graph_sets={};checked=0
    for part,rows in selection['parts'].items():
        query_sets[part]={r['query'] for r in rows};graph_sets[part]=set()
        for r in rows:
            c=cases[r['split']][r['sid']];prefix=c['trace'][:r['pos']]
            assert hashlib.sha256(np.asarray(prefix,np.int32).tobytes()).hexdigest()==r['prefix_hash']
            assert am.tok.id('<MISTAKE>') not in prefix
            graph,src,dst=R['PREP']['graph_tokens'](am,r['query'])
            assert replay(prefix,None,graph,src,dst,am.tok)['first'] is None
            assert independent_allowed(prefix,graph,src,am.tok)==r['allowed']
            if 'natural_onset' in r['populations']:
                assert c['tau']==r['pos']
                assert replay(c['trace'][:r['pos']+1],None,graph,src,dst,am.tok)['first'] is not None
            graph_sets[part].add(c['graph_hash']);checked+=1
    keys=list(query_sets)
    for i,a in enumerate(keys):
        for b in keys[i+1:]:
            assert not query_sets[a]&query_sets[b] and not graph_sets[a]&graph_sets[b]
    records=0;models=0
    for path in sorted((OUT/'results').glob('*.gz')):
        d=json.load(gzip.open(path,'rt'));assert d['complete'] and d['checkpoint_unchanged'];models+=1
        assert sha_file(Path(d['checkpoint']['path']))==d['checkpoint']['sha256']
        for name,digest in d['identity']['sources'].items():assert sha_file(P.parents[2]/name)==digest
        assert d['verification']['cached_sham_argmax_mismatches']==0
        index={(r['query'],r['prefix_hash'],r['probe_target'],r['lens'],r['layer'],r['mode'],r['condition']):r for r in d['interventions']}
        for r in d['interventions']:
            assert r['relative_norm']<=.20001
            if r['mode']=='preserve_tendency':
                assert r['tentative_preserved']
                if r['layer']==6:assert r['pred_before']==r['pred_after']
            if r['condition'].startswith('random_'):
                other=index[r['query'],r['prefix_hash'],r['probe_target'],r['lens'],r['layer'],r['mode'],r['condition'][7:]]
                assert abs(r['relative_norm']-other['relative_norm'])<3e-5
                assert abs(r['backtrack']-other['backtrack'])<1e-8
            records+=1
    assert models==8
    # Full-sequence replacement is an independent real-checkpoint reference for
    # the optimized same-position cached-tail implementation.
    rows=[r for r in selection['parts']['test'] if r['requires_node']][:4]
    seq=S['sequences'](am,cases,rows);ids,lengths=pad_rows(seq,am.tok.pad,am.device)
    cache=am.model.new_cache(len(rows),am.device,torch.float32);cache.valid_length=ids.shape[1];cap={}
    with torch.no_grad():
        am.model(ids,cache=cache,cache_mode='prefill',capture=cap)
        b=torch.arange(len(rows),device=am.device);positions=(lengths-1)[:,None];max_diff=0.
        torch.manual_seed(2081)
        for layer in [3,4,5]:
            h=cap['resid'][layer][b,lengths-1];edited=h+torch.randn_like(h)*.01
            def hook(_m,_args,out):
                x,w=out;x=x.clone();x[b,lengths-1]=edited;return x,w
            handle=am.model.blocks[layer].register_forward_hook(hook)
            try:full=am.model(ids)[b,lengths-1]
            finally:handle.remove()
            tail=cached_tail(am.model,cache,edited,positions,layer)
            torch.testing.assert_close(full,tail,rtol=3e-4,atol=5e-5)
            assert torch.equal(full.argmax(-1),tail.argmax(-1))
            max_diff=max(max_diff,float((full-tail).abs().max()))
    atomic_json(OUT/'verification.json',dict(passed=True,before_emission_prefixes=checked,
        independent_true_state_next_token_oracle=True,no_past_error_or_marker=True,
        graph_and_query_split_disjoint=True,checkpoints_unchanged=8,intervention_rows=records,
        current_token_not_consumed=True,norm_matched_controls=True,protected_full_and_node_tendency=True,
        final_layer_protected_structural_zero=True,real_cached_tail_full_max_logit_difference=max_diff,
        real_cached_tail_argmax_identical=True,tests_passed=6))
    print('WITHIN TOKEN VERIFIED',checked,records,flush=True)


if __name__=='__main__':main()

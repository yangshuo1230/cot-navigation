"""Full native CoT replay and history-consistent search-state pairs.

Uses validation-best scale checkpoints, the previous query IDs, and frozen
inference only. Main semantic replay stops at the first invalid transition;
separate local checks can inspect later printed records without treating them
as an authoritative search history. No trace/activation bulk caches are saved.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
import os
from pathlib import Path
import time


def state_audit(ex,body,pred,tok,fifo):
    """Order-invariant sets; distinguish algorithm errors from display priority."""
    nodes=set(ex.nodes);adj={ex.nodes[u]:{ex.nodes[v] for v in vs} for u,vs in enumerate(ex.adj)}
    frontier=[ex.nodes[ex.source]];seen=set(frontier);target=ex.nodes[ex.target]
    first=None;terminal=None;strict_records=0;start=0;local=Counter();shown_records=[]
    def issue(name,index,detail=None):
        nonlocal first
        if first is None:first={'kind':name,'record':index,'detail':detail or {}}
    for end,token in enumerate(body):
        if token!=tok.id(';'):continue
        raw=body[start:end];begin=start;start=end+1
        fields=[[]]
        for x in raw:
            if x==tok.id('|'):fields.append([])
            else:fields[-1].append(x)
        index=len(shown_records)
        if len(fields)!=4:
            local['malformed_records']+=1;issue('malformed_record',index);shown_records.append(None);continue
        ff,vv,cc,aa=fields;item={'frontier':ff,'visited':vv,'candidates':cc,'action':aa,'start':begin}
        shown_records.append(item);local['parseable_records']+=1
        if ff and set(ff+vv)<=nodes:
            cur=ff[0 if fifo else -1];eligible=adj[cur]-set(vv)
            if set(cc)-adj[cur]:local['candidate_nonedge_records']+=1
            if set(cc)&set(vv):local['candidate_visited_records']+=1
            if len(set(cc))!=len(cc):local['candidate_duplicate_records']+=1
            if set(cc)!=eligible or len(set(cc))!=len(cc):local['candidate_incorrect_records']+=1
            if len(aa)==2 and aa[0]==tok.id('<PUSH>'):
                local['push_records']+=1
                if aa[1] not in adj[cur]:local['nonedge_push_records']+=1
                if aa[1] in vv:local['visited_push_records']+=1
        if len(shown_records)>=2:
            prev=shown_records[-2]
            if prev and prev['frontier']:
                expected=list(prev['frontier']);vs=set(prev['visited']);a=prev['action'];known=False
                if len(a)==2 and a[0]==tok.id('<PUSH>'):expected.append(a[1]);vs.add(a[1]);known=True
                elif a==[tok.id('<POP>')]:expected.pop(0 if fifo else -1);known=True
                if known:
                    local['local_transition_checks']+=1
                    if ff!=expected:local['frontier_jump_records']+=1
                    if set(vv)!=vs:local['visited_update_error_records']+=1
        if first is not None:continue
        if terminal:issue('record_after_terminal',index);continue
        if ff!=frontier:issue('frontier_mismatch',index,{'expected':frontier,'shown':ff});continue
        if len(set(vv))!=len(vv) or set(vv)!=seen:issue('visited_mismatch',index);continue
        cur=frontier[0 if fifo else -1];eligible=adj[cur]-seen
        if set(cc)!=eligible or len(set(cc))!=len(cc):issue('candidate_mismatch',index,{'expected':sorted(eligible),'shown':cc});continue
        action='FOUND' if cur==target else 'PUSH' if eligible else 'POP' if len(frontier)>1 else 'EXHAUST'
        if not aa or aa[0]!=tok.id('<'+action+'>') or len(aa)!=(2 if action=='PUSH' else 1):issue('wrong_action',index);continue
        if action=='PUSH':
            if aa[1] not in eligible:issue('illegal_push',index);continue
            if cc and aa[1]!=cc[0]:local['push_not_first_displayed_candidate']+=1
            frontier.append(aa[1]);seen.add(aa[1])
        elif action=='POP':frontier.pop(0 if fifo else -1)
        else:terminal=action
        strict_records+=1
    if start!=len(body):issue('unfinished_record',len(shown_records))
    complete=first is None and ((terminal=='FOUND' and pred==1) or (terminal=='EXHAUST' and pred==0))
    return {'complete_valid':complete,'prefix_valid':first is None,'first_error':first,
            'strict_valid_records':strict_records,'printed_records':len(shown_records),'local':dict(local)}


def other_audit(ex,body,pred,protocol,tok):
    """Node sequences are expansion orders, NOT adjacent-node paths."""
    import sys
    sys.path.insert(0,str(Path.cwd()/'research/scale_trace_audit_20261019'))
    from legacy_trace_audit import validate_trace
    adj={ex.nodes[u]:[ex.nodes[v] for v in vs] for u,vs in enumerate(ex.adj)}
    result=validate_trace(body,pred,protocol,adj,ex.nodes[ex.source],ex.nodes[ex.target],tok,protocol=='path')
    return {'complete_valid':result['valid'],'first_error':None if result['valid'] else {'kind':result.get('error')},
            'prefix_valid':None,'local':{}}


def audit(am,examples):
    import numpy as np
    from .scale_followup import generate
    counts=Counter();local=Counter();errors=Counter();saved=[];rows=[]
    for at in range(0,len(examples),64):
        chunk=examples[at:at+64]
        for ex,out in zip(chunk,generate(am,[e.prompt for e in chunk])):
            body=out['trace_tokens'];stop=am.tok.id('<STOP>');stopped=stop in body
            if stopped:body=body[:body.index(stop)]
            if am.protocol.endswith('_state'):r=state_audit(ex,body,out['pred'],am.tok,am.protocol=='bfs_state')
            else:r=other_audit(ex,body,out['pred'],am.protocol,am.tok)
            correct=out['pred']==ex.label;complete=r['complete_valid'] and not stopped
            c=Counter(n=1,answer_correct=int(correct),complete_valid=int(complete),
                answer_correct_invalid_trace=int(correct and not complete),stop=int(stopped),forced_stop=int(out['forced_stop']),
                format_failure=int(out['pred'] is None),prefix_valid=int(r.get('prefix_valid') is True))
            for k in ['nonedge_push_records','visited_push_records','frontier_jump_records','visited_update_error_records']:
                c[k.replace('_records','_queries')]=int(r['local'].get(k,0)>0)
            counts.update(c);local.update(r['local'])
            if r['first_error']:errors[r['first_error']['kind']]+=1
            row={'query':ex.index,'label':ex.label,'pred':out['pred'],'correct':correct,'complete_valid':complete,
                 'stop':stopped,'forced_stop':out['forced_stop'],'first_error':r['first_error']['kind'] if r['first_error'] else None,
                 'local':r['local']}
            rows.append(row)
            if len(saved)<6 and (r['first_error'] or correct and not complete):
                saved.append({'query':ex.index,'source':am.tok.name(ex.nodes[ex.source]),'target':am.tok.name(ex.nodes[ex.target]),
                    'label':ex.label,'pred':out['pred'],'first_error':r['first_error'],
                    'prefix_preview':am.tok.decode(body[:220]),'stop':stopped,'forced_stop':out['forced_stop']})
    n=counts['n']
    return {'counts':dict(counts),'local_counts':dict(local),'first_errors':dict(errors),
            'accuracy':counts['answer_correct']/n,'complete_valid_rate':counts['complete_valid']/n,
            'correct_answer_invalid_trace_rate':counts['answer_correct_invalid_trace']/n,
            'examples':saved,'by_query':rows}


def find_pairs(am,ex,variants=64):
    """Prefer identical frontier, equal VIS cardinality and exact prefix length.

    BFS relaxed matches are explicitly marked: same current/queue length/VIS
    count. On strict layered DAGs an identical queue often fixes the discovered
    set; these pairs cannot isolate VIS from queue changes.
    """
    import numpy as np
    from cot_navigation.protocols.dfs import dfs_search
    from cot_navigation.protocols.bfs import bfs_search
    fifo=am.protocol=='bfs_state';search=bfs_search if fifo else dfs_search
    rng=np.random.default_rng(20261019+ex.index)
    exact=defaultdict(list);relaxed=defaultdict(list);equivalent=defaultdict(list)
    best=None;control=None
    def candidate(a,b,relax=False):
        ca,cb=set(a['step'].cands),set(b['step'].cands)
        if ca==cb:return None
        if not ca-cb:a,b=b,a;ca,cb=cb,ca
        # Prefer one changed candidate and a common stable candidate.
        rank=(int(relax),int(len(a['step'].visited)!=len(b['step'].visited)),
              int(a['length']!=b['length']),abs(a['length']-b['length']),len(ca^cb),-len(ca&cb))
        return rank,a,b,min(ca-cb)
    for variant in range(variants):
        adj=[rng.permutation(vs).astype(int).tolist() for vs in ex.adj]
        steps=search(adj,ex.source,ex.target);used=len(ex.prompt)
        for i,s in enumerate(steps):
            partial=used+len(s.stack)+len(s.visited)+2
            if partial+24>=am.model.max_seq_len:break
            if s.action not in {'FOUND','EXHAUST'}:
                state={'steps':steps,'index':i,'step':s,'length':partial,'variant':variant}
                frontier=tuple(s.stack)
                for old in exact[frontier]:
                    c=candidate(state,old)
                    if c and (best is None or c[0]<best[0]):best=c
                key=(s.cur,len(s.stack),len(s.visited))
                if fifo:
                    for old in relaxed[key]:
                        c=candidate(state,old,True)
                        if c and (best is None or c[0]<best[0]):best=c
                eq=(frontier,tuple(sorted(s.visited)))
                for old in equivalent[eq]:
                    if old['variant']!=variant and (old['length']!=partial or old['step'].visited!=s.visited):
                        score=(int(old['length']!=partial),abs(old['length']-partial))
                        if control is None or score<control[0]:control=(score,state,old)
                # Different orders of an identical state need not occupy memory repeatedly.
                if not any(set(x['step'].visited)==set(s.visited) for x in exact[frontier]):exact[frontier].append(state)
                if fifo and len(relaxed[key])<16:relaxed[key].append(state)
                if len(equivalent[eq])<4:equivalent[eq].append(state)
            used+=len(s.stack)+len(s.visited)+len(s.cands)+5+int(s.action=='PUSH')
    from cot_navigation.protocols.render import render_steps
    def serialize(state):
        s=state['step']
        prefix=ex.prompt+render_steps(state['steps'][:state['index']],ex.nodes,am.tok)
        for part in [s.stack,s.visited]:prefix += [ex.nodes[v] for v in part]+[am.tok.id('|')]
        assert len(prefix)==state['length']
        # Independent first-error oracle accepts any legal priority/set order.
        from .error_detection import semantic_slots
        body=prefix[len(ex.prompt):];slots=list(semantic_slots(ex,am.protocol,am.tok,body))
        assert len(slots)==len(body) and all(body[i] in good for i,good,_ in slots)
        return {'prefix':prefix,'frontier':[ex.nodes[v] for v in s.stack],
                'visited':[ex.nodes[v] for v in s.visited],'candidates':[ex.nodes[v] for v in s.cands],
                'record':state['index'],'variant':state['variant']}
    out={}
    if best:
        score,a,b,node=best
        out['changed']={'a':serialize(a),'b':serialize(b),'focus':ex.nodes[node],
            'match':'relaxed_queue' if score[0] else 'exact_frontier','equal_vis_count':score[1]==0,'equal_length':score[2]==0}
    if control:
        _,a,b=control;out['same_state']={'a':serialize(a),'b':serialize(b),'match':'exact_state','equal_length':a['length']==b['length']}
    return out


def pair_stage(am,examples,max_cases=40,variants=64):
    import numpy as np
    import torch
    from .scale_followup import generate
    from .common import records
    cases={'changed':[],'same_state':[]}
    for ex in examples:
        if all(len(v)>=max_cases for v in cases.values()):break
        found=find_pairs(am,ex,variants)
        for key,c in found.items():
            if len(cases[key])<max_cases:cases[key].append({'query':ex.index,'label':ex.label,'prompt_length':len(ex.prompt),**c})
    result={}
    for key,items in cases.items():
        if not items:result[key]={'n':0,'status':'no eligible legal history pairs'};continue
        prefixes=[c[arm]['prefix'] for c in items for arm in ['a','b']]
        outs=generate(am,prefixes,cap=32)
        scores=[]
        for start in range(0,len(prefixes),8):
            chunk=prefixes[start:start+8];logits,_,_=am.forward(chunk)
            for i,prefix in enumerate(chunk):scores.append(torch.softmax(logits[i,len(prefix)-1].float(),-1).cpu().numpy())
        observations=[];saved=[]
        for i,c in enumerate(items):
            row={'query':c['query'],'match':c['match'],'equal_length':c['equal_length']}
            for j,arm in enumerate(['a','b']):
                s=c[arm];out=outs[2*i+j]
                # Parse the completed record from its actual beginning, not from the cut.
                start=s['prefix'][:-1]
                semi=am.tok.id(';');prior=max((k for k,v in enumerate(start) if v==semi),default=c['prompt_length']-1)
                actual=records(out['tokens'],prior+1,am.tok)
                r=actual[0] if actual else None
                eligible=set(s['candidates']);allowed=eligible or {am.tok.id('|')};prob=scores[2*i+j]
                row[arm]={'next_legal_mass':float(prob[list(allowed)].sum()),'next_argmax_legal':int(prob.argmax()) in allowed,
                          'candidate_set_correct':r is not None and set(r['candidates'])==eligible and len(r['candidates'])==len(eligible),
                          'record_parseable':r is not None,
                          'focus_listed':c.get('focus') in r['candidates'] if r and 'focus' in c else None}
                if 'focus' in c:row[arm]['focus_probability']=float(prob[c['focus']])
                if r:
                    expected='PUSH' if eligible else 'POP' if len(s['frontier'])>1 else 'EXHAUST'
                    act=r['action'];row[arm]['action_legal']=bool(act and act[0]==am.tok.id('<'+expected+'>') and
                        ((len(act)==2 and act[1] in eligible) if expected=='PUSH' else len(act)==1))
                else:row[arm]['action_legal']=False
            observations.append(row)
            if len(saved)<4:
                saved.append({'query':c['query'],'match':c['match'],'equal_length':c['equal_length'],
                    'focus':am.tok.name(c['focus']) if 'focus' in c else None,
                    'arms':{arm:{'frontier':c[arm]['frontier'],'visited':c[arm]['visited'],'expected_candidates':c[arm]['candidates'],
                        'prefix':c[arm]['prefix'],'prefix_tail':am.tok.decode(c[arm]['prefix'][-160:]),
                        'continuation':am.tok.decode(outs[2*i+j]['generated']),'measurements':row[arm]} for j,arm in enumerate(['a','b'])}})
        summary={'n':len(items),'match_counts':dict(Counter(c['match'] for c in items)),
                 'equal_length_n':sum(c['equal_length'] for c in items),'examples':saved,'by_query':observations}
        for arm in ['a','b']:
            summary[arm]={metric:float(np.mean([r[arm][metric] for r in observations])) for metric in
                ['next_legal_mass','next_argmax_legal','candidate_set_correct','record_parseable','action_legal']}
        if key=='changed':
            delta=np.asarray([r['a']['focus_probability']-r['b']['focus_probability'] for r in observations])
            rng=np.random.default_rng(20261019);boot=delta[rng.integers(len(delta),size=(2000,len(delta)))].mean(-1)
            summary.update(focus_probability_drop=float(delta.mean()),drop_ci95=np.quantile(boot,[.025,.975]).tolist(),
                focus_listed_when_eligible=float(np.mean([r['a']['focus_listed'] is True for r in observations])),
                focus_listed_when_visited=float(np.mean([r['b']['focus_listed'] is True for r in observations])))
        result[key]=summary
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',type=int,choices=range(1,8),required=True)
    parser.add_argument('--models',nargs='+',required=True)
    parser.add_argument('--output',type=Path,default=Path('research/valid_history_20261019/results'))
    args=parser.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=f'0,{args.gpu}';os.environ['WANDB_MODE']='disabled'
    import torch
    from .scale_followup import make_adapter
    from .common import write_json,sha256
    torch.set_num_threads(3)
    previous=Path('research/scale_mechanistic_20261019/results')
    selection=json.loads((previous/'selection.json').read_text())
    for entry in selection['models']:
        name=entry['id']
        if name not in args.models:continue
        target=args.output/'models'/f'{name}.json'
        if target.exists() and json.loads(target.read_text()).get('complete'):continue
        print('MODEL',name,flush=True);started=time.time()
        old=json.loads((previous/'models'/f'{name}.json').read_text())
        am=make_adapter(entry,Path.cwd(),'cuda:1');assert am.checkpoint['sha256']==entry['checkpoint_sha256']
        examples=[am.example(q,'test') for q in old['query_ids']['test']]
        result={'entry':entry,'checkpoint':am.checkpoint,'source_sha256':sha256(Path(__file__)),'query_ids':[e.index for e in examples]}
        print(' AUDIT',flush=True);result['audit']=audit(am,examples);write_json(target,result)
        if am.protocol.endswith('_state'):
            print(' LEGAL PAIRS',flush=True);result['pairs']=pair_stage(am,examples)
        else:result['pairs']={'status':'no explicit search state'}
        result['checkpoint_unchanged']=sha256(am.run/'checkpoints/best.pt')==entry['checkpoint_sha256']
        assert result['checkpoint_unchanged'];result.update(complete=True,elapsed_seconds=time.time()-started)
        write_json(target,result);print('DONE',name,round(time.time()-started,1),flush=True)
        del am;torch.cuda.empty_cache()


if __name__=='__main__':main()

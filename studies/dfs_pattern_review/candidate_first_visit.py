"""Native DFS first-visit/revisit candidate hazard, only before first error.

Observe candidate checks only after frontier and visited pass grounded replay.
First visit to a node is distinct from the first record of a trajectory.
"""
import sys,json,gzip,hashlib,time
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import torch
import measure as m
sys.path.insert(0,str(m.ROOT/'studies/fixed100_legality'))
from replay import StateReplay
from stream import generate_checked
from cache_prefix import enable
m.SETTINGS['context_hard_n100']=(
    m.ROOT/'outputs/hard_dag_v1/launches/hard_dag_v1_s0/context/curriculum/dfs_state',
    m.ROOT/'data/hard_dag_v1/context_r10b10_n100_m200_train50000_s0')

class VisitReplay(StateReplay):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.expanded=set();self.events=[];self.fields=[[]];self.pending=None
    def candidate_fail(self,kind,body):
        if self.pending is not None:self.pending['bad']=True;self.pending['failure']=kind
        return self.fail(kind,dict(current=self.frontier[-1],
            expected=sorted(set(self.adj[self.frontier[-1]])-self.seen),
            shown=self.fields[2].copy() if len(self.fields)>2 else [],offending_token=body[-1] if body else None))
    def check(self,body):
        token=body[-1]
        if token==self.tok.id('|') and not self.terminal:
            if len(self.fields)==1:
                if self.fields[0]!=self.frontier:return self.fail('frontier_mismatch')
            elif len(self.fields)==2:
                visited=self.fields[1]
                if len(visited)!=len(set(visited)) or set(visited)!=self.seen:return self.fail('visited_mismatch')
                cur=self.frontier[-1];fresh=cur not in self.expanded
                assert fresh==(cur==self.history[-1])
                expected=set(self.adj[cur])-self.seen
                event=dict(record=self.records+1,fresh=fresh,root=self.records==0,current=cur,
                    visited_n=len(self.seen),stack_depth=len(self.frontier),prefix_tokens=self.start,
                    nonempty=bool(expected),at_goal=cur==self.target,bad=False,censored=False,
                    blocked_successors=len(set(self.adj[cur])&self.seen),out_degree=len(self.adj[cur]),
                    completed=False,failure=None,missing=0,extra=0,duplicate=False)
                self.events.append(event);self.pending=event;self.expanded.add(cur)
            elif len(self.fields)==3:
                cands=self.fields[2];expected=set(self.adj[self.frontier[-1]])-self.seen
                self.pending.update(completed=True,missing=len(expected-set(cands)),extra=len(set(cands)-expected),
                                    duplicate=len(cands)!=len(set(cands)))
                if set(cands)!=expected or len(cands)!=len(set(cands)):
                    return self.candidate_fail('candidate_mismatch',body)
                self.pending=None
            self.fields.append([])
        elif len(self.fields)==3 and not self.terminal:
            # A candidate field can fail before its closing separator.
            if token not in self.adj:
                kind='candidate_outside_graph' if self.tok.node_lo<=token<self.tok.node_lo+self.tok.n_node_tokens else 'candidate_invalid_token'
                return self.candidate_fail(kind,body)
            self.fields[2].append(token)
        elif token!=self.tok.id(';'):self.fields[-1].append(token)
        stopped=super().check(body)
        if token==self.tok.id(';') and not stopped:self.fields=[[]]
        return stopped
    def finish(self,body,pred,interrupted=False):
        if self.pending is not None and not self.error:
            if interrupted:self.pending['censored']=True
            else:self.candidate_fail('candidate_unclosed',body)
        return super().finish(body,pred,interrupted)

def summarize(events):
    if not events:return dict(n=0,bad=0,rate=None)
    bad=sum(e['bad'] for e in events)
    return dict(n=len(events),bad=bad,rate=bad/len(events))

def original_fixed():
    result={};old=m.ROOT/'studies/fixed100_legality'
    for setting in ('simple','hard'):
        rows=[json.loads(line) for line in gzip.open(old/f'{setting}_dfs_state_full_test_semantic.jsonl.gz','rt')]
        errors=[r['audit']['first_error'] for r in rows if (r['audit']['first_error'] or {}).get('kind')=='candidate_mismatch']
        fresh=[e for e in errors if e['detail']['current']==e['detail']['visited'][-1]]
        result[setting]=dict(total_queries=len(rows),candidate_errors=len(errors),fresh=len(fresh),revisit=len(errors)-len(fresh),
            root=sum(e['record']==0 for e in errors),fresh_pushed=sum(e['record']>0 for e in fresh),
            note='Exact full-pool FIRST-error classifications, not exposure-adjusted hazard.')
    return result

def run(name):
    folder,data=m.SETTINGS[name];cfg,tok,model,step=m.base.load_model(folder,torch.device('cuda'))
    model.requires_grad_(False);enable(model)
    context=name.startswith('context');source=m.ContextStore(data) if context else m.ParametricWorld(data)
    arrays=source.arrays if context else source.queries
    ids=source.indices('test')
    if not context:ids=np.array([i for i in ids if arrays['t_topo'][i]//10>arrays['s_topo'][i]//10])
    shapes=arrays['shape_id'] if context else np.zeros(len(arrays['label']),np.int8)
    ids=list(map(int,m.stratified_ids(ids,shapes,arrays['distance'],arrays['label'],128,20261027)))
    rows=[];all_events=[];start=time.time();examples=[]
    for offset in range(0,len(ids),64):
        chosen=ids[offset:offset+64];prompts=[];replays=[];node_names=[]
        for i in chosen:
            if context:
                item=m.base.prepare_context(source,tok,i);prompt=item['tokens'][:item['prompt_len']]
                nodes=source.node_vocab_ids(i,tok);adj=source.adj(i)
            else:
                row=source.render_query(i,'dfs_state',tok);header=[m.base.world_token(cfg,tok,data)] if cfg.train.world_roots else []
                prompt=header+row.tokens[:row.prompt_len];nodes=source.node_vocab_ids(tok);adj=source.adj()
            tokenadj={nodes[u]:set(nodes[v] for v in vs) for u,vs in enumerate(adj)}
            s,t=int(arrays['s_topo'][i]),int(arrays['t_topo'][i]);prompts.append(prompt)
            replays.append(VisitReplay(tokenadj,nodes[s],nodes[t],tok))
            node_names.append({v:f'V{j}' for j,v in enumerate(nodes)})
        generated=generate_checked(model,prompts,tok,cfg.train.precision,replays,cfg.protocol.cot_margin)
        for i,rep,out,names in zip(chosen,replays,generated,node_names):
            es=[dict(e,query_id=i) for e in rep.events if not e['censored']];all_events+=es
            failure=out['audit']['first_error'];candidate_failure=bool(failure and failure['kind'].startswith('candidate_'))
            first_phase=None
            if candidate_failure:
                assert es and es[-1]['bad']
                first_phase='root' if es[-1]['root'] else 'fresh_pushed' if es[-1]['fresh'] else 'revisit'
                if len(examples)<6:
                    detail=failure['detail'];lab=lambda xs:[names.get(v,tok.name(v)) for v in xs]
                    examples.append(dict(query_id=i,phase=first_phase,record=es[-1]['record'],
                        current=names[detail['current']],visited_n=es[-1]['visited_n'],
                        expected=lab(detail.get('expected',[])),shown=lab(detail.get('shown',[]))))
            rows.append(dict(query_id=i,label=int(arrays['label'][i]),candidate_failure=candidate_failure,
                first_error=None if failure is None else failure['kind'],first_candidate_error_phase=first_phase,
                semantic_valid=out['audit']['semantic_valid'],
                fresh=summarize([e for e in es if e['fresh']]),revisit=summarize([e for e in es if not e['fresh']])))
        print(name,'generated',len(rows),'seconds',round(time.time()-start,1),flush=True)
    by_phase={key:summarize([e for e in all_events if predicate(e)]) for key,predicate in {
        'root':lambda e:e['root'], 'fresh_pushed':lambda e:e['fresh'] and not e['root'],
        'fresh_all':lambda e:e['fresh'],'revisit':lambda e:not e['fresh']}.items()}
    buckets={}
    for name2,predicate in [('root',lambda e:e['root']),('fresh_pushed',lambda e:e['fresh'] and not e['root']),('revisit',lambda e:not e['fresh'])]:
        for nonempty in (False,True):
            buckets[f'{name2}_{"nonempty" if nonempty else "empty"}']=summarize([e for e in all_events if predicate(e) and e['nonempty']==nonempty])
    temporal={}
    for lo,hi in [(1,1),(2,5),(6,20),(21,50),(51,100000)]:
        for fresh in (False,True):
            temporal[f'{lo}-{hi}_{"fresh" if fresh else "revisit"}']=summarize([e for e in all_events if lo<=e['record']<=hi and e['fresh']==fresh])
    result=dict(name=name,run=str(folder),checkpoint_step=step,checkpoint_sha256=hashlib.sha256((folder/'checkpoints/best.pt').read_bytes()).hexdigest(),
        n_queries=len(rows),precision=cfg.train.precision,ids=ids,by_phase=by_phase,by_nonempty=buckets,by_record=temporal,
        fresh_with_visited_successors=summarize([e for e in all_events if e['fresh'] and e['blocked_successors']>0]),
        fresh_without_visited_successors=summarize([e for e in all_events if e['fresh'] and not e['blocked_successors']]),
        candidate_failure_kinds=dict(Counter(e['failure'] for e in all_events if e['bad'])),
        first_candidate_error_counts=dict(Counter(r['first_candidate_error_phase'] for r in rows if r['candidate_failure'])),
        first_errors=dict(Counter(r['first_error'] for r in rows)),rows=rows,examples=examples,
        definition='Native greedy until first semantic error, original context/STOP; count candidate attempts at the second separator after correct frontier+visited, including outside-graph nodes, invalid tokens and unclosed fields. Forced-budget unfinished fields are censored. Rates conditional on still-correct prefixes.')
    del model;torch.cuda.empty_cache();return result

if __name__=='__main__':
    torch.set_num_threads(2);out=m.OUT/'candidate_first_visit.json'
    result=dict(version=2,original_fixed_full_pool=original_fixed(),native_panels={})
    if out.exists() and json.loads(out.read_text()).get('version')==2:result=json.loads(out.read_text())
    for name in ('context_n16','context_n100','fixed_simple_n100','fixed_hard_n100','context_hard_n100'):
        if name not in result['native_panels']:
            result['native_panels'][name]=run(name)
            out.write_text(json.dumps(result,indent=2)+'\n')

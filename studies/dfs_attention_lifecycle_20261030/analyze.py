"""Lifecycle-aligned attention statistics and adjacent-layer relay audits.

Frozen CPU FP32. No training; no persistent activations or dense attention.
Discovery/confirmation panels chosen before inference, by label/length only.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='4'
import sys,json,time,hashlib,gzip
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
import torch
ROOT=Path('/mnt/ningxuefei/yangshuo/cot-navigation-unified')
sys.path.insert(0,str(ROOT/'src'))
from cot_navigation.config import load_config
from cot_navigation.data.dataset import ContextStore
from cot_navigation.models.transformer import build_model
from cot_navigation.protocols.validate_trace import parse_completed_steps
from cot_navigation.protocols.search_replay import replay_state
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.evaluation.rollout import generate

OUT=ROOT/'studies/dfs_attention_lifecycle_20261030'
RUN=ROOT/'outputs/scale_v1/launches/scale_v1_s0/context/curriculum/dfs_state'
SETS={'n16':'context_l4w4_n16_m24_train50000_s0','n100':'context_l10w10_n100_m150_train50000_s0'}
GROUPS=['graph_EDGE','graph_source','graph_destination','query_source','query_target','prompt_bar',
        'current_stack','current_visited','current_candidates','current_bars','current_action',
        'history_stack','history_visited','history_candidates','history_bars','history_action','history_semicolon','self','other']
ROLES=['graph_EDGE','graph_source','graph_destination','query_marker','query_source','query_target','prompt_bar',
       'stack','visited','candidates','stack_end_bar','visited_end_bar','candidate_end_bar','action','argument','semicolon','answer']

def save(name,data):
    p=OUT/name;p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def parse(tokens,P,tok):
    recs=[];start=P
    for end in range(P,len(tokens)):
        if tokens[end]!=tok.id(';'):continue
        bars=[k for k in range(start,end) if tokens[k]==tok.id('|')]
        if len(bars)!=3:break
        a,b,c=bars
        recs.append(dict(start=start,end=end+1,a=a,b=b,c=c,stack=tokens[start:a],visited=tokens[a+1:b],
                         candidates=tokens[b+1:c],action=tok.name(tokens[c+1]).strip('<>'),
                         argument=tokens[c+2] if c+2<end else None))
        start=end+1
    return recs

def metadata(tokens,P,recs,tok):
    n=len(tokens);roles=np.full(n,3,np.int16);rids=np.full(n,-1,np.int16)
    roles[:P-4]=np.tile([0,1,2],(P-4)//3);roles[P-3:P]=[4,5,6]
    for ri,r in enumerate(recs):
        s,a,b,c,end=[r[k] for k in ['start','a','b','c','end']]
        roles[s:a]=7;roles[a+1:b]=8;roles[b+1:c]=9
        roles[[a,b,c]]=[10,11,12];roles[c+1:end-1]=13
        if r['argument'] is not None:roles[c+2]=14
        roles[end-1]=15;rids[s:end]=ri
    if tokens[-1] in [tok.yes,tok.no]:roles[-1]=16
    return roles,rids

def select(store,tag,split):
    buckets=defaultdict(list);excluded=0
    for ix in store.indices(split).tolist():
        row=store.render(ix,'dfs_state',TOK);length=len(row.tokens)
        if length>CFG.model.max_seq_len:excluded+=1;continue
        band=('short' if length<=1024 else 'middle' if length<=2048 else 'long') if tag=='n100' else ('short' if row.meta['steps']<=7 else 'long')
        buckets[(row.label,band)].append((ix,length,row.meta['steps']))
    rng=np.random.default_rng(20261030+(0 if split=='val' else 100)+(0 if tag=='n16' else 1000))
    chosen=[];each=2 if tag=='n16' else 3
    for (label,band),items in sorted(buckets.items()):
        order=rng.permutation(len(items))[:each]
        for i in order:
            ix,length,nrec=items[int(i)];chosen.append(dict(index=ix,label=label,length=length,records=nrec,band=band))
    return chosen,excluded,{f'{k[0]}_{k[1]}':len(v) for k,v in buckets.items()}

def events(tokens,P,recs):
    ev=[];seen=set()
    def add(phase,target,ri,sub=''):
        if target>0 and target<len(tokens):ev.append(dict(phase=phase,q=target-1,target=target,ri=ri,sub=sub))
    for ri,r in enumerate(recs):
        s,a,b,c=[r[k] for k in ['start','a','b','c']]
        prev=recs[ri-1]['action'] if ri else 'ROOT'
        add('stack_init_'+prev,s,ri)
        for p in range(s+1,a):add('stack_continue',p,ri,prev)
        add('stack_end',a,ri,prev)
        add('visited_init',a+1,ri,prev)
        for p in range(a+2,b):add('visited_continue',p,ri,prev)
        add('visited_end',b,ri,prev)
        fresh=r['stack'][-1] not in seen
        sub='root' if ri==0 else 'fresh' if fresh else 'revisit'
        if b+1==c:add('candidate_empty',c,ri,sub)
        else:
            add('candidate_init_'+sub,b+1,ri)
            for p in range(b+2,c):add('candidate_continue',p,ri,sub)
            add('candidate_end',c,ri,sub)
        add('action_'+r['action'],c+1,ri,sub)
        if r['action']=='PUSH':add('push_argument',c+2,ri,sub)
        add('record_end',r['end']-1,ri,r['action'])
        seen.add(r['stack'][-1])
    if tokens[-1] in [TOK.yes,TOK.no]:add('answer_'+TOK.name(tokens[-1]).strip('<>'),len(tokens)-1,len(recs)-1)
    # Equal query weighting, and evenly spaced opportunities within each phase.
    selected=[];byphase=defaultdict(list)
    for e in ev:byphase[e['phase']].append(e)
    for phase,values in sorted(byphase.items()):
        for i in np.unique(np.linspace(0,len(values)-1,min(12,len(values)),dtype=int)):
            selected.append(values[int(i)])
    return sorted(selected,key=lambda e:(e['q'],e['phase']))

def groupmask(roles,rids,ri,q):
    groups=np.full(q+1,len(GROUPS)-1,np.int16)
    for p in range(q+1):
        role=int(roles[p]);old=rids[p]<ri
        if role in [0,1,2]:g=role
        elif role in [4,5,6]:g=role-1
        elif role in [7,8,9]:g=(11 if old else 6)+(role-7)
        elif role in [10,11,12]:g=14 if old else 9
        elif role in [13,14]:g=15 if old else 10
        elif role==15:g=16 if old else 18
        else:g=18
        groups[p]=g
    groups[q]=17
    return groups

def routes(tokens,P,recs,e):
    q,ri,target=e['q'],e['ri'],tokens[e['target']];r=recs[ri];s,a,b,c=[r[k] for k in ['start','a','b','c']]
    out=dict(prompt_bar=[P-1],query_source=[P-3],query_target=[P-2],current_stack_top=[a-1],
             current_stack_root=[s],current_visited_root=[a+1],current_candidates=list(range(b+1,c)),
             current_visited=list(range(a+1,b)),current_stack=list(range(s,a)),
             current_stack_bar=[a],current_visited_bar=[b],current_candidate_bar=[c])
    if ri:
        prev=recs[ri-1];out['previous_semicolon']=[prev['end']-1]
        out['previous_action']=[prev['c']+1]
        if prev['argument'] is not None:out['previous_push_argument']=[prev['c']+2]
    samecur=[p for p in recs[:ri] if p['stack'][-1]==r['stack'][-1] and p['action']=='PUSH']
    if samecur:out['previous_same_current_PUSH_pair']=[samecur[-1]['c']+1,samecur[-1]['c']+2]
    if TOK.node_lo<=target<TOK.node_lo+TOK.n_node_tokens:
        out['same_target_token']=[p for p in range(q+1) if tokens[p]==target]
        out['chosen_candidate']=[p for p in range(b+1,c) if tokens[p]==target]
        out['previous_same_field_target']=[p for past in recs[:ri] for p in (range(past['start'],past['a']) if e['phase'].startswith('stack') else range(past['a']+1,past['b']) if e['phase'].startswith('visited') else range(past['b']+1,past['c'])) if tokens[p]==target]
        out['parent_matched_first_visited_carrier']=[p['a']+1 for p in recs[:ri] if len(p['stack'])>=2 and p['stack'][-2]==target]
        out['history_target_prediction_predecessor']=[p-1 for p in range(P,s) if tokens[p]==target]
        for field,lo,hi in [('stack','start','a'),('visited','a','b'),('candidate','b','c')]:
            out['history_'+field+'_target_predecessor']=[p-1 for past in recs[:ri] for p in range(past[lo]+(0 if field=='stack' else 1),past[hi]) if tokens[p]==target]
        out['current_stack_target_predecessor']=[p-1 for p in range(s,a) if tokens[p]==target]
        out['current_visited_target_predecessor']=[p-1 for p in range(a+1,b) if tokens[p]==target]
    out['stack_penultimate']=[a-2] if a-s>=2 else []
    if e['phase'] in ['stack_end','visited_end','candidate_end']:
        field={'stack_end':'a','visited_end':'b','candidate_end':'c'}[e['phase']]
        out['historical_homologous_field_end']=[past[field] for past in recs[:ri]]
        out['previous_homologous_field_end']=[recs[ri-1][field]] if ri else []
    out['history_same_current_candidate_end']=[past['c'] for past in recs[:ri] if past['stack'][-1]==r['stack'][-1]]
    edge=np.asarray(tokens[:P-4]).reshape(-1,3)
    relevant=np.flatnonzero(edge[:,1]==r['stack'][-1])
    out['current_adjacency_raw_destinations']=[3*int(i)+2 for i in relevant]
    out['current_adjacency_destination_shift3']=[3*int(i)+5 for i in relevant]
    if ri==len(recs)-1:out['terminal_action']=[r['c']+1]
    return {name:[p for p in dict.fromkeys(pos) if 0<=p<=q] for name,pos in out.items()}

def desc(p,tokens,roles,rids):
    return dict(position=int(p),token=TOK.name(tokens[p]),role=ROLES[int(roles[p])],record=int(rids[p])+1)

def analyze_one(store,tag,split,spec,mode='reference',generated=None):
    start=time.time();ix=spec['index'];row=store.render(ix,'dfs_state',TOK)
    tokens=row.tokens[:-1] if generated is None else generated
    P=row.prompt_len;recs=parse(tokens,P,TOK);roles,rids=metadata(tokens,P,recs,TOK)
    es=events(tokens,P,recs)
    ids=torch.tensor(tokens)[None]
    with torch.inference_mode():logits,ws=MODEL(ids,return_attn=True)
    A=[w[0,0].numpy() for w in ws]
    assert all(np.max(np.abs(a.sum(-1)-1))<3e-6 for a in A)
    assert all(np.max(np.triu(a,1))==0 for a in A)
    output=[];examples=[];target_cache={};third_cache={};all_tokens=np.asarray(tokens)
    def target_read(layer,node):
        key=(layer,int(node))
        if key not in target_cache:
            pos=np.flatnonzero(all_tokens==node)
            target_cache[key]=A[layer][:,pos].sum(axis=1) if len(pos) else np.zeros(len(tokens))
        return target_cache[key]
    for e in es:
        q,ri=e['q'],e['ri'];r=recs[ri];g=groupmask(roles,rids,ri,q);rt=routes(tokens,P,recs,e)
        rows=np.stack([a[q,:q+1] for a in A]);masses=[[float(z[g==j].sum()) for j in range(len(GROUPS))] for z in rows]
        route_mass={name:[float(z[pos].sum()) if pos else 0.0 for z in rows] for name,pos in rt.items()}
        target=tokens[e['target']];isnode=TOK.node_lo<=target<TOK.node_lo+TOK.n_node_tokens
        symbol=np.isin(roles[:q+1],[6,10,11,12,15]);action=np.isin(roles[:q+1],[13,14])
        two=[];best_relays=[];three=[]
        if isnode:
            matches=np.flatnonzero(np.asarray(tokens[:q+1])==target)
            candidates=[v for v in np.unique(tokens[:q+1]) if TOK.node_lo<=v<TOK.node_lo+TOK.n_node_tokens and v!=target]
            target_last=max(matches) if len(matches) else -1
            wrong=min(candidates,key=lambda v:(abs(sum(t==v for t in tokens[:q+1])-len(matches)),abs(max(p for p in range(q+1) if tokens[p]==v)-target_last),int(v))) if candidates else None
            wrongpos=np.flatnonzero(np.asarray(tokens[:q+1])==wrong) if wrong is not None else []
            for li in range(1,6):
                upstream=target_read(li-1,target)[:q+1]
                wrongup=target_read(li-1,wrong)[:q+1] if wrong is not None else np.zeros(q+1)
                via=rows[li]*upstream
                two.append(dict(layer=li+1,total=float(via.sum()),via_symbol=float(via[symbol].sum()),via_action=float(via[action].sum()),
                    via_symbol_control=float((rows[li]*wrongup)[symbol].sum())))
                if symbol.any():
                    pos=int(np.where(symbol,via,-1).argmax());up=A[li-1][pos,:pos+1]
                    best_relays.append(dict(layer=li+1,key=desc(pos,tokens,roles,rids),first_weight=float(rows[li,pos]),
                        upstream_target_mass=float(upstream[pos]),path_mass=float(via[pos]),
                        upstream_top=[dict(**desc(int(k),tokens,roles,rids),weight=float(up[k])) for k in np.argsort(-up)[:3]]))
                if li>=2:
                    key=(li,int(target))
                    if key not in third_cache:third_cache[key]=A[li-1]@target_read(li-2,target)
                    mid=third_cache[key][:q+1]
                    three.append(dict(layer=li+1,via_first_symbol=float((rows[li]*mid)[symbol].sum())))
        comparison=[]
        if e['phase'].startswith('action_'):
            for li in range(1,6):
                stackmask=np.zeros(q+1,bool);stackmask[r['start']:r['a']]=True
                goal=A[li-1][:q+1,P-2];source=A[li-1][:q+1,P-3]
                comparison.append(dict(layer=li+1,goal_via_stack=float((rows[li]*goal)[stackmask].sum()),source_via_stack=float((rows[li]*source)[stackmask].sum())))
        top=[]
        for li,z in enumerate(rows):
            top.append([dict(**desc(int(k),tokens,roles,rids),weight=float(z[k])) for k in np.argsort(-z)[:5]])
        result=dict(**e,tag=tag,split=split,mode=mode,index=ix,label=row.label,n_records=len(recs),length=len(tokens),
             record_fraction=ri/max(1,len(recs)-1),visited_size=len(r['visited']),stack_depth=len(r['stack']),candidate_size=len(r['candidates']),
             input=TOK.name(tokens[q]),next=TOK.name(target),next_correct=int(logits[0,q].argmax())==target,
             group_mass=masses,route_mass=route_mass,two_hop=two,three_hop=three,top_keys=top,best_symbol_relays=best_relays,goal_comparison=comparison)
        output.append(result)
    # Compact raw rows for reproducing independent row checks and later drawings.
    if len(recs)>=30 and split=='test':
        for e in es:
            if e['ri'] in [0,len(recs)//2,len(recs)-1] and e['phase'].startswith(('action','answer')):
                ex=next(z for z in output if z['q']==e['q'] and z['phase']==e['phase'])
                examples.append(ex)
    summary=dict(tag=tag,split=split,mode=mode,index=ix,length=len(tokens),records=len(recs),events=len(es),
                 complete_reference=mode=='reference',label=row.label,teacher_forced_next_accuracy=float(np.mean([e['next_correct'] for e in output])),
                 elapsed_sec=time.time()-start)
    print(json.dumps(summary),flush=True)
    del logits,ws,A
    return output,summary

def main():
    global MODEL,TOK,CFG
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    CFG=load_config([RUN/'resolved_config.yaml']);TOK=Tokenizer.from_config(CFG)
    MODEL=build_model(CFG.model,len(TOK),TOK.pad).eval();cp=RUN/'checkpoints/best.pt'
    state=torch.load(cp,map_location='cpu',weights_only=False);MODEL.load_state_dict(state['model']);MODEL.requires_grad_(False)
    stores={tag:ContextStore(ROOT/'data/scale_v1'/name) for tag,name in SETS.items()};plan={};excluded={}
    for tag,store in stores.items():
        plan[tag]={};excluded[tag]={}
        for split in ['val','test']:
            plan[tag][split],count,pools=select(store,tag,split);excluded[tag][split]=dict(over_context=count,pool_counts=pools)
    save('plan.json',dict(seed=20261030,panels=plan,excluded=excluded,checkpoint=str(cp),checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),
      step=int(state['step']),selection='N16 label x <=7/>7 records, 2 per cell per split; N100 label x <=1024/1025..2048/2049..4096 full tokens, 3 per cell per split; random without performance selection.',
      mode='Complete canonical teacher-forced reference; actual free panel reported separately',groups=GROUPS,
      route_composition='Adjacent-layer attention-only composition; no residual or FFN attribution. Not causal influence.'))
    all_summary=[]
    with gzip.open(OUT/'events.jsonl.gz','wt',encoding='utf-8') as file:
        for tag,store in stores.items():
            for split,specs in plan[tag].items():
                for spec in specs:
                    es,summary=analyze_one(store,tag,split,spec)
                    for e in es:file.write(json.dumps(e,separators=(',',':'))+'\n')
                    file.flush();all_summary.append(summary);save('progress.json',all_summary)
    # Independent, performance-unselected native rollouts: one per label in
    # middle and long test bands. Retain all failures and only analyze the
    # fully legal prefix before the first semantic failure.
    selected=[]
    for label in [0,1]:
        for band in ['middle','long']:
            selected.append(next(s for s in plan['n100']['test'] if s['label']==label and s['band']==band))
    free_summary=[]
    with gzip.open(OUT/'free_events.jsonl.gz','wt',encoding='utf-8') as file:
        for spec in selected:
            store=stores['n100'];ix=spec['index'];row=store.render(ix,'dfs_state',TOK);P=row.prompt_len
            torch.set_num_threads(2)
            with torch.inference_mode():out=generate(MODEL,[row.tokens[:P]],'dfs_state',TOK,'fp32',cot_margin=CFG.protocol.cot_margin)[0]
            torch.set_num_threads(4)
            inverse={t:i for i,t in enumerate(store.node_vocab_ids(ix,TOK))};adj=store.adj(ix)
            src=int(store.arrays['s_topo'][ix]);dst=int(store.arrays['t_topo'][ix])
            native=parse(out['tokens'],P,TOK);valid_end=P;legal=0
            for r in native:
                audit=replay_state(adj,src,dst,out['tokens'][P:r['end']],TOK,inverse)
                if audit['records']!=legal+1:break
                if not audit['valid'] and audit['error']!='missing_terminal':break
                legal+=1;valid_end=r['end']
            audit=replay_state(adj,src,dst,out['trace_tokens'],TOK,inverse)
            summary=dict(index=ix,generated=len(out['tokens'])-P,valid_records=legal,full_valid=audit['valid'],semantic_error=audit['error'],
                         correct=out['pred']==spec['label'],pred=out['pred'],label=spec['label'],interrupted=out.get('cot_interrupted'),
                         band=spec['band'],tokens=out['tokens'],legal_end=valid_end)
            if legal:
                end=out['tokens'][:valid_end]
                if audit['valid'] and out['pred'] is not None:end=out['tokens']
                es,ss=analyze_one(store,'n100','test',spec,'free_legal_prefix',end)
                for e in es:file.write(json.dumps(e,separators=(',',':'))+'\n')
            free_summary.append(summary);save('free_rollouts.json',free_summary)
            print('FREE',json.dumps({k:v for k,v in summary.items() if k!='tokens'}),flush=True)
    save('summary_queries.json',all_summary)
    save('COMPLETE.json',dict(reference_queries=len(all_summary),free_queries=len(free_summary),finished=True))

if __name__=='__main__':main()

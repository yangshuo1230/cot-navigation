"""Fixed N100 PATH: legal goal hitting, useful-successor probabilities,
exact random-walk baseline, changed-goal pairs and unfiltered sampling.

Reuse the original test queries/free generations. All probability readouts
use original legal prefixes, not oracle paths. GPU1 sampling uses full-vocab
temperature-one probabilities without adjacency masks or retries. No training.
"""
import os, json, importlib.util, itertools
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import torch
torch.set_num_threads(2)
ROOT=Path.cwd(); OUT=ROOT/'studies/path_goal_preference'
spec=importlib.util.spec_from_file_location('storage_previous',ROOT/'studies/fixed_graph_storage/study.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
from cot_navigation.models.attention import KVCache


def walk_prob(p,t):
    prob=np.zeros(len(p['nodes']));prob[t]=1
    for u in reversed(range(len(prob))):
        if u!=t and p['a'][u].any():prob[u]=prob[p['a'][u]].mean()
    assert np.all((prob>0)==p['reach'][:,t])
    return prob


def audit(p):
    rows=[r for r in p['rows'] if r['split']==2]; inv={int(v):i for i,v in enumerate(p['nodes'])}; results=[]
    for row in rows:
        s,t=row['source'],row['target']; nodes=[]; legal=True; hit=False; first_error=None
        for j,token in enumerate(row['trace']):
            u=inv.get(token,-1)
            if u<0 or (j==0 and u!=s) or (j>0 and not p['a'][nodes[-1],u]):
                legal=False;first_error=j;break
            nodes.append(u)
            if u==t:hit=True
        valid_start=bool(nodes and nodes[0]==s)
        complete=bool(legal and valid_start and nodes[-1]==t)
        results.append(dict(query=row['query_ids'],source=s,target=t,label=row['label'],
            pred=row['pred'],legal_goal_hit=hit,full_trace_legal_end_goal=complete,
            full_trace_legal=legal and valid_start,first_error=first_error,
            random_goal_probability=float(walk_prob(p,t)[s]),hops=row['hops']))
    return rows,results


def grouped_mean(items,key):
    groups={}
    for row in items:groups.setdefault(row['query'],[]).append(row[key])
    return float(np.mean([np.mean(v) for v in groups.values()])) if groups else None


@torch.inference_mode()
def probabilities(p,rows):
    inv={int(v):i for i,v in enumerate(p['nodes'])}; result=[]; first={};examples=[]; mismatches=0; mismatch_details=[]
    for start in range(0,len(rows),64):
        sub=rows[start:start+64]
        seqs=[p['header']+[p['tok'].id('<QUERY>'),int(p['nodes'][r['source']]),int(p['nodes'][r['target']]),p['tok'].id('|')]+r['trace'] for r in sub]
        lengths=[len(s) for s in seqs]; ids=torch.full((len(sub),max(lengths)),p['tok'].pad,dtype=torch.long)
        for i,seq in enumerate(seqs):ids[i,:len(seq)]=torch.tensor(seq)
        logits=p['model'](ids).numpy(); plen=len(p['header'])+4
        for b,row in enumerate(sub):
            if not row['label']:continue
            source,target=row['source'],row['target']; previous=None
            for j,token in enumerate(row['trace']):
                u=inv.get(token,-1)
                if u<0 or (j==0 and u!=source) or (previous is not None and not p['a'][previous,u]):break
                previous=u
                if u==target:break
                adjacent=p['a'][u]; useful=adjacent&p['reach'][:,target]
                if not adjacent.any():break
                logit=logits[b,plen+j]; prob=np.exp(logit-np.logaddexp.reduce(logit)); npnode=prob[p['nodes']]
                succmass=float(npnode[adjacent].sum()); goodmass=float(npnode[useful].sum()); chance=float(useful.sum()/adjacent.sum())
                pred=int(logit.argmax()); replay_v=inv.get(pred,-1)
                saved_next=row['trace'][j+1] if j+1<len(row['trace']) else (p['tok'].yes if row['pred']==1 else p['tok'].no if row['pred']==0 else None)
                if saved_next is not None and pred!=saved_next:
                    mismatches+=1
                    mismatch_details.append(dict(query=row['query_ids'],k=j,saved=int(saved_next),replay=pred,
                        mixed=bool(useful.any() and (adjacent&~useful).any()),saved_is_node=saved_next in inv,
                        saved_useful=bool(inv.get(saved_next,-1)>=0 and useful[inv[saved_next]]),
                        replay_useful=bool(replay_v>=0 and useful[replay_v]),logit_gap=float(logit[pred]-logit[saved_next])))
                # Empirical choices use the saved free generation; FP32 replay supplies probabilities.
                v=inv.get(saved_next,-1)
                rec=dict(query=row['query_ids'],source=source,target=target,current=u,k=j,
                    mixed=bool(useful.any() and (adjacent&~useful).any()),dead=not useful.any(),
                    actual_legal=bool(v>=0 and adjacent[v]),actual_useful=bool(v>=0 and useful[v]),
                    successor_probability=succmass,useful_probability=goodmass,
                    useful_conditional=goodmass/max(succmass,1e-30),random_useful_probability=chance)
                result.append(rec)
                if j==0:first[row['query_ids']]=dict(record=rec,prob=npnode,useful=useful,chosen=v)
                if rec['mixed'] and not rec['actual_useful'] and len(examples)<5:
                    examples.append(dict(query=rec['query'],source=p['tok'].name(int(p['nodes'][source])),current=p['tok'].name(int(p['nodes'][u])),
                        target=p['tok'].name(int(p['nodes'][target])),chosen=p['tok'].name(saved_next),
                        useful=[p['tok'].name(int(vv)) for vv in p['nodes'][useful]],conditional_useful=rec['useful_conditional'],final_pred=row['pred']))
    groups={}
    for name,keep in [('all_decisions',lambda r:True),('mixed_only',lambda r:r['mixed']),('first_step_mixed',lambda r:r['mixed'] and r['k']==0)]:
        selected=[r for r in result if keep(r)]
        groups[name]=dict(decisions=len(selected),queries=len(set(r['query'] for r in selected)))
        for key in ('actual_legal','actual_useful','successor_probability','useful_probability','useful_conditional','random_useful_probability'):
            groups[name][key]=grouped_mean(selected,key)
            groups[name][key+'_micro']=float(np.mean([r[key] for r in selected])) if selected else None
    # Graph-selected changed-goal pairs at the SAME source and no prior history.
    pairs=[];pair_examples=[];depth=np.zeros(len(p['nodes']),int)
    for u in range(len(depth)):
        for v in np.flatnonzero(p['a'][u]):depth[v]=max(depth[v],depth[u]+1)
    keys=list(first)
    for qa,qb in itertools.combinations(keys,2):
        a,b=first[qa],first[qb];ra,rb=a['record'],b['record']
        if ra['source']!=rb['source'] or not ra['mixed'] or not rb['mixed'] or np.any(a['useful']&b['useful']):continue
        sa=float(a['prob'][p['a'][ra['source']]].sum());sb=float(b['prob'][p['a'][rb['source']]].sum())
        progress=.5*((a['prob'][a['useful']].sum()/sa-b['prob'][a['useful']].sum()/sb)+(b['prob'][b['useful']].sum()/sb-a['prob'][b['useful']].sum()/sa))
        both=ra['actual_useful'] and rb['actual_useful']
        rec=dict(query_a=qa,query_b=qb,source=ra['source'],target_a=ra['target'],target_b=rb['target'],
            both_useful=bool(both),changes=bool(a['chosen']!=b['chosen']),probability_goal_shift=float(progress),
            random_both=ra['random_useful_probability']*rb['random_useful_probability'],
            matched_depth_band=bool(depth[ra['target']]==depth[rb['target']] and ra['target']//10==rb['target']//10))
        pairs.append(rec)
        if len(pair_examples)<4 and both:
            pair_examples.append(dict(source=p['tok'].name(int(p['nodes'][ra['source']])),
                target_a=p['tok'].name(int(p['nodes'][ra['target']])),target_b=p['tok'].name(int(p['nodes'][rb['target']])),
                useful_a=[p['tok'].name(int(v)) for v in p['nodes'][a['useful']]], useful_b=[p['tok'].name(int(v)) for v in p['nodes'][b['useful']]],
                chosen_a=p['tok'].name(int(p['nodes'][a['chosen']])),chosen_b=p['tok'].name(int(p['nodes'][b['chosen']]))) )
    pair_results={}
    for name,selected in [('all',pairs),('matched_depth_band',[r for r in pairs if r['matched_depth_band']])]:
        qids=set(q for r in selected for q in (r['query_a'],r['query_b']))
        pair_results[name]=dict(pairs=len(selected),unique_queries=len(qids),sources=len(set(r['source'] for r in selected)),
            unique_query_useful=float(np.mean([first[q]['record']['actual_useful'] for q in qids])) if qids else None,
            **{key:float(np.mean([r[key] for r in selected])) if selected else None for key in ('both_useful','changes','probability_goal_shift','random_both')})
    return dict(groups=groups,goal_pairs=pair_results,pair_examples=pair_examples,failure_examples=examples,replay_argmax_mismatches=mismatches,mismatch_details=mismatch_details)


@torch.inference_mode()
def sample(p,rows,k=16,device='cuda:1'):
    rows=[r for r in rows if r['label']==1]; model=p['model']; tok=p['tok']; n=len(p['nodes']); inv={int(v):i for i,v in enumerate(p['nodes'])}
    prompts=[p['header']+[tok.id('<QUERY>'),int(p['nodes'][r['source']]),int(p['nodes'][r['target']]),tok.id('|'),int(p['nodes'][r['source']])] for r in rows]
    check=torch.tensor(prompts[:8]); cpu_logits=model(check)[:,-1].numpy();model.to(device)
    torch.backends.cuda.matmul.allow_tf32=False
    gpu_logits=model(check.to(device))[:,-1].cpu().numpy();device_error=float(np.max(abs(cpu_logits-gpu_logits)))
    assert device_error<1e-3,device_error
    generator=torch.Generator(device=device).manual_seed(20261024)
    jobs=[(i,j) for i in range(len(rows)) for j in range(k)];success=np.zeros((len(rows),k),bool);failures={};cache_error=0
    for start in range(0,len(jobs),256):
        batch=jobs[start:start+256]; ids=torch.tensor([prompts[i] for i,j in batch],device=device);length=ids.shape[1]
        cache=KVCache(model.n_layers,len(batch),model.n_heads,length+n+2,model.d_model//model.n_heads,device,torch.float32)
        logits=model(ids,cache=cache,cache_mode='prefill')[:,-1]
        if start==0:
            chosen=logits.argmax(-1)[:,None];pos=torch.full((len(batch),1),length,device=device,dtype=torch.long);cache.valid_length=length+1
            cached=model(chosen,positions=pos,cache=cache,cache_mode='decode')[:,-1]
            plain=model(torch.cat([ids,chosen],1))[:,-1];cache_error=float((cached-plain).abs().max())
            assert cache_error<1e-3,cache_error
        active=np.ones(len(batch),bool);current=np.array([rows[i]['source'] for i,j in batch]);goals=np.array([rows[i]['target'] for i,j in batch])
        for step in range(n):
            take=np.flatnonzero(active);sampled=torch.multinomial(logits[take].softmax(-1),1,generator=generator)[:,0].cpu().numpy()
            feed=np.full(len(batch),tok.pad,int)
            for b,token in zip(take,sampled):
                node=inv.get(int(token),-1);reason=None
                if node<0:reason='non_node_before_goal'
                elif not p['a'][current[b],node]:reason='invalid_edge'
                elif node==goals[b]:
                    success[batch[b][0],batch[b][1]]=True;reason='goal'
                elif not p['reach'][node,goals[b]]:reason='dead_branch'
                if reason:
                    active[b]=False;failures[reason]=failures.get(reason,0)+1
                else:current[b]=node;feed[b]=token
            if not active.any():break
            pos=torch.full((len(batch),1),length+step,device=device,dtype=torch.long);cache.valid_length=length+step+1
            logits=model(torch.tensor(feed[:,None],device=device),positions=pos,cache=cache,cache_mode='decode')[:,-1]
        failures['budget']=failures.get('budget',0)+int(active.sum())
        if start%2048==0:print('sample',start,'/',len(jobs),flush=True)
    ps=success.mean(1);random=np.array([walk_prob(p,r['target'])[r['source']] for r in rows])
    result=dict(queries=len(rows),samples_per_query=k,successes=int(success.sum()),trajectories=len(jobs),success_rate=float(ps.mean()),
        exact_random_rate=float(random.mean()),query_win_fraction=float((ps>random).mean()),
        monte_carlo_se=float(np.sqrt(np.sum(ps*(1-ps)/(k-1)))/len(ps)),stops=failures,
        CPU_GPU_logit_max_error=device_error,cache_decode_logit_max_error=cache_error,
        convention='Start from prompt plus source token. Full-vocab temperature1 sampling; no edge masks/retry. Goal is absorbing; first illegal/protocol step or graph-dead branch is failure. No final answer requirement.')
    model.to('cpu');torch.cuda.empty_cache()
    return result


def run(setting):
    p=base.load(setting,'path','cpu');rows,audited=audit(p)
    probabilities_=probabilities(p,rows)
    groups={}
    for label,name in [(1,'positive'),(0,'negative')]:
        subset=[r for r in audited if r['label']==label]
        groups[name]=dict(n=len(subset),answer_accuracy=float(np.mean([r['pred']==label for r in subset])),
            legal_goal_hit=int(sum(r['legal_goal_hit'] for r in subset)),
            legal_goal_rate=float(np.mean([r['legal_goal_hit'] for r in subset])),
            complete_legal_end_goal=int(sum(r['full_trace_legal_end_goal'] for r in subset)),
            exact_random_goal_rate=float(np.mean([r['random_goal_probability'] for r in subset])))
    hops=[]
    for hop in sorted(set(r['hops'] for r in audited if r['label'])):
        subset=[r for r in audited if r['label'] and r['hops']==hop]
        hops.append(dict(hops=hop,n=len(subset),legal_goal_rate=float(np.mean([r['legal_goal_hit'] for r in subset])),random_rate=float(np.mean([r['random_goal_probability'] for r in subset]))))
    out=dict(manifest=p['manifest'],greedy=groups,by_hops=hops,preference=probabilities_,sampling=sample(p,rows))
    base.write(OUT/f'{setting}.json',out);print(setting,'COMPLETE',flush=True)


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    for setting in ('simple','hard'):run(setting)

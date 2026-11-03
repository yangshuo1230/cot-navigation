"""Two PATH figure datasets: hop-stratified behavior and sublayer probabilities.

Original fixed N100 simple/hard best checkpoints and saved greedy test traces.
Frozen CPU FP32; no fitting/training. Intermediate probabilities use the same
final_norm/LM head (uncalibrated logit lens), not new probes or early exits.
"""
from pathlib import Path
import os, json, importlib.util
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import torch
torch.set_num_threads(2)
ROOT=Path.cwd();HERE=ROOT/'studies/path_behavior_figures'
spec=importlib.util.spec_from_file_location('storage',ROOT/'studies/fixed_graph_storage/study.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)

def random_walk(p,target):
    prob=np.zeros(len(p['nodes']));prob[target]=1
    for u in reversed(range(len(prob))):
        if u!=target and p['a'][u].any():prob[u]=prob[p['a'][u]].mean()
    return prob

def behavior(p,rows):
    inv={int(v):i for i,v in enumerate(p['nodes'])};records=[]
    distance={}
    for row in rows:
        s,t=row['source'],row['target']
        if s not in distance:
            d=np.full(len(p['nodes']),-1,int);d[s]=0;queue=[s]
            for u in queue:
                for v in np.flatnonzero(p['a'][u]):
                    if d[v]<0:d[v]=d[u]+1;queue.append(v)
            distance[s]=d
        assert distance[s][t]==row['hops']
        path=[inv.get(token,-1) for token in row['trace']]
        legal=bool(path and path[0]==s and all(v>=0 for v in path))
        if legal:legal=all(p['a'][u,v] for u,v in zip(path,path[1:]))
        complete=bool(legal and path[-1]==t)
        records.append(dict(query=row['query_ids'],hops=row['hops'],label=row['label'],
            answer_correct=row['pred']==row['label'],full_legal_goal=complete,
            random_goal_probability=float(random_walk(p,t)[s])))
    groups=[]
    for h in sorted(set(r['hops'] for r in records if r['label'])):
        part=[r for r in records if r['label'] and r['hops']==h]
        groups.append(dict(hops=h,n=len(part),correct_answers=sum(r['answer_correct'] for r in part),
            answer_accuracy=float(np.mean([r['answer_correct'] for r in part])),
            legal_goal_count=sum(r['full_legal_goal'] for r in part),
            legal_goal_rate=float(np.mean([r['full_legal_goal'] for r in part])),
            random_goal_probability=float(np.mean([r['random_goal_probability'] for r in part]))))
    negatives=[r for r in records if not r['label']]
    positives=[r for r in records if r['label']]
    return dict(by_hops=groups,negative=dict(n=len(negatives),answer_accuracy=float(np.mean([r['answer_correct'] for r in negatives]))),
        positive=dict(n=len(positives),answer_accuracy=float(np.mean([r['answer_correct'] for r in positives])),
            legal_goal_rate=float(np.mean([r['full_legal_goal'] for r in positives]))))

def aggregate(records):
    result={}
    for name,part in [('mixed_all',records),('mixed_first_step',[r for r in records if r['step']==0])]:
        ids=sorted(set(r['query'] for r in part));byquery={q:[r for r in part if r['query']==q] for q in ids}
        entry=dict(decisions=len(part),queries=len(ids))
        for key in ('successor_full_vocab','useful_full_vocab','useful_given_successor','successor_given_node','useful_given_node','uniform_useful'):
            entry[key]=np.mean([np.mean([r[key] for r in group],axis=0) for group in byquery.values()],axis=0).tolist()
            entry[key+'_decision_weighted']=np.mean([r[key] for r in part],axis=0).tolist()
        result[name]=entry
    return result

@torch.inference_mode()
def probabilities(p,rows):
    tok=p['tok'];model=p['model'];inv={int(v):i for i,v in enumerate(p['nodes'])};observed={}
    def before(layer):
        def hook(module,args):observed[layer,'input']=args[0]
        return hook
    def after(layer):
        def hook(module,args,output):observed[layer,'after_attention']=observed[layer,'input']+output[0]
        return hook
    handles=[]
    for layer,block in enumerate(model.blocks):
        handles.extend([block.register_forward_pre_hook(before(layer)),block.attn.register_forward_hook(after(layer))])
    records=[];final_error=0
    try:
        for start in range(0,len(rows),32):
            sub=rows[start:start+32];plen=len(p['header'])+4
            seqs=[p['header']+[tok.id('<QUERY>'),int(p['nodes'][r['source']]),int(p['nodes'][r['target']]),tok.id('|')]+r['trace'] for r in sub]
            ids=torch.full((len(sub),max(map(len,seqs))),tok.pad,dtype=torch.long)
            for b,seq in enumerate(seqs):ids[b,:len(seq)]=torch.tensor(seq)
            decisions=[]
            for b,row in enumerate(sub):
                if not row['label']:continue
                previous=None
                for step,token in enumerate(row['trace']):
                    u=inv.get(token,-1)
                    if u<0 or (step==0 and u!=row['source']) or (previous is not None and not p['a'][previous,u]):break
                    previous=u
                    if u==row['target']:break
                    adjacent=p['a'][u];useful=adjacent&p['reach'][:,row['target']]
                    if useful.any() and (adjacent&~useful).any():
                        decisions.append((b,plen+step,u,row['target'],row['query_ids'],step))
            if not decisions:continue
            cap={};logits=model(ids,capture=cap)
            states=[model.embed(ids)]
            for layer in range(6):states.extend([observed[layer,'after_attention'],cap['resid'][layer]])
            bb=[r[0] for r in decisions];pos=[r[1] for r in decisions]
            hidden=torch.stack([state[bb,pos] for state in states])
            head=model.lm_head(model.final_norm(hidden))
            error=float((head[-1]-logits[bb,pos]).abs().max());final_error=max(final_error,error)
            assert error<1e-4,error
            for j,(b,position,u,target,q,step) in enumerate(decisions):
                scores=head[:,j];adjacent=p['a'][u];useful=adjacent&p['reach'][:,target]
                log_total=torch.logsumexp(scores,dim=-1)
                log_nodes=torch.logsumexp(scores[:,p['nodes']],dim=-1)
                log_succ=torch.logsumexp(scores[:,p['nodes'][adjacent]],dim=-1)
                log_good=torch.logsumexp(scores[:,p['nodes'][useful]],dim=-1)
                records.append(dict(query=q,step=step,
                    successor_full_vocab=(log_succ-log_total).exp().numpy(),
                    useful_full_vocab=(log_good-log_total).exp().numpy(),
                    useful_given_successor=(log_good-log_succ).exp().numpy(),
                    successor_given_node=(log_succ-log_nodes).exp().numpy(),
                    useful_given_node=(log_good-log_nodes).exp().numpy(),
                    uniform_useful=float(useful.sum()/adjacent.sum())))
    finally:
        for handle in handles:handle.remove()
    return aggregate(records),dict(final_stage_head_matches_native_max_error=final_error)

def run(setting):
    p=base.load(setting,'path','cpu');rows=[r for r in p['rows'] if r['split']==2]
    layers=['E']+[f'L{i}_{kind}' for i in range(1,7) for kind in ('attention','FFN')]
    b=behavior(p,rows);prob,checks=probabilities(p,rows)
    # Compare final-layer probabilities against the independently audited prior result.
    previous=json.loads((ROOT/f'studies/path_goal_preference/{setting}.json').read_text())
    comparisons={}
    for group,old_group in [('mixed_all','mixed_only'),('mixed_first_step','first_step_mixed')]:
        old=previous['preference']['groups'][old_group];new=prob[group]
        assert new['decisions']==old['decisions'] and new['queries']==old['queries']
        for current,prior in [('useful_given_successor','useful_conditional'),('successor_full_vocab','successor_probability'),('useful_full_vocab','useful_probability')]:
            difference=abs(new[current][-1]-old[prior]);assert difference<1e-5,difference
            comparisons[group+'_'+current]=difference
    base.write(HERE/f'{setting}.json',dict(manifest=p['manifest'],layers=layers,behavior=b,probabilities=prob,
        checks=checks|dict(final_stage_vs_previous_max_errors=comparisons),
        method='Original greedy query-test traces, actual BFS shortest hops. Positive queries only for hop panels. '
               'Entire trace must follow real edges and END at goal for legal-goal metric. '
               'Probability curves use positive mixed-branch legal prefixes before goal/error. '
               'Final norm + tied LM head at each intermediate residual, CPU FP32 uncalibrated logit lens. '
               'Per-query mean then across-query mean; no new probes/training.'))
    print(setting,'COMPLETE',b['positive'],prob['mixed_all']['decisions'],flush=True)

if __name__=='__main__':
    HERE.mkdir(exist_ok=True)
    for setting in ('simple','hard'):run(setting)

"""PATH node embeddings: node-heldout relation probes and sparse causal edits.

Only frozen inference on GPU2, closed-form probe fits on GPU0. Edits replace
one prompt input vector, never tied embedding/head parameters. Greedy rollouts
have no graph masks/retries. Main causal pairs are BOTH original query-test
and BOTH probe-test focal nodes. Direction constructor / audit decoder use
disjoint 30-node halves of the 60 probe-training nodes.
"""
from pathlib import Path
import os,json,importlib.util
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import numpy as np
import torch
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False
ROOT=Path.cwd();OUT=ROOT/'studies/path_embedding_causal'
spec=importlib.util.spec_from_file_location('storage',ROOT/'studies/fixed_graph_storage/study.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
from cot_navigation.models.attention import KVCache
import linear
SEEDS=(17,31,47);CONTROLS=(201,303,407);RANKS=(1,2,4,8,16)
DEVICE='cuda:2'

def fit(x,y,train,val):
    if y.ndim==1:y=y[:,None]
    z=torch.tensor(x,device='cuda:0');yy=torch.tensor(y,dtype=torch.float32,device='cuda:0')
    mean=z[train].mean(0);std=z[train].std(0).clamp_min(.05)
    norm=(z-mean)/std;norm=torch.cat([norm,torch.ones((len(x),1),device=z.device)],1)
    zz=norm[train];gram=zz.T@zz/len(train);rhs=zz.T@yy[train]/len(train)
    eye=torch.eye(norm.shape[1],device=z.device);eye[-1,-1]=0
    jitter=1e-6*torch.eye(norm.shape[1],device=z.device);best=np.inf
    for penalty in (.001,.01,.1,1.,10.):
        w=torch.linalg.solve(gram+penalty*eye+jitter,rhs)
        loss=float(((norm[val]@w-yy[val])**2).mean())
        if loss<=best:
            best=loss;ww=(w[:-1]/std[:,None]).cpu().numpy();bb=(w[-1]-mean@(w[:-1]/std[:,None])).cpu().numpy();chosen=penalty
    return ww,bb,chosen

def threshold(scores,y,ids):
    mask=~np.eye(len(y),dtype=bool)[ids]
    truth=y[ids].astype(bool)&mask;cuts=np.unique(np.r_[0,np.quantile(scores[ids][mask],np.linspace(0,1,41))])
    def f1(cut):
        guess=(scores[ids]>=cut)&mask;tp=(guess&truth).sum();fp=(guess&~truth).sum();fn=(~guess&truth).sum()
        return 2*tp/max(2*tp+fp+fn,1)
    return float(max(cuts,key=f1))

def readout_metrics(y,scores,test,forward,cut):
    n=len(y);mask=~np.eye(n,dtype=bool)[test]
    valid=mask&((np.arange(n)[None,:]>test[:,None]) if forward else (np.arange(n)[None,:]<test[:,None]))
    aucs=[]
    for c in range(n):
        keep=valid[:,c];a=linear.auc(y[test[keep],c],scores[test[keep],c])
        if a is not None:aucs.append(a)
    truth=y[test].astype(bool)&mask;guess=(scores[test]>=cut)&mask
    tp=(truth&guess).sum();fp=(~truth&guess).sum();fn=(truth&~guess).sum();rec=[];chance=[]
    for i,t in enumerate(test):
        yy=y[t,valid[i]].astype(bool);ss=scores[t,valid[i]];k=int(yy.sum())
        if 0<k<len(yy):rec.append(float(yy[np.argsort(ss)[-k:]].mean()));chance.append(k/len(yy))
    return dict(auc=float(np.mean(aucs)) if aucs else None,candidates=len(aucs),test_nodes=len(test),
        f1=float(2*tp/max(2*tp+fp+fn,1)),precision=float(tp/max(tp+fp,1)),recall=float(tp/max(tp+fn,1)),
        exact_set=float((truth==guess).all(1).mean()),recall_at_true_k=float(np.mean(rec)) if rec else None,
        random_recall=float(np.mean(chance)) if chance else None,threshold=cut)

def prompt(p,s,t,copy_source=False):
    ids=p['header']+[p['tok'].id('<QUERY>'),int(p['nodes'][s]),int(p['nodes'][t]),p['tok'].id('|')]
    return ids+[int(p['nodes'][s])] if copy_source else ids

@torch.inference_mode()
def prefill(p,sequences,edited=None,position=None,cache=None):
    ids=torch.tensor(sequences,device=DEVICE);handles=[]
    if edited is not None:
        values=torch.as_tensor(edited,device=DEVICE)
        def hook(module,args,output):
            changed=output.clone();changed[:,position]=values;return changed
        handles=[p['model'].embed.register_forward_hook(hook)]
    try:return p['model'](ids,cache=cache,cache_mode='prefill' if cache is not None else None)[:,-1]
    finally:
        for handle in handles:handle.remove()

@torch.inference_mode()
def rollout(p,sequences,edited=None,position=None):
    results=[];model=p['model'];tok=p['tok'];n=len(p['nodes']);inv={int(v):i for i,v in enumerate(p['nodes'])}
    for start in range(0,len(sequences),128):
        seq=sequences[start:start+128];batch=len(seq);length=len(seq[0]);limit=n+3
        cache=KVCache(6,batch,1,length+limit+8,128,DEVICE,torch.float32)
        logits=prefill(p,seq,None if edited is None else edited[start:start+128],position,cache)
        active=np.ones(batch,bool);traces=[[] for _ in seq];answers=np.full(batch,-1,int)
        for step in range(limit):
            tokens=logits.argmax(-1).cpu().numpy();feed=np.full(batch,tok.pad,int)
            for i in np.flatnonzero(active):
                token=int(tokens[i]);feed[i]=token
                if token==tok.yes or token==tok.no:answers[i]=int(token==tok.yes);active[i]=False
                elif token==tok.id('<EOS>') or token==tok.pad:active[i]=False
                else:traces[i].append(inv.get(token,-1))
            if not active.any():break
            positions=torch.full((batch,1),length+step,device=DEVICE,dtype=torch.long)
            cache.valid_length=length+step+1
            logits=model(torch.tensor(feed[:,None],device=DEVICE),positions=positions,cache=cache,cache_mode='decode')[:,-1]
        for i in range(batch):
            trace=traces[i];legal=bool(trace and all(u>=0 for u in trace))
            if legal:legal=all(p['a'][u,v] for u,v in zip(trace,trace[1:]))
            results.append(dict(answer=int(answers[i]),first=trace[0] if trace else -1,last=trace[-1] if trace else -1,
                legal=legal,budget=bool(active[i]),trace=trace))
    return results

def pairs(p,test_nodes,role):
    # Original heldout query pairs, opposite labels and common partner.
    focus='target' if role=='ancestors' else 'source';partner='source' if role=='ancestors' else 'target'
    pool=[r for r in p['rows'] if r['split']==2 and r[focus] in test_nodes]
    grouped={}
    for row in pool:grouped.setdefault(row[partner],[]).append(row)
    candidates=[]
    for rows in grouped.values():
        for a in rows:
            if a['label']!=0:continue
            for b in rows:
                if b['label']==1:candidates.append((a,b))
    np.random.default_rng(881).shuffle(candidates);selected=[];uses={}
    for a,b in candidates:
        if uses.get(a['query_ids'],0)>=3 or uses.get(b['query_ids'],0)>=3:continue
        selected.extend([(a,b),(b,a)])
        for row in (a,b):uses[row['query_ids']]=uses.get(row['query_ids'],0)+1
        if len(selected)>=128:break
    return selected

def knn(p,edited,training,relation):
    difference=edited[:,None,:]-p['embedding'][training][None,:,:]
    nearest=np.argsort((difference*difference).sum(-1),axis=1)[:,:5]
    return relation[training[nearest]].mean(1)

def edit_summary(p,panel,role,edited,native,baseline,reader,relation,training,constructor_reader):
    focus='target' if role=='ancestors' else 'source';partner='source' if role=='ancestors' else 'target'
    a=np.array([r[0][focus] for r in panel]);b=np.array([r[1][focus] for r in panel]);other=np.array([r[0][partner] for r in panel])
    original_labels=np.array([r[0]['label'] for r in panel]);desired=1-original_labels
    before=np.array([r['answer'] for r in baseline]);after=np.array([r['answer'] for r in native]);valid=after>=0
    # Natural donor results were computed before any edit, not chosen by outcome.
    donor_answers=np.array([p['baseline'][r[1]['query_ids']]['answer'] for r in panel])
    eligible=(before==original_labels)&(donor_answers==desired)
    w,bias,cut=reader;semantic=edited@w+bias;truth=relation[b].astype(bool)
    semantic_before=p['embedding'][a]@w+bias
    cw,cb,ccut=constructor_reader;constructed=edited@cw+cb
    masks=np.ones_like(truth,bool)
    for ids in (a,b,other):masks[np.arange(len(panel)),ids]=False
    guess=(semantic>=cut)&masks;truth&=masks
    cosine=edited@p['embedding'].T/(np.linalg.norm(edited,axis=1,keepdims=True)*np.linalg.norm(p['embedding'],axis=1)[None,:])
    identity=cosine.argmax(-1);ks=knn(p,edited,training,relation)
    result={}
    for name,keep in [('all',np.ones(len(panel),bool)),('baseline_both_correct',eligible),
                     ('positive_to_negative',eligible&(original_labels==1)),('negative_to_positive',eligible&(original_labels==0))]:
        if not keep.any():result[name]=dict(n=0);continue
        tp=(guess[keep]&truth[keep]).sum();fp=(guess[keep]&~truth[keep]).sum();fn=(~guess[keep]&truth[keep]).sum()
        result[name]=dict(n=int(keep.sum()),counterfactual_answer_success=float((valid&(after==desired))[keep].mean()),
            original_answer_accuracy=float((valid&(after==original_labels))[keep].mean()),valid_answer_rate=float(valid[keep].mean()),
            first_node_original_source=float(np.mean([native[i]['first']==panel[i][0]['source'] for i in np.flatnonzero(keep)])),
            first_node_donor_source=float(np.mean([native[i]['first']==panel[i][1]['source'] for i in np.flatnonzero(keep)])),
            full_original_witness=float(np.mean([native[i]['legal'] and native[i]['first']==panel[i][0]['source'] and native[i]['last']==panel[i][0]['target'] for i in np.flatnonzero(keep)])),
            full_donor_witness=float(np.mean([native[i]['legal'] and native[i]['first']==panel[i][1]['source'] and native[i]['last']==panel[i][1]['target'] for i in np.flatnonzero(keep)])),
            trajectory_edges_legal=float(np.mean([native[i]['legal'] for i in np.flatnonzero(keep)])),
            nearest_embedding_identity_retained=float((identity==a)[keep].mean()),nearest_embedding_donor_adopted=float((identity==b)[keep].mean()),
            audit_relation_donor_f1=float(2*tp/max(2*tp+fp+fn,1)),
            constructor_query_membership_counterfactual=float(((constructed[np.arange(len(panel)),other]>=ccut)==desired)[keep].mean()),
            audit_query_membership_original_before=float(((semantic_before[np.arange(len(panel)),other]>=cut)==original_labels)[keep].mean()),
            audit_query_membership_counterfactual=float(((semantic[np.arange(len(panel)),other]>=cut)==desired)[keep].mean()),
            knn_query_membership_counterfactual=float(((ks[np.arange(len(panel)),other]>=.5)==desired)[keep].mean()),
            mean_edit_norm=float(np.linalg.norm(edited-p['embedding'][a],axis=1)[keep].mean()),
            relative_edit_norm=float((np.linalg.norm(edited-p['embedding'][a],axis=1)/np.linalg.norm(p['embedding'][a],axis=1))[keep].mean()),
            mean_norm_ratio=float((np.linalg.norm(edited,axis=1)/np.linalg.norm(p['embedding'][a],axis=1))[keep].mean()))
    return result

def matched(delta,reference):
    result=delta*np.linalg.norm(reference,axis=1,keepdims=True)/np.maximum(np.linalg.norm(delta,axis=1,keepdims=True),1e-12)
    assert np.allclose(np.linalg.norm(result,axis=1),np.linalg.norm(reference,axis=1),rtol=1e-5,atol=1e-5)
    return result.astype(np.float32)

def variants(p,panel,role,basis,shuffled,pca,constructor_reader=None):
    focus='target' if role=='ancestors' else 'source';a=np.array([r[0][focus] for r in panel]);b=np.array([r[1][focus] for r in panel])
    original=p['embedding'][a];donor=p['embedding'][b];difference=donor-original
    yield 'no_op',0,0,original
    yield 'full_swap',0,0,donor
    for rank in RANKS:
        u=basis[:,:rank];delta=difference@u@u.T
        yield 'semantic_swap',rank,0,original+delta
        yield 'complement_swap',rank,0,donor-delta
        yield 'PCA_equal_norm',rank,0,original+matched(difference@pca[:,:rank]@pca[:,:rank].T,delta)
        for seed in CONTROLS:
            random=np.linalg.qr(np.random.default_rng(seed).normal(size=(128,128)))[0].astype(np.float32)[:,:rank]
            yield 'random_equal_norm',rank,seed,original+matched(difference@random@random.T,delta)
            u=shuffled[seed][:,:rank]
            yield 'shuffled_equal_norm',rank,seed,original+matched(difference@u@u.T,delta)
        if constructor_reader is not None:
            # Minimum-norm single-fact edit constrained to the SHARED rank-r
            # semantic subspace. Target is val-F1 threshold +/- .5. Oracle
            # desired label defines the edit objective, not a test solver.
            w,bias,cut=constructor_reader;partner='source' if role=='ancestors' else 'target'
            candidate=np.array([r[0][partner] for r in panel]);wanted=np.array([r[1]['label'] for r in panel])
            gradient=w[:,candidate].T;projected=gradient@basis[:,:rank]@basis[:,:rank].T
            score=(original*gradient).sum(1)+bias[candidate]
            norm=(projected*projected).sum(1);supported=norm>1e-12
            amount=np.where(supported,(cut+(2*wanted-1)*.5-score)/np.maximum(norm,1e-12),0)
            targeted=(amount[:,None]*projected).astype(np.float32)
            # Bound out-of-distribution perturbations; report whether the
            # constructor objective is achieved after this predeclared cap.
            cap=np.linalg.norm(original,axis=1,keepdims=True)
            targeted*=np.minimum(1,cap/np.maximum(np.linalg.norm(targeted,axis=1,keepdims=True),1e-12))
            yield 'targeted_semantic',rank,0,original+targeted
            v=pca[:,:rank]
            yield 'targeted_PCA_equal_norm',rank,0,original+matched(difference@v@v.T,targeted)
            for seed in CONTROLS:
                v=np.linalg.qr(np.random.default_rng(seed).normal(size=(128,128)))[0].astype(np.float32)[:,:rank]
                yield 'targeted_random_equal_norm',rank,seed,original+matched(difference@v@v.T,targeted)
                v=shuffled[seed][:,:rank]
                yield 'targeted_shuffled_equal_norm',rank,seed,original+matched(difference@v@v.T,targeted)

@torch.inference_mode()
def branch_scores(p,panel,edited):
    # Both goals are reachable, but their useful successor sets are disjoint.
    sequences=[prompt(p,r[0]['source'],r[0]['target'],True) for r in panel]
    position=len(p['header'])+2;out=[]
    for start in range(0,len(panel),128):
        scores=prefill(p,sequences[start:start+128],edited[start:start+128],position)
        probabilities=scores.softmax(-1).cpu().numpy();pred=scores.argmax(-1).cpu().numpy()
        for j,(a,b) in enumerate(panel[start:start+128]):
            u=a['source'];adj=p['a'][u];good=adj&p['reach'][:,b['target']];old_good=adj&p['reach'][:,a['target']]
            mass=probabilities[j,p['nodes'][adj]].sum();node=int(pred[j]);chosen=np.flatnonzero(p['nodes']==node)
            v=int(chosen[0]) if len(chosen) else -1
            out.append(dict(donor_useful_probability=float(probabilities[j,p['nodes'][good]].sum()),
                donor_useful_given_successor=float(probabilities[j,p['nodes'][good]].sum()/max(mass,1e-30)),
                donor_useful_argmax=bool(v>=0 and good[v]),original_useful_argmax=bool(v>=0 and old_good[v]),
                edge_legal=bool(v>=0 and adj[v])))
    return {key:float(np.mean([r[key] for r in out])) for key in out[0]}

def run(setting):
    p=base.load(setting,'path','cpu');n=len(p['nodes']);p['embedding']=p['model'].embed.weight.detach().numpy()[p['nodes']].copy()
    saved_embedding=p['model'].embed.weight.detach().clone();p['model'].to(DEVICE)
    relations={'ancestors':p['reach'].T.copy(),'descendants':p['reach'].copy(),'successors':p['a'].copy()}
    for value in relations.values():np.fill_diagonal(value,False)
    x=p['embedding'];test_rows=[r for r in p['rows'] if r['split']==2]
    baseline=rollout(p,[prompt(p,r['source'],r['target']) for r in test_rows])
    p['baseline']={r['query_ids']:v for r,v in zip(test_rows,baseline)}
    result=dict(analysis_version=2,manifest=p['manifest'],ranks=RANKS,probes={},causal={},branching=[],examples=[],
        method='Pure node embeddings. Probes use 60/20/20 node holdout. Causal directions use first30 train nodes, audit uses other30; val20 selects ridge; both original/donor focal nodes are test20. '
               'Main causal queries both original query-test. Frozen GPU2 FP32 rollouts; fits GPU0. Only one input embedding occurrence is replaced; tied weights untouched.',
        baseline=dict(test_queries=len(test_rows),FP32_vs_saved_answer_agreement=float(np.mean([v['answer']==r['pred'] for r,v in zip(test_rows,baseline)]))))
    for node_seed in SEEDS:
        order=np.random.default_rng(node_seed).permutation(n);training,val,test=order[:60],order[60:80],order[80:]
        constructor,audit=training[:30],training[30:]
        for role,ybool in relations.items():
            y=ybool.astype(np.float32);w,b,reg=fit(x,y,training,val);scores=x@w+b;cut=threshold(scores,y,val)
            metric=readout_metrics(y,scores,test,role!='ancestors',cut)
            controls=[]
            for seed in CONTROLS:
                random=np.random.default_rng(seed).normal(size=x.shape).astype(np.float32)
                random=random*x[training].std(0)+x[training].mean(0)
                rw,rb,_=fit(random,y,training,val);pred=random@rw+rb
                controls.append(readout_metrics(y,pred,test,role!='ancestors',threshold(pred,y,val)))
            result['probes'].setdefault(role,[]).append(dict(node_seed=node_seed,actual=metric,random=controls,ridge=reg))
            if role=='successors':continue
            panel=pairs(p,set(test.tolist()),role)
            if not panel:result['causal'].setdefault(role,[]).append(dict(node_seed=node_seed,n=0,curves=[]));continue
            w,b,reg=fit(x,y,constructor,val);u,singular,_=np.linalg.svd(w,full_matrices=False)
            constructor_reader=(w,b,threshold(x@w+b,y,val))
            aw,ab,_=fit(x,y,audit,val);reader=(aw,ab,threshold(x@aw+ab,y,val))
            shuffled={}
            for seed in CONTROLS:
                rng=np.random.default_rng(seed);sy=y.copy();sy[constructor]=y[rng.permutation(constructor)];sy[val]=y[rng.permutation(val)]
                sw,_,_=fit(x,sy,constructor,val);shuffled[seed]=np.linalg.svd(sw,full_matrices=False)[0]
            pca=np.linalg.svd(x[constructor]-x[constructor].mean(0),full_matrices=False)[2].T
            focus='target' if role=='ancestors' else 'source';position=len(p['header'])+(2 if role=='ancestors' else 1)
            sequences=[prompt(p,a['source'],a['target']) for a,b in panel]
            base_outputs=[p['baseline'][a['query_ids']] for a,b in panel]
            entry=dict(node_seed=node_seed,n=len(panel),unique_queries=len({r['query_ids'] for pair in panel for r in pair}),
                focus_test_nodes=len({r[focus] for pair in panel for r in pair}),constructor_nodes=len(constructor),audit_nodes=len(audit),
                singular_values=singular.tolist(),curves=[])
            for method,rank,seed,edited in variants(p,panel,role,u,shuffled,pca,constructor_reader):
                edited=edited.astype(np.float32)
                generated=rollout(p,sequences,edited,position)
                metrics=edit_summary(p,panel,role,edited,generated,base_outputs,reader,y,training,constructor_reader)
                entry['curves'].append(dict(method=method,rank=rank,seed=seed,metrics=metrics))
                if method=='no_op':assert all(v['answer']==b['answer'] for v,b in zip(generated,base_outputs))
                if method=='full_swap':
                    donor_seq=[prompt(p,b['source'],b['target']) for a,b in panel]
                    logits=prefill(p,sequences[:16],edited[:16],position);natural=prefill(p,donor_seq[:16])
                    error=float((logits-natural).abs().max());assert error<1e-4
                    entry['full_swap_prefill_error']=error
                if method=='semantic_swap' and rank==8:
                    for i in np.flatnonzero([v['answer']!=b['answer'] for v,b in zip(generated,base_outputs)])[:2]:
                        a,b=panel[i]
                        def name(v):return p['tok'].name(int(p['nodes'][v]))
                        result['examples'].append(dict(node_seed=node_seed,role=role,source=name(a['source']),target=name(a['target']),
                            donor_source=name(b['source']),donor_target=name(b['target']),original_label=a['label'],desired_label=b['label'],
                            edited_answer=generated[i]['answer'],trace=[name(v) if v>=0 else 'INVALID' for v in generated[i]['trace']]))
            result['causal'].setdefault(role,[]).append(entry)
            # Secondary ancestor steering: recipient is query-test, donor goal
            # probe-test; counterfactual query need not be main-model query-test.
            if role=='ancestors':
                branching=[]
                for a in test_rows:
                    if not a['label'] or a['target'] not in test:continue
                    adj=p['a'][a['source']];old_good=adj&p['reach'][:,a['target']]
                    if not old_good.any():continue
                    choices=[]
                    for goal in test:
                        good=adj&p['reach'][:,goal]
                        if good.any() and not (good&old_good).any():choices.append(int(goal))
                    for goal in choices[:2]:branching.append((a,dict(source=a['source'],target=goal,label=1)))
                    if len(branching)>=96:break
                splits={(r['source'],r['target']):r['split'] for r in p['rows']}
                branch=dict(node_seed=node_seed,n=len(branching),unique_recipient_queries=len({a['query_ids'] for a,b in branching}),
                    unique_goals=len({r['target'] for pair in branching for r in pair}),
                    counterfactual_query_split_counts={str(s):sum(splits.get((b['source'],b['target']),-1)==s for a,b in branching) for s in (-1,0,1,2)},
                    curves=[],note='Recipient original query-test; BOTH goals probe-test. Counterfactual query may be main-model train/val. Prefix includes copied source only.')
                if branching:
                    for method,rank,seed,edited in variants(p,branching,role,u,shuffled,pca):
                        score=branch_scores(p,branching,edited.astype(np.float32))
                        branch['curves'].append(dict(method=method,rank=rank,seed=seed,metrics=score))
                result['branching'].append(branch)
            print(setting,node_seed,role,'COMPLETE',len(panel),flush=True)
        base.write(OUT/f'{setting}.json',result)
    assert torch.equal(p['model'].embed.weight.cpu(),saved_embedding)
    assert p['model'].embed.weight.data_ptr()==p['model'].lm_head.weight.data_ptr()
    result['checks']=dict(embedding_and_tied_head_unchanged=True,all_focal_test_nodes_excluded_from_fit_and_validation=True,
        closed_form_direction_audit_training_nodes_disjoint=True,graph_reachability_verified_by_BFS=True)
    base.write(OUT/f'{setting}.json',result);print(setting,'COMPLETE',flush=True)

if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    for setting in ('simple','hard'):run(setting)

"""Probe-free native-head successor score intervention, plus probe-edit audit.

Work in final_norm standardized coordinates. Project donor-recipient difference
onto columns of the ORIGINAL node output head for the union of old/new true
successors. No trained diagnostic decoder defines these directions. Restore
recipient mean and variance. Random sphere / unrelated-node controls preserve
the edit angle, hence per-case L2 norm, with 3 seeds each. These controls still
do not guarantee semantic disentanglement or unchanged node identity.
"""
from successor_causal import *

def native_scores(pack,h):
    with torch.inference_mode():
        tensor=torch.tensor(h,device=next(pack['model'].parameters()).device)
        logits=pack['model'].lm_head(pack['model'].final_norm(tensor))
        return logits[:,pack['nodes']].cpu().numpy(),logits.softmax(-1)[:,pack['nodes']].cpu().numpy()

def head_metric(score,prob,labels):
    rows=[];exact=[];mass=[]
    for s,p,y in zip(score,prob,labels):
        ids=np.flatnonzero(y);k=len(ids)
        if not k:continue
        best=np.argsort(s)[-k:];hit=np.isin(best,ids).mean()
        rows.append(hit);exact.append(hit==1);mass.append(float(p[ids].sum()))
    return dict(n_nonempty=len(rows),recall_at_true_k=float(np.mean(rows)) if rows else None,
        exact_topk=float(np.mean(exact)) if rows else None,
        full_vocab_relation_mass=float(np.mean(mass)) if rows else None)

def basis(columns):
    if columns.shape[1]==0:return np.zeros((columns.shape[0],0))
    # LayerNorm annihilates the constant coordinate; keep directions centered.
    columns=columns-columns.mean(0,keepdims=True)
    u,s,_=np.linalg.svd(columns,full_matrices=False)
    return u[:,s>s[0]*1e-8]

def run(setting):
    pack=load(setting,device='cpu');m=pack['m'];n=len(pack['nodes']);sp=[np.flatnonzero(m['split']==i) for i in range(3)]
    panel=squares(pack,2);a,b,d,c=panel['indices'].T;eligible=panel['all_four_correct']
    output_weight=pack['model'].lm_head.weight.detach().cpu().numpy()[pack['nodes']]
    gamma=pack['model'].final_norm.weight.detach().cpu().numpy();eps=pack['model'].final_norm.eps
    sensitivity=gamma[:,None]*output_weight.T
    result=dict(layer=1,roles={},method='native output head, actual successor candidate union, standardized-coordinate transplant',
        caveat='Selects original head directions associated with successor output tokens; can also affect identity or other graph features. Not a semantic-isolation guarantee.')
    for role,name in ((1,'target'),(0,'source')):
        focal=m[name];donor=b if role==1 else c;position=-2 if role==1 else -3
        base=pack['x'][a,role,1];natural=pack['x'][donor,role,1]
        old_set=pack['adjacency'][focal[a]].astype(bool);new_set=pack['adjacency'][focal[donor]].astype(bool)
        equal=(old_set==new_set).all(1);groups=dict(all_four_correct=eligible,
            different_successor_sets=eligible&~equal,same_successor_sets=eligible&equal,
            different_nonempty_new_successors=eligible&~equal&(new_set.sum(1)>0))
        original_identity=np.eye(n,dtype=np.float32)[focal]
        iw,ib,_=coefficient_fit(pack['x'][:,role,1:2],original_identity,sp)
        old_id=(base@iw[0]+ib[0]).argmax(-1)
        records=[];examples=[]
        def measure(method,seed,edited):
            margin,valid=forward_patch(pack,a,1,edited.astype(np.float32),position=position)
            ns,np_=native_scores(pack,edited.astype(np.float32));id_new=(edited@iw[0]+ib[0]).argmax(-1)
            rec=dict(method=method,seed=seed,mean_edit_norm=float(np.linalg.norm(edited-base,axis=1).mean()))
            for group,mask in groups.items():
                rec[group]=dict(n=int(mask.sum()),counterfactual_accuracy=float(((((margin>0)==m['label'][donor])&valid)[mask]).mean()) if mask.any() else None,
                    valid_answer_token_fraction=float(valid[mask].mean()) if mask.any() else None,
                    native_head_new_successors=head_metric(ns[mask],np_[mask],new_set[mask]),
                    native_head_old_successors=head_metric(ns[mask],np_[mask],old_set[mask]),
                    original_identity_retained=float((id_new[mask]==focal[a[mask]]).mean()) if mask.any() else None,
                    donor_identity_adopted=float((id_new[mask]==focal[donor[mask]]).mean()) if mask.any() else None)
            if method=='native_successor_union':
                original_score,_=native_scores(pack,base)
                recall=np.full(len(a),np.nan);original_recall=recall.copy()
                for j in range(len(a)):
                    k=int(new_set[j].sum())
                    if not k:continue
                    recall[j]=new_set[j,np.argsort(ns[j])[-k:]].mean()
                    original_recall[j]=new_set[j,np.argsort(original_score[j])[-k:]].mean()
                strong=eligible&~equal&(recall>=.8)&(original_recall<=.2)
                rec['strong_native_head_change']=dict(n=int(strong.sum()),
                    counterfactual_accuracy=float(((((margin>0)==m['label'][donor])&valid)[strong]).mean()) if strong.any() else None,
                    note='Descriptive post-edit quality subset, not a fair method-comparison cohort.')
                for desired_behavior in (False,True):
                    mask=strong&((((margin>0)==m['label'][donor])&valid)==desired_behavior)
                    for j in np.flatnonzero(mask)[:2]:
                        def name_node(v):return pack['tok'].name(pack['nodes'][int(v)])
                        examples.append(dict(source=name_node(m['source'][a[j]]),target=name_node(m['target'][a[j]]),
                            donor_source=name_node(m['source'][donor[j]]),donor_target=name_node(m['target'][donor[j]]),
                            original_query_id=int(m['query_ids'][a[j]]),donor_query_id=int(m['query_ids'][donor[j]]),
                            original_label=int(m['label'][a[j]]),desired_label=int(m['label'][donor[j]]),
                            edited_answer=int(margin[j]>0) if valid[j] else None,
                            old_successors=[name_node(v) for v in np.flatnonzero(old_set[j])],
                            new_successors=[name_node(v) for v in np.flatnonzero(new_set[j])],
                            new_successor_recall_before=float(original_recall[j]),new_successor_recall_after=float(recall[j])))
            records.append(rec)
        measure('no_op',0,base);measure('full_swap',0,natural)
        # Audit the previous diagnostic successor-probe rank64 edit with the
        # ORIGINAL output head. It may change a probe without changing head.
        w,_,_=coefficient_fit(pack['x'][:,role,1:2],pack['adjacency'][focal],sp)
        u=projection(w[0])[:,:64];probe_edit=base+(natural-base)@u@u.T
        measure('probe_rank64',0,probe_edit)
        mean=base.mean(1,keepdims=True);std=np.sqrt(base.var(1,keepdims=True)+eps)
        xb=(base-mean)/std
        xd=(natural-natural.mean(1,keepdims=True))/np.sqrt(natural.var(1,keepdims=True)+eps)
        xb=xb.astype(np.float64);xd=xd.astype(np.float64)
        native=[];unions=[];rank=[]
        for i in range(len(a)):
            nodes=np.flatnonzero(old_set[i]|new_set[i]);unions.append(nodes)
            uu=basis(sensitivity[:,nodes].astype(np.float64));rank.append(uu.shape[1])
            zz=xb[i]+(xd[i]-xb[i])@uu@uu.T
            zz-=zz.mean();zz*=np.linalg.norm(xb[i])/max(np.linalg.norm(zz),1e-30)
            native.append(zz)
        native=np.asarray(native);native_edit=(mean+std*native).astype(np.float32)
        measure('native_successor_union',0,native_edit)
        radii=np.linalg.norm(xb,axis=1);cosine=np.clip((native*xb).sum(1)/(radii*radii),-1,1)
        sine=np.sqrt(np.maximum(1-cosine*cosine,0))
        for seed in SEEDS:
            rng=np.random.default_rng(seed)
            for method in ('random_sphere_equal_norm','unrelated_output_nodes_equal_norm'):
                random=[]
                for i in range(len(a)):
                    if method.startswith('random'):
                        direction=rng.normal(size=xb.shape[1])
                    else:
                        pool=np.setdiff1d(np.arange(n),unions[i]);k=min(len(unions[i]),len(pool))
                        chosen=rng.choice(pool,k,replace=False)
                        uu=basis(sensitivity[:,chosen].astype(np.float64))
                        direction=(xd[i]-xb[i])@uu@uu.T
                    direction-=direction.mean();direction-=np.dot(direction,xb[i])/(radii[i]**2)*xb[i]
                    if np.linalg.norm(direction)<1e-20:
                        direction=rng.normal(size=xb.shape[1]);direction-=direction.mean()
                        direction-=np.dot(direction,xb[i])/(radii[i]**2)*xb[i]
                    direction*=radii[i]/np.linalg.norm(direction)
                    random.append(cosine[i]*xb[i]+sine[i]*direction)
                edited=(mean+std*np.asarray(random)).astype(np.float32)
                assert np.allclose(np.linalg.norm(edited-base,axis=1),np.linalg.norm(native_edit-base,axis=1),atol=2e-4,rtol=2e-5)
                measure(method,seed,edited)
        assert np.allclose(native_edit.mean(1),base.mean(1),atol=2e-5,rtol=2e-5)
        assert np.allclose(native_edit.var(1),base.var(1),atol=2e-4,rtol=2e-5)
        result['roles'][name]=dict(curves=records,union_rank_min=int(min(rank)),union_rank_max=int(max(rank)),examples=examples,
            norm_and_moment_checks_passed=True,native_identity_baseline_accuracy=float((old_id[eligible]==focal[a[eligible]]).mean()))
        print(setting,name,'native control complete',flush=True)
    write(OUT/f'{setting}_native.json',result);del pack;torch.cuda.empty_cache()

if __name__=='__main__':
    for setting in ('simple','hard'):run(setting)

"""Causal interchange of successor-decoder directions at source/target L1.

Target: same-source donor b=(sa,tb), alternative = ancestors(target).
Source: same-target donor c=(sb,ta), alternative = descendants(source).
All query inputs stay unchanged; only ONE residual position after block1 is
patched. Downstream recomputed. Native full-token swap is a positive control.
Probe subspaces are supervised diagnostics, NOT semantic guarantees.
CPU frozen forward avoids busy training GPUs; all ridge fits on GPU0.
"""
from study import *
OUT=HERE/'successor_causal';OUT.mkdir(exist_ok=True)
SEEDS=(201,303,407)

def projection(w):return np.linalg.svd(w,full_matrices=False)[0]
def norm_match(delta,reference):
    out=delta*np.linalg.norm(reference,axis=1,keepdims=True)/np.maximum(np.linalg.norm(delta,axis=1,keepdims=True),1e-12)
    assert np.allclose(np.linalg.norm(out,axis=1),np.linalg.norm(reference,axis=1),rtol=1e-5,atol=1e-5)
    return out.astype(np.float32)

def threshold(pred,y,ids):
    candidates=np.unique(np.r_[np.linspace(-.1,1.1,61),np.quantile(pred[ids],np.linspace(0,1,41))])
    def f1(t):
        a=pred[ids]>=t;b=y[ids].astype(bool);tp=(a&b).sum();fp=(a&~b).sum();fn=(~a&b).sum()
        return 2*tp/max(2*tp+fp+fn,1)
    return float(max(candidates,key=f1))

def stat(pack,panel,donor,role,margin,valid,edited,readers,full_success):
    m=pack['m'];a=panel['indices'][:,0];focal=m['target' if role==1 else 'source']
    other=m['source' if role==1 else 'target'];desired=m['label'][donor]
    base=pack['x'][a,role,1];natural=pack['x'][donor,role,1]
    w=readers['successor_w'];b=readers['successor_b'];t=readers['threshold']
    before=base@w+b;after=edited@w+b;wanted=natural@w+b
    ds=wanted-before;den=(ds*ds).sum(1);has=den>1e-10
    progress=((after-before)*ds).sum(1)/np.maximum(den,1e-30)
    true_new=pack['adjacency'][focal[donor]].astype(bool);new_readout=after>=t
    exact=(new_readout==true_new).all(1)
    equal_sets=np.all(pack['adjacency'][focal[a]]==pack['adjacency'][focal[donor]],axis=1)
    if role==0:assert not equal_sets.any()  # same successors => same reach to these later goals
    original_id=(base@readers['focal_w']+readers['focal_b']).argmax(-1)
    new_id=(edited@readers['focal_w']+readers['focal_b']).argmax(-1)
    old_other=(base@readers['other_w']+readers['other_b']).argmax(-1)
    new_other=(edited@readers['other_w']+readers['other_b']).argmax(-1)
    eligible=panel['all_four_correct'];identity_baseline=eligible&(original_id==focal[a])
    groups=dict(all=np.ones(len(a),bool),all_four_correct=eligible,
        equal_successor_sets=eligible&equal_sets,different_successor_sets=eligible&~equal_sets,
        full_swap_sensitive=eligible&full_success,matched_depth_band=eligible&panel['matched'],
        all_four_heldout=eligible&panel['all_four_heldout'])
    result=dict(mean_edit_norm=float(np.linalg.norm(edited-base,axis=1).mean()),
        focal_identity=dict(baseline_accuracy=float((original_id[eligible]==focal[a[eligible]]).mean()),
            original_retained=float((new_id[eligible]==focal[a[eligible]]).mean()),
            donor_adopted=float((new_id[eligible]==focal[donor[eligible]]).mean()),
            baseline_correct_n=int(identity_baseline.sum()),
            retained_from_baseline_n=int((identity_baseline&(new_id==focal[a])).sum())),
        other_endpoint_identity=dict(baseline_accuracy=float((old_other[eligible]==other[a[eligible]]).mean()),
            edited_accuracy=float((new_other[eligible]==other[a[eligible]]).mean()),
            note='Diagnostic only: a weak identity decoder cannot establish preservation of all endpoint information.'))
    for name,mask in groups.items():
        h=mask&has;tp=int((new_readout[mask]&true_new[mask]).sum());fp=int((new_readout[mask]&~true_new[mask]).sum());fn=int((~new_readout[mask]&true_new[mask]).sum())
        result[name]=dict(n=int(mask.sum()),counterfactual_accuracy=float((((margin>0)==desired)&valid)[mask].mean()) if mask.any() else None,
            desired_margin_shift=float(((margin-pack['margin'][a])*(2*desired-1))[mask].mean()) if mask.any() else None,
            valid_answer_token_fraction=float(valid[mask].mean()) if mask.any() else None,
            successor_newset_exact=float(exact[mask].mean()) if mask.any() else None,
            successor_newset_f1=2*tp/max(2*tp+fp+fn,1),
            reader_interchange_progress=float(progress[h].mean()) if h.any() else None,
            reader_progress_n=int(h.sum()))
    return result

def run(setting):
    pack=load(setting,device='cpu');m=pack['m'];n=len(pack['nodes']);sp=[np.flatnonzero(m['split']==i) for i in range(3)]
    panels={name:squares(pack,i) for name,i in [('val',1),('test',2)]}
    out=dict(manifest=pack['manifest'],layer=1,ranks=list(RANKS),roles={},checks={},
        inference_device='CPU FP32 native SDPA',probe_fit_device='GPU0',
        caveat='Oracle graph-label affine decoder SVD directions, not guaranteed to isolate successor semantics; identity and other structure may change.')
    examples=[]
    for role,name in ((1,'target'),(0,'source')):
        focal=m[name];other=m['source' if name=='target' else 'target'];x=pack['x'][:,role,1:2]
        ys=pack['adjacency'][focal]
        ya=(pack['reach'].T if role==1 else pack['reach'])[focal].astype(np.float32)
        sw,sb,sreg=coefficient_fit(x,ys,sp);aw,ab,areg=coefficient_fit(x,ya,sp)
        fw,fb,_=coefficient_fit(x,np.eye(n,dtype=np.float32)[focal],sp)
        ow,ob,_=coefficient_fit(x,np.eye(n,dtype=np.float32)[other],sp)
        pred=x[:,0]@sw[0]+sb[0];cut=threshold(pred,ys,sp[1])
        readers=dict(successor_w=sw[0],successor_b=sb[0],focal_w=fw[0],focal_b=fb[0],
            other_w=ow[0],other_b=ob[0],threshold=cut)
        su=projection(sw[0]);au=projection(aw[0]);random_u={};shuffle_u={}
        for seed in SEEDS:
            random_u[seed]=np.linalg.qr(np.random.default_rng(seed).normal(size=(x.shape[-1],x.shape[-1])))[0].astype(np.float32)
            perm=np.random.default_rng(seed).permutation(n);labels=pack['adjacency'][perm[focal]]
            ww,_,_=coefficient_fit(x,labels,sp);shuffle_u[seed]=projection(ww[0])
        result=dict(alternative_relation='ancestors' if role==1 else 'descendants',curves=[],panels={},
            successor_probe_ridge=sreg,alternative_probe_ridge=areg,successor_threshold=cut)
        for splitname,panel in panels.items():
            a,b,d,c=panel['indices'].T;donor=b if role==1 else c
            assert np.array_equal(other[a],other[donor])
            base=pack['x'][a,role,1];natural=pack['x'][donor,role,1];difference=natural-base;position=-2 if role==1 else -3
            full_margin,full_valid=forward_patch(pack,a,1,natural,position=position)
            full_success=((full_margin>0)==m['label'][donor])&full_valid
            result['panels'][splitname]=panel_info(pack,panel)
            eligible=panel['all_four_correct'];equal=np.all(pack['adjacency'][focal[a]]==pack['adjacency'][focal[donor]],axis=1)
            result['panels'][splitname]['equal_successor_sets_correct']=int((eligible&equal).sum())
            for method,edited,margin,valid in [('full_swap',natural,full_margin,full_valid),
                ('no_op',base,pack['margin'][a],m['native_pred'][a]>=0)]:
                result['curves'].append(dict(split=splitname,method=method,rank=0,seed=0,
                    result=stat(pack,panel,donor,role,margin,valid,edited,readers,full_success)))
            for rank in RANKS:
                delta=difference@su[:,:rank]@su[:,:rank].T
                alt=difference@au[:,:rank]@au[:,:rank].T
                variants=[('successor',0,base+delta),('successor_complement',0,natural-delta),
                    ('alternative_raw',0,base+alt),('alternative_equal_norm',0,base+norm_match(alt,delta))]
                for seed in SEEDS:
                    for method,basis in [('random_equal_norm',random_u[seed]),('shuffled_equal_norm',shuffle_u[seed])]:
                        dd=difference@basis[:,:rank]@basis[:,:rank].T
                        variants.append((method,seed,base+norm_match(dd,delta)))
                for method,seed,edited in variants:
                    edited=edited.astype(np.float32);margin,valid=forward_patch(pack,a,1,edited,position=position)
                    result['curves'].append(dict(split=splitname,method=method,rank=rank,seed=seed,
                        result=stat(pack,panel,donor,role,margin,valid,edited,readers,full_success)))
                    if splitname=='test' and rank==32 and method=='successor':
                        for j in np.flatnonzero(eligible&full_success)[:3]:
                            def node(u):return pack['tok'].name(pack['nodes'][int(u)])
                            examples.append(dict(role=name,source=node(m['source'][a[j]]),target=node(m['target'][a[j]]),
                                donor_source=node(m['source'][donor[j]]),donor_target=node(m['target'][donor[j]]),
                                original_query_id=int(m['query_ids'][a[j]]),donor_query_id=int(m['query_ids'][donor[j]]),
                                native_label=int(m['label'][a[j]]),desired_label=int(m['label'][donor[j]]),
                                edited_answer=int(margin[j]>0) if valid[j] else None,
                                old_successors=[node(v) for v in np.flatnonzero(pack['adjacency'][focal[a[j]]])],
                                new_successors=[node(v) for v in np.flatnonzero(pack['adjacency'][focal[donor[j]]])]))
            if splitname=='test' and role==1:
                for j in np.flatnonzero(eligible&equal)[:3]:
                    def node(u):return pack['tok'].name(pack['nodes'][int(u)])
                    examples.append(dict(type='identical_successor_sets_different_reach',source=node(m['source'][a[j]]),
                        old_target=node(focal[a[j]]),new_target=node(focal[donor[j]]),
                        successors=[node(v) for v in np.flatnonzero(pack['adjacency'][focal[a[j]]])],
                        old_label=int(m['label'][a[j]]),new_label=int(m['label'][donor[j]]),full_swap_answer=int(full_margin[j]>0)))
            print(setting,name,splitname,'complete',flush=True)
        # Predefined val objective; do not choose primary rank by test results.
        candidates=[]
        for rank in RANKS:
            rows=[q for q in result['curves'] if q['split']=='val' and q['rank']==rank]
            main=next(q for q in rows if q['method']=='successor')['result']['all_four_correct']['counterfactual_accuracy']
            controls=[q['result']['all_four_correct']['counterfactual_accuracy'] for q in rows if q['method'] in ('random_equal_norm','shuffled_equal_norm')]
            candidates.append((main-np.mean(controls),-rank,rank))
        result['validation_selected_rank']=max(candidates)[2]
        result['validation_criterion']='successor counterfactual success minus mean random/shuffled equal-norm success, tie favors smaller rank'
        out['roles'][name]=result
    out['examples']=examples
    write(OUT/f'{setting}.json',out);del pack;torch.cuda.empty_cache();print(setting,'DONE',flush=True)

if __name__=='__main__':
    for setting in sys.argv[1:] or ('simple','hard'):run(setting)

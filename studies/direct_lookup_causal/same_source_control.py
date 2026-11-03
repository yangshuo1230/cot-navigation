"""Control source-context contamination in target-subspace interchange.

Only target L1, same original validation/test crossed panels. Donor b=(sa,tb)
shares the exact source with recipient a=(sa,ta), unlike earlier donor d.
Original source token precedes target, so its later causal activations cannot
be changed by this patch; source-identity decoding below concerns its COPY in
the target state. Partial mixtures can still be off-manifold / entangled.
"""
from study import *

def run(setting):
    pack=load(setting);m=pack['m'];sp=[np.flatnonzero(m['split']==i) for i in range(3)]
    n=len(pack['nodes']);x=pack['x'][:,1,1:2]
    y=pack['reach'][:,m['target']].T.astype(np.float32)
    w,bb,_=coefficient_fit(x,y,sp)
    source_w,source_b,_=coefficient_fit(x,np.eye(n,dtype=np.float32)[m['source']],sp)
    goal_w,goal_b,_=coefficient_fit(x,np.eye(n,dtype=np.float32)[m['target']],sp)
    source_w,source_b=source_w[0],source_b[0]
    coefs=dict(w=np.stack([w[0]]*2),b=np.stack([bb[0]]*2),
        identity_w=np.stack([goal_w[0]]*2),identity_b=np.stack([goal_b[0]]*2))
    u=np.linalg.svd(w[0],full_matrices=False)[0]
    result=dict(layer=1,donor='b=(same source, changed target)',ranks=list(RANKS),curves=[],checks={})
    for splitname,panel in [('val',squares(pack,1)),('test',squares(pack,2))]:
        a,b,d,c=panel['indices'].T;base=pack['x'][a,1,1];donor=pack['x'][b,1,1];diff=donor-base
        assert np.array_equal(m['source'][a],m['source'][b])
        source_pred=(base@source_w+source_b).argmax(-1);source_correct=source_pred==m['source'][a]
        eligible=panel['all_four_correct'];baseline=eligible&source_correct
        result[splitname+'_panel']=panel_info(pack,panel)
        for rank in RANKS:
            us=u[:,:rank];delta=diff@us@us.T;norm=np.linalg.norm(delta,axis=1,keepdims=True)
            variants=[('ancestor_subspace',0,base+delta),('ancestor_complement',0,donor-delta)]
            for seed in (201,303,407):
                ur=np.linalg.qr(np.random.default_rng(seed).normal(size=(base.shape[1],base.shape[1])))[0][:,:rank]
                rd=diff@ur@ur.T;rd*=norm/np.maximum(np.linalg.norm(rd,axis=1,keepdims=True),1e-12)
                variants.append(('random_equal_norm',seed,(base+rd).astype(np.float32)))
            if rank==32:
                for seed in (201,303,407):
                    perm=np.random.default_rng(seed).permutation(n)
                    sy=pack['reach'][:,perm[m['target']]].T.astype(np.float32)
                    sw,_,_=coefficient_fit(x,sy,sp);su=np.linalg.svd(sw[0],full_matrices=False)[0][:,:rank]
                    rd=diff@su@su.T;rd*=norm/np.maximum(np.linalg.norm(rd,axis=1,keepdims=True),1e-12)
                    variants.append(('shuffled_equal_norm',seed,(base+rd).astype(np.float32)))
            for method,seed,edited in variants:
                margin,valid=forward_patch(pack,a,1,edited)
                stat=metrics(pack,panel,margin,edited,1,coefs,valid)
                new_source=(edited@source_w+source_b).argmax(-1)
                retained=baseline&(new_source==m['source'][a])
                stat['source_copy_readout']=dict(baseline_accuracy=float(source_correct[eligible].mean()),
                    edited_accuracy=float((new_source[eligible]==m['source'][a[eligible]]).mean()),
                    baseline_correct_n=int(baseline.sum()),retained_n=int(retained.sum()),
                    retention_among_baseline_correct=float(retained.sum()/max(baseline.sum(),1)),
                    counterfactual_success_among_retained=float((((margin>0)==m['label'][b])&valid)[retained].mean()) if retained.any() else None,
                    warning='Post-treatment subsets differ between methods; do not compare them as identical cohorts.')
                result['curves'].append(dict(split=splitname,method=method,seed=seed,rank=rank,result=stat))
        # Same-source transfer cannot alter source-position states: numerical
        # prefix capture at the source agrees across the two natural queries.
        source_error=float(np.max(abs(pack['x'][a,0]-pack['x'][b,0])))
        assert source_error<1e-4
        result['checks'][splitname+'_same_source_prefix_max_difference']=source_error
        print(setting,splitname,'same-source complete',flush=True)
    write(HERE/f'{setting}_same_source.json',result)
    del pack;torch.cuda.empty_cache()

if __name__=='__main__':
    for setting in ('simple','hard'):run(setting)

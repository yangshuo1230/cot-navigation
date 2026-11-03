"""Disambiguate goal-only vs source-conditioned target messages.

Donor square: a=(sa,ta), b=(sa,tb), c=(sb,ta), d=(sb,tb), labels a=d != b=c.
Goal-only message predicts counterfactual for b/d; a joint answer predicts b/c.
Also test exact readout-nullspace edits and replicate equal-norm controls.
"""
from study import *
from cot_navigation.analysis.attention import DistanceBaseline

def summarize_probe(pack,panel):
    x=pack['x'];m=pack['m'];nl=x.shape[2];sp=[np.flatnonzero(m['split']==i) for i in range(3)]
    pred,meta=linear.fit(x.reshape(len(x),3*nl,-1),m['label'],sp,binary=True)
    result=[];square=panel['indices'];truth=m['label'][square]
    for h in range(3*nl):
        threshold=meta['threshold'][h][0];allpred=pred[:,h,0]>=threshold
        result.append(dict(role=('source','target','separator')[h//nl],layer=h%nl,
            test=linear.classification(m['label'][sp[2]],pred[sp[2],h,0],threshold),
            square_all_four_accuracy=float((allpred[square]==truth).all(1).mean()),
            square_all_four_accuracy_native_correct=float((allpred[square[panel['all_four_correct']]]==truth[panel['all_four_correct']]).all(1).mean())))
    n=len(pack['nodes']);add=np.concatenate([np.eye(n)[m['source']],np.eye(n)[m['target']]],1).astype(np.float32)
    p,fit=linear.fit(add,m['label'],sp,binary=True);thr=fit['threshold'][0][0]
    return dict(residual=result,additive_endpoint_baseline=dict(test=linear.classification(m['label'][sp[2]],p[sp[2],0,0],thr),
        square_all_four_accuracy=float(((p[square,0,0]>=thr)==truth).all(1).mean())),
        note='Query-train answer labels only, val threshold. All-four XOR score rules out source-only, goal-only and additive source+goal answers.')

@torch.inference_mode()
def attention(pack):
    model=pack['model'];ids=pack['ids'];length=ids.shape[1];nl=model.n_layers
    observed={}
    def hook(l):
        def capture(module,args):
            xx,pos=args[:2];b,le,_=xx.shape
            q,k,_=module.qkv(xx).view(b,le,3,module.n_heads,module.head_dim).unbind(2)
            q=apply_rope(q.transpose(1,2),pos,module.rope_base);k=apply_rope(k.transpose(1,2),pos,module.rope_base)
            score=q@k.transpose(-2,-1)/module.head_dim**.5
            observed[l]=score.masked_fill(~torch.ones(le,le,device=xx.device,dtype=torch.bool).tril(),float('-inf')).softmax(-1)
        return capture
    handles=[block.attn.register_forward_pre_hook(hook(l)) for l,block in enumerate(model.blocks)]
    baseline=DistanceBaseline(nl,model.n_heads,length);test=[];max_difference=0
    try:
        for start in range(0,len(ids),128):
            logits=model(ids[start:start+128]);margin=(logits[:,-1,pack['tok'].yes]-logits[:,-1,pack['tok'].no]).cpu().numpy()
            max_difference=max(max_difference,float(np.max(abs(margin-pack['margin'][start:start+128]))))
            a=torch.stack([observed[l] for l in range(nl)]).cpu().numpy();splits=pack['m']['split'][start:start+128]
            tr=a[:,splits==0]
            for d in range(length):
                baseline.sums[...,d]+=np.diagonal(tr,offset=-d,axis1=-2,axis2=-1).sum((1,3))
                baseline.counts[d]+=tr.shape[1]*(length-d)
            if (splits==2).any():test.append(a[:,splits==2])
    finally:
        for h in handles:h.remove()
    a=np.concatenate(test,axis=1).mean((1,2));mu=baseline.mean.mean(1)
    roles=np.zeros(length,int);roles[-3:]=[1,2,3]
    result={}
    for receiver,name in ((length-2,'target'),(length-1,'answer_separator')):
        raw=np.zeros((nl,4));correct=np.zeros_like(raw)
        for role in range(4):
            js=np.flatnonzero((roles==role)&(np.arange(length)<=receiver))
            raw[:,role]=a[:,receiver,js].sum(-1)
            correct[:,role]=(a[:,receiver,js]-mu[:,receiver-js]).sum(-1)
        result[name]=dict(raw_mass=raw.tolist(),distance_centered_mass=correct.tolist())
    assert max_difference<1e-4
    return dict(roles=['syntax','source','target','separator'],receivers=result,
        observer_margin_error=max_difference,note='Train-pair weighted exact distance mean subtracted; prompt length is fixed, so role/position remain partly confounded.')

def run(setting):
    pack=load(setting);prior=json.loads((HERE/f'{setting}.json').read_text())
    panels={name:squares(pack,split) for name,split in [('val',1),('test',2)]}
    result=dict(joint_answer_probes=summarize_probe(pack,panels['test']),attention=attention(pack),curves=[],checks={})
    nl=pack['x'].shape[2]
    for name,panel in panels.items():
        a,b,d,c=panel['indices'].T
        for l in range(nl):
            margin,valid=forward_patch(pack,a,l,pack['x'][b,1,l])
            result['curves'].append(dict(split=name,method='same_source_other_target_residual',layer=l,
                result=metrics(pack,panel,margin,valid=valid)))
            if l==nl-1:continue
            for method,donor in [('same_source_other_target_edge',b),('same_target_other_source_edge',c),('other_source_other_target_edge',d)]:
                margin,valid=forward_patch(pack,a,l,donor_v=values(pack,donor,l))
                result['curves'].append(dict(split=name,method=method,layer=l+1,result=metrics(pack,panel,margin,valid=valid)))
            margin,valid=forward_patch(pack,a,l,donor_v=values(pack,c,l,role=0),sender_position=-3,receiver_position=-2)
            result['curves'].append(dict(split=name,method='other_source_to_target_value_edge',layer=l+1,
                result=metrics(pack,panel,margin,valid=valid)))
            if l==0:
                margin0,_=forward_patch(pack,a[:16],l,donor_v=values(pack,a[:16],l,role=0),sender_position=-3,receiver_position=-2)
                error=float(np.max(abs(margin0-pack['margin'][a[:16]])));assert error<1e-4
                result['checks'][name+'_source_target_edge_noop']=error
    # Exact nullspace of the fitted ancestor decoder, not only a low-rank
    # complement. Preserves all 100 linear ancestor scores by construction.
    m=pack['m'];sp=[np.flatnonzero(m['split']==i) for i in range(3)]
    y=pack['reach'][:,m['target']].T.astype(np.float32)
    w,bb,_=coefficient_fit(pack['x'][:,1],y,sp)
    iw,ib,_=coefficient_fit(pack['x'][:,1],np.eye(len(pack['nodes']),dtype=np.float32)[m['target']],sp)
    coefs=dict(w=w,b=bb,identity_w=iw,identity_b=ib)
    p=panels['test'];a,b,d,c=p['indices'].T;l=prior['causal']['validation_selected']['layer'];rank=prior['causal']['validation_selected']['rank']
    base=pack['x'][a,1,l];diff=pack['x'][d,1,l]-base
    u,sv,_=np.linalg.svd(w[l].astype(np.float64),full_matrices=True)
    matrix_rank=int((sv>sv[0]*1e-5).sum());null=u[:,matrix_rank:]
    delta=(diff@null@null.T).astype(np.float32);edited=base+delta
    error=float(np.max(abs(delta@w[l])));assert error<1e-4
    margin,valid=forward_patch(pack,a,l,edited)
    result['exact_ancestor_readout_nullspace']=dict(layer=l,decoder_matrix_rank=matrix_rank,null_dimension=len(u)-matrix_rank,
        max_all_ancestor_score_change=error,result=metrics(pack,p,margin,edited,l,coefs,valid),
        warning='Preserving this linear decoder does not guarantee preserving all nonlinear graph information.')
    anc=u[:,:rank];anc_delta=(diff@anc@anc.T).astype(np.float32);norm=np.linalg.norm(anc_delta,axis=1,keepdims=True)
    result['replicated_norm_matched_controls']=[]
    for seed in (201,303,407):
        ur=np.linalg.qr(np.random.default_rng(seed).normal(size=(base.shape[1],base.shape[1])))[0][:,:rank]
        delta=diff@ur@ur.T;delta*=norm/np.maximum(np.linalg.norm(delta,axis=1,keepdims=True),1e-12)
        edited=(base+delta).astype(np.float32);margin,valid=forward_patch(pack,a,l,edited)
        result['replicated_norm_matched_controls'].append(dict(method='random',seed=seed,result=metrics(pack,p,margin,edited,l,coefs,valid)))
        perm=np.random.default_rng(seed).permutation(len(pack['nodes']))
        shuffled=pack['reach'][:,perm[m['target']]].T.astype(np.float32)
        sw,_,_=coefficient_fit(pack['x'][:,1,l:l+1],shuffled,sp)
        us=np.linalg.svd(sw[0],full_matrices=False)[0][:,:rank]
        delta=diff@us@us.T;delta*=norm/np.maximum(np.linalg.norm(delta,axis=1,keepdims=True),1e-12)
        edited=(base+delta).astype(np.float32);margin,valid=forward_patch(pack,a,l,edited)
        result['replicated_norm_matched_controls'].append(dict(method='shuffled_ancestors',seed=seed,result=metrics(pack,p,margin,edited,l,coefs,valid)))
    write(HERE/f'{setting}_followup.json',result);del pack;torch.cuda.empty_cache();print(setting,'followup COMPLETE',flush=True)

if __name__=='__main__':
    for setting in sys.argv[1:] or ('simple','hard'):run(setting)

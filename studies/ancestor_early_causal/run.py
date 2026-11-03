"""Ancestor interventions at target embedding and L1 post-attention residual.

Frozen CPU FP32; ridge only on GPU0. Same-source natural donors and original
query-test crossed panels. Node-heldout variants additionally exclude both
recipient and donor target identities from probe training and validation.
Only compact final statistics/examples are saved, no weights/activations.
"""
from pathlib import Path
import os,sys,json,importlib.util
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import numpy as np
import torch
torch.set_num_threads(2)
ROOT=Path.cwd();OUT=ROOT/'studies/ancestor_early_causal'
spec=importlib.util.spec_from_file_location('early_previous',ROOT/'studies/direct_lookup_causal/study.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
RANKS=(1,2,4,8,16,32,64);SEEDS=(201,303,407)

@torch.inference_mode()
def attention_states(pack,position=-2):
    model=pack['model'];observed={};outputs=[]
    def before(module,args):observed['input']=args[0]
    def after(module,args,output):observed['state']=observed['input']+output[0]
    handles=[model.blocks[0].register_forward_pre_hook(before),model.blocks[0].attn.register_forward_hook(after)]
    error=0
    try:
        for start in range(0,len(pack['ids']),128):
            ids=pack['ids'][start:start+128];logits=model(ids)
            outputs.append(observed['state'][:,position].numpy().copy())
            margin=(logits[:,-1,pack['tok'].yes]-logits[:,-1,pack['tok'].no]).numpy()
            error=max(error,float(np.max(abs(margin-pack['margin'][start:start+128]))))
    finally:
        for handle in handles:handle.remove()
    assert error==0
    return np.concatenate(outputs),error

@torch.inference_mode()
def forward(pack,indices,stage,edited,position=-2):
    """Patch the residual itself, including its FFN skip path.

    A pre-hook on norm2 alone would change only the FFN input, leaving the
    skip residual unchanged; instead modify the attention delta so their sum
    equals the desired post-attention state. Embedding weights/head untouched.
    """
    model=pack['model'];tok=pack['tok'];margins=[];valid=[];input_error=0
    for start in range(0,len(indices),128):
        ix=indices[start:start+128];values=torch.as_tensor(edited[start:start+128]);seen={}
        if stage=='embedding':
            def embedding_hook(module,args,output):
                result=output.clone();result[:,position]=values;return result
            handles=[model.embed.register_forward_hook(embedding_hook)]
        else:
            def before(module,args):seen['input']=args[0]
            def change(module,args,output):
                delta=output[0].clone();delta[:,position]=values-seen['input'][:,position]
                return (delta,)+output[1:]
            def check(module,args):seen['actual']=args[0][:,position]
            handles=[model.blocks[0].register_forward_pre_hook(before),
                model.blocks[0].attn.register_forward_hook(change),model.blocks[0].norm2.register_forward_pre_hook(check)]
        try:
            logits=model(pack['ids'][ix]);head=logits[:,-1];top=head.argmax(-1)
            margins.append((head[:,tok.yes]-head[:,tok.no]).numpy())
            valid.append(((top==tok.yes)|(top==tok.no)).numpy())
            if stage!='embedding':input_error=max(input_error,float((seen['actual']-values).abs().max()))
        finally:
            for handle in handles:handle.remove()
    assert input_error<1e-4,input_error
    return np.concatenate(margins),np.concatenate(valid),input_error

def fit(x,y,train,val,test):
    w,b,reg=old.coefficient_fit(x[:,None],y,[train,val,test]);return w[0],b[0],reg

def norm_match(delta,reference):
    norm=np.linalg.norm(reference,axis=1,keepdims=True)
    result=delta*norm/np.maximum(np.linalg.norm(delta,axis=1,keepdims=True),1e-12)
    assert np.allclose(np.linalg.norm(result,axis=1),norm[:,0],atol=1e-4,rtol=1e-5)
    return result.astype(np.float32)

def cutoff(scores,labels,mask):
    cuts=np.unique(np.r_[0,np.quantile(scores[mask],np.linspace(0,1,41))]);truth=labels.astype(bool)&mask
    def quality(cut):
        pred=(scores>=cut)&mask;tp=(pred&truth).sum();fp=(pred&~truth).sum();fn=(~pred&truth).sum()
        return 2*tp/max(2*tp+fp+fn,1)
    return float(max(cuts,key=quality))

def summarize(pack,a,b,x,edited,margin,valid,readers,mean,erase):
    m=pack['m'];wanted=m['label'][b];baseline=m['native_pred'][a]
    original=x[a];donor=x[b];source=m['source'][a];target=m['target'][a];newtarget=m['target'][b]
    aw,ab,cut=readers['ancestor'];before=original@aw+ab;after=edited@aw+ab;desired=donor@aw+ab
    masks=np.ones_like(after,bool);masks[np.arange(len(a)),source]=False;masks[np.arange(len(a)),target]=False;masks[np.arange(len(a)),newtarget]=False
    truth=pack['reach'][:,newtarget].T.astype(bool);guess=(after>=cut)&masks;truth&=masks
    tp=int((guess&truth).sum());fp=int((guess&~truth).sum());fn=int((~guess&truth).sum())
    diff=desired-before;den=(diff*diff).sum(1);progress=((after-before)*diff).sum(1)/np.maximum(den,1e-20)
    sw,sb=readers['target_id'];ow,ob=readers['source_id']
    original_id=(original@sw+sb).argmax(-1);new_id=(edited@sw+sb).argmax(-1)
    original_source=(original@ow+ob).argmax(-1);new_source=(edited@ow+ob).argmax(-1)
    result=dict(n=len(a),unique_recipient_queries=len(set(a)),unique_targets=len(set(target)|set(newtarget)),
        original_accuracy=float((baseline==m['label'][a]).mean()),
        edited_original_accuracy=float((((margin>0)==m['label'][a])&valid).mean()),
        counterfactual_success=float((((margin>0)==wanted)&valid).mean()),
        valid_answer_rate=float(valid.mean()),desired_margin_shift=float(((margin-pack['margin'][a])*(2*wanted-1)).mean()),
        edit_norm=float(np.linalg.norm(edited-original,axis=1).mean()),
        target_state_norm_ratio=float(np.mean(np.linalg.norm(edited,axis=1)/np.linalg.norm(original,axis=1))),
        independent_ancestor_donor_F1=2*tp/max(2*tp+fp+fn,1),
        independent_ancestor_score_progress=float(progress[den>1e-10].mean()) if (den>1e-10).any() else None,
        independent_source_membership_original_accuracy=float(((before[np.arange(len(a)),source]>=cut)==m['label'][a]).mean()),
        independent_source_membership_counterfactual_accuracy=float(((after[np.arange(len(a)),source]>=cut)==wanted).mean()),
        target_identity_baseline=float((original_id==target).mean()),target_identity_retained=float((new_id==target).mean()),
        donor_identity_adopted=float((new_id==newtarget).mean()),
        source_copy_baseline=float((original_source==source).mean()),source_copy_edited=float((new_source==source).mean()))
    # Audit leakage into other graph features, using separately fitted readers.
    for name in ('successors','descendants','depth'):
        w,bias=readers[name];natural=donor@w+bias;base=original@w+bias;changed=edited@w+bias;dd=natural-base;norm=(dd*dd).sum(1)
        take=norm>1e-10
        result[name+'_score_progress']=float((((changed-base)*dd).sum(1)/np.maximum(norm,1e-20))[take].mean()) if take.any() else None
    return result

def run(setting):
    pack=old.load(setting,device='cpu');m=pack['m'];n=len(pack['nodes']);attn,error=attention_states(pack)
    original_embedding=pack['model'].embed.weight.detach().clone()
    stages={'embedding':pack['x'][:,1,0],'L1_attention':attn}
    panel=old.squares(pack,2);a,b,_,_=panel['indices'].T
    assert np.array_equal(m['source'][a],m['source'][b]);assert np.all(m['label'][a]!=m['label'][b])
    # Fixed before edits, common to all methods. Keep full panel and original-correct summary.
    eligible=panel['all_four_correct'];a=a[eligible];b=b[eligible]
    query_split=[np.flatnonzero(m['split']==i) for i in range(3)]
    y=pack['reach'][:,m['target']].T.astype(np.float32)
    out=dict(manifest=pack['manifest'],ranks=RANKS,stages={},examples=[],
        fit_note='Direction constructor and independent audit decoder use disjoint halves of query-train. '
            'Both select penalties on query-val only. Node-holdout variants use 60/20/20 target identities '
            'intersected with original query split; both edited/donor targets must be test-fold nodes. '
            'Identity audit alone uses all query-train targets so identity retention is measurable.',
        checks={'observer_margin_error':error},inference='CPU FP32',probe_fitting='GPU0')
    for stage,x in stages.items():
        entries=[]
        # Full-target positive control and no-op must precede interpretation.
        base=x[a];donor=x[b]
        margin,valid,err=forward(pack,a,stage,base)
        noop_error=float(np.max(abs(margin-pack['margin'][a])));assert noop_error<1e-4
        full_margin,full_valid,err=forward(pack,a,stage,donor)
        if stage=='embedding':assert float(np.max(abs(full_margin-pack['margin'][b])))<1e-4
        out['checks'][stage]=dict(noop_margin_error=noop_error,patched_residual_error=err,
            full_swap_counterfactual_success=float((((full_margin>0)==m['label'][b])&full_valid).mean()),n=len(a))
        # Separate diagnostics; identity decoder sees train node IDs intentionally.
        iw,ib,_=fit(x,np.eye(n,dtype=np.float32)[m['target']],*query_split)
        ow,ob,_=fit(x,np.eye(n,dtype=np.float32)[m['source']],*query_split)
        for fold_seed in (None,17,31,47):
            if fold_seed is None:
                splits=query_split;keep=np.ones(len(a),bool)
            else:
                order=np.random.default_rng(fold_seed).permutation(n);fold=np.zeros(n,int)
                fold[order[60:80]]=1;fold[order[80:]]=2
                splits=[np.flatnonzero((m['split']==i)&(fold[m['target']]==i)) for i in range(3)]
                keep=(fold[m['target'][a]]==2)&(fold[m['target'][b]]==2)
            aa,bb=a[keep],b[keep]
            if not len(aa):entries.append(dict(node_seed=fold_seed,n=0,curves=[]));continue
            train,val,test=splits;shuffled_train=np.random.default_rng(431).permutation(train)
            constructor,audit=np.array_split(shuffled_train,2)
            assert not set(constructor)&set(audit)
            w,bias,penalty=fit(x,y,constructor,val,test);u,singular,_=np.linalg.svd(w,full_matrices=False)
            aw,ab,_=fit(x,y,audit,val,test)
            mask=np.ones((len(val),n),bool);mask[np.arange(len(val)),m['target'][val]]=False;mask[np.arange(len(val)),m['source'][val]]=False
            cut=cutoff(x[val]@aw+ab,y[val],mask)
            readers=dict(ancestor=(aw,ab,cut),target_id=(iw,ib),source_id=(ow,ob))
            for name,labels in [('successors',pack['adjacency'][m['target']]),('descendants',pack['reach'][m['target']].astype(np.float32)),('depth',pack['depth'][m['target'],None].astype(np.float32))]:
                rw,rb,_=fit(x,labels,audit,val,test);readers[name]=(rw,rb)
            random_u={seed:np.linalg.qr(np.random.default_rng(seed).normal(size=(128,128)))[0].astype(np.float32) for seed in SEEDS}
            shuffled_u={}
            for seed in SEEDS:
                permutation=np.random.default_rng(seed).permutation(n);sy=pack['reach'][:,permutation[m['target']]].T.astype(np.float32)
                sw,_,_=fit(x,sy,constructor,val,test);shuffled_u[seed]=np.linalg.svd(sw,full_matrices=False)[0]
            base=x[aa];donor=x[bb];difference=donor-base;mean=x[constructor].mean(0)
            record=dict(node_seed=fold_seed,n=len(aa),unique_targets=len(set(m['target'][aa])|set(m['target'][bb])),
                constructor_queries=len(constructor),audit_queries=len(audit),validation_queries=len(val),
                constructor_target_nodes=len(set(m['target'][constructor])),ridge=penalty,
                singular_values=singular.tolist(),effective_rank_1e5=int((singular>singular[0]*1e-5).sum()),
                control_subspace_overlap={},curves=[])
            variants=[('no_op',0,0,base),('full_swap',0,0,donor)]
            for rank in RANKS:
                basis=u[:,:rank];delta=difference@basis@basis.T;remove=-(base-mean)@basis@basis.T
                variants.extend([('ancestor_swap',rank,0,base+delta),('complement_swap',rank,0,donor-delta),('ancestor_erase',rank,0,base+remove)])
                for seed in SEEDS:
                    for kind,bases in [('random',random_u),('shuffled',shuffled_u)]:
                        v=bases[seed][:,:rank]
                        variants.append((kind+'_swap',rank,seed,base+norm_match(difference@v@v.T,delta)))
                        variants.append((kind+'_erase',rank,seed,base+norm_match(-(base-mean)@v@v.T,remove)))
                record['control_subspace_overlap'][str(rank)]={kind:[float(np.linalg.norm(basis.T@bases[seed][:,:rank])**2/rank) for seed in SEEDS]
                    for kind,bases in [('random',random_u),('shuffled',shuffled_u)]}
            for method,rank,seed,edited in variants:
                edited=edited.astype(np.float32);margin,valid,err=forward(pack,aa,stage,edited)
                metrics=summarize(pack,aa,bb,x,edited,margin,valid,readers,mean,'erase' in method)
                record['curves'].append(dict(method=method,rank=rank,seed=seed,metrics=metrics))
                if fold_seed is None and rank==32 and method=='ancestor_swap':
                    audit_after=edited@aw+ab
                    for j in np.flatnonzero(valid&((margin>0)==m['label'][aa]))[:2]:
                        def node(v):return pack['tok'].name(int(pack['nodes'][v]))
                        out['examples'].append(dict(stage=stage,source=node(m['source'][aa[j]]),target=node(m['target'][aa[j]]),
                            donor_target=node(m['target'][bb[j]]),original_label=int(m['label'][aa[j]]),
                            desired_label=int(m['label'][bb[j]]),edited_answer=int(margin[j]>0),
                            ancestor_source_score_before=float((base[j]@aw+ab)[m['source'][aa[j]]]),
                            ancestor_source_score_after=float(audit_after[j,m['source'][aa[j]]]),threshold=cut))
            entries.append(record);print(setting,stage,'node_seed',fold_seed,'pairs',len(aa),'COMPLETE',flush=True)
        out['stages'][stage]=entries
    assert torch.equal(pack['model'].embed.weight,original_embedding)
    assert pack['model'].lm_head.weight.data_ptr()==pack['model'].embed.weight.data_ptr()
    out['checks']['embedding_and_tied_head_unchanged']=True
    old.write(OUT/f'{setting}.json',out)
    print(setting,'COMPLETE',flush=True)

if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    for setting in ('simple','hard'):run(setting)

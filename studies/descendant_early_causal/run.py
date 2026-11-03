"""Source reachable-descendant edits, matched to the target ancestor experiment.

Only source-position activations are replaced; target stays fixed. Reuse
tested patch/fit primitives with an explicit source position (-3). Native
frozen inference CPU FP32; all ridge fitting GPU0; no saved activations/weights.
"""
from pathlib import Path
import os,sys,json,importlib.util
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import numpy as np
import torch
torch.set_num_threads(2)
ROOT=Path.cwd();OUT=ROOT/'studies/descendant_early_causal'
spec=importlib.util.spec_from_file_location('early_shared',ROOT/'studies/ancestor_early_causal/run.py')
shared=importlib.util.module_from_spec(spec);spec.loader.exec_module(shared)
old=shared.old;fit=shared.fit;RANKS=shared.RANKS;SEEDS=shared.SEEDS

def summarize(pack,a,b,x,edited,margin,valid,readers):
    m=pack['m'];source=m['source'][a];newsource=m['source'][b];target=m['target'][a]
    original=x[a];donor=x[b];wanted=m['label'][b]
    w,bias,cut=readers['descendant'];before=original@w+bias;after=edited@w+bias;desired=donor@w+bias
    mask=np.ones_like(after,bool)
    for ids in (source,newsource,target):mask[np.arange(len(a)),ids]=False
    truth=pack['reach'][newsource].astype(bool)&mask;guess=(after>=cut)&mask
    tp=int((guess&truth).sum());fp=int((guess&~truth).sum());fn=int((~guess&truth).sum())
    original_truth=pack['reach'][source].astype(bool)&mask
    def original_ranking_auc(scores):
        aucs=[old.linear.auc(truth_row[eligible],score_row[eligible])
              for truth_row,score_row,eligible in zip(original_truth,scores,mask)]
        aucs=[v for v in aucs if v is not None]
        return float(np.mean(aucs)) if aucs else None
    def f1_original(scores):
        prediction=(scores>=cut)&mask
        true_positive=int((prediction&original_truth).sum())
        false_positive=int((prediction&~original_truth).sum())
        false_negative=int((~prediction&original_truth).sum())
        return 2*true_positive/max(2*true_positive+false_positive+false_negative,1)
    diff=desired-before;den=(diff*diff).sum(1);take=den>1e-10
    progress=((after-before)*diff).sum(1)/np.maximum(den,1e-20)
    iw,ib=readers['source_id'];base_id=(original@iw+ib).argmax(-1);edited_id=(edited@iw+ib).argmax(-1)
    result=dict(n=len(a),unique_recipient_queries=len(set(a)),unique_sources=len(set(source)|set(newsource)),
        unique_targets=len(set(target)),donor_query_test_fraction=float((m['split'][b]==2).mean()),
        original_accuracy=float((m['native_pred'][a]==m['label'][a]).mean()),
        edited_original_accuracy=float((((margin>0)==m['label'][a])&valid).mean()),
        counterfactual_success=float((((margin>0)==wanted)&valid).mean()),valid_answer_rate=float(valid.mean()),
        desired_margin_shift=float(((margin-pack['margin'][a])*(2*wanted-1)).mean()),
        edit_norm=float(np.linalg.norm(edited-original,axis=1).mean()),
        source_state_norm_ratio=float(np.mean(np.linalg.norm(edited,axis=1)/np.linalg.norm(original,axis=1))),
        independent_descendant_donor_F1=2*tp/max(2*tp+fp+fn,1),
        independent_descendant_original_F1_before=f1_original(before),
        independent_descendant_original_F1_after=f1_original(after),
        original_set_within_query_AUC_before=original_ranking_auc(before),
        original_set_within_query_AUC_after=original_ranking_auc(after),
        independent_descendant_score_progress=float(progress[take].mean()) if take.any() else None,
        independent_target_membership_original_accuracy=float(((before[np.arange(len(a)),target]>=cut)==m['label'][a]).mean()),
        independent_target_membership_edited_original_accuracy=float(((after[np.arange(len(a)),target]>=cut)==m['label'][a]).mean()),
        independent_target_membership_counterfactual_accuracy=float(((after[np.arange(len(a)),target]>=cut)==wanted).mean()),
        source_identity_baseline=float((base_id==source).mean()),source_identity_retained=float((edited_id==source).mean()),
        donor_identity_adopted=float((edited_id==newsource).mean()))
    for name in ('successors','ancestors','depth'):
        rw,rb=readers[name];base=original@rw+rb;change=edited@rw+rb-base;delta=donor@rw+rb-base
        # Aggregate projection avoids unstable per-pair division when donor
        # depth/readout difference is almost zero. Not a probability or R².
        denominator=float((delta*delta).sum())
        result[name+'_aggregate_score_progress']=float((change*delta).sum()/denominator) if denominator>1e-10 else None
    return result

def run(setting):
    pack=old.load(setting,device='cpu');m=pack['m'];n=len(pack['nodes'])
    attention,error=shared.attention_states(pack,position=-3)
    original_embedding=pack['model'].embed.weight.detach().clone()
    stages={'embedding':pack['x'][:,0,0],'L1_attention':attention}
    panel=old.squares(pack,2);a,_,_,b=panel['indices'].T
    assert np.array_equal(m['target'][a],m['target'][b]);assert np.all(m['label'][a]!=m['label'][b])
    eligible=panel['all_four_correct'];a=a[eligible];b=b[eligible]
    query_split=[np.flatnonzero(m['split']==i) for i in range(3)]
    y=pack['reach'][m['source']].astype(np.float32)
    out=dict(manifest=pack['manifest'],relation='strict reachable descendants of source',ranks=RANKS,
        edited_position='source (-3); target fixed',stages={},examples=[],checks={'observer_margin_error':error},
        fit_note='Disjoint query-train halves construct subspace and audit. Query-val selects ridge/threshold. '
            'Source-node-heldout variants: 60/20/20 identities, both recipient/donor sources in test fold. '
            'Donor is c=(changed source,same target) from original crossed square and may belong to any original query split; '
            'recipient belongs to query-test. Original main model saw all graph nodes.',
        inference='CPU FP32',probe_fitting='GPU0')
    for stage,x in stages.items():
        base=x[a];donor=x[b]
        margin,valid,err=shared.forward(pack,a,stage,base,position=-3)
        noop_error=float(np.max(abs(margin-pack['margin'][a])));assert noop_error<1e-4
        full_margin,full_valid,err=shared.forward(pack,a,stage,donor,position=-3)
        natural_error=float(np.max(abs(full_margin-pack['margin'][b])))
        if stage=='embedding':assert natural_error<1e-4
        out['checks'][stage]=dict(noop_margin_error=noop_error,patched_residual_error=err,
            full_swap_vs_natural_donor_margin_error=natural_error,
            full_swap_counterfactual_success=float((((full_margin>0)==m['label'][b])&full_valid).mean()),n=len(a))
        iw,ib,_=fit(x,np.eye(n,dtype=np.float32)[m['source']],*query_split)
        entries=[]
        for node_seed in (None,17,31,47):
            if node_seed is None:splits=query_split;keep=np.ones(len(a),bool)
            else:
                order=np.random.default_rng(node_seed).permutation(n);fold=np.zeros(n,int)
                fold[order[60:80]]=1;fold[order[80:]]=2
                splits=[np.flatnonzero((m['split']==i)&(fold[m['source']]==i)) for i in range(3)]
                keep=(fold[m['source'][a]]==2)&(fold[m['source'][b]]==2)
            aa,bb=a[keep],b[keep]
            if not len(aa):entries.append(dict(node_seed=node_seed,n=0,curves=[]));continue
            train,val,test=splits;constructor,audit=np.array_split(np.random.default_rng(431).permutation(train),2)
            assert not set(constructor)&set(audit)
            if node_seed is not None:
                assert not (set(m['source'][aa])|set(m['source'][bb]))&set(m['source'][np.r_[constructor,audit,val]])
            w,bias,penalty=fit(x,y,constructor,val,test);u,singular,_=np.linalg.svd(w,full_matrices=False)
            aw,ab,_=fit(x,y,audit,val,test)
            mask=np.ones((len(val),n),bool);mask[np.arange(len(val)),m['source'][val]]=False;mask[np.arange(len(val)),m['target'][val]]=False
            cut=shared.cutoff(x[val]@aw+ab,y[val],mask)
            readers=dict(descendant=(aw,ab,cut),source_id=(iw,ib))
            for name,labels in [('successors',pack['adjacency'][m['source']]),('ancestors',pack['reach'].T[m['source']].astype(np.float32)),('depth',pack['depth'][m['source'],None].astype(np.float32))]:
                rw,rb,_=fit(x,labels,audit,val,test);readers[name]=(rw,rb)
            random_u={seed:np.linalg.qr(np.random.default_rng(seed).normal(size=(128,128)))[0].astype(np.float32) for seed in SEEDS}
            shuffled_u={}
            for seed in SEEDS:
                permutation=np.random.default_rng(seed).permutation(n);sy=pack['reach'][permutation[m['source']]].astype(np.float32)
                sw,_,_=fit(x,sy,constructor,val,test);shuffled_u[seed]=np.linalg.svd(sw,full_matrices=False)[0]
            base=x[aa];donor=x[bb];difference=donor-base;mean=x[constructor].mean(0)
            record=dict(node_seed=node_seed,n=len(aa),unique_sources=len(set(m['source'][aa])|set(m['source'][bb])),
                donor_query_test_fraction=float((m['split'][bb]==2).mean()),constructor_queries=len(constructor),audit_queries=len(audit),
                validation_queries=len(val),constructor_source_nodes=len(set(m['source'][constructor])),ridge=penalty,
                singular_values=singular.tolist(),effective_rank_1e5=int((singular>singular[0]*1e-5).sum()),control_subspace_overlap={},curves=[])
            variants=[('no_op',0,0,base),('full_swap',0,0,donor)]
            for rank in RANKS:
                basis=u[:,:rank];delta=difference@basis@basis.T;remove=-(base-mean)@basis@basis.T
                variants.extend([('descendant_swap',rank,0,base+delta),('complement_swap',rank,0,donor-delta),('descendant_erase',rank,0,base+remove)])
                for seed in SEEDS:
                    for kind,bases in [('random',random_u),('shuffled',shuffled_u)]:
                        v=bases[seed][:,:rank]
                        variants.append((kind+'_swap',rank,seed,base+shared.norm_match(difference@v@v.T,delta)))
                        variants.append((kind+'_erase',rank,seed,base+shared.norm_match(-(base-mean)@v@v.T,remove)))
                record['control_subspace_overlap'][str(rank)]={kind:[float(np.linalg.norm(basis.T@bases[seed][:,:rank])**2/rank) for seed in SEEDS]
                    for kind,bases in [('random',random_u),('shuffled',shuffled_u)]}
            for method,rank,seed,edited in variants:
                edited=edited.astype(np.float32);margin,valid,err=shared.forward(pack,aa,stage,edited,position=-3)
                metrics=summarize(pack,aa,bb,x,edited,margin,valid,readers)
                record['curves'].append(dict(method=method,rank=rank,seed=seed,metrics=metrics))
                if node_seed is None and rank==32 and method=='descendant_swap':
                    before=base@aw+ab;after=edited@aw+ab;target=m['target'][aa]
                    flipped_reader=((before[np.arange(len(aa)),target]>=cut)==m['label'][aa])&((after[np.arange(len(aa)),target]>=cut)==m['label'][bb])
                    candidates=np.flatnonzero(valid&((margin>0)==m['label'][aa])&flipped_reader)
                    for j in candidates[:2]:
                        def node(v):return pack['tok'].name(int(pack['nodes'][v]))
                        out['examples'].append(dict(stage=stage,source=node(m['source'][aa[j]]),target=node(target[j]),donor_source=node(m['source'][bb[j]]),
                            original_label=int(m['label'][aa[j]]),desired_label=int(m['label'][bb[j]]),edited_answer=int(margin[j]>0),
                            target_descendant_score_before=float(before[j,target[j]]),target_descendant_score_after=float(after[j,target[j]]),threshold=cut))
            entries.append(record);print(setting,stage,'node_seed',node_seed,'pairs',len(aa),'COMPLETE',flush=True)
        out['stages'][stage]=entries
    assert torch.equal(pack['model'].embed.weight,original_embedding)
    assert pack['model'].lm_head.weight.data_ptr()==pack['model'].embed.weight.data_ptr()
    out['checks']['embedding_and_tied_head_unchanged']=True
    old.write(OUT/f'{setting}.json',out);print(setting,'COMPLETE',flush=True)

if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    for setting in ('simple','hard'):run(setting)

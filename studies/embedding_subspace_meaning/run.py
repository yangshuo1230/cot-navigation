"""Interpret embedding causal subspaces with strictly node-heldout probes.

Compare fixed original rank8 spaces with a stricter pipeline that learns both
the causal space and the explanatory probe without test-node identities.
Recover actual cumulative exposures from saved sampler states. All fitting
and projection training GPU0; original model weights frozen throughout.
"""
from pathlib import Path
import os,sys,importlib.util,json
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import numpy as np
import torch
torch.set_num_threads(2)
ROOT=Path.cwd();HERE=ROOT/'studies/embedding_subspace_meaning'
spec=importlib.util.spec_from_file_location('causal_search',ROOT/'studies/causal_subspace_search/run.py')
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
from exposure import recover
RANK=8;SPLITS=(17,31,47)

def node_split(seed):
    order=np.random.default_rng(seed).permutation(100)
    return [order[:60],order[60:80],order[80:]]

def correlation(y,p):
    if np.std(y)<1e-10 or np.std(p)<1e-10:return None
    return float(np.corrcoef(y,p)[0,1])

def rank_values(x):
    order=np.argsort(x,kind='stable');_,start,count=np.unique(x[order],return_index=True,return_counts=True)
    ranks=np.empty(len(x));ranks[order]=np.repeat(start+(count-1)/2,count);return ranks

def regress(features,y,split):
    pred,fit=C.old.linear.fit(features,y,split)
    pred=pred[:,0];test=split[2];metrics=[]
    for k in range(y.shape[1]):
        metric=C.old.linear.regression(y[test,k],pred[test,k])
        metric['pearson']=correlation(y[test,k],pred[test,k])
        metric['spearman']=correlation(rank_values(y[test,k]),rank_values(pred[test,k]))
        metrics.append(metric)
    return metrics,fit

def properties(pack):
    attrs,audit=recover(pack);a=pack['adjacency'];n=len(a)
    depth=pack['depth'];height=np.zeros(n)
    for u in reversed(range(n)):
        for v in np.flatnonzero(a[u]):height[u]=max(height[u],height[v]+1)
    attrs.update(in_degree=a.sum(0),out_degree=a.sum(1),depth=depth,height=height,
        topological_rank=np.arange(n),rank_band=np.arange(n)//10,
        ancestor_count=pack['reach'].sum(0),descendant_count=pack['reach'].sum(1))
    names=list(attrs);labels=np.column_stack([attrs[name] for name in names]).astype(np.float32)
    return attrs,names,labels,audit

def matched_panels(pack,panel,role,attrs,split):
    if not len(panel):return {}
    a,b=panel.T;nodes_a=pack['m'][role][a];nodes_b=pack['m'][role][b]
    degree=(attrs['in_degree'][nodes_a]==attrs['in_degree'][nodes_b])&(attrs['out_degree'][nodes_a]==attrs['out_degree'][nodes_b])
    position=(attrs['depth'][nodes_a]==attrs['depth'][nodes_b])&(attrs['rank_band'][nodes_a]==attrs['rank_band'][nodes_b])
    frequency=np.ones(len(panel),bool)
    for name in ('query_source_frequency','query_target_frequency'):
        cuts=np.unique(np.quantile(attrs[name][split[0]],[.25,.5,.75]))
        bins=np.searchsorted(cuts,attrs[name],side='right')
        frequency&=bins[nodes_a]==bins[nodes_b]
    return dict(same_in_out_degree=panel[degree],same_depth_and_band=panel[position],
        same_source_target_frequency_bins=panel[frequency],all_matched=panel[degree&position&frequency])

@torch.inference_mode()
def reconstructed_swap(pack,panel,role,e,u,coordinates,predicted):
    if not len(panel):return dict(n=0)
    successes=[];norms=[];caps=[]
    for start in range(0,len(panel),128):
        pp=panel[start:start+128];a,b=pp.T;na=pack['m'][role][a];nb=pack['m'][role][b]
        delta=torch.tensor(predicted[nb]-predicted[na],device=C.DEVICE)
        actual=torch.tensor(coordinates[nb]-coordinates[na],device=C.DEVICE)
        ratio=actual.norm(dim=1,keepdim=True)/delta.norm(dim=1,keepdim=True).clamp_min(1e-12)
        caps.extend((ratio[:,0]<1).cpu().tolist());delta=delta*ratio.clamp_max(1)
        edited=e[na]+delta@u.T
        logits=C.patched_logits(pack,pp,(role,'E'),edited)
        desired=torch.tensor(pack['m']['label'][b],device=C.DEVICE)
        token=torch.where(desired==1,pack['tok'].yes,pack['tok'].no)
        successes.extend((logits.argmax(-1)==token).cpu().tolist());norms.extend(delta.norm(dim=1).cpu().tolist())
    return dict(n=len(panel),success=float(np.mean(successes)),mean_edit_norm=float(np.mean(norms)),
        capped_fraction=float(np.mean(caps)),note='Linear attributes->subspace coordinates; swap predicted donor-recipient difference; '
        'cap each edit to natural learned-subspace edit norm. Oracle attribute diagnostic, not model task accuracy or a causal mediation percentage.')

def explain(pack,role,e,u,attrs,names,y,split,panel):
    tr,va,te=split;embedding=e.cpu().numpy();basis=u.cpu().numpy();z=embedding@basis
    complement=np.linalg.qr(basis,mode='complete')[0][:,RANK:]
    pca=np.linalg.svd(embedding[tr]-embedding[tr].mean(0),full_matrices=False)[2][:RANK].T
    features=dict(learned=z,full_embedding=embedding,complement=embedding@complement,PCA8=embedding@pca,
        position=np.column_stack([attrs[k] for k in ('topological_rank','rank_band','depth','height')]).astype(np.float32))
    for seed in C.RANDOM_SEEDS:
        q=np.linalg.qr(np.random.default_rng(seed).normal(size=(128,RANK)))[0]
        features[f'random_subspace_{seed}']=(embedding@q).astype(np.float32)
        features[f'random_node_identity_{seed}']=np.random.default_rng(seed).normal(size=(100,RANK)).astype(np.float32)
    result=dict(node_train=tr.tolist(),node_val=va.tolist(),node_test=te.tolist(),probes={},coordinate_reconstruction={},
        native_subspace=C.evaluate(pack,panel,(role,'E'),e[pack['m'][role]],u),matched_pairs={})
    for name,x in features.items():
        metrics,fit=regress(x,y,split);result['probes'][name]=dict(metrics=dict(zip(names,metrics)),fit=fit)
    augmented=np.c_[features['position'],z]
    metrics,fit=regress(augmented,y,split)
    result['probes']['position_plus_learned']=dict(metrics=dict(zip(names,metrics)),fit=fit)
    # Reverse explanation: how much of the learned coordinates can attributes
    # predict on heldout nodes? Uniform Euclidean loss preserves basis rotation.
    freq=[k for k in names if k.startswith('query_') or k.startswith('walk_')]
    local=['in_degree','out_degree','depth','height','topological_rank','rank_band']
    structure=local+['ancestor_count','descendant_count']
    for group,columns in [('frequency',freq),('local_structure',local),('all_structure',structure),('combined',freq+structure)]:
        x=np.column_stack([attrs[k] for k in columns]).astype(np.float32)
        w,b,penalties=C.old.coefficient_fit(x[:,None],z,split);pred=x@w[0]+b[0]
        numerator=float(((pred[te]-z[te])**2).sum());den=float(((z[te]-z[te].mean(0))**2).sum())
        result['coordinate_reconstruction'][group]=dict(columns=columns,heldout_r2=1-numerator/den if den>1e-10 else None,
            ridge=penalties,intervention=reconstructed_swap(pack,panel,role,e,u,z,pred))
    for name,pp in matched_panels(pack,panel,role,attrs,split).items():
        result['matched_pairs'][name]=C.evaluate(pack,pp,(role,'E'),e[pack['m'][role]],u)
    return result

def run(setting):
    pack=C.old.load(setting,device='cpu');attrs,names,y,audit=properties(pack)
    print(setting,'EXPOSURE VERIFIED',audit['query_actual_count'],audit['task_walk_actual_count'],audit['knowledge_walk_actual_count'],flush=True)
    model=C.old.build_model(pack['cfg'].model,len(pack['tok']),pack['tok'].pad).eval().requires_grad_(False)
    model.load_state_dict({k:v.detach().clone() for k,v in pack['model'].state_dict().items()});pack['model']=model.to(C.DEVICE)
    pack['ids']=pack['ids'].clone().to(C.DEVICE)
    e=torch.tensor(pack['embed'],device=C.DEVICE);saved_embedding=model.embed.weight.detach().clone()
    out=dict(manifest=pack['manifest'],rank=RANK,attribute_names=names,node_attributes=y.tolist(),exposure_audit=audit,roles={},
        note='Every explanatory probe uses 60/20/20 node identities, not query rows. Strict branch also learns causal U without probe-test nodes. '
             'Main model already saw all nodes; graph attributes are diagnostic oracles. R2 is heldout prediction, not complete recovery or causal use.')
    previous=json.loads((ROOT/f'studies/causal_subspace_search/{setting}.json').read_text())
    for role,site_i in [('source',0),('target',3)]:
        site=(role,'E');x=e[pack['m'][role]]
        panels=[C.pairs(pack,role,part,limit) for part,limit in enumerate((2048,512,1024))]
        u,training=C.train(pack,panels,site,x,RANK,seed=1301+site_i*20+RANK)
        baseline=C.evaluate(pack,panels[2],site,x,u)
        prior=next(c for c in previous['candidates'] if c['site']==[role,'E'] and c['rank']==RANK)['test']['swap']
        assert abs(baseline['success']-prior['success'])<.02,(baseline,prior)
        entry=dict(original_reproduction=dict(training=training,test=baseline,previous_test_success=prior['success']),
            original_space_nodeheld_probes=[],strict_nodeheld_pipeline=[])
        for seed in SPLITS:
            split=node_split(seed);focus=pack['m'][role]
            keep=np.isin(focus[panels[2][:,0]],split[2])&np.isin(focus[panels[2][:,1]],split[2])
            entry['original_space_nodeheld_probes'].append(dict(seed=seed,**explain(pack,role,e,u,attrs,names,y,split,panels[2][keep])))
        for seed in SPLITS:
            split=node_split(seed)
            pp=[C.pairs(pack,role,part,limit,node_seed=seed) for part,limit in enumerate((2048,512,1024))]
            record=dict(seed=seed,pair_counts=list(map(len,pp)))
            if min(map(len,pp))<8:record['skipped']='Insufficient pairs under original split; no forced fallback.'
            else:
                q,training=C.train(pack,pp,site,x,RANK,seed=7001+seed)
                record.update(training=training,**explain(pack,role,e,q,attrs,names,y,split,pp[2]))
            entry['strict_nodeheld_pipeline'].append(record)
            print(setting,role,'strict split',seed,'pairs',record['pair_counts'],'COMPLETE',flush=True)
        out['roles'][role]=entry;C.old.write(HERE/f'{setting}.json',out)
    assert torch.equal(model.embed.weight,saved_embedding) and model.embed.weight.data_ptr()==model.lm_head.weight.data_ptr()
    out['complete']=True;C.old.write(HERE/f'{setting}.json',out);print(setting,'COMPLETE',flush=True)

if __name__=='__main__':
    HERE.mkdir(exist_ok=True)
    for setting in ('simple','hard'):run(setting)

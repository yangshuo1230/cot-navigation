"""Readout orientation: incoming/outgoing edges vs transitive relations.

Frozen CPU inference avoids occupying busy training GPUs. All closed-form
probe fits on GPU0. Query AND focal-node identity disjoint; 3 node split seeds,
3 random identity vector seeds. Random features match each layer's train-only
coordinate means/stds. No activations or weights written. Original output head
is evaluated independently, including FULL-vocabulary probability mass.
"""
from study import *
OUT=HERE/'relation_orientation';OUT.mkdir(exist_ok=True)
RELATIONS=('predecessors','successors','ancestors','descendants')
RANDOM_SEEDS=(201,303,407)

def mask_for(focal,other,n,forward):
    full=np.ones((len(focal),n),bool);full[np.arange(len(focal)),focal]=False
    full[np.arange(len(focal)),other]=False
    valid=(np.arange(n)[None,:]>focal[:,None]) if forward else (np.arange(n)[None,:]<focal[:,None])
    return {'all_other_nodes':full,'direction_valid_candidates':full&valid}

@torch.inference_mode()
def native_head(pack,role,relation,maskdict):
    x=torch.tensor(pack['x'][:,role]);model=pack['model'];te=np.flatnonzero(pack['m']['split']==2)
    logits=model.lm_head(model.final_norm(x[te]));prob=logits.softmax(-1)
    scores=logits[...,pack['nodes']].cpu().numpy();node_prob=prob[...,pack['nodes']].cpu().numpy()
    node_conditional=node_prob/np.maximum(node_prob.sum(-1,keepdims=True),1e-30)
    labels=relation[te];out=[]
    for l in range(x.shape[1]):
        # Empty relation sets are excluded from probability and set-recall
        # aggregation; separately report how many decision rows contribute.
        count=labels.sum(1);nonempty=count>0
        full=ranking(labels,scores[:,l],maskdict['all_other_nodes'][te])
        valid=ranking(labels,scores[:,l],maskdict['direction_valid_candidates'][te])
        out.append(dict(all_other_nodes=full,direction_valid_candidates=valid,
            nonempty_rows=int(nonempty.sum()),
            full_vocab_relation_mass=float((node_prob[:,l]*labels).sum(1)[nonempty].mean()) if nonempty.any() else None,
            node_conditional_relation_mass=float((node_conditional[:,l]*labels).sum(1)[nonempty].mean()) if nonempty.any() else None))
    return out

def run(setting):
    pack=load(setting,device='cpu');m=pack['m'];n=len(pack['nodes']);nl=pack['x'].shape[2]
    mats=dict(predecessors=pack['adjacency'].T,successors=pack['adjacency'],
        ancestors=pack['reach'].T.astype(np.float32),descendants=pack['reach'].astype(np.float32))
    out=dict(manifest=pack['manifest'],relations=RELATIONS,layers=['E']+[f'L{i}' for i in range(1,nl)],roles={},
        inference_device='CPU FP32',probe_fit_device='GPU0',random_vector_seeds=RANDOM_SEEDS,
        note='One-hop predecessor/successor vs strict transitive ancestor/descendant. Do not equate different tasks AUROC with representational dominance: difficulty/support differ. Heldout focal nodes were seen in main-model training.')
    random={seed:np.random.default_rng(seed).normal(size=(n,pack['x'].shape[-1])).astype(np.float32) for seed in RANDOM_SEEDS}
    for role,name in ((0,'source'),(1,'target')):
        focal=m[name];other=m['target' if name=='source' else 'source'];features=pack['x'][:,role]
        entry={}
        for relation in RELATIONS:
            labels=mats[relation][focal];forward=relation in ('successors','descendants')
            maskdict=mask_for(focal,other,n,forward)
            result={'original_head':native_head(pack,role,labels,maskdict),'heldout':[]}
            for seed in (17,31,47):
                order=np.random.default_rng(seed).permutation(n);fold=np.zeros(n,int)
                fold[order[int(.6*n):int(.8*n)]]=1;fold[order[int(.8*n):]]=2
                split=[np.flatnonzero((m['split']==i)&(fold[focal]==i)) for i in range(3)]
                mean=features[split[0]].mean(0);std=features[split[0]].std(0)
                controls=[random[rs][focal,None,:]*std[None,:,:]+mean[None,:,:] for rs in RANDOM_SEEDS]
                xx=np.concatenate([features]+controls,axis=1)
                pred=ridge_predictions(xx,labels,split)[0]
                # One no-input baseline: marginal relation-node frequency.
                freq=np.broadcast_to(labels[split[0]].mean(0),(len(focal),1,n)).copy()
                pred=np.concatenate([pred,freq],axis=1)
                metrics_all=evaluate(labels,pred,split,maskdict,focal)
                record=dict(seed=seed,query_counts=list(map(len,split)),focal_node_counts=[len(np.unique(focal[ii])) for ii in split],
                    actual=metrics_all[:nl],random={},frequency=metrics_all[-1])
                for k,rs in enumerate(RANDOM_SEEDS):record['random'][str(rs)]=metrics_all[(k+1)*nl:(k+2)*nl]
                result['heldout'].append(record)
            entry[relation]=result
            print(setting,name,relation,'complete',flush=True)
        out['roles'][name]=entry
    write(OUT/f'{setting}.json',out);del pack;torch.cuda.empty_cache()

if __name__=='__main__':
    for setting in sys.argv[1:] or ('simple','hard'):run(setting)

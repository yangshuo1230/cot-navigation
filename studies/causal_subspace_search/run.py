"""Learn shared low-rank interchange subspaces without graph-semantic probes.

Frozen Direct checkpoints. Natural donors change ONE endpoint; desired labels
come only from the donor query's relation. No edit amplification. Query-disjoint
train/validation/test, plus focal-node-heldout replication of the val-selected
site/rank. All learned basis optimization on GPU0. No saved basis/activations.
"""
from pathlib import Path
import os,sys,importlib.util,json,time
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import numpy as np
import torch
import torch.nn.functional as F
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False
ROOT=Path.cwd();HERE=ROOT/'studies/causal_subspace_search'
spec=importlib.util.spec_from_file_location('shared_early',ROOT/'studies/ancestor_early_causal/run.py')
shared=importlib.util.module_from_spec(spec);spec.loader.exec_module(shared)
old=shared.old
RANKS=(1,2,4,8,16);RANDOM_SEEDS=(201,303,407)
SITES=[('source','E'),('source','1A'),('source','1F'),('target','E'),('target','1A'),('target','1F'),('target','2F')]
DEVICE='cuda:0';STEPS=300

def pairs(pack,role,part,limit,node_seed=None):
    """Graph-selected, balanced opposite-label pairs with a fixed other endpoint.

    At most two uses of each recipient; add both directions together. Main
    comparisons retain pairs where both natural queries have legal correct
    predictions, before any intervention. Both queries share original split.
    """
    m=pack['m'];other='target' if role=='source' else 'source'
    keep=m['split']==part
    if node_seed is not None:
        order=np.random.default_rng(node_seed).permutation(len(pack['nodes']));fold=np.zeros(len(order),int)
        fold[order[60:80]]=1;fold[order[80:]]=2
        keep&=fold[m[role]]==part
    candidates=[]
    for endpoint in np.unique(m[other][keep]):
        ids=np.flatnonzero(keep&(m[other]==endpoint))
        positive=ids[m['label'][ids]==1];negative=ids[m['label'][ids]==0]
        candidates.extend((int(a),int(b)) for a in positive for b in negative)
    rng=np.random.default_rng(922+part);rng.shuffle(candidates)
    selected=[];uses={};eligible=0
    for a,b in candidates:
        if m['native_pred'][a]!=m['label'][a] or m['native_pred'][b]!=m['label'][b]:continue
        eligible+=1
        if uses.get(a,0)>=2 or uses.get(b,0)>=2:continue
        selected.extend([(a,b),(b,a)]);uses[a]=uses.get(a,0)+1;uses[b]=uses.get(b,0)+1
        if len(selected)>=limit:break
    result=np.array(selected,dtype=int).reshape(-1,2)
    if len(result):
        a,b=result.T
        assert np.all(m['split'][a]==part) and np.all(m['split'][b]==part)
        assert np.array_equal(m[other][a],m[other][b]) and np.all(m['label'][a]!=m['label'][b])
        assert m['label'][b].mean()==.5
    return result

def meta(pack,panel,role):
    if not len(panel):return dict(n=0)
    a,b=panel.T;m=pack['m']
    return dict(n=len(panel),unique_queries=len(set(a)|set(b)),focal_nodes=len(set(m[role][a])|set(m[role][b])),
        desired_yes_fraction=float(m['label'][b].mean()))

def install(model,site,values):
    """Inject differentiable values into the entire residual at one position."""
    role,stage=site;position={'source':-3,'target':-2,'separator':-1}[role]
    if stage=='E':
        def hook(module,args,output):
            edited=output.clone();edited[:,position]=values;return edited
        return [model.embed.register_forward_hook(hook)]
    layer=int(stage[:-1])-1
    if stage.endswith('F'):
        def hook(module,args,output):
            edited=output[0].clone();edited[:,position]=values;return (edited,)+output[1:]
        return [model.blocks[layer].register_forward_hook(hook)]
    seen={}
    def before(module,args):seen['input']=args[0]
    def after(module,args,output):
        delta=output[0].clone();delta[:,position]=values-seen['input'][:,position]
        return (delta,)+output[1:]
    return [model.blocks[layer].register_forward_pre_hook(before),model.blocks[layer].attn.register_forward_hook(after)]

def patched_logits(pack,panel,site,edited):
    handles=install(pack['model'],site,edited)
    try:return pack['model'](pack['ids'][panel[:,0]])[:,-1]
    finally:
        for handle in handles:handle.remove()

@torch.inference_mode()
def capture(pack):
    matrices={}
    for role,stage in SITES:
        r=0 if role=='source' else 1
        if stage=='E':matrices[role,stage]=pack['x'][:,r,0]
        elif stage.endswith('F'):matrices[role,stage]=pack['x'][:,r,int(stage[:-1])]
    for role in ('source','target'):
        matrices[role,'1A']=shared.attention_states(pack,position=-3 if role=='source' else -2)[0]
    matrices['separator','6F']=pack['x'][:,2,6]
    return matrices

@torch.inference_mode()
def evaluate(pack,panel,site,x,u,method='swap',reference=None):
    if not len(panel):return dict(n=0)
    results=[];norms=[];ratios=[]
    for start in range(0,len(panel),128):
        pp=panel[start:start+128];a,b=pp.T;base=x[a];difference=x[b]-base
        delta=difference if method=='full' else torch.zeros_like(base) if method=='noop' else (difference@u)@u.T
        if method=='complement':delta=difference-delta
        if reference is not None:
            desired=(difference@reference)@reference.T
            delta=delta*desired.norm(dim=1,keepdim=True)/delta.norm(dim=1,keepdim=True).clamp_min(1e-12)
            assert float((delta.norm(dim=1)-desired.norm(dim=1)).abs().max())<1e-3
        logits=patched_logits(pack,pp,site,base+delta);top=logits.argmax(-1)
        tok=pack['tok'];wanted=torch.tensor(pack['m']['label'][b],device=DEVICE)
        desired_tokens=torch.where(wanted==1,tok.yes,tok.no)
        valid=(top==tok.yes)|(top==tok.no);success=top==desired_tokens
        margin=logits[:,tok.yes]-logits[:,tok.no]
        results.extend(zip(success.cpu().tolist(),valid.cpu().tolist(),wanted.cpu().tolist(),margin.cpu().tolist()))
        norms.extend(delta.norm(dim=1).cpu().tolist())
        ratios.extend((delta.norm(dim=1)/difference.norm(dim=1).clamp_min(1e-12)).cpu().tolist())
    values=np.array(results);yes=values[:,2]==1
    return dict(n=len(panel),success=float(values[:,0].mean()),yes_success=float(values[yes,0].mean()),
        no_success=float(values[~yes,0].mean()),valid_answer_rate=float(values[:,1].mean()),
        desired_margin=float((values[:,3]*(2*values[:,2]-1)).mean()),
        edit_norm=float(np.mean(norms)),edit_to_full_difference_ratio=float(np.mean(ratios)))

def controls(pack,panel,site,x,u):
    result={}
    for method in ('swap','complement','full','noop'):
        result[method]=evaluate(pack,panel,site,x,u,method)
    random=[]
    for seed in RANDOM_SEEDS:
        q=np.linalg.qr(np.random.default_rng(seed).normal(size=(128,u.shape[1])))[0].astype(np.float32)
        random.append(evaluate(pack,panel,site,x,torch.tensor(q,device=DEVICE),reference=u))
    result['random_equal_norm']=random
    return result

def train(pack,panels,site,x,rank,seed):
    """Optimize only A; QR gives an orthonormal global basis with exact rank r.

    Edit uses donor coefficients, no learned per-query scaling or free offset.
    Best state selected at fixed checkpoints 100/200/300 by val native argmax.
    """
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    parameter=torch.nn.Parameter(torch.randn(128,rank,device=DEVICE)*.1)
    optimizer=torch.optim.Adam([parameter],lr=.02)
    train_pairs=panels[0];best=None;best_score=(-1.,-np.inf);history=[]
    for step in range(1,STEPS+1):
        pp=train_pairs[rng.integers(len(train_pairs),size=64)];a,b=pp.T
        u=torch.linalg.qr(parameter,mode='reduced')[0]
        edited=x[a]+((x[b]-x[a])@u)@u.T
        logits=patched_logits(pack,pp,site,edited)
        wanted=torch.tensor(pack['m']['label'][b],device=DEVICE)
        tokens=torch.where(wanted==1,pack['tok'].yes,pack['tok'].no)
        loss=F.cross_entropy(logits,tokens)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_([parameter],1.)
        optimizer.step()
        if step%100==0:
            with torch.no_grad():q=torch.linalg.qr(parameter,mode='reduced')[0].detach()
            scores=evaluate(pack,panels[1],site,x,q)
            key=(scores['success'],scores['desired_margin'])
            history.append(dict(step=step,training_loss=float(loss.detach()),validation=scores))
            if key>best_score:best_score=key;best=q.clone();chosen=step
    assert float((best.T@best-torch.eye(rank,device=DEVICE)).abs().max())<1e-5
    assert all(p.grad is None for p in pack['model'].parameters())
    return best,dict(seed=seed,steps=STEPS,chosen_step=chosen,history=history)

def run(setting):
    path=HERE/f'{setting}.json'
    pack=old.load(setting,device='cpu');features=capture(pack)
    # Previous loader constructs inference-mode tensors. Rebuild outside that
    # mode so autograd can differentiate INPUTS through frozen LayerNorms.
    model=old.build_model(pack['cfg'].model,len(pack['tok']),pack['tok'].pad).eval().requires_grad_(False)
    model.load_state_dict({k:v.detach().clone() for k,v in pack['model'].state_dict().items()})
    pack['model']=model;original=model.embed.weight.detach().clone()
    model.to(DEVICE);pack['ids']=pack['ids'].clone().to(DEVICE)
    features={site:torch.tensor(x,device=DEVICE) for site,x in features.items()}
    out=dict(manifest=pack['manifest'],ranks=RANKS,candidates=[],heldout_nodes=[],checks={},
        scope='Frozen old N100 Direct; learned global activation subspace, not a graph-semantic decoder.',
        train='Only orthonormal projection learned; GPU0, FP32, natural donor coefficients; no edit amplification.',
        selection='Validation only. Prefer smallest rank reaching >=90% success and >=20pp over equal-norm random; otherwise maximize val excess over random, tie favors lower rank.',
        warning='Query/node pairs share endpoints; no independent-model significance claims. Last-layer separator is an answer-writing control.')
    panels={role:[pairs(pack,role,part,limit) for part,limit in enumerate((2048,512,1024))] for role in ('source','target')}
    out['panels']={role:[meta(pack,p,role) for p in pp] for role,pp in panels.items()}
    for role,pp in panels.items():
        assert all(len(p)>0 for p in pp)
        assert not (set(pp[0].flat)&set(pp[1].flat)) and not (set(pp[0].flat)&set(pp[2].flat))
    winner=None;winner_key=None
    for site_i,site in enumerate(SITES):
        pp=panels[site[0]];x=features[site]
        for rank in RANKS:
            u,training=train(pack,pp,site,x,rank,seed=1301+site_i*20+rank)
            validation=controls(pack,pp[1],site,x,u)
            test=controls(pack,pp[2],site,x,u)
            random=np.mean([v['success'] for v in validation['random_equal_norm']])
            score=validation['swap']['success'];gap=score-random;qualifies=score>=.9 and gap>=.2
            key=(int(qualifies),-rank if qualifies else gap,gap if qualifies else -rank,score)
            entry=dict(site=list(site),rank=rank,training=training,validation=validation,test=test)
            out['candidates'].append(entry)
            if winner_key is None or key>winner_key:winner_key=key;winner=(site,rank,u.clone(),len(out['candidates'])-1)
            print(setting,site,rank,'val',round(score,3),'test',round(test['swap']['success'],3),flush=True)
            # Compact partial final-result record enables inspection during run.
            old.write(path,out)
    site,rank,u,index=winner;out['selected_candidate_index']=index
    out['selected_site']=list(site);out['selected_rank']=rank
    # Fresh initialization at the preselected site/rank; do not select using test.
    q,training=train(pack,panels[site[0]],site,features[site],rank,seed=20261024)
    out['fresh_initialization']=dict(training=training,validation=controls(pack,panels[site[0]][1],site,features[site],q),
        test=controls(pack,panels[site[0]][2],site,features[site],q))
    for node_seed in (17,31,47):
        pp=[pairs(pack,site[0],part,limit,node_seed=node_seed) for part,limit in enumerate((2048,512,1024))]
        item=dict(node_seed=node_seed,panels=[meta(pack,p,site[0]) for p in pp])
        if min(map(len,pp))<8:item['skipped']='Fewer than 8 pairs in train/val/test; report support, no invented result.'
        else:
            q,training=train(pack,pp,site,features[site],rank,seed=7001+node_seed)
            item.update(training=training,validation=controls(pack,pp[1],site,features[site],q),test=controls(pack,pp[2],site,features[site],q))
        out['heldout_nodes'].append(item)
        print(setting,'node holdout',node_seed,'counts',[len(p) for p in pp],flush=True)
    # Predefined output-writing control; not counted among discovered sites.
    gamma=model.final_norm.weight.detach();direction=gamma*(model.lm_head.weight[pack['tok'].yes]-model.lm_head.weight[pack['tok'].no])
    direction=direction-direction.mean();direction=(direction/direction.norm())[:,None]
    out['last_separator_control']=controls(pack,panels['target'][2],('separator','6F'),features['separator','6F'],direction)
    out['checks']['embedding_and_tied_head_unchanged']=bool(torch.equal(model.embed.weight.cpu(),original))
    assert out['checks']['embedding_and_tied_head_unchanged']
    assert model.lm_head.weight.data_ptr()==model.embed.weight.data_ptr()
    out['complete']=True;old.write(path,out);print(setting,'COMPLETE',flush=True)

if __name__=='__main__':
    HERE.mkdir(exist_ok=True)
    for setting in ('simple','hard'):run(setting)

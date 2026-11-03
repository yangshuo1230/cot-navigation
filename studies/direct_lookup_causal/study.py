"""Direct fixed-graph lookup hypothesis: readouts and causal interchange.

Frozen FP32 native SDPA on GPU1, ridge fitting on GPU0. No model training or
activation/checkpoint files. Interchange squares have crossed graph labels:
R(sa,ta)=R(sb,tb) != R(sa,tb)=R(sb,ta). Thus donor answer copying predicts the
WRONG counterfactual. Both recipient target queries belong to val/test; donor
queries may belong to any original split, explicitly recorded in summaries.
Low-rank semantic subspaces are diagnostics, not automatically disentangled
concepts: identity retention, shuffled-label and equal-norm controls are reported.
"""
from pathlib import Path
import os,sys,json,gzip,hashlib,itertools
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import numpy as np
import torch
torch.set_num_threads(2)
ROOT=Path.cwd();HERE=ROOT/'studies/direct_lookup_causal'
PREV=ROOT/'studies/fixed100_layer_probes'
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'studies/path_direction'),str(PREV)]
from analyze import fit as ridge_predictions, ranking, write
from ancestor_readout import evaluate,masks
import linear
from cot_navigation.config import load_config
from cot_navigation.data.dataset import ParametricWorld
from cot_navigation.models.transformer import build_model
from cot_navigation.models.attention import apply_rope
from cot_navigation.protocols.tokenizer import Tokenizer
RANKS=(1,2,4,8,16,32,64)

def coefficient_fit(x,y,split):
    """Validation-selected affine ridge in ORIGINAL residual coordinates.

    SVD on this W defines Euclidean activation directions; all fits GPU0.
    Train-only scaling. A single regularizer per layer minimizes val MSE.
    """
    if x.ndim==2:x=x[:,None,:]
    tr,va,te=split;z=torch.tensor(x,device='cuda:0').transpose(0,1)
    mean=z[:,tr].mean(1,keepdim=True);sd=z[:,tr].std(1,keepdim=True).clamp(min=.05)
    norm=(z-mean)/sd;norm=torch.cat([norm,torch.ones((*norm.shape[:2],1),device=z.device)],-1)
    yy=torch.tensor(y,device=z.device);zt=norm[:,tr]
    gram=zt.transpose(1,2)@zt/len(tr);rhs=zt.transpose(1,2)@yy[tr]/len(tr)
    eye=torch.eye(norm.shape[-1],device=z.device);eye[-1,-1]=0
    best=np.full(x.shape[1],np.inf);ws=[None]*x.shape[1];bs=[None]*x.shape[1];penalties=[None]*x.shape[1]
    for penalty in (.001,.01,.1,1.,10.):
        w=torch.linalg.solve(gram+penalty*eye+1e-6*torch.eye(norm.shape[-1],device=z.device),rhs)
        pred=(norm[:,va]@w).transpose(0,1)
        losses=((pred-yy[va,None,:])**2).mean((0,2)).cpu().numpy()
        for l,loss in enumerate(losses):
            if loss<=best[l]:
                best[l]=loss;ww=w[l,:-1]/sd[l,0,:,None]
                bb=w[l,-1]-mean[l,0]@ww
                ws[l]=ww.cpu().numpy();bs[l]=bb.cpu().numpy();penalties[l]=penalty
    return np.stack(ws),np.stack(bs),penalties

@torch.inference_mode()
def load(setting,device='cuda:1'):
    manifest=json.loads((PREV/f'{setting}_direct.json').read_text())['manifest']
    run=Path(manifest['run']);ck=run/'checkpoints/best.pt'
    assert hashlib.sha256(ck.read_bytes()).hexdigest()==manifest['checkpoint_sha256']
    cfg=load_config([run/'resolved_config.yaml']);tok=Tokenizer.from_config(cfg)
    world=ParametricWorld(Path(cfg.data.root)/'main-seed0');adj=world.adj();nodes=world.node_vocab_ids(tok)
    n=len(adj);reach=np.eye(n,dtype=bool);depth=np.zeros(n,int)
    for u in range(n):
        for v in adj[u]:assert v>u;depth[v]=max(depth[v],depth[u]+1)
    for u in reversed(range(n)):
        for v in adj[u]:reach[u]|=reach[v]
    for u in range(n):
        seen={u};todo=[u]
        for v in todo:
            for w in adj[v]:
                if w not in seen:seen.add(w);todo.append(w)
        assert seen==set(np.flatnonzero(reach[u]))
    reach[np.arange(n),np.arange(n)]=False
    rows=[json.loads(l) for l in gzip.open(PREV/f'{setting}_direct_queries.jsonl.gz','rt')]
    m={k:np.array([r[k] for r in rows]) for k in ('query_ids','source','target','split','label','pred')}
    assert np.array_equal(reach[m['source'],m['target']],m['label'].astype(bool))
    header=[tok.id('<WORLD_0>')] if cfg.train.world_roots else []
    ids=torch.tensor([header+[tok.id('<QUERY>'),nodes[s],nodes[t],tok.id('|')] for s,t in zip(m['source'],m['target'])],device=device)
    model=build_model(cfg.model,len(tok),tok.pad).to(device).eval().requires_grad_(False)
    model.load_state_dict(torch.load(ck,map_location='cpu',weights_only=False)['model'])
    x=[];margins=[];preds=[]
    for start in range(0,len(rows),128):
        cap={};tokens=ids[start:start+128];logits=model(tokens,capture=cap)
        hidden=torch.stack([model.embed(tokens)]+cap['resid'])[:,:,-3:]
        x.append(hidden.permute(1,2,0,3).cpu().numpy())
        margins.append((logits[:,-1,tok.yes]-logits[:,-1,tok.no]).cpu().numpy())
        top=logits[:,-1].argmax(-1)
        preds.append(torch.where(top==tok.yes,1,torch.where(top==tok.no,0,-1)).cpu().numpy())
    x=np.concatenate(x);margin=np.concatenate(margins)
    m['native_pred']=np.concatenate(preds)
    emb=model.embed.weight.detach().cpu().numpy()[nodes]
    adjacency=np.zeros((n,n),np.float32)
    for u in range(n):adjacency[u,adj[u]]=1
    return dict(manifest=manifest,cfg=cfg,tok=tok,model=model,ids=ids,x=x,m=m,margin=margin,
        reach=reach,adjacency=adjacency,depth=depth,nodes=nodes,embed=emb)

def squares(pack,split,limit=384):
    """Graph-defined selection, never consult model predictions or probe fits.

    Recipients (sa,ta) and their counterfactuals (sa,tb) must be held-out.
    Donors (sb,tb),(sb,ta) are natural queries in any split. Max two uses of
    each recipient to limit replication. Include a strict all-four-held-out
    panel and targets matched for generator band AND actual longest depth.
    """
    m=pack['m'];lookup={(s,t):i for i,(s,t) in enumerate(zip(m['source'],m['target']))}
    candidates=[];sources=np.unique(m['source']);width=next(iter(pack['cfg'].data.shapes.values())).width
    for sa in sources:
        targets=m['target'][(m['source']==sa)&(m['split']==split)]
        for ta,tb in itertools.combinations(targets,2):
            a=lookup[sa,ta];b=lookup[sa,tb]
            if m['label'][a]==m['label'][b]:continue
            for sb in sources:
                if sb==sa or (sb,tb) not in lookup or (sb,ta) not in lookup:continue
                d=lookup[sb,tb];c=lookup[sb,ta]
                if m['label'][d]!=m['label'][a] or m['label'][c]!=m['label'][b]:continue
                candidates.append((a,b,d,c))
                candidates.append((b,a,c,d))
    rng=np.random.default_rng(901+split);rng.shuffle(candidates)
    # Reserve budget separately for the more informative matched-depth panel.
    matched=[];other=[]
    for item in candidates:
        a,b,d,c=item;ta=m['target'][a];tb=m['target'][b]
        (matched if ta//width==tb//width and pack['depth'][ta]==pack['depth'][tb] else other).append(item)
    selected=[];uses={};seen=set()
    for group,cap in ((matched,limit//2),(other,limit)):
        for item in group:
            a,b,d,c=item
            if (a,b) in seen or uses.get(a,0)>=2:continue
            selected.append(item);seen.add((a,b));uses[a]=uses.get(a,0)+1
            if len(selected)>=cap:break
    if not selected:raise ValueError('No crossed interchange squares')
    ar=np.asarray(selected,int);a,b,d,c=ar.T
    matched=np.array([m['target'][i]//width==m['target'][j]//width and pack['depth'][m['target'][i]]==pack['depth'][m['target'][j]] for i,j in zip(a,b)])
    return dict(indices=ar,matched=matched,all_four_heldout=np.all(m['split'][ar]==split,axis=1),
        all_four_correct=np.all(m['native_pred'][ar]==m['label'][ar],axis=1))

def readouts(pack):
    x=pack['x'];m=pack['m'];n=len(pack['nodes']);nl=x.shape[2]
    sp=[np.flatnonzero(m['split']==i) for i in range(3)]
    y=pack['reach'][:,m['target']].T.astype(np.float32)
    ancestor_masks=masks(m['target'],m['source'],n)
    random=np.random.default_rng(2026).normal(0,.02,(n,x.shape[-1])).astype(np.float32)
    out={'ancestor_target_heldout':[],'successor_source_heldout':[]}
    for seed in (17,31,47):
        order=np.random.default_rng(seed).permutation(n);fold=np.zeros(n,int)
        fold[order[int(.6*n):int(.8*n)]]=1;fold[order[int(.8*n):]]=2
        splits=[np.flatnonzero((m['split']==i)&(fold[m['target']]==i)) for i in range(3)]
        entry=dict(seed=seed,query_counts=list(map(len,splits)),target_counts=[len(np.unique(m['target'][z])) for z in splits],features={})
        features={'target_and_separator':x[:,1:].reshape(len(x),2*nl,-1),
            'random_target_identity':random[m['target']]}
        for name,xx in features.items():
            p=ridge_predictions(xx,y,splits)[0]
            entry['features'][name]=evaluate(y,p,splits,ancestor_masks,m['target'])
        out['ancestor_target_heldout'].append(entry)
        source_split=[np.flatnonzero((m['split']==i)&(fold[m['source']]==i)) for i in range(3)]
        ys=pack['adjacency'][m['source']];te=source_split[2]
        p=ridge_predictions(x[:,0],ys,source_split)[0]
        out['successor_source_heldout'].append(dict(seed=seed,query_counts=list(map(len,source_split)),
            layers=[ranking(ys[te],p[te,l]) for l in range(nl)]))
        print('heldout readouts',seed,'done',flush=True)
    # Query-split diagnostics also define causal projections. They receive
    # extra oracle ancestor labels, so are NOT independent reachability solvers.
    w,b,reg=coefficient_fit(x[:,1],y,sp)
    identity_w,identity_b,_=coefficient_fit(x[:,1],np.eye(n,dtype=np.float32)[m['target']],sp)
    perm=np.random.default_rng(117).permutation(n)
    shuffled=pack['reach'][:,perm[m['target']]].T.astype(np.float32)
    shuffled_w,_,_=coefficient_fit(x[:,1],shuffled,sp)
    bases=[];shuffled_bases=[];random_bases=[]
    for l in range(nl):
        bases.append(np.linalg.svd(w[l],full_matrices=False)[0])
        shuffled_bases.append(np.linalg.svd(shuffled_w[l],full_matrices=False)[0])
        random_bases.append(np.linalg.qr(np.random.default_rng(110+l).normal(size=(x.shape[-1],x.shape[-1])))[0].astype(np.float32))
    out['ancestor_subspace_fit']=dict(ridge=reg,n_train=len(sp[0]),n_val=len(sp[1]),
        warning='Supervised diagnostic subspace fitted to full graph ancestor labels on query-train goals; identity and ancestry may overlap.')
    return out,dict(w=w,b=b,identity_w=identity_w,identity_b=identity_b,bases=bases,shuffled_bases=shuffled_bases,random_bases=random_bases)

@torch.inference_mode()
def forward_patch(pack,indices,layer,patched=None,donor_v=None,position=-2,sender_position=-2,receiver_position=-1):
    """Patch one residual token, or ONLY the target->answer attention edge.

    layer=0 means embedding output; layer=L means block L output. Edge uses
    the input to block layer+1, recipient Q/K and all other attention edges.
    Returns the YES-NO margin. Hooks never modify shared training code.
    """
    model=pack['model'];tok=pack['tok'];result=[];valid=[]
    for start in range(0,len(indices),128):
        ix=indices[start:start+128];tokens=pack['ids'][ix];handles=[]
        if donor_v is None:
            values=torch.as_tensor(patched[start:start+128],device=next(model.parameters()).device)
            def hook(module,args,output):
                if isinstance(output,tuple):
                    xx=output[0].clone();xx[:,position]=values;return (xx,)+output[1:]
                xx=output.clone();xx[:,position]=values;return xx
            module=model.embed if layer==0 else model.blocks[layer-1]
            handles=[module.register_forward_hook(hook)]
        else:
            values=torch.as_tensor(donor_v[start:start+128],device=next(model.parameters()).device)
            def edge_hook(module,args,output):
                xx,positions=args[:2];batch,length,_=xx.shape
                qkv=module.qkv(xx).view(batch,length,3,module.n_heads,module.head_dim)
                q,k,v=qkv.unbind(2)
                q=apply_rope(q.transpose(1,2),positions,module.rope_base)
                k=apply_rope(k.transpose(1,2),positions,module.rope_base)
                receiver=receiver_position%length
                score=q[:,:,receiver:receiver+1]@k.transpose(-2,-1)/module.head_dim**.5
                score=score.masked_fill(torch.arange(length,device=xx.device)>receiver,float('-inf'))
                weight=score.softmax(-1)[:,:,:,sender_position]
                dv=values.view(batch,module.n_heads,module.head_dim)-v[:,sender_position]
                change=(weight*dv).reshape(batch,-1)
                change=torch.nn.functional.linear(change,module.proj.weight,None)
                delta=output[0].clone();delta[:,receiver_position]+=change
                return (delta,output[1])
            handles=[model.blocks[layer].attn.register_forward_hook(edge_hook)]
        try:
            logits=model(tokens);result.append((logits[:,-1,tok.yes]-logits[:,-1,tok.no]).cpu().numpy())
            top=logits[:,-1].argmax(-1);valid.append(((top==tok.yes)|(top==tok.no)).cpu().numpy())
        finally:
            for handle in handles:handle.remove()
    return np.concatenate(result),np.concatenate(valid)

@torch.inference_mode()
def values(pack,indices,layer,role=1):
    model=pack['model'];xx=torch.tensor(pack['x'][indices,role,layer],device=next(model.parameters()).device)
    block=model.blocks[layer];d=model.d_model
    return block.attn.qkv(block.norm1(xx))[:,2*d:].cpu().numpy()

def metrics(pack,panel,margins,patched=None,layer=None,coefs=None,valid=None):
    a,b,d,c=panel['indices'].T;m=pack['m'];desired=m['label'][b];pred=margins>0
    base=pack['margin'][a];base_correct=(base>0)==m['label'][a]
    if valid is None:valid=m['native_pred'][a]>=0
    allcorrect=panel['all_four_correct'];out={}
    groups={'all':np.ones(len(a),bool),'all_four_correct':allcorrect,
        'matched_depth_band':panel['matched']&allcorrect,'all_four_heldout':panel['all_four_heldout']&allcorrect}
    if patched is not None:
        w=coefs['identity_w'][layer];bb=coefs['identity_b'][layer]
        old=(pack['x'][a,1,layer]@w+bb).argmax(-1)
        new=(patched@w+bb).argmax(-1)
        groups['identity_retained']=allcorrect&(old==m['target'][a])&(new==m['target'][a])
        out['identity']=dict(base_correct=float((old==m['target'][a]).mean()),recipient_retained=float((new==m['target'][a]).mean()),
            donor_adopted=float((new==m['target'][b]).mean()))
        delta=patched-pack['x'][a,1,layer]
        out['mean_edit_norm']=float(np.linalg.norm(delta,axis=1).mean())
        before=(pack['x'][a,1,layer]@coefs['w'][layer]+coefs['b'][layer])[np.arange(len(a)),m['source'][a]]
        after=(patched@coefs['w'][layer]+coefs['b'][layer])[np.arange(len(a)),m['source'][a]]
        out['ancestor_reader_desired_score_shift']=float(((after-before)*(2*desired-1)).mean())
    for name,mask in groups.items():
        out[name]=dict(n=int(mask.sum()),counterfactual_accuracy=float(((pred[mask]==desired[mask])&valid[mask]).mean()) if mask.any() else None,
            valid_answer_token_fraction=float(valid[mask].mean()) if mask.any() else None,
            fraction_answer_changed=float((pred[mask]!=(base[mask]>0)).mean()) if mask.any() else None,
            desired_margin_shift=float(((margins-base)*(2*desired-1))[mask].mean()) if mask.any() else None)
    return out

def panel_info(pack,panel):
    m=pack['m'];a,b,d,c=panel['indices'].T
    return dict(n=len(a),unique_recipient_queries=len(set(a)),unique_sources=len(set(m['source'][a])),
        unique_targets=len(set(m['target'][a])|set(m['target'][b])),matched_depth_band=int(panel['matched'].sum()),
        all_four_heldout=int(panel['all_four_heldout'].sum()),all_four_correct=int(panel['all_four_correct'].sum()),
        recipient_positive_fraction=float(m['label'][a].mean()),
        same_graph_and_length=True,donor_answer_always_equals_recipient_original=True,
        note='Pairs share nodes and queries; percentages are descriptive, not independent sample estimates.')

def interventions(pack,coefs):
    nl=pack['x'].shape[2];panels={name:squares(pack,split) for name,split in [('val',1),('test',2)]}
    result={'panels':{name:panel_info(pack,p) for name,p in panels.items()},'curves':[],'checks':{}}
    model=pack['model'];saved_examples=[]
    # Native donor values are computed once for every layer; no model changes.
    for name,panel in panels.items():
        a,b,d,c=panel['indices'].T
        result['curves'].append(dict(split=name,method='no_op',layer=0,rank=0,
            result=metrics(pack,panel,pack['margin'][a])))
        for l in range(nl):
            base=pack['x'][a,1,l];donor=pack['x'][d,1,l];diff=donor-base
            # Full target transfer is a compositional positive control.
            for method,edited in [('full_target',donor),('same_target_other_source',pack['x'][c,1,l])]:
                margin,valid=forward_patch(pack,a,l,edited)
                stat=metrics(pack,panel,margin,edited,l,coefs,valid)
                result['curves'].append(dict(split=name,method=method,layer=l,rank=0,result=stat))
                if l==0 and method=='full_target':
                    error=float(np.max(abs(margin-pack['margin'][b])))
                    assert error<1e-4;result['checks'][name+'_embedding_swap_equals_counterfactual_logits']=error
                if l==nl-1:
                    error=float(np.max(abs(margin-pack['margin'][a])))
                    assert error<1e-4;result['checks'][name+'_last_layer_target_cannot_change_answer']=error
            if l==nl-1:continue
            dv=values(pack,d,l);margin,valid=forward_patch(pack,a,l,donor_v=dv)
            result['curves'].append(dict(split=name,method='target_to_answer_value_edge',layer=l+1,rank=0,
                result=metrics(pack,panel,margin,valid=valid)))
            # True no-op edge message must exactly preserve the original answer.
            if l==0:
                same,_=forward_patch(pack,a[:16],l,donor_v=values(pack,a[:16],l))
                err=float(np.max(abs(same-pack['margin'][a[:16]])))
                assert err<1e-4;result['checks'][name+'_same_value_edge_noop_error']=err
            for rank in RANKS:
                u=coefs['bases'][l][:,:rank];ur=coefs['random_bases'][l][:,:rank];ush=coefs['shuffled_bases'][l][:,:rank]
                delta=diff@u@u.T
                rd=diff@ur@ur.T
                rd*=np.linalg.norm(delta,axis=1,keepdims=True)/np.maximum(np.linalg.norm(rd,axis=1,keepdims=True),1e-12)
                variants={'ancestor_subspace':base+delta,'ancestor_complement':donor-delta,
                    'random_equal_norm':base+rd,'shuffled_ancestor_subspace':base+diff@ush@ush.T}
                for method,edited in variants.items():
                    margin,valid=forward_patch(pack,a,l,edited)
                    result['curves'].append(dict(split=name,method=method,layer=l,rank=rank,
                        result=metrics(pack,panel,margin,edited,l,coefs,valid)))
                    if name=='test' and method=='ancestor_subspace' and rank==8 and len(saved_examples)<12:
                        yes=panel['all_four_correct']&((margin>0)==pack['m']['label'][b])&valid
                        for j in np.flatnonzero(yes)[:3]:
                            tok=pack['tok'];nodes=pack['nodes'];m=pack['m']
                            saved_examples.append(dict(layer=l,rank=rank,source=tok.name(nodes[m['source'][a[j]]]),
                                old_target=tok.name(nodes[m['target'][a[j]]]),new_target=tok.name(nodes[m['target'][b[j]]]),
                                donor_source=tok.name(nodes[m['source'][d[j]]]),old_label=int(m['label'][a[j]]),
                                new_label=int(m['label'][b[j]]),donor_label=int(m['label'][d[j]]),
                                old_margin=float(pack['margin'][a[j]]),patched_margin=float(margin[j]),
                                matched_depth_band=bool(panel['matched'][j])))
            print(name,'causal layer',l,'done',flush=True)
    # Select ONLY on validation. Test curves are descriptive; primary config is
    # selected before its test metric is consulted. Favor lower rank on ties.
    candidates=[]
    for item in result['curves']:
        if item['split']!='val' or item['method']!='ancestor_subspace':continue
        l,r=item['layer'],item['rank'];controls=[]
        for method in ('random_equal_norm','shuffled_ancestor_subspace'):
            other=next(z for z in result['curves'] if z['split']=='val' and z['method']==method and z['layer']==l and z['rank']==r)
            controls.append(other['result']['all_four_correct']['counterfactual_accuracy'])
        success=item['result']['all_four_correct']['counterfactual_accuracy']
        candidates.append((success-np.mean(controls),-r,-l,l,r))
    _,_,_,l,r=max(candidates)
    result['validation_selected']=dict(layer=l,rank=r,criterion='val all-four-correct counterfactual accuracy minus mean(random equal-norm, shuffled ancestor)',
        comparisons=[z for z in result['curves'] if z['split']=='test' and z['layer']==l and (z['rank']==r or z['method'] in ('full_target','same_target_other_source'))])
    result['examples']=saved_examples
    return result

def run(setting):
    pack=load(setting);m=pack['m'];te=m['split']==2
    result=dict(manifest=pack['manifest'],precision='FP32 native SDPA frozen',
        native_test=dict(n=int(te.sum()),accuracy=float((m['native_pred'][te]==m['label'][te]).mean()),
            label_token_fraction=float((m['native_pred'][te]>=0).mean()),
            agreement_with_prior_bf16=float((m['native_pred'][te]==m['pred'][te]).mean())),
        task='N100 fixed DAG, original forward query partitions, best validation checkpoint')
    probe,coefs=readouts(pack);result['readouts']=probe
    result['causal']=interventions(pack,coefs)
    write(HERE/f'{setting}.json',result)
    del pack;torch.cuda.empty_cache();print(setting,'COMPLETE',flush=True)

if __name__=='__main__':
    HERE.mkdir(parents=True,exist_ok=True)
    for setting in sys.argv[1:] or ('simple','hard'):run(setting)

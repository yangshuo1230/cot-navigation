"""Does fixed-DAG PATH expose reachability BEFORE any trace token?

Original heldout queries; CPU FP32 frozen inference; balanced affine answer
probes on GPU0. Compare prompt separator, target, joint endpoint states, and
the first copied-source step (no graph edge traversed). Input identity and
shuffled-label controls; both-endpoint-heldout sensitivity checks.
"""
from pathlib import Path
import os,json,importlib.util
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import numpy as np
import torch
torch.set_num_threads(2)
ROOT=Path.cwd();OUT=ROOT/'studies/path_before_trace'
spec=importlib.util.spec_from_file_location('storage',ROOT/'studies/fixed_graph_storage/study.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
import linear

@torch.inference_mode()
def collect(p):
    tok=p['tok'];model=p['model'];rows=p['rows'];cap={};observed={};handles=[]
    def before(layer):
        def hook(module,args):observed[layer,'input']=args[0]
        return hook
    def after(layer):
        def hook(module,args,output):observed[layer,'attention']=observed[layer,'input']+output[0]
        return hook
    for l,block in enumerate(model.blocks):
        handles.extend([block.register_forward_pre_hook(before(l)),block.attn.register_forward_hook(after(l))])
    features={'prompt_separator':[],'prompt_target':[],'prompt_joint':[],'copied_source_separator':[]}
    raw=[];copy_raw=[];copied=0;observation_error=0
    try:
        for start in range(0,len(rows),128):
            sub=rows[start:start+128]
            seq=[p['header']+[tok.id('<QUERY>'),int(p['nodes'][r['source']]),int(p['nodes'][r['target']]),tok.id('|')] for r in sub]
            ids=torch.tensor(seq);native=model(ids,capture=cap)
            states=[model.embed(ids)]
            for l in range(6):states.extend([observed[l,'attention'],cap['resid'][l]])
            hidden=torch.stack(states,1)
            features['prompt_separator'].append(hidden[:,:,-1].numpy().copy())
            features['prompt_target'].append(hidden[:,:,-2].numpy().copy())
            features['prompt_joint'].append(hidden[:,:,-3:].reshape(len(ids),13,-1).numpy().copy())
            lens=model.lm_head(model.final_norm(hidden[:,:,-1]))
            assert torch.equal(lens[:,-1],native[:,-1])
            raw.append(lens.numpy())
            first=native[:,-1].argmax(-1).numpy();sources=p['nodes'][[r['source'] for r in sub]]
            copied+=int((first==sources).sum())
            # Counterfactual fixed copy is natural only when saved/native first token agrees.
            assert np.array_equal(first,sources)
            ids_copy=torch.cat([ids,torch.tensor(sources[:,None])],1);cap={};out=model(ids_copy,capture=cap)
            copied_states=[model.embed(ids_copy)[:,-1]]
            for l in range(6):copied_states.extend([observed[l,'attention'][:,-1],cap['resid'][l][:,-1]])
            copied_hidden=torch.stack(copied_states,1)
            features['copied_source_separator'].append(copied_hidden.numpy().copy())
            copy_raw.append(model.lm_head(model.final_norm(copied_hidden)).numpy())
            # Causality: copied future token cannot change prompt states.
            observation_error=max(observation_error,float((cap['resid'][-1][:,-2]-hidden[:,-1,-1]).abs().max()))
            cap={}
    finally:
        for handle in handles:handle.remove()
    assert observation_error<1e-4
    return {name:np.concatenate(values) for name,values in features.items()},np.concatenate(raw),np.concatenate(copy_raw),dict(native_first_token_copies_source=copied,total=len(rows),future_copy_prompt_state_max_error=observation_error)

def metrics(y,p,fit,test):
    return [linear.classification(y[test],p[test,l,0],fit['threshold'][l][0]) for l in range(p.shape[1])]

def xor_squares(m,limit=256):
    """All FOUR queries must be original test; graph-selected, not model-selected."""
    lookup={(s,t):i for i,(s,t,sp) in enumerate(zip(m['source'],m['target'],m['split'])) if sp==2}
    candidates=[];sources=sorted(set(s for s,t in lookup))
    for j,sa in enumerate(sources):
        for sb in sources[j+1:]:
            targets=sorted(set(t for s,t in lookup if s==sa)&set(t for s,t in lookup if s==sb))
            groups={0:[],1:[]}
            for t in targets:
                a=lookup[sa,t];c=lookup[sb,t]
                if m['label'][a]!=m['label'][c]:groups[int(m['label'][a])].append(t)
            for ta in groups[0]:
                for tb in groups[1]:candidates.append([lookup[sa,ta],lookup[sa,tb],lookup[sb,tb],lookup[sb,ta]])
    np.random.default_rng(917).shuffle(candidates)
    chosen=[];uses={}
    for square in candidates:
        if any(uses.get(i,0)>=3 for i in square):continue
        chosen.append(square)
        for i in square:uses[i]=uses.get(i,0)+1
        if len(chosen)==limit:break
    return np.array(chosen,int).reshape(-1,4)

def run(setting):
    p=base.load(setting,'path','cpu');rows=p['rows'];features,raw,copy_raw,checks=collect(p)
    m={key:np.array([r[key] for r in rows]) for key in ('source','target','split','label','pred')};y=m['label'];n=len(p['nodes'])
    split=[np.flatnonzero(m['split']==i) for i in range(3)];test=split[2]
    embedding=p['model'].embed.weight.detach().numpy()[p['nodes']]
    features['endpoint_embeddings']=np.concatenate([embedding[m['source']],embedding[m['target']]],1)
    features['endpoint_onehot']=np.concatenate([np.eye(n)[m['source']],np.eye(n)[m['target']]],1).astype(np.float32)
    for seed in (201,303,407):
        vectors=np.random.default_rng(seed).normal(size=(n,128)).astype(np.float32)
        features[f'random_endpoints_{seed}']=np.concatenate([vectors[m['source']],vectors[m['target']]],1)
    squares=xor_squares(m);result=dict(manifest=p['manifest'],checks=checks,layers=['E']+[f'L{i}_{stage}' for i in range(1,7) for stage in ('attention','FFN')],
        method='Original query holdout balanced affine probes; train-only scaling, val AUROC selects ridge and val BA selects threshold. '
            'Prompt has no trace or answer token. Copy-source phase adds exactly the natural first node token and traverses no edge. '
            'Joint state concatenates source/target/separator at same stage. Random controls are fixed per-node vectors.',
        features={},node_heldout=[],native_prompt={},native_after_copy={},xor_panel=dict(squares=len(squares),unique_queries=len(set(squares.ravel().tolist())),
            all_queries_original_test=True,note='Each square has crossed labels. Additive endpoint-only threshold classifier cannot get all four right; pairs share queries.'))
    for name,x in features.items():
        pred,fit=linear.fit(x,y,split,binary=True);scores=metrics(y,pred,fit,test)
        entry=dict(fit=fit,metrics=scores)
        if len(squares):
            entry['xor_all_four_correct']=[float(((pred[squares,l,0]>=fit['threshold'][l][0])==y[squares]).all(1).mean()) for l in range(pred.shape[1])]
            entry['xor_balanced_accuracy']=[float(((pred[squares,l,0]>=fit['threshold'][l][0])==y[squares]).mean()) for l in range(pred.shape[1])]
        result['features'][name]=entry
    for name,scores in [('native_prompt',raw),('native_after_copy',copy_raw)]:
        entry=[]
        for l in range(13):
            margin=scores[:,l,p['tok'].yes]-scores[:,l,p['tok'].no]
            probs=torch.tensor(scores[:,l]).softmax(-1).numpy();argmax=scores[:,l].argmax(-1)
            entry.append(dict(forced_yesno=linear.classification(y[test],margin[test],0),
                full_vocab_yesno_mass=float((probs[test,p['tok'].yes]+probs[test,p['tok'].no]).mean()),
                native_argmax_is_yesno=float(np.isin(argmax[test],[p['tok'].yes,p['tok'].no]).mean())))
        result[name]=entry
    shuffled=y.copy();rng=np.random.default_rng(181)
    for subset in split:shuffled[subset]=rng.permutation(shuffled[subset])
    pred,fit=linear.fit(features['prompt_separator'],shuffled,split,binary=True)
    result['shuffled_prompt_separator']=metrics(shuffled,pred,fit,test)
    # Strict probe-only node generalization; main-model training saw all nodes.
    for seed in (17,31,47):
        order=np.random.default_rng(seed).permutation(n);fold=np.zeros(n,int);fold[order[60:80]]=1;fold[order[80:]]=2
        sp=[np.flatnonzero((m['split']==i)&(fold[m['source']]==i)&(fold[m['target']]==i)) for i in range(3)]
        record=dict(seed=seed,counts=list(map(len,sp)),positive_counts=[int(y[v].sum()) for v in sp],features={})
        if any(len(np.unique(y[v]))<2 for v in sp):record['skipped']='a partition lacks both labels'
        else:
            for name in ('prompt_separator','prompt_target','prompt_joint','copied_source_separator','endpoint_embeddings','endpoint_onehot','random_endpoints_201'):
                pred,fit=linear.fit(features[name],y,sp,binary=True)
                record['features'][name]=dict(fit=fit,metrics=metrics(y,pred,fit,sp[2]))
        result['node_heldout'].append(record)
    result['native_final_answer']=dict(n=len(test),accuracy=float((m['pred'][test]==y[test]).mean()),
        yes_recall=float((m['pred'][test[y[test]==1]]==1).mean()),no_recall=float((m['pred'][test[y[test]==0]]==0).mean()))
    prior=json.loads((ROOT/f'studies/fixed100_layer_probes/{setting}_path.json').read_text())['answers']['linear_readout']['prompt']['residual']['layers']
    result['checks']['FP32_vs_previous_BF16_prompt_BA_deltas']=[result['features']['prompt_separator']['metrics'][2*i]['balanced_accuracy']-prior[i]['all']['balanced_accuracy'] for i in range(7)]
    base.write(OUT/f'{setting}.json',result)
    print(setting,'COMPLETE',[(k,round(v['metrics'][-1]['balanced_accuracy'],4)) for k,v in result['features'].items()],flush=True)

if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    for setting in ('simple','hard'):run(setting)

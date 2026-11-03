"""Native PATH attention by trajectory phase, with train-only distance baseline.

CPU FP32 frozen forward, no fitting/training. Reconstruct QK+RoPE from actual
attention inputs, verify reconstructed value output equals native SDPA delta.
Only compact role aggregates saved, no full attention/activation files.
"""
from pathlib import Path
import os,importlib.util,json
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import torch
torch.set_num_threads(2)
ROOT=Path.cwd();OUT=ROOT/'studies/path_attention_progress'
spec=importlib.util.spec_from_file_location('storage',ROOT/'studies/fixed_graph_storage/study.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
from cot_navigation.models.attention import apply_rope
ROLES=['syntax','query source','query target','separator','earlier path','current node']

@torch.inference_mode()
def run(setting):
    p=base.load(setting,'path','cpu');tok=p['tok'];model=p['model'];rows=p['rows'];inv={int(v):i for i,v in enumerate(p['nodes'])}
    sequences=[p['header']+[tok.id('<QUERY>'),int(p['nodes'][r['source']]),int(p['nodes'][r['target']]),tok.id('|')]+r['trace'] for r in rows]
    max_length=max(map(len,sequences));sums=np.zeros((6,max_length));counts=np.zeros(max_length,int)
    observed={};reconstruction_error=0
    def before(layer):
        def hook(module,args):
            xx,pos=args[:2];batch,length,_=xx.shape
            q,k,v=module.qkv(xx).view(batch,length,3,module.n_heads,module.head_dim).unbind(2)
            q=apply_rope(q.transpose(1,2),pos,module.rope_base);k=apply_rope(k.transpose(1,2),pos,module.rope_base)
            score=q@k.transpose(-2,-1)/module.head_dim**.5
            weight=score.masked_fill(~torch.ones(length,length,dtype=torch.bool).tril(),float('-inf')).softmax(-1)
            observed[layer,'weights']=weight[:,0].numpy().copy()
            observed[layer,'delta']=module.proj((weight@v.transpose(1,2)).transpose(1,2).reshape(batch,length,-1))
        return hook
    def after(layer):
        def hook(module,args,output):
            nonlocal reconstruction_error
            reconstruction_error=max(reconstruction_error,float((output[0]-observed[layer,'delta']).abs().max()))
        return hook
    handles=[]
    for layer,block in enumerate(model.blocks):handles.extend([block.attn.register_forward_pre_hook(before(layer)),block.attn.register_forward_hook(after(layer))])
    groups={name:[] for name in ('mixed_all','source_copy','middle','last_before_goal')};plen=len(p['header'])+4
    try:
        for start in range(0,len(rows),64):
            sub=rows[start:start+64];seqs=sequences[start:start+64];length=max(map(len,seqs))
            ids=torch.full((len(sub),length),tok.pad,dtype=torch.long)
            for b,seq in enumerate(seqs):ids[b,:len(seq)]=torch.tensor(seq)
            model(ids);weights=np.stack([observed[layer,'weights'] for layer in range(6)])
            for b,row in enumerate(sub):
                L=len(seqs[b]);attention=weights[:,b,:L,:L]
                if row['split']==0:
                    for distance in range(L):
                        sums[:,distance]+=np.diagonal(attention,offset=-distance,axis1=-2,axis2=-1).sum(1)
                        counts[distance]+=L-distance
                if row['split']!=2 or not row['label']:continue
                path=[inv.get(v,-1) for v in row['trace']]
                legal=bool(path and path[0]==row['source'] and all(u>=0 for u in path))
                if legal:legal=all(p['a'][u,v] for u,v in zip(path,path[1:]))
                success=bool(legal and path[-1]==row['target'])
                h=len(path)-1
                anchors={'source_copy':0,'middle':(h-1)//2,'last_before_goal':h-1} if success else {}
                previous=None
                for step,u in enumerate(path):
                    if u<0 or (step==0 and u!=row['source']) or (previous is not None and not p['a'][previous,u]):break
                    previous=u
                    if u==row['target']:break
                    if not p['a'][u].any():continue
                    pos=plen+step;role=np.zeros(pos+1,int);role[plen-3:plen]=[1,2,3];role[plen:pos]=4;role[pos]=5
                    record=dict(query=row['query_ids'],weights=attention[:,pos,:pos+1].copy(),role=role,step=step)
                    good=p['a'][u]&p['reach'][:,row['target']]
                    if good.any() and (p['a'][u]&~good).any():groups['mixed_all'].append(record)
                    for name,j in anchors.items():
                        if j==step:groups[name].append(record)
    finally:
        for handle in handles:handle.remove()
    assert reconstruction_error<1e-4,reconstruction_error
    mean=sums/np.maximum(counts,1)[None,:];aggregates={}
    for name,records in groups.items():
        queries={}
        for r in records:
            length=len(r['role']);distance=length-1-np.arange(length)
            assert np.all(counts[distance]>0)
            raw=np.stack([r['weights'][:,r['role']==role].sum(1) for role in range(6)],1)
            centered=np.stack([(r['weights']-mean[:,distance])[:,r['role']==role].sum(1) for role in range(6)],1)
            assert np.max(abs(raw.sum(1)-1))<1e-5
            queries.setdefault(r['query'],[]).append((raw,centered))
        aggregates[name]=dict(decisions=len(records),queries=len(queries),
            raw_query_balanced=np.mean([np.mean([v[0] for v in values],0) for values in queries.values()],0).tolist(),
            corrected_query_balanced=np.mean([np.mean([v[1] for v in values],0) for values in queries.values()],0).tolist(),
            raw_decision_weighted=np.mean([v[0] for values in queries.values() for v in values],0).tolist(),
            corrected_decision_weighted=np.mean([v[1] for values in queries.values() for v in values],0).tolist())
    previous=json.loads((ROOT/f'studies/path_goal_preference/{setting}.json').read_text())['preference']['groups']['mixed_only']
    assert aggregates['mixed_all']['decisions']==previous['decisions']
    assert aggregates['mixed_all']['queries']==previous['queries']
    base.write(OUT/f'{setting}.json',dict(manifest=p['manifest'],roles=ROLES,groups=aggregates,
        checks=dict(reconstructed_attention_value_delta_max_error=reconstruction_error,
            train_only_distance_baseline=True,distance_counts=counts.tolist()),
        method='Single head per layer. Native CPU FP32, unchanged weights. Actual input QK+RoPE softmax reconstructed; native V-output checked. '
            'Train full causal token pairs, no padding or YES/NO, define distance baseline. '
            'Mixed group uses all positive test legal pre-goal mixed branches; phase groups use strict successful paths only, one anchor/query. '
            'Middle/last anchors chosen retrospectively; short paths may share positions. Corrected role sums are signed deviations, not probabilities.'))
    print(setting,'COMPLETE',[(name,v['queries']) for name,v in aggregates.items()],flush=True)

if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    for setting in ('simple','hard'):run(setting)

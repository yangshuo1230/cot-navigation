"""Recover exact cumulative item multiplicities from saved sampler state.

Every completed IndexSampler epoch contains each item once. Its current prefix
adds one count per consumed item. No historical shuffle replay is necessary.
Metadata is validated against checkpoint counters before calling this actual
training exposure. The knowledge parent is hashed and counted separately.
"""
from pathlib import Path
import json,hashlib
import numpy as np
import torch

def counts(state):
    if state.get('kind')=='scale_mix_v1':
        assert len(state['samplers'])==1
        return counts(state['samplers'][0])
    if 'sampler' in state:
        result=counts(state['sampler'])
        np.add.at(result,np.asarray(state['pending'],int),-1)
        assert np.all(result>=0);return result
    if state.get('kind')=='fixed_query_v1':
        size=max(max(v) for v in state['groups'].values())+1;result=np.zeros(size,np.int64)
        for key,ids in state['groups'].items():
            local=counts(state['samplers'][key]);assert len(ids)==len(local)
            np.add.at(result,np.array(ids),local)
        return result
    n=state['n_items'];result=np.full(n,int(state['epoch']),np.int64)
    prefix=np.asarray(state['order'],int)[:state['cursor']]
    assert np.array_equal(np.sort(state['order']),np.arange(n))
    np.add.at(result,prefix,1)
    assert result.sum()==int(state['epoch'])*n+state['cursor']
    return result

def recover(pack):
    from cot_navigation.data.dataset import ParametricWorld
    run=Path(pack['manifest']['run']);cfg=pack['cfg']
    world=ParametricWorld(Path(cfg.data.root)/'main-seed0');q=world.queries;n=world.n_nodes
    checkpoint=run/'checkpoints/best.pt'
    blob=torch.load(checkpoint,map_location='cpu',weights_only=False)
    state=blob['query_sampler'];query_counts=np.zeros(len(q['label']),np.int64)
    if state.get('kind')=='short_hop_quotas_v1':
        # Historical wrapper groups are LOCAL to multi-hop + one-hop train_ids,
        # then its explicit forward/reverse signed-gap filter. Reconstruct that
        # mapping and validate all six semantic strata, not just array lengths.
        train_ids=np.r_[world.indices('train','multi_hop'),world.indices('train','one_hop')]
        gap=q['signed_gap'][train_ids];maximum=3 if cfg.tracking.variant=='short3' else 9
        train_ids=train_ids[((gap>0)&(gap<=maximum))|((gap<0)&(-gap<=3))]
        delegate=state['delegate'];assert delegate['kind']=='scale_mix_v1'
        expected=[]
        for one in (True,False):
            hop=q['distance'][train_ids]==1 if one else q['distance'][train_ids]>=2
            for label,direction in ((1,1),(0,1),(0,-1)):
                expected.append(np.flatnonzero(hop&(q['label'][train_ids]==label)&(np.sign(q['signed_gap'][train_ids])==direction)))
        for ids,wanted,local_state in zip(state['groups'],expected,delegate['samplers']):
            assert np.array_equal(ids,wanted)
            weights=counts(local_state);assert len(ids)==len(weights)
            np.add.at(query_counts,train_ids[np.array(ids)],weights)
    else:
        train_ids=np.r_[world.indices('train','multi_hop'),world.indices('train','one_hop')]
        inner=state['samplers'][0] if state.get('kind')=='scale_mix_v1' else state
        if 'sampler' in inner:inner=inner['sampler']
        assert inner['kind']=='fixed_query_v1'
        for key,ids in inner['groups'].items():
            hop,label=key.split('/')
            mask=(q['distance'][train_ids]==1 if hop=='one' else q['distance'][train_ids]>=2)&(q['label'][train_ids]==int(label))
            assert np.array_equal(ids,np.flatnonzero(mask))
        weights=counts(state);assert len(weights)==len(train_ids),(len(weights),len(train_ids))
        np.add.at(query_counts,train_ids,weights)
    assert int(query_counts.sum())==int(blob['queries_seen'])
    assert np.all(q['split'][query_counts>0]==0)
    task_walk=counts(blob['walk_sampler'])
    assert int(task_walk.sum())==int(blob['walks_seen'])
    parent_info=json.loads((run/'init_from.json').read_text());parent=Path(parent_info['init_from'])
    digest=hashlib.sha256(parent.read_bytes()).hexdigest();assert digest==parent_info['init_sha256']
    knowledge=torch.load(parent,map_location='cpu',weights_only=False)
    assert knowledge['query_sampler'] is None and int(knowledge['queries_seen'])==0
    from cot_navigation.config import load_config
    parent_cfg=load_config([parent.parent.parent/'resolved_config.yaml'])
    parent_world=Path(parent_cfg.data.root)/f'{parent_cfg.data.graph_shape}-seed{parent_cfg.data.graph_seed}'
    for name in ('graph.npz','walks.npz'):
        assert hashlib.sha256((parent_world/name).read_bytes()).hexdigest()==hashlib.sha256((world.root/name).read_bytes()).hexdigest()
    parent_walk=counts(knowledge['walk_sampler'])
    assert int(parent_walk.sum())==int(knowledge['walks_seen'])
    assert len(task_walk)==len(parent_walk)==world.n_walks()
    attributes={}
    for role,column in [('source','s_topo'),('target','t_topo')]:
        ids=q[column].astype(int)
        total=np.bincount(ids,weights=query_counts,minlength=n)
        yes=np.bincount(ids,weights=query_counts*(q['label']==1),minlength=n)
        attributes[f'query_{role}_frequency']=total
        attributes[f'query_{role}_yes_rate']=(yes+1)/(total+2)  # Laplace; unqueried nodes -> 1/2.
    attributes['query_total_frequency']=attributes['query_source_frequency']+attributes['query_target_frequency']
    start=np.zeros(n);end=np.zeros(n);total=np.zeros(n)
    for index,weight in enumerate(task_walk+parent_walk):
        nodes=world.walk_nodes_at(index);assert len(nodes)>=2
        start[nodes[0]]+=weight;end[nodes[-1]]+=weight
        np.add.at(total,nodes,weight)
    attributes.update(walk_start_frequency=start,walk_end_frequency=end,walk_total_frequency=total)
    audit=dict(query_actual_count=int(query_counts.sum()),query_counter=int(blob['queries_seen']),
        task_walk_actual_count=int(task_walk.sum()),task_walk_counter=int(blob['walks_seen']),
        knowledge_walk_actual_count=int(parent_walk.sum()),knowledge_walk_counter=int(knowledge['walks_seen']),
        parent_sha256=digest,task_checkpoint_step=int(blob['step']),knowledge_step=int(knowledge['step']),
        node_names=[pack['tok'].name(int(v)) for v in pack['nodes']],
        definition='Cumulative query occurrences through best checkpoint; WALK appearances include hashed common Stage A parent + Stage B. '
                   'Sampler epochs/current prefixes, buffered pending corrected; all totals equal saved actual counters. '
                   'No per-update ordering claimed. Yes rates use Laplace (yes+1)/(total+2).')
    return attributes,audit

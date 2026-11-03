"""Oracle graph-structure readout baselines, never fed to the main model.

Separate whether embedding relation AUC exceeds rank/depth/band shortcuts.
Only closed-form GPU0 fits; main model loaded CPU, no frozen GPU inference.
"""
import run as study
import numpy as np

def run(setting):
    p=study.base.load(setting,'path','cpu');n=len(p['nodes'])
    height=np.zeros(n,int)
    for u in reversed(range(n)):
        if p['a'][u].any():height[u]=1+height[p['a'][u]].max()
    # On hard DAG this is the hidden generator rank band, not a unique layer.
    band=np.eye(10,dtype=np.float32)[np.arange(n)//10]
    rank=np.arange(n,dtype=np.float32)[:,None]/n
    features=dict(rank_only=rank,rank_band_onehot=band,
        graph_structure=np.column_stack([rank,p['depth']/n,height/n,p['a'].sum(0)/10,p['a'].sum(1)/10,band]).astype(np.float32))
    relations=dict(ancestors=p['reach'].T.copy(),descendants=p['reach'].copy(),successors=p['a'].copy())
    output={key:[] for key in relations}
    for role,y in relations.items():
        np.fill_diagonal(y,False);y=y.astype(np.float32)
        for seed in study.SEEDS:
            order=np.random.default_rng(seed).permutation(n);train,val,test=order[:60],order[60:80],order[80:]
            entry=dict(node_seed=seed)
            for name,x in features.items():
                w,b,penalty=study.fit(x,y,train,val);pred=x@w+b
                entry[name]=study.readout_metrics(y,pred,test,role!='ancestors',study.threshold(pred,y,val))
            output[role].append(entry)
    path=study.OUT/f'{setting}.json';data=study.json.loads(path.read_text())
    data['structural_controls']=output
    data['structural_control_note']='Oracle features not supplied to main model: topological rank, hidden generator rank-band onehot, rank+longest depth/height+in/out degree+band. Same 60/20/20 node split and ridge fitting. Not a causal proof the model uses these shortcuts.'
    study.base.write(path,data);print(setting,'STRUCTURAL CONTROLS COMPLETE',flush=True)

if __name__=='__main__':
    for setting in ('simple','hard'):run(setting)

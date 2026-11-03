"""Deterministic, bounded preparation; never starts training."""
from pathlib import Path
from collections import Counter
import random,json,hashlib
HERE=Path(__file__).resolve().parent

def build(seed):
    rng=random.Random(74000+seed)
    n=200;width=20
    adj=[[v for v in range((u//width+1)*width,min((u//width+2)*width,n)) if rng.random()<.05] for u in range(n)]
    names=list(range(n));rng.shuffle(names)
    paths={}
    for s in range(n):
        todo=[[s]]
        while todo:
            p=todo.pop()
            for t in adj[p[-1]]:
                q=p+[t];paths.setdefault((s,t),[]).append(q);todo.append(q)
                if sum(map(len,paths.values()))>1000000:raise RuntimeError('Path enumeration exceeds declared bound')
    edges={(u,v) for u,vs in enumerate(adj) for v in vs}
    positives=sorted(paths)
    negatives=[(s,t) for s in range(n) for t in range(n) if s!=t and (s,t) not in paths]
    # Independent reverse-topological reachability validates path enumeration.
    reach=[set() for _ in range(n)]
    for u in reversed(range(n)):
        for v in adj[u]:reach[u].add(v);reach[u].update(reach[v])
    assert set(paths)=={(u,v) for u in range(n) for v in reach[u]}
    worlds={}
    for mode in ('random20','short3'):
        r=random.Random(87000+seed)
        eligible=[p for p in positives if mode=='random20' or p[1]//width-p[0]//width<=3]
        selected=set(r.sample(eligible,max(1,round(.2*len(eligible)))))|edges
        neg_train=set(r.sample(negatives,len(selected)))
        train=[[s,t,1] for s,t in sorted(selected)]+[[s,t,0] for s,t in sorted(neg_train)]
        testpos=[p for p in positives if p not in selected and (mode=='random20' or p[1]//width-p[0]//width>=4)]
        # Same-layer forward nonedges and reverse-order pairs are easy controls;
        # report forward-layer subset separately to expose that shortcut.
        testneg=[p for p in negatives if p not in neg_train and (mode=='random20' or p[1]//width-p[0]//width>=4)]
        assert len(testpos)>=20 and len(testneg)>=len(testpos)
        r.shuffle(testpos);r.shuffle(testneg)
        m=max(1,len(testpos)//4)
        monitor=[[s,t,1] for s,t in testpos[:m]]+[[s,t,0] for s,t in testneg[:m]]
        final=[[s,t,1] for s,t in testpos[m:]]+[[s,t,0] for s,t in testneg[m:len(testpos)]]
        for a,b in ((train,monitor),(train,final),(monitor,final)):
            assert not ({tuple(x[:2]) for x in a}&{tuple(x[:2]) for x in b})
        sequences=[]
        for s,t,y in train:
            for p in paths[(s,t)] if y else [[]]:sequences.append(dict(s=s,t=t,y=y,path=p))
        assert all((u,v) in selected for u,v in edges)
        witnesses={tuple(p) for k in selected for p in paths[k]}
        disclosed={(p[i],p[j]) for p in witnesses for i in range(len(p)) for j in range(i+1,len(p))}
        worlds[mode]=dict(train=train,monitor=monitor,final=final,sequences=sequences,
            audit=dict(all_edges_in_query_train=True,n_train_pairs=len(train),n_train_sequences=len(sequences),
                labels_sequence=dict(Counter(str(x['y']) for x in sequences)),n_monitor=len(monitor),n_final=len(final),
                final_positive_subpath_exposed=sum((s,t) in disclosed for s,t,y in final if y),
                max_train_positive_hops=max(t//width-s//width for s,t in selected)))
    return dict(seed=seed,n_nodes=n,width=width,layers=10,adj=adj,node_names=names,n_edges=len(edges),
                n_paths=sum(map(len,paths.values())),splits=worlds,
                connectivity_note='Unconditioned layered Bernoulli p=.05; no connectivity rejection or repair.')

def main():
    (HERE/'data').mkdir(parents=True,exist_ok=True)
    jobs=[];audits=[]
    for seed in range(3):
        w=build(seed);p=HERE/'data'/f'graph_{seed}.json';p.write_text(json.dumps(w))
        audits.append(dict(graph=seed,n_edges=w['n_edges'],n_paths=w['n_paths'],splits={k:v['audit'] for k,v in w['splits'].items()}))
        for mode in ('random20','short3'):
            for protocol in ('direct','path'):
                jobs.append(dict(id=f'g{seed}/{mode}/{protocol}',graph=seed,model_seed=seed,mode=mode,protocol=protocol,
                    steps=125000,eval_interval=5000,checkpoint_interval=25000,batch_size=64,lr=1e-4,
                    layers=2,heads=4,width=64,ffn=256,context=32,dropout=0.,optimizer='Adam',betas=[.9,.95],weight_decay=0.,
                    data_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    (HERE/'manifest.json').write_text(json.dumps(dict(group='stepwise_reference',jobs=jobs,audits=audits),indent=2))
    print(json.dumps(audits,indent=2))
if __name__=='__main__':main()

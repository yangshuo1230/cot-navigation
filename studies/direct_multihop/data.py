"""Original N16/E24 graph pools, now supervised at hops 1, 2 and 3.

Every label is rechecked by both explicit BFS and topological bitset closure.
No trajectory or extra tokens are given to the model. Source val/test remain
held out; query-counterfactual pairs are derived from test graphs only.
"""
from pathlib import Path
import hashlib,json
import numpy as np

PAD,EOS,EDGE,QUERY,BAR,SEMI,YES,NO=range(8)
NODE_LO=14
SOURCE='data/explore_v1/context_standard_n16_m24_train50000_val10000_test10000_s0'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def key(kind,vocab,edges):
    # Compatibility with the frozen one-hop helper; its old Pools is not used.
    return f'{kind}_v{vocab}_m{edges}'


def reachability(adj,source):
    reached={source};stack=[source]
    while stack:
        u=stack.pop()
        for v in adj[u]:
            if v not in reached:reached.add(v);stack.append(v)
    return reached


def prepare(folder):
    from cot_navigation.data.dataset import ContextStore
    from cot_navigation.protocols.render import context_prompt
    from cot_navigation.protocols.tokenizer import Tokenizer
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    path=folder/'manifest.json'
    if path.exists():
        m=json.loads(path.read_text())
        assert hashlib.sha256((folder/'arrays.npz').read_bytes()).hexdigest()==m['arrays_sha256']
        return m
    store=ContextStore(SOURCE);a=store.arrays;tok=Tokenizer();arrays={};summary={};graph_sets={}
    for split in ('train','val','test'):
        ids=store.indices(split);xs=[];ys=[];hops=[];identities=set();counter={h:[] for h in (1,2,3)}
        for index in ids:
            n=16;adj=store.adj(int(index));nodes=store.node_vocab_ids(int(index),tok)
            s,t=int(a['s_topo'][index]),int(a['t_topo'][index]);y=int(a['label'][index]);hop=int(a['distance'][index])
            assert hop in (1,2,3) and t//4-s//4==hop and s!=t
            edges=list(zip(a['edge_src'][index].tolist(),a['edge_dst'][index].tolist()))
            assert len(set(edges))==24 and all(v//4==u//4+1 for u,v in edges)
            bits=[0]*n
            for u in range(n-1,-1,-1):
                for v in adj[u]:bits[u]|=(1<<v)|bits[v]
            assert int(t in reachability(adj,s))==int((bits[s]>>t)&1)==y
            x=context_prompt(tok,nodes,edges,s,t)
            assert len(x)==76 and x[-4]==QUERY and x[-1]==BAR
            # Recheck from the ACTUAL serialized input, independently of token IDs.
            inv={v:i for i,v in enumerate(nodes)};shown=[[] for _ in range(n)]
            for _,u,v in np.asarray(x[:-4]).reshape(-1,3):shown[inv[int(u)]].append(inv[int(v)])
            assert int(inv[x[-2]] in reachability(shown,inv[x[-3]]))==y
            identities.add(tuple(sorted((nodes[u],nodes[v]) for u,v in edges)))
            xs.append(x);ys.append(y);hops.append(hop)
            if split=='test' and len(counter[hop])<256:
                # Same graph and same layer gap, change one endpoint to flip label.
                target_options=[v for v in range((t//4)*4,(t//4+1)*4) if int((bits[s]>>v)&1)!=y]
                source_options=[u for u in range((s//4)*4,(s//4+1)*4) if int((bits[u]>>t)&1)!=y]
                new=x.copy();kind=None
                if target_options:new[-2]=nodes[target_options[0]];kind='target'
                elif source_options:new[-3]=nodes[source_options[0]];kind='source'
                if kind:
                    ns,nt=inv[new[-3]],inv[new[-2]]
                    assert int(nt in reachability(adj,ns))==1-y
                    counter[hop].append((x,new,y,kind,int(index)))
        graph_sets[split]=identities
        arrays[split+'_x']=np.asarray(xs,np.int64);arrays[split+'_y']=np.asarray(ys,np.int64)
        arrays[split+'_hop']=np.asarray(hops,np.int64);arrays[split+'_index']=ids
        summary[split]={str(h):{'n':int(np.sum(arrays[split+'_hop']==h)),
                              'yes':int(np.sum(arrays[split+'_y'][arrays[split+'_hop']==h]))} for h in (1,2,3)}
        if split=='test':
            for h,items in counter.items():
                arrays[f'counter{h}_x']=np.asarray([x for p in items for x in p[:2]],np.int64)
                arrays[f'counter{h}_y']=np.asarray([y for p in items for y in (p[2],1-p[2])],np.int64)
                arrays[f'counter{h}_index']=np.asarray([p[4] for p in items],np.int64)
    for i,x in enumerate(['train','val','test']):
        for y in ['train','val','test'][i+1:]:assert not graph_sets[x]&graph_sets[y],(x,y,'graph leakage')
    np.savez_compressed(folder/'arrays.npz',**arrays)
    m={'version':1,'source':SOURCE,'source_fingerprint':store.fingerprint,'counts':summary,
       'independent_labels_checked':sum(len(store.indices(s)) for s in ('train','val','test')),
       'graph_split_overlap':0,'counterfactual_pairs':{str(h):len(arrays[f'counter{h}_y'])//2 for h in (1,2,3)},
       'prompt_length':76,'node_vocab':256,'vocab_size':270,'n_nodes':16,'n_edges':24,
       'arrays_sha256':hashlib.sha256((folder/'arrays.npz').read_bytes()).hexdigest()}
    atomic_json(path,m);return m


class Pools:
    def __init__(self,folder):
        import torch
        self.parts={};self.counter={}
        with np.load(Path(folder)/'arrays.npz') as f:
            for split in ('train','val','test'):
                for hop in (1,2,3):
                    chosen=np.flatnonzero(f[split+'_hop']==hop)
                    for view in ('full','probe'):
                        ids=chosen
                        if view=='probe':
                            limit=2048 if split=='train' else 512
                            ids=np.concatenate([chosen[f[split+'_y'][chosen]==y][:limit//2] for y in (0,1)])
                        p={'x':torch.as_tensor(f[split+'_x'][ids],device='cuda'),
                           'y':torch.as_tensor(f[split+'_y'][ids],device='cuda')}
                        p['target']=torch.where(p['y'].bool(),YES,NO)
                        p['yes']=torch.where(p['y']==1)[0];p['no']=torch.where(p['y']==0)[0]
                        self.parts[(split,hop,view)]=p
            for hop in (1,2,3):
                y=torch.as_tensor(f[f'counter{hop}_y'],device='cuda')
                self.counter[hop]={'x':torch.as_tensor(f[f'counter{hop}_x'],device='cuda'),
                                   'y':y,'target':torch.where(y.bool(),YES,NO)}

    def get(self,split,hop,full=False):return self.parts[(split,hop,'full' if full else 'probe')]

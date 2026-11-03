"""Identity-capacity sanity checks, with the exact same native decision rows."""
from analyze import *

def main(setting):
    p=HERE/f'{setting}.json';out=json.loads(p.read_text())
    cfg=load_config([Path(out['manifest']['run'])/'resolved_config.yaml'])
    tok=Tokenizer.from_config(cfg);world=ParametricWorld(Path(cfg.data.root)/'main-seed0')
    adj=world.adj();nodes=world.node_vocab_ids(tok);inv={int(v):i for i,v in enumerate(nodes)}
    reach=np.eye(100,dtype=bool);a=np.zeros((100,100),np.float32)
    for u in reversed(range(100)):
        a[u,adj[u]]=1
        for v in adj[u]:reach[u]|=reach[v]
    meta=[]
    for line in gzip.open(OLD/f'{setting}_path_queries.jsonl.gz','rt'):
        r=json.loads(line)
        for j,token in enumerate(r['trace'][:-1]):
            if token not in inv:break
            u=inv[token];v=inv.get(r['trace'][j+1],-1)
            if j==0 and u!=r['source']:break
            if u==r['target'] or v<0:break
            meta.append(dict(current=u,target=r['target'],split=r['split'],next=v))
            if v not in adj[u]:break
    m={k:np.array([r[k] for r in meta]) for k in meta[0]}
    assert len(meta)==out['n_decisions']
    ys=a[m['current']];yu=ys*reach[:,m['target']].T
    yn=np.eye(100,dtype=np.float32)[m['next']]
    split=[np.flatnonzero(m['split']==i) for i in range(3)];te=split[2]
    mixed=(ys.sum(1)>yu.sum(1))&(yu.sum(1)>0);mte=te[mixed[te]]
    emb=np.random.default_rng(2026).normal(0,.02,(100,128)).astype(np.float32)
    for name,xx in [('random_current_embedding',emb[m['current']]),('random_target_embedding',emb[m['target']])]:
        entry={}
        for task,y in [('successor',ys),('useful',yu),('native_next',yn)]:
            pred,ridge=fit(xx,y,split);s=pred[:,0];metric=ranking(y[te],s[te])
            if task=='useful':metric['within_successors']=ranking(y[mte],s[mte],ys[mte].astype(bool))
            if task=='native_next':metric['mixed_top1_exact']=float((s[mte].argmax(1)==m['next'][mte]).mean())
            entry[task]=dict(ridge=ridge,layers=[metric])
        out['probes']['query_split']['features'][name]=entry
    out['random_control_note']='Untrained Gaussian identity vectors, 128 dimensions for 100 nodes. High query-split probe accuracy can arise from probe memorization.'
    write(p,out);print(setting,'random controls COMPLETE',flush=True)

if __name__=='__main__':
    for setting in ('simple','hard'):main(setting)

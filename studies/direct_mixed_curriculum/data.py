"""Paired mixed-hop curriculum data; no model-dependent labels."""
from pathlib import Path
import importlib.util
import json
import numpy as np

OLD = Path(str(next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'pyproject.toml').is_file() and (_p / 'studies').is_dir())) + '/studies/direct_learning/data.py')
spec = importlib.util.spec_from_file_location('old_data', OLD)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
BAR, EDGE, EOS, NODE_LO, NO, QUERY, YES = old.BAR, old.EDGE, old.EOS, old.NODE_LO, old.NO, old.QUERY, old.YES
atomic_json, digest, key = old.atomic_json, old.digest, old.key

def reachable(edges, s, t):
    seen = {s}; stack = [s]
    while stack:
        u = stack.pop()
        for a, b in edges:
            if a == u and b not in seen:
                seen.add(b); stack.append(b)
    return t in seen

def group(rng, vocab, kind):
    nodes = rng.choice(vocab, 16, replace=False) + NODE_LO
    if kind == 'fixed_source':
        pos, token = 0, NODE_LO
        idx = np.flatnonzero(nodes == token)
        if len(idx): nodes[pos], nodes[idx[0]] = nodes[idx[0]], nodes[pos]
        else: nodes[pos] = token
    elif kind == 'fixed_target':
        # Common target token at layer 3; all hops end here in this variant.
        pos, token = 12, NODE_LO + 1
        idx = np.flatnonzero(nodes == token)
        if len(idx): nodes[pos], nodes[idx[0]] = nodes[idx[0]], nodes[pos]
        else: nodes[pos] = token
    # Source-anchored and full use layer0 -> layer h. Target-anchored
    # uses layer(3-h) -> layer3, with the graph flip at the final boundary.
    flip = 2 if kind == 'fixed_target' else 0
    cores = []
    for variant in (0, 1):
        cores.append([(4*l+i, 4*(l+1)+(1-i if variant and l == flip else i))
                      for l in range(3) for i in range(2)])
    possible = [(a,b) for a in range(16) for b in range(16) if b//4 == a//4+1]
    # Insert a shared distractor only if all intended paired labels remain true.
    queries = {}
    for h in (1,2,3):
        start = 3-h if kind == 'fixed_target' else 0
        queries[h] = [(4*start+i,4*(start+h)+j) for i,j in ((0,0),(0,1),(1,0),(1,1))]
    graphs = [list(c) for c in cores]
    for idx in rng.permutation(len(possible)):
        edge = possible[idx]
        if any(edge in g for g in graphs): continue
        candidates = [g+[edge] for g in graphs]
        if all(reachable(candidates[v],s,t) == bool(old.LABELS[v*4+q])
               for v in (0,1) for h in (1,2,3) for q,(s,t) in enumerate(queries[h])):
            graphs = candidates
            if len(graphs[0]) == 24: break
    if len(graphs[0]) != 24: return None
    return [[(int(nodes[a]),int(nodes[b])) for a,b in g] for g in graphs], {
        h:[(int(nodes[s]),int(nodes[t])) for s,t in qs] for h,qs in queries.items()}

def prepare(folder):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    manifest = folder/'manifest.json'
    if manifest.exists(): return json.loads(manifest.read_text())
    ownership = {}; summaries = {}; checked = 0
    for ki,(kind,vocab) in enumerate([(k,v) for v in (16,64,256) for k in ('full','fixed_source','fixed_target')]):
        rng=np.random.default_rng(2026102100+ki); groups={}
        for split,count in [('train',512),('val',128),('test',128)]:
            accepted=[]; local=set(); attempts=0
            while len(accepted)<count:
                attempts+=1
                if attempts>100000: raise RuntimeError('Cannot construct disjoint pools')
                g=group(rng,vocab,kind)
                if g is None: continue
                identities=[tuple(sorted(edges[:m])) for edges in g[0] for m in (6,8,16,24)]
                if any(k in local or (k in ownership and ownership[k]!=split) for k in identities): continue
                for k in identities: local.add(k); ownership[k]=split
                accepted.append(g)
            groups[split]=accepted
        for m in (6,8,16,24):
            arrays={}; info={}
            for split,gs in groups.items():
                rows=[]; labels=[]; gids=[]; variants=[]; hops=[]
                for gi,(graphs,queries) in enumerate(gs):
                    for h in (1,2,3):
                        selected=old.VARIANTS.get(kind,old.VARIANTS['full'])
                        for vi in selected:
                            gv,q=divmod(vi,4); edges=graphs[gv][:m]; s,t=queries[h][q]
                            y=int(old.LABELS[vi]); assert int(reachable(edges,s,t))==y
                            # Independent boolean closure audit.
                            adj=np.zeros((270,270),dtype=bool)
                            for a,b in edges: adj[a,b]=True
                            seen={s};front={s}
                            for _ in range(16):
                                nxt={int(v) for u in front for v in np.flatnonzero(adj[u])}-seen
                                if not nxt: break
                                seen |= nxt; front=nxt
                            assert int(t in seen)==y
                            rows.append([z for a,b in edges for z in (EDGE,a,b)]+[QUERY,s,t,BAR])
                            labels.append(y); gids.append(gi*3+h-1); variants.append(vi); hops.append(h); checked+=1
                part={'x':np.asarray(rows,np.int64),'y':np.asarray(labels,np.int64),
                      'group':np.asarray(gids,np.int64),'variant':np.asarray(variants,np.int64),
                      'hop':np.asarray(hops,np.int64)}
                for h in (1,2,3):
                    y=part['y'][part['hop']==h]; assert len(y) and y.sum()*2==len(y)
                for k,v in part.items(): arrays[split+'_'+k]=v
                info[split]={'n':len(labels),'per_hop':{str(h):hops.count(h) for h in (1,2,3)}}
            name=key(kind,vocab,m); np.savez_compressed(folder/(name+'.npz'),**arrays);summaries[name]=info
        print('PREPARED',kind,vocab,flush=True)
    result={'version':1,'checked_rows':checked,'pools':summaries,
            'split_contract':'Global graph identity ownership across pools and edge scales; paired blocks stay together.',
            'construction':'Two length-3 chains with paired graph flip; distractors preserve every hop label.'}
    atomic_json(manifest,result);return result

if __name__=='__main__':
    import sys
    prepare(sys.argv[1])

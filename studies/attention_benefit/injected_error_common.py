"""Shared, query-disjoint artificial DFS errors for paired behavior/probes.

All injected nodes belong to the input graph. Candidate replacements exclude
visited nodes unless the named error explicitly tests visited membership.
Visited corruptions preserve source/current nodes, testing history rather
than deleting the obvious current-node cue. No correct repair is supplied.
"""
import json,hashlib,runpy
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import torch
P=Path(__file__).resolve().parent
H=runpy.run_path(str(P/'harder.py'));M=H['M'];D=M['D']
OUT=P/'injected_errors'
SEED=20261024
KINDS=['candidate_nonedge','candidate_visited','candidate_duplicate','candidate_omission',
       'visited_unvisited','visited_duplicate']
HEADS=[a+'_'+h for a in ('at_error','before_action') for h in ('L4','L5','L6','token_position','lm_surprise')]


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);H['write'](path,data)


def load_model(arm,device):
    folder=D['FOLDERS']['dfs_'+arm];cfg=D['load_config']([folder/'resolved_config.yaml'])
    tok=D['Tokenizer'].from_config(cfg);ckpt=folder/'checkpoints/latest.pt'
    model=D['build_model'](cfg.model,len(tok),tok.pad).to(device).eval().requires_grad_(False)
    blob=torch.load(ckpt,map_location='cpu',weights_only=False);assert blob['step']==200000
    model.load_state_dict(blob['model'],strict=True);del blob
    return cfg,tok,model,dict(checkpoint=str(ckpt),step=200000,
        sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),max_seq_len=model.max_seq_len,
        trace_lambda=cfg.train.trace_loss_lambda,training_graph_family=cfg.data.graph_family,
        training_data_root=cfg.data.root)


class Data:
    def __init__(self,domain):
        self.domain=domain
        cfg=D['load_config']([D['FOLDERS']['dfs_clean']/'resolved_config.yaml'])
        self.tok=D['Tokenizer'].from_config(cfg)
        if domain=='hard':cfg=D['load_config']([H['HARD']/'resolved_config.yaml']);root=cfg.data.root
        elif domain=='main':root=cfg.data.root
        elif domain=='deep':root=dict(s.split('=',1) for s in cfg.evaluation.extra_sets)['deep']
        else:raise ValueError(domain)
        self.source=D['ContextStore'](root)
        self.cache=D['ScalePreparedRows'](self.source,cfg,self.tok)
        self.source_name=str(self.source.root)

    def ids(self,split,n):
        a=self.source.arrays
        return list(map(int,M['stratified_ids'](self.source.indices(split),a['shape_id'],a['distance'],a['label'],n,
            SEED+{'train':0,'val':1,'test':2}[split]*7919)))

    def row(self,e):return self.cache.row(e['query'],e['variant'])

    def pair(self,e):
        row=self.row(e);anchor=e['anchor'];clean=list(row.tokens[:anchor+1])
        wrong=clean.copy()
        if e['kind']=='candidate_omission':del wrong[e['at']]
        else:wrong[e['at']]=e['bad']
        assert len(clean)<=4080 and len(wrong)<=4080
        return clean,wrong

    def action_pair(self,e):
        row=self.row(e);clean=list(row.tokens[:e['action_start']]);wrong=clean.copy()
        if e['kind']=='candidate_omission':del wrong[e['at']]
        else:wrong[e['at']]=e['bad']
        return clean,wrong

    def example(self,e):
        from cot_navigation.analysis.common import Example
        q=e['query'];a=self.source.arrays;nodes=self.source.node_vocab_ids(q,self.tok)
        return Example(q,'test',int(a['label'][q]),self.source.adj(q),nodes,
                       int(a['s_topo'][q]),int(a['t_topo'][q]),self.row(e))

    def verify(self,events,per_kind=5):
        """Independent previously-tested prefix oracle, not the sampler sets."""
        from cot_navigation.analysis.error_detection import semantic_slots
        checked=Counter()
        for e in events:
            if checked[e['kind']]>=per_kind:continue
            ex=self.example(e)
            for label,seq in enumerate(self.pair(e)):
                body=seq[ex.row.prompt_len:];slots=list(semantic_slots(ex,'dfs_state',self.tok,body))
                bad=[i for i,allowed,field in slots if body[i] not in allowed]
                assert len(slots)==len(body),(e,'oracle stopped before anchor')
                assert bad==([len(body)-1] if label else []),(e,bad,label)
            checked[e['kind']]+=1
        return dict(checked)

    def events(self,split,n_queries,n_pairs):
        from cot_navigation.data.mistakes import records,audit_reference
        pools=defaultdict(list);ids=self.ids(split,n_queries)
        tok=self.tok;bar=tok.id('|');counter=Counter()
        for q in ids:
            variant=q%4;row=self.cache.row(q,variant);recs=records(row,tok)
            audit_reference(row,recs,tok,fifo=False)
            nodes=self.source.node_vocab_ids(q,tok);raw=self.source.adj(q)
            adj={nodes[u]:set(nodes[v] for v in vs) for u,vs in enumerate(raw)}
            target=nodes[int(self.source.arrays['t_topo'][q])];source=nodes[int(self.source.arrays['s_topo'][q])]
            rng=np.random.default_rng(SEED+q*104729+{'train':0,'val':1,'test':2}[split])
            order=list(rng.permutation(len(recs)));local=defaultdict(list);seen=set()
            for ri in order:
                r=recs[int(ri)]
                if r['action'][0] not in (tok.id('<PUSH>'),tok.id('<POP>')) or r['action_start']>=4080:continue
                cur=r['stack'][-1];visited=set(r['visited']);outgoing=adj.get(cur,set())
                cands=r['cands'];clo,chi=r['spans']['cands'];vlo,vhi=r['spans']['visited']
                def add(kind,field,at,bad,anchor=None):
                    if len(local[kind])>=2:return
                    sig=(kind,int(ri),int(at),int(bad))
                    if sig in seen:return
                    seen.add(sig)
                    e=dict(query=q,variant=variant,record=int(ri),kind=kind,field=field,
                        at=int(at),bad=int(bad),anchor=int(at if anchor is None else anchor),
                        action_start=r['action_start'],record_start=r['start'],label=int(row.label),
                        prompt_len=row.prompt_len,target=int(target),current=int(cur))
                    if e['anchor']<4080:local[kind].append(e)
                if cands:
                    positions=list(rng.permutation(len(cands)))
                    for j in positions[:3]:
                        if cands[j]==target:continue
                        pool=sorted(set(nodes)-outgoing-visited-{target})
                        if pool:add('candidate_nonedge','candidates',clo+int(j),rng.choice(pool))
                        pool=sorted((outgoing&visited)-{target})
                        if pool:add('candidate_visited','candidates',clo+int(j),rng.choice(pool))
                        pool=[v for v in cands[:j] if v!=target]
                        if pool:add('candidate_duplicate','candidates',clo+int(j),rng.choice(pool))
                    j=int(rng.integers(len(cands)))
                    add('candidate_omission','candidates',clo+j,bar,anchor=chi)
                protected={source,r['stack'][0],cur,target}
                removable=[j for j,v in enumerate(r['visited']) if v not in protected]
                rng.shuffle(removable)
                for j in removable[:3]:
                    pool=sorted(set(nodes)-visited-{target})
                    if pool:add('visited_unvisited','visited',vlo+j,rng.choice(pool))
                    pool=[v for v in r['visited'][:j] if v not in protected]
                    if pool:add('visited_duplicate','visited',vlo+j,rng.choice(pool))
                if all(len(local[k])>=2 for k in KINDS):break
            for k,items in local.items():pools[k].extend(items)
        rng=np.random.default_rng(SEED+{'train':11,'val':12,'test':13}[split])
        for k in KINDS:rng.shuffle(pools[k])
        # Round-robin types; rare kinds never get fabricated to fill a quota.
        selected=[]
        while len(selected)<n_pairs:
            added=False
            for k in KINDS:
                if pools[k] and len(selected)<n_pairs:selected.append(pools[k].pop());added=True
            if not added:break
        rng.shuffle(selected)
        assert len({(e['query'],e['variant'],e['kind'],e['at'],e['bad']) for e in selected})==len(selected)
        digest=hashlib.sha256(json.dumps(selected,sort_keys=True).encode()).hexdigest()
        return selected,dict(source=self.source_name,split=split,pairs=len(selected),requested_pairs=n_pairs,
            selected_queries=len(ids),queries=len({e['query'] for e in selected}),
            kinds=dict(Counter(e['kind'] for e in selected)),event_sha256=digest,
            oracle_checks=self.verify(selected))


def repair_events(data,n_per_kind=16):
    pool,meta=data.events('test',512,6144)
    rng=np.random.default_rng(SEED+99);selected=[];used=set()
    for kind in KINDS:
        candidates=[e for e in pool if e['kind']==kind and e['action_start']<=2048]
        rng.shuffle(candidates)
        for label in (0,1):
            got=0
            for e in candidates:
                if e['label']!=label or e['query'] in used:continue
                selected.append(e);used.add(e['query']);got+=1
                if got>=n_per_kind//2:break
    meta.update(pairs=len(selected),queries=len(used),kinds=dict(Counter(e['kind'] for e in selected)),
        labels=dict(Counter(e['label'] for e in selected)),selection='one event per test query; up to 8 per label per kind; action prefix <=2048',
        event_sha256=hashlib.sha256(json.dumps(selected,sort_keys=True).encode()).hexdigest())
    assert len(used)==len(selected)
    return selected,meta

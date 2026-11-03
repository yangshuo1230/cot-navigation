"""Frozen N100 trace reuse audit against the complete eligible training bank.

No optimizer, checkpoint writes or activation dumps. Training-bank matches are
upper bounds on literal observed-example reuse: variant exposure is not logged.
"""
from pathlib import Path
import os
import sys
import json
import hashlib
from collections import Counter
import time

os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
ROOT=Path(str(next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'pyproject.toml').is_file() and (_p / 'studies').is_dir())))
OUT=ROOT/'studies/fixed_trace_reuse'
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'studies/fixed100_legality'),str(ROOT/'studies/embedding_subspace_meaning')]
import numpy as np
import torch
from cot_navigation.config import load_config
from cot_navigation.data.dataset import ParametricWorld
from cot_navigation.data.scale_prepared import ScalePreparedRows
from cot_navigation.models.transformer import build_model
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.protocols.render import parametric_prompt,render_task
from cot_navigation.protocols.dfs import dfs_search
from cot_navigation.protocols.bfs import bfs_search,first_expansion_order
from exposure import counts
from stream import generate_checked
from cache_prefix import enable
from replay import StateReplay,SequenceReplay
import audit as legality

RUNS={'simple':ROOT/'outputs/fixed_short3_v1/launches/short3_n100_s0/all_hops',
      'hard':ROOT/'outputs/hard_dag_v1/launches/hard_dag_v1_s0/fixed/n100/stage_b'}

def digest(values):
    return hashlib.blake2b(np.asarray(values,dtype='<i2').tobytes(),digest_size=16).digest()

def exposed_queries(world,cfg,blob):
    q=world.queries;ids=np.r_[world.indices('train','multi_hop'),world.indices('train','one_hop')]
    state=blob['query_sampler'];result=np.zeros(len(q['label']),np.int64)
    if state.get('kind')=='short_hop_quotas_v1':
        gap=q['signed_gap'][ids];ids=ids[((gap>0)&(gap<=9))|((gap<0)&(-gap<=3))]
        assert len(state['groups'])==len(state['delegate']['samplers'])
        for group,sub in zip(state['groups'],state['delegate']['samplers']):
            np.add.at(result,ids[np.asarray(group)],counts(sub))
    else:
        np.add.at(result,ids,counts(state))
    assert result.sum()==blob['queries_seen']
    assert np.all(q['split'][result>0]==0)
    return np.flatnonzero(result>0),int(result.sum())

def unpack(body,protocol,tok):
    """Record IDs retain complete token order; expansion order ignores repeats."""
    if protocol.endswith('_nodes'):
        return list(body),[]
    records=[];currents=[];start=0
    for end in np.flatnonzero(np.asarray(body)==tok.id(';')):
        raw=list(body[start:int(end)+1]);start=int(end)+1
        try:
            first=raw.index(tok.id('|'))
            frontier=raw[:first]
            current=frontier[0 if protocol.startswith('bfs') else -1]
            if current not in currents:currents.append(current)
        except (ValueError,IndexError):
            pass
        records.append(raw)
    return currents,records

class Bank:
    def __init__(self,protocol,tok):
        self.protocol=protocol;self.tok=tok;self.full={};self.node_full={}
        self.node_trie={};self.record_trie={};self.record_set=set();self.node_ngrams=set();self.semantic_states=set()
        self.variants=0;self.qids=set();self.source_counts=Counter()

    @staticmethod
    def insert(root,seq):
        for value in seq:root=root.setdefault(value,{})

    @staticmethod
    def prefix(root,seq):
        n=0
        for value in seq:
            if value not in root:break
            root=root[value];n+=1
        return n

    def state_key(self,record):
        """Exact frontier order plus visited SET: independent of print ordering."""
        separators=[i for i,v in enumerate(record) if v==self.tok.id('|')]
        if len(separators)!=3:return None
        a,b,_=separators
        frontier=record[:a];visited=sorted(set(record[a+1:b]))
        return digest([len(frontier),*frontier,*visited])

    def add(self,body,qid,variant):
        nodes,records=unpack(body,self.protocol,self.tok)
        witness=[int(qid),int(variant)]
        self.full.setdefault(digest(body),witness);self.node_full.setdefault(digest(nodes),witness)
        self.insert(self.node_trie,nodes)
        rh=[digest(r) for r in records]
        self.insert(self.record_trie,rh);self.record_set.update(rh)
        self.semantic_states.update(self.state_key(r) for r in records)
        self.node_ngrams.update(digest(nodes[i:i+8]) for i in range(max(0,len(nodes)-7)))
        self.variants+=1;self.qids.add(int(qid))
        if nodes:self.source_counts[nodes[0]]+=1

    def measure(self,body,valid_records=None):
        nodes,records=unpack(body,self.protocol,self.tok)
        rh=[digest(r) for r in records]
        # Terminal FOUND/EXHAUST records depend on the queried target. Exclude
        # them only in the prefix diagnostic, never in whole-trace equality.
        core=rh[:-1] if records and any(t in records[-1] for t in [self.tok.id('<FOUND>'),self.tok.id('<EXHAUST>')]) else rh
        node_prefix=self.prefix(self.node_trie,nodes)
        record_prefix=self.prefix(self.record_trie,core)
        valid=records[:valid_records] if valid_records is not None else records
        keys=[self.state_key(r) for r in valid[2:]]
        return dict(tokens=len(body),expansions=len(nodes),records=len(records),
            valid_states_after_first2=len(keys),novel_valid_states_after_first2=sum(k not in self.semantic_states for k in keys),
            exact_tokens=digest(body) in self.full,exact_expansions=digest(nodes) in self.node_full,
            node_prefix_length=node_prefix,node_prefix_fraction=node_prefix/len(nodes) if nodes else None,
            whole_node_prefix=bool(nodes) and node_prefix==len(nodes),
            record_prefix_length=record_prefix,record_prefix_fraction=record_prefix/len(core) if core else None,
            records_seen_excluding_first2=sum(v in self.record_set for v in rh[2:]),records_after_first2=len(rh[2:]),
            ngrams8_seen=sum(digest(nodes[i:i+8]) in self.node_ngrams for i in range(max(0,len(nodes)-7))),
            ngrams8_total=max(0,len(nodes)-7),
            token_match=self.full.get(digest(body)),node_match=self.node_full.get(digest(nodes)))

def aggregate(rows):
    if not rows:return dict(n=0)
    result=dict(n=len(rows))
    for field in ['exact_tokens','exact_expansions','whole_node_prefix']:
        result[field]=sum(r[field] for r in rows)/len(rows)
    for field in ['tokens','expansions','records','node_prefix_fraction','record_prefix_fraction']:
        values=[r[field] for r in rows if r[field] is not None]
        result[field]=float(np.mean(values)) if values else None
    for name,num,den in [('novel_valid_state_fraction','novel_valid_states_after_first2','valid_states_after_first2'),('record_seen_fraction','records_seen_excluding_first2','records_after_first2'),('ngram8_seen_fraction','ngrams8_seen','ngrams8_total')]:
        total=sum(r[den] for r in rows)
        result[name]=sum(r[num] for r in rows)/total if total else None
        result[name+'_denominator']=total
    return result

def run(setting,protocol):
    dest=OUT/f'{setting}_{protocol}.json'
    if dest.exists() and json.loads(dest.read_text()).get('version')==2:return
    start=time.time();folder=RUNS[setting]/protocol
    cfg=load_config([folder/'resolved_config.yaml']);tok=Tokenizer.from_config(cfg)
    world=ParametricWorld(Path(cfg.data.root)/f'{cfg.data.graph_shape}-seed{cfg.data.graph_seed}')
    q=world.queries;nodes=world.node_vocab_ids(tok);adj=world.adj()
    checkpoint=folder/'checkpoints/best.pt';checksum=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    blob=torch.load(checkpoint,map_location='cpu',weights_only=False)
    trainids,exposures=exposed_queries(world,cfg,blob)
    ids=legality.panel_ids([int(i) for i in world.indices('test') if int(q['t_topo'][i])//10>int(q['s_topo'][i])//10],q)
    assert not set(ids)&set(trainids)
    train_pairs={(int(q['s_topo'][i]),int(q['t_topo'][i])) for i in trainids}
    assert all((int(q['s_topo'][i]),int(q['t_topo'][i])) not in train_pairs for i in ids)
    cache=ScalePreparedRows(world,cfg,tok)
    bank=Bank(protocol,tok)
    for i in trainids:
        for variant in range(cache.variants(i)):
            row=cache.row(i,variant)
            bank.add(row.tokens[row.prompt_len:row.answer_index],i,variant)
    print(setting,protocol,'bank',len(trainids),bank.variants,'test',len(ids),flush=True)
    reference=[]
    for i in ids:
        s,t=int(q['s_topo'][i]),int(q['t_topo'][i])
        # A held-out cached variant is not a training example. It provides a
        # correct algorithm-generated baseline under the same synthesis rules.
        for variant in range(cache.variants(i)):
            row=cache.row(i,variant)
            reference.append(dict(query_id=i,kind='heldout_random_reference',label=int(q['label'][i]),**bank.measure(row.tokens[row.prompt_len:row.answer_index])))
        for order_seed in (1001,1002,1003):
            priorities=np.random.default_rng(order_seed).permutation(world.n_nodes)
            ordered=[sorted(vs,key=lambda v:int(priorities[v])) for vs in adj]
            row=render_task(tok,protocol,int(q['label'][i]),parametric_prompt(tok,nodes,s,t),ordered,s,t,nodes)
            reference.append(dict(query_id=i,kind='global_priority_reference',label=int(q['label'][i]),**bank.measure(row.tokens[row.prompt_len:row.answer_index])))
    header=[tok.id('<WORLD_0>')] if cfg.train.world_roots else []
    model=build_model(cfg.model,len(tok),tok.pad).cuda().eval().requires_grad_(False)
    model.load_state_dict(blob['model'],strict=True);step=int(blob['step']);del blob
    enable(model)
    tokenadj={nodes[u]:set(nodes[v] for v in vs) for u,vs in enumerate(adj)}
    generated=[];examples=[]
    for offset in range(0,len(ids),64):
        selected=ids[offset:offset+64];prompts=[];replays=[]
        for i in selected:
            s,t=int(q['s_topo'][i]),int(q['t_topo'][i])
            prompts.append(header+parametric_prompt(tok,nodes,s,t))
            if protocol.endswith('_state'):replays.append(StateReplay(tokenadj,nodes[s],nodes[t],tok,protocol.startswith('bfs')))
            else:replays.append(SequenceReplay(tokenadj,nodes[s],nodes[t],int(q['label'][i]),protocol,tok,legality.previous.sequence_audit))
        outs=generate_checked(model,prompts,tok,cfg.train.precision,replays,cfg.protocol.cot_margin)
        for i,out in zip(selected,outs):
            body=out['trace'];m=bank.measure(body,out['audit'].get('prefix_valid_records',0))
            row=dict(query_id=i,source=int(q['s_topo'][i]),target=int(q['t_topo'][i]),label=int(q['label'][i]),
                     censored=out['early_censored'],interrupted=out['interrupted'],complete=out['pred'] is not None,
                     answer_correct=out['pred']==int(q['label'][i]),semantic_valid=out['audit']['semantic_valid'],
                     first_error=out['audit']['first_error'],**m)
            generated.append(row)
            # Save just a few short examples; no complete trace/activation dumps.
            if len(examples)<4 and (row['exact_tokens'] or (row['semantic_valid'] and not row['whole_node_prefix'])):
                match=m['token_match'] or m['node_match']
                examples.append(dict(query_id=i,source=tok.name(nodes[row['source']]),target=tok.name(nodes[row['target']]),
                    label=row['label'],exact_tokens=row['exact_tokens'],node_prefix_length=row['node_prefix_length'],
                    expansion_preview=[tok.name(v) for v in unpack(body,protocol,tok)[0][:24]],
                    matched_training_query=match,
                    matched_training_target=tok.name(nodes[int(q['t_topo'][match[0]])]) if match else None,
                    matched_training_label=int(q['label'][match[0]]) if match else None))
        print(setting,protocol,'generated',len(generated),'seconds',round(time.time()-start,1),flush=True)
    complete=[r for r in generated if r['complete'] and not r['interrupted']]
    result=dict(version=2,setting=setting,protocol=protocol,run=str(folder),checkpoint_step=step,checkpoint_sha256=checksum,
                training_queries=len(trainids),training_query_exposures=exposures,training_bank_variants=bank.variants,
                variants_per_query=cfg.protocol.trajectory_variants,
                disjoint_query_pairs=True,panel_ids=ids,
                generated_all=aggregate(generated),generated_complete=aggregate(complete),
                generated_valid=aggregate([r for r in generated if r['semantic_valid']]),
                complete_count=len(complete),semantic_valid_count=sum(r['semantic_valid'] for r in generated),
                censored_count=sum(r['censored'] for r in generated),
                by_label={str(label):aggregate([r for r in complete if r['label']==label]) for label in (0,1)},
                references={kind:aggregate([r for r in reference if r['kind']==kind]) for kind in ('heldout_random_reference','global_priority_reference')},
                matched_reference={kind:aggregate([r for r in reference if r['kind']==kind and r['query_id'] in {v['query_id'] for v in complete}]) for kind in ('heldout_random_reference','global_priority_reference')},
                reference_by_label={str(label):aggregate([r for r in reference if r['kind']=='heldout_random_reference' and r['label']==label and r['query_id'] in {v['query_id'] for v in complete}]) for label in (0,1)},
                generated_long8=aggregate([r for r in complete if r['expansions']>=8]),
                reference_long8=aggregate([r for r in reference if r['kind']=='heldout_random_reference' and r['expansions']>=8 and r['query_id'] in {v['query_id'] for v in complete}]),
                rows=generated,examples=examples,seconds=time.time()-start,
                caveats=['Bank contains all four available variants of actually sampled training queries; per-variant cumulative exposure was not logged.',
                         'At first semantic error generation is censored; whole-trace matching is reported separately for completed, uninterrupted outputs.',
                         'Same-source unreachable queries legitimately have identical target-independent traversals.',
                         'Prefix/fragment overlap alone does not establish copying or an algorithmic mechanism.'])
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    del model,cache,bank;torch.cuda.empty_cache()

if __name__=='__main__':
    torch.set_num_threads(2);OUT.mkdir(exist_ok=True)
    for setting in ('simple','hard'):
        for protocol in ('dfs_nodes','bfs_nodes','dfs_state','bfs_state'):run(setting,protocol)

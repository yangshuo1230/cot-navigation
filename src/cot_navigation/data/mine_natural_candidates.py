"""Mine frozen-model candidate completions; no synthetic token edits."""
from collections import Counter
import gzip
import json
import time
from pathlib import Path

import numpy as np
import torch

from cot_navigation.config import load_config
from cot_navigation.data.dataset import ContextStore
from cot_navigation.data.scale_prepared import ScalePreparedRows
from cot_navigation.data.mistakes import records,audit_reference
from cot_navigation.data.natural_mistakes import describe_error,build_episode,pool_path,identity,write_pool
from cot_navigation.data.shards import file_sha256,is_complete,load_json
from cot_navigation.evaluation.rollout import pad_rows,_gather_row,_decode_step,_autocast
from cot_navigation.evaluation.sampling import stratified_ids
from cot_navigation.models.transformer import build_model
from cot_navigation.models.compiled_inference import CompiledInference
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.training.checkpoint import load_checkpoint


@torch.no_grad()
def complete_fields(model,prompts,tok,cap=64):
    """Greedy completion to the actual emitted |; never force a delimiter."""
    if not prompts:return []
    device=next(model.parameters()).device
    tokens,lengths=pad_rows(prompts,tok.pad,device)
    cache=model.new_cache(len(prompts),device,torch.bfloat16)
    positions=torch.arange(tokens.shape[1],device=device).expand(len(prompts),-1)
    with _autocast(device,'bf16'):
        logits=model(tokens,positions=positions,cache=cache,cache_mode='prefill')
    cache.lengths.copy_(lengths)
    cache.valid_length=tokens.shape[1]
    next_logits=_gather_row(logits,lengths-1)
    generated=[[] for _ in prompts];reasons=[None]*len(prompts)
    live=list(range(len(prompts)))
    while live:
        picked=next_logits.argmax(-1).tolist()
        keep=[];feed=[]
        for local,(index,token) in enumerate(zip(live,picked)):
            if token==tok.id('|'):
                reasons[index]='closed';continue
            if not tok.node_lo<=token<tok.node_lo+tok.n_node_tokens:
                generated[index].append(token)  # preserve the actual malformed output
                reasons[index]='non_node';continue
            generated[index].append(token)
            if len(generated[index])>=cap or len(prompts[index])+len(generated[index])>=model.max_seq_len:
                reasons[index]='unclosed_at_cap';continue
            keep.append(local);feed.append(token)
        if not keep:break
        if len(keep)!=len(live):
            cache.select_batch(torch.tensor(keep,device=device,dtype=torch.long))
            live=[live[j] for j in keep]
        cache.valid_length=max(len(prompts[i])+len(generated[i])-1 for i in live)
        next_logits=_decode_step(model,cache,torch.tensor(feed,device=device).unsqueeze(1),'bf16')
    return list(zip(generated,reasons))


def mine_one(config,root,train_queries=8192,heldout_queries=512,states_per_query=0,batch_size=32):
    if states_per_query != 0:
        raise ValueError('Capped mining is retired; states_per_query=0 scans ALL reference records')
    started=time.perf_counter();torch.set_num_threads(2)
    cfg=load_config([config]);tok=Tokenizer.from_config(cfg)
    source=ContextStore(root);prepared=ScalePreparedRows(source,cfg,tok)
    path=pool_path(source,cfg,tok)
    settings=dict(train_queries=train_queries,heldout_queries=heldout_queries,
                  states_per_query=states_per_query,batch_size=batch_size,
                  completion='greedy_to_emitted_bar_or_native_context_limit',candidate_cap='remaining_context',
                  selection='all_records_no_position_cap',compiled_inference=True,seed=cfg.protocol.mistake_seed)
    if is_complete(path):
        header=load_json(path.with_suffix('.json'))
        if header['identity']!=identity(source,cfg,tok) or header['provenance']['settings']!=settings:
            raise ValueError('immutable natural corpus identity/settings differ')
        return dict(source=root,path=str(path),**header)
    old=Path('outputs/scale_v1/launches/scale_v1_s0/context/curriculum')/cfg.protocol.name
    donor_cfg=load_config([old/'resolved_config.yaml'])
    donor_tok=Tokenizer.from_config(donor_cfg)
    assert tok.itos[:len(donor_tok)]==donor_tok.itos
    model=build_model(donor_cfg.model,len(donor_tok),pad_id=donor_tok.pad).cuda().eval()
    queries={};contexts=[];stats=Counter();features=Counter();donors=[];raw_extra=[]
    for split_id,split in enumerate(('train','val','test')):
        ids=source.indices(split)
        limit=train_queries if split=='train' else heldout_queries
        selected=stratified_ids(ids,source.arrays['shape_id'],source.arrays['distance'],
                                source.arrays['label'],limit,cfg.protocol.mistake_seed+split_id*7919)
        for index in selected:
            index=int(index);rng=np.random.default_rng(cfg.protocol.mistake_seed+index*104729)
            variant=int(rng.integers(prepared.variants(index)))
            row=prepared.row(index,variant)
            assert int(row.meta['split'])==split_id
            recs=records(row,tok);audit_reference(row,recs,tok,fifo=cfg.protocol.name=='bfs_state')
            query_key=(split_id,index)
            queries[query_key]=dict(row=row,events=[],variant=variant)
            donor=int(rng.integers(2))
            selected_records=np.arange(len(recs))
            for rec_index in selected_records:
                rec=recs[int(rec_index)];lo,hi=rec['spans']['cands']
                contexts.append(dict(key=query_key,donor=donor,prefix=row.tokens[:lo],
                    event=dict(start=rec['start'],end=rec['end'],lo=lo,hi=hi,action=rec['action_start']),
                    correct=rec['cands'],visited=rec['visited']))
    stats['queries_sampled']=len(queries);stats['fields_requested']=len(contexts)
    raw_path=path.with_suffix('.raw.json.gz')
    def write_raw(complete=False):
        raw_path.parent.mkdir(parents=True,exist_ok=True)
        temp=raw_path.with_suffix('.tmp')
        data=dict(complete=complete,identity=identity(source,cfg,tok),settings=settings,donors=donors,
                  statistics=dict(stats),
                  query_errors=[dict(split=key[0],index=key[1],variant=q['variant'],events=q['events'])
                                for key,q in queries.items() if q['events']],
                  unclosed_or_malformed=raw_extra)
        with gzip.open(temp,'wt',compresslevel=3) as handle:json.dump(data,handle,separators=(',',':'))
        temp.replace(raw_path)
    inference=CompiledInference(model).eval()
    for donor,name in enumerate(('best.pt','latest.pt')):
        checkpoint=old/'checkpoints'/name
        blob=load_checkpoint(checkpoint,map_location='cpu')
        if blob['arch']['vocab_hash']!=donor_tok.hash():raise ValueError('donor vocabulary changed')
        model.load_state_dict(blob['model'])
        donors.append(dict(checkpoint=str(checkpoint),sha256=file_sha256(checkpoint),step=int(blob['step'])))
        del blob
        todo=sorted([c for c in contexts if c['donor']==donor],key=lambda c:len(c['prefix']))
        for start in range(0,len(todo),batch_size):
            chunk=todo[start:start+batch_size]
            predictions=complete_fields(inference,[c['prefix'] for c in chunk],donor_tok,cap=model.max_seq_len)
            for case,(predicted,reason) in zip(chunk,predictions):
                stats[reason]+=1
                if reason!='closed':
                    raw_extra.append(dict(split=case['key'][0],index=case['key'][1],
                        variant=queries[case['key']]['variant'],event=case['event'],predicted=predicted,reason=reason))
                    continue
                query=queries[case['key']];row=query['row']
                graph_nodes={t for t in row.tokens[:row.prompt_len]
                             if tok.node_lo<=t<tok.node_lo+tok.n_node_tokens}
                error=describe_error(case['correct'],predicted,case['visited'],graph_nodes)
                if error is None:
                    stats['correct_membership']+=1;continue
                stats['observed_wrong_fields']+=1
                pattern=','.join(f'{k}={error[k]}' for k in ('missing','extra','duplicates') if error[k])
                features[pattern]+=1
                for key,value in error.items():
                    if value:stats[f'observed_{key}']+=int(value)
                query['events'].append(dict(**case['event'],predicted=predicted,features=error))
            if start%(batch_size*128)==0:
                write_raw()
                print(f"MINE {cfg.protocol.name} {source.root.name} donor={donor} {start}/{len(todo)} fields; errors={stats['observed_wrong_fields']}",flush=True)
    assert sum(stats[reason] for reason in ['closed','non_node','unclosed_at_cap'])==stats['fields_requested']
    write_raw(complete=True)
    rows=[];splits=[];seen=set();episode_stats=Counter();previews=[]
    for (split_id,index),query in queries.items():
        row=query['row']
        if not query['events']:continue
        episode=build_episode(row,query['events'],tok,cfg)
        if episode is None:
            stats['no_complete_repair_fits']+=1;continue
        assert (split_id,index) not in seen;seen.add((split_id,index))
        episode.meta['natural_variant']=query['variant']
        rows.append(episode);splits.append(split_id)
        split=('train','val','test')[split_id]
        episode_stats[f'{split}_episodes']+=1
        episode_stats[f'{split}_label{row.label}']+=1
        episode_stats[f'{split}_complex']+=int(episode.meta['natural_complex'])
        episode_stats[f'{split}_multi_event']+=int(len(episode.meta['natural_events'])>1)
        episode_stats[f'{split}_events']+=len(episode.meta['natural_events'])
        if split_id==0 and len(previews)<4:
            event=episode.meta['natural_events'][0]
            previews.append(dict(query=index,variant=query['variant'],
                observed_candidates=tok.decode(event['predicted']),features=event['features'],
                wrong_then_repair=tok.decode(episode.tokens[event['bad_start']:event['repair_end']])))
    for split in ('train','val','test'):
        if episode_stats[f'{split}_episodes']==0:
            raise ValueError(f'no actual model errors in {split}: increase mining, never substitute synthetic errors')
    for label in (0,1):
        if episode_stats[f'train_label{label}']<16:
            raise ValueError('too few observed training errors for a label; increase --train-queries')
    statistics=dict(**dict(stats),episodes=dict(episode_stats),observed_patterns=dict(features),
                    raw_error_file=str(raw_path),raw_error_sha256=file_sha256(raw_path),
                    raw_closed_error_events=sum(len(q['events']) for q in queries.values()),
                    raw_nonclosed_events=len(raw_extra),
                    packed_events=sum(len(r.meta['natural_events']) for r in rows),
                    preview=previews,seconds=time.perf_counter()-started)
    path=write_pool(source,cfg,tok,rows,splits,dict(donors=donors,settings=settings),statistics)
    print(f'MINE READY {cfg.protocol.name} {source.root.name}: {dict(episode_stats)}',flush=True)
    header=load_json(path.with_suffix('.json'))
    return dict(source=root,path=str(path),**header)

"""Immutable repair episodes containing model-produced candidate fields.

No candidate token is synthesized or remapped. Each retained field was
completed by a frozen donor on this exact graph and canonical search prefix.
Episodes may contain several observed errors; every MISTAKE is followed by
the original correct record without advancing the search machine twice.
"""
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from cot_navigation.data.mistakes import records, audit_reference
from cot_navigation.data.prepared import PreparedRows
from cot_navigation.data.provenance import digest
from cot_navigation.data.scale_prepared import identity as clean_identity
from cot_navigation.data.shards import atomic_json,atomic_savez,file_sha256,is_complete,load_json,load_npz,mark_complete
from cot_navigation.protocols.render import truncate_example

VERSION = 2


def identity(source,cfg,tok):
    return dict(version=VERSION,source=source.fingerprint,
                canonical=clean_identity(source,cfg,tok),kind='observed_all_candidate_positions_v2')


def pool_path(source,cfg,tok):
    root=Path(cfg.protocol.mistake_cache_root)
    if (root/'DEPRECATED.json').exists():
        raise ValueError(f'Error pool retired: {root}; use data/natural_correction/all_positions_v2_20261102')
    return root/f'{cfg.protocol.name}_{digest(identity(source,cfg,tok))[:24]}.npz'


def describe_error(correct,predicted,visited,graph_nodes):
    """Measured features, never a recipe for generating a wrong field."""
    missing=set(correct)-set(predicted)
    extra=set(predicted)-set(correct)
    duplicates=len(predicted)-len(set(predicted))
    if not missing and not extra and not duplicates:
        return None  # a legal reordering is not an error
    return dict(missing=len(missing),extra=len(extra),duplicates=duplicates,
                extra_visited=len(extra&set(visited)),out_of_graph=len(extra-set(graph_nodes)),
                edits_lower_bound=max(len(missing),len(extra))+duplicates,
                false_empty=bool(correct and not predicted))


def build_episode(row,events,tok,cfg):
    """Insert actual observed lists; discard events whose repair cannot fit."""
    tokens=[];cursor=0;accepted=[]
    for event in sorted(events,key=lambda e:e['start']):
        start,end,lo,hi,action=(event[k] for k in ('start','end','lo','hi','action'))
        bad=(row.tokens[start:lo]+event['predicted']+row.tokens[hi:action]
             +[tok.id('<MISTAKE>'),tok.id(';')])
        shift=len(tokens)-cursor
        repair_end=end+shift+len(bad)
        if repair_end>cfg.model.max_seq_len-cfg.protocol.cot_margin-3:
            continue
        tokens.extend(row.tokens[cursor:start])
        bad_start=len(tokens);tokens.extend(bad)
        repair_start=len(tokens);tokens.extend(row.tokens[start:end]);cursor=end
        accepted.append(dict(bad_start=bad_start,mistake_action=repair_start-2,
            repair_start=repair_start,repair_end=len(tokens),
            repair_action=action+shift+len(bad),original_start=start,original_end=end,
            candidate_start=lo+shift,candidate_end=lo+shift+len(event['predicted']),
            predicted=event['predicted'],
            features=event['features']))
    if not accepted:
        return None
    tokens.extend(row.tokens[cursor:])
    answer=len(tokens)-len(row.tokens[row.answer_index:])
    result=replace(row,tokens=tokens,answer_index=answer,
        trace_indices=list(range(row.prompt_len,answer))+[answer+1],
        meta={**row.meta,'mistake':True,'mistake_field':'cands','natural_events':accepted,
              'natural_complex':len(accepted)>1 or any(e['features']['edits_lower_bound']>=2 for e in accepted)})
    result=truncate_example(result,tok,cfg.model.max_seq_len,cfg.protocol.cot_margin)
    for event in accepted:
        if event['repair_end']>result.answer_index:
            raise ValueError('natural repair was truncated')
        if result.tokens[event['repair_start']:event['repair_end']]!=row.tokens[event['original_start']:event['original_end']]:
            raise ValueError('natural repair differs from its canonical record')
        if result.tokens[event['candidate_start']:event['candidate_end']]!=event['predicted']:
            raise ValueError('observed candidate tokens were modified')
    if result.tokens[result.answer_index]!=tok.label_id(row.label):
        raise ValueError('natural episode changed its label')
    return result


def write_pool(source,cfg,tok,rows,splits,provenance,statistics):
    path=pool_path(source,cfg,tok)
    flat=[];offsets=[0];fields=[];metadata=[]
    for row in rows:
        flat.extend(row.tokens);offsets.append(len(flat))
        fields.append([row.prompt_len,row.answer_index,row.label,row.filler_tokens])
        meta={k:v for k,v in row.meta.items() if k not in {'reference_steps','node_token','full_body'}}
        metadata.append(json.dumps(meta,separators=(',',':')))
    atomic_savez(path,tokens=np.asarray(flat,dtype=np.int32),offsets=np.asarray(offsets,dtype=np.int64),
        fields=np.asarray(fields,dtype=np.int32).reshape(-1,4),metadata=np.asarray(metadata,dtype=str),
        query_offsets=np.arange(len(rows)+1,dtype=np.int64),split=np.asarray(splits,dtype=np.int8))
    atomic_json(path.with_suffix('.json'),dict(identity=identity(source,cfg,tok),
        provenance=provenance,statistics=statistics,bytes=path.stat().st_size))
    mark_complete(path,file_sha256(path))
    return path


class NaturalCorrectionPool(PreparedRows):
    """RAM-only sampling of observed episodes from the active graph scale."""
    def __init__(self,source,cfg,tok):
        path=pool_path(source,cfg,tok)
        if not is_complete(path) or load_json(path.with_suffix('.json'))['identity']!=identity(source,cfg,tok):
            raise FileNotFoundError(f'missing/stale natural episode pool: {path}')
        self.protocol=cfg.protocol.name
        self.arrays=load_npz(path)
        self.tok=tok
        self.train_rows=[self.row(int(i),0) for i in np.flatnonzero(self.arrays['split']==0)]
        self.by_label={label:np.asarray([i for i,r in enumerate(self.train_rows) if r.label==label],dtype=np.int64)
                       for label in (0,1)}
        self.complex={label:np.asarray([i for i in self.by_label[label] if self.train_rows[i].meta['natural_complex']],dtype=np.int64)
                      for label in (0,1)}
        if any(len(ids)==0 for ids in self.by_label.values()):
            raise ValueError('natural training pool must include both labels')
        self._cases={}

    def sample(self,label,rng,complex_share):
        prefer=bool(rng.random()<complex_share)
        ids=self.complex[label] if prefer and len(self.complex[label]) else self.by_label[label]
        return self.train_rows[int(ids[int(rng.integers(len(ids)))])]

    def cases(self,split,limit,seed):
        key=(split,limit,seed)
        if key in self._cases:return self._cases[key]
        split_id={'train':0,'val':1,'test':2}[split]
        if split_id==0:raise ValueError('training errors cannot be used as held-out diagnostics')
        ids=np.flatnonzero(self.arrays['split']==split_id)
        rng=np.random.default_rng(seed)
        ids=rng.permutation(ids)[:limit or len(ids)]
        cases=[]
        for index in ids:
            row=self.row(int(index),0)
            event=row.meta['natural_events'][int(rng.integers(len(row.meta['natural_events'])))]
            cases.append(dict(field='cands',bad=row.tokens[:event['mistake_action']],
                clean=row.tokens[:event['bad_start']]+row.tokens[event['repair_start']:event['repair_action']],
                expected=row.tokens[event['repair_start']:event['repair_end']]))
        if not cases:raise ValueError(f'no observed {split} errors available for diagnostics')
        self._cases[key]=cases
        return cases

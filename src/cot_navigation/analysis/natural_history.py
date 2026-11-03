"""Match unedited natural wrong/clean readouts without repairing either text.

Every pair has the same generating source, grammatical field and within-record
offset; token identity is matched when available, otherwise token family.
Natural clean prefixes need not finish with a correct answer. No position at
or after an explicit MISTAKE is included in the primary population.
"""
from bisect import bisect_left
from collections import Counter,defaultdict
import numpy as np

FIELDS=['frontier','visited','candidates','action','semicolon']


def spans(trace,semi):
    start=0;out=[]
    for p,t in enumerate(trace):
        if t==semi:out.append((start,p+1));start=p+1
    if start<len(trace):out.append((start,len(trace)))
    return out


def token_fields(trace,bar,semi):
    result=[];bars=0
    for token in trace:
        if token==semi:result.append('semicolon');bars=0;continue
        if token==bar:bars+=1
        result.append(FIELDS[min(bars,3)])
    return result


def annotate(cases,tok):
    out=[]
    for sid,case in enumerate(cases):
        trace=list(case['trace']);marker=case['marker_at']
        cut=len(trace) if marker is None else marker
        trace=trace[:cut]
        first=case['first'];tau=first['token']-1 if first is not None else None
        out.append({**case,'sid':sid,'trace':trace,'spans':spans(trace,tok.id(';')),
            'fields':token_fields(trace,tok.id('|'),tok.id(';')),
            'tau':tau,'has_observed_error':tau is not None and tau<len(trace)})
    return out


def match_readouts(cases,tok,seed=20261030,positive_source=None,*,error_offsets=None,anchor='first_error'):
    """One natural positive trajectory per query, all its eligible focus tokens.

    The negative is a separately generated legal prefix, never a repaired or
    canonical replacement. All splitting occurs before calling this function.
    The same original negative query is used at most once per cell. Source
    matching prevents generator identity from becoming the binary label.
    """
    # Optional error-relative window preserves the target and natural texts.
    # Negatives still match the ACTUAL record offset, not the error-relative one.
    if error_offsets is not None:
        error_offsets=tuple(error_offsets)
        if not error_offsets or any(not isinstance(o,int) or o<0 for o in error_offsets):
            raise ValueError('error_offsets must contain nonnegative integer offsets')
        if len(set(error_offsets))!=len(error_offsets):raise ValueError('Duplicate error offset')
    if anchor not in {'first_error','action_boundary'}:raise ValueError('Unknown readout anchor')
    if anchor!='first_error' and error_offsets is None:raise ValueError('Anchor requires explicit error_offsets')
    rng=np.random.default_rng(seed);legal=defaultdict(list);exact=defaultdict(list)
    for c in cases:
        legal_end=min(len(c['trace']),c['tau'] if c['tau'] is not None else len(c['trace']))
        for ri,(begin,end) in enumerate(c['spans']):
            for p in range(begin,min(end,legal_end)):
                token=c['trace'][p];field=c['fields'][p];off=p-begin
                item=(p,ri,c['sid'],c['query'],token)
                legal[(c['source'],off,field)].append(item)
                exact[(c['source'],off,field,token)].append(item)
    # Sorting by reasoning position supports nearest-position matching without
    # scanning the whole legal corpus for every token.
    for pool in [legal,exact]:
        for k,v in pool.items():
            v.sort(key=lambda r:(r[0],r[2]));pool[k]=(v,[r[0] for r in v])
    positive_by_query=defaultdict(list)
    for c in cases:
        if c['has_observed_error'] and (positive_source is None or c['source']==positive_source):
            positive_by_query[c['query']].append(c)
    positives=[v[int(rng.integers(len(v)))] for v in positive_by_query.values()]
    rng.shuffle(positives)
    selected=[];used=defaultdict(set);stats=Counter();source_counts=Counter();kinds=Counter()
    def choose(pool,key,p,ri,query,token,used_queries):
        if key not in pool:return None
        options,positions=pool[key];j=bisect_left(positions,p)
        # Expand when nearby queries have already been used; no random negative
        # can cross the legal prefix boundary because pools contain legal tokens only.
        for radius in [32,128,len(options)]:
            nearby=[r for r in options[max(0,j-radius):min(len(options),j+radius)]
                    if r[3]!=query and r[3] not in used_queries
                    and (tok.node_lo<=r[4]<tok.node_lo+tok.n_node_tokens)==
                        (tok.node_lo<=token<tok.node_lo+tok.n_node_tokens)]
            if nearby:
                return min(nearby,key=lambda r:abs(r[0]-p)+4*abs(r[1]-ri))
        return None
    for event,c in enumerate(positives):
        first_record=c['first']['record']-1
        if first_record>=len(c['spans']):continue
        source_counts[c['source']]+=1;kinds[c['first']['kind']]+=1
        if error_offsets is None:
            windows=[(phase,ri,range(max(begin,c['tau']),end))
                     for phase in range(3) for ri in [first_record+phase]
                     if ri<len(c['spans']) for begin,end in [c['spans'][ri]]]
        else:
            starts=[a for a,b in c['spans']]
            anchor_pos=c['tau']
            if anchor=='action_boundary':
                begin,end=c['spans'][first_record]
                bars=[p for p in range(begin,end) if c['trace'][p]==tok.id('|')]
                if len(bars)<3 or bars[2]<c['tau']:continue
                anchor_pos=bars[2]
            windows=[(0,bisect_left(starts,p+1)-1,[p]) for o in error_offsets
                     for p in [anchor_pos+o] if p<len(c['trace'])]
        for phase,ri,positions in windows:
            begin,end=c['spans'][ri]
            for p in positions:
                record_off=p-begin
                off=record_off if error_offsets is None else p-anchor_pos
                cell=phase*10000+off;field=c['fields'][p];token=c['trace'][p]
                key=(c['source'],record_off,field)
                neg=choose(exact,key+(token,),p,ri,c['query'],token,used[cell])
                if neg is None:neg=choose(legal,key,p,ri,c['query'],token,used[cell])
                if neg is None:stats['unmatched_tokens']+=1;continue
                np_,nri,nsid,nq,ntoken=neg;used[cell].add(nq)
                stats['pairs']+=1;stats['exact_token']+=int(ntoken==token)
                stats['absolute_progress_gap']+=abs(np_-p)
                for y,sid,pos,query in [(1,c['sid'],p,c['query']),(0,nsid,np_,nq)]:
                    selected.append(dict(sid=sid,pos=pos,query=query,event=event,
                        positive_query=c['query'],y=y,cell=cell,phase=phase,offset=off,
                        field=field,kind=c['first']['kind'],exact_token=ntoken==token,
                        source=c['source']))
    return selected,dict(stats,positive_queries=len(positives),positive_sources=dict(source_counts),
        first_types=dict(kinds),exact_token_fraction=stats['exact_token']/max(stats['pairs'],1),
        mean_progress_gap=stats['absolute_progress_gap']/max(stats['pairs'],1))

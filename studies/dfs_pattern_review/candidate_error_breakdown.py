"""Reclassify original fixed DFS-state candidate FIRST errors independently.

No generation, checkpoint loading, or training. Validate each expected set
against the graph and the already-audited visited state. Also count duplicate
entries, which set-only missing/extra summaries do not capture fully.
"""
from pathlib import Path
from collections import Counter
import sys,json,gzip
import numpy as np
ROOT=Path(str(next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'pyproject.toml').is_file() and (_p / 'studies').is_dir())))
sys.path.insert(0,str(ROOT/'src'))
from cot_navigation.config import load_config
from cot_navigation.data.dataset import ParametricWorld
from cot_navigation.protocols.tokenizer import Tokenizer

OUT=ROOT/'studies/dfs_pattern_review'
OLD=ROOT/'studies/fixed100_legality'
results={}
for setting in ('simple','hard'):
    manifest=json.loads((OLD/f'{setting}_dfs_state.json').read_text())
    cfg=load_config([Path(manifest['run'])/'resolved_config.yaml']);tok=Tokenizer.from_config(cfg)
    world=ParametricWorld(Path(cfg.data.root)/f'{cfg.data.graph_shape}-seed{cfg.data.graph_seed}')
    nodes=world.node_vocab_ids(tok);inverse={v:i for i,v in enumerate(nodes)}
    graph={nodes[u]:set(nodes[v] for v in vs) for u,vs in enumerate(world.adj())}
    labels=lambda values:[f'V{inverse[v]}' if v in inverse else tok.name(v) for v in values]
    rows=[json.loads(line) for line in gzip.open(OLD/f'{setting}_dfs_state_full_test_semantic.jsonl.gz','rt')]
    counts=Counter();patterns=Counter();cases=[];lengths=Counter();pushed=Counter()
    for row in rows:
        err=row['audit']['first_error']
        if not err or err['kind']!='candidate_mismatch':continue
        d=err['detail'];cur=d['current'];visited=set(d['visited']);neighbors=graph[cur]
        expected=neighbors-visited;shown=d['shown'];actual=set(shown)
        missing=expected-actual;extra=actual-expected;dups=[v for v,n in Counter(shown).items() if n>1]
        assert expected==set(d['expected']) and missing==set(d['missing']) and extra==set(d['extra'])
        flags=dict(missing=bool(missing),extra=bool(extra),duplicate=bool(dups),
            extra_visited_any=bool(extra&visited),extra_visited_neighbor=bool(extra&visited&neighbors),
            extra_non_neighbor=bool(extra-neighbors),first_record=err['record']==0,
            empty_shown=not shown,nonempty_when_expected_empty=not expected and bool(shown))
        counts.update(k for k,v in flags.items() if v)
        patterns['+'.join(k for k in ('missing','extra','duplicate') if flags[k])]+=1
        lengths['shorter' if len(shown)<len(expected) else 'longer' if len(shown)>len(expected) else 'same']+=1
        action=d['shown_action'];choice=action[1] if len(action)==2 and action[0]==tok.id('<PUSH>') else None
        if choice is not None:
            pushed['PUSH']+=1
            pushed['valid']+=choice in expected
            pushed['nonedge']+=choice not in neighbors
            pushed['revisited']+=choice in visited
        cases.append(dict(query_id=row['query_id'],label=row['label'],record=err['record']+1,
            current=f'V{inverse[cur]}',source=f'V{row["source"]}',target=f'V{row["target"]}',
            visited=labels(d['visited']),neighbors=labels(sorted(neighbors,key=inverse.get)),
            expected=labels(sorted(expected,key=inverse.get)),shown=labels(shown),
            missing=labels(sorted(missing,key=inverse.get)),extra=labels(sorted(extra,key=inverse.get)),
            duplicate=labels(dups),extra_visited_neighbor=labels(sorted(extra&visited&neighbors,key=inverse.get)),
            extra_non_neighbor=labels(sorted(extra-neighbors,key=inverse.get)),
            chosen=None if choice is None else labels([choice])[0],flags=flags))
    examples={}
    for kind in ('missing','extra_visited_neighbor','extra_non_neighbor','duplicate'):
        pool=[r for r in cases if r['flags'][kind]]
        if pool:
            if kind=='missing':pool.sort(key=lambda r:(bool(r['extra']) or bool(r['duplicate']),len(r['visited']),r['query_id']))
            else:pool.sort(key=lambda r:(len(r['visited']),r['query_id']))
            examples[kind]=pool[0]
    results[setting]=dict(total_queries=len(rows),candidate_first_errors=len(cases),counts=dict(counts),
                         disjoint_patterns=dict(patterns),list_lengths=dict(lengths),shown_push=dict(pushed),
                         examples=examples,rows=cases,checkpoint=manifest['run'],step=manifest['checkpoint_step'])
    print(setting,'counts',dict(counts),'patterns',dict(patterns),'lengths',dict(lengths),'PUSH',dict(pushed))
    for kind,e in examples.items():
        print(kind,{k:e[k] for k in ['query_id','record','current','visited','neighbors','expected','shown','missing','extra','duplicate','chosen']})
(OUT/'candidate_error_breakdown.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')

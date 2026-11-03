"""Check whether the actual generated witness occurred as a training subpath."""
import gzip
import importlib.util
import json
from pathlib import Path

s=importlib.util.spec_from_file_location('audit',Path(__file__).with_name('audit.py'))
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
from cot_navigation.data.scale_prepared import ScalePreparedRows

for n in (36,100):
    folder=m.RUN/f'fixed/n{n}/stage_b/path';cfg=m.load_config([folder/'resolved_config.yaml'])
    tok=m.Tokenizer.from_config(cfg);world=m.ParametricWorld(Path(cfg.data.root)/'main-seed0')
    p=m.OUT/f'n{n}_path_queries.jsonl.gz'
    with gzip.open(p,'rt') as f:rows=[json.loads(x) for x in f]
    queries={};pending={}
    for r in rows:
        if not r['label'] or not r['path']['valid_positive_witness']:continue
        sequence=tuple(r['path']['nodes']);queries[r['query_id']]=sequence
        pending.setdefault(sequence[0],{}).setdefault(len(sequence),{})[sequence]=r['query_id']
    walk_seen=set()
    for i in range(world.n_walks()):
        seq=world.walk_nodes_at(i)
        for q,path in queries.items():
            if len(path)>len(seq):continue
            if any(tuple(seq[j:j+len(path)])==path for j in range(len(seq)-len(path)+1)):walk_seen.add(q)
    found={};cached=ScalePreparedRows(world,cfg,tok);inverse={v:u for u,v in enumerate(world.node_vocab_ids(tok))}
    for q in world.indices('train').tolist():
        for v in range(cached.variants(q)):
            row=cached.row(q,v);nodes=[inverse[x] for x in row.tokens[row.prompt_len:row.answer_index] if x in inverse]
            for j,u in enumerate(nodes):
                if u not in pending:continue
                lengths=pending[u]
                for length,options in list(lengths.items()):
                    candidate=tuple(nodes[j:j+length])
                    if candidate in options:
                        i=options.pop(candidate);found[i]=dict(training_query=q,variant=v,training_label=int(world.queries['label'][q]))
                    if not options:lengths.pop(length)
                if not lengths:pending.pop(u)
            if not pending:break
        if not pending:break
    for r in rows:
        r['generated_witness_seen_in_walk']=r['query_id'] in walk_seen
        r['generated_witness_seen_in_cached_training_path']=r['query_id'] in found
        r['exact_generated_witness_training_example']=found.get(r['query_id'])
    with gzip.open(p,'wt') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    valid=[x for x in rows if x['label'] and x['path']['valid_positive_witness']]
    novel=[x for x in valid if not x['generated_witness_seen_in_walk'] and not x['generated_witness_seen_in_cached_training_path']]
    report=json.loads((m.OUT/f'n{n}_path.json').read_text())
    report['exact_generated_witness_exposure']=dict(valid_positive_witnesses=len(valid),seen_in_walk=len(walk_seen),
        seen_in_cached_training_path=len(found),novel_to_both=len(novel),novel_query_ids=[x['query_id'] for x in novel])
    m.write(m.OUT/f'n{n}_path.json',report)
    m.write(m.OUT/f'n{n}_path_novel_examples.json',novel[:5])
    print(n,report['exact_generated_witness_exposure'],flush=True)

"""Audit actual cached positive AND negative PATH supervision, CPU only."""
import gzip
import importlib.util
import json
from pathlib import Path

spec=importlib.util.spec_from_file_location('audit',Path(__file__).with_name('audit.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
from cot_navigation.data.scale_prepared import ScalePreparedRows


for n in (36,100):
    folder=m.RUN/f'fixed/n{n}/stage_b/path'
    cfg=m.load_config([folder/'resolved_config.yaml']);tok=m.Tokenizer.from_config(cfg)
    world=m.ParametricWorld(Path(cfg.data.root)/'main-seed0')
    meta=m.metadata(world);pending={}
    for i,r in meta.items():
        if r['label']:pending.setdefault(r['source'],{})[r['target']]=i
    prepared=ScalePreparedRows(world,cfg,tok)
    inverse={v:u for u,v in enumerate(world.node_vocab_ids(tok))}
    found={};checked=0
    for q in world.indices('train').tolist():
        for v in range(prepared.variants(q)):
            row=prepared.row(q,v);body=row.tokens[row.prompt_len:row.answer_index]
            nodes=[inverse[x] for x in body if x in inverse];checked+=1
            for j,u in enumerate(nodes):
                if u not in pending:continue
                targets=pending[u]
                hits=set(nodes[j+1:]) & set(targets)
                for t in hits:
                    i=targets.pop(t)
                    found[i]=dict(training_query=q,variant=v,training_label=int(world.queries['label'][q]),
                        training_source=int(world.queries['s_topo'][q]),training_target=int(world.queries['t_topo'][q]),
                        witness=nodes[j:nodes.index(t)+1])
                if not targets:pending.pop(u)
            if not pending:break
        if not pending:break
    print('EXPOSURE',n,'found',len(found),'remaining',sum(map(len,pending.values())),'cached_rows_checked',checked,flush=True)
    for protocol in ('direct','path'):
        p=m.OUT/f'n{n}_{protocol}_queries.jsonl.gz'
        with gzip.open(p,'rt') as f:rows=[json.loads(x) for x in f]
        for r in rows:
            r['any_cached_training_path_witness']=r['query_id'] in found
            r['cached_training_path_example']=found.get(r['query_id'])
        with gzip.open(p,'wt') as f:
            for r in rows:f.write(json.dumps(r)+'\n')
        report=json.loads((m.OUT/f'n{n}_{protocol}.json').read_text())
        strict=[r for r in rows if r['label'] and not r['walk_witness'] and not r['any_cached_training_path_witness']]
        report['summary']['no_walk_no_cached_training_path']=m.rate(strict)
        if protocol=='path':
            report['summary']['no_walk_no_cached_training_path']['positive_witness_validity']=m.rate([
                dict(correct=x['path']['valid_positive_witness']) for x in strict])
        report['exposure']=dict(positive_test_queries=sum(r['label'] for r in rows),
            positive_with_cached_training_path=len(found),
            positive_with_negative_training_path_example=sum(x['training_label']==0 for x in found.values()),
            cached_training_rows_checked=checked,
            note='Cache availability, not verified optimizer exposure. Direct receives no PATH supervision; PATH exposure is a diagnostic annotation for its shared query panel.')
        m.write(m.OUT/f'n{n}_{protocol}.json',report)
        m.write(m.OUT/f'n{n}_{protocol}_strict_examples.json',strict[:5])
    m.write(m.OUT/f'n{n}_path_exposure.json',found)

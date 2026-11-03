"""Export concrete state records only; reuse already measured attention."""
from pathlib import Path
import json
import measure as m

cases=[('context_n16',50520,6),('context_n16',50512,5),
       ('fixed_simple_n100',151,5),('fixed_hard_n100',780,7)]
result=[]
for name,index,keep in cases:
    folder,data=m.SETTINGS[name];cfg=m.base.load_config([folder/'resolved_config.yaml'])
    tok=m.base.Tokenizer.from_config(cfg)
    context=name.startswith('context')
    if context:
        source=m.ContextStore(data);item=m.base.prepare_context(source,tok,index);q=source.arrays
    else:
        source=m.ParametricWorld(data);q=source.queries
        row=source.render_query(index,'dfs_state',tok)
        header=[m.base.world_token(cfg,tok,data)] if cfg.train.world_roots else []
        item=dict(tokens=header+row.tokens,prompt_len=row.prompt_len+len(header),node_tokens=source.node_vocab_ids(tok))
    tokens=item['tokens'];inverse={v:f'V{i}' for i,v in enumerate(item['node_tokens'])}
    label=lambda t:inverse.get(t,tok.name(t))
    records=[]
    for ri,rec in enumerate(m.records(tokens,item['prompt_len'],tok)[:keep]):
        a,b,c=rec['separators']
        records.append(dict(record=ri+1,start=rec['start'],end=rec['end'],
            stack=[label(t) for t in rec['frontier']],visited=[label(t) for t in rec['visited']],
            candidates=[label(t) for t in rec['candidates']],action=[label(t) for t in rec['action']],
            positioned_tokens=[dict(position=p,token=label(tokens[p])) for p in range(rec['start'],rec['end']+1)]))
    result.append(dict(model=name,query_id=index,source='V'+str(int(q['s_topo'][index])),
        target='V'+str(int(q['t_topo'][index])),label=int(q['label'][index]),
        prompt=[dict(position=p,token=label(tokens[p])) for p in range(max(0,item['prompt_len']-5),item['prompt_len'])],
        records=records))
(m.OUT/'concrete_examples.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
for case in result:
    print(case['model'],case['query_id'],case['source'],case['target'])
    for r in case['records']:
        print(r['record'],r['start'],' '.join(r['stack']),'|',' '.join(r['visited']),'|',' '.join(r['candidates']),'|',' '.join(r['action']))

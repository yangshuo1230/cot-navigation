"""Concrete first/continuation/end candidate-token reads, no model training."""
import json
import numpy as np
import torch
import measure as m
from candidate_generation import decode

torch.set_num_threads(2);output=[]
for name,index,selected in [('context_n16',50520,[0,2]),('fixed_simple_n100',151,[0,2,4,5]),('fixed_hard_n100',780,[0,2])]:
    folder,data=m.SETTINGS[name];cfg,tok,model,step=m.base.load_model(folder,torch.device('cuda'))
    context=name.startswith('context');source=m.ContextStore(data) if context else m.ParametricWorld(data)
    if context:item=m.base.prepare_context(source,tok,index)
    else:
        row=source.render_query(index,'dfs_state',tok)
        header=[m.base.world_token(cfg,tok,data)] if cfg.train.world_roots else []
        item=dict(tokens=header+row.tokens,prompt_len=row.prompt_len+len(header),node_tokens=source.node_vocab_ids(tok))
    tokens=item['tokens'];P=item['prompt_len'];recs=m.records(tokens,P,tok)
    inverse={t:f'V{j}' for j,t in enumerate(item['node_tokens'])}
    label=lambda t:inverse.get(t,tok.name(t))
    es=[]
    for ri in selected:
        r=recs[ri];a,b,c=r['separators']
        for q in range(b,c):es.append(dict(name='candidate',q=q,keys=[q],record=ri))
    reader=m.Reader(model);weights,qmap,logits=reader.forward(item,es);reader.close()
    states=[]
    for ri in selected:
        r=recs[ri];a,b,c=r['separators'];cur=r['frontier'][-1]
        oldsame=[old for old in recs[:ri] if old['frontier'][-1]==cur]
        oldcands=[k for old in oldsame for k in range(old['separators'][1]+1,old['separators'][2])]
        oldpush=[old for old in oldsame if old['action'][0]==tok.id('<PUSH>')]
        pp=oldpush[-1]['separators'][2]+1 if oldpush else None
        events=[]
        for q in range(b,c):
            qi=qmap[q];mat=np.stack([weights[li][qi,:q+1] for li in range(6)])
            groups=dict(current_visited=[k for k in range(a+1,b) if tokens[k]==cur],stack_top=[a-1],
                        previous_same_current_candidates=oldcands,
                        previous_candidates_same_next=[k for k in oldcands if tokens[k]==tokens[q+1]],
                        previous_push_pair=[] if pp is None else [pp,pp+1],
                        candidate_prefix=list(range(b+1,q+1)))
            events.append(dict(q=q,input=label(tokens[q]),reference_next=label(tokens[q+1]),
                model_next=label(int(logits[q].argmax())),
                groups={key:mat[:,ks].sum(1).tolist() if ks else [0.]*6 for key,ks in groups.items()},
                top_keys=[[dict(position=int(k),token=label(tokens[k]),mass=float(mat[li,k])) for k in np.argsort(mat[li])[-3:][::-1]] for li in range(6)]))
        states.append(dict(record=ri+1,current=label(cur),stack=[label(t) for t in r['frontier']],
            visited=[label(t) for t in r['visited']],candidates=[label(t) for t in r['candidates']],events=events))
    field_inputs=[]
    for ri in selected:
        r=recs[ri];b=r['separators'][1]
        field_inputs.append(dict(start=r['start'],prompt_len=P,prefix=tokens[:b+1],node_set=set(item['node_tokens'])))
    generated,identity=decode(model,tok,field_inputs,'full')
    assert identity['argmax_drift']==0
    for s,g in zip(states,generated):
        s['autoreg_generated']=[label(t) for t in g['tokens']]
        s['autoreg_closed']=g['closed'];s['autoreg_error']=g['error']
        s['autoreg_set_correct']=g['closed'] and len(g['tokens'])==len(set(g['tokens'])) and set(s['autoreg_generated'])==set(s['candidates'])
    output.append(dict(model=name,query_id=index,checkpoint_step=step,states=states))
    del model;torch.cuda.empty_cache()
(m.OUT/'candidate_token_examples.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
for r in output:
    print(r['model'],r['query_id'])
    for s in r['states']:
        print(s['record'],s['current'],s['candidates'],[(e['input'],e['model_next']) for e in s['events']])

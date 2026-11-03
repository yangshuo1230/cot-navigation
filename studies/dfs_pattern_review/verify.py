"""Independent full-attention cross-check on the original pictured example."""
import json
import numpy as np
import torch
import measure as m

torch.set_num_threads(2)
cfg,tok,model,step=m.base.load_model(m.base.CONTEXT_RUN,torch.device('cuda:0'))
store=m.ContextStore(m.base.CONTEXT_DATA)
item=m.base.prepare_context(store,tok,50520);es=m.events(item,tok)
reader=m.Reader(model);weights,qmap,_=reader.forward(item,es)
with torch.inference_mode():
    _,full=model(torch.tensor(item['tokens'][:max(qmap)+1],device='cuda')[None],return_attn=True)
errors=[]
for li,w in weights.items():
    direct=full[li][0,0].cpu().float().numpy()
    errors.append(float(np.max(np.abs(w-direct[sorted(qmap)]))))
assert max(errors)<1e-4
P=item['prompt_len'];source_token=item['tokens'][P-3]
edge_offsets=[]
attn0=full[0][0,0].cpu().float().numpy()
for edge_start in range(0,P-4,3):
    if item['tokens'][edge_start+1]!=source_token:continue
    src=edge_start+1;dst=edge_start+2
    edge_offsets.append(dict(edge_start=edge_start,source_position=src,destination_position=dst,
        source_plus2_reads_source=float(attn0[src+2,src]),
        source_plus3_reads_source=float(attn0[src+3,src]),
        destination_plus3_reads_destination=float(attn0[dst+3,dst])))
reader.close()
checked=[]
for name in m.SETTINGS:
    r=json.loads((m.OUT/f'{name}.json').read_text());assert r['version']==3
    assert not set(r['ids']['val'])&set(r['ids']['test'])
    assert r['readonly_logits_identity'] and len(r['ids']['test'])==20
    for kind,metric in r['metrics'].items():
        assert 0<metric['n_queries']<=20
        assert metric['n_queries']<=metric['n_events']<=8*metric['n_queries']
        assert np.all(np.asarray(metric['query_macro'])>=-1e-7)
        assert np.all(np.asarray(metric['query_macro'])<=1+1e-5)
    for e in r['examples'].values():
        assert all(k['position']<=e['q'] for k in e['keys'])
    checked.append(name)
q=json.loads((m.OUT/'context_n16.json').read_text())['pictured_queries']['50520']
assert abs(q['push_three_2'][0]['total'][0]-.947)<.003
assert abs(q['found_stack_top'][0]['total'][3]-.965)<.003
assert abs(q['prompt_source'][0]['total'][0]-.915)<.003
result=dict(full_attention_row_max_errors=errors,fp32_full_attention_agreement=True,
    causal_key_positions=True,heldout_validation_calibration=True,readonly_logits_identical=True,
    independent_pictured_example_checks=True,models=checked,
    query50520_source_offset_audit=edge_offsets)
(m.OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(result)

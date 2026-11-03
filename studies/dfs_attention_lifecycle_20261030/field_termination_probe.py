"""Node identity edits and legal candidate permutations; exact route audit."""
import json,torch,numpy as np
import analyze as a

def main():
    torch.set_num_threads(4);cfg=a.load_config([a.RUN/'resolved_config.yaml']);tok=a.Tokenizer.from_config(cfg)
    a.TOK=tok
    model=a.build_model(cfg.model,len(tok),tok.pad).eval()
    model.load_state_dict(torch.load(a.RUN/'checkpoints/best.pt',map_location='cpu',weights_only=False)['model']);model.requires_grad_(False)
    store=a.ContextStore(a.ROOT/'data/scale_v1'/a.SETS['n100']);panel=json.loads((a.OUT/'plan.json').read_text())['panels']['n100']['test']
    edits=[];permutations=[];routes=[];shams=[]
    pairs=json.loads((a.OUT/'field_termination.json').read_text())['pairs']
    for spec in panel:
        row=store.render(spec['index'],'dfs_state',tok);seq=row.tokens[:-1];recs=a.parse(seq,row.prompt_len,tok)
        ids=torch.tensor(seq)[None]
        with torch.inference_mode():base,weights=model(ids,return_attn=True)
        for e in a.events(seq,row.prompt_len,recs):
            if e['phase'] not in ['stack_end','stack_continue','visited_end','visited_continue','candidate_end','candidate_continue','candidate_empty']:continue
            q,ri=e['q'],e['ri'];r=recs[ri];x=seq[q]
            # A new record's stack root after PUSH reads the added node at L1.
            # Index root by old visited tail (the predecessor in discovery order).
            successor_roots=[past['start'] for k,past in enumerate(recs[:ri+1]) if k and recs[k-1]['action']=='PUSH' and recs[k-1]['visited'][-1]==x]
            root_carried_current=[past['start'] for past in recs[:ri+1] if past['visited'][-1]==x]
            fieldbars=[p for past in recs[:ri+1] for p in [past['a'],past['b'],past['c']] if p<=q]
            masks=dict(minus3=[q-3],successor_roots=successor_roots,root_carried_current=root_carried_current,
                all_field_bars=fieldbars,current_stack_top=[r['a']-1],current_stack_root=[r['start']],
                current_visited_bar=[r['b']],previous_push_argument=[recs[ri-1]['c']+2] if ri and recs[ri-1]['action']=='PUSH' else [])
            rt={name:[float(w[0,0,q,[p for p in ps if 0<=p<=q]].sum()) for w in weights] for name,ps in masks.items()}
            routes.append(dict(index=spec['index'],phase=e['phase'],sub=e['sub'],q=q,record=ri+1,
                next_correct=int(base[0,q].argmax())==seq[q+1],has_successor_root=bool(successor_roots),routes=rt))
        del weights,base
        if spec['band']!='long':continue
        for field,lo,hi in [('stack','start','a'),('visited','a','b'),('candidates','b','c')]:
            possible=[(ri,r) for ri,r in enumerate(recs) if r[hi]-r[lo]-(field!='stack')>=2]
            if not possible:continue
            ri,r=min(possible,key=lambda pair:(abs(pair[0]-len(recs)/2),pair[0]))
            end=r[hi];last=end-1;pen=end-2
            for direction,q,donor,expected in [('end_node_to_penultimate',last,pen,seq[last]),('penultimate_to_end_node',pen,last,tok.id('|'))]:
                original=seq[:q+1];changed=original[:];changed[q]=seq[donor]
                with torch.inference_mode():b=model(torch.tensor(original)[None]);c=model(torch.tensor(changed)[None])
                edits.append(dict(index=spec['index'],field=field,record=ri+1,direction=direction,q=q,
                    original_token=tok.name(seq[q]),edited_token=tok.name(seq[donor]),baseline_prediction=tok.name(int(b[0,q].argmax())),
                    baseline_correct=int(b[0,q].argmax())==seq[q+1],prediction=tok.name(int(c[0,q].argmax())),
                    expected_identity_lookup=tok.name(expected),follows_identity_lookup=int(c[0,q].argmax())==expected,
                    earlier_logit_max_difference=float((b[0,:q]-c[0,:q]).abs().max())))
            if field=='candidates':
                altered=seq[:];start=r['b']+1
                altered[start],altered[last]=altered[last],altered[start]
                for q,expected,kind in [(last-1,seq[start],'one_remaining'),(last,tok.id('|'),'all_emitted')]:
                    with torch.inference_mode():c=model(torch.tensor(altered[:q+1])[None])
                    pred=int(c[0,q].argmax());emitted=altered[start:q+1]
                    remaining=set(r['candidates'])-set(emitted)
                    permutations.append(dict(index=spec['index'],record=ri+1,kind=kind,q=q,
                        original_candidates=[tok.name(v) for v in r['candidates']],emitted=[tok.name(v) for v in emitted],
                        remaining=[tok.name(v) for v in remaining],prediction=tok.name(pred),
                        set_correct=(pred in remaining if remaining else pred==tok.id('|')),
                        predicted_stop=pred==tok.id('|')))
        for pair in [p for p in pairs if p['index']==spec['index'] and p['available']]:
            q=pair['stop_q'];ids=torch.tensor(seq[:q+1])[None]
            for li in [4,5]:
                module=model.blocks[li].attn;cap={}
                def grab(m,args,out):cap['out']=out[0].clone()
                h=module.register_forward_hook(grab)
                with torch.inference_mode():base=model(ids)
                h.remove()
                def identity(m,args,out):
                    z=out[0].clone();z[0,q]=cap['out'][0,q];return z,out[1]
                h=module.register_forward_hook(identity)
                with torch.inference_mode():same=model(ids)
                h.remove();assert torch.equal(base,same)
                shams.append(dict(index=spec['index'],field=pair['field'],layer=li+1,identical=True))
        a.save('field_termination_probe.json',dict(routes=routes,identity_edits=edits,candidate_permutations=permutations,shams=shams,complete=False))
        print(json.dumps(dict(index=spec['index'],routes=len(routes),edits=len(edits))),flush=True)
    a.save('field_termination_probe.json',dict(routes=routes,identity_edits=edits,candidate_permutations=permutations,shams=shams,complete=True))

if __name__=='__main__':main()

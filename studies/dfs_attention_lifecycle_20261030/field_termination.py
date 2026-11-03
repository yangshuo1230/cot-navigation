"""Exploratory stop/continue contrasts with same visible receiver token.

Local attention/FFN output swaps, not legal alternate trajectories. One receiver
only; historical activations are unchanged. Full results include failed swaps.
"""
import json,time,torch
import analyze as a

def main():
    torch.set_num_threads(4)
    cfg=a.load_config([a.RUN/'resolved_config.yaml']);tok=a.Tokenizer.from_config(cfg)
    model=a.build_model(cfg.model,len(tok),tok.pad).eval()
    model.load_state_dict(torch.load(a.RUN/'checkpoints/best.pt',map_location='cpu',weights_only=False)['model'])
    model.requires_grad_(False)
    store=a.ContextStore(a.ROOT/'data/scale_v1'/a.SETS['n100'])
    panel=json.loads((a.OUT/'plan.json').read_text())['panels']['n100']['test']
    specs=[s for s in panel if s['band']=='long']
    results=[];pairs=[]
    for spec in specs:
        row=store.render(spec['index'],'dfs_state',tok);seq=row.tokens[:-1]
        recs=a.parse(seq,row.prompt_len,tok);jobs=[]
        for field,lo,hi in [('stack','start','a'),('visited','a','b'),('candidates','b','c')]:
            ends=[];continues=[]
            for ri,r in enumerate(recs):
                start=r[lo]+(field!='stack');end=r[hi]
                if end<=start:continue
                ends.append((end-1,ri))
                continues.extend((q,ri) for q in range(start,end-1))
            matches=[(q,d,ri,di) for q,ri in ends for d,di in continues if seq[q]==seq[d]]
            if not matches:
                pairs.append(dict(index=spec['index'],field=field,available=False));continue
            # Prefer nearby records near the middle, independent of model outputs.
            q,d,ri,di=min(matches,key=lambda p:(abs(p[2]-len(recs)/2)+abs(p[3]-len(recs)/2),abs(p[0]-p[1]),p[0],p[1]))
            pair=dict(index=spec['index'],field=field,available=True,stop_q=q,continue_q=d,
                      stop_record=ri+1,continue_record=di+1,visible=tok.name(seq[q]),
                      continue_target=tok.name(seq[d+1]),stop_target='|')
            pairs.append(pair)
            jobs.extend([dict(**pair,direction='stop_to_continue',q=q,donor=d,expected=seq[d+1]),
                         dict(**pair,direction='continue_to_stop',q=d,donor=q,expected=tok.id('|'))])
        if not jobs:continue
        n=max(max(j['q'],j['donor']) for j in jobs)+1;ids=torch.tensor(seq[:n])[None]
        capture={};hooks=[]
        for li,block in enumerate(model.blocks):
            for kind,module in [('attention',block.attn),('ffn',block.ffn)]:
                def grab(module,args,out,li=li,kind=kind):capture[(li,kind)]=(out[0] if kind=='attention' else out).detach().clone()
                hooks.append(module.register_forward_hook(grab))
        with torch.inference_mode():base=model(ids)
        for h in hooks:h.remove()
        with torch.inference_mode():mathlog,weights=model(ids,return_attn=True)
        diff=float((base-mathlog).abs().max());assert diff<.01
        for job in jobs:
            q,d=job['q'],job['donor'];target=seq[q+1]
            routes=[]
            for li in range(6):
                z=weights[li][0,0,q,:q+1]
                routes.append(dict(layer=li+1,self_weight=float(z[q]),minus3=float(z[q-3]) if q>=3 else 0,
                    top=[dict(key=int(k),token=tok.name(seq[k]),weight=float(z[k])) for k in z.topk(5).indices]))
            for li in range(6):
                for kind in ['attention','ffn']:
                    block=model.blocks[li];module=block.attn if kind=='attention' else block.ffn
                    old=capture[(li,kind)][0,q];new=capture[(li,kind)][0,d]
                    def patch(module,args,out):
                        changed=(out[0] if kind=='attention' else out).clone();changed[0,q]=new
                        return (changed,out[1]) if kind=='attention' else changed
                    h=module.register_forward_hook(patch)
                    with torch.inference_mode():changed=model(ids[:,:q+1])
                    h.remove()
                    # Prefix SDPA may round slightly differently from a longer input.
                    earlier=float((changed[0,:q]-base[0,:q]).abs().max())
                    assert earlier<.01
                    pred=int(changed[0,q].argmax());original=int(base[0,q].argmax())
                    result={k:v for k,v in job.items() if k!='expected'}
                    result.update(layer=li+1,component=kind,target=tok.name(target),
                        baseline_prediction=tok.name(original),baseline_correct=original==target,
                        prediction=tok.name(pred),predicts_stop=pred==tok.id('|'),
                        follows_donor_class=(pred==tok.id('|')) if job['direction']=='continue_to_stop' else (tok.node_lo<=pred<tok.node_lo+tok.n_node_tokens),
                        exact_donor_target=pred==job['expected'],
                        stop_logit_margin_change=float((changed[0,q,tok.id('|')]-changed[0,q,tok.node_lo:tok.node_lo+tok.n_node_tokens].max())-(base[0,q,tok.id('|')]-base[0,q,tok.node_lo:tok.node_lo+tok.n_node_tokens].max())),
                        delta_norm=float((new-old).norm()),earlier_logit_max_difference=earlier,
                        routes=routes,math_sdpa_max_difference=diff)
                    results.append(result)
            a.save('field_termination.json',dict(definition=__doc__,pairs=pairs,results=results,complete=False))
        print(json.dumps(dict(index=spec['index'],pairs=len(jobs)//2,results=len(results),elapsed=time.time())),flush=True)
        del weights,capture
    a.save('field_termination.json',dict(definition=__doc__,pairs=pairs,results=results,complete=True))

if __name__=='__main__':main()

"""Matched literal-symbol value swaps at putative boundary-plan carriers."""
import json,torch
import analyze as a

def main():
    torch.set_num_threads(4);cfg=a.load_config([a.RUN/'resolved_config.yaml']);tok=a.Tokenizer.from_config(cfg)
    model=a.build_model(cfg.model,len(tok),tok.pad).eval();model.requires_grad_(False)
    model.load_state_dict(torch.load(a.RUN/'checkpoints/best.pt',map_location='cpu',weights_only=False)['model'])
    store=a.ContextStore(a.ROOT/'data/scale_v1'/a.SETS['n100']);panel=json.loads((a.OUT/'plan.json').read_text())['panels']['n100']['test'];results=[];missing=[]
    for spec in [s for s in panel if s['band']=='long']:
        row=store.render(spec['index'],'dfs_state',tok);seq=row.tokens[:-1];recs=a.parse(seq,row.prompt_len,tok);jobs=[]
        # Stack: previous semicolon summarizes the action before the new state.
        options=[]
        for ri,r in enumerate(recs):
            if ri<2:continue
            prev=recs[ri-1]
            for di,past in enumerate(recs[:ri-1]):
                expected=len(past['stack'])+(1 if past['action']=='PUSH' else -1 if past['action']=='POP' else 0)
                if past['action']==prev['action'] and expected>len(r['stack']):
                    options.append((ri,di,expected))
        if options:
            ri,di,n=min(options,key=lambda x:(abs(x[0]-len(recs)/2),abs(x[2]-len(recs[x[0]]['stack'])),x[1]));r=recs[ri]
            for layer in [2,4]:jobs.append(dict(field='stack',layer=layer,q=r['a']-1,key=recs[ri-1]['end']-1,donor=recs[di]['end']-1,record=ri+1,expected='continue',original_count=len(r['stack']),donor_count=n))
        else:missing.append(dict(index=spec['index'],field='stack'))
        # Candidate: old visit of same current node had more remaining neighbors.
        options=[]
        for ri,r in enumerate(recs):
            if not r['candidates']:continue
            for di,past in enumerate(recs[:ri]):
                if past['stack'][-1]==r['stack'][-1] and len(past['candidates'])>len(r['candidates']):options.append((ri,di))
        if options:
            ri,di=min(options,key=lambda x:(abs(x[0]-len(recs)/2),abs(x[0]-x[1])));r=recs[ri]
            for layer in [3,6]:jobs.append(dict(field='candidates',layer=layer,q=r['c']-1,key=r['b'],donor=recs[di]['b'],record=ri+1,expected='continue',original_count=len(r['candidates']),donor_count=len(recs[di]['candidates'])))
        else:missing.append(dict(index=spec['index'],field='candidates'))
        for job in jobs:
            li=job['layer']-1;q,key,donor=job['q'],job['key'],job['donor'];assert seq[key]==seq[donor]
            ids=torch.tensor(seq[:q+1])[None];attn=model.blocks[li].attn;cap={}
            def grab(m,args,out):cap['v']=out.view(1,out.shape[1],3,1,cfg.model.d_model)[0,:,2,0].clone()
            h=attn.qkv.register_forward_hook(grab)
            with torch.inference_mode():base,ws=model(ids,return_attn=True)
            h.remove();v=cap['v'];z=ws[li][0,0,q];controls=[p for p in range(q+1) if seq[p]==seq[key] and p!=key];control=min(controls,key=lambda p:float(z[p]));del ws
            for kind,k,d in [('main',key,donor),('sham',key,key),('low_attention',control,donor)]:
                delta=torch.nn.functional.linear((v[d]-v[k])*z[k],attn.proj.weight)
                def patch(m,args,out):
                    x=out[0].clone();x[0,q]+=delta;return x,out[1]
                h=attn.register_forward_hook(patch)
                with torch.inference_mode():changed,_=model(ids,return_attn=True)
                h.remove();assert torch.equal(base[:,:q],changed[:,:q])
                if kind=='sham':assert torch.equal(base,changed)
                pred=int(changed[0,q].argmax())
                results.append(dict(index=spec['index'],**job,condition=kind,actual_key=k,actual_donor=d,
                    symbol=tok.name(seq[key]),attention=float(z[k]),delta_norm=float(delta.norm()),
                    baseline_prediction=tok.name(int(base[0,q].argmax())),prediction=tok.name(pred),
                    predicts_stop=pred==tok.id('|'),changed_prediction=pred!=int(base[0,q].argmax()),earlier_logits_unchanged=True))
        a.save('field_boundary_carriers.json',dict(results=results,missing=missing,complete=False))
        print(json.dumps(dict(index=spec['index'],jobs=len(jobs))),flush=True)
    a.save('field_boundary_carriers.json',dict(results=results,missing=missing,complete=True))

if __name__=='__main__':main()

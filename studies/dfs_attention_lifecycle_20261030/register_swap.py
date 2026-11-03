"""Local value-path counterfactuals at symbol carriers, preserving Q/K weights.

Only the selected attention output at the last query position is changed.
No attention masking, no input edits, no global changes to historical states.
"""
import json,time
import numpy as np
import torch
import analyze as a

def main():
    torch.set_num_threads(4)
    cfg=a.load_config([a.RUN/'resolved_config.yaml']);tok=a.Tokenizer.from_config(cfg)
    model=a.build_model(cfg.model,len(tok),tok.pad).eval()
    state=torch.load(a.RUN/'checkpoints/best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(state['model']);model.requires_grad_(False)
    store=a.ContextStore(a.ROOT/'data/scale_v1'/a.SETS['n100'])
    plan=json.loads((a.OUT/'plan.json').read_text())['panels']['n100']['test']
    # All six long-band queries for initialization. All positive middle/long
    # queries for the final-semicolon experiment. No performance selection.
    jobs=[]
    for spec in plan:
        row=store.render(spec['index'],'dfs_state',tok);tokens=row.tokens[:-1];recs=a.parse(tokens,row.prompt_len,tok)
        if spec['band']=='long':
            ri=len(recs)//2;r=recs[ri]
            for phase,q in [('stack_init',r['start']-1),('visited_init',r['a'])]:
                jobs.append(dict(index=spec['index'],phase=phase,q=q,layer=5,key=row.prompt_len-1,
                                 donor=recs[0]['a'],control_key=recs[max(0,ri-1)]['a'],tokens=tokens[:q+1],target=tokens[q+1]))
        if spec['label']==1 and spec['band'] in ['middle','long']:
            q=len(tokens)-2
            jobs.append(dict(index=spec['index'],phase='final_semicolon_YES',q=q,layer=5,key=q,
                             donor=recs[-2]['end']-1,control_key=recs[0]['end']-1,tokens=tokens[:q+1],target=tokens[q+1]))
    results=[]
    for job in jobs:
        start=time.time();q,li=job['q'],job['layer']-1;attn=model.blocks[li].attn
        ids=torch.tensor(job['tokens'])[None];captured={}
        def grab(module,args,out):captured['v']=out.view(1,out.shape[1],3,1,cfg.model.d_model)[0,:,2,0,:].detach().clone()
        handle=attn.qkv.register_forward_hook(grab)
        with torch.inference_mode():base,weights=model(ids,return_attn=True)
        handle.remove();v=captured['v'];scores=weights[li][0,0,q].clone();del weights
        target=job['target'];base_logp=float(base[0,q].log_softmax(-1)[target]);base_pred=int(base[0,q].argmax())
        conditions=[]
        for kind,key,donor in [('swap_main',job['key'],job['donor']),('sham',job['key'],job['key']),('swap_low_attention_control',job['control_key'],job['donor'])]:
            # Frozen, baseline Q/K => exact A[q,key]*(Vdonor-Vkey)*WO.
            delta=torch.nn.functional.linear((v[donor]-v[key])*scores[key],attn.proj.weight)
            def edit(module,args,out):
                changed=out[0].clone();changed[0,q]+=delta
                return changed,out[1]
            h=attn.register_forward_hook(edit)
            with torch.inference_mode():changed,w=model(ids,return_attn=True)
            h.remove();del w
            if kind=='sham':assert torch.equal(base,changed)
            earlier=float((base[:,:q]-changed[:,:q]).abs().max()) if q else 0
            assert earlier==0
            logp=float(changed[0,q].log_softmax(-1)[target]);pred=int(changed[0,q].argmax())
            conditions.append(dict(kind=kind,key=key,donor=donor,key_token=tok.name(job['tokens'][key]),donor_token=tok.name(job['tokens'][donor]),
              attention=float(scores[key]),delta_norm=float(delta.norm()),gold_logp=logp,logp_drop=base_logp-logp,pred=tok.name(pred),
              next_correct=pred==target,changed_prediction=pred!=base_pred,earlier_logit_max_difference=earlier))
        result={k:v for k,v in job.items() if k not in ['tokens','target']}
        result.update(target=tok.name(target),baseline_prediction=tok.name(base_pred),baseline_correct=base_pred==target,baseline_gold_logp=base_logp,
                      conditions=conditions,elapsed_sec=time.time()-start)
        results.append(result);a.save('register_swaps.json',dict(definition='Local attention-value replacement; Q,K,attention weights preserved at intervened layer, same visible symbol donor for main. Downstream recomputed. Sham exact identity; earlier logits unchanged.',results=results))
        print(json.dumps(result),flush=True)
    a.save('REGISTER_SWAPS_COMPLETE.json',dict(jobs=len(jobs),sham_identity=True,earlier_positions_unchanged=True))

if __name__=='__main__':main()

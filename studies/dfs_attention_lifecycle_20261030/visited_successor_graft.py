"""Counterfactual insertion of a future discovery register into an earlier stop.
Exploratory sufficiency diagnostic, not a legal causal-generation trajectory.
Only one L5 receiver attention row changes; all other positions are untouched.
"""
import json,torch,math
import analyze as a
from cot_navigation.models.attention import apply_rope

def main():
    torch.set_num_threads(4);cfg=a.load_config([a.RUN/'resolved_config.yaml']);tok=a.Tokenizer.from_config(cfg)
    model=a.build_model(cfg.model,len(tok),tok.pad).eval();model.requires_grad_(False)
    model.load_state_dict(torch.load(a.RUN/'checkpoints/best.pt',map_location='cpu',weights_only=False)['model'])
    store=a.ContextStore(a.ROOT/'data/scale_v1'/a.SETS['n100']);pairs=json.loads((a.OUT/'field_termination.json').read_text())['pairs'];results=[]
    for pair in [p for p in pairs if p['field']=='visited' and p['available']]:
        row=store.render(pair['index'],'dfs_state',tok);seq=row.tokens[:-1];recs=a.parse(seq,row.prompt_len,tok)
        q=pair['stop_q'];r=recs[pair['stop_record']-1];x=seq[q]
        donors=[past for k,past in enumerate(recs) if k and recs[k-1]['action']=='PUSH' and recs[k-1]['visited'][-1]==x and past['start']>q]
        if not donors:continue
        donor=donors[0]['start'];key=r['start'];assert seq[key]==seq[donor]
        expected=donors[0]['visited'][-1];attn=model.blocks[4].attn;cap={}
        def grab(m,args,out):cap['qkv']=out.detach().clone()
        h=attn.qkv.register_forward_hook(grab)
        with torch.inference_mode():model(torch.tensor(seq[:donor+1])[None])
        donor_qkv=cap['qkv'][0,donor].view(3,cfg.model.d_model).clone()
        ids=torch.tensor(seq[:q+1])[None]
        with torch.inference_mode():base,weights=model(ids,return_attn=True)
        original=cap['qkv'][0].view(q+1,3,cfg.model.d_model).clone();h.remove()
        positions=torch.arange(q+1)[None]
        query=apply_rope(original[:,0][None,None],positions,attn.rope_base)[0,0,q]
        base_keys=apply_rope(original[:,1][None,None],positions,attn.rope_base)[0,0]
        old_w=(base_keys@query/math.sqrt(cfg.model.d_model)).softmax(-1)
        row_roundoff=float((old_w-weights[4][0,0,q]).abs().max());assert row_roundoff<2e-5
        old_out=old_w@original[:,2];del weights
        for kind in ['sham','K_only','V_only','K_and_V']:
            altered=original.clone()
            if kind in ['K_only','K_and_V']:altered[key,1]=donor_qkv[1]
            if kind in ['V_only','K_and_V']:altered[key,2]=donor_qkv[2]
            keys=apply_rope(altered[:,1][None,None],positions,attn.rope_base)[0,0]
            new_w=(keys@query/math.sqrt(cfg.model.d_model)).softmax(-1)
            delta=torch.nn.functional.linear(new_w@altered[:,2]-old_out,attn.proj.weight)
            def patch(m,args,out):
                z=out[0].clone();z[0,q]+=delta;return z,out[1]
            h=attn.register_forward_hook(patch)
            with torch.inference_mode():log,_=model(ids,return_attn=True)
            h.remove();earlier=float((log[:,:q]-base[:,:q]).abs().max());assert earlier==0
            if kind=='sham':assert torch.equal(base,log)
            pred=int(log[0,q].argmax())
            results.append(dict(index=pair['index'],condition=kind,q=q,key=key,donor=donor,
                baseline_prediction=tok.name(int(base[0,q].argmax())),prediction=tok.name(pred),
                expected_inserted_successor=tok.name(expected),follows_inserted_successor=pred==expected,
                old_key_attention=float(old_w[key]),new_key_attention=float(new_w[key]),
                earlier_logit_max_difference=earlier,delta_norm=float(delta.norm()),row_roundoff=row_roundoff,
                sham_logit_max_difference=float((log-base).abs().max()) if kind=='sham' else None))
        a.save('visited_successor_graft.json',dict(definition=__doc__,results=results,complete=False))
        print(json.dumps(results[-1]),flush=True)
    a.save('visited_successor_graft.json',dict(definition=__doc__,results=results,complete=True))

if __name__=='__main__':main()

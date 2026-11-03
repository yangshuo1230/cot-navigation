"""Exploratory same-visible-node register swaps during visited continuation."""
import json,torch,numpy as np
import analyze as a
def main():
    torch.set_num_threads(4);cfg=a.load_config([a.RUN/'resolved_config.yaml']);tok=a.Tokenizer.from_config(cfg)
    model=a.build_model(cfg.model,len(tok),tok.pad).eval();model.load_state_dict(torch.load(a.RUN/'checkpoints/best.pt',map_location='cpu',weights_only=False)['model']);model.requires_grad_(False)
    store=a.ContextStore(a.ROOT/'data/scale_v1'/a.SETS['n100']);plan=json.loads((a.OUT/'plan.json').read_text())['panels']['n100']['test'];results=[]
    for spec in [s for s in plan if s['band']=='long']:
        row=store.render(spec['index'],'dfs_state',tok);recs=a.parse(row.tokens,row.prompt_len,tok);ri=len(recs)//2;r=recs[ri]
        j=max(1,len(r['visited'])//2);p=r['a']+1+j;q=p-1;target=row.tokens[p];seq=row.tokens[:p];ids=torch.tensor(seq)[None]
        keys=[past['start'] for past in recs[:ri] if past['visited'][-1]==target]
        donor=recs[0]['start'];attn=model.blocks[4].attn;cap={}
        def grab(module,args,out):cap['v']=out.view(1,len(seq),3,1,cfg.model.d_model)[0,:,2,0,:].clone()
        h=attn.qkv.register_forward_hook(grab)
        with torch.inference_mode():base,w=model(ids,return_attn=True)
        h.remove();v=cap['v'];weights=w[4][0,0,q].clone();del w
        other=[past['start'] for past in recs[:ri] if past['start'] not in keys]
        control=sorted(other,key=lambda k:float(weights[k]))[:len(keys)]
        conditions=[]
        for kind,positions in [('swap_matched_discovery_roots',keys),('swap_low_weight_roots',control),('sham',keys)]:
            delta=torch.zeros(cfg.model.d_model)
            if kind!='sham':
                for key in positions:delta+=torch.nn.functional.linear((v[donor]-v[key])*weights[key],attn.proj.weight)
            def inject(module,args,out):
                z=out[0].clone();z[0,q]+=delta;return z,out[1]
            hh=attn.register_forward_hook(inject)
            with torch.inference_mode():changed,w=model(ids,return_attn=True)
            hh.remove();del w
            if kind=='sham':assert torch.equal(base,changed)
            assert float((base[:,:q]-changed[:,:q]).abs().max())==0
            conditions.append(dict(kind=kind,n_keys=len(positions),attention_mass=float(weights[positions].sum()),
               prediction=tok.name(int(changed[0,q].argmax())),changed_prediction=int(changed[0,q].argmax())!=int(base[0,q].argmax()),
               gold_logp_drop=float(base[0,q].log_softmax(-1)[target]-changed[0,q].log_softmax(-1)[target])))
        result=dict(index=spec['index'],record=ri+1,q=q,next=tok.name(target),baseline=tok.name(int(base[0,q].argmax())),baseline_correct=int(base[0,q].argmax())==target,
                    register_visible_token=tok.name(seq[donor]),keys=keys,donor=donor,conditions=conditions)
        results.append(result);a.save('visited_register_swaps.json',dict(definition='Exploratory: six long test queries, middle record, middle visited element; replace local L5 value contributions from historical stack-root positions whose reference visited[-1] is the target. Donor is first stack-root, identical visible node token. Q/K/weights fixed; no earlier logits change.',results=results));print(json.dumps(result),flush=True)
if __name__=='__main__':main()

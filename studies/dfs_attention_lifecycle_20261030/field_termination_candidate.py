"""Same candidate set and receiver identity, opposite boundary via permutation.
Also probe fixed stack/visited boundary under whole-field node replacement and
length edits. Malformed identity/length probes are diagnostics, not valid DFS.
"""
import json,torch
import analyze as a

def main():
    torch.set_num_threads(4);cfg=a.load_config([a.RUN/'resolved_config.yaml']);tok=a.Tokenizer.from_config(cfg)
    model=a.build_model(cfg.model,len(tok),tok.pad).eval()
    model.load_state_dict(torch.load(a.RUN/'checkpoints/best.pt',map_location='cpu',weights_only=False)['model']);model.requires_grad_(False)
    store=a.ContextStore(a.ROOT/'data/scale_v1'/a.SETS['n100']);panel=json.loads((a.OUT/'plan.json').read_text())['panels']['n100']['test']
    results=[];probes=[];shams=[]
    for spec in [s for s in panel if s['band']=='long']:
        row=store.render(spec['index'],'dfs_state',tok);seq=row.tokens[:-1];recs=a.parse(seq,row.prompt_len,tok)
        for field,lo,hi in [('stack','start','a'),('visited','a','b'),('candidates','b','c')]:
            possible=[(ri,r) for ri,r in enumerate(recs) if r[hi]-r[lo]-(field!='stack')>=2]
            ri,r=min(possible,key=lambda pair:(abs(pair[0]-len(recs)/2),pair[0]));start=r[lo]+(field!='stack');last=r[hi]-1
            for kind,edited in [('all_nodes_replaced',seq[:start]+[seq[start]]*(last-start+1)),
                                ('delete_penultimate',seq[:last-1]+seq[last:last+1]),
                                ('insert_duplicate_before_last',seq[:last]+[seq[last-1]]+seq[last:last+1])]:
                with torch.inference_mode():log=model(torch.tensor(edited)[None])
                pred=int(log[0,-1].argmax())
                probes.append(dict(index=spec['index'],field=field,record=ri+1,kind=kind,original_length=last-start+1,
                    prediction=tok.name(pred),predicts_stop=pred==tok.id('|')))
            if field!='candidates':continue
            original=seq[:last+1];altered=original[:];altered[start],altered[last]=altered[last],altered[start]
            conditions=[]
            for label,tokens,q in [('stop',original,last),('continue',altered,start)]:
                captured={};handles=[]
                for li,block in enumerate(model.blocks):
                    for component,module in [('attention',block.attn),('ffn',block.ffn)]:
                        def grab(m,args,out,li=li,component=component):captured[(li,component)]=(out[0] if component=='attention' else out).detach().clone()
                        handles.append(module.register_forward_hook(grab))
                with torch.inference_mode():base=model(torch.tensor(tokens[:q+1])[None])
                for h in handles:h.remove()
                conditions.append(dict(label=label,tokens=tokens[:q+1],q=q,capture=captured,base=base))
            assert conditions[0]['tokens'][-1]==conditions[1]['tokens'][-1]
            for receiver,donor in [conditions,conditions[::-1]]:
                q=receiver['q'];base=receiver['base'];ids=torch.tensor(receiver['tokens'])[None]
                for li in range(6):
                    for component in ['attention','ffn']:
                        module=model.blocks[li].attn if component=='attention' else model.blocks[li].ffn
                        replacement=donor['capture'][(li,component)][0,donor['q']]
                        def patch(m,args,out):
                            z=(out[0] if component=='attention' else out).clone();z[0,q]=replacement
                            return (z,out[1]) if component=='attention' else z
                        h=module.register_forward_hook(patch)
                        with torch.inference_mode():log=model(ids)
                        h.remove();earlier=float((base[0,:q]-log[0,:q]).abs().max());assert earlier==0
                        pred=int(log[0,q].argmax());original_pred=int(base[0,q].argmax())
                        results.append(dict(index=spec['index'],record=ri+1,layer=li+1,component=component,
                            direction=receiver['label']+'_to_'+donor['label'],visible=tok.name(ids[0,q].item()),
                            baseline_prediction=tok.name(original_pred),baseline_class_correct=(original_pred==tok.id('|'))==(receiver['label']=='stop'),
                            prediction=tok.name(pred),follows_donor_class=(pred==tok.id('|')) if donor['label']=='stop' else (tok.node_lo<=pred<tok.node_lo+tok.n_node_tokens),
                            earlier_logit_max_difference=earlier))
            # Exact identity on both decisive high attention-layer candidates.
            for li in [4,5]:
                receiver=conditions[0];q=receiver['q'];base=receiver['base'];module=model.blocks[li].attn
                def sham(m,args,out):
                    z=out[0].clone();z[0,q]=receiver['capture'][(li,'attention')][0,q];return z,out[1]
                h=module.register_forward_hook(sham)
                with torch.inference_mode():log=model(torch.tensor(receiver['tokens'])[None])
                h.remove();assert torch.equal(base,log);shams.append(dict(index=spec['index'],layer=li+1,identical=True))
        a.save('field_termination_candidate.json',dict(results=results,length_identity_probes=probes,shams=shams,complete=False))
        print(json.dumps(dict(index=spec['index'],results=len(results))),flush=True)
    a.save('field_termination_candidate.json',dict(results=results,length_identity_probes=probes,shams=shams,complete=True))

if __name__=='__main__':main()

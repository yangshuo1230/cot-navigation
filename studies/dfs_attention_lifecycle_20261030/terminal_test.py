"""Terminal-action -> final-answer tests, separate from search correctness."""
import json,torch
import analyze as a
def main():
    torch.set_num_threads(4);cfg=a.load_config([a.RUN/'resolved_config.yaml']);tok=a.Tokenizer.from_config(cfg)
    model=a.build_model(cfg.model,len(tok),tok.pad).eval();state=torch.load(a.RUN/'checkpoints/best.pt',map_location='cpu',weights_only=False);model.load_state_dict(state['model']);model.requires_grad_(False)
    store=a.ContextStore(a.ROOT/'data/scale_v1'/a.SETS['n100']);plan=json.loads((a.OUT/'plan.json').read_text())['panels']['n100']['test'];results=[]
    for spec in plan:
        row=store.render(spec['index'],'dfs_state',tok);tokens=row.tokens[:-2];q=len(tokens)-1;recs=a.parse(tokens,row.prompt_len,tok);key=recs[-1]['c']+1
        alternate=tok.id('<EXHAUST>') if spec['label'] else tok.id('<FOUND>');expected=tok.no if spec['label'] else tok.yes
        altered=tokens[:];altered[key]=alternate
        attn=model.blocks[0].attn;capture={}
        def grab(module,args,out):capture['v']=out.view(1,len(tokens),3,1,cfg.model.d_model)[0,:,2,0,:].clone()
        hook=attn.qkv.register_forward_hook(grab)
        with torch.inference_mode():base,w=model(torch.tensor(tokens)[None],return_attn=True)
        original_v=capture['v'];weight=float(w[0][0,0,q,key]);del w
        with torch.inference_mode():edited=model(torch.tensor(altered)[None])
        altered_v=capture['v'];hook.remove()
        delta=torch.nn.functional.linear((altered_v[key]-original_v[key])*weight,attn.proj.weight)
        def inject(module,args,out):
            x=out[0].clone();x[0,q]+=delta;return x,out[1]
        h=attn.register_forward_hook(inject)
        with torch.inference_mode():local,w=model(torch.tensor(tokens)[None],return_attn=True)
        h.remove();del w
        assert float((local[:,:q]-base[:,:q]).abs().max())==0
        result=dict(index=spec['index'],label=spec['label'],q=q,key=key,original_action=tok.name(tokens[key]),alternate_action=tok.name(alternate),
                    original_prediction=tok.name(int(base[0,q].argmax())),full_token_edit_prediction=tok.name(int(edited[0,q].argmax())),
                    local_L1_value_swap_prediction=tok.name(int(local[0,q].argmax())),expected_counterfactual=tok.name(expected),
                    full_edit_follows_alternate=int(edited[0,q].argmax())==expected,local_swap_follows_alternate=int(local[0,q].argmax())==expected,
                    terminal_action_attention_L1=weight,
                    logit_margin_change=float((local[0,q,expected]-local[0,q,tok.yes if spec['label'] else tok.no])-(base[0,q,expected]-base[0,q,tok.yes if spec['label'] else tok.no])))
        results.append(result);a.save('terminal_tests.json',dict(definition='Swap FOUND/EXHAUST at the terminal action as an explicitly inconsistent input diagnostic; compare full edit with local L1 action-value replacement into final semicolon, preserving original Q/K there. Not a legal alternate DFS trajectory.',results=results));print(json.dumps(result),flush=True)
if __name__=='__main__':main()

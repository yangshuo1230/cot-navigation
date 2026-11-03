"""Inspect the low-layer read that creates a discovery-register candidate."""
import json,torch,numpy as np
import analyze as a
def main():
    torch.set_num_threads(4);cfg=a.load_config([a.RUN/'resolved_config.yaml']);tok=a.Tokenizer.from_config(cfg)
    model=a.build_model(cfg.model,len(tok),tok.pad).eval();model.load_state_dict(torch.load(a.RUN/'checkpoints/best.pt',map_location='cpu',weights_only=False)['model']);model.requires_grad_(False)
    store=a.ContextStore(a.ROOT/'data/scale_v1'/a.SETS['n100']);plan=json.loads((a.OUT/'plan.json').read_text())['panels']['n100']['test'];rows=[]
    for spec in [s for s in plan if s['band']=='long']:
        row=store.render(spec['index'],'dfs_state',tok);seq=row.tokens[:-1];recs=a.parse(seq,row.prompt_len,tok)
        with torch.inference_mode():_,weights=model(torch.tensor(seq)[None],return_attn=True)
        for ri in range(1,len(recs)):
            prev,r=recs[ri-1],recs[ri]
            if prev['action']!='PUSH':continue
            key=prev['c']+2;q=r['start']
            rows.append(dict(index=spec['index'],record=ri+1,root=q,source_argument=key,root_visible_token=tok.name(seq[q]),
               introduced_node=tok.name(seq[key]),read_push_arg=[float(w[0,0,q,key]) for w in weights],
               last_visited_matches_argument=r['visited'][-1]==seq[key]))
    by={}
    for ix in sorted(set(r['index'] for r in rows)):
        by[ix]=np.mean([r['read_push_arg'] for r in rows if r['index']==ix],axis=0).tolist()
    a.save('writer_audit.json',dict(definition='After a PUSH, inspect the next record stack-root token attention to the previous PUSH argument. This is an observed write-route candidate, not proof of a unique unchanged representation across L2-L4.',n_queries=len(by),n_events=len(rows),query_balanced_mean=np.mean(list(by.values()),axis=0).tolist(),per_query=by,rows=rows))
    print(json.dumps(dict(n_queries=len(by),n_events=len(rows),query_balanced_mean=np.mean(list(by.values()),axis=0).tolist())),flush=True)
if __name__=='__main__':main()

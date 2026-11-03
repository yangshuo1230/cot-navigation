"""Distance-centered display statistics, preserving all original research data."""
import json,torch,numpy as np
from collections import defaultdict
import analyze as a
import summarize as s

def main():
    torch.set_num_threads(4);cfg=a.load_config([a.RUN/'resolved_config.yaml']);tok=a.Tokenizer.from_config(cfg);a.TOK=tok
    model=a.build_model(cfg.model,len(tok),tok.pad).eval();model.requires_grad_(False)
    model.load_state_dict(torch.load(a.RUN/'checkpoints/best.pt',map_location='cpu',weights_only=False)['model'])
    store=a.ContextStore(a.ROOT/'data/scale_v1'/a.SETS['n100']);panel=json.loads((a.OUT/'plan.json').read_text())['panels']['n100']['test'];events=[];baselines=[]
    for spec in panel:
        row=store.render(spec['index'],'dfs_state',tok);seq=row.tokens[:-1];n=len(seq);recs=a.parse(seq,row.prompt_len,tok);roles,rids=a.metadata(seq,row.prompt_len,recs,tok)
        with torch.inference_mode():_,ws=model(torch.tensor(seq)[None],return_attn=True)
        A=[w[0,0].numpy() for w in ws]
        mean=np.array([[mat.diagonal(-d).mean(dtype=np.float64) for d in range(n)] for mat in A])
        total=[float(sum((mat.diagonal(-d)-mean[li,d]).sum(dtype=np.float64) for d in range(n))) for li,mat in enumerate(A)]
        assert max(abs(x) for x in total)<.002
        for e in a.events(seq,row.prompt_len,recs):
            q=e['q'];groups=a.groupmask(roles,rids,e['ri'],q);routes=a.routes(seq,row.prompt_len,recs,e)
            residual=np.array([mat[q,:q+1] for mat in A])-mean[:,np.arange(q,-1,-1)]
            events.append(dict(**e,index=spec['index'],group_mass=[[float(z[groups==g].sum()) for g in range(len(a.GROUPS))] for z in residual],
                route_mass={name:residual[:,keys].sum(1).tolist() for name,keys in routes.items()}))
        baselines.append(dict(index=spec['index'],length=n,mean_by_layer_distance=mean.tolist(),count_by_distance=list(range(n,0,-1)),centered_total=total))
        print(json.dumps(dict(index=spec['index'],events=len(events))),flush=True)
        del A,ws
    def summary(es):
        names=sorted(set(name for e in es for name in e['route_mass']))
        return dict(groups=s.balanced(es,lambda e:e['group_mass']),routes={name:s.balanced(es,lambda e:e['route_mass'].get(name,[0]*6)) for name in names})
    phases=defaultdict(list);subs=defaultdict(list)
    for e in events:
        phases[e['phase']].append(e)
        if e['sub']:subs[e['phase']+'::'+e['sub']].append(e)
    a.save('statistics_distance_adjusted.json',dict(definition='For each sequence and attention layer/head separately, mu[d]=mean(A[q,k] for all causal q-k=d), including prompt and CoT. Display A[q,k]-mu[q-k]. Groups/routes sum these residuals BEFORE query averaging. No row renormalization. Bootstrap queries.',
        panels={'n100':{'test':{p:summary(es) for p,es in phases.items()}}},subphases={p:summary(es) for p,es in subs.items()},n_queries=18,n_events=len(events)))
    a.save('distance_baselines.json',dict(baselines=baselines))

if __name__=='__main__':main()

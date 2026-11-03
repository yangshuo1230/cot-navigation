"""Test discovery-register labels against all saved top-5 observations."""
import json,gzip,torch
from collections import defaultdict
import numpy as np
import analyze as a
def main():
    cfg=a.load_config([a.RUN/'resolved_config.yaml']);tok=a.Tokenizer.from_config(cfg)
    store=a.ContextStore(a.ROOT/'data/scale_v1'/a.SETS['n100']);cache={};rows=[]
    with gzip.open(a.OUT/'events.jsonl.gz','rt') as f:
        for line in f:
            e=json.loads(line)
            if e['tag']!='n100' or e['phase']!='visited_continue':continue
            ix=e['index']
            if ix not in cache:
                rr=store.render(ix,'dfs_state',tok);cache[ix]=(rr,a.parse(rr.tokens,rr.prompt_len,tok))
            rr,recs=cache[ix];target=tok.id(e['next']);inp=tok.id(e['input']);source=rr.tokens[rr.prompt_len-3]
            matching={r['start'] for r in recs[:e['ri']] if r['visited'][-1]==target}
            control={r['start'] for r in recs[:e['ri']] if r['visited'][-1]==inp}
            top=e['top_keys'][4]
            rows.append(dict(index=ix,split=e['split'],q=e['q'],record=e['ri']+1,target=e['next'],
                top5_matching_last_visited=sum(k['weight'] for k in top if k['position'] in matching),
                top5_previous_input_control=sum(k['weight'] for k in top if k['position'] in control),
                top5_all_history_stack_roots=sum(k['weight'] for k in top if k['role']=='stack' and k['token']==tok.name(source) and k['record']<=e['ri']),
                full_history_stack_mass=e['group_mass'][4][11],
                top5_coverage=sum(k['weight'] for k in top),matching_positions=sorted(matching)))
    summary={}
    for split in ['val','test']:
        selected=[r for r in rows if r['split']==split];by=defaultdict(list)
        for r in selected:by[r['index']].append(r)
        summary[split]=dict(n_queries=len(by),n_events=len(selected))
        for key in ['top5_matching_last_visited','top5_previous_input_control','top5_all_history_stack_roots','full_history_stack_mass','top5_coverage']:
            summary[split][key]=float(np.mean([np.mean([r[key] for r in rr]) for rr in by.values()]))
    a.save('carrier_audit.json',dict(definition='L5 top-5 lower bound: historical stack-root positions whose reference visited[-1] equals the next visited token. No access to future in activations; reference semantics used only to label past positions. Candidate rule proposed after inspecting discovery-val and confirmation-test routes; exploratory, not preregistered.',summary=summary,rows=rows))
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()

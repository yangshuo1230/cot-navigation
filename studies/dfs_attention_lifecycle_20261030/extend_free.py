"""Complete the middle/long test-band native panel; retain every outcome."""
import json,gzip,torch
import analyze as a
def main():
    torch.set_num_threads(4);a.CFG=a.load_config([a.RUN/'resolved_config.yaml']);a.TOK=a.Tokenizer.from_config(a.CFG)
    a.MODEL=a.build_model(a.CFG.model,len(a.TOK),a.TOK.pad).eval()
    state=torch.load(a.RUN/'checkpoints/best.pt',map_location='cpu',weights_only=False);a.MODEL.load_state_dict(state['model']);a.MODEL.requires_grad_(False)
    store=a.ContextStore(a.ROOT/'data/scale_v1'/a.SETS['n100']);plan=json.loads((a.OUT/'plan.json').read_text())['panels']['n100']['test']
    rolls=json.loads((a.OUT/'free_rollouts.json').read_text());done={r['index'] for r in rolls}
    jobs=[s for s in plan if s['band'] in ['middle','long'] and s['index'] not in done]
    a.save('free_extension_plan.json',dict(criteria='Every remaining middle/long reference-test-panel query, no performance selection; preserve original four outcomes.',indices=[s['index'] for s in jobs]))
    with gzip.open(a.OUT/'free_events.jsonl.gz','at',encoding='utf-8') as file:
        for spec in jobs:
            ix=spec['index'];row=store.render(ix,'dfs_state',a.TOK);P=row.prompt_len
            torch.set_num_threads(2)
            with torch.inference_mode():out=a.generate(a.MODEL,[row.tokens[:P]],'dfs_state',a.TOK,'fp32',cot_margin=a.CFG.protocol.cot_margin)[0]
            torch.set_num_threads(4)
            inverse={t:i for i,t in enumerate(store.node_vocab_ids(ix,a.TOK))};adj=store.adj(ix)
            src=int(store.arrays['s_topo'][ix]);dst=int(store.arrays['t_topo'][ix]);native=a.parse(out['tokens'],P,a.TOK);legal=0;end=P
            for r in native:
                test=a.replay_state(adj,src,dst,out['tokens'][P:r['end']],a.TOK,inverse)
                if test['records']!=legal+1 or (not test['valid'] and test['error']!='missing_terminal'):break
                legal+=1;end=r['end']
            audit=a.replay_state(adj,src,dst,out['trace_tokens'],a.TOK,inverse)
            result=dict(index=ix,generated=len(out['tokens'])-P,valid_records=legal,full_valid=audit['valid'],semantic_error=audit['error'],
                        correct=out['pred']==spec['label'],pred=out['pred'],label=spec['label'],interrupted=out.get('cot_interrupted'),band=spec['band'],tokens=out['tokens'],legal_end=end)
            if legal:
                seq=out['tokens'] if audit['valid'] else out['tokens'][:end]
                es,_=a.analyze_one(store,'n100','test',spec,'free_legal_prefix',seq)
                for e in es:file.write(json.dumps(e,separators=(',',':'))+'\n')
                file.flush()
            rolls.append(result);a.save('free_rollouts.json',rolls)
            print('FREE',json.dumps({k:v for k,v in result.items() if k!='tokens'}),flush=True)
    a.save('COMPLETE.json',dict(reference_queries=52,free_queries=len(rolls),finished=True))
if __name__=='__main__':main()

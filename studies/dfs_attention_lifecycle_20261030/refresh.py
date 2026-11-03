"""Recompute added semantic routes on the exact frozen panel; reuse free rolls."""
import json,gzip,torch
import analyze as a
def main():
    torch.set_num_threads(4)
    a.CFG=a.load_config([a.RUN/'resolved_config.yaml']);a.TOK=a.Tokenizer.from_config(a.CFG)
    a.MODEL=a.build_model(a.CFG.model,len(a.TOK),a.TOK.pad).eval()
    state=torch.load(a.RUN/'checkpoints/best.pt',map_location='cpu',weights_only=False)
    a.MODEL.load_state_dict(state['model']);a.MODEL.requires_grad_(False)
    stores={tag:a.ContextStore(a.ROOT/'data/scale_v1'/name) for tag,name in a.SETS.items()}
    plan=json.loads((a.OUT/'plan.json').read_text())['panels']
    with gzip.open(a.OUT/'events.jsonl.gz','wt',encoding='utf-8') as file:
        for tag,parts in plan.items():
            for split,specs in parts.items():
                for spec in specs:
                    es,_=a.analyze_one(stores[tag],tag,split,spec)
                    for e in es:file.write(json.dumps(e,separators=(',',':'))+'\n')
                    file.flush()
    rolls=json.loads((a.OUT/'free_rollouts.json').read_text())
    with gzip.open(a.OUT/'free_events.jsonl.gz','wt',encoding='utf-8') as file:
        for roll in rolls:
            if not roll['valid_records']:continue
            spec=next(s for s in plan['n100']['test'] if s['index']==roll['index'])
            tokens=roll['tokens'] if roll['full_valid'] else roll['tokens'][:roll['legal_end']]
            es,_=a.analyze_one(stores['n100'],'n100','test',spec,'free_legal_prefix',tokens)
            for e in es:file.write(json.dumps(e,separators=(',',':'))+'\n')
if __name__=='__main__':main()

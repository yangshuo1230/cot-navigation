"""Score legal next content separately from node/bar class and canonical order."""
from experiment import *

def main():
 cfg=a.load_config([a.RUN/'resolved_config.yaml']);tok=a.Tokenizer.from_config(cfg);store=a.ContextStore(a.ROOT/'data/scale_v1'/a.SETS['n100'])
 for prefix in ['', 'stack_tail.','candidate_empty.']:
  rows=json.loads((OUT/(prefix+'results.json')).read_text());pairs=json.loads((OUT/(prefix+'pairs.json')).read_text());allowed={}
  for pair in pairs:
   if not pair['available']:continue
   ix=pair['index'];field=pair['field'];data=store.render(ix,'dfs_state',tok);seq=data.tokens[:-1];recs=a.parse(seq,data.prompt_len,tok)
   if field=='candidates':
    r=recs[pair['records'][1]-1];first=r['b']+1;last=r['c']-1;edited=seq[:last+1];edited[first],edited[last]=edited[last],edited[first];nodes=edited[first+1:]
   elif field=='candidates_empty':nodes=recs[pair['records'][1]-1]['candidates']
   else:nodes=[seq[pair['continue_q']+1]]
   allowed[ix,field]=[tok.name(t) for t in nodes]
  for r in rows:
   continuation=allowed[r['index'],r['field']];isstop=r['direction']=='stop_to_continue';original=['|'] if isstop else continuation;donor=continuation if isstop else ['|'];r['allowed_original_next']=original;r['allowed_donor_next']=donor;r['legal_original_next']=r['prediction'] in original;r['legal_donor_next']=r['prediction'] in donor;r['baseline']['legal_original_next']=r['baseline']['prediction'] in original
  (OUT/(prefix+'results.json')).write_text(json.dumps(rows,ensure_ascii=False));print(json.dumps(dict(file=prefix+'results.json',conditions=len(rows))))
if __name__=='__main__':main()

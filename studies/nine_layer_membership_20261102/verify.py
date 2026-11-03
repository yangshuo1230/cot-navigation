from common import *
PREFIX='route_aligned.' if '--route-aligned' in sys.argv else ''

def main():
 rows=json.loads((OUT/(PREFIX+'interventions.json')).read_text());store,_=store_plan();binding=bindings()[0];cfg=a.load_config([Path(binding['run'])/'resolved_config.yaml']);tok=a.Tokenizer.from_config(cfg);a.TOK=tok;bad=[];graph_audits=[]
 for r in rows:
  ix=r['index'];row=store.render(ix,'dfs_state',tok);seq=row.tokens[:-1];P=row.prompt_len;recs=a.parse(seq,P,tok);record=recs[r['record']-1];prefix=seq[:record['b']+1];original_hash=hashlib.sha256(np.asarray(prefix,dtype=np.int64).tobytes()).hexdigest()
  assert original_hash==r['original_prefix_sha256']
  if r['arm']=='push_residual':assert r['input_sha256']==original_hash and not r['edited_positions']
  if r['arm']=='visited_only':assert all(record['a']<p<record['b'] for p in r['edited_positions']) and 'membership_shift' not in r
  if r['arm']=='graph_only':
   assert len(r['edited_positions'])==1 and r['edited_positions']==[r['edge_position']]
   edited=prefix[:];edited[r['edge_position']]=tok.id(r['alternate']);assert hashlib.sha256(np.asarray(edited,dtype=np.int64).tobytes()).hexdigest()==r['input_sha256'];ids=[int(v) for v in store.node_vocab_ids(ix,tok)];inv={v:i for i,v in enumerate(ids)};edge=np.asarray(edited[:P-4]).reshape(-1,3);dag=all(inv[int(u)]<inv[int(v)] for _,u,v in edge);adj=[[] for _ in ids]
   for _,u,v in edge:adj[inv[int(u)]].append(inv[int(v)])
   audit=a.replay_state(adj,int(store.arrays['s_topo'][ix]),int(store.arrays['t_topo'][ix]),edited[P:recs[r['record']-2]['end']],tok,inv);legal=audit['error']=='missing_terminal' and audit['records']==r['record']-1
   graph_audits.append(dict(model=r['model'],index=ix,condition=r['condition'],dag=dag,previous_records_legal=legal))
   if not dag or not legal:bad.append(graph_audits[-1])
 baseline={(r['model'],r['index']):r for r in rows if r['condition']=='baseline'}
 assert all(r['output']==baseline[r['model'],r['index']]['output'] for r in rows if r['condition']=='push_sham')
 result=dict(conditions=len(rows),model_state_pairs=len(baseline),push_conditions=sum(r['arm']=='push_residual' for r in rows),visited_conditions=sum(r['arm']=='visited_only' for r in rows),graph_conditions=len(graph_audits),independence_pass=True,sham_pass=True,graph_audits=graph_audits,graph_failures=bad)
 save(PREFIX+'validation',result);print(json.dumps(result))
if __name__=='__main__':main()

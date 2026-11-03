"""Separate pristine-input arms: graph, current visited, and PUSH residual."""
from common import *
ROUTE_ALIGNED='--route-aligned' in sys.argv
_save=save
def save(name,value):_save(('route_aligned.' if ROUTE_ALIGNED else '')+name,value)

def digest(x):return hashlib.sha256(np.asarray(x,dtype=np.int64).tobytes()).hexdigest()

def main():
 store,plan=store_plan();selected=json.loads((OUT/'selected_probe.json').read_text());results=[];selection=[]
 if ROUTE_ALIGNED:
  curves=json.loads((OUT/'probe_results.json').read_text())
  for name,layer in [('corr9_s0',8),('corr9_s1',5)]:
   selected[name]=next(r for r in curves[name]['curves'] if r['site']=='argument' and r['layer']==layer)
 for binding in bindings():
  cfg,tok,model=load(binding);name=binding['name'];specification=selected[name];layer=specification['layer'];site=specification['site'];B=np.load(OUT/(name+'.probe_weights.npz'))[site+'_L'+str(layer)]
  for spec in plan['test']:
   ix=spec['index'];row=store.render(ix,'dfs_state',tok);seq=row.tokens[:-1];P=row.prompt_len;recs=a.parse(seq,P,tok);edge=np.array(seq[:P-4]).reshape(-1,3);ids=[int(v) for v in store.node_vocab_ids(ix,tok)];inv={v:i for i,v in enumerate(ids)};choices=[]
   for ri,r in enumerate(recs):
    u=r['stack'][-1]
    if ri==0 or any(p['stack'][-1]==u for p in recs[:ri]) or r['action']=='FOUND':continue
    neighbors=set(edge[edge[:,1]==u,2].tolist())
    for v in sorted(neighbors&set(r['visited'])):
     if v in [seq[P-3],seq[P-2]]:continue
     ws=[w for w in ids if w not in neighbors|set(r['visited'])|{seq[P-3],seq[P-2],u} and inv[w]//10==inv[v]//10]
     if ws:choices.append((ri,r,v,ws[0],neighbors))
   if not choices:continue
   ri,r,v,w,neighbors=min(choices,key=lambda x:(abs(x[0]-len(recs)/2),x[0],x[2]));u=r['stack'][-1];gold=set(r['candidates']);prefix=seq[:r['b']+1];original_hash=digest(prefix);prior=recs[ri-1];assert prior['action']=='PUSH' and prior['argument']==u
   pos=prior['c']+(1 if site=='action' else 2);captured={}
   def capture(m,args,out):captured['h']=out[0][0,pos].detach().numpy().astype(float).copy()
   h=model.blocks[layer-1].register_forward_hook(capture)
   try:
    with torch.inference_mode():model(torch.tensor(prefix)[None])
   finally:h.remove()
   x=captured['h'];scores=np.r_[x,1.]@B;identity=scores[257:];uid=u-tok.node_lo;vid=v-tok.node_lo;competitor=max((j for j in range(256) if j!=uid),key=lambda j:identity[j]);C=np.stack([B[:-1,vid],B[:-1,256],B[:-1,257+uid]-B[:-1,257+competitor]])
   direction=C.T@np.linalg.pinv(C@C.T)@np.array([1.,0.,0.]);rng=np.random.default_rng(20261102+ix);random=rng.normal(size=128);random-=C.T@np.linalg.pinv(C@C.T)@(C@random);random/=np.linalg.norm(random)
   common=dict(model=name,index=ix,record=ri+1,cur=tok.name(u),excluded=tok.name(v),alternate=tok.name(w),original_prefix_sha256=original_hash,push_position=pos,push_site=site,push_layer=layer,probe_membership=float(scores[vid]),probe_threshold=specification['threshold'],probe_identity_correct=int(identity.argmax())==uid)
   selection.append(common)
   baseline_logits={}
   def run(condition,arm,edited,expected,delta=None,**extra):
    if ROUTE_ALIGNED and arm in ['visited_only','graph_only']:return
    assert not model.blocks[layer-1]._forward_hooks
    if arm=='push_residual':assert edited==prefix
    hook=None
    if delta is not None:
     dt=torch.tensor(delta,dtype=torch.float32)
     def patch(m,args,out):
      z=out[0].clone();z[0,pos]+=dt;return (z,)+out[1:]
     hook=model.blocks[layer-1].register_forward_hook(patch)
    observed={}
    def observe(m,args,out):
     if 'logits' not in observed:observed['logits']=out[0,-1].detach().numpy().astype(float).copy()
    observer=model.register_forward_hook(observe)
    try:out=generate(model,tok,edited)
    finally:
     observer.remove()
     if hook is not None:hook.remove()
    assert not model.blocks[layer-1]._forward_hooks
    nodes=out[:-1] if out[-1]==tok.id('|') else out;metrics=evaluate(out,expected,tok)
    result=dict(**common,condition=condition,arm=arm,input_sha256=digest(edited),edited_positions=[i for i,(a,b) in enumerate(zip(prefix,edited)) if a!=b],output=[tok.name(t) for t in out],expected=[tok.name(t) for t in sorted(expected)],expected_correct=metrics['set_correct'],original_correct=evaluate(out,gold,tok)['set_correct'],excluded_returned=v in nodes,alternate_returned=w in nodes,STOP=metrics['STOP'],**extra)
    if condition=='baseline':baseline_logits['value']=observed['logits']
    z=observed['logits'];base=baseline_logits['value'];prob=np.exp(z-z.max());prob/=prob.sum();bp=np.exp(base-base.max());bp/=bp.sum();result.update(first_query_logit_max_change=float(np.max(np.abs(z-base))),first_query_kl=float(np.sum(bp*(np.log(np.maximum(bp,1e-300))-np.log(np.maximum(prob,1e-300))))),first_query_excluded_probability=float(prob[v]),first_query_excluded_probability_change=float(prob[v]-bp[v]))
    if delta is not None:
     after=np.r_[x+delta,1.]@B;result.update(relative_norm=float(np.linalg.norm(delta)/np.linalg.norm(x)),membership_shift=float(after[vid]-scores[vid]),count_shift=float(after[256]-scores[256]),identity_margin_shift=float((after[257+uid]-after[257+competitor])-(identity[uid]-identity[competitor])),identity_after_correct=int(after[257:].argmax())==uid)
    results.append(result);save('interventions',results);print(json.dumps(dict(model=name,index=ix,condition=condition,output=result['output'],correct=result['expected_correct'])),flush=True)
   run('baseline','baseline',prefix,gold)
   # Visited edits are confined to this record; graph and all historical PUSHes stay original.
   edited=prefix[:]
   for p in range(r['a']+1,r['b']):
    if edited[p]==v:edited[p]=w
   run('visited_remove_excluded','visited_only',edited,gold|{v})
   if gold:
    target=min(gold);edited=prefix[:];p=next(p for p in range(r['a']+1,r['b']) if prefix[p] not in neighbors and prefix[p] not in [u,seq[P-3]])
    edited[p]=target;run('visited_add_eligible','visited_only',edited,gold-{target},added=tok.name(target))
   # Changing an outgoing edge of this never-before-expanded node preserves earlier records.
   edited=prefix[:];ep=next(3*i+2 for i,e in enumerate(edge) if int(e[1])==u and int(e[2])==v);edited[ep]=w
   run('edge_current_source','graph_only',edited,gold|{w},edge_position=ep)
   expanded={p['stack'][-1] for p in recs[:ri]};wrong=[i for i,e in enumerate(edge) if int(e[1]) not in expanded|{u} and int(e[2])==v]
   if wrong:
    wp=3*min(wrong,key=lambda j:abs(3*j+2-ep))+2;edited=prefix[:];edited[wp]=w;run('edge_wrong_source_same_destination','graph_only',edited,gold,edge_position=wp)
   # A separate untouched input is used for every residual intervention.
   run('push_sham','push_residual',prefix,gold,np.zeros(128))
   for dose in [.25,.5]:
    delta=direction*dose;scale=min(1.,.25*np.linalg.norm(x)/max(np.linalg.norm(delta),1e-12));delta*=scale
    run('push_remove_'+str(dose),'push_residual',prefix,gold|{v},-delta,requested_shift=-dose)
    run('push_increase_'+str(dose),'push_residual',prefix,gold,delta,requested_shift=dose)
    run('push_random_'+str(dose),'push_residual',prefix,gold,random*np.linalg.norm(delta),requested_shift=0.)
   # Stronger, explicitly off-distribution stress tests: identical residual norm
   # across the membership and random directions. Zero readout directions are skipped.
   if np.linalg.norm(direction)>1e-10:
    for norm in [.1,.25]:
     delta=direction/np.linalg.norm(direction)*np.linalg.norm(x)*norm
     run('push_remove_norm_'+str(norm),'push_residual',prefix,gold|{v},-delta,requested_relative_norm=norm)
     run('push_increase_norm_'+str(norm),'push_residual',prefix,gold,delta,requested_relative_norm=norm)
     run('push_random_norm_'+str(norm),'push_residual',prefix,gold,random*np.linalg.norm(delta),requested_relative_norm=norm)
  save('selection',selection)
 save('INTERVENTIONS_COMPLETE',dict(models=2,conditions=len(results),selection=len(selection),independence='All PUSH arms use original input hashes; no visited edit and activation edit coexist; hooks removed after each condition.'))
if __name__=='__main__':main()

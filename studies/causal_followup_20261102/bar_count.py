"""Virtual separator-count changes in a single L1 attention row.
Input syntax and all historical hidden states stay unchanged in virtual tests.
Also compare real token edits and count-preserving relocations.
"""
from common import *
from cot_navigation.models.attention import apply_rope
import math
def main():
 cfg,tok,model,store,panel=setup();probes=np.load(PREV/'probe_weights.npz');results=[]
 for spec in [s for s in panel if s['band']=='long']:
  row=store.render(spec['index'],'dfs_state',tok);seq=row.tokens[:-1];P=row.prompt_len;recs=a.parse(seq,P,tok);options=[(i,r) for i,r in enumerate(recs) if i>=2 and len(r['stack'])>=4];ri,r=min(options,key=lambda x:abs(x[0]-len(recs)/2));bar=tok.id('|');near=list(range(r['start']+1,r['a']))[:3];far_end=recs[ri-1]['start'];far_nodes=[p for p in range(P,far_end) if tok.node_lo<=seq[p]<tok.node_lo+tok.n_node_tokens][:3];far_bars=[p for p in range(P,far_end) if seq[p]==bar][-3:];assert len(near)==len(far_nodes)==len(far_bars)==3
  for cls,keyname in enumerate(['a','b','c']):
   q=r[keyname];ids=torch.tensor(seq[:q+1])[None];cap={};hooks=[]
   def grab_qkv(m,args,out):cap['qkv']=out[0].view(q+1,3,cfg.model.d_model).clone()
   hooks.append(model.blocks[0].attn.qkv.register_forward_hook(grab_qkv))
   for layer in range(1,7):
    def grab(m,args,out,layer=layer):cap[layer]=out[0][0,q].clone()
    hooks.append(model.blocks[layer-1].register_forward_hook(grab))
   with torch.inference_mode():baseline,weights=model(ids,return_attn=True)
   for h in hooks:h.remove()
   original=cap['qkv'];attn=model.blocks[0].attn;positions=torch.arange(q+1)[None];query=apply_rope(original[:,0][None,None],positions,attn.rope_base)[0,0,q];orig_keys=apply_rope(original[:,1][None,None],positions,attn.rope_base)[0,0];w0=(orig_keys@query/math.sqrt(cfg.model.d_model)).softmax(-1);oldout=w0@original[:,2];roundoff=float((w0-weights[0][0,0,q]).abs().max());assert roundoff<2e-5
   jobs=[('sham',{},'virtual')]
   for k in [1,2,3]:
    jobs += [('add_current_stack_'+str(k),{p:P-1 for p in near[:k]},'virtual'),('add_far_'+str(k),{p:P-1 for p in far_nodes[:k]},'virtual'),('remove_far_'+str(k),{p:r['start'] for p in far_bars[:k]},'virtual'),('real_add_current_stack_'+str(k),{p:P-1 for p in near[:k]},'input'),('real_node_control_'+str(k),{p:r['start'] for p in near[:k]},'input')]
   for k in [1,3]:jobs.append(('preserve_count_relocate_'+str(k),dict([(p,P-1) for p in near[:k]]+[(p,r['start']) for p in far_bars[:k]]),'virtual'))
   for condition,changes,mode in jobs:
    modified_tokens=seq[:q+1];altered=original.clone()
    for p,donor in changes.items():
     modified_tokens[p]=seq[donor];altered[p,1:]=original[donor,1:]
    count_before=sum(t==bar for t in seq[:q]);count_after=sum(t==bar for t in modified_tokens[:q]);count_delta=count_after-count_before;expected=(cls+count_delta)%3
    observed={};handles=[]
    if mode=='virtual':
     keys=apply_rope(altered[:,1][None,None],positions,attn.rope_base)[0,0];w1=(keys@query/math.sqrt(cfg.model.d_model)).softmax(-1);delta=torch.nn.functional.linear(w1@altered[:,2]-oldout,attn.proj.weight)
     def patch(m,args,out):
      x=out[0].clone();x[0,q]+=delta;return x,out[1]
     handles.append(attn.register_forward_hook(patch));input_ids=ids
    else:delta=None;input_ids=torch.tensor(modified_tokens)[None]
    for layer in range(1,7):
     def grab(m,args,out,layer=layer):observed[layer]=out[0][0,q].clone()
     handles.append(model.blocks[layer-1].register_forward_hook(grab))
    with torch.inference_mode():log,_=model(input_ids,return_attn=True)
    for h in handles:h.remove()
    if mode=='virtual':assert torch.equal(log[:,:q],baseline[:,:q])
    if condition=='sham':assert torch.equal(log,baseline)
    bp={};ap={}
    for layer in range(1,7):
     b=probes['bars_L'+str(layer)];bp[str(layer)]=int((np.r_[cap[layer].numpy(),1]@b).argmax());ap[str(layer)]=int((np.r_[observed[layer].numpy(),1]@b).argmax())
    before_distance=q-max(p for p in range(q) if seq[p]==bar);after_distance=q-max(p for p in range(q) if modified_tokens[p]==bar)
    results.append(dict(index=spec['index'],record=ri+1,field=cls,q=q,condition=condition,mode=mode,count_delta=count_delta,expected_global_count_role=expected,nearest_bar_distance_before=before_distance,nearest_bar_distance_after=after_distance,probe_before=bp,probe_after=ap,baseline=tok.name(int(baseline[0,q].argmax())),prediction=tok.name(int(log[0,q].argmax())),output_changed=int(log[0,q].argmax())!=int(baseline[0,q].argmax()),delta_norm=None if delta is None else float(delta.norm()),roundoff=roundoff,earlier_logits_unchanged=True if mode=='virtual' else None))
   save('bar_count',results)
if __name__=='__main__':main()

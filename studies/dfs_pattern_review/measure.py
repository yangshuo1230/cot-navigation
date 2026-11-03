"""Audit concrete DFS token routes in FP32, without changing model outputs.

Every event distinguishes INPUT receiver q from predicted token q+1. Exact
distance baselines use independent validation queries, separately per route.
Only small aggregate tables and illustrative observed arrows are persisted.
"""
from pathlib import Path
import os,sys,json,hashlib,math
from collections import defaultdict,Counter
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
ROOT=Path(str(next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'pyproject.toml').is_file() and (_p / 'studies').is_dir())));OUT=ROOT/'studies/dfs_pattern_review'
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
import plot_dfs_cot_attention as base
from cot_navigation.analysis.common import records
from cot_navigation.models.attention import apply_rope
from cot_navigation.evaluation.sampling import stratified_ids
from cot_navigation.data.dataset import ContextStore,ParametricWorld

SETTINGS={
 'context_n16':(base.CONTEXT_RUN,base.CONTEXT_DATA),
 'context_n100':(base.CONTEXT_RUN,ROOT/'data/scale_v1/context_l10w10_n100_m150_train50000_s0'),
 'fixed_scale_n100':(base.FIXED_RUN,base.FIXED_DATA),
 'fixed_simple_n100':(ROOT/'outputs/fixed_short3_v1/launches/short3_n100_s0/all_hops/dfs_state',ROOT/'data/fixed_short3_v1/n100_m150_s0/main-seed0'),
 'fixed_hard_n100':(ROOT/'outputs/hard_dag_v1/launches/hard_dag_v1_s0/fixed/n100/stage_b/dfs_state',ROOT/'data/hard_dag_v1/fixed_r10b10_n100_m200_s0/main-seed0')}
LIMIT=4096

def events(item,tok):
    t=item['tokens'];P=item['prompt_len'];recs=records(t,P,tok);es=[]
    for r in recs:r['stack']=r['frontier']
    context=t[0]==tok.id('<EDGE>');edges=np.asarray(t[:P-4]).reshape(-1,3) if context else np.empty((0,3),int)
    def add(name,q,keys,ri=-1,description=''):
        keys=list(dict.fromkeys(int(k) for k in keys if 0<=k<=q))
        if keys and 0<=q<min(LIMIT,len(t)-2):
            es.append(dict(name=name,q=int(q),keys=keys,record=ri,description=description))
    add('prompt_source',P-1,[P-3])
    if context:
        for ei,(_,u,v) in enumerate(edges):
            add('edge_dest_shift3',3*ei+5,[3*ei+2])
            add('edge_source_shift2',3*ei+3,[3*ei+1])
    seen=set()
    for ri,r in enumerate(recs):
        a,b,c=r['separators'];start=r['start'];actpos=c+1;action=tok.name(t[actpos]);cur=t[a-1]
        if start>=LIMIT:break
        prev=recs[ri-1] if ri else None;vis=list(range(a+1,b));cand=list(range(b+1,c))
        if ri:
            add('stack_root_prompt_bar',start-1,[P-1],ri)
            add('stack_root_query_source',start-1,[P-3],ri)
            add('stack_root_history_source',start-1,[k for k in range(P,start) if t[k]==t[P-3]],ri)
        else:add('first_stack_input_source',start,[P-3],ri)
        add('visited_first_prompt_bar',a,[P-1],ri)
        add('visited_first_stack_root',a,[start],ri)
        add('visited_first_query_source',a,[P-3],ri)
        add('visited_first_history_source',a,[k for k in range(P,start) if t[k]==t[P-3]],ri)
        if len(r['stack'])>=2:add('visited_first_second_stack',a+1,[a-2],ri)
        for j,pos in enumerate(vis[1:],1):
            sameprev=[k for k in range(prev['separators'][0]+1,prev['separators'][1]) if t[k]==t[pos]] if prev else []
            oldvis=[k for past in recs[:ri] for k in range(past['separators'][0]+1,past['separators'][1]) if t[k]==t[pos]]
            add('visited_next_previous_same',pos-1,sameprev,ri)
            add('visited_next_history_same',pos-1,oldvis,ri)
            add('visited_next_current_source',pos-1,[a+1],ri)
            add('visited_next_earlier_sources',pos-1,[past['separators'][0]+1 for past in recs[:ri]],ri)
            add('visited_next_history_source_any',pos-1,[k for k in range(P,start) if t[k]==t[P-3]],ri)
            add('visited_next_current_stack_source',pos-1,[start],ri)
            add('visited_next_self',pos-1,[pos-1],ri)
        if prev and prev['action'][0]==tok.id('<PUSH>') and len(r['stack'])==len(prev['stack'])+1:
            newtop=a-1;argpos=prev['separators'][2]+2
            if t[newtop]==t[argpos]:
                add('stack_new_top_push_arg',newtop-1,[argpos],ri)
                add('stack_new_top_INPUT_push_arg',newtop,[argpos],ri)
        if prev and prev['action'][0]==tok.id('<POP>'):
            for pos in range(start+1,a):
                matching=[past['separators'][0]+1 for past in recs[:ri] if len(past['stack'])>=2 and past['stack'][-2]==t[pos]]
                add('stack_kept_predict_matching_carrier',pos-1,matching,ri)
                add('stack_kept_INPUT_matching_carrier',pos,matching,ri)
                add('stack_kept_history_same',pos-1,[k for k in range(P,start) if t[k]==t[pos]],ri)
                prev_stack=list(range(prev['start'],prev['separators'][0]))
                add('stack_kept_previous_stack_same',pos-1,[k for k in prev_stack if t[k]==t[pos]],ri)
        if cand:
            typ='candidate_fresh' if cur not in seen else 'candidate_repeat'
            current_vis=[k for k in vis if t[k]==cur]
            rel=np.flatnonzero(edges[:,1]==cur) if context else []
            pastpush=[past for past in recs[:ri] if past['stack'][-1]==cur and past['action'][0]==tok.id('<PUSH>')]
            for bucket in [typ,'candidate_first_record' if ri==0 else 'candidate_later_record']:
                add(bucket+'_current_visited',b,current_vis,ri)
                add(bucket+'_stack_top',b,[a-1],ri)
                add(bucket+'_raw_edge_dest',b,[3*int(e)+2 for e in rel],ri)
                add(bucket+'_shifted_edge_dest',b,[3*int(e)+5 for e in rel],ri)
                if pastpush:
                    old=pastpush[-1]['separators'][2]+1
                    add(bucket+'_previous_push_pair',b,[old,old+1],ri)
                    add(bucket+'_previous_push_token',b,[old],ri)
                    add(bucket+'_previous_push_arg',b,[old+1],ri)
        if action=='<PUSH>':
            add('push_arg_candidates',actpos,cand,ri)
            add('push_arg_chosen',actpos,[k for k in cand if t[k]==r['action'][1]],ri)
            add('push_arg_first',actpos,cand[:1],ri)
            add('push_arg_last',actpos,cand[-1:],ri)
            if len(cand)==3:
                for j,k in enumerate(cand):add('push_three_'+str(j+1),actpos,[k],ri)
        if action in ('<POP>','<FOUND>'):
            add(action[1:-1].lower()+'_stack_top',c,[a-1],ri)
            add(action[1:-1].lower()+'_INPUT_stack_top',actpos,[a-1],ri)
        seen.add(cur)
    # Deterministic cap prevents one long query from dominating a route.
    counts=Counter();chosen=[]
    for e in es:
        if counts[e['name']]<8:chosen.append(e);counts[e['name']]+=1
    return chosen

class Reader:
    def __init__(self,model):
        self.model=model;self.handles=[];self.qs=[];self.weights={};self.active=True;self.checked=False
        for li,block in enumerate(model.blocks):
            self.handles.append(block.attn.qkv.register_forward_hook(self.hook(li,block.attn)))
    def hook(self,li,attn):
        def hook(module,args,out):
            if not self.active:return
            B,L,_=out.shape;H=attn.n_heads;D=attn.head_dim
            q,k,v=out.view(B,L,3,H,D).unbind(2)
            pos=torch.arange(L,device=out.device)[None]
            q=apply_rope(q.transpose(1,2),pos,attn.rope_base);k=apply_rope(k.transpose(1,2),pos,attn.rope_base)
            indices=torch.tensor(self.qs,device=out.device)
            score=q[:,:,indices,:]@k.transpose(-1,-2)/math.sqrt(D)
            keep=torch.arange(L,device=out.device)[None,:]<=indices[:,None]
            self.weights[li]=torch.softmax(score.masked_fill(~keep[None,None],-torch.inf),-1)[0,0].detach().cpu().numpy()
        return hook
    @torch.inference_mode()
    def forward(self,item,es):
        self.qs=sorted({e['q'] for e in es});self.weights={}
        ids=torch.tensor(item['tokens'][:max(self.qs)+1],device='cuda')[None]
        if not self.checked:
            self.active=False;original=self.model(ids);self.active=True
        logits=self.model(ids)
        if not self.checked:assert torch.equal(original,logits);self.checked=True
        assert all(np.max(np.abs(a.sum(-1)-1))<2e-6 for a in self.weights.values())
        return self.weights,{q:i for i,q in enumerate(self.qs)},logits[0].detach().cpu().float().numpy()
    def close(self):
        for h in self.handles:h.remove()

def main(name,folder,data):
    dest=OUT/f'{name}.json'
    if dest.exists() and json.loads(dest.read_text()).get('version')==3:return
    cfg,tok,model,step=base.load_model(folder,torch.device('cuda:0'));model.requires_grad_(False)
    context=name.startswith('context');source=ContextStore(data) if context else ParametricWorld(data)
    worldid=base.world_token(cfg,tok,data) if not context and cfg.train.world_roots else None
    arrays=source.arrays if context else source.queries
    def item(i):
        if context:return base.prepare_context(source,tok,int(i))
        row=source.render_query(int(i),'dfs_state',tok);prefix=[] if worldid is None else [worldid]
        return dict(tokens=prefix+row.tokens,prompt_len=row.prompt_len+len(prefix),node_tokens=source.node_vocab_ids(tok),index=int(i))
    chosen={}
    for split in ('val','test'):
        ids=source.indices(split)
        if not context:ids=np.array([i for i in ids if arrays['t_topo'][i]//10>arrays['s_topo'][i]//10])
        shapes=arrays['shape_id'] if context else np.zeros(len(arrays['label']),np.int8)
        chosen[split]=list(map(int,stratified_ids(ids,shapes,arrays['distance'],arrays['label'],20,20261026)))
    # Original pictured queries are audited separately, not silently counted
    # as part of the fixed diagnostic sample.
    examples_ids=[50520,50512] if name=='context_n16' else []
    reader=Reader(model);sums={};counts={};raw=defaultdict(list);corrected=defaultdict(list);examples={};special={}
    for split,ids in [('val',chosen['val']),('test',chosen['test']),('pictured',examples_ids)]:
        for i in ids:
            it=item(i);es=events(it,tok)
            if not es:continue
            weights,qmap,logits=reader.forward(it,es)
            local=defaultdict(list);localcorrect=defaultdict(list)
            for e in es:
                kind=e['name'];q=e['q'];qs=qmap[q];keys=e['keys'];a=np.stack([weights[li][qs,:q+1] for li in range(6)])
                if split=='val':
                    if kind not in sums:sums[kind]=np.zeros((6,LIMIT));counts[kind]=np.zeros(LIMIT,int)
                    sums[kind][:,:q+1]+=a[:,::-1];counts[kind][:q+1]+=1
                    continue
                value=a[:,keys].sum(1);local[kind].append(value)
                if kind in counts:
                    distance=q-np.array(keys);good=counts[kind][distance]>0
                    mu=np.divide(sums[kind][:,distance],counts[kind][distance],out=np.zeros((6,len(keys))),where=counts[kind][distance]>0)
                    delta=(a[:,keys]-mu)[:,good].mean(1) if good.any() else np.zeros(6)
                    localcorrect[kind].append(delta)
                if split=='pictured' or kind not in examples:
                    inverse={v:f'V{j}' for j,v in enumerate(it['node_tokens'])}
                    def label(pos):return inverse.get(it['tokens'][pos],tok.name(it['tokens'][pos]))
                    ex=dict(query_id=i,record=e['record'],q=q,input_token=label(q),predicted_reference=label(q+1),
                            native_prediction=inverse.get(int(logits[q].argmax()),tok.name(int(logits[q].argmax()))),
                            keys=[dict(position=k,token=label(k),mass=[float(a[l,k]) for l in range(6)]) for k in keys],
                            total=value.tolist(),top_keys_per_layer=[[
                                dict(position=int(k),token=label(int(k)),mass=float(a[l,k]))
                                for k in np.argsort(a[l])[-3:][::-1]] for l in range(6)])
                    if split=='pictured':special.setdefault(str(i),{}).setdefault(kind,[]).append(ex)
                    else:examples[kind]=ex
            if split=='test':
                for k,v in local.items():raw[k].append(dict(query_id=i,n=len(v),mean=np.mean(v,0).tolist(),sum=np.sum(v,0).tolist()))
                for k,v in localcorrect.items():corrected[k].append(dict(query_id=i,mean=np.mean(v,0).tolist()))
        print(name,split,'done',len(ids),flush=True)
    reader.close()
    metrics={}
    for k,rows in raw.items():
        n=sum(r['n'] for r in rows)
        metrics[k]=dict(n_queries=len(rows),n_events=n,query_macro=np.mean([r['mean'] for r in rows],0).tolist(),
                        event_mean=(np.sum([r['sum'] for r in rows],0)/n).tolist(),
                        distance_centered_density=np.mean([r['mean'] for r in corrected[k]],0).tolist() if corrected[k] else None)
    ckpt=folder/'checkpoints/best.pt'
    result=dict(version=3,name=name,run=str(folder),data_root=str(data),step=step,checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),
                precision='fp32',max_receiver=LIMIT-1,trace='canonical teacher-forced reference',ids=chosen,
                readonly_logits_identity=reader.checked,metrics=metrics,examples=examples,pictured_queries=special)
    dest.write_text(json.dumps(result,indent=2)+'\n')
    del model;torch.cuda.empty_cache()

if __name__=='__main__':
    torch.set_num_threads(2);OUT.mkdir(exist_ok=True)
    for name,(folder,data) in SETTINGS.items():main(name,folder,data)

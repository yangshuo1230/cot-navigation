"""Matched-query / matched-reference / free-trajectory cross-seed attention audit.

CPU only. Layers are selected on N12+N16 and checked on held-out N24.
Only consumed prefixes valid under independent DFS replay receive state labels.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,hashlib,time,zipfile
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/mnt/ningxuefei/yangshuo/cot-navigation-unified')
sys.path.insert(0,str(ROOT/'src'))
from cot_navigation.config import load_config
from cot_navigation.models.transformer import build_model
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.data.dataset import ContextStore
from cot_navigation.evaluation.rollout import generate
from cot_navigation.protocols.search_replay import replay_nodes

OUT=ROOT/'studies/cross_seed_attention_20261101'
OLD=ROOT/'studies/dfs_nodes_attention_generalization_20261030/summary.json'
GROUPS=['down','backtrack','found','exhaust']

def measurements(att,tokens,p,nt,edges,adj,s,t,seed,n,ix,mode,valid):
    """Rows for legal consumed state, including predictions that become illegal."""
    inv={v:i for i,v in enumerate(nt)};seen=set();stack=[];positions={};result=[]
    for pos in range(p,len(tokens)-1):
        node=inv.get(tokens[pos]);nxt=inv.get(tokens[pos+1])
        if node is None:break
        if not seen:
            if node!=s:break
            stack=[node]
        else:
            if stack[-1]==t or node in seen:break
            while stack and not any(v not in seen for v in adj[stack[-1]]):stack.pop()
            if not stack or node not in adj[stack[-1]]:break
            stack.append(node)
        seen.add(node);positions[node]=pos
        available=[u for u in stack if any(v not in seen for v in adj[u])]
        eligible=available[-1] if available else None
        kind='found' if node==t else 'down' if eligible==node else 'backtrack' if eligible is not None else 'exhaust'
        hist=list(range(p,pos+1));anc=[positions[u] for u in stack[:-1]];ep=positions[eligible] if eligible is not None else None
        r=dict(seed=seed,n_nodes=n,index=ix,mode=mode,position=pos,node=node,next=nxt,kind=kind,case_valid=valid,
          history_size=len(hist),ancestor_count=len(anc),eligible=eligible,stack=stack[:],skip_parent=kind=='backtrack' and eligible!=stack[-2],
          ancestor_uniform=len(anc)/len(hist),eligible_uniform=1/len(hist),
          parent_baseline=int(kind=='backtrack' and eligible==stack[-2]),root_baseline=int(eligible==stack[0]))
        layers=[]
        for l,w in enumerate(att[:,pos]):
            hmass=float(w[p:pos+1].sum());htop=hist[int(w[hist].argmax())];top=int(w.argmax())
            z=dict(history_mass=hmass,ancestor_mass=float(w[anc].sum()) if anc else 0,
              ancestor_hist_top=int(htop in anc),ancestor_global_top=int(top in anc),
              eligible_mass=float(w[ep]) if ep is not None else 0,
              eligible_hist_share=float(w[ep])/max(hmass,1e-20) if ep is not None else 0,
              eligible_hist_top=int(htop==ep),eligible_global_top=int(top==ep),
              parent_mass=float(w[positions[stack[-2]]]) if len(stack)>1 else 0,
              current_mass=float(w[pos]),root_mass=float(w[p]),query_marker=float(w[p-4]),
              target_mass=float(w[p-2]),target_top=int(top==p-2),source_mass=float(w[p-3]),bar_mass=float(w[p-1]),bar_top=int(top==p-1))
            for slot,offset in [('src',1),('dst',2),('relay',3)]:
                for label,src,unvisited in [('out',node,False),('unvisited',node,True),('eligible_out',eligible,True)]:
                    keys=[3*j+offset for j,(u,v) in enumerate(edges) if (slot!='relay' or j<len(edges)-1) and u==src and (not unvisited or v not in seen)]
                    z[label+'_'+slot+'_mass']=float(w[keys].sum()) if keys else 0
                    z[label+'_'+slot+'_top']=int(top in keys)
            layers.append(z)
        r['layers']=layers;result.append(r)
    return result

def aggregate(rows):
    return dict(n=len(rows),layers=[{k:float(np.mean([r['layers'][l][k]for r in rows]))for k in rows[0]['layers'][l]}for l in range(6)]) if rows else dict(n=0,layers=[])

def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(2);start=time.time()
    old=json.loads(OLD.read_text());selected={n:[c['index']for c in old['cases']if c['n_nodes']==n]for n in [12,16,24]}
    stores={}
    for n in selected:
        cfg=load_config([ROOT/f'configs/generated/fine_node_curriculum_20261028/data_simple_n{n}.yaml'])
        stores[n]=(ContextStore(ROOT/cfg.data.root),str(ROOT/cfg.data.root))
    allrows=[];inputs=[];cases=[];checkpoints=[];examples=[];maxdiff=0;greedy_pass=0
    for seed in [0,1,2]:
        run=ROOT/f'outputs/fine_node_curriculum_20261028/s{seed}/dfs_nodes/simple'
        cfg=load_config([run/'resolved_config.yaml']);tok=Tokenizer.from_config(cfg);cp=run/'checkpoints/latest.pt'
        state=torch.load(cp,map_location='cpu',weights_only=False)
        model=build_model(cfg.model,len(tok),tok.pad).eval();model.load_state_dict(state['model']);model.requires_grad_(False)
        assert cfg.data.knowledge=='context_graph' and cfg.protocol.name=='dfs_nodes'
        assert cfg.train.model_seed==seed and state['step']==1000000
        sha=hashlib.sha256(cp.read_bytes()).hexdigest();checkpoints.append(dict(seed=seed,path=str(cp),sha256=sha,step=int(state['step'])))
        for n,indices in selected.items():
            store,data=stores[n]
            for ci,ix in enumerate(indices):
                row=store.render(ix,'dfs_nodes',tok);p=row.prompt_len;nt=store.node_vocab_ids(ix,tok);inv={v:i for i,v in enumerate(nt)}
                edges=list(zip(store.arrays['edge_src'][ix].astype(int).tolist(),store.arrays['edge_dst'][ix].astype(int).tolist()))
                adj=store.adj(ix);s=int(store.arrays['s_topo'][ix]);t=int(store.arrays['t_topo'][ix])
                with torch.inference_mode():
                    out=generate(model,[row.tokens[:p]],'dfs_nodes',tok,'fp32',max_new_tokens=n+4)[0]
                rep=replay_nodes(adj,s,t,out['trace_tokens'],inv)
                case=dict(seed=seed,n_nodes=n,index=ix,sample_id=int(store.arrays['sample_id'][ix]),label=row.label,
                    pred=out['pred'],correct=out['pred']==row.label,valid=rep['valid'],error=rep['error'],limit=out['generation_limit_reached'],generated=out['generated'])
                cases.append(case)
                for mode,tokens in [('free',out['tokens']),('reference',row.tokens[:-1])]:
                    with torch.inference_mode():
                        ids=torch.tensor(tokens)[None];normal=model(ids);logits,weights=model(ids,return_attn=True)
                    if mode=='free':
                        assert normal[0,p-1:-1].argmax(-1).tolist()==tokens[p:];greedy_pass+=1
                    diff=float((normal-logits).abs().max());maxdiff=max(maxdiff,diff)
                    att=np.stack([w[0,0].numpy()for w in weights])
                    assert np.allclose(att.sum(-1),1,atol=2e-6) and np.max(np.triu(att,1))==0
                    if mode=='free':
                        inp=[]
                        for l in range(6):
                            z={}
                            for name,receiver,key in [('relay_dst',3,2),('relay_src',3,1),('dst_src',2,1)]:
                                ws=[att[l,3*j+receiver,3*j+key]for j in range(len(edges)-1)]
                                hit=[int(att[l,3*j+receiver].argmax()==3*j+key)for j in range(len(edges)-1)]
                                z[name]=float(np.mean(ws));z[name+'_top']=float(np.mean(hit))
                            z['bar_target']=float(att[l,p-1,p-2]);z['bar_target_top']=int(att[l,p-1].argmax()==p-2);z['bar_source']=float(att[l,p-1,p-3]);z['last_query_src']=float(att[l,p-4,p-6]);z['last_query_dst']=float(att[l,p-4,p-5]);inp.append(z)
                        inputs.append(dict(seed=seed,n_nodes=n,index=ix,layers=inp))
                    allrows.extend(measurements(att,tokens,p,nt,edges,adj,s,t,seed,n,ix,mode,rep['valid'] if mode=='free'else True))
                    if mode=='free' and n==16 and ci in [0,16]:
                        names={v:chr(65+i)for i,v in enumerate(nt)}
                        d=dict(seed=seed,checkpoint=str(cp),checkpoint_sha256=sha,step=int(state['step']),dataset=data,index=ix,sample_id=case['sample_id'],
                            split='test',source=chr(65+s),target=chr(65+t),edges=edges,node_tokens=nt,labels=[names.get(v,tok.name(v).strip('<>'))for v in tokens],
                            tokens=tokens,prompt_len=p,attention=np.round(att,8).tolist(),strict_replay=rep,case=case,full_forward_greedy_match=True,max_math_sdpa_logit_difference=diff)
                        name=f'seed{seed}_n{n}_label{row.label}_idx{ix}';(OUT/(name+'.json')).write_text(json.dumps(d,separators=(',',':')));examples.append(name)
            print(f'seed{seed} N{n} done; elapsed {time.time()-start:.1f}s',flush=True)
        del model,state
    # Fixed grouping, no layer selection on held-out N24.
    ag={}
    for seed in [0,1,2]:
        ag[str(seed)]={}
        for mode in ['free','reference']:
            ag[str(seed)][mode]={}
            for split,ns in [('discovery',[12,16]),('heldout',[24]),('all',[12,16,24])]:
                rr=[r for r in allrows if r['seed']==seed and r['mode']==mode and r['n_nodes']in ns]
                ag[str(seed)][mode][split]={k:aggregate([r for r in rr if r['kind']==k])for k in GROUPS}
    inp={str(seed):{split:aggregate([r for r in inputs if r['seed']==seed and r['n_nodes']in ns])for split,ns in [('discovery',[12,16]),('heldout',[24]),('all',[12,16,24])]}for seed in [0,1,2]}
    result=dict(design='Same 96 held-out queries as 20261030, 32 per N12/N16/N24, labels balanced, no success filtering. Seeds 0/1/2, same 1M simple-family training recipe. CPU FP32. Free generation and identical reference prefixes. Layer selection only on N12+N16; evaluate N24 separately. State labels stop before first illegal consumed node. Input aggregation excludes final QUERY boundary.',
      checkpoints=checkpoints,selected=selected,cases=cases,groups=ag,input=inp,examples=examples,greedy_checks=greedy_pass,
      max_math_sdpa_logit_difference=maxdiff,elapsed_seconds=time.time()-start)
    (OUT/'summary.json').write_text(json.dumps(result,indent=2))
    (OUT/'rows.json').write_text(json.dumps(allrows,separators=(',',':')))
    (OUT/'inputs.json').write_text(json.dumps(inputs,separators=(',',':')))
    with zipfile.ZipFile('/tmp/cross_seed_attention_20261101.zip','w',zipfile.ZIP_DEFLATED)as z:
        for f in OUT.glob('*.json'):z.write(f,f.name)
    print(json.dumps(dict(greedy_checks=greedy_pass,rows=len(allrows),examples=examples,elapsed=time.time()-start)),flush=True)

if __name__=='__main__':main()

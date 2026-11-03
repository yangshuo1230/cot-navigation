"""Predeclared, label-balanced test scan; no success/attention-based selection."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,hashlib,csv,time
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

OUT=ROOT/'studies/dfs_nodes_attention_generalization_20261030'
SEED=20261030

def summary(rows,keys):
    return dict(n=len(rows),**{k:float(np.mean([r[k] for r in rows])) if rows else None for k in keys})

def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(2);start=time.time()
    run=ROOT/'outputs/fine_node_curriculum_20261028/s0/dfs_nodes/simple'
    cfg=load_config([run/'resolved_config.yaml']);tok=Tokenizer.from_config(cfg);cp=run/'checkpoints/latest.pt'
    state=torch.load(cp,map_location='cpu',weights_only=False)
    model=build_model(cfg.model,len(tok),tok.pad).eval();model.load_state_dict(state['model']);model.requires_grad_(False)
    allrows=[];inputs=[];cases=[];examples=[];maxdiff=0
    for n in [12,16,24]:
        c=load_config([ROOT/f'configs/generated/fine_node_curriculum_20261028/data_simple_n{n}.yaml'])
        store=ContextStore(ROOT/c.data.root);rng=np.random.default_rng(SEED+n);chosen=[]
        for label in [0,1]:
            ids=[int(i) for i in store.indices('test') if int(store.arrays['label'][i])==label]
            chosen.extend(rng.choice(ids,size=16,replace=False).tolist())
        for ci,ix in enumerate(chosen):
            row=store.render(ix,'dfs_nodes',tok);p=row.prompt_len;nt=store.node_vocab_ids(ix,tok);inv={v:i for i,v in enumerate(nt)}
            adj=store.adj(ix);s=int(store.arrays['s_topo'][ix]);t=int(store.arrays['t_topo'][ix]);edges=list(zip(store.arrays['edge_src'][ix].astype(int).tolist(),store.arrays['edge_dst'][ix].astype(int).tolist()));e=len(edges)
            with torch.inference_mode():
                out=generate(model,[row.tokens[:p]],'dfs_nodes',tok,'fp32',max_new_tokens=n+4)[0]
                tokens=out['tokens'];ids=torch.tensor(tokens)[None];normal=model(ids);logits,weights=model(ids,return_attn=True)
            att=np.stack([w[0,0].numpy() for w in weights]);diff=float((normal-logits).abs().max());maxdiff=max(maxdiff,diff)
            assert normal[0,p-1:-1].argmax(-1).tolist()==tokens[p:]
            assert np.allclose(att.sum(-1),1,atol=2e-6) and np.max(np.triu(att,1))==0
            rep=replay_nodes(adj,s,t,out['trace_tokens'],inv)
            case=dict(n_nodes=n,index=ix,sample_id=int(store.arrays['sample_id'][ix]),label=row.label,pred=out['pred'],correct=out['pred']==row.label,
                      valid=bool(rep['valid']),error=rep['error'],terminal=rep['terminal'],limit=out['generation_limit_reached'],generated=out['generated'],maxdiff=diff)
            cases.append(case)
            # A clean relay is the NEXT EDGE marker, not the marker before u,v.
            # Final edge ends at QUERY and is tracked separately.
            for ei in range(e):
                relay=3*(ei+1);dest=3*ei+2;src=3*ei+1
                inputs.append(dict(n_nodes=n,index=ix,final=ei==e-1,
                    l1_dest=float(att[0,relay,dest]),l1_dest_top=int(att[0,relay].argmax()==dest),
                    l2_src=float(att[1,relay,src]),l2_src_top=int(att[1,relay].argmax()==src)))
            seen=set();stack=[];positions={};broken=False;case_rows=[]
            for pos in range(p,len(tokens)-1):
                consumed=tokens[pos]
                if consumed not in inv:break
                node=inv[consumed]
                if not stack:
                    if node!=s:broken=True;break
                    stack=[node]
                else:
                    if stack[-1]==t:broken=True;break
                    while stack and not any(v not in seen for v in adj[stack[-1]]):stack.pop()
                    if not stack or node in seen or node not in adj[stack[-1]]:broken=True;break
                    stack.append(node)
                seen.add(node);positions[node]=pos
                available=[u for u in stack if any(v not in seen for v in adj[u])]
                selected=available[-1] if available else None
                if node==t:kind='found'
                elif selected==node:kind='down'
                elif selected is not None:kind='backtrack'
                else:kind='exhaust'
                ancestors=stack[:-1];hpositions=list(range(p,pos+1));ancestorpos=[positions[u] for u in ancestors]
                targetpos=positions[selected] if selected is not None else None
                expected=set(adj[selected])-seen if selected is not None else set()
                nxt=inv.get(tokens[pos+1]);next_legal=(tokens[pos+1]==tok.yes if kind=='found' else tokens[pos+1]==tok.no if kind=='exhaust' else nxt in expected)
                r=dict(n_nodes=n,index=ix,label=row.label,case_valid=case['valid'],position=pos,node=node,next_node=nxt,kind=kind,
                       stack=stack[:],ancestors=ancestors[:],selected_ancestor=selected,history_size=len(hpositions),ancestor_count=len(ancestors),next_legal=bool(next_legal))
                relaypos=[3*(j+1) for j,(u,v) in enumerate(edges[:-1]) if u==node]
                legalrelay=[3*(j+1) for j,(u,v) in enumerate(edges[:-1]) if u==node and v not in seen]
                for l in range(6):
                    w=att[l,pos];histmass=float(w[p:pos+1].sum());hs=w[hpositions];histtop=hpositions[int(hs.argmax())];globaltop=int(w.argmax())
                    r.update({f'l{l+1}_history_mass':histmass,f'l{l+1}_query_mass':float(w[p-4:p].sum()),
                        f'l{l+1}_ancestor_mass':float(w[ancestorpos].sum()) if ancestorpos else 0,
                        f'l{l+1}_ancestor_hist_share':float(w[ancestorpos].sum())/max(histmass,1e-20) if ancestorpos else 0,
                        f'l{l+1}_ancestor_hist_top':int(histtop in ancestorpos),
                        f'l{l+1}_eligible_mass':float(w[targetpos]) if targetpos is not None else 0,
                        f'l{l+1}_eligible_hist_share':float(w[targetpos])/max(histmass,1e-20) if targetpos is not None else 0,
                        f'l{l+1}_eligible_hist_top':int(histtop==targetpos),f'l{l+1}_eligible_global_top':int(globaltop==targetpos)})
                r.update(l4_outedge_mass=float(att[3,pos,relaypos].sum()) if relaypos else 0,
                    l4_outedge_top=int(att[3,pos].argmax() in relaypos),l4_legal_outedge_mass=float(att[3,pos,legalrelay].sum()) if legalrelay else 0,
                    outedge_count=len(relaypos),has_final_outedge=edges[-1][0]==node,
                    l6_target_mass=float(att[5,pos,p-2]),l6_target_top=int(att[5,pos].argmax()==p-2),
                    l4_sink_first_triple=float(att[3,pos,:3].sum()),l3_query_marker=float(att[2,pos,p-4]),
                    ancestor_uniform=len(ancestorpos)/len(hpositions),eligible_uniform=1/len(hpositions))
                allrows.append(r);case_rows.append(r)
            # Retain first positive/negative example at each size, fixed by sampling order.
            if ci in [0,16]:
                names={v:chr(65+i) for i,v in enumerate(nt)}
                ex=dict(checkpoint=str(cp),checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),step=int(state['step']),dataset=str(ROOT/c.data.root),
                    index=ix,sample_id=case['sample_id'],split='test',source=chr(65+s),target=chr(65+t),edges=edges,node_tokens=nt,
                    labels=[names.get(v,tok.name(v).strip('<>')) for v in tokens],tokens=tokens,prompt_len=p,attention=np.round(att,8).tolist(),
                    generated_order=[names[v] for v in out['trace_tokens'] if v in names],strict_replay=rep,precision='fp32',device='cpu',
                    full_forward_greedy_match=True,max_math_sdpa_logit_difference=diff,case=case)
                name=f'n{n}_label{row.label}_idx{ix}';(OUT/(name+'.json')).write_text(json.dumps(ex,separators=(',',':')))
                examples.append(name)
            if (ci+1)%8==0:print(f'N{n}: {ci+1}/32 elapsed {time.time()-start:.1f}s',flush=True)
        del store
    keys=['l3_history_mass','l3_query_marker','l3_ancestor_mass','l3_ancestor_hist_share','l3_ancestor_hist_top',
          'l3_eligible_mass','l3_eligible_hist_top','l5_history_mass','l5_ancestor_mass','l5_ancestor_hist_share','l5_ancestor_hist_top',
          'l5_eligible_mass','l5_eligible_hist_share','l5_eligible_hist_top','l5_eligible_global_top',
          'l4_outedge_mass','l4_outedge_top','l4_legal_outedge_mass','l4_sink_first_triple','l6_target_mass','l6_target_top','ancestor_uniform','eligible_uniform','next_legal']
    groups={k:summary([r for r in allrows if r['kind']==k],keys) for k in ['down','backtrack','found','exhaust']}
    bysize={str(n):dict(cases=summary([c for c in cases if c['n_nodes']==n],['correct','valid','limit']),
              groups={k:summary([r for r in allrows if r['n_nodes']==n and r['kind']==k],keys) for k in groups})for n in [12,16,24]}
    result=dict(checkpoint=str(cp),checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),step=int(state['step']),seed=SEED,
      design='Same seed0 simple checkpoint; N12/N16/N24; 16 positive and 16 negative test queries per size, sampled before generation; no success filtering; CPU FP32 greedy; cap N+4; only semantically valid consumed prefixes receive state labels.',
      n_cases=len(cases),cases=cases,n_valid_prefixes=len(allrows),groups=groups,bysize=bysize,
      input_clean=summary([r for r in inputs if not r['final']],['l1_dest','l1_dest_top','l2_src','l2_src_top']),
      input_final_query=summary([r for r in inputs if r['final']],['l1_dest','l1_dest_top','l2_src','l2_src_top']),
      max_math_sdpa_logit_difference=maxdiff,greedy_checks_passed=len(cases),elapsed=time.time()-start,examples=examples)
    (OUT/'summary.json').write_text(json.dumps(result,indent=2))
    (OUT/'rows.json').write_text(json.dumps(allrows))
    (OUT/'inputs.json').write_text(json.dumps(inputs))
    print(json.dumps({k:v for k,v in result.items() if k not in ['cases','bysize']},indent=2),flush=True)

if __name__=='__main__':main()
